"""Primary user-facing dataset/model workflow API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import requests

from dcat_ap_hub.files.files import FileCollection, scan_directory
from dcat_ap_hub.integrations.discovery import (
    build_model_load_plan,
    collect_search_paths,
    detect_local_model_roles,
)
from dcat_ap_hub.integrations.integrations import load_model as load_with_integration
from dcat_ap_hub.integrations.models import BackendName, LoadOptions, ModelSource
from dcat_ap_hub.metadata.models import DatasetMetadata, Distribution
from dcat_ap_hub.metadata.parsing import (
    JSONLD_ACCEPT_HEADER,
    fetch_and_parse,
    parse_local_file,
)
from dcat_ap_hub.metadata.transfer import download_dataset_files
from dcat_ap_hub.processing.pipeline import (
    find_existing_processed_path,
    run_processing_pipeline,
)


class Dataset:
    """Main entry point for interacting with DCAT-AP datasets and model artifacts."""

    def __init__(
        self, meta: DatasetMetadata, local_data_path: Optional[Path] = None
    ) -> None:
        """
        Initialize a dataset wrapper from parsed metadata.

        Args:
            meta: Parsed DCAT-AP metadata representation.
            local_data_path: Optional local path where artifacts already exist.
        """
        self._meta = meta

        # Local artifact state is lazily populated by download/process/load calls.
        self._local_data_path = local_data_path
        self._local_processed_path: Optional[Path] = None
        self._local_model_path: Optional[Path] = None

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def load(cls, source: Union[str, Path], verbose: bool = False) -> Dataset:
        """
        Construct a ``Dataset`` from URL, local metadata file, or local directory.

        Args:
            source: URL/path identifying where metadata or artifacts are located.
            verbose: Whether remote loading/parsing should log progress.
        """
        source_str = str(source)
        path_obj = Path(source)

        if source_str.startswith(("http://", "https://")):
            return cls.from_url(source_str, verbose=verbose)
        if path_obj.is_file():
            return cls.from_file(path_obj)
        if path_obj.is_dir():
            return cls.from_directory(path_obj)

        raise ValueError(
            f"Invalid source: '{source}'. Must be URL, file, or directory."
        )

    @classmethod
    def from_url(cls, url: str, verbose: bool = False) -> Dataset:
        """Load metadata from a remote URL and return a ``Dataset`` instance."""
        meta = fetch_and_parse(url, verbose=verbose)
        return cls(meta)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> Dataset:
        """Load metadata from a local JSON/JSON-LD file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        meta = parse_local_file(p)
        return cls(meta, local_data_path=None)

    @classmethod
    def from_directory(cls, path: Union[str, Path]) -> Dataset:
        """
        Build a dataset wrapper from a local directory.

        The method first tries to restore full metadata from a local ``*.jsonld``
        file. If none is available, it creates virtual metadata from the files
        in the directory and infers whether the directory likely contains a model.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Directory not found: {p}")

        # Restore full metadata first when available.
        meta = None
        for candidate in p.glob("*.jsonld"):
            try:
                # We use the internal parser directly to get the object
                meta = parse_local_file(candidate)
                break
            except:  # noqa: E722
                continue

        # Fall back to local artifact heuristics when metadata is missing/incomplete.
        local_model_roles = detect_local_model_roles([p])
        is_model_guess = bool(local_model_roles) or bool(meta and meta.is_model)

        if not meta:
            files = [f for f in p.iterdir() if f.is_file()]
            distros = [
                Distribution(f.name, "Local file", f.suffix.lstrip("."), f.as_uri())
                for f in files
            ]
            meta = DatasetMetadata(
                title=p.name,
                description="Virtual dataset from local directory",
                distributions=distros,
                is_model=is_model_guess,
                source_url=str(p.absolute()),
            )
        elif is_model_guess and not meta.is_model:
            meta.is_model = True

        ds = cls(meta)

        if ds.is_model:
            ds._local_model_path = p
            ds._local_data_path = p
        else:
            ds._local_data_path = p

        # Reuse existing processed outputs when present.
        ds._local_processed_path = find_existing_processed_path(
            local_data_path=p,
            known_processed_path=None,
            processed_dir="processed",
        )

        return ds

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def title(self) -> str:
        """Human-readable dataset title from metadata."""
        return self._meta.title

    @property
    def is_model(self) -> bool:
        """Return whether metadata marks this dataset as a model entry."""
        return self._meta.is_model

    @property
    def local_path(self) -> Optional[Path]:
        """Return the data path if available, else the model path."""
        return self._local_data_path or self._local_model_path

    @property
    def processed_path(self) -> Optional[Path]:
        """Path to processed outputs when available, otherwise ``None``."""
        return self._local_processed_path

    # =========================================================================
    # Core Operations
    # =========================================================================

    def _save_metadata(self, directory: Path, verbose: bool = False) -> None:
        """
        Save remote source metadata as ``dcat-metadata.jsonld`` for offline reuse.

        No-op for local-only metadata sources.
        """
        if not self._meta.source_url.startswith(("http://", "https://")):
            return

        target_file = directory / "dcat-metadata.jsonld"
        if target_file.exists():
            return

        try:
            if verbose:
                print("Saving metadata for offline usage...")
            response = requests.get(
                self._meta.source_url,
                headers={"Accept": JSONLD_ACCEPT_HEADER},
                timeout=10,
            )
            if response.status_code == 200:
                # Re-serialize to ensure stable, readable formatting.
                data = response.json()
                target_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            if verbose:
                print(f"Warning: Could not save metadata file: {e}")

    def download(
        self,
        data_dir: Union[str, Path] = "./data",
        force: bool = False,
        verbose: bool = True,
    ) -> FileCollection:
        """
        Download referenced dataset artifacts and return a lazy ``FileCollection``.

        Behavior is idempotent by default: if local data already exists and
        ``force=False``, the existing artifacts are loaded instead of redownloaded.
        """
        if self._local_data_path and self._local_data_path.exists() and not force:
            if verbose:
                print(f"Using existing local data at '{self._local_data_path}'")
            return FileCollection(
                self._local_data_path, scan_directory(self._local_data_path)
            )

        path = download_dataset_files(
            self._meta, Path(data_dir), force=force, verbose=verbose
        )

        self._local_data_path = path

        self._save_metadata(path, verbose=verbose)
        return FileCollection(path, scan_directory(path))

    def process(
        self,
        processed_dir: str = "processed",
        force: bool = False,
        verbose: bool = True,
    ) -> FileCollection:
        """
        Process downloaded artifacts using metadata-linked processor resources.

        Behavior mirrors ``download()``: when processed outputs already exist and
        ``force=False``, existing artifacts are loaded instead of reprocessed.
        """
        if not self._local_data_path:
            raise RuntimeError("Data not downloaded. Call .download() first.")

        if not force:
            existing = find_existing_processed_path(
                local_data_path=self._local_data_path,
                known_processed_path=self._local_processed_path,
                processed_dir=processed_dir,
            )
            if existing:
                self._local_processed_path = existing
                if verbose:
                    print(
                        f"Processed data found at '{existing.name}'. Skipping (use force=True to rerun)."
                    )
                return FileCollection(existing, scan_directory(existing))

        self._local_processed_path = run_processing_pipeline(
            local_data_path=self._local_data_path,
            resources=self._meta.related_resources,
            processed_dir=processed_dir,
            verbose=verbose,
        )
        return FileCollection(
            self._local_processed_path,
            scan_directory(self._local_processed_path),
        )

    # =========================================================================
    # Model Loading Helpers
    # =========================================================================

    def _model_search_paths(self) -> list[Path]:
        """Local search roots used to locate model artifacts and sidecars."""
        return collect_search_paths(self._local_model_path, self._local_data_path)

    def _load_with_sources(
        self,
        backend: BackendName,
        sources: list[ModelSource],
        options: LoadOptions,
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Attempt model loading from ordered candidate sources.

        Returns:
            Tuple of ``(model, adapter, metadata)``.
        """
        errors: list[str] = []
        for source in sources:
            try:
                loaded = load_with_integration(
                    source=source,
                    options=options,
                    backend=backend,
                )
                return loaded.as_tuple()
            except Exception as e:
                location = (
                    source.path.name if source.path else source.model_id or "<unknown>"
                )
                errors.append(f"{location}: {e}")

        raise RuntimeError(
            f"Failed to load {backend.value} model from available candidates. "
            + " | ".join(errors)
        )

    def load_model(
        self,
        model_dir: Union[str, Path] = "./models",
        token: Optional[str] = None,
        device_map: Union[str, Dict[str, Any]] = "auto",
        dtype: str = "auto",
        trust_remote_code: bool = False,
        load_task_specific_head: bool = True,
        onnx_providers: Optional[list[str]] = None,
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Detect and load exactly one model backend from local artifacts/metadata.

        Returns:
            A tuple ``(model, adapter, metadata)`` where adapter is backend-specific
            (for example tokenizer for Hugging Face, or ``None`` for ONNX/sklearn).
        """
        plan = build_model_load_plan(
            metadata=self._meta,
            search_paths=self._model_search_paths(),
            model_dir=model_dir,
            token=token,
            device_map=device_map,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            load_task_specific_head=load_task_specific_head,
            onnx_providers=onnx_providers,
        )
        return self._load_with_sources(
            backend=plan.backend,
            sources=plan.sources,
            options=plan.options,
        )

    def __repr__(self) -> str:
        """Compact human-readable dataset summary for interactive use."""
        icon = "🧠" if self.is_model else "📊"
        locs = []
        if self._local_data_path:
            locs.append(f"Data: {self._local_data_path.name}")
        if self._local_processed_path:
            locs.append("Processed: ✓")
        if self._local_model_path:
            locs.append(f"Model: {self._local_model_path.name}")

        loc_str = f" [{', '.join(locs)}]" if locs else ""
        return f"{icon} Dataset('{self.title}', {len(self._meta.distributions)} distros){loc_str}"
