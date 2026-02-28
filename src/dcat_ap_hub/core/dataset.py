from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union

import requests

from dcat_ap_hub.core.files import FileCollection
from dcat_ap_hub.internals.constants import HF_FORMAT, ONNX_FORMAT
from dcat_ap_hub.internals.integrations import (
    load_hf_model,
    load_onnx_model,
    load_sklearn_model,
)
from dcat_ap_hub.internals.loaders import scan_directory
from dcat_ap_hub.internals.models import DatasetMetadata, Distribution
from dcat_ap_hub.internals.parser import (
    JSONLD_ACCEPT_HEADER,
    fetch_and_parse,
    parse_local_file,
)
from dcat_ap_hub.internals.processor import apply_processor_logic
from dcat_ap_hub.internals.transfer import download_dataset_files


class Dataset:
    """
    The main entry point for interacting with DCAT-AP datasets and models.
    """

    _MODEL_DIST_ROLES = ("huggingface_model", "onnx_model", "sklearn_model")
    _MODEL_ROLES = Literal["huggingface_model", "onnx_model", "sklearn_model"]

    def __init__(
        self, meta: DatasetMetadata, local_data_path: Optional[Path] = None
    ) -> None:
        self._meta = meta

        # 1. State for Data
        self._local_data_path = local_data_path

        # 2. State for Processed Data
        # We start as None. It is set by process(), load_processed(), or auto-detection.
        self._local_processed_path: Optional[Path] = None

        # 3. State for Models
        self._local_model_path: Optional[Path] = None

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def load(cls, source: Union[str, Path], verbose: bool = False) -> Dataset:
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
        meta = fetch_and_parse(url, verbose=verbose)
        return cls(meta)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> Dataset:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        meta = parse_local_file(p)
        return cls(meta, local_data_path=None)

    @classmethod
    def from_directory(cls, path: Union[str, Path]) -> Dataset:
        """
        Load from a directory.
        1. Tries to find a stored 'dcat-metadata.jsonld' file to restore full metadata.
        2. If not found, scans files to create a 'virtual' dataset.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Directory not found: {p}")

        # 1. Try to restore full metadata from saved file
        meta = None
        for candidate in p.glob("*.jsonld"):
            try:
                # We use the internal parser directly to get the object
                meta = parse_local_file(candidate)
                break
            except:  # noqa: E722
                continue

        # 2. If no metadata, create virtual
        is_model_guess = (
            (p / "config.json").exists()
            or any(p.glob("*.onnx"))
            or any("model" in f.stem.lower() for f in p.glob("*.py"))
            or any("model" in f.stem.lower() for f in p.glob("*.txt"))
        )

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

        # Create instance
        ds = cls(meta)

        # Assign paths based on what we found
        if is_model_guess:
            ds._local_model_path = p
            # FIX: Also set data path for models so .process() works
            ds._local_data_path = p
        else:
            ds._local_data_path = p

        # Auto-detect processed folder so load_processed works immediately
        processed_guess = p / "processed"
        if processed_guess.exists() and any(processed_guess.iterdir()):
            ds._local_processed_path = processed_guess

        return ds

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def title(self) -> str:
        return self._meta.title

    @property
    def is_model(self) -> bool:
        return self._meta.is_model

    @property
    def local_path(self) -> Optional[Path]:
        """Return the data path if available, else the model path."""
        return self._local_data_path or self._local_model_path

    @property
    def processed_path(self) -> Optional[Path]:
        """Public accessor for the processed data path."""
        return self._local_processed_path

    # =========================================================================
    # Core Operations
    # =========================================================================

    def _save_metadata(self, directory: Path, verbose: bool = False) -> None:
        """Internal helper to fetch and save the original metadata to disk."""
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
                # Re-serialize to ensure clean formatting
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
        Download dataset files to the data directory.
        """
        # If we already have a data path, use it
        if self._local_data_path and self._local_data_path.exists() and not force:
            if verbose:
                print(f"Using existing local data at '{self._local_data_path}'")
            return FileCollection(
                self._local_data_path, scan_directory(self._local_data_path)
            )

        # Perform download
        path = download_dataset_files(
            self._meta, Path(data_dir), force=force, verbose=verbose
        )

        # Set DATA path specifically
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
        Executes the attached processor script on the downloaded data.

        :param processed_dir: Name of the output folder relative to the data path.
        :param force: If True, runs the processor even if the output folder exists.
        :return: FileCollection of the processed files.
        """
        # 1. Pre-flight checks
        if not self._local_data_path:
            raise RuntimeError("Data not downloaded. Call .download() first.")

        # 2. Determine Output Directory early
        output_dir = self._local_data_path / processed_dir

        # Check if already processed
        if output_dir.exists() and any(output_dir.iterdir()) and not force:
            if verbose:
                print(
                    f"Processed data found at '{output_dir.name}'. Skipping (use force=True to rerun)."
                )

            # Update state
            self._local_processed_path = output_dir
            return FileCollection(output_dir, scan_directory(output_dir))

        # 3. Find the processor
        # Per strict rules: Processors are found in related_resources
        resources = self._meta.related_resources

        # 3a. Explicit role
        processor_item = next((r for r in resources if r.role == "processor"), None)
        notebook_item = next((r for r in resources if r.role == "notebook"), None)

        if not processor_item:
            raise ValueError("No processor found in related resources.")

        # 4. Resolve paths
        processor_filename = processor_item.get_filename()
        notebook_filename = notebook_item.get_filename() if notebook_item else None

        processor_path = self._local_data_path / processor_filename
        notebook_path = (
            self._local_data_path / notebook_filename if notebook_filename else None
        )

        # Fallback for extension
        if not processor_path.exists():
            processor_path = self._local_data_path / f"{processor_filename}.py"

        if not processor_path.exists():
            raise FileNotFoundError(f"Processor script not found at {processor_path}")

        # 5. Separate inputs (data) from the tool (processor)
        # Filter out the script itself AND the metadata file to be safe
        input_paths = [
            f
            for f in self._local_data_path.iterdir()
            if f.is_file()
            and f.name != processor_path.name
            and f.name != "dcat-metadata.jsonld"
        ]

        # filter out notebook if it exists
        if notebook_path and notebook_path.exists():
            input_paths = [p for p in input_paths if p.name != notebook_path.name]

        # 6. Prepare output and run
        output_dir.mkdir(parents=True, exist_ok=True)

        apply_processor_logic(processor_path, input_paths, output_dir, verbose=verbose)

        # 7. Update State
        self._local_processed_path = output_dir

        return FileCollection(output_dir, scan_directory(output_dir))

    def load_processed(self) -> FileCollection:
        """
        Loads data from the processed directory without re-running logic.
        """
        # 1. Check known state
        if self._local_processed_path and self._local_processed_path.exists():
            return FileCollection(
                self._local_processed_path, scan_directory(self._local_processed_path)
            )

        # 2. Check convention
        if self._local_data_path:
            # Assume default "processed" folder
            candidate = self._local_data_path / "processed"
            if candidate.exists() and any(candidate.iterdir()):
                self._local_processed_path = candidate
                return FileCollection(candidate, scan_directory(candidate))

        raise FileNotFoundError("No processed data found. Run .process() first.")

    # =========================================================================
    # Model Loading Helpers
    # =========================================================================

    def _get_search_paths(self) -> list[Path]:
        """Local paths used when locating model artifacts."""
        results: list[Path] = []
        for p in [self._local_model_path, self._local_data_path]:
            if p and p.exists() and p not in results:
                results.append(p)
        return results

    def _iter_files_with_extensions(self, *extensions: str) -> list[Path]:
        """Return unique files from search paths matching the given extensions."""
        seen: set[Path] = set()
        matches: list[Path] = []
        normalized = [ext.lstrip(".") for ext in extensions]
        for root in self._get_search_paths():
            for ext in normalized:
                for path in root.glob(f"*.{ext}"):
                    if path not in seen:
                        seen.add(path)
                        matches.append(path)
        return matches

    def _first_distribution_by_role(
        self, role: str, target_format: Optional[str] = None
    ) -> Optional[Distribution]:
        return next(
            (
                d
                for d in self._meta.distributions
                if d.role == role
                and (target_format is None or d.format == target_format)
            ),
            None,
        )

    def _find_file_by_extension(self, extension: str) -> Optional[Path]:
        """Helper to find a file with a specific extension in available paths."""
        matches = self._iter_files_with_extensions(extension)
        return matches[0] if matches else None

    def _has_local_hf_artifacts(self) -> bool:
        for p in self._get_search_paths():
            if (p / "config.json").exists():
                return True
        return False

    def _has_local_sklearn_artifacts(self) -> bool:
        for source_file in self._iter_files_with_extensions("py", "txt"):
            try:
                if "SKLearnModel" in source_file.read_text(encoding="utf-8"):
                    return True
            except Exception:
                continue
        return False

    def _distribution_model_roles(
        self,
    ) -> set[_MODEL_ROLES]:
        return {
            d.role for d in self._meta.distributions if d.role in self._MODEL_DIST_ROLES
        }

    def _detect_model_type(
        self,
    ) -> _MODEL_ROLES:
        """
        Detect a single model type from local artifacts first, then metadata roles.
        Raises when none or multiple candidates are detected.
        """
        detectors: dict[Dataset._MODEL_ROLES, Callable[[], bool]] = {
            "onnx_model": lambda: self._find_file_by_extension("onnx") is not None,
            "huggingface_model": self._has_local_hf_artifacts,
            "sklearn_model": self._has_local_sklearn_artifacts,
        }
        local_candidates = {role for role, detect in detectors.items() if detect()}

        if len(local_candidates) == 1:
            return next(iter(local_candidates))  # type: ignore
        if len(local_candidates) > 1:
            detected = ", ".join(sorted(local_candidates))
            raise ValueError(
                f"Ambiguous local model artifacts detected ({detected}). Keep only one model type."
            )

        roles = self._distribution_model_roles()
        if len(roles) == 1:
            return next(iter(roles))
        if len(roles) > 1:
            detected = ", ".join(sorted(roles))
            raise ValueError(
                f"Ambiguous model roles in metadata ({detected}). Keep only one role."
            )

        raise ValueError(
            "Could not detect model type. Expected exactly one of: ONNX (.onnx), "
            "Hugging Face (config.json), or sklearn (SKLearnModel source)."
        )

    def _load_sidecar_metadata(
        self, target_role: str, target_format: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Helper to load sidecar metadata JSON for a specific model role/format."""
        if not self._get_search_paths():
            return None

        dist = self._first_distribution_by_role(target_role, target_format)
        if not dist:
            return None

        # Try exact filename and with .json extension
        base_name = dist.get_filename()
        candidates: list[Path] = []
        for root in self._get_search_paths():
            candidates.extend([root / base_name, root / f"{base_name}.json"])

        for c in candidates:
            if c.exists():
                try:
                    return json.loads(c.read_text(encoding="utf-8"))
                except Exception as e:
                    print(
                        f"Warning: Failed to parse local {target_format} metadata: {e}"
                    )
        return None

    def _sklearn_candidate_paths(self) -> list[Path]:
        """Build sklearn source candidates with metadata-preferred paths first."""
        candidate_paths: list[Path] = []
        seen: set[Path] = set()

        sklearn_dist = self._first_distribution_by_role("sklearn_model")
        if sklearn_dist:
            base_name = sklearn_dist.get_filename()
            suffix_candidates = ("", ".py", ".txt")
            for root in self._get_search_paths():
                for suffix in suffix_candidates:
                    candidate = root / f"{base_name}{suffix}"
                    if candidate.exists() and candidate not in seen:
                        seen.add(candidate)
                        candidate_paths.append(candidate)

        for candidate in self._iter_files_with_extensions("py", "txt"):
            if candidate not in seen:
                seen.add(candidate)
                candidate_paths.append(candidate)

        return candidate_paths

    def _load_as_onnx_model(
        self, providers: Optional[list]
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        """Internal handler for ONNX models."""
        onnx_path = self._find_file_by_extension("onnx")
        if not onnx_path:
            # Fallback check path directly if it was just a mis-detection
            if self._local_model_path and str(self._local_model_path).endswith(".onnx"):
                onnx_path = self._local_model_path

        if not onnx_path:
            raise FileNotFoundError("ONNX file not found in local paths.")

        meta = self._load_sidecar_metadata("onnx_model", target_format=ONNX_FORMAT)
        return load_onnx_model(onnx_path, providers=providers, preloaded_metadata=meta)

    def _load_as_hf_model(
        self,
        model_dir: Union[str, Path],
        token: Optional[str],
        device_map: Union[str, Dict],
        dtype: str,
        trust_remote_code: bool,
        load_task_specific_head: bool,
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        """Internal handler for Hugging Face models."""
        # 1. Determine source
        local_hf_dir = next(
            (p for p in self._get_search_paths() if (p / "config.json").exists()), None
        )
        model_source = str(local_hf_dir.absolute()) if local_hf_dir else self.title

        # 2. Metadata
        meta = self._load_sidecar_metadata("huggingface_model", target_format=HF_FORMAT)

        # 3. Load
        return load_hf_model(
            model_id=model_source,
            token=token,
            device_map=device_map,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            load_task_specific_head=load_task_specific_head,
            cache_dir=model_dir,
            preloaded_metadata=meta,
        )

    def _load_as_sklearn_model(self) -> Tuple[Any, Any, Dict[str, Any]]:
        """Internal handler for sklearn-style models."""
        candidate_paths = self._sklearn_candidate_paths()
        meta = self._load_sidecar_metadata("sklearn_model") or {}

        if not candidate_paths:
            raise FileNotFoundError(
                "No sklearn model source found. Expected a Python script implementing SKLearnModel."
            )

        errors = []
        for candidate in candidate_paths:
            try:
                model = load_sklearn_model(candidate)
                return model, None, meta
            except Exception as e:
                errors.append(f"{candidate.name}: {e}")

        raise RuntimeError(
            "Failed to load sklearn model from available candidates. "
            + " | ".join(errors)
        )

    def load_model(
        self,
        model_dir: Union[str, Path] = "./models",
        token: Optional[str] = None,
        device_map: Union[str, Dict] = "auto",
        dtype: str = "auto",
        trust_remote_code: bool = False,
        load_task_specific_head: bool = True,
        onnx_providers: Optional[list] = None,
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Detect and load exactly one model type (huggingface_model, onnx_model, or
        sklearn_model). Returns a triplet of (model, processor/tokenizer_or_none, metadata).
        """
        model_type = self._detect_model_type()
        loaders: dict[
            Dataset._MODEL_ROLES, Callable[[], Tuple[Any, Any, Dict[str, Any]]]
        ] = {
            "onnx_model": lambda: self._load_as_onnx_model(onnx_providers),
            "huggingface_model": lambda: self._load_as_hf_model(
                model_dir,
                token,
                device_map,
                dtype,
                trust_remote_code,
                load_task_specific_head,
            ),
            "sklearn_model": self._load_as_sklearn_model,
        }
        return loaders[model_type]()

    def __repr__(self) -> str:
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
