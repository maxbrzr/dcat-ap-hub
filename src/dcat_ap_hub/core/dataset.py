"""
Core interface for the DCAT-AP Hub.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import requests

from dcat_ap_hub.core.files import FileCollection
from dcat_ap_hub.internals.constants import HF_FORMAT, ONNX_FORMAT
from dcat_ap_hub.internals.integrations import load_hf_model, load_onnx_model
from dcat_ap_hub.internals.loaders import scan_directory
from dcat_ap_hub.internals.models import DatasetMetadata, Distribution
from dcat_ap_hub.internals.parser import fetch_and_parse, parse_local_file
from dcat_ap_hub.internals.processor import apply_processor_logic
from dcat_ap_hub.internals.transfer import download_dataset_files


class Dataset:
    """
    The main entry point for interacting with DCAT-AP datasets and models.
    """

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
        is_model_guess = (p / "config.json").exists() or any(p.glob("*.onnx"))

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
            response = requests.get(self._meta.source_url, timeout=10)
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

        # 3. Find the processor distribution
        processor_dist = next(
            (d for d in self._meta.distributions if d.role == "processor"), None
        )
        if not processor_dist:
            raise ValueError("No processor distribution found in metadata.")

        # 4. Resolve paths
        processor_filename = processor_dist.get_filename()
        processor_path = self._local_data_path / processor_filename

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

    def _find_file_by_extension(self, extension: str) -> Optional[Path]:
        """Helper to find a file with a specific extension in available paths."""
        search_paths = [self._local_model_path, self._local_data_path]
        for p in search_paths:
            if p and p.exists():
                candidates = list(p.glob(f"*.{extension.lstrip('.')}"))
                if candidates:
                    return candidates[0]
        return None

    def _load_sidecar_metadata(self, target_format: str) -> Optional[Dict[str, Any]]:
        """Helper to load sidecar metadata JSON for a specific model format."""
        if not self._local_data_path:
            return None

        dist = next(
            (
                d
                for d in self._meta.distributions
                if d.role == "model" and d.format == target_format
            ),
            None,
        )
        if not dist:
            return None

        # Try exact filename and with .json extension
        base_name = dist.get_filename()
        candidates = [
            self._local_data_path / base_name,
            self._local_data_path / f"{base_name}.json",
        ]

        for c in candidates:
            if c.exists():
                try:
                    return json.loads(c.read_text(encoding="utf-8"))
                except Exception as e:
                    print(
                        f"Warning: Failed to parse local {target_format} metadata: {e}"
                    )
        return None

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

        meta = self._load_sidecar_metadata(ONNX_FORMAT)
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
        if self._local_model_path and (self._local_model_path / "config.json").exists():
            model_source = str(self._local_model_path.absolute())
        else:
            model_source = self.title

        # 2. Metadata
        meta = self._load_sidecar_metadata(HF_FORMAT)

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
        Load as Hugging Face or ONNX model.
        """
        # Detection
        has_onnx = self._find_file_by_extension("onnx") is not None
        should_be_onnx = has_onnx or any(
            d.role == "model" and d.format == ONNX_FORMAT
            for d in self._meta.distributions
        )

        # Dispatch
        if should_be_onnx:
            return self._load_as_onnx_model(onnx_providers)

        return self._load_as_hf_model(
            model_dir,
            token,
            device_map,
            dtype,
            trust_remote_code,
            load_task_specific_head,
        )

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
