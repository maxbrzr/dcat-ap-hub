"""
Core interface for the DCAT-AP Hub.
"""

from __future__ import annotations
import requests
import json
from pathlib import Path
from typing import Optional, Any, Tuple, List, Iterator, Union, Dict

from dcat_ap_hub.internals.models import DatasetMetadata, Distribution
from dcat_ap_hub.internals.parser import fetch_and_parse, parse_local_file
from dcat_ap_hub.internals.transfer import download_dataset_files
from dcat_ap_hub.internals.loaders import scan_directory, LazyAsset
from dcat_ap_hub.internals.integrations import load_hf_model


class FileCollection:
    """Smart container for downloaded files."""

    def __init__(self, root: Path, assets: Dict[str, LazyAsset]):
        self.root = root
        self._assets = assets

    def __getitem__(self, key: str) -> LazyAsset:
        if key in self._assets:
            return self._assets[key]
        matches = [k for k in self._assets if key in k]
        if len(matches) == 1:
            return self._assets[matches[0]]
        if not matches:
            raise KeyError(f"File '{key}' not found in {self.root.name}.")
        raise KeyError(f"Ambiguous key '{key}'. Matches: {matches}")

    def __iter__(self) -> Iterator[LazyAsset]:
        return iter(self._assets.values())

    def __len__(self) -> int:
        return len(self._assets)

    def filter_by(self, ext: str) -> List[LazyAsset]:
        target = ext.lower().lstrip(".")
        return [
            f
            for f in self._assets.values()
            if f.path.suffix.lower().lstrip(".") == target
        ]

    @property
    def dataframes(self) -> List[Any]:
        return [
            f.data
            for f in self._assets.values()
            if f.path.suffix.lower() in [".csv", ".parquet", ".xlsx", ".xls"]
            and f.data is not None
        ]

    def __repr__(self) -> str:
        return f"<FileCollection: {len(self._assets)} files in '{self.root.name}'>"


class Dataset:
    """
    The main entry point for interacting with DCAT-AP datasets and models.
    """

    def __init__(
        self, meta: DatasetMetadata, local_data_path: Optional[Path] = None
    ) -> None:
        self._meta = meta
        # Path to generic files (CSVs, JSONs, etc.)
        self._local_data_path = local_data_path
        # Path to model weights (set only after load_model is called or detected)
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
        is_model_guess = (p / "config.json").exists()

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
            # If it's a model, we usually consider the data path the same
            # unless specified otherwise, but strict separation is safer.
        else:
            ds._local_data_path = p

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
        target_dir: Union[str, Path] = "./data",
        force: bool = False,
        verbose: bool = True,
    ) -> FileCollection:
        """
        Download dataset files to the data directory.
        Does NOT affect where load_model looks for weights.
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
            self._meta, Path(target_dir), force=force, verbose=verbose
        )

        # Set DATA path specifically
        self._local_data_path = path

        self._save_metadata(path, verbose=verbose)
        return FileCollection(path, scan_directory(path))

    def load_model(
        self,
        token: Optional[str] = None,
        device_map: Union[str, Dict] = "auto",
        dtype: str = "auto",
        trust_remote_code: bool = False,
        load_task_specific_head: bool = True,
        cache_dir: Union[str, Path] = Path("./models"),
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Load as Hugging Face model.
        Uses 'base_dir' for model weights (completely separate from data download dir).
        """
        if not self.is_model and not (
            self._local_model_path and (self._local_model_path / "config.json").exists()
        ):
            raise ValueError(
                f"Dataset '{self.title}' is not marked as a Machine Learning Model."
            )

        # 1. Determine Source
        # Priority:
        # A. Existing _local_model_path (we loaded from a model dir)
        # B. Title (Model ID) -> Let Hugging Face manage the download into 'base_dir'

        if self._local_model_path and (self._local_model_path / "config.json").exists():
            model_source = str(self._local_model_path.absolute())
            if self._local_data_path:
                pass  # We ignore the data path entirely for model loading
        else:
            model_source = self.title

        # 2. Call Integration
        # Hugging Face will cache into 'cache_dir' if provided.
        model, tokenizer, meta = load_hf_model(
            model_id=model_source,
            token=token,
            device_map=device_map,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            load_task_specific_head=load_task_specific_head,
            cache_dir=cache_dir,
        )

        # 3. Update model path if successful
        # If we used the HF Hub, 'base_dir' acts as the cache,
        # but HF manages the internal structure (blobs/snapshots).
        # We generally don't set _local_model_path to the specific snapshot hash
        # unless we want to lock it, but for simplicity we leave it None if managed by HF cache.

        return model, tokenizer, meta

    def __repr__(self) -> str:
        icon = "🧠" if self.is_model else "📊"
        # Show both paths if they exist
        locs = []
        if self._local_data_path:
            locs.append(f"Data: {self._local_data_path.name}")
        if self._local_model_path:
            locs.append(f"Model: {self._local_model_path.name}")

        loc_str = f" [{', '.join(locs)}]" if locs else ""
        return f"{icon} Dataset('{self.title}', {len(self._meta.distributions)} distros){loc_str}"
