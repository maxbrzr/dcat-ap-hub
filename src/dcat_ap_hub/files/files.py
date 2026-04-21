from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from dcat_ap_hub.files.loaders import LOADER_REGISTRY
from dcat_ap_hub.utils.logging import logger


@dataclass
class LazyFile:
    """Represents a file whose content is loaded only on access."""

    path: Path
    _data: Any = field(default=None, repr=False)
    _error: Optional[str] = field(default=None, repr=False)

    @property
    def data(self) -> Any:
        """Load and return the file content."""
        # Return cached data after first successful load.
        if self._data is not None:
            return self._data

        # Avoid retrying after a known failure.
        if self._error:
            return None

        try:
            loader = LOADER_REGISTRY.resolve(self.path)
        except ValueError as exc:
            # Persist unsupported-type errors to keep access idempotent.
            self._error = str(exc)
            return None

        try:
            self._data = loader.load(self.path)
            return self._data
        except Exception as e:
            # Memoize runtime load errors and surface them via repr/logger.
            self._error = str(e)
            logger.error(f"Error loading {self.path.name}: {e}")
            return None

    def __repr__(self) -> str:
        state = "✅ Loaded" if self._data is not None else "💤 Lazy"
        if self._error:
            state = f"❌ Error: {self._error}"
        return f"<File: {self.path.name} ({state})>"


def scan_directory(directory: Path) -> Dict[str, LazyFile]:
    """Scan directory and return a dict of LazyFiles keyed by filename."""
    # Default to the shared registry unless a custom one is injected.
    results = {}
    for f in directory.rglob("*"):
        if f.is_file():
            # Keyed by basename; duplicate filenames in subfolders will overwrite.
            results[f.name] = LazyFile(f)
    return results


class FileCollection:
    """Smart container for downloaded files."""

    def __init__(self, root: Path, files: Dict[str, LazyFile]):
        self.root = root
        self._files = files

    def __getitem__(self, key: str) -> LazyFile:
        if key in self._files:
            return self._files[key]
        matches = [k for k in self._files if key in k]
        if len(matches) == 1:
            return self._files[matches[0]]
        if not matches:
            raise KeyError(f"File '{key}' not found in {self.root.name}.")
        raise KeyError(f"Ambiguous key '{key}'. Matches: {matches}")

    def __iter__(self) -> Iterator[LazyFile]:
        return iter(self._files.values())

    def __len__(self) -> int:
        return len(self._files)

    def filter_by(self, ext: str) -> List[LazyFile]:
        target = ext.lower().lstrip(".")
        return [
            f
            for f in self._files.values()
            if f.path.suffix.lower().lstrip(".") == target
        ]

    @property
    def dataframes(self) -> List[Any]:
        return [
            f.data
            for f in self._files.values()
            if f.path.suffix.lower() in [".csv", ".parquet", ".xlsx", ".xls"]
            and f.data is not None
        ]

    def __repr__(self) -> str:
        return f"<FileCollection: {len(self._files)} files in '{self.root.name}'>"
