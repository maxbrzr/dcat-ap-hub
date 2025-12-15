from pathlib import Path
from typing import Any, Dict, Iterator, List
from dcat_ap_hub.internals.loaders import LazyAsset


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
