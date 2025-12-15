"""Lazy loading logic for various file formats."""

import json
from pathlib import Path
from enum import Enum
from typing import Any, Callable, Dict, Optional, TypeAlias
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import cv2
from pypdf import PdfReader
from bs4 import BeautifulSoup

from dcat_ap_hub.internals.logging import logger


class FileType(Enum):
    CSV = "csv"
    XLSX = "xlsx"
    JSON = "json"
    JSONLD = "jsonld"
    PARQUET = "parquet"
    PNG = "png"
    JPG = "jpg"
    TXT = "txt"
    PDF = "pdf"
    HTML = "html"
    XML = "xml"
    NPY = "npy"


LoadFunc: TypeAlias = Callable[[Path], Any]

# --- Loader Implementations ---


def _load_csv(p: Path) -> Any:
    return pd.read_csv(p)


def _load_excel(p: Path) -> Any:
    return pd.read_excel(p)


def _load_json(p: Path) -> Any:
    return json.loads(p.read_text())


def _load_parquet(p: Path) -> Any:
    return pd.read_parquet(p)


def _load_img(p: Path) -> Any:
    return np.array(cv2.imread(str(p)))


def _load_txt(p: Path) -> Any:
    return p.read_text()


def _load_pdf(p: Path) -> Any:
    return PdfReader(p)


def _load_bs4(p: Path, parser="html.parser") -> Any:
    return BeautifulSoup(p.read_text(), parser)


def _load_npy(p: Path) -> Any:
    return np.load(p)


LOADERS: Dict[FileType, LoadFunc] = {
    FileType.CSV: _load_csv,
    FileType.XLSX: _load_excel,
    FileType.JSON: _load_json,
    FileType.JSONLD: _load_json,
    FileType.PARQUET: _load_parquet,
    FileType.PNG: _load_img,
    FileType.JPG: _load_img,
    FileType.TXT: _load_txt,
    FileType.PDF: _load_pdf,
    FileType.HTML: lambda p: _load_bs4(p, "html.parser"),
    FileType.XML: lambda p: _load_bs4(p, "xml"),
    FileType.NPY: _load_npy,
}


@dataclass
class LazyAsset:
    """Represents a file whose content is loaded only on access."""

    path: Path
    _data: Any = field(default=None, repr=False)
    _error: Optional[str] = field(default=None, repr=False)

    @property
    def data(self) -> Any:
        """Load and return the file content."""
        if self._data is not None:
            return self._data

        if self._error:
            return None

        # Determine type
        ext = self.path.suffix.lower().lstrip(".")
        try:
            ft = FileType(ext)
            loader = LOADERS.get(ft)
        except ValueError:
            self._error = f"Unsupported extension: {ext}"
            return None

        if not loader:
            self._error = f"No loader for {ext}"
            return None

        try:
            self._data = loader(self.path)
            return self._data
        except Exception as e:
            self._error = str(e)
            logger.error(f"Error loading {self.path.name}: {e}")
            return None

    def __repr__(self) -> str:
        state = "✅ Loaded" if self._data is not None else "💤 Lazy"
        if self._error:
            state = f"❌ Error: {self._error}"
        return f"<File: {self.path.name} ({state})>"


def scan_directory(directory: Path) -> Dict[str, LazyAsset]:
    """Scan directory and return a dict of LazyAssets keyed by filename."""
    results = {}
    for f in directory.rglob("*"):
        if f.is_file():
            results[f.name] = LazyAsset(f)
    return results
