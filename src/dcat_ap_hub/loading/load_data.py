"""Utility loaders for many file formats with lazy-loading support.

Provides simple per-file loader functions, a FileType enum and mapping to the
appropriate loader, a LoadedFile dataclass that lazily loads content on first
access, and a load_data helper to scan a path and register files.

This module is intentionally lightweight: each loader returns a parsed object
(or raw content) appropriate for the file type. Use LoadedFile.data to access
parsed content; loading is deferred until first access unless lazy=False.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from pypdf import PdfReader
import cv2
from bs4 import BeautifulSoup
from enum import Enum
from typing import Any, Dict, TypeAlias, Callable

from dataclasses import dataclass, field
from typing import List, Optional

from dcat_ap_hub.logging import logger


# ============================================================
# === 1. Individual File Loaders =============================
# ============================================================


def load_pdf(filepath: Path) -> PdfReader:
    """Load a PDF file and return a PdfReader instance."""
    return PdfReader(filepath)


def load_csv(filepath: Path) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(filepath)


def load_xlsx_xls(filepath: Path) -> pd.DataFrame:
    """Load an Excel file (xlsx or xls) into a pandas DataFrame."""
    return pd.read_excel(filepath)


def load_json(filepath: Path) -> dict:
    """Load a JSON file and return a Python dict."""
    with open(filepath, "r") as f:
        return json.load(f)


def load_parquet(filepath: Path) -> pd.DataFrame:
    """Load a Parquet file into a pandas DataFrame."""
    return pd.read_parquet(filepath)


def load_png_jpeg_jpg(filepath: Path) -> np.ndarray:
    """Load an image (png/jpg/jpeg) and return as a numpy array (BGR via cv2).

    Note: cv2.imread returns None on read failure; callers should handle None
    or check LoadedFile.error when using lazy loading.
    """
    return np.array(cv2.imread(str(filepath)))


def load_npy_npz(filepath: Path) -> np.ndarray:
    """Load a NumPy .npy or .npz file."""
    return np.load(filepath)


def load_txt(filepath: Path) -> str:
    """Read and return the full contents of a text file."""
    with open(filepath, "r") as f:
        return f.read()


def load_html(filepath: Path) -> BeautifulSoup:
    """Parse HTML and return a BeautifulSoup object using the HTML parser."""
    with open(filepath, "r", encoding="utf-8") as f:
        return BeautifulSoup(f.read(), "html.parser")


def load_xml(filepath: Path) -> BeautifulSoup:
    """Parse XML and return a BeautifulSoup object using an XML parser."""
    with open(filepath, "r", encoding="utf-8") as f:
        # Use the generic "xml" parser name for BeautifulSoup
        return BeautifulSoup(f.read(), "xml")


# ============================================================
# === 2. File Type Definitions ===============================
# ============================================================


class FileType(Enum):
    CSV = "csv"
    XLSX = "xlsx"
    XLS = "xls"
    JSON = "json"
    PARQUET = "parquet"
    PNG = "png"
    JPG = "jpg"
    JPEG = "jpeg"
    MP4 = "mp4"
    MOV = "mov"
    XML = "xml"
    HTML = "html"
    PDF = "pdf"
    TXT = "txt"
    NPY = "npy"
    NPZ = "npz"


LoadFunc: TypeAlias = Callable[[Path], Any]

# Mapping from FileType -> loader function used by load_data.
# Keep this mapping explicit to make it easy to extend with new loaders.
FileTypeToLoadFunc: Dict[FileType, LoadFunc] = {
    FileType.PDF: load_pdf,
    FileType.HTML: load_html,
    FileType.XML: load_xml,
    FileType.CSV: load_csv,
    FileType.XLSX: load_xlsx_xls,
    FileType.XLS: load_xlsx_xls,
    FileType.JSON: load_json,
    FileType.PARQUET: load_parquet,
    FileType.PNG: load_png_jpeg_jpg,
    FileType.JPG: load_png_jpeg_jpg,
    FileType.JPEG: load_png_jpeg_jpg,
    FileType.TXT: load_txt,
    FileType.NPY: load_npy_npz,
    FileType.NPZ: load_npy_npz,
}


# ============================================================
# === 3. Data Classes ========================================
# ============================================================


@dataclass
class LoadedFile:
    """Represents a single file entry with metadata and lazy-loaded content.

    Accessing .data will call the provided loader exactly once, caching the
    result. If the loader raises, the exception is caught and a concise
    message is stored in `.error` for inspection; further accesses return None.
    """

    path: str
    filetype: FileType
    size: int
    mtime: float
    loader: LoadFunc
    _path_obj: Path
    _data: Any = field(default=None, repr=False)
    error: Optional[str] = None

    @property
    def data(self) -> Any:
        """Return parsed file data, loading it on first access.

        Loading behavior:
        - If _data is already set, return it.
        - If not, call loader(_path_obj) and cache the result in _data.
        - If an exception occurs, set `error` to a short message and leave
          _data as None so callers can detect failure via .error.
        """
        if self._data is None and self.error is None:
            try:
                # actual load happens here; may be expensive
                self._data = self.loader(self._path_obj)
            except Exception as e:
                # Store a concise error message for inspection later
                self.error = f"{e.__class__.__name__}: {e}"
        return self._data

    def summary(self) -> str:
        """Return a one-line summary string for logging or display."""
        dtype = type(self._data).__name__ if self._data is not None else "Lazy"
        size_kb = self.size / 1024
        return (
            f"{self.path:<40} | {self.filetype.value:<8} | {dtype:<12} "
            f"| {size_kb:>8.1f} KB"
        )


class LoadedFiles(dict[str, LoadedFile]):
    """Container mapping relative path -> LoadedFile with convenience helpers."""

    def summary(self) -> None:
        """Log a table-like summary for all tracked files."""
        logger.info(f"{'Path':<40} | {'Type':<8} | {'Data Type':<12} | {'Size':>10}")
        logger.info("-" * 80)
        for lf in self.values():
            logger.info(lf.summary())

    def get_errors(self) -> list[LoadedFile]:
        """Return list of LoadedFile instances that recorded loading errors."""
        return [lf for lf in self.values() if lf.error]

    def get_by_type(self, filetype: FileType) -> list[LoadedFile]:
        """Return list of LoadedFile instances matching the given FileType."""
        return [lf for lf in self.values() if lf.filetype == filetype]

    def get_dataframes(self) -> list[pd.DataFrame]:
        """Return a list of loaded pandas DataFrames (forces load for each file)."""
        return [lf.data for lf in self.values() if isinstance(lf.data, pd.DataFrame)]


# ============================================================
# === 4. Main Loader =========================================
# ============================================================


def load_data(
    dataset_dir: Path,
    file_types: Optional[List[FileType]] = None,
    summarize: bool = False,
    lazy: bool = True,
) -> LoadedFiles:
    """
    Scan a directory (or single file) and register supported files for loading.

    Args:
        dataset_dir: Directory path or single file path to scan/register.
        file_types: Optional whitelist of FileType values to include.
        summarize: If True, log a summary table after scanning.
        lazy: If True, defer parsing until file.data is accessed.

    Returns:
        LoadedFiles: mapping of relative path -> LoadedFile

    Notes:
        - Unsupported extensions are skipped and logged.
        - If lazy is False, files are loaded immediately so any errors will be
          captured in LoadedFile.error at registration time.
    """
    results = LoadedFiles()
    root = dataset_dir

    def register_file(file_path: Path) -> None:
        # Extract extension without leading dot and normalize to lower-case
        ext = file_path.suffix.lower().lstrip(".")
        try:
            # Convert extension string to FileType enum; ValueError means unsupported
            filetype = FileType(ext)
        except ValueError:
            logger.info(f"Skipping unsupported file: {file_path.name}")
            return

        # If a whitelist is provided, skip types not in it
        if file_types and filetype not in file_types:
            return

        # Lookup loader for this FileType
        loader = FileTypeToLoadFunc.get(filetype)
        if not loader:
            logger.info(f"No loader defined for: {filetype}")
            return

        # Gather file metadata once (size, mtime)
        stat = file_path.stat()

        # Compute a relative path for display; if root is a single file, use its name
        rel_path = str(file_path.relative_to(root)) if root.is_dir() else file_path.name

        lf = LoadedFile(
            path=rel_path,
            filetype=filetype,
            size=stat.st_size,
            mtime=stat.st_mtime,
            loader=loader,
            _path_obj=file_path,
        )

        # If eager loading requested, access .data to force parse now
        if not lazy:
            _ = lf.data

        results[rel_path] = lf

    if root.is_file():
        register_file(root)
    else:
        for file in root.rglob("*"):
            if file.is_file():
                register_file(file)

    if summarize:
        results.summary()

    return results
