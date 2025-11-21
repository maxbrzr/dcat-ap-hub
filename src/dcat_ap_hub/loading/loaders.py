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
    return PdfReader(filepath)


def load_csv(filepath: Path) -> pd.DataFrame:
    return pd.read_csv(filepath)


def load_xlsx_xls(filepath: Path) -> pd.DataFrame:
    return pd.read_excel(filepath)


def load_json(filepath: Path) -> dict:
    with open(filepath, "r") as f:
        return json.load(f)


def load_parquet(filepath: Path) -> pd.DataFrame:
    return pd.read_parquet(filepath)


def load_png_jpeg_jpg(filepath: Path) -> np.ndarray:
    return np.array(cv2.imread(str(filepath)))


def load_npy_npz(filepath: Path) -> np.ndarray:
    return np.load(filepath)


def load_txt(filepath: Path) -> str:
    with open(filepath, "r") as f:
        return f.read()


def load_html(filepath: Path) -> BeautifulSoup:
    with open(filepath, "r", encoding="utf-8") as f:
        return BeautifulSoup(f.read(), "html.parser")


def load_xml(filepath: Path) -> BeautifulSoup:
    with open(filepath, "r", encoding="utf-8") as f:
        return BeautifulSoup(f.read(), "xml.parser")


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
    """Represents one file and its lazy-loadable parsed content."""

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
        """Lazily load file data when first accessed."""
        if self._data is None and self.error is None:
            try:
                self._data = self.loader(self._path_obj)
            except Exception as e:
                self.error = f"{e.__class__.__name__}: {e}"
        return self._data

    def summary(self) -> str:
        dtype = type(self._data).__name__ if self._data is not None else "Lazy"
        size_kb = self.size / 1024
        return (
            f"{self.path:<40} | {self.filetype.value:<8} | {dtype:<12} "
            f"| {size_kb:>8.1f} KB"
        )


class LoadedFiles(dict[str, LoadedFile]):
    """Container for multiple LoadedFile objects with convenience methods."""

    def summary(self) -> None:
        logger.info(f"{'Path':<40} | {'Type':<8} | {'Data Type':<12} | {'Size':>10}")
        logger.info("-" * 80)
        for lf in self.values():
            logger.info(lf.summary())

    def get_errors(self) -> list[LoadedFile]:
        return [lf for lf in self.values() if lf.error]

    def get_by_type(self, filetype: FileType) -> list[LoadedFile]:
        return [lf for lf in self.values() if lf.filetype == filetype]

    def get_dataframes(self) -> list[pd.DataFrame]:
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
    Load files from a directory or single file, with optional lazy loading.

    Args:
        path: Directory or single file to load.
        file_types: Optional list of file types to restrict loading.
        summarize: If True, logger.info a summary table of loaded files.
        lazy: If True, defer file parsing until first data access.

    Returns:
        LoadedFiles container.
    """
    results = LoadedFiles()
    root = dataset_dir

    def register_file(file_path: Path) -> None:
        ext = file_path.suffix.lower().lstrip(".")
        try:
            filetype = FileType(ext)
        except ValueError:
            logger.info(f"Skipping unsupported file: {file_path.name}")
            return

        if file_types and filetype not in file_types:
            return

        loader = FileTypeToLoadFunc.get(filetype)
        if not loader:
            logger.info(f"No loader defined for: {filetype}")
            return

        stat = file_path.stat()
        rel_path = str(file_path.relative_to(root)) if root.is_dir() else file_path.name

        lf = LoadedFile(
            path=rel_path,
            filetype=filetype,
            size=stat.st_size,
            mtime=stat.st_mtime,
            loader=loader,
            _path_obj=file_path,
        )

        # if not lazy, load immediately
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
