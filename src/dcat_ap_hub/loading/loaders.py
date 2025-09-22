import json
from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd
from pypdf import PdfReader
import cv2
from bs4 import BeautifulSoup

from dcat_ap_hub.loading.supported import FileTypeToParseMap, SupportedFileType


def load_files_from_metadata(path: Path) -> Dict[str, Any]:
    """
    Recursively loads supported files from a directory or a single file.
    Returns a dictionary mapping relative file paths (as str) to parsed content.
    """
    results: Dict[str, Any] = {}
    root = path.resolve()

    def load_file(file_path: Path) -> None:
        """Helper to load a single file if supported."""
        ext = file_path.suffix.lower().lstrip(".")
        try:
            filetype = SupportedFileType(ext)
        except ValueError:
            return  # unsupported

        loader = FileTypeToParseMap.get(filetype)
        if loader:
            rel_path = (
                str(file_path.relative_to(root)) if root.is_dir() else file_path.name
            )
            try:
                results[rel_path] = loader(file_path)
            except Exception as e:
                results[rel_path] = {"error": str(e)}

    if root.is_file():
        load_file(root)
    else:
        for file in root.rglob("*"):
            if file.is_file():
                load_file(file)

    return results


def load_pdf(filepath: Path) -> PdfReader:
    return PdfReader(filepath)


def load_csv(filepath: Path) -> pd.DataFrame:
    return pd.read_csv(filepath)


def load_xlsx_xls(filepath: Path) -> pd.DataFrame:
    return pd.read_excel(filepath)


def load_json(filepath: Path) -> dict:
    with open(filepath, "r") as file:
        data = json.load(file)
    return data


def load_parquet(filepath: Path) -> pd.DataFrame:
    return pd.read_parquet(filepath)


def load_png_jpeg_jpg(filepath: Path) -> np.ndarray:
    return np.array(cv2.imread(str(filepath)))


def load_npy_npz(filepath: Path) -> np.ndarray:
    return np.load(filepath)


def load_txt(filepath: Path) -> str:
    with open(filepath, "r") as file:
        data = file.read()
    return data


def load_html(filepath: Path) -> BeautifulSoup:
    with open(filepath, "r", encoding="utf-8") as file:
        contents = file.read()
    return BeautifulSoup(contents, "html.parser")


def load_xml(filepath: Path) -> BeautifulSoup:
    with open(filepath, "r", encoding="utf-8") as file:
        contents = file.read()
    return BeautifulSoup(contents, "xml.parser")
