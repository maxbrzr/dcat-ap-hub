import json
from pathlib import Path
import numpy as np
import pandas as pd
from pypdf import PdfReader
import cv2
from bs4 import BeautifulSoup


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
