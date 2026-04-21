"""Lazy loading logic for various file formats."""

import json
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import chardet
import cv2
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from pypdf import PdfReader


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


class BaseLoader(ABC):
    """Base class for file loaders."""

    file_types: tuple[FileType, ...] = ()

    @abstractmethod
    def load(self, path: Path) -> Any:
        """Load content from path."""


class CsvLoader(BaseLoader):
    file_types = (FileType.CSV,)

    def _detect_encoding(self, path: Path, nbytes: int = 100000) -> str | None:
        with open(path, "rb") as f:
            raw = f.read(nbytes)
        result = chardet.detect(raw)
        return result["encoding"]

    def load(self, path: Path) -> Any:
        encoding = self._detect_encoding(path)
        return pd.read_csv(path, encoding=encoding, sep=None, engine="python")


class ExcelLoader(BaseLoader):
    file_types = (FileType.XLSX,)

    def load(self, path: Path) -> Any:
        return pd.read_excel(path)


class JsonLoader(BaseLoader):
    file_types = (FileType.JSON, FileType.JSONLD)

    def load(self, path: Path) -> Any:
        return json.loads(path.read_text())


class ParquetLoader(BaseLoader):
    file_types = (FileType.PARQUET,)

    def load(self, path: Path) -> Any:
        return pd.read_parquet(path)


class ImageLoader(BaseLoader):
    file_types = (FileType.PNG, FileType.JPG)

    def load(self, path: Path) -> Any:
        return np.array(cv2.imread(str(path)))


class TextLoader(BaseLoader):
    file_types = (FileType.TXT,)

    def load(self, path: Path) -> Any:
        return path.read_text()


class PdfLoader(BaseLoader):
    file_types = (FileType.PDF,)

    def load(self, path: Path) -> Any:
        return PdfReader(path)


class HtmlLoader(BaseLoader):
    file_types = (FileType.HTML,)

    def load(self, path: Path) -> Any:
        return BeautifulSoup(path.read_text(), "html.parser")


class XmlLoader(BaseLoader):
    file_types = (FileType.XML,)

    def load(self, path: Path) -> Any:
        return BeautifulSoup(path.read_text(), "xml")


class NpyLoader(BaseLoader):
    file_types = (FileType.NPY,)

    def load(self, path: Path) -> Any:
        return np.load(path)


LOADER_CLASSES = (
    CsvLoader,
    ExcelLoader,
    JsonLoader,
    ParquetLoader,
    ImageLoader,
    TextLoader,
    PdfLoader,
    HtmlLoader,
    XmlLoader,
    NpyLoader,
)


class LoaderRegistry:
    """Registry responsible for resolving loaders by file extension."""

    _instance: Optional["LoaderRegistry"] = None
    loaders: Dict[FileType, BaseLoader]

    def __new__(cls) -> "LoaderRegistry":
        if cls._instance is None:
            instance = super().__new__(cls)
            instance.loaders = {}
            # Build the default registry once, then share it process-wide.
            for loader_cls in LOADER_CLASSES:
                instance.register(loader_cls())
            cls._instance = instance
        return cls._instance

    def __init__(self):
        # Singleton setup is handled in __new__.
        return

    def register(self, loader: BaseLoader) -> None:
        # Later registrations override earlier ones for the same extension.
        for file_type in loader.file_types:
            self.loaders[file_type] = loader

    def resolve(self, path: Path) -> BaseLoader:
        # Normalize extension values like ".CSV" to enum keys.
        ext = path.suffix.lower().lstrip(".")
        try:
            ft = FileType(ext)
        except ValueError as exc:
            raise ValueError(f"Unsupported extension: {ext}") from exc

        try:
            return self.loaders[ft]
        except KeyError as exc:
            raise ValueError(f"No loader for {ext}") from exc


LOADER_REGISTRY = LoaderRegistry()
