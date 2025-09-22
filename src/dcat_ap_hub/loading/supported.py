from enum import Enum
from pathlib import Path
from typing import Any, Dict, TypeAlias, Callable

from dcat_ap_hub.loading.loaders import (
    load_html,
    load_xml,
    load_pdf,
    load_csv,
    load_xlsx_xls,
    load_json,
    load_parquet,
    load_png_jpeg_jpg,
)


class SupportedFileType(Enum):
    # Tabular / structured data
    CSV = "csv"
    XLSX = "xlsx"
    XLS = "xls"
    JSON = "json"
    PARQUET = "parquet"

    # Images
    PNG = "png"
    JPG = "jpg"
    JPEG = "jpeg"

    # Video
    MP4 = "mp4"
    MOV = "mov"

    # Web data
    XML = "xml"
    HTML = "html"

    # Documents
    PDF = "pdf"


Load: TypeAlias = Callable[[Path], Any]


FileTypeToParseMap: Dict[SupportedFileType, Load] = {
    SupportedFileType.PDF: load_pdf,
    SupportedFileType.HTML: load_html,
    SupportedFileType.XML: load_xml,
    SupportedFileType.CSV: load_csv,
    SupportedFileType.XLSX: load_xlsx_xls,
    SupportedFileType.XLS: load_xlsx_xls,
    SupportedFileType.JSON: load_json,
    SupportedFileType.PARQUET: load_parquet,
    SupportedFileType.PNG: load_png_jpeg_jpg,
    SupportedFileType.JPEG: load_png_jpeg_jpg,
    SupportedFileType.JPG: load_png_jpeg_jpg,
}
