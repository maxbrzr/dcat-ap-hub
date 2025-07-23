import json
import os
import shutil
import zipfile
import tarfile
from pathlib import Path
from typing import Any, Callable, Optional, Tuple
from urllib import request
from importlib.util import spec_from_file_location, module_from_spec

import requests
from tqdm import tqdm


def download_metadata(json_ld_handle: str) -> dict[str, Any]:
    """
    Downloads and parses a JSON-LD metadata file.
    """
    if not json_ld_handle.endswith(".jsonld"):
        raise ValueError(f"Expected a .jsonld file, got: {json_ld_handle}")

    try:
        with request.urlopen(json_ld_handle) as response:
            return json.load(response)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download or parse metadata from {json_ld_handle}"
        ) from e


def download_file(url: str, dest_path: Path) -> None:
    """
    Downloads a file from a URL to a destination path.
    """
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                shutil.copyfileobj(r.raw, f)
    except Exception as e:
        raise RuntimeError(f"Failed to download file from {url}") from e


def download_file_with_progress(
    url: str, dest_path: Path, chunk_size: int = 8192
) -> None:
    """
    Downloads a file with a progress bar.
    """
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with (
                open(dest_path, "wb") as f,
                tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc=f"Downloading {dest_path.name}",
                ) as pbar,
            ):
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(len(chunk))
    except Exception as e:
        raise RuntimeError(f"Failed to download file from {url}") from e


def extract_archive(filepath: Path, target_dir: Path) -> None:
    """
    Recursively extracts .zip, .tar.gz, or .tgz files, including nested archives.
    """

    def is_archive(file: Path) -> bool:
        return (
            file.suffix == ".zip"
            or file.suffixes[-2:] in [[".tar", ".gz"]]
            or file.suffix == ".tgz"
        )

    def extract_one(file: Path, extract_to: Path) -> None:
        if file.suffix == ".zip":
            with zipfile.ZipFile(file, "r") as zip_ref:
                zip_ref.extractall(extract_to)
        elif file.suffixes[-2:] in [[".tar", ".gz"]] or file.suffix == ".tgz":
            with tarfile.open(file, "r:gz") as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            raise ValueError(f"Unsupported archive format: {file.name}")
        file.unlink()  # remove archive after extraction

    try:
        queue = [(filepath, target_dir)]

        while queue:
            archive_path, dest_dir = queue.pop(0)

            extract_one(archive_path, dest_dir)

            # Scan for newly extracted archives
            for root, _, files in os.walk(dest_dir):
                for name in files:
                    path = Path(root) / name
                    if is_archive(path):
                        queue.append((path, Path(root)))

    except Exception as e:
        raise RuntimeError(f"Failed to extract archive: {filepath}") from e


def download_dataset(download_url: str, output_dir: Path) -> Path:
    """
    Downloads and extracts a dataset to the specified directory.
    """
    if output_dir.exists():
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    filename = download_url.split("/")[-1]
    filepath = output_dir / filename

    download_file_with_progress(download_url, filepath)

    if filepath.suffix in [".zip", ".tgz", ".gz"] or filepath.name.endswith(".tar.gz"):
        extract_archive(filepath, output_dir)

    return output_dir


def download_parser(download_url: str, output_dir: Path) -> Callable[[str], Any]:
    """
    Downloads and dynamically loads a parser.py from a zip archive.
    """
    parser_path = output_dir / "parser.py"
    if not parser_path.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        zip_path = output_dir / "parser.zip"

        download_file_with_progress(download_url, zip_path)
        extract_archive(zip_path, output_dir)

    # Dynamically load parser.py
    spec = spec_from_file_location("parser_module", parser_path)
    if spec is None or spec.loader is None:
        raise ImportError("Failed to load parser module.")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "parse"):
        raise AttributeError("Parser module must define a 'parse' function.")

    return module.parse


def download(
    dataset_metadata_handle: str,
    parser_metadata_handle: Optional[str] = None,
    base_dir: str = "./datasets",
) -> Tuple[Path, Optional[Callable[[str], Any]]]:
    """
    Downloads dataset and optionally a parser using JSON-LD metadata.
    Returns:
        - Dataset path
        - Parser function (or None)
    """
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    dataset_metadata = download_metadata(dataset_metadata_handle)
    dataset_title = dataset_metadata.get("dct:title")
    dataset_download_url = dataset_metadata.get("dcat:downloadURL", {}).get("@id")

    if not dataset_title or not dataset_download_url:
        raise KeyError(
            "Dataset metadata missing required fields: 'dct:title' or 'dcat:downloadURL.@id'"
        )

    dataset_dir = base_path / dataset_title
    dataset_path = download_dataset(dataset_download_url, dataset_dir)

    if not parser_metadata_handle:
        return dataset_path, None

    parser_metadata = download_metadata(parser_metadata_handle)
    parser_title = parser_metadata.get("dct:title")
    parser_download_url = parser_metadata.get("dcat:downloadURL", {}).get("@id")

    if not parser_title or not parser_download_url:
        raise KeyError(
            "Parser metadata missing required fields: 'dct:title' or 'dcat:downloadURL.@id'"
        )

    parser_dir = base_path / parser_title
    parse = download_parser(parser_download_url, parser_dir)

    return dataset_path, parse
