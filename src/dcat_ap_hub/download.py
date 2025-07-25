import os
import shutil
import zipfile
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

from dcat_ap_hub.metadata import fetch_metadata, get_data_download_url, get_dataset_dir


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


def download_data(json_ld_handle: str, base_dir: Path = Path("./datasets")) -> dict:
    """
    Downloads dataset and optionally a parser using JSON-LD metadata.
    Returns:
        - Dataset path
        - Parser function (or None)
    """
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    metadata = fetch_metadata(json_ld_handle)
    data_download_url = get_data_download_url(metadata)
    dataset_dir = get_dataset_dir(metadata, base_path)

    if dataset_dir.exists():
        return metadata

    dataset_dir.mkdir(parents=True, exist_ok=True)

    filename = data_download_url.split("/")[-1]
    filepath = dataset_dir / filename

    download_file_with_progress(data_download_url, filepath)

    if filepath.suffix in [".zip", ".tgz", ".gz"] or filepath.name.endswith(".tar.gz"):
        extract_archive(filepath, dataset_dir)

    return metadata
