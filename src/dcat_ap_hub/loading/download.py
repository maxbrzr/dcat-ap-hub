import os
import zipfile
import tarfile
from pathlib import Path
import mimetypes

import requests
from tqdm import tqdm

from dcat_ap_hub.loading.metadata import (
    fetch_metadata,
    get_dataset_dir,
    parse_metadata,
)


# -------------------------------
# Download helpers
# -------------------------------


def download_file_with_mime(url: str, dest_path: Path, chunk_size: int = 8192) -> Path:
    """
    Download a file from URL, automatically appending the correct file extension
    based on the MIME type returned by the server.
    """
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            content_type = r.headers.get("Content-Type")
            ext = None
            if content_type:
                # Guess extension from MIME type
                ext = mimetypes.guess_extension(content_type.split(";")[0])
            # Fallback: use URL suffix if MIME guess fails
            if not ext and dest_path.suffix:
                ext = dest_path.suffix
            # Append extension if not already present
            if ext and not dest_path.suffix == ext:
                dest_path = dest_path.with_suffix(ext)

            # Download with progress bar
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
        return dest_path
    except Exception as e:
        raise RuntimeError(f"Failed to download file from {url}") from e


# -------------------------------
# Archive extraction
# -------------------------------


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


# -------------------------------
# Main dataset download
# -------------------------------


def download_data(url: str, base_dir: Path | str = Path("./datasets")) -> Path:
    """
    Downloads a dataset using JSON-LD metadata and saves it with correct extensions.
    Automatically extracts archives.
    Returns:
        - Metadata dictionary
    """
    base_dir = Path(base_dir) if isinstance(base_dir, str) else base_dir
    base_dir.mkdir(parents=True, exist_ok=True)

    metadata = fetch_metadata(url)
    dataset, distros = parse_metadata(metadata)
    dataset_dir = get_dataset_dir(dataset, base_dir)

    if dataset_dir.exists():
        print(f"Dataset {dataset_dir} already exists. Skipping download.")
        return dataset_dir

    dataset_dir.mkdir(parents=True, exist_ok=True)

    for distro in distros:
        # Prepare a temporary path without extension
        temp_path = dataset_dir / distro.title
        print(f"Downloading {distro.access_url} to {temp_path}")
        filepath = download_file_with_mime(distro.access_url, temp_path)

        # Extract if it's an archive
        if filepath.suffix in [".zip", ".tgz", ".gz"] or filepath.name.endswith(
            ".tar.gz"
        ):
            extract_archive(filepath, dataset_dir)

    return dataset_dir


# -------------------------------
# Example
# -------------------------------

if __name__ == "__main__":
    dataset_dir = download_data(
        "https://ki-daten.hlrs.de/hub/repo/datasets/dcc5faea-10fd-430b-944b-4ac03383ca9f~~1.jsonld"
    )
