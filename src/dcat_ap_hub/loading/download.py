import os
import zipfile
import tarfile
from pathlib import Path
import mimetypes

import requests
from tqdm import tqdm

from dcat_ap_hub.loading.metadata import Dataset, get_metadata
from dcat_ap_hub.logging import logger


# -------------------------------
# Download helpers
# -------------------------------


def get_dataset_dir(dataset: Dataset, base_dir: Path) -> Path:
    base_path = Path(base_dir)
    dataset_dir = base_path / dataset.title
    return dataset_dir


def download_file_with_mime(
    url: str, dest_path: Path, chunk_size: int = 8192, verbose: bool = False
) -> Path:
    """
    Download a file from URL, automatically appending the correct file extension
    based on the MIME type returned by the server.
    """
    try:
        if verbose:
            logger.info(
                f"[download] Starting download: url='{url}', base_path='{dest_path}'"
            )
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            content_type = r.headers.get("Content-Type")
            if verbose:
                logger.info(f"[download] Content-Type header: {content_type}")
            ext = None
            if content_type:
                ext = mimetypes.guess_extension(content_type.split(";")[0])
                if verbose:
                    logger.info(f"[download] Guessed extension from MIME: {ext}")
            if not ext and dest_path.suffix:
                ext = dest_path.suffix
                if verbose:
                    logger.info(
                        f"[download] Fallback extension from path suffix: {ext}"
                    )
            if ext and not dest_path.suffix == ext:
                if verbose:
                    logger.info(f"[download] Applying extension: {ext}")
                dest_path = dest_path.with_suffix(ext)

            total = int(r.headers.get("content-length", 0))
            if verbose:
                logger.info(f"[download] Content-Length: {total} bytes")
            with (
                open(dest_path, "wb") as f,
                tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    disable=not verbose,
                    desc=f"Downloading {dest_path.name}",
                ) as pbar,
            ):
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(len(chunk))
        if verbose:
            size = dest_path.stat().st_size if dest_path.exists() else 0
            logger.info(f"[download] Finished: saved='{dest_path}', size={size} bytes")
        return dest_path
    except Exception as e:
        raise RuntimeError(f"Failed to download file from {url}") from e


# -------------------------------
# Archive extraction
# -------------------------------


def extract_archive(filepath: Path, target_dir: Path, verbose: bool = False) -> None:
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
        if verbose:
            logger.info(f"[extract] Extracting '{file.name}' into '{extract_to}'")
        if file.suffix == ".zip":
            with zipfile.ZipFile(file, "r") as zip_ref:
                zip_ref.extractall(extract_to)
        elif file.suffixes[-2:] in [[".tar", ".gz"]] or file.suffix == ".tgz":
            with tarfile.open(file, "r:gz") as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            raise ValueError(f"Unsupported archive format: {file.name}")
        if verbose:
            logger.info(f"[extract] Removing archive '{file.name}'")
        file.unlink()  # remove archive after extraction

    try:
        queue = [(filepath, target_dir)]
        if verbose:
            logger.info(f"[extract] Initial archive queue: {[str(filepath)]}")
        while queue:
            archive_path, dest_dir = queue.pop(0)
            extract_one(archive_path, dest_dir)
            # Scan for newly extracted archives
            for root, _, files in os.walk(dest_dir):
                for name in files:
                    path = Path(root) / name
                    if is_archive(path):
                        queue.append((path, Path(root)))
                        if verbose:
                            logger.info(f"[extract] Queued nested archive: {path}")
        if verbose:
            logger.info(f"[extract] Extraction complete for '{filepath}'")
    except Exception as e:
        raise RuntimeError(f"Failed to extract archive: {filepath}") from e


# -------------------------------
# Main dataset download
# -------------------------------


def download_data(
    url: str, base_dir: Path | str = Path("./datasets"), verbose: bool = False
) -> Path:
    """
    Downloads a dataset using JSON-LD metadata and saves it with correct extensions.
    Automatically extracts archives.
    Returns:
        - Metadata dictionary
    """
    if verbose:
        logger.info(f"[dataset] Starting download workflow for metadata URL: {url}")
    base_dir = Path(base_dir) if isinstance(base_dir, str) else base_dir
    base_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        logger.info(f"[dataset] Base directory: {base_dir}")

    dataset, distros = get_metadata(url, verbose=verbose)

    dataset_dir = get_dataset_dir(dataset, base_dir)
    if verbose:
        logger.info(
            f"[dataset] Dataset title='{dataset.title}' -> directory='{dataset_dir}'"
        )
        logger.info(f"[dataset] Distributions count: {len(distros)}")

    if dataset_dir.exists():
        if verbose:
            logger.info(
                "[dataset] Dataset directory already exists. Skipping downloads."
            )
        else:
            logger.info(f"Dataset {dataset_dir} already exists. Skipping download.")
        return dataset_dir

    dataset_dir.mkdir(parents=True, exist_ok=True)

    for i, distro in enumerate(distros):
        # Prepare a temporary path without extension
        temp_path = dataset_dir / distro.title
        if distro.download_url:
            if verbose:
                logger.info(
                    f"[distro {i}] Downloading download_url='{distro.download_url}' -> temp_path='{temp_path}'"
                )
            else:
                logger.info(f"Downloading {distro.download_url} to {temp_path}")
            filepath = download_file_with_mime(
                distro.download_url, temp_path, verbose=verbose
            )
        else:
            if verbose:
                logger.info(
                    f"[distro {i}] No download_url for distribution '{distro.title}'. Using access_url instead."
                )
            else:
                logger.info(
                    f"No download_url for distribution '{distro.title}'. Using access_url instead."
                )
            filepath = download_file_with_mime(
                distro.access_url, temp_path, verbose=verbose
            )

        # Extract if it's an archive
        if filepath.suffix in [".zip", ".tgz", ".gz"] or filepath.name.endswith(
            ".tar.gz"
        ):
            if verbose:
                logger.info(
                    f"[distro {i}] Archive detected: '{filepath.name}' -> extracting"
                )
            extract_archive(filepath, dataset_dir, verbose=verbose)
        else:
            if verbose:
                logger.info(
                    f"[distro {i}] Non-archive file retained: '{filepath.name}'"
                )

    if verbose:
        files = list(dataset_dir.rglob("*"))
        logger.info(f"[dataset] Download complete. Total files: {len(files)}")
    return dataset_dir


# -------------------------------
# Example
# -------------------------------

if __name__ == "__main__":
    dataset_dir = download_data(
        "https://ki-daten.hlrs.de/hub/repo/datasets/dcc5faea-10fd-430b-944b-4ac03383ca9f~~1.jsonld",
        verbose=True,
    )
