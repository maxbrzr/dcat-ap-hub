"""
Utilities for downloading datasets described by JSON-LD metadata and for
handling archive extraction.

- download_file_with_mime: stream download and ensure file extension matches
  server MIME type (fallback to provided path suffix).
- extract_archive: extract zip and tar.gz/tgz archives, including nested ones.
- download_data: orchestrate metadata retrieval, downloads, and extraction.

All functions keep existing logging behavior; only documentation and inline
comments were added for clarity.
"""

import os
import zipfile
import tarfile
from pathlib import Path
import mimetypes

import requests
from tqdm import tqdm

from dcat_ap_hub.metadata.metadata import Dataset, get_metadata
from dcat_ap_hub.logging import logger


# -------------------------------
# Download helpers
# -------------------------------


def get_dataset_dir(dataset: Dataset, base_dir: Path) -> Path:
    """Return the filesystem directory for a dataset based on its title.

    Args:
        dataset: Dataset metadata object containing at least a 'title'.
        base_dir: Base path under which the dataset directory will be created.

    Returns:
        Path to the dataset directory (base_dir / dataset.title).
    """
    base_path = Path(base_dir)
    dataset_dir = base_path / dataset.title
    return dataset_dir


def download_file_with_mime(
    url: str, dest_path: Path, chunk_size: int = 8192, verbose: bool = False
) -> Path:
    """Download a file from URL, adjusting file extension based on MIME type.

    This function:
    - streams the response to disk to avoid large memory usage,
    - attempts to determine the correct extension from the Content-Type header,
    - falls back to the provided path suffix if MIME inference fails,
    - returns the final Path where the file was saved.

    Args:
        url: The URL to download.
        dest_path: Desired destination path (may be adjusted to include extension).
        chunk_size: Size of chunks to read from the response stream.
        verbose: Whether to emit detailed logging and show a tqdm progress bar.

    Raises:
        RuntimeError: If any network or filesystem error occurs.
    """
    try:
        if verbose:
            logger.info(
                f"[download] Starting download: url='{url}', base_path='{dest_path}'"
            )
        with requests.get(url, stream=True) as r:
            r.raise_for_status()

            # Determine extension from Content-Type header (e.g. 'text/csv; charset=utf-8')
            content_type = r.headers.get("Content-Type")
            if verbose:
                logger.info(f"[download] Content-Type header: {content_type}")

            ext = None
            if content_type:
                ext = mimetypes.guess_extension(content_type.split(";")[0])
                if verbose:
                    logger.info(f"[download] Guessed extension from MIME: {ext}")

            # If MIME-based extension not found, fall back to provided suffix
            if not ext and dest_path.suffix:
                ext = dest_path.suffix
                if verbose:
                    logger.info(
                        f"[download] Fallback extension from path suffix: {ext}"
                    )

            # Apply the chosen extension if different from current
            if ext and not dest_path.suffix == ext:
                if verbose:
                    logger.info(f"[download] Applying extension: {ext}")
                dest_path = dest_path.with_suffix(ext)

            # Stream to disk with a progress bar when verbose
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
                    # write chunk to disk; skip empty keep-alive chunks if any
                    if not chunk:
                        continue
                    f.write(chunk)
                    pbar.update(len(chunk))

        if verbose:
            size = dest_path.stat().st_size if dest_path.exists() else 0
            logger.info(f"[download] Finished: saved='{dest_path}', size={size} bytes")
        return dest_path
    except Exception as e:
        # Preserve original exception semantics but add context
        raise RuntimeError(f"Failed to download file from {url}") from e


# -------------------------------
# Archive extraction
# -------------------------------


def extract_archive(filepath: Path, target_dir: Path, verbose: bool = False) -> None:
    """Recursively extract supported archive formats into a target directory.

    Supports:
        - .zip
        - .tar.gz and .tgz

    Nested archives found in the extracted contents are detected and extracted
    in turn (breadth-first).

    Args:
        filepath: Path to the archive to extract.
        target_dir: Directory where extracted contents should be placed.
        verbose: Whether to emit detailed logging.

    Raises:
        RuntimeError: If extraction fails for any reason.
    """

    def is_archive(file: Path) -> bool:
        """Return True if the given file looks like a supported archive."""
        return (
            file.suffix == ".zip"
            or file.suffixes[-2:] in [[".tar", ".gz"]]
            or file.suffix == ".tgz"
        )

    def extract_one(file: Path, extract_to: Path) -> None:
        """Extract a single archive into extract_to and delete the archive file."""
        if verbose:
            logger.info(f"[extract] Extracting '{file.name}' into '{extract_to}'")
        if file.suffix == ".zip":
            with zipfile.ZipFile(file, "r") as zip_ref:
                zip_ref.extractall(extract_to)
        elif file.suffixes[-2:] in [[".tar", ".gz"]] or file.suffix == ".tgz":
            # tarfile can handle gz-compressed tar archives
            with tarfile.open(file, "r:gz") as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            raise ValueError(f"Unsupported archive format: {file.name}")

        # Remove the archive after successful extraction to avoid re-processing.
        if verbose:
            logger.info(f"[extract] Removing archive '{file.name}'")
        file.unlink()

    try:
        # Breadth-first queue of (archive_path, destination_dir)
        queue = [(filepath, target_dir)]
        if verbose:
            logger.info(f"[extract] Initial archive queue: {[str(filepath)]}")

        while queue:
            archive_path, dest_dir = queue.pop(0)
            extract_one(archive_path, dest_dir)

            # Scan the newly extracted files for nested archives and enqueue them.
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
        # Preserve original exception semantics but add context
        raise RuntimeError(f"Failed to extract archive: {filepath}") from e


# -------------------------------
# Main dataset download
# -------------------------------


def download_data(
    url: str, base_dir: Path | str = Path("./datasets"), verbose: bool = False
) -> Path:
    """High-level workflow to download dataset distributions and extract archives.

    Steps:
        1. Ensure base_dir exists.
        2. Retrieve dataset metadata and distributions via get_metadata().
        3. For each distribution, download the file using download_file_with_mime().
        4. If the downloaded file is an archive, extract it into the dataset directory.

    Args:
        url: Metadata JSON-LD URL for the dataset.
        base_dir: Base directory to store datasets.
        verbose: Whether to emit detailed logging.

    Returns:
        Path to the dataset directory (created if necessary).
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
        # Avoid re-downloading an existing dataset directory.
        if verbose:
            logger.info(
                "[dataset] Dataset directory already exists. Skipping downloads."
            )
        else:
            logger.info(f"Dataset {dataset_dir} already exists. Skipping download.")
        return dataset_dir

    dataset_dir.mkdir(parents=True, exist_ok=True)

    for i, distro in enumerate(distros):
        # Prepare a temporary path without extension (we'll apply correct extension after download)
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
            # Fall back to access_url if download_url is not provided
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

        # Detect common archive types and extract them
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
