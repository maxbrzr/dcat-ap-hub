"""Download and extraction utilities."""

import mimetypes
import os
import zipfile
import tarfile
import requests
from pathlib import Path
from tqdm import tqdm

from dcat_ap_hub.internals.logging import logger
from dcat_ap_hub.internals.models import DatasetMetadata


def _extract_archive(filepath: Path, target_dir: Path) -> None:
    """Recursively extract zip/tar/tgz archives."""

    def is_archive(f: Path) -> bool:
        return f.suffix == ".zip" or f.name.endswith((".tar.gz", ".tgz"))

    # Queue of (archive_path, extract_to_dir)
    queue = [(filepath, target_dir)]

    while queue:
        current_file, current_target = queue.pop(0)

        try:
            extracted = False
            if current_file.suffix == ".zip":
                with zipfile.ZipFile(current_file, "r") as z:
                    z.extractall(current_target)
                extracted = True
            elif current_file.name.endswith((".tar.gz", ".tgz")):
                with tarfile.open(current_file, "r:gz") as t:
                    t.extractall(current_target)
                extracted = True

            if extracted:
                logger.info(f"[extract] Extracted {current_file.name}")
                current_file.unlink()  # Delete archive after extraction

                # Scan for nested archives
                for root, _, files in os.walk(current_target):
                    for name in files:
                        p = Path(root) / name
                        if is_archive(p):
                            queue.append((p, Path(root)))
        except Exception as e:
            logger.error(f"Failed to extract {current_file}: {e}")


def _download_file(url: str, dest_path: Path, verbose: bool = False) -> Path:
    """Stream download, correct extension via MIME, and return final path."""
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()

            # Correct extension based on Content-Type
            content_type = r.headers.get("Content-Type", "")
            ext = mimetypes.guess_extension(content_type.split(";")[0])

            if ext and dest_path.suffix != ext:
                dest_path = dest_path.with_suffix(ext)

            total = int(r.headers.get("content-length", 0))

            # Write to disk
            with (
                open(dest_path, "wb") as f,
                tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc=dest_path.name,
                    disable=not verbose,
                ) as pbar,
            ):
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        return dest_path
    except Exception as e:
        raise RuntimeError(f"Download failed for {url}") from e


def download_dataset_files(
    metadata: DatasetMetadata,
    base_dir: Path,
    force: bool = False,
    verbose: bool = False,
) -> Path:
    """Orchestrate download and extraction for a dataset."""
    dataset_dir = base_dir / metadata.title

    if dataset_dir.exists() and not force:
        if verbose:
            logger.info(f"Dataset directory exists: {dataset_dir}. Skipping download.")
        return dataset_dir

    dataset_dir.mkdir(parents=True, exist_ok=True)

    for distro in metadata.distributions:
        # Initial filename derived from title (extension may change during download)
        temp_path = dataset_dir / distro.get_filename()
        url = distro.best_url

        if not url:
            logger.warning(f"No URL found for distribution '{distro.title}'")
            continue

        if verbose:
            logger.info(f"Downloading: {distro.title}")

        try:
            final_path = _download_file(url, temp_path, verbose=verbose)

            # Check for archive extraction
            if final_path.suffix in [".zip", ".tgz"] or final_path.name.endswith(
                ".tar.gz"
            ):
                _extract_archive(final_path, dataset_dir)

        except Exception as e:
            logger.error(f"Failed to process distribution '{distro.title}': {e}")

    return dataset_dir
