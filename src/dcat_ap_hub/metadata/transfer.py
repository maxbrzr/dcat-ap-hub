"""Dataset artifact download and archive extraction utilities."""

import mimetypes
import os
import tarfile
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from dcat_ap_hub.metadata.models import DatasetMetadata
from dcat_ap_hub.utils.logging import logger


def _extract_archive(filepath: Path, target_dir: Path) -> None:
    """
    Recursively extract zip/tar/tgz archives.

    Nested archives discovered during extraction are extracted in breadth-first
    order until no archive files remain.
    """

    def is_archive(f: Path) -> bool:
        return f.suffix == ".zip" or f.name.endswith((".tar.gz", ".tgz"))

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
                # Remove extracted archives to keep dataset directories clean.
                current_file.unlink()

                # Scan extracted files for nested archives.
                for root, _, files in os.walk(current_target):
                    for name in files:
                        p = Path(root) / name
                        if is_archive(p):
                            queue.append((p, Path(root)))
        except Exception as e:
            logger.error(f"Failed to extract {current_file}: {e}")


def _download_file(url: str, dest_path: Path, verbose: bool = False) -> Path:
    """Download a file stream to disk and return the final output path."""
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()

            # Adjust extension using MIME type when safe.
            content_type = r.headers.get("Content-Type", "")
            ext = mimetypes.guess_extension(content_type.split(";")[0])

            # Keep common source/code extensions stable even if server MIME is generic.
            protected_exts = {".py", ".ipynb", ".sh", ".json", ".md", ".yaml", ".yml"}

            should_update = (
                ext
                and dest_path.suffix != ext
                and dest_path.suffix not in protected_exts
            )

            if should_update:
                assert ext is not None  # For type checker
                dest_path = dest_path.with_suffix(ext)

            total = int(r.headers.get("content-length", 0))

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
    """
    Download all distributions and related resources into a dataset directory.

    Returns:
        Path to the dataset directory containing downloaded artifacts.
    """
    dataset_dir = base_dir / metadata.title

    if dataset_dir.exists() and not force:
        if verbose:
            logger.info(f"Dataset directory exists: {dataset_dir}. Skipping download.")
        return dataset_dir

    dataset_dir.mkdir(parents=True, exist_ok=True)

    for distro in metadata.distributions:
        # Extension may be corrected later based on response headers.
        temp_path = dataset_dir / distro.get_filename()
        url = distro.best_url

        if not url:
            logger.warning(f"No URL found for distribution '{distro.title}'")
            continue

        if verbose:
            logger.info(f"Downloading: {distro.title}")

        try:
            final_path = _download_file(url, temp_path, verbose=verbose)

            if final_path.suffix in [".zip", ".tgz"] or final_path.name.endswith(
                ".tar.gz"
            ):
                _extract_archive(final_path, dataset_dir)

        except Exception as e:
            logger.error(f"Failed to process distribution '{distro.title}': {e}")

    for resource in metadata.related_resources:
        temp_path = dataset_dir / resource.get_filename()

        if verbose:
            logger.info(f"Downloading: {resource.title}")

        try:
            final_path = _download_file(
                resource.download_url, temp_path, verbose=verbose
            )

            if final_path.suffix in [".zip", ".tgz"] or final_path.name.endswith(
                ".tar.gz"
            ):
                _extract_archive(final_path, dataset_dir)

        except Exception as e:
            logger.error(f"Failed to process related resource '{resource.title}': {e}")

    return dataset_dir
