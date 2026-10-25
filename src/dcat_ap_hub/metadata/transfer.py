"""Dataset artifact download and archive extraction utilities."""

import mimetypes
import os
import re
import tarfile
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from dcat_ap_hub.metadata.models import DatasetMetadata
from dcat_ap_hub.utils.logging import logger


# A real file extension, unlike the tail of "Data v1.2 archive_<hash>".
_EXTENSION = re.compile(r"\.[A-Za-z0-9]{1,10}$")


def dataset_dir_name(title: str) -> str:
    """
    Return a dataset title as one folder name that is valid on every OS.

    Path separators and characters Windows rejects (``<>:"/\\|?*``) become
    spaces; whitespace is collapsed and trailing dots or spaces are removed.
    """
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", title)
    name = " ".join(name.split()).strip(" .")
    return name or "untitled_dataset"


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

            # Servers often send generic types (``application/octet-stream`` for
            # Parquet), so the MIME type only fills in a missing extension. HTML
            # is the exception: it marks a landing or login page, not the data.
            mime = r.headers.get("Content-Type", "").split(";")[0].split(",")[0]
            mime = mime.strip().lower()
            ext = mimetypes.guess_extension(mime)
            has_extension = bool(_EXTENSION.search(dest_path.name))
            is_html_page = mime == "text/html" and dest_path.suffix not in {
                ".html",
                ".htm",
            }

            if ext and (not has_extension or is_html_page):
                dest_path = (
                    dest_path.with_suffix(ext)
                    if has_extension
                    else dest_path.with_name(dest_path.name + ext)
                )

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
    dataset_dir = base_dir / dataset_dir_name(metadata.title)
    # Reuse a folder created by older versions from the unsanitized title.
    legacy_dir = base_dir / metadata.title
    if (
        legacy_dir != dataset_dir
        and legacy_dir.parent == base_dir
        and legacy_dir.is_dir()
        and not dataset_dir.exists()
    ):
        dataset_dir = legacy_dir

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

        if not distro.download_url:
            logger.warning(
                f"Distribution '{distro.title}' has no dcat:downloadURL; falling "
                f"back to its dcat:accessURL {url}, which is often a landing or "
                "login page rather than the data (access may be restricted)."
            )

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
