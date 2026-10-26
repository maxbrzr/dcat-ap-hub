"""Dataset artifact download and archive extraction utilities."""

import hashlib
import json
import mimetypes
import os
import re
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from dcat_ap_hub.metadata import auth
from dcat_ap_hub.metadata.constants import SOURCES_MANIFEST
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
        with auth.get(url, stream=True) as r:
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

    Also writes ``SOURCES_MANIFEST`` there, recording for every file the URL it
    came from (archive members the archive's URL), its SHA-256, and every
    download that failed or returned an HTML page instead of data.

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
    sources: list[dict] = []

    for distro in metadata.distributions:
        # Extension may be corrected later based on response headers.
        temp_path = dataset_dir / distro.get_filename()
        url = distro.best_url

        if not url:
            logger.warning(f"No URL found for distribution '{distro.title}'")
            sources.append(_source("distribution", distro.title, "", "no_url"))
            continue

        if not distro.download_url:
            logger.warning(
                f"Distribution '{distro.title}' has no dcat:downloadURL; falling "
                f"back to its dcat:accessURL {url}, which is often a landing or "
                "login page rather than the data (access may be restricted)."
            )

        if verbose:
            logger.info(f"Downloading: {distro.title}")

        sources.append(
            _fetch_into(
                dataset_dir,
                url,
                temp_path,
                _source(
                    "distribution",
                    distro.title,
                    url,
                    "downloaded",
                    access_url_fallback=not distro.download_url,
                ),
                verbose=verbose,
            )
        )

    for resource in metadata.related_resources:
        temp_path = dataset_dir / resource.get_filename()

        if verbose:
            logger.info(f"Downloading: {resource.title}")

        sources.append(
            _fetch_into(
                dataset_dir,
                resource.download_url,
                temp_path,
                _source(f"resource:{resource.role}", resource.title, resource.download_url, "downloaded"),
                verbose=verbose,
            )
        )

    _write_sources_manifest(dataset_dir, metadata.source_url, sources)
    return dataset_dir


def _source(kind: str, title: str, url: str, status: str, **extra) -> dict:
    return {"kind": kind, "title": title, "url": url, "status": status, **extra, "files": []}


def _snapshot(directory: Path) -> dict[str, tuple[int, int]]:
    return {
        str(path.relative_to(directory)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in directory.rglob("*")
        if path.is_file()
    }


def _fetch_into(
    dataset_dir: Path, url: str, temp_path: Path, source: dict, verbose: bool = False
) -> dict:
    """Download (and extract) one URL, recording the files it produced in ``source``."""
    before = _snapshot(dataset_dir)
    try:
        final_path = _download_file(url, temp_path, verbose=verbose)
        if final_path.suffix in {".html", ".htm"}:
            # The server answered with a web page, not the advertised data.
            source["status"] = "html_page"
        if final_path.suffix in [".zip", ".tgz"] or final_path.name.endswith(".tar.gz"):
            source["archive"] = final_path.name
            _extract_archive(final_path, dataset_dir)
    except Exception as e:
        logger.error(f"Failed to process '{source['title']}': {e}")
        source["status"] = "failed"
        source["error"] = str(e.__cause__ or e)
    after = _snapshot(dataset_dir)
    # New or rewritten files belong to this source (an archive may overwrite).
    source["files"] = sorted(name for name, state in after.items() if before.get(name) != state)
    return source


def _write_sources_manifest(dataset_dir: Path, record: str, sources: list[dict]) -> None:
    owner: dict[str, int] = {}
    for index, source in enumerate(sources):
        for name in source["files"]:
            owner[name] = index
    for index, source in enumerate(sources):
        # A later source that rewrote a file owns it.
        source["files"] = [name for name in source["files"] if owner.get(name) == index]
    files = {
        name: {"source": index, "sha256": _sha256(dataset_dir / name)}
        for name, index in sorted(owner.items())
        if (dataset_dir / name).is_file()
    }
    manifest = {
        "record": record,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sources": sources,
        "files": files,
    }
    (dataset_dir / SOURCES_MANIFEST).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def read_sources_manifest(dataset_dir: Path) -> Optional[dict]:
    """Return a folder's sources manifest, or ``None`` if it was not downloaded by the hub."""
    path = Path(dataset_dir) / SOURCES_MANIFEST
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
