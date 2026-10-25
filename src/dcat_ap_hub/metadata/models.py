"""Domain models for the DCAT-AP Hub."""

from dataclasses import dataclass, field
from hashlib import blake2b
from typing import List, Literal, Optional


def _sanitize_filename(name: str, fallback: str) -> str:
    """Sanitize a display name into a filesystem-safe filename."""
    safe = "".join(c for c in name if c.isalnum() or c in " ._-").strip()
    return safe or fallback


def _short_hash(value: str, digest_size: int = 6) -> str:
    """Return a compact deterministic hash for stable filename suffixes."""
    return blake2b(value.encode("utf-8"), digest_size=digest_size).hexdigest()


def _append_hash_before_extension(filename: str, hash_suffix: str) -> str:
    """Append hash while preserving the last file extension when present."""
    if "." in filename and not filename.startswith("."):
        stem, extension = filename.rsplit(".", 1)
        if stem:
            return f"{stem}_{hash_suffix}.{extension}"
    return f"{filename}_{hash_suffix}"


@dataclass
class Distribution:
    """Represents one downloadable/accessed distribution entry."""

    title: str
    description: str
    format: str
    access_url: str
    download_url: Optional[str] = None
    role: Literal["data", "onnx_model", "huggingface_model", "sklearn_model"] = "data"

    @property
    def best_url(self) -> str:
        """Return the preferred download URL (fallback to access URL)."""
        return self.download_url or self.access_url

    def get_filename(self) -> str:
        """Create a safe filename with deterministic URL-based disambiguation."""
        base = _sanitize_filename(self.title, "untitled_distribution")
        source_url = self.download_url or self.access_url
        filename = _append_hash_before_extension(base, _short_hash(source_url))
        return filename


@dataclass
class RelatedResource:
    """Represents an auxiliary resource (for example processor or notebook)."""

    title: str
    description: str
    format: str
    download_url: str
    role: Literal["processor", "notebook", "other"] = "other"
    # True when the dataset entry references this resource (dct:relation etc.).
    linked: bool = False

    def get_filename(self) -> str:
        """
        Create a safe filename with deterministic URL-based disambiguation.

        Processors keep their plain title (``processor.py``), so a dataset
        folder looks the same whether it was downloaded or assembled locally.
        """
        base = _sanitize_filename(self.title, "untitled_resource")
        if self.role == "processor":
            return base
        filename = _append_hash_before_extension(base, _short_hash(self.download_url))
        return filename

    def candidate_filenames(self) -> list[str]:
        """
        Local names this resource may have, most specific first.

        ``get_filename()`` is the name used by ``download()``; the plain title
        and the last URL path segment cover locally assembled dataset folders.
        """
        from urllib.parse import urlparse

        names = [
            self.get_filename(),
            _sanitize_filename(self.title, ""),
            urlparse(self.download_url).path.rsplit("/", 1)[-1],
        ]
        return list(dict.fromkeys(name for name in names if name))


@dataclass
class DatasetMetadata:
    """Internal normalized metadata representation used by the library."""

    title: str
    description: str
    distributions: List[Distribution] = field(default_factory=list)
    related_resources: List[RelatedResource] = field(default_factory=list)
    is_model: bool = False
    source_url: str = ""
    # Records this dataset derives from (prov:wasDerivedFrom), e.g. the raw
    # dataset of an AI-ready one.
    derived_from: List[str] = field(default_factory=list)
