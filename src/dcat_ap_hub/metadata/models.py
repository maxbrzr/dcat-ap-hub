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
    """Represents a specific representation of a dataset (file/resource)."""

    title: str
    description: str
    format: str
    access_url: str
    download_url: Optional[str] = None
    role: Literal["data", "onnx_model", "huggingface_model", "sklearn_model"] = "data"

    @property
    def best_url(self) -> str:
        """Return download_url if available, else access_url."""
        return self.download_url or self.access_url

    def get_filename(self) -> str:
        """Create a safe filename with deterministic URL-based disambiguation."""
        base = _sanitize_filename(self.title, "untitled_distribution")
        source_url = self.download_url or self.access_url
        filename = _append_hash_before_extension(base, _short_hash(source_url))
        return filename


@dataclass
class RelatedResource:
    """Represents a related resource (e.g. processor script, notebook)."""

    title: str
    description: str
    format: str
    download_url: str
    role: Literal["processor", "notebook"] = "processor"

    def get_filename(self) -> str:
        """Create a safe filename with deterministic URL-based disambiguation."""
        base = _sanitize_filename(self.title, "untitled_resource")
        filename = _append_hash_before_extension(base, _short_hash(self.download_url))
        return filename


@dataclass
class DatasetMetadata:
    """Internal metadata representation."""

    title: str
    description: str
    distributions: List[Distribution] = field(default_factory=list)
    related_resources: List[RelatedResource] = field(default_factory=list)
    is_model: bool = False
    source_url: str = ""
