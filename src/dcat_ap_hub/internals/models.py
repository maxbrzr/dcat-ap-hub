"""Domain models for the DCAT-AP Hub."""

from dataclasses import dataclass, field
from typing import List, Literal, Optional


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
        """Sanitize title to create a safe filename."""
        # Basic sanitization: allow alphanumerics, dots, dashes, underscores
        safe = "".join(c for c in self.title if c.isalnum() or c in " ._-")
        return safe.strip() or "untitled_distribution"


@dataclass
class RelatedResource:
    """Represents a related resource (e.g. processor script, notebook)."""

    title: str
    description: str
    format: str
    download_url: Optional[str] = None
    role: Literal["processor", "notebook"] = "processor"

    @property
    def best_url(self) -> str:
        """Return download_url."""
        return self.download_url or ""

    def get_filename(self) -> str:
        """Derive filename from URL if possible, else sanitize title."""
        if self.download_url:
            from urllib.parse import urlparse

            path = urlparse(self.download_url).path
            name = path.split("/")[-1]
            if name:
                return name

        # Fallback
        safe = "".join(c for c in self.title if c.isalnum() or c in " ._-")
        return safe.strip() or "untitled_resource"


@dataclass
class DatasetMetadata:
    """Internal metadata representation."""

    title: str
    description: str
    distributions: List[Distribution] = field(default_factory=list)
    related_resources: List[RelatedResource] = field(default_factory=list)
    is_model: bool = False
    source_url: str = ""
