"""Typed contracts and value objects for model backend integrations."""

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Sequence, Union, runtime_checkable


class BackendName(StrEnum):
    """Supported model backend identifiers."""

    HUGGINGFACE = "huggingface"
    ONNX = "onnx"
    SKLEARN = "sklearn"


@dataclass(frozen=True)
class ModelSource:
    """
    Canonical model source descriptor consumed by backend loaders.

    Attributes:
        path: Optional local path to model artifacts.
        model_id: Optional remote/model-hub identifier.
        metadata: Optional preloaded metadata attached by discovery.
        role: Optional metadata-derived role hint (e.g. ``onnx_model``).
    """

    path: Optional[Path] = None
    model_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    role: Optional[str] = None


@dataclass(frozen=True)
class LoadOptions:
    """
    Cross-backend load options.

    Most fields are backend-specific and safely ignored by other backends.
    """

    token: Optional[str] = None
    cache_dir: Union[Path, str] = Path("./models")
    device_map: Optional[Union[str, Dict[str, Any]]] = "auto"
    dtype: str = "auto"
    trust_remote_code: bool = False
    load_task_specific_head: bool = True
    onnx_providers: Optional[list[str]] = None
    sklearn_class_name: Optional[str] = None


@dataclass(frozen=True)
class LoadedModel:
    """
    Normalized model load result.

    ``adapter`` typically contains tokenizer/processor-like objects when
    available (for example Hugging Face tokenizers), otherwise ``None``.
    """

    backend: BackendName
    model: Any
    adapter: Any | None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_tuple(self) -> tuple[Any, Any, Dict[str, Any]]:
        """Return `(model, adapter, metadata)` tuple for caller convenience."""
        return (self.model, self.adapter, self.metadata)


@runtime_checkable
class ModelLoader(Protocol):
    """Interface implemented by all backend-specific model loaders."""

    backend: BackendName

    def can_load(self, source: ModelSource) -> bool:
        """Return whether this loader can handle the provided source."""
        ...

    def load(self, source: ModelSource, options: LoadOptions) -> LoadedModel:
        """Load model artifacts and return a normalized ``LoadedModel``."""
        ...


class IntegrationRegistry:
    """
    Registry + dispatcher for backend loaders.

    The registry supports explicit backend selection and conservative
    auto-detection that fails on ambiguous matches.
    """

    def __init__(self, loaders: Optional[Sequence[ModelLoader]] = None) -> None:
        self._loaders: dict[BackendName, ModelLoader] = {}
        if loaders:
            for loader in loaders:
                self.register(loader)

    def register(self, loader: ModelLoader) -> None:
        """Register or replace a loader for its declared backend."""
        self._loaders[loader.backend] = loader

    def _normalize_backend(self, backend: BackendName | str) -> BackendName:
        """Normalize string/enum backend values to ``BackendName``."""
        if isinstance(backend, BackendName):
            return backend
        try:
            return BackendName(backend)
        except ValueError as e:
            available = ", ".join(b.value for b in BackendName)
            raise ValueError(
                f"Unknown backend '{backend}'. Available backends: {available}."
            ) from e

    def get(self, backend: BackendName | str) -> ModelLoader:
        """Return a registered loader for an explicit backend."""
        normalized = self._normalize_backend(backend)
        try:
            return self._loaders[normalized]
        except KeyError as e:
            available = (
                ", ".join(sorted(b.value for b in self._loaders.keys())) or "none"
            )
            raise ValueError(
                f"Unknown backend '{normalized.value}'. Available backends: {available}."
            ) from e

    def detect(self, source: ModelSource) -> ModelLoader:
        """
        Detect a loader from source hints.

        The method intentionally requires exactly one match to avoid silent
        misloads when artifacts for multiple backends coexist.
        """
        matches = [
            loader for loader in self._loaders.values() if loader.can_load(source)
        ]
        if not matches:
            raise ValueError("Could not detect model backend from source.")
        if len(matches) > 1:
            backends = ", ".join(sorted(loader.backend.value for loader in matches))
            raise ValueError(f"Ambiguous model backend detection: {backends}.")
        return matches[0]

    def load(
        self,
        source: ModelSource,
        options: Optional[LoadOptions] = None,
        backend: Optional[BackendName | str] = None,
    ) -> LoadedModel:
        """Load a model via explicit backend or auto-detection."""
        opts = options or LoadOptions()
        loader = self.get(backend) if backend else self.detect(source)
        return loader.load(source, opts)
