"""Shared models and interfaces for backend integrations."""

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Sequence, Union, runtime_checkable


class BackendName(StrEnum):
    """Supported backend identifiers."""

    HUGGINGFACE = "huggingface"
    ONNX = "onnx"
    SKLEARN = "sklearn"


@dataclass(frozen=True)
class ModelSource:
    """Input source used by backend loaders."""

    path: Optional[Path] = None
    model_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    role: Optional[str] = None


@dataclass(frozen=True)
class LoadOptions:
    """Cross-backend loading options."""

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
    """Normalized result produced by integrations."""

    backend: BackendName
    model: Any
    adapter: Any | None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_tuple(self) -> tuple[Any, Any, Dict[str, Any]]:
        """Compatibility helper for older tuple-based APIs."""
        return (self.model, self.adapter, self.metadata)


@runtime_checkable
class ModelLoader(Protocol):
    """Common interface implemented by all backend integrations."""

    backend: BackendName

    def can_load(self, source: ModelSource) -> bool:
        """Return whether this loader can handle the source."""
        ...

    def load(self, source: ModelSource, options: LoadOptions) -> LoadedModel:
        """Load model artifacts and return a normalized LoadedModel."""
        ...


class IntegrationRegistry:
    """Registry and dispatcher for backend loaders."""

    def __init__(self, loaders: Optional[Sequence[ModelLoader]] = None) -> None:
        self._loaders: dict[BackendName, ModelLoader] = {}
        if loaders:
            for loader in loaders:
                self.register(loader)

    def register(self, loader: ModelLoader) -> None:
        self._loaders[loader.backend] = loader

    def _normalize_backend(self, backend: BackendName | str) -> BackendName:
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
        # We intentionally require exactly one matching backend to avoid
        # silent misloads when artifacts for multiple backends coexist.
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
        opts = options or LoadOptions()
        loader = self.get(backend) if backend else self.detect(source)
        return loader.load(source, opts)
