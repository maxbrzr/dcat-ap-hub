"""Backend-agnostic integration dispatcher and compatibility helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

from dcat_ap_hub.integrations.huggingface import HuggingFaceLoader, load_hf_model
from dcat_ap_hub.integrations.models import (
    BackendName,
    IntegrationRegistry,
    LoadedModel,
    LoadOptions,
    ModelSource,
)
from dcat_ap_hub.integrations.onnx import OnnxLoader, load_onnx_model
from dcat_ap_hub.integrations.sklearn import SklearnLoader, load_sklearn_model

DEFAULT_REGISTRY = IntegrationRegistry(
    [OnnxLoader(), HuggingFaceLoader(), SklearnLoader()]
)


def load_model(
    source: ModelSource,
    options: Optional[LoadOptions] = None,
    backend: Optional[BackendName | str] = None,
    registry: IntegrationRegistry = DEFAULT_REGISTRY,
) -> LoadedModel:
    """Load a model with automatic or explicit backend selection."""
    return registry.load(source=source, options=options, backend=backend)


def _load_with_backend(
    *,
    backend: BackendName,
    source: ModelSource,
    options: LoadOptions,
) -> LoadedModel:
    """Shared dispatcher used by compatibility wrappers."""
    return load_model(source=source, options=options, backend=backend)


def load_hf_loaded_model(
    model_id: str,
    token: Optional[str] = None,
    device_map: Optional[Union[str, Dict]] = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
    load_task_specific_head: bool = True,
    cache_dir: Path | str = Path("./models"),
    preloaded_metadata: Optional[Dict] = None,
) -> LoadedModel:
    """
    Compatibility helper returning a normalized LoadedModel for Hugging Face.
    """
    options = LoadOptions(
        token=token,
        cache_dir=cache_dir,
        device_map=device_map,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        load_task_specific_head=load_task_specific_head,
    )
    source = ModelSource(
        path=Path(model_id) if Path(model_id).exists() else None,
        model_id=model_id,
        metadata=preloaded_metadata,
        role="huggingface_model",
    )
    return _load_with_backend(
        backend=BackendName.HUGGINGFACE,
        source=source,
        options=options,
    )


def load_onnx_loaded_model(
    model_path: Union[str, Path],
    providers: Optional[list[str]] = None,
    preloaded_metadata: Optional[Dict] = None,
) -> LoadedModel:
    """
    Compatibility helper returning a normalized LoadedModel for ONNX.
    """
    options = LoadOptions(onnx_providers=providers)
    source = ModelSource(
        path=Path(model_path), metadata=preloaded_metadata, role="onnx_model"
    )
    return _load_with_backend(
        backend=BackendName.ONNX,
        source=source,
        options=options,
    )


def load_sklearn_loaded_model(
    model_path: Union[str, Path], class_name: Optional[str] = None
) -> LoadedModel:
    """
    Compatibility helper returning a normalized LoadedModel for sklearn.
    """
    options = LoadOptions(sklearn_class_name=class_name)
    source = ModelSource(path=Path(model_path), role="sklearn_model")
    return _load_with_backend(
        backend=BackendName.SKLEARN,
        source=source,
        options=options,
    )


__all__ = [
    "BackendName",
    "DEFAULT_REGISTRY",
    "IntegrationRegistry",
    "LoadedModel",
    "LoadOptions",
    "ModelSource",
    "load_hf_loaded_model",
    "load_hf_model",
    "load_model",
    "load_onnx_loaded_model",
    "load_onnx_model",
    "load_sklearn_loaded_model",
    "load_sklearn_model",
]
