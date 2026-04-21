"""Backend-agnostic integration dispatcher."""

from __future__ import annotations

from typing import Optional

from dcat_ap_hub.integrations.huggingface import HuggingFaceLoader
from dcat_ap_hub.integrations.models import (
    BackendName,
    IntegrationRegistry,
    LoadedModel,
    LoadOptions,
    ModelSource,
)
from dcat_ap_hub.integrations.onnx import OnnxLoader
from dcat_ap_hub.integrations.sklearn import SklearnLoader

# Global default registry used by the package-level `load_model` helper.
DEFAULT_REGISTRY = IntegrationRegistry(
    [OnnxLoader(), HuggingFaceLoader(), SklearnLoader()]
)


def load_model(
    source: ModelSource,
    options: Optional[LoadOptions] = None,
    backend: Optional[BackendName | str] = None,
) -> LoadedModel:
    """
    Load a model with automatic or explicit backend selection.

    Args:
        source: Input path/id + optional metadata used by the backends.
        options: Cross-backend loading options.
        backend: Optional backend override. When omitted, auto-detection is used.
    """
    return DEFAULT_REGISTRY.load(source=source, options=options, backend=backend)
