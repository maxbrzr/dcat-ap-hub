"""Public integration APIs and typed model-loading interfaces."""

from dcat_ap_hub.integrations.integrations import (
    DEFAULT_REGISTRY,
    IntegrationRegistry,
    load_hf_loaded_model,
    load_hf_model,
    load_model,
    load_onnx_loaded_model,
    load_onnx_model,
    load_sklearn_loaded_model,
    load_sklearn_model,
)
from dcat_ap_hub.integrations.models import (
    BackendName,
    LoadedModel,
    LoadOptions,
    ModelLoader,
    ModelSource,
)
from dcat_ap_hub.integrations.sklearn import SKLearnModel

__all__ = [
    "BackendName",
    "DEFAULT_REGISTRY",
    "IntegrationRegistry",
    "LoadedModel",
    "LoadOptions",
    "ModelLoader",
    "ModelSource",
    "SKLearnModel",
    "load_hf_loaded_model",
    "load_hf_model",
    "load_model",
    "load_onnx_loaded_model",
    "load_onnx_model",
    "load_sklearn_loaded_model",
    "load_sklearn_model",
]
