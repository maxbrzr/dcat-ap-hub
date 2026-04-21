"""Public integration APIs and typed model-loading interfaces.

Consumers should generally import from this module instead of submodules to keep
their code resilient to internal refactors.
"""

from dcat_ap_hub.integrations.integrations import (
    DEFAULT_REGISTRY,
    IntegrationRegistry,
    load_model,
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
    "load_model",
]
