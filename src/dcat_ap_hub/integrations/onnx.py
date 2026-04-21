"""ONNX backend integration."""

from __future__ import annotations

import importlib
from typing import Any

from dcat_ap_hub.integrations.models import (
    BackendName,
    LoadedModel,
    LoadOptions,
    ModelLoader,
    ModelSource,
)
from dcat_ap_hub.utils.logging import logger


def _extract_session_metadata(session: Any) -> dict[str, Any]:
    """Best-effort extraction of ONNX model metadata from a runtime session."""
    try:
        model_meta = session.get_modelmeta()
    except Exception as e:
        logger.warning(f"Could not extract metadata from ONNX model: {e}")
        return {}

    if not model_meta:
        return {}
    return {
        "description": model_meta.description,
        "producer_name": model_meta.producer_name,
        "graph_name": model_meta.graph_name,
        "domain": model_meta.domain,
        "version": model_meta.version,
        "custom_metadata_map": model_meta.custom_metadata_map,
    }


class OnnxLoader(ModelLoader):
    """ModelLoader implementation for ONNX models."""

    backend = BackendName.ONNX

    def can_load(self, source: ModelSource) -> bool:
        """Return whether the source points to ONNX artifacts."""
        role = source.role or ""
        if role in {"onnx", "onnx_model"}:
            return True
        if source.path is None:
            return False
        if source.path.is_file() and source.path.suffix.lower() == ".onnx":
            return True
        if source.path.is_dir() and any(source.path.glob("*.onnx")):
            return True
        return False

    def load(self, source: ModelSource, options: LoadOptions) -> LoadedModel:
        """Load an ONNX model into an ``onnxruntime.InferenceSession``."""
        if source.path is None:
            raise ValueError("ONNX loader requires a filesystem path.")

        model_path = source.path
        if model_path.is_dir():
            candidates = sorted(model_path.glob("*.onnx"))
            if not candidates:
                raise FileNotFoundError(f"No .onnx file found in '{model_path}'.")
            model_path = candidates[0]

        try:
            ort = importlib.import_module("onnxruntime")
        except ImportError as e:
            raise ImportError(
                "The 'onnxruntime' library is required to load ONNX models. "
                'Install the ONNX variant with: pip install "dcat-ap-hub[onnx]".'
            ) from e

        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model file not found at: {model_path}")

        providers = options.onnx_providers or ["CPUExecutionProvider"]
        logger.info(f"Loading ONNX model from '{model_path}'...")
        # Session is returned as model object; no standard adapter/tokenizer exists.
        session = ort.InferenceSession(str(model_path), providers=providers)

        metadata = source.metadata or _extract_session_metadata(session)
        return LoadedModel(
            backend=self.backend,
            model=session,
            adapter=None,
            metadata=metadata,
        )
