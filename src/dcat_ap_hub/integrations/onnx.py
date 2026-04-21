"""ONNX model loading integration."""

import importlib
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from dcat_ap_hub.integrations.models import (
    BackendName,
    LoadedModel,
    LoadOptions,
    ModelLoader,
    ModelSource,
)
from dcat_ap_hub.utils.logging import logger


class OnnxLoader(ModelLoader):
    """ModelLoader implementation for ONNX models."""

    backend = BackendName.ONNX

    def can_load(self, source: ModelSource) -> bool:
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
        if source.path is None:
            raise ValueError("ONNX loader requires a filesystem path.")

        model_path = source.path
        if model_path.is_dir():
            # If multiple models exist, we use deterministic ordering.
            candidates = sorted(model_path.glob("*.onnx"))
            if not candidates:
                raise FileNotFoundError(f"No .onnx file found in '{model_path}'.")
            model_path = candidates[0]

        model, adapter, metadata = self._load_onnx_model(
            model_path=model_path,
            providers=options.onnx_providers,
            preloaded_metadata=source.metadata,
        )
        return LoadedModel(
            backend=self.backend,
            model=model,
            adapter=adapter,
            metadata=metadata,
        )

    def _load_onnx_model(
        self,
        model_path: Union[str, Path],
        providers: Optional[list[str]] = None,
        preloaded_metadata: Optional[Dict] = None,
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Load an ONNX model using onnxruntime.
        """
        try:
            ort = importlib.import_module("onnxruntime")
        except ImportError as e:
            raise ImportError(
                "The 'onnxruntime' library is required to load ONNX models. "
                'Install the ONNX variant with: pip install "dcat-ap-hub[onnx]".'
            ) from e

        path_str = str(model_path)
        if not os.path.exists(path_str):
            raise FileNotFoundError(f"ONNX model file not found at: {path_str}")

        logger.info(f"Loading ONNX model from '{path_str}'...")

        if providers is None:
            providers = ["CPUExecutionProvider"]

        session = ort.InferenceSession(path_str, providers=providers)

        # Extract metadata from the model file if not provided
        meta = preloaded_metadata or {}
        if not meta:
            try:
                # Try to get metadata from the session if available
                model_meta = session.get_modelmeta()
                if model_meta:
                    # convert to dict
                    meta = {
                        "description": model_meta.description,
                        "producer_name": model_meta.producer_name,
                        "graph_name": model_meta.graph_name,
                        "domain": model_meta.domain,
                        "version": model_meta.version,
                        "custom_metadata_map": model_meta.custom_metadata_map,
                    }
            except Exception as e:
                logger.warning(f"Could not extract metadata from ONNX model: {e}")

        # No tokenizer standard for ONNX usually, unless wrapped. Returning None for now.
        return session, None, meta


def load_onnx_model(
    model_path: Union[str, Path],
    providers: Optional[list[str]] = None,
    preloaded_metadata: Optional[Dict] = None,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Backward-compatible tuple API for ONNX loading."""
    loader = OnnxLoader()
    return loader._load_onnx_model(
        model_path=model_path,
        providers=providers,
        preloaded_metadata=preloaded_metadata,
    )
