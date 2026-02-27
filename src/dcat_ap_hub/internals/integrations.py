"""Integrations with external libraries like Hugging Face."""

import importlib
import importlib.machinery
import importlib.util
import inspect
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import requests

from dcat_ap_hub.core.interfaces import SKLearnModel
from dcat_ap_hub.internals.logging import logger

PIPELINE_TO_AUTO_CLASS = {
    "text-generation": "AutoModelForCausalLM",
    "text-classification": "AutoModelForSequenceClassification",
    "token-classification": "AutoModelForTokenClassification",
    "question-answering": "AutoModelForQuestionAnswering",
    "summarization": "AutoModelForSeq2SeqLM",
    "translation": "AutoModelForSeq2SeqLM",
    "fill-mask": "AutoModelForMaskedLM",
}


def fetch_hf_metadata(model_id: str, token: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetch Hugging Face metadata. Returns empty dict if failed or offline.
    """
    # If it's a local path, skip API call
    if os.path.isdir(model_id):
        return {}

    # Log that we are about to hit the network
    logger.info(f"Fetching Hugging Face metadata for '{model_id}' from API...")

    url = f"https://huggingface.co/api/models/{model_id}"
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass  # Fail gracefully (might be offline or private)

    return {}


def _get_model_class_name(
    hf_metadata: Dict[str, Any], load_task_specific_head: bool
) -> str:
    """Determine AutoModel class. Defaults to AutoModel if metadata is missing."""
    if not load_task_specific_head:
        return "AutoModel"

    # 1. Try pipeline tag from metadata
    pipeline_tag = hf_metadata.get("pipeline_tag")
    if pipeline_tag in PIPELINE_TO_AUTO_CLASS:
        return PIPELINE_TO_AUTO_CLASS[pipeline_tag]

    # 2. Fallback: check transformersInfo
    info = hf_metadata.get("transformersInfo", {})
    return info.get("auto_model", "AutoModel")


def load_hf_model(
    model_id: str,
    token: Optional[str] = None,
    device_map: Optional[Union[str, Dict]] = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
    load_task_specific_head: bool = True,
    cache_dir: Path | str = Path("./models"),
    preloaded_metadata: Optional[Dict] = None,  # Added parameter
) -> Tuple[Any, Any, Dict[str, Any]]:
    # Step 1: Get metadata (use preloaded if available, else fetch)
    if preloaded_metadata is not None:
        logger.info("Using preloaded Hugging Face metadata from distribution.")
        hf_metadata = preloaded_metadata
    else:
        hf_metadata = fetch_hf_metadata(model_id, token=token)

    # Step 2: Determine Model Class
    try:
        transformers = importlib.import_module("transformers")
    except ImportError as e:
        raise ImportError(
            "The 'transformers' library is required. Install the Hugging Face "
            'variant with: pip install "dcat-ap-hub[huggingface]".'
        ) from e

    cls_name = _get_model_class_name(hf_metadata, load_task_specific_head)

    model_class = getattr(transformers, cls_name)

    logger.info(f"Loading '{model_id}' using {cls_name}...")

    # Step 3: Load
    try:
        model = model_class.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            token=token,
            device_map=device_map,
            dtype=dtype,
            cache_dir=cache_dir,
        )

        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=trust_remote_code,
                token=token,
                cache_dir=cache_dir,
            )
        except Exception:
            tokenizer = None

        return model, tokenizer, hf_metadata

    except Exception as e:
        logger.error(f"Failed to load model '{model_id}': {e}")
        raise


def load_onnx_model(
    model_path: Union[str, Path],
    providers: Optional[list] = None,
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


def load_sklearn_model(
    model_path: Union[str, Path], class_name: Optional[str] = None
) -> SKLearnModel:
    """
    Load a sklearn model by dynamically importing a Python module and
    instantiating a class that inherits from SKLearnModel.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"SKLearn model file not found at: {path}")

    # Python module path containing SKLearnModel subclass
    if path.suffix != ".py" and path.suffix != ".txt":
        raise ValueError(
            f"Unsupported sklearn model source '{path.name}'. Expected a .py or .txt module."
        )

    unique_mod_name = f"dcat_sklearn_model_{path.stem}_{uuid.uuid4().hex[:8]}"

    # Non-.py scripts (e.g. .txt) need an explicit source loader.
    if path.suffix == ".txt":
        loader = importlib.machinery.SourceFileLoader(unique_mod_name, str(path))
        spec = importlib.util.spec_from_loader(unique_mod_name, loader)
    else:
        spec = importlib.util.spec_from_file_location(unique_mod_name, path)
    if not spec or not spec.loader:
        raise ImportError(f"Could not load module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_mod_name] = module
    try:
        spec.loader.exec_module(module)

        selected_class = None
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if not issubclass(obj, SKLearnModel) or obj is SKLearnModel:
                continue
            if class_name and obj.__name__ != class_name:
                continue
            selected_class = obj
            break

        if not selected_class:
            hint = f" named '{class_name}'" if class_name else ""
            raise AttributeError(
                f"No class{hint} inheriting SKLearnModel found in '{path.name}'."
            )

        return selected_class()
    finally:
        sys.modules.pop(unique_mod_name, None)
