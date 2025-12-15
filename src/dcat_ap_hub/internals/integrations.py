"""Integrations with external libraries like Hugging Face."""

import importlib
import requests
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

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
    dtype: str = "auto",  # Added dtype support
    trust_remote_code: bool = False,
    load_task_specific_head: bool = True,
    cache_dir: Path | str = Path("./models"),
) -> Tuple[Any, Any, Dict[str, Any]]:
    # Step 1: Try to fetch metadata (soft fail)
    hf_metadata = fetch_hf_metadata(model_id, token=token)

    # Step 2: Determine Model Class
    try:
        transformers = importlib.import_module("transformers")
    except ImportError as e:
        raise ImportError("The 'transformers' library is required.") from e

    cls_name = _get_model_class_name(hf_metadata, load_task_specific_head)

    # If metadata failed (local model), force AutoModel (or let user override via kwargs logic in future)
    # But often AutoModelForCausalLM works generically if we guess wrong.
    # For robust local loading, usually AutoModel is safest, OR we inspect config.json locally.
    # For now, we default to the class derived or AutoModel.
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

        # Smart tokenizer loading
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
