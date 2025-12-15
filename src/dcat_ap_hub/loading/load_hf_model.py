"""Helpers to load models from Hugging Face using metadata from the DCAT-AP Hub.

This module provides:
- fetch_hf_metadata: fetches model metadata from huggingface.co API
- load_hf_model: loads a model and optional tokenizer/processor given a DCAT-AP Hub URL
"""

import importlib
from pathlib import Path
import requests
import torch
from typing import Any, Dict, Optional, Tuple, Union
from transformers import PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin

from dcat_ap_hub.metadata.metadata import get_metadata
from dcat_ap_hub.logging import logger

PIPELINE_TO_AUTO_CLASS = {
    "text-generation": "AutoModelForCausalLM",
    "text-classification": "AutoModelForSequenceClassification",
    "token-classification": "AutoModelForTokenClassification",
    "question-answering": "AutoModelForQuestionAnswering",
    "summarization": "AutoModelForSeq2SeqLM",
    "translation": "AutoModelForSeq2SeqLM",
    "fill-mask": "AutoModelForMaskedLM",
}


def fetch_hf_metadata(
    model_name: str, token: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Fetch Hugging Face metadata for a given model name."""
    url = f"https://huggingface.co/api/models/{model_name}"
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()
        logger.warning(
            "Failed to fetch HF metadata for '%s' (status: %s)",
            model_name,
            response.status_code,
        )
    except requests.RequestException as e:
        logger.error("Network error fetching metadata for '%s': %s", model_name, e)

    return None


def _get_model_class_name(
    hf_metadata: Dict[str, Any], load_task_specific_head: bool
) -> str:
    """Determine the correct AutoModel class name based on metadata and pipeline tag."""
    transformers_info = hf_metadata.get("transformersInfo", {})

    if not load_task_specific_head:
        return transformers_info.get("auto_model", "AutoModel")

    pipeline_tag = hf_metadata.get("pipeline_tag")
    if pipeline_tag in PIPELINE_TO_AUTO_CLASS:
        return PIPELINE_TO_AUTO_CLASS[pipeline_tag]

    return transformers_info.get("auto_model", "AutoModel")


def load_hf_model(
    url: str,
    token: Optional[str] = None,
    device_map: Optional[Union[str, Dict]] = "auto",
    dtype: Optional[Union[str, torch.dtype]] = "auto",
    trust_remote_code: bool = False,
    load_task_specific_head: bool = True,
    base_dir: Path | str = Path("./models"),
) -> Tuple[
    PreTrainedModel,
    Optional[Union[PreTrainedTokenizerBase, ProcessorMixin]],
    Dict[str, Any],
]:
    """Load a Hugging Face model using DCAT-AP Hub metadata.

    Args:
        url: URL pointing to a DCAT-AP Hub JSON-LD metadata file.
        token: Hugging Face authentication token.
        device_map: "auto" for GPU/CPU allocation.
        dtype: "auto" for best precision (float16/bfloat16).
        trust_remote_code: Whether to allow custom code execution.
        load_task_specific_head: If True, tries to load AutoModelForX (e.g. CausalLM).
        base_dir: Path to directory where model weights should be cached.

    Returns:
        (model, processor_or_tokenizer, raw_hf_metadata)
    """
    # Step 1: Load metadata
    dataset, _ = get_metadata(url)
    if not dataset.is_model:
        raise ValueError(f"Metadata at {url} does not describe a model.")

    model_id: str = dataset.title

    # Step 2: Fetch HF Metadata
    hf_metadata = fetch_hf_metadata(model_id, token=token)
    if not hf_metadata:
        raise ValueError(f"Could not fetch metadata for '{model_id}'.")

    # Step 3: Determine Model Class
    transformers = importlib.import_module("transformers")
    auto_model_class_name = _get_model_class_name(hf_metadata, load_task_specific_head)
    model_class = getattr(transformers, auto_model_class_name)

    # Step 4: Load Model
    logger.info("Loading model '%s' to base_dir: %s", model_id, base_dir or "default")

    try:
        model = model_class.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            token=token,
            device_map=device_map,
            dtype=dtype,
            cache_dir=base_dir,
        )
    except Exception as e:
        logger.error("Failed to instantiate model '%s': %s", model_id, e)
        raise

    # Step 5: Load Processor/Tokenizer
    transformers_info = hf_metadata.get("transformersInfo", {})
    processor_class_name = transformers_info.get("processor")
    tags = hf_metadata.get("tags", [])
    processor = None

    try:
        if processor_class_name:
            proc_cls = getattr(transformers, processor_class_name)
            processor = proc_cls.from_pretrained(
                model_id,
                trust_remote_code=trust_remote_code,
                token=token,
                cache_dir=base_dir,
            )
        elif any(t in tags for t in ["text", "nlp", "language", "transformers"]):
            processor = transformers.AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=trust_remote_code,
                token=token,
                cache_dir=base_dir,
            )
    except Exception as e:
        logger.warning("Could not load processor/tokenizer: %s", e)

    return model, processor, hf_metadata


if __name__ == "__main__":
    # Example Usage
    test_url = "https://ki-daten.hlrs.de/hub/repo/datasets/6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837.jsonld"

    # Example: Loading with a task head (e.g. for text generation) and GPU
    try:
        model, processor, meta = load_hf_model(
            test_url,
            device_map="auto",  # Use GPU if available
            load_task_specific_head=True,  # Ensure we get CausalLM/SeqClassify head
        )

        print("Model loaded:", model.__class__.__name__)
        if processor:
            print("Processor/Tokenizer loaded:", processor.__class__.__name__)

        # # Simple inference check if text model
        # if processor and "text-generation" in meta.get("pipeline_tag", ""):
        #     inputs = processor("Hello, world!", return_tensors="pt").to(model.device)
        #     output = model.generate(**inputs, max_new_tokens=20)
        #     print("Output:", processor.decode(output[0]))

    except Exception as e:
        logger.error("Main execution failed: %s", e)
