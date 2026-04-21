"""Hugging Face backend integration."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Optional

import requests

from dcat_ap_hub.integrations.models import (
    BackendName,
    LoadedModel,
    LoadOptions,
    ModelLoader,
    ModelSource,
)
from dcat_ap_hub.utils.logging import logger

PIPELINE_TO_AUTO_CLASS = {
    "text-generation": "AutoModelForCausalLM",
    "text-classification": "AutoModelForSequenceClassification",
    "token-classification": "AutoModelForTokenClassification",
    "question-answering": "AutoModelForQuestionAnswering",
    "summarization": "AutoModelForSeq2SeqLM",
    "translation": "AutoModelForSeq2SeqLM",
    "fill-mask": "AutoModelForMaskedLM",
}


def _fetch_hf_metadata(model_id: str, token: Optional[str] = None) -> dict[str, Any]:
    """Fetch metadata for remote Hugging Face models; return {} on failure."""
    if Path(model_id).is_dir():
        return {}

    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        response = requests.get(
            f"https://huggingface.co/api/models/{model_id}",
            headers=headers,
            timeout=5,
        )
        if response.status_code == 200:
            return response.json()
    except requests.RequestException:
        pass
    return {}


def _resolve_model_class_name(
    metadata: dict[str, Any], load_task_specific_head: bool
) -> str:
    """Resolve the transformers auto-model class name from metadata hints."""
    if not load_task_specific_head:
        return "AutoModel"

    pipeline_tag = metadata.get("pipeline_tag")
    if pipeline_tag in PIPELINE_TO_AUTO_CLASS:
        return PIPELINE_TO_AUTO_CLASS[pipeline_tag]

    info = metadata.get("transformersInfo", {})
    return info.get("auto_model", "AutoModel")


class HuggingFaceLoader(ModelLoader):
    """ModelLoader implementation for Hugging Face models."""

    backend = BackendName.HUGGINGFACE

    def can_load(self, source: ModelSource) -> bool:
        """Return whether the source looks like a Hugging Face model source."""
        role = source.role or ""
        if role in {"huggingface", "huggingface_model"}:
            return True
        if source.path and source.path.is_dir() and (source.path / "config.json").exists():
            return True
        return bool(source.model_id and not source.path)

    def load(self, source: ModelSource, options: LoadOptions) -> LoadedModel:
        """Load a Hugging Face model and optional tokenizer adapter."""
        if source.path and source.path.is_dir():
            model_source = str(source.path.absolute())
        elif source.model_id:
            model_source = source.model_id
        else:
            raise ValueError(
                "Hugging Face loader requires either a model directory path or model_id."
            )

        metadata = source.metadata or _fetch_hf_metadata(
            model_source, token=options.token
        )
        class_name = _resolve_model_class_name(metadata, options.load_task_specific_head)

        try:
            transformers = importlib.import_module("transformers")
        except ImportError as e:
            raise ImportError(
                "The 'transformers' library is required. Install the Hugging Face "
                'variant with: pip install "dcat-ap-hub[huggingface]".'
            ) from e

        try:
            model_class = getattr(transformers, class_name)
        except AttributeError as e:
            raise ValueError(
                f"Unsupported transformers auto class '{class_name}' for model '{model_source}'."
            ) from e

        logger.info(f"Loading '{model_source}' using {class_name}...")
        # We intentionally pass backend-neutral options from LoadOptions directly
        # to transformers so Dataset-level options map cleanly to HF APIs.
        model = model_class.from_pretrained(
            model_source,
            trust_remote_code=options.trust_remote_code,
            token=options.token,
            device_map=options.device_map,
            dtype=options.dtype,
            cache_dir=options.cache_dir,
        )

        try:
            adapter = transformers.AutoTokenizer.from_pretrained(
                model_source,
                trust_remote_code=options.trust_remote_code,
                token=options.token,
                cache_dir=options.cache_dir,
            )
        except Exception:
            adapter = None

        return LoadedModel(
            backend=self.backend,
            model=model,
            adapter=adapter,
            metadata=metadata,
        )
