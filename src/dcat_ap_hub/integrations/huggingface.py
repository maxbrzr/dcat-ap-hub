"""Hugging Face model loading integration."""

import importlib
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

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


class HuggingFaceLoader(ModelLoader):
    """ModelLoader implementation for Hugging Face models."""

    backend = BackendName.HUGGINGFACE

    def can_load(self, source: ModelSource) -> bool:
        role = source.role or ""
        if role in {"huggingface", "huggingface_model"}:
            return True
        if (
            source.path
            and source.path.is_dir()
            and (source.path / "config.json").exists()
        ):
            return True
        # A remote model id is a valid HF candidate if no explicit path is given.
        return bool(source.model_id and not source.path)

    def load(self, source: ModelSource, options: LoadOptions) -> LoadedModel:
        model_source: str
        if source.path and source.path.is_dir():
            model_source = str(source.path.absolute())
        elif source.model_id:
            model_source = source.model_id
        else:
            raise ValueError(
                "Hugging Face loader requires either a model directory path or model_id."
            )

        model, tokenizer, metadata = self._load_hf_model(
            model_id=model_source,
            token=options.token,
            device_map=options.device_map,
            dtype=options.dtype,
            trust_remote_code=options.trust_remote_code,
            load_task_specific_head=options.load_task_specific_head,
            cache_dir=options.cache_dir,
            preloaded_metadata=source.metadata,
        )
        return LoadedModel(
            backend=self.backend,
            model=model,
            adapter=tokenizer,
            metadata=metadata,
        )

    def _load_hf_model(
        self,
        model_id: str,
        token: Optional[str] = None,
        device_map: Optional[Union[str, Dict]] = "auto",
        dtype: str = "auto",
        trust_remote_code: bool = False,
        load_task_specific_head: bool = True,
        cache_dir: Path | str = Path("./models"),
        preloaded_metadata: Optional[Dict] = None,
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        if preloaded_metadata is not None:
            logger.info("Using preloaded Hugging Face metadata from distribution.")
            hf_metadata = preloaded_metadata
        else:
            hf_metadata = self.fetch_hf_metadata(model_id, token=token)

        try:
            transformers = importlib.import_module("transformers")
        except ImportError as e:
            raise ImportError(
                "The 'transformers' library is required. Install the Hugging Face "
                'variant with: pip install "dcat-ap-hub[huggingface]".'
            ) from e

        cls_name = self._get_model_class_name(hf_metadata, load_task_specific_head)
        try:
            model_class = getattr(transformers, cls_name)
        except AttributeError as e:
            raise ValueError(
                f"Unsupported transformers auto class '{cls_name}' for model '{model_id}'."
            ) from e

        logger.info(f"Loading '{model_id}' using {cls_name}...")

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

    def fetch_hf_metadata(
        self, model_id: str, token: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fetch Hugging Face metadata. Returns empty dict if failed or offline.
        """
        if os.path.isdir(model_id):
            return {}

        logger.info(f"Fetching Hugging Face metadata for '{model_id}' from API...")

        url = f"https://huggingface.co/api/models/{model_id}"
        headers = {"Accept": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        try:
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                return response.json()
        except requests.RequestException:
            pass  # Fail gracefully (might be offline or private)

        return {}

    def _get_model_class_name(
        self, hf_metadata: Dict[str, Any], load_task_specific_head: bool
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
    preloaded_metadata: Optional[Dict] = None,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Backward-compatible tuple API for Hugging Face loading."""
    loader = HuggingFaceLoader()
    return loader._load_hf_model(
        model_id=model_id,
        token=token,
        device_map=device_map,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        load_task_specific_head=load_task_specific_head,
        cache_dir=cache_dir,
        preloaded_metadata=preloaded_metadata,
    )
