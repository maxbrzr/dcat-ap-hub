import importlib
import requests
from typing import Any, Dict, Optional, Tuple, Union

from dcat_ap_hub.loading.metadata import get_metadata
from dcat_ap_hub.logging import logger

# Import HF base types for typing
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)


def fetch_hf_metadata(model_name: str) -> Optional[Dict[str, Any]]:
    """Fetch Hugging Face metadata for a given model name."""
    url = f"https://huggingface.co/api/models/{model_name}"
    headers = {"Accept": "application/json"}
    response = requests.get(url, headers=headers)
    return response.json() if response.status_code == 200 else None


def load_hf_model(
    url: str,
) -> Tuple[
    PreTrainedModel, PreTrainedTokenizerBase | ProcessorMixin | None, Dict[str, Any]
]:
    # 1️⃣ Load metadata from the DCAT-AP Hub
    dataset, _ = get_metadata(url)
    assert dataset.is_model, "The provided metadata must describe a model."

    model_name: str = dataset.title

    # 2️⃣ Fetch Hugging Face model metadata
    hf_metadata = fetch_hf_metadata(model_name)
    if not hf_metadata:
        raise ValueError(f"Could not fetch metadata for model '{model_name}'.")

    info: Dict[str, Any] = hf_metadata.get("transformersInfo", {})
    tags = hf_metadata.get("tags", [])
    model_id: str = str(hf_metadata.get("id"))

    # 3️⃣ Determine whether remote code needs to be trusted
    trust_remote_code: bool = any("custom" in t or "remote_code" in t for t in tags)

    # 4️⃣ Dynamically import transformers
    transformers = importlib.import_module("transformers")

    # 5️⃣ Determine which Auto classes to use
    auto_model_class_name: str = info.get("auto_model", "AutoModel")
    processor_class_name: Optional[str] = info.get("processor")

    # 6️⃣ Load model
    model_class = getattr(transformers, auto_model_class_name)
    model: PreTrainedModel = model_class.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    )

    # 7️⃣ Load processor/tokenizer if available
    processor: Optional[Union[PreTrainedTokenizerBase, ProcessorMixin]] = None
    if processor_class_name:
        processor_class = getattr(transformers, processor_class_name)
        processor = processor_class.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
    else:
        # Default: for text-based models, use AutoTokenizer
        if any(
            tag.lower()
            in ["pytorch", "transformers", "bert", "text", "nlp", "language"]
            for tag in tags
        ):
            tokenizer_class = getattr(transformers, "AutoTokenizer")
            processor = tokenizer_class.from_pretrained(
                model_id, trust_remote_code=trust_remote_code
            )

    # 8️⃣ Log summary
    logger.info(f"Model loaded: {model_id} ({model.__class__.__name__})")
    if processor:
        logger.info(f"Processor loaded: {processor.__class__.__name__}")
    if trust_remote_code:
        logger.info("Used trust_remote_code=True for this model")
    return model, processor, hf_metadata


if __name__ == "__main__":
    url = "https://ki-daten.hlrs.de/hub/repo/datasets/6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837.jsonld"
    # url = "https://ki-daten.hlrs.de/hub/repo/datasets/df1b051c49625cf57a3d0d8d3863ed4d13564fe4.jsonld"
    model, processor, _ = load_hf_model(url)

    # if processor:
    #     inputs = processor("Hello world!", return_tensors="pt")
    #     outputs = model(**inputs)
    #     logger.info("Forward pass successful:", outputs.last_hidden_state.shape)
