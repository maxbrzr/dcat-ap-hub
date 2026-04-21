"""JSON-LD parsing and metadata normalization utilities."""

import json
from pathlib import Path
from typing import Dict, List, Union
from urllib import request

from dcat_ap_hub.metadata.constants import (
    HF_FORMAT,
    HF_METADATA_PROFILE_URI,
    MODEL_TYPE,
    ONNX_FORMAT,
    ONNX_METADATA_PROFILE_URI,
    PROCESSOR_PROFILE_URI,
    SKLEARN_METADATA_PROFILE_URI,
)
from dcat_ap_hub.metadata.models import DatasetMetadata, Distribution, RelatedResource
from dcat_ap_hub.utils.logging import logger


def _extract_value(field: Union[str, dict, None]) -> str:
    """Normalize a JSON-LD field to a string."""
    if isinstance(field, dict):
        return field.get("@id") or field.get("@value") or ""
    return field if isinstance(field, str) else ""


def _extract_list(field: Union[str, List, Dict, None]) -> List[str]:
    """Helper to extract a list of strings/URIs from a field."""
    if not field:
        return []
    if isinstance(field, str):
        return [field]
    if isinstance(field, dict):
        return [_extract_value(field)]
    if isinstance(field, list):
        return [_extract_value(item) for item in field]
    return []


def _extract_lang_value(field: Union[str, List[dict], dict], lang: str = "en") -> str:
    """Extract language-specific value with sensible fallbacks."""
    if isinstance(field, str):
        return field
    if isinstance(field, list):
        for item in field:
            if isinstance(item, dict) and lang in item.get("@language", ""):
                return _extract_value(item)
        if field:
            return _extract_value(field[0])
    if isinstance(field, dict):
        return _extract_value(field)
    return ""


def _parse_json_content(data: Dict, source_name: str) -> DatasetMetadata:
    """
    Convert raw JSON-LD content to normalized ``DatasetMetadata``.

    Parsing happens in two passes:
    1. Resolve dataset-level metadata.
    2. Resolve distributions and related resources.
    """
    entries: List[dict] = data.get("@graph", [])
    dataset_meta = None
    distros = []

    for entry in entries:
        types = _extract_list(entry.get("@type", []))

        if "dcat:Dataset" in types:
            is_model = MODEL_TYPE in types
            dataset_meta = DatasetMetadata(
                title=_extract_lang_value(entry.get("dct:title", "")),
                description=_extract_lang_value(entry.get("dct:description", "")),
                is_model=is_model,
                source_url=source_name,
            )

    if not dataset_meta:
        raise ValueError(f"No dcat:Dataset found in {source_name}")

    # Second pass: parse distributions and related resources.
    related_resources = []

    for entry in entries:
        types = _extract_list(entry.get("@type", []))

        conforms_to = _extract_list(entry.get("dct:conformsTo", []))
        format = _extract_value(entry.get("dct:format", ""))

        if "dcat:Distribution" in types:
            # Infer distribution role for backend/model dispatch.
            dist_role = "data"

            if HF_METADATA_PROFILE_URI in conforms_to or format == HF_FORMAT:
                dist_role = "huggingface_model"
            elif ONNX_METADATA_PROFILE_URI in conforms_to or format == ONNX_FORMAT:
                dist_role = "onnx_model"
            elif SKLEARN_METADATA_PROFILE_URI in conforms_to:
                dist_role = "sklearn_model"

            distros.append(
                Distribution(
                    title=_extract_lang_value(entry.get("dct:title", "")),
                    description=_extract_lang_value(entry.get("dct:description", "")),
                    format=format,
                    access_url=_extract_value(entry.get("dcat:accessURL", "")),
                    download_url=_extract_value(entry.get("dcat:downloadURL", "")),
                    role=dist_role,
                )
            )

        elif "rdfs:Resource" in types:
            rel_role = "processor"
            title = _extract_lang_value(entry.get("dct:title", "")).lower()

            if PROCESSOR_PROFILE_URI in conforms_to:
                rel_role = "processor"
            elif "ipynb" in format or "notebook" in title:
                rel_role = "notebook"

            related_resources.append(
                RelatedResource(
                    title=_extract_lang_value(entry.get("dct:title", "")),
                    description=_extract_lang_value(entry.get("dct:description", "")),
                    format=format,
                    download_url=_extract_value(entry.get("dcat:downloadURL", "")),
                    role=rel_role,
                )
            )

    dataset_meta.distributions = distros
    dataset_meta.related_resources = related_resources
    return dataset_meta


JSONLD_ACCEPT_HEADER = "application/ld+json, application/json;q=0.9, */*;q=0.1"


def fetch_and_parse(url: str, verbose: bool = False) -> DatasetMetadata:
    """Fetch JSON-LD metadata from the web and parse it."""
    if verbose:
        logger.info(f"Fetching: {url}")
    req = request.Request(url, headers={"Accept": JSONLD_ACCEPT_HEADER})
    with request.urlopen(req) as response:
        data = json.load(response)
    return _parse_json_content(data, url)


def parse_local_file(path: Path) -> DatasetMetadata:
    """Read local JSON/JSON-LD metadata and parse it."""
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    return _parse_json_content(data, str(path))
