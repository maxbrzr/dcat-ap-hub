"""JSON-LD fetching and parsing logic."""

import json
from pathlib import Path
from typing import Dict, List, Union
from urllib import request

from dcat_ap_hub.internals.constants import (
    HF_FORMAT,
    HF_METADATA_PROFILE_URI,
    ONNX_FORMAT,
    ONNX_METADATA_PROFILE_URI,
    PROCESSOR_PROFILE_URI,
    SKLEARN_METADATA_PROFILE_URI,
)
from dcat_ap_hub.internals.logging import logger
from dcat_ap_hub.internals.models import DatasetMetadata, Distribution, RelatedResource

JSONLD_ACCEPT_HEADER = "application/ld+json, application/json;q=0.9, */*;q=0.1"


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
    """Extract language-specific value."""
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


def parse_json_content(data: Dict, source_name: str) -> DatasetMetadata:
    """
    Pure logic: converts a raw JSON-LD dictionary into a DatasetMetadata object.
    """
    entries: List[dict] = data.get("@graph", [])
    dataset_meta = None
    distros = []

    for entry in entries:
        types = _extract_list(entry.get("@type", []))

        if "dcat:Dataset" in types:
            is_model = "http://www.w3.org/ns/mls#Model" in types
            dataset_meta = DatasetMetadata(
                title=_extract_lang_value(entry.get("dct:title", "")),
                description=_extract_lang_value(entry.get("dct:description", "")),
                is_model=is_model,
                source_url=source_name,
            )

    if not dataset_meta:
        raise ValueError(f"No dcat:Dataset found in {source_name}")

    # 2. Second pass: Parse distributions and related resources
    related_resources = []

    for entry in entries:
        types = _extract_list(entry.get("@type", []))

        # Determine role and attributes
        conforms_to = _extract_list(entry.get("dct:conformsTo", []))
        format = _extract_value(entry.get("dct:format", ""))

        if "dcat:Distribution" in types:
            # Determine role for Distribution (data, model)
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
            # It's a related resource (processor, notebook)
            rel_role = "processor"  # Default
            title = _extract_lang_value(entry.get("dct:title", "")).lower()

            if PROCESSOR_PROFILE_URI in conforms_to:
                rel_role = "processor"
            elif "ipynb" in format or "notebook" in title:
                rel_role = "notebook"

            # The definition of "processor" is broad (script, tool), so default is acceptable if not explicitly notebook

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


def fetch_and_parse(url: str, verbose: bool = False) -> DatasetMetadata:
    """Fetch from web and parse."""
    if verbose:
        logger.info(f"Fetching: {url}")
    req = request.Request(url, headers={"Accept": JSONLD_ACCEPT_HEADER})
    with request.urlopen(req) as response:
        data = json.load(response)
    return parse_json_content(data, url)


def parse_local_file(path: Path) -> DatasetMetadata:
    """Read from disk and parse."""
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    return parse_json_content(data, str(path))
