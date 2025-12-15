"""JSON-LD fetching and parsing logic."""

import json
from urllib import request
from typing import List, Union, Dict
from pathlib import Path

from dcat_ap_hub.internals.logging import logger
from dcat_ap_hub.internals.models import (
    DatasetMetadata,
    Distribution,
    PROCESSOR_PROFILE_URI,
    HF_METADATA_PROFILE_URI,
)


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

        if "dcat:Distribution" in types:
            # Check 'dct:conformsTo' to determine role
            conforms_to = _extract_list(entry.get("dct:conformsTo", []))

            if PROCESSOR_PROFILE_URI in conforms_to:
                role = "processor"
            elif HF_METADATA_PROFILE_URI in conforms_to:
                role = "hf-metadata"
            else:
                role = "data"

            distros.append(
                Distribution(
                    title=_extract_lang_value(entry.get("dct:title", "")),
                    description=_extract_lang_value(entry.get("dct:description", "")),
                    format=_extract_value(entry.get("dct:format", "")),
                    access_url=_extract_value(entry.get("dcat:accessURL", "")),
                    download_url=_extract_value(entry.get("dcat:downloadURL", "")),
                    role=role,
                )
            )

    if not dataset_meta:
        raise ValueError(f"No dcat:Dataset found in {source_name}")

    dataset_meta.distributions = distros
    return dataset_meta


def fetch_and_parse(url: str, verbose: bool = False) -> DatasetMetadata:
    """Fetch from web and parse."""
    if verbose:
        logger.info(f"Fetching: {url}")
    with request.urlopen(url) as response:
        data = json.load(response)
    return parse_json_content(data, url)


def parse_local_file(path: Path) -> DatasetMetadata:
    """Read from disk and parse."""
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    return parse_json_content(data, str(path))
