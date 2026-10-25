"""JSON-LD parsing and metadata normalization utilities."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Union
from urllib import request
from urllib.parse import urljoin, urlparse

from dcat_ap_hub.metadata.constants import (
    HF_FORMAT,
    HF_METADATA_PROFILE_URI,
    CANONICAL_PREFIXES,
    MODEL_TYPE,
    ONNX_FORMAT,
    ONNX_METADATA_PROFILE_URI,
    PROCESSOR_PROFILE_URI,
    SKLEARN_METADATA_PROFILE_URI,
)
from dcat_ap_hub.metadata.models import DatasetMetadata, Distribution, RelatedResource
from dcat_ap_hub.utils.logging import logger


def _extract_value(field: Union[str, dict, list, None]) -> str:
    """Normalize a JSON-LD field to a string (the first value of a list)."""
    if isinstance(field, list):
        return next((value for value in map(_extract_value, field) if value), "")
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


# Dataset properties that link a dataset to auxiliary resources.
_RELATION_KEYS = ("dct:relation", "dct:references", "dct:hasPart", "foaf:page")


def _context_prefixes(context: Any) -> Dict[str, str]:
    """Collect ``prefix -> namespace`` mappings from a JSON-LD ``@context``."""
    prefixes: Dict[str, str] = {}
    contexts = context if isinstance(context, list) else [context]
    for item in contexts:
        if not isinstance(item, dict):
            continue
        for key, value in item.items():
            if key.startswith("@"):
                continue
            iri = value.get("@id") if isinstance(value, dict) else value
            if isinstance(iri, str) and iri.endswith(("#", "/")):
                prefixes[key] = iri
    return prefixes


def _expand(term: str, prefixes: Dict[str, str]) -> str:
    """Expand a compact IRI (``prefix:name``) using known prefixes."""
    prefix, sep, local = term.partition(":")
    if sep and not local.startswith("//") and prefix in prefixes:
        return prefixes[prefix] + local
    return term


def _compact(iri: str) -> str:
    """Compact a full IRI to the canonical prefix this parser reads."""
    for prefix, namespace in CANONICAL_PREFIXES.items():
        if iri.startswith(namespace):
            return f"{prefix}:{iri[len(namespace):]}"
    return iri


def _resolve(url: str, source_name: str) -> str:
    """
    Resolve a relative URL against the record it came from.

    A record fetched from ``https://host/run/dcat-metadata.jsonld`` that names
    ``processor.py`` points to ``https://host/run/processor.py``, so a dataset
    folder can be published as is. Records read from local files keep relative
    names; they refer to files in the same folder.
    """
    if not url or not source_name.startswith(("http://", "https://")):
        return url
    if urlparse(url).scheme:
        return url
    return urljoin(source_name, url)


def _normalize_term(term: str, prefixes: Dict[str, str]) -> str:
    return _compact(_expand(term, prefixes))


def _normalize_node(node: Any, prefixes: Dict[str, str]) -> Any:
    """
    Rewrite keys and ``@type`` values to canonical prefixes.

    This lets metadata using other prefixes (``dcterms:``), full IRIs, or a
    custom ``@context`` parse the same way as the canonical compact form.
    ``@id`` values that are compact IRIs are expanded to full IRIs.
    """
    if isinstance(node, list):
        return [_normalize_node(item, prefixes) for item in node]
    if not isinstance(node, dict):
        return node
    normalized: Dict[str, Any] = {}
    for key, value in node.items():
        if key == "@type":
            types = value if isinstance(value, list) else [value]
            normalized[key] = [
                _normalize_term(item, prefixes) if isinstance(item, str) else item
                for item in types
            ]
        elif key == "@id" and isinstance(value, str):
            normalized[key] = _expand(value, prefixes)
        elif key.startswith("@"):
            normalized[key] = value
        else:
            normalized[_normalize_term(key, prefixes)] = _normalize_node(value, prefixes)
    return normalized


def _graph_entries(data: Any) -> List[dict]:
    """Return normalized graph nodes from a JSON-LD document of any common shape."""
    if isinstance(data, list):
        return [entry for item in data for entry in _graph_entries(item)]
    if not isinstance(data, dict):
        return []
    prefixes = {**CANONICAL_PREFIXES, "dcterms": CANONICAL_PREFIXES["dct"]}
    prefixes.update(_context_prefixes(data.get("@context")))
    nodes = data["@graph"] if "@graph" in data else [data]
    if isinstance(nodes, dict):
        nodes = [nodes]
    return [
        _normalize_node(node, prefixes) for node in nodes if isinstance(node, dict)
    ]


def _parse_json_content(data: Dict, source_name: str) -> DatasetMetadata:
    """
    Convert raw JSON-LD content to normalized ``DatasetMetadata``.

    Parsing happens in two passes:
    1. Resolve dataset-level metadata.
    2. Resolve distributions and related resources.
    """
    entries = _graph_entries(data)
    dataset_meta = None
    distros = []
    linked_ids: set[str] = set()

    for entry in entries:
        types = _extract_list(entry.get("@type", []))

        if "dcat:Dataset" in types:
            if dataset_meta:
                logger.warning(
                    f"Several dcat:Dataset entries in {source_name}; using the first."
                )
                continue
            is_model = MODEL_TYPE in types
            dataset_meta = DatasetMetadata(
                title=_extract_lang_value(entry.get("dct:title", "")),
                description=_extract_lang_value(entry.get("dct:description", "")),
                is_model=is_model,
                source_url=source_name,
            )
            for key in _RELATION_KEYS:
                linked_ids.update(_extract_list(entry.get(key, [])))

    if not dataset_meta:
        raise ValueError(f"No dcat:Dataset found in {source_name}")

    # Second pass: parse distributions and related resources.
    related_resources = []

    for entry in entries:
        types = _extract_list(entry.get("@type", []))

        # Profiles may be written compactly ("hammerhai:Processor").
        conforms_to = [
            _expand(value, CANONICAL_PREFIXES)
            for value in _extract_list(entry.get("dct:conformsTo", []))
        ]
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

            # DCAT-AP allows several download URLs; each one is a separate file.
            download_urls = [
                _resolve(url, source_name)
                for url in _extract_list(entry.get("dcat:downloadURL"))
                if url
            ]
            for download_url in download_urls or [""]:
                distros.append(
                    Distribution(
                        title=_extract_lang_value(entry.get("dct:title", "")),
                        description=_extract_lang_value(
                            entry.get("dct:description", "")
                        ),
                        format=format,
                        access_url=_resolve(
                            _extract_value(entry.get("dcat:accessURL", "")), source_name
                        ),
                        download_url=download_url,
                        role=dist_role,
                    )
                )

        elif "rdfs:Resource" in types:
            title = _extract_lang_value(entry.get("dct:title", "")).lower()

            if PROCESSOR_PROFILE_URI in conforms_to:
                rel_role = "processor"
            elif "ipynb" in format or "notebook" in title:
                rel_role = "notebook"
            else:
                rel_role = "other"

            related_resources.append(
                RelatedResource(
                    title=_extract_lang_value(entry.get("dct:title", "")),
                    description=_extract_lang_value(entry.get("dct:description", "")),
                    format=format,
                    download_url=_resolve(
                        _extract_value(entry.get("dcat:downloadURL", ""))
                        or _extract_value(entry.get("dcat:accessURL", "")),
                        source_name,
                    ),
                    role=rel_role,
                    linked=entry.get("@id") in linked_ids,
                )
            )

    dataset_meta.distributions = distros
    dataset_meta.related_resources = related_resources
    return dataset_meta


JSONLD_ACCEPT_HEADER = "application/ld+json, application/json;q=0.9, */*;q=0.1"
METADATA_BEARER_TOKEN_ENV = "DCAT_AP_HUB_METADATA_BEARER_TOKEN"


def _load_env_var_from_dotenv(var_name: str, dotenv_path: Path = Path(".env")) -> str:
    """Best-effort lookup of a variable from a local .env file."""
    try:
        if not dotenv_path.exists():
            return ""
        for line in dotenv_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            if key.strip() != var_name:
                continue
            value = value.strip()
            if (
                len(value) >= 2
                and value[0] == value[-1]
                and value[0] in {"'", '"'}
            ):
                value = value[1:-1]
            return value
    except Exception:
        return ""
    return ""


def build_metadata_request_headers() -> dict[str, str]:
    """Build metadata request headers with optional bearer auth from env/.env."""
    headers = {"Accept": JSONLD_ACCEPT_HEADER}
    token = os.getenv(METADATA_BEARER_TOKEN_ENV) or _load_env_var_from_dotenv(
        METADATA_BEARER_TOKEN_ENV
    )
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def fetch_and_parse(url: str, verbose: bool = False) -> DatasetMetadata:
    """Fetch JSON-LD metadata from the web and parse it."""
    if verbose:
        logger.info(f"Fetching: {url}")
    req = request.Request(url, headers=build_metadata_request_headers())
    with request.urlopen(req) as response:
        data = json.load(response)
    return _parse_json_content(data, url)


def parse_local_file(path: Path) -> DatasetMetadata:
    """Read local JSON/JSON-LD metadata and parse it."""
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    return _parse_json_content(data, str(path))
