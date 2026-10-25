"""Build DCAT-AP records that dcat-ap-hub can process.

Producers such as the agentic data engineer use these helpers instead of
writing processor entries themselves, so the record shape is defined once.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

from dcat_ap_hub.metadata.constants import (
    CANONICAL_PREFIXES,
    PROCESSOR_FILENAME,
    PROCESSOR_PROFILE_URI,
)
from dcat_ap_hub.metadata.parsing import _context_prefixes, _extract_list, _normalize_node

_DATASET_TYPE = "dcat:Dataset"
_RELATION_KEY = "dct:relation"


def add_processor(
    document: Dict[str, Any],
    *,
    dataset_id: Optional[str] = None,
    title: str = "Dataset",
    filename: str = PROCESSOR_FILENAME,
    download_url: Optional[str] = None,
    description: str = "Processor producing train.csv and test.csv from the raw files.",
) -> Dict[str, Any]:
    """
    Return a copy of a JSON-LD record that declares ``filename`` as its processor.

    The dataset node is found the way the parser finds it, whatever prefixes
    or ``@context`` the record uses; if there is none, one is created from
    ``dataset_id`` and ``title``. The processor is an ``rdfs:Resource`` with
    ``dct:conformsTo`` the HammerHAI Processor profile, linked from the dataset
    with ``dct:relation``. ``download_url`` defaults to ``filename``: relative,
    so it names the file next to the record, locally or once published.
    """
    document = copy.deepcopy(document)
    graph = document.get("@graph")
    if not isinstance(graph, list):
        # A single-node record: move the node into a @graph.
        node = {key: value for key, value in document.items() if key != "@context"}
        graph = [node] if node.get("@type") else []
        document = {key: value for key, value in document.items() if key == "@context"}
    prefixes = {**CANONICAL_PREFIXES, "dcterms": CANONICAL_PREFIXES["dct"]}
    prefixes.update(_context_prefixes(document.get("@context")))

    dataset = next(
        (
            node
            for node in graph
            if isinstance(node, dict)
            and _DATASET_TYPE in _extract_list(_normalize_node(node, prefixes).get("@type"))
        ),
        None,
    )
    if dataset is None:
        dataset = {"@id": dataset_id, "@type": _DATASET_TYPE, "dct:title": title}
        graph.insert(0, dataset)
    base_id = dataset.get("@id") or dataset_id or "_:dataset"
    processor_id = f"{base_id}#processor"

    graph = [
        node
        for node in graph
        if not (isinstance(node, dict) and node.get("@id") == processor_id)
    ]
    graph.append(
        {
            "@id": processor_id,
            "@type": "rdfs:Resource",
            "dct:title": filename,
            "dct:description": description,
            "dct:format": "text/x-python",
            "dct:conformsTo": {"@id": PROCESSOR_PROFILE_URI},
            "dcat:downloadURL": download_url or filename,
        }
    )

    # Extend whichever key the record already uses for dct:relation.
    relation_key = next(
        (
            key
            for key in dataset
            if not key.startswith("@")
            and next(iter(_normalize_node({key: None}, prefixes))) == _RELATION_KEY
        ),
        _RELATION_KEY,
    )
    relations = dataset.get(relation_key, [])
    relations = relations if isinstance(relations, list) else [relations]
    if processor_id not in _extract_list(relations):
        relations.append({"@id": processor_id})
    dataset[relation_key] = relations

    document["@graph"] = graph
    document["@context"] = _with_prefixes(document.get("@context"))
    return document


def _with_prefixes(context: Any) -> Any:
    """Make sure the prefixes used by ``add_processor`` resolve in JSON-LD."""
    needed = {prefix: CANONICAL_PREFIXES[prefix] for prefix in ("dcat", "dct", "rdfs")}
    if context is None:
        return needed
    if isinstance(context, dict):
        return {**needed, **context}
    contexts = context if isinstance(context, list) else [context]
    return [*contexts, needed]
