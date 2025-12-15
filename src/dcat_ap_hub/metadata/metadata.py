"""Helpers to fetch and parse DCAT-AP JSON-LD metadata.

This module provides:
- fetch_metadata: download and parse a JSON-LD metadata file from a URL
- parse_metadata: extract Dataset and Distribution objects from a JSON-LD @graph
- helper extractors to normalize JSON-LD value shapes into simple strings

The functions include additional inline comments to explain JSON-LD shapes
(common forms: plain string, {'@id': ...}, {'@value': ...}, or language-tagged lists).
"""

from dataclasses import dataclass
import json
from typing import List, Tuple, Optional, Union
from urllib import request

from dcat_ap_hub.logging import logger


@dataclass
class Distribution:
    """
    Represents a DCAT distribution (a specific representation of a dataset).

    Attributes:
        title: The title of the distribution
        description: A description of what this distribution contains
        format: The file format (e.g., 'CSV', 'JSON', 'RDF')
        access_url: URL where the distribution can be accessed
        download_url: Optional direct download URL for the distribution
    """

    title: str
    description: str
    format: str
    access_url: str
    download_url: str | None = None


@dataclass
class Dataset:
    """
    Represents a DCAT dataset with metadata.

    Attributes:
        title: The title of the dataset
        description: A description of the dataset contents
        is_model: Whether this dataset represents an ML model (mls:Model type)
    """

    title: str
    description: str
    is_model: bool = False


def fetch_metadata(url: str, verbose: bool = False) -> dict:
    """
    Fetches and parses a JSON-LD metadata file from a URL.

    Notes:
    - Validates that the response Content-Type is application/ld+json (parameters like
      charset are tolerated).
    - Uses the Content-Length header for logging when available.
    """
    try:
        if verbose:
            logger.info(f"[metadata] Fetching URL: {url}")

        # Open the URL and fetch the metadata
        with request.urlopen(url) as response:
            # Validate content type (ignore MIME parameters like charset)
            content_type = response.headers.get("Content-Type", "") or ""
            mime = content_type.split(";", 1)[0].strip().lower()
            if mime != "application/ld+json":
                raise ValueError(
                    f"Invalid MIME type: {content_type!r}. Expected application/ld+json."
                )
            if verbose:
                logger.info(f"[metadata] Content-Type: {content_type}")

            # Parse the JSON content from the response stream
            metadata = json.load(response)

            # Use Content-Length header if available to log size; response objects
            # don't always expose a .length attribute.
            content_length = response.headers.get("Content-Length")
            if verbose:
                logger.info(
                    f"[metadata] Downloaded bytes: {content_length or 'unknown'}"
                )
    except Exception as e:
        # Provide context in the raised error while preserving original exception
        raise RuntimeError(f"Failed to download or parse metadata from {url}") from e

    return metadata


def extract_value(field: Union[str, dict, None]) -> str:
    """
    Normalize a JSON-LD field to a plain string.

    Handles common shapes:
    - plain string: "Example"
    - object reference: {"@id": "http://..."}
    - literal object: {"@value": "example"}
    - anything else -> empty string
    """
    # If it's a dict, check for @id (reference) first, then @value (literal)
    if isinstance(field, dict):
        if "@id" in field:
            return field["@id"]
        if "@value" in field:
            return field["@value"]

    # Plain string value
    if isinstance(field, str):
        return field

    # Unknown/unsupported shape -> return empty string
    return ""


def extract_lang_value(field: Union[str, List[dict], dict], lang: str = "en") -> str:
    """
    Extract a language-specific value from a JSON-LD multilingual field.

    Common field shapes:
    - plain string -> returned as-is
    - dict like {"@value": "..."} -> handled by extract_value
    - list of dicts like [{"@value":"...","@language":"en"}, ...] -> prefer requested lang,
      fallback to first item if requested language not found.
    """
    # Already a plain string, return it directly
    if isinstance(field, str):
        return field

    # If it's a list of language-tagged literals, try to find the requested language
    if isinstance(field, list):
        for item in field:
            # item expected to be a dict with @language and either @value or @id
            if isinstance(item, dict):
                lang_tag = item.get("@language", "")
                # Exact or prefix match (some systems use 'en-GB' etc.)
                if lang in lang_tag:
                    return extract_value(item)
        # Fallback to first item's value if present
        if field and isinstance(field[0], dict):
            return extract_value(field[0])

    # Single dict value -> normalize using extract_value
    if isinstance(field, dict):
        return extract_value(field)

    # Nothing we can extract -> empty string
    return ""


def parse_metadata(
    metadata: dict, verbose: bool = False
) -> Tuple[Dataset, List[Distribution]]:
    """
    Parse DCAT-AP JSON-LD metadata to extract the primary dataset and its distributions.

    Behaviour summary:
    - Expects a JSON-LD @graph list in the metadata (metadata.get('@graph', []))
    - Scans each node: if node has dcat:Dataset type it becomes the primary dataset
      (last seen dataset node wins). Distribution nodes (dcat:Distribution) are collected.
    - Recognizes ML model type by presence of the mls:Model full IRI in the @type array.
    """
    # Get all entries from the JSON-LD graph
    entries: List[dict] = metadata.get("@graph", [])

    distros: List[Distribution] = []
    dataset: Optional[Dataset] = None

    # Process each entry in the graph
    for idx, entry in enumerate(entries):
        try:
            entry_type = entry.get("@type")

            # Normalize @type to a Python list (it can be a string or list in JSON-LD)
            if isinstance(entry_type, str):
                types = [entry_type]
            elif isinstance(entry_type, list):
                types = entry_type
            else:
                types = []

            if verbose:
                logger.info(f"[parse_metadata] Entry {idx} types: {types}")

            # If this node is a Dataset, extract title/description and mark model-flag
            if "dcat:Dataset" in types:
                is_ml_model = "http://www.w3.org/ns/mls#Model" in types

                dataset = Dataset(
                    title=extract_lang_value(entry.get("dct:title", "")),
                    description=extract_lang_value(entry.get("dct:description", "")),
                    is_model=is_ml_model,
                )
                if verbose:
                    logger.info(
                        f"[parse_metadata] Dataset detected: title='{dataset.title}', is_model={dataset.is_model}"
                    )

            # If this node is a Distribution, collect it
            if "dcat:Distribution" in types:
                distro = Distribution(
                    title=extract_lang_value(entry.get("dct:title", "")),
                    description=extract_lang_value(entry.get("dct:description", "")),
                    format=extract_value(entry.get("dct:format", "")),
                    access_url=extract_value(entry.get("dcat:accessURL", "")),
                    download_url=extract_value(entry.get("dcat:downloadURL", "")),
                )
                distros.append(distro)
                if verbose:
                    logger.info(
                        f"[parse_metadata] Distribution added: title='{distro.title}', format='{distro.format}', access_url='{distro.access_url}'"
                    )

        except KeyError:
            # Handle missing keys gracefully and continue processing other entries
            logger.info(extract_lang_value(entry.get("dct:title", "")))
            if verbose:
                logger.info(
                    "[parse_metadata] KeyError encountered; skipping entry safely."
                )

    # Ensure we found a dataset in the metadata
    assert dataset is not None

    if verbose:
        logger.info(
            f"[parse_metadata] Parsed dataset title: {dataset.title if dataset else 'None'}"
        )
        logger.info(f"[parse_metadata] Total distributions: {len(distros)}")

    return dataset, distros


def get_metadata(url: str, verbose: bool = False) -> Tuple[Dataset, List[Distribution]]:
    """
    Convenience function to fetch and parse metadata in one step.

    Args:
        url: The URL of the JSON-LD metadata file
        verbose: If True, logs detailed information about fetch and parse process

    Returns:
        A tuple of (Dataset, List[Distribution])

    Raises:
        ValueError: If the content type is not application/ld+json
        RuntimeError: If download or parsing fails
        AssertionError: If no dataset is found in the metadata
    """
    metadata = fetch_metadata(url, verbose=verbose)
    dataset, distros = parse_metadata(metadata, verbose=verbose)
    return dataset, distros


if __name__ == "__main__":
    # Example usage: fetch and parse metadata from a sample URL
    metadata = fetch_metadata(
        "https://ki-daten.hlrs.de/hub/repo/datasets/dcc5faea-10fd-430b-944b-4ac03383ca9f~~1.jsonld",
        verbose=True,
    )
    dataset, distros = parse_metadata(metadata, verbose=True)
