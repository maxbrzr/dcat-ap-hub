from dataclasses import dataclass
import json
from typing import List, Tuple
from urllib import request

from dcat_ap_hub.logging import logger


@dataclass
class Distribution:
    title: str
    description: str
    format: str
    access_url: str
    download_url: str | None = None


@dataclass
class Dataset:
    title: str
    description: str
    is_model: bool = False


def fetch_metadata(url: str, verbose: bool = False) -> dict:
    """
    Fetches and parses a JSON-LD metadata file.
    """

    try:
        if verbose:
            logger.info(f"[metadata] Fetching URL: {url}")
        with request.urlopen(url) as response:
            content_type = response.headers.get("Content-Type", "")
            if verbose:
                logger.info(f"[metadata] Content-Type: {content_type}")
            if not content_type.startswith("application/ld+json"):
                raise ValueError(
                    f"Invalid MIME type: {content_type}. Expected application/ld+json."
                )

            metadata = json.load(response)
            if verbose:
                logger.info(
                    f"[metadata] Downloaded bytes: {response.length or 'unknown'}"
                )
    except Exception as e:
        raise RuntimeError(f"Failed to download or parse metadata from {url}") from e

    return metadata


def extract_value(field) -> str:
    """Extract @id if present, otherwise return the value directly."""
    if isinstance(field, dict):
        if "@id" in field:
            return field["@id"]
        if "@value" in field:
            return field["@value"]
    if isinstance(field, str):
        return field
    return ""


def extract_lang_value(field: str | List[dict] | dict, lang: str = "en") -> str:
    """Extract the English value from a multilingual field if available."""
    if isinstance(field, str):
        return field
    if isinstance(field, list):
        for item in field:
            if isinstance(item, dict) and lang in item.get("@language", ""):
                return extract_value(item)
        # fallback: first item's value
        if field and isinstance(field[0], dict):
            return extract_value(field[0])
    if isinstance(field, dict):
        return extract_value(field)
    return ""


def parse_metadata(
    metadata: dict, verbose: bool = False
) -> Tuple[Dataset, List[Distribution]]:
    entries: List[dict] = metadata.get("@graph", [])

    distros: List[Distribution] = []
    dataset: Dataset | None = None

    for idx, entry in enumerate(entries):
        try:
            entry_type = entry.get("@type")

            if isinstance(entry_type, str):
                types = [entry_type]
            elif isinstance(entry_type, list):
                types = entry_type
            else:
                types = []

            if verbose:
                logger.info(f"[parse_metadata] Entry {idx} types: {types}")

            if "dcat:Dataset" in types:
                dataset = Dataset(
                    title=extract_lang_value(entry.get("dct:title", "")),
                    description=extract_lang_value(entry.get("dct:description", "")),
                    is_model=True
                    if "http://www.w3.org/ns/mls#Model" in types
                    else False,
                )
                if verbose:
                    logger.info(
                        f"[parse_metadata] Dataset detected: title='{dataset.title}', is_model={dataset.is_model}"
                    )

            if "dcat:Distribution" in types:
                distro = Distribution(
                    title=extract_lang_value(entry.get("dct:title", "")),
                    description=extract_lang_value(entry.get("dct:description", "")),
                    format=extract_value(entry.get("dct:format", "")),
                    access_url=extract_value(entry.get("dcat:accessURL", "")),
                )
                distros.append(distro)
                if verbose:
                    logger.info(
                        f"[parse_metadata] Distribution added: title='{distro.title}', format='{distro.format}', access_url='{distro.access_url}'"
                    )

        except KeyError:
            # Safe access in error path
            logger.info(extract_lang_value(entry.get("dct:title", "")))
            if verbose:
                logger.info(
                    "[parse_metadata] KeyError encountered; skipping entry safely."
                )
    assert dataset is not None

    if verbose:
        logger.info(
            f"[parse_metadata] Parsed dataset title: {dataset.title if dataset else 'None'}"
        )
        logger.info(f"[parse_metadata] Total distributions: {len(distros)}")
    return dataset, distros


def get_metadata(url: str, verbose: bool = False) -> Tuple[Dataset, List[Distribution]]:
    metadata = fetch_metadata(url, verbose=verbose)
    dataset, distros = parse_metadata(metadata, verbose=verbose)
    return dataset, distros


if __name__ == "__main__":
    metadata = fetch_metadata(
        "https://ki-daten.hlrs.de/hub/repo/datasets/dcc5faea-10fd-430b-944b-4ac03383ca9f~~1.jsonld",
        verbose=True,
    )
    dataset, distros = parse_metadata(metadata, verbose=True)
