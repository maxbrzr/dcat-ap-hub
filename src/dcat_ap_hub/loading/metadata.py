from dataclasses import dataclass
import json
from typing import List, Tuple
from urllib import request


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


def fetch_metadata(url: str) -> dict:
    """
    Fetches and parses a JSON-LD metadata file.
    """

    try:
        with request.urlopen(url) as response:
            content_type = response.headers.get("Content-Type", "")
            if not content_type.startswith("application/ld+json"):
                raise ValueError(
                    f"Invalid MIME type: {content_type}. Expected application/ld+json."
                )

            metadata = json.load(response)
    except Exception as e:
        raise RuntimeError(f"Failed to download or parse metadata from {url}") from e

    return metadata


def parse_metadata(metadata: dict) -> Tuple[Dataset, List[Distribution]]:
    entries: List[dict] = metadata.get("@graph", [])

    distros: List[Distribution] = []
    dataset: Dataset | None = None

    for entry in entries:
        try:
            entry_type = entry.get("@type")

            if isinstance(entry_type, str):
                types = [entry_type]
            elif isinstance(entry_type, list):
                types = entry_type
            else:
                types = []

            if "dcat:Dataset" in types:
                dataset = Dataset(
                    title=extract_lang_value(entry["dct:title"]),
                    description=extract_lang_value(entry["dct:description"]),
                    is_model=True
                    if "http://www.w3.org/ns/mls#Model" in types
                    else False,
                )

            if "dcat:Distribution" in types:
                distros.append(
                    Distribution(
                        title=extract_lang_value(entry["dct:title"]),
                        description=extract_lang_value(entry["dct:description"]),
                        format=entry["dct:format"],
                        access_url=entry["dcat:accessURL"]["@id"],
                    )
                )

        except KeyError:
            print(extract_lang_value(entry["dct:title"]))

    assert dataset is not None

    return dataset, distros


def extract_lang_value(field: str | List[dict] | dict, lang: str = "en") -> str:
    """Extract the English value from a multilingual field if available."""
    if isinstance(field, str):
        return field
    if isinstance(field, list):
        for item in field:
            if isinstance(item, dict) and item.get("@language") == lang:
                return item.get("@value", "")
        # fallback: first item’s value
        if field and isinstance(field[0], dict):
            return field[0].get("@value", "")
    if isinstance(field, dict):
        return field.get("@value", "")
    return ""


def get_metadata(url: str) -> Tuple[Dataset, List[Distribution]]:
    metadata = fetch_metadata(url)
    dataset, distros = parse_metadata(metadata)
    return dataset, distros


if __name__ == "__main__":
    metadata = fetch_metadata(
        "https://ki-daten.hlrs.de/hub/repo/datasets/dcc5faea-10fd-430b-944b-4ac03383ca9f~~1.jsonld"
    )
    dataset, distros = parse_metadata(metadata)
