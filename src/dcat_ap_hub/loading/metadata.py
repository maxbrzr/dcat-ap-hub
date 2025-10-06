from dataclasses import dataclass
import json
from pathlib import Path
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
            if entry["@type"] == "dcat:Distribution":
                distros.append(
                    Distribution(
                        title=entry["dct:title"],
                        description=entry["dct:description"],
                        format=entry["dct:format"],
                        access_url=entry["dcat:accessURL"]["@id"],
                    )
                )
            elif entry["@type"] == "dcat:Dataset":
                dataset = Dataset(
                    title=entry["dct:title"],
                    description=entry["dct:description"],
                )
        except KeyError:
            print(entry["dct:title"])

    assert dataset is not None

    return dataset, distros


def get_dataset_dir(dataset: Dataset, base_dir: Path) -> Path:
    base_path = Path(base_dir)
    dataset_dir = base_path / dataset.title
    return dataset_dir


if __name__ == "__main__":
    metadata = fetch_metadata(
        "https://ki-daten.hlrs.de/hub/repo/datasets/dcc5faea-10fd-430b-944b-4ac03383ca9f~~1.jsonld"
    )
    dataset, distros = parse_metadata(metadata)
