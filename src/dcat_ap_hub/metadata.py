import json
from pathlib import Path
from typing import List
from urllib import request


def fetch_metadata(json_ld_handle: str) -> dict:
    """
    Fetches and parses a JSON-LD metadata file.
    """

    try:
        with request.urlopen(json_ld_handle) as response:
            content_type = response.headers.get("Content-Type", "")
            if not content_type.startswith("application/ld+json"):
                raise ValueError(
                    f"Invalid MIME type: {content_type}. Expected application/ld+json."
                )

            metadata = json.load(response)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download or parse metadata from {json_ld_handle}"
        ) from e

    return metadata


def get_dataset_title(metadata: dict) -> str:
    distros: List[dict] = metadata.get("@graph", [])

    match len(distros):
        case 2:
            dataset_title = distros[1]["dct:title"]
        case 3:
            dataset_title = distros[2]["dct:title"]
        case _:
            raise ValueError("Metadata file is not valid")

    return dataset_title


def get_dataset_dir(metadata: dict, base_dir: Path) -> Path:
    base_path = Path(base_dir)

    distros: List[dict] = metadata.get("@graph", [])

    match len(distros):
        case 2:
            dataset_title = distros[1]["dct:title"]
        case 3:
            dataset_title = distros[2]["dct:title"]
        case _:
            raise ValueError("Metadata file is not valid")

    return Path(base_path / dataset_title)


def get_data_download_url(metadata: dict) -> str:
    distros: List[dict] = metadata.get("@graph", [])

    match len(distros):
        case 2:
            data_download_url = distros[0].get("dcat:downloadURL", {}).get("@id")
        case 3:
            data_download_url = distros[0].get("dcat:downloadURL", {}).get("@id")
        case _:
            raise ValueError("Metadata file is not valid")

    return data_download_url


def get_parser_download_url(metadata: dict) -> str | None:
    distros: List[dict] = metadata.get("@graph", [])

    match len(distros):
        case 3:
            parser_download_url = distros[1].get("dcat:downloadURL", {}).get("@id")
        case _:
            parser_download_url = None

    return parser_download_url
