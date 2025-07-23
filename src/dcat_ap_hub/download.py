import os
from typing import Any, Callable, Optional, Tuple
from urllib import request
import json
import requests  # type: ignore
import shutil
import zipfile
import tarfile
from importlib.util import spec_from_file_location, module_from_spec


def download_metadata(json_ld_handle: str) -> dict:
    assert json_ld_handle.split(".")[-1] == "jsonld"

    with request.urlopen(json_ld_handle) as response:
        data = json.load(response)

    return data


def download_dataset(download_url: str, dir: str) -> str:
    # check cache
    if os.path.exists(dir):
        return dir

    # create directory
    os.makedirs(dir, exist_ok=True)

    # get filename from url
    filename = download_url.split("/")[-1]
    filepath = os.path.join(dir, filename)

    # download the file
    with requests.get(download_url, stream=True) as r:
        r.raise_for_status()
        with open(filepath, "wb") as f:
            shutil.copyfileobj(r.raw, f)

    # extract if archive
    if filename.endswith(".zip"):
        with zipfile.ZipFile(filepath, "r") as zip_ref:
            zip_ref.extractall(dir)
        os.remove(filepath)
    elif filename.endswith(".tar.gz") or filename.endswith(".tgz"):
        with tarfile.open(filepath, "r:gz") as tar_ref:
            tar_ref.extractall(dir)
        os.remove(filepath)

    return dir


def download_parser(download_url: str, dir: str) -> Callable[[str], Any]:
    parser_path = os.path.join(dir, "parser.py")

    # Check cache
    if not os.path.exists(parser_path):
        # Create directory
        os.makedirs(dir, exist_ok=True)

        # Download zip
        zip_path = os.path.join(dir, "parser.zip")
        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                shutil.copyfileobj(r.raw, f)

        # Extract parser.py from zip
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(dir)
        os.remove(zip_path)

    # Dynamically import parser.py
    spec = spec_from_file_location("parser_module", parser_path)
    assert spec is not None

    module = module_from_spec(spec)

    loader = spec.loader
    assert loader is not None
    loader.exec_module(module)

    # Return the parse function
    return module.parse


def download(
    dir: str, dataset_metadata_handle: str, parser_metadata_handle: Optional[str]
) -> Tuple[str, Optional[Callable[[str], Any]]]:
    dataset_metadata = download_metadata(dataset_metadata_handle)
    dataset_title = dataset_metadata["dct:title"]
    dataset_download_url = dataset_metadata["dcat:downloadURL"]["@id"]

    assert dataset_title is not None
    assert dataset_download_url is not None

    dataset_dir = os.path.join(dir, dataset_title)
    dataset_path = download_dataset(dataset_download_url, dataset_dir)

    if parser_metadata_handle is None:
        return dataset_path, None

    parser_metadata = download_metadata(parser_metadata_handle)
    parser_title = parser_metadata["dct:title"]
    parser_download_url = parser_metadata["dcat:downloadURL"]["@id"]

    assert parser_title is not None
    assert parser_download_url is not None

    parser_dir = os.path.join(dir, parser_title)
    parse = download_parser(parser_download_url, parser_dir)

    return dataset_path, parse
