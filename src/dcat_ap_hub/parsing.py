import importlib
from pathlib import Path
from dcat_ap_hub.metadata import (
    get_dataset_dir,
    get_dataset_title,
    get_parser_download_url,
)


def apply_parsing(metadata: dict, base_dir: Path = Path("./datasets")) -> dict:
    dataset_title = get_dataset_title(metadata)
    dataset_dir = get_dataset_dir(metadata, base_dir)
    parser_download_url = get_parser_download_url(metadata)

    if parser_download_url is None:
        raise ValueError(
            f"Metadata for {dataset_title} does not include a parser download URL"
        )

    package_name = f"{dataset_title}_parser"

    try:
        parser_module = importlib.import_module(package_name)
        parsed_dataset = parser_module.parse(dataset_dir)
    except ImportError:
        raise ImportError(
            f"Parser package for {dataset_title} not found. Please install with `pip install {parser_download_url}`"
        )

    return parsed_dataset


# def download_parser(download_url: str, output_dir: Path) -> Callable[[str], Any]:
#     """
#     Downloads and dynamically loads a parser.py from a zip archive.
#     """
#     parser_path = output_dir / "parser.py"
#     if not parser_path.exists():
#         output_dir.mkdir(parents=True, exist_ok=True)
#         zip_path = output_dir / "parser.zip"

#         download_file_with_progress(download_url, zip_path)
#         extract_archive(zip_path, output_dir)

#     # Dynamically load parser.py
#     spec = spec_from_file_location("parser_module", parser_path)
#     if spec is None or spec.loader is None:
#         raise ImportError("Failed to load parser module.")

#     module = module_from_spec(spec)
#     spec.loader.exec_module(module)

#     if not hasattr(module, "parse"):
#         raise AttributeError("Parser module must define a 'parse' function.")

#     return module.parse
