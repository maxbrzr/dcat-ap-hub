"""Processing workflow helpers used by :class:`dcat_ap_hub.dataset.Dataset`."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from dcat_ap_hub.metadata.models import RelatedResource
from dcat_ap_hub.processing.processor import apply_processor_logic


def find_existing_processed_path(
    local_data_path: Optional[Path],
    known_processed_path: Optional[Path] = None,
    processed_dir: str = "processed",
) -> Optional[Path]:
    """
    Resolve an existing processed output directory when available.

    ``known_processed_path`` is preferred when still valid; otherwise the
    conventional ``<local_data_path>/<processed_dir>`` location is checked.
    """
    if known_processed_path and known_processed_path.exists():
        return known_processed_path
    if not local_data_path:
        return None

    candidate = local_data_path / processed_dir
    if candidate.exists() and any(candidate.iterdir()):
        return candidate
    return None


def _resolve_processor_paths(
    local_data_path: Path,
    resources: Sequence[RelatedResource],
) -> tuple[Path, Optional[Path]]:
    """Resolve local processor and notebook paths from related resources."""
    processor_item = next((resource for resource in resources if resource.role == "processor"), None)
    notebook_item = next((resource for resource in resources if resource.role == "notebook"), None)

    if not processor_item:
        raise ValueError("No processor found in related resources.")

    processor_path = local_data_path / processor_item.get_filename()
    notebook_path = local_data_path / notebook_item.get_filename() if notebook_item else None

    if not processor_path.exists():
        processor_path = local_data_path / f"{processor_item.get_filename()}.py"
    if not processor_path.exists():
        raise FileNotFoundError(f"Processor script not found at {processor_path}")

    if notebook_path and not notebook_path.exists():
        notebook_path = None

    return processor_path, notebook_path


def build_processor_input_paths(
    local_data_path: Path,
    processor_path: Path,
    notebook_path: Optional[Path] = None,
) -> list[Path]:
    """
    Return processor input files from a dataset directory.

    The processor script, optional notebook, and saved metadata sidecar are
    excluded from the input list.
    """
    input_paths = [
        path
        for path in local_data_path.iterdir()
        if path.is_file()
        and path.name != processor_path.name
        and path.name != "dcat-metadata.jsonld"
    ]
    if notebook_path:
        input_paths = [path for path in input_paths if path.name != notebook_path.name]
    return input_paths


def run_processing_pipeline(
    local_data_path: Path,
    resources: Sequence[RelatedResource],
    processed_dir: str = "processed",
    verbose: bool = False,
) -> Path:
    """
    Execute dataset processing and return the output directory.

    The output directory is created when missing and processor logic is invoked
    exactly once for the selected processor resource.
    """
    output_dir = local_data_path / processed_dir
    processor_path, notebook_path = _resolve_processor_paths(local_data_path, resources)
    input_paths = build_processor_input_paths(
        local_data_path=local_data_path,
        processor_path=processor_path,
        notebook_path=notebook_path,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    apply_processor_logic(processor_path, input_paths, output_dir, verbose=verbose)
    return output_dir
