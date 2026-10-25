"""Processing workflow helpers used by :class:`dcat_ap_hub.dataset.Dataset`."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from dcat_ap_hub.metadata.constants import PROCESSOR_PROFILE_URI, SPLIT_FILES
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

    # Only a complete split counts; a failed run may leave partial output.
    candidate = local_data_path / processed_dir
    if all((candidate / name).is_file() for name in SPLIT_FILES):
        return candidate
    return None


def resolve_processor_path(
    local_data_path: Path,
    resources: Sequence[RelatedResource],
) -> Path:
    """Return the local processor script, preferring ones the dataset links to."""
    processors = [resource for resource in resources if resource.role == "processor"]
    processors.sort(key=lambda resource: not resource.linked)
    if not processors:
        raise ValueError(
            "No processor found: the metadata needs an rdfs:Resource with "
            f"dct:conformsTo {PROCESSOR_PROFILE_URI}."
        )
    processor = processors[0]
    path = _find_local_file(local_data_path, processor)
    if path is None:
        raise FileNotFoundError(
            f"Processor script not found in {local_data_path}; tried "
            f"{', '.join(processor.candidate_filenames())}"
        )
    return path


def _find_local_file(local_data_path: Path, resource: RelatedResource) -> Optional[Path]:
    names = resource.candidate_filenames()
    for name in names:
        for candidate in (local_data_path / name, local_data_path / f"{name}.py"):
            if candidate.is_file():
                return candidate
    # The downloader adds an extension from the server MIME type when the name
    # has none, e.g. ``.txt`` for a processor served as ``text/plain``.
    for name in names:
        matches = sorted(
            path
            for path in local_data_path.iterdir()
            if path.is_file() and path.stem == name
        )
        if matches:
            return matches[0]
    return None


def run_processing_pipeline(
    local_data_path: Path,
    resources: Sequence[RelatedResource],
    processed_dir: str = "processed",
    verbose: bool = False,
) -> Path:
    """
    Run the dataset's processor on its folder and return the output directory.

    The processor receives the dataset folder itself and must write every file
    in ``SPLIT_FILES`` to ``<dataset folder>/<processed_dir>``.
    """
    output_dir = local_data_path / processed_dir
    processor_path = resolve_processor_path(local_data_path, resources)
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_processor_logic(processor_path, local_data_path, output_dir, verbose=verbose)
    missing = [name for name in SPLIT_FILES if not (output_dir / name).is_file()]
    if missing:
        written = sorted(path.name for path in output_dir.iterdir())
        raise RuntimeError(
            f"Processor {processor_path.name} did not write {', '.join(missing)} "
            f"to {output_dir} (found: {', '.join(written) or 'nothing'})."
        )
    return output_dir
