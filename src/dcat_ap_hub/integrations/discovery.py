"""Model discovery and source-building helpers used by :class:`Dataset`."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence, Union, cast

from dcat_ap_hub.integrations.models import BackendName, LoadOptions, ModelSource
from dcat_ap_hub.metadata.constants import HF_FORMAT, ONNX_FORMAT
from dcat_ap_hub.metadata.models import DatasetMetadata, Distribution

ModelRole = Literal["huggingface_model", "onnx_model", "sklearn_model"]
MODEL_DIST_ROLES = ("huggingface_model", "onnx_model", "sklearn_model")
ROLE_TO_BACKEND: dict[ModelRole, BackendName] = {
    "huggingface_model": BackendName.HUGGINGFACE,
    "onnx_model": BackendName.ONNX,
    "sklearn_model": BackendName.SKLEARN,
}


@dataclass(frozen=True)
class ModelLoadPlan:
    """
    Resolved model-loading plan for one backend.

    This object keeps backend selection, source candidates, and load options
    bundled so callers can execute loading in one step.
    """

    backend: BackendName
    sources: list[ModelSource]
    options: LoadOptions


def collect_search_paths(*paths: Optional[Path]) -> list[Path]:
    """Normalize local search paths while removing duplicates and missing paths."""
    results: list[Path] = []
    for path in paths:
        if path and path.exists() and path not in results:
            results.append(path)
    return results


def iter_files_with_extensions(search_paths: Sequence[Path], *extensions: str) -> list[Path]:
    """Return unique files from search paths matching the given extensions."""
    normalized = {ext.lower().lstrip(".") for ext in extensions}
    seen: set[Path] = set()
    matches: list[Path] = []
    for root in search_paths:
        if root.is_file():
            ext = root.suffix.lower().lstrip(".")
            if ext in normalized and root not in seen:
                seen.add(root)
                matches.append(root)
            continue

        for ext in normalized:
            for path in root.glob(f"*.{ext}"):
                if path not in seen:
                    seen.add(path)
                    matches.append(path)
    return matches


def _first_distribution_by_role(
    metadata: DatasetMetadata, role: str, target_format: Optional[str] = None
) -> Optional[Distribution]:
    """Return the first distribution matching role and optional format."""
    return next(
        (
            d
            for d in metadata.distributions
            if d.role == role and (target_format is None or d.format == target_format)
        ),
        None,
    )


def _find_file_by_extension(search_paths: Sequence[Path], extension: str) -> Optional[Path]:
    """Return the first matching file by extension across search paths."""
    matches = iter_files_with_extensions(search_paths, extension)
    return matches[0] if matches else None


def _has_local_hf_artifacts(search_paths: Sequence[Path]) -> bool:
    """Return whether local files indicate a Hugging Face repository structure."""
    for path in search_paths:
        if path.is_dir() and (path / "config.json").exists():
            return True
        if path.is_file() and path.name == "config.json":
            return True
    return False


def _looks_like_sklearn_source(path: Path) -> bool:
    """Heuristic check for sklearn scripted model source files."""
    lower_name = path.stem.lower()
    if "sklearn" in lower_name or "model" in lower_name:
        return True
    try:
        return "SKLearnModel" in path.read_text(encoding="utf-8")
    except Exception:
        return False


def detect_local_model_roles(search_paths: Sequence[Path]) -> set[ModelRole]:
    """Detect possible local model roles from filesystem artifacts."""
    detected: set[ModelRole] = set()
    if _find_file_by_extension(search_paths, "onnx"):
        detected.add("onnx_model")
    if _has_local_hf_artifacts(search_paths):
        detected.add("huggingface_model")
    if any(
        _looks_like_sklearn_source(path)
        for path in iter_files_with_extensions(search_paths, "py", "txt")
    ):
        detected.add("sklearn_model")
    return detected


def _distribution_model_roles(metadata: DatasetMetadata) -> set[ModelRole]:
    """Collect model roles declared in metadata distributions."""
    return cast(
        set[ModelRole],
        {d.role for d in metadata.distributions if d.role in MODEL_DIST_ROLES},
    )


def detect_model_role(metadata: DatasetMetadata, search_paths: Sequence[Path]) -> ModelRole:
    """
    Detect one model role from local artifacts first, then metadata roles.
    Raises when none or multiple candidates are detected.
    """
    local_candidates = detect_local_model_roles(search_paths)
    if len(local_candidates) == 1:
        return next(iter(local_candidates))
    if len(local_candidates) > 1:
        detected = ", ".join(sorted(local_candidates))
        raise ValueError(
            f"Ambiguous local model artifacts detected ({detected}). Keep only one model type."
        )

    roles = _distribution_model_roles(metadata)
    if len(roles) == 1:
        return next(iter(roles))
    if len(roles) > 1:
        detected = ", ".join(sorted(roles))
        raise ValueError(
            f"Ambiguous model roles in metadata ({detected}). Keep only one role."
        )

    raise ValueError(
        "Could not detect model type. Expected exactly one of: ONNX (.onnx), "
        "Hugging Face (config.json), or sklearn (SKLearnModel source)."
    )


def load_sidecar_metadata(
    metadata: DatasetMetadata,
    search_paths: Sequence[Path],
    target_role: str,
    target_format: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Load sidecar JSON metadata for a specific role/format when available.

    Sidecar metadata is looked up from deterministic distribution-derived
    filenames in local search paths.
    """
    if not search_paths:
        return None

    dist = _first_distribution_by_role(metadata, target_role, target_format)
    if not dist:
        return None

    base_name = dist.get_filename()
    candidates: list[Path] = []
    for root in search_paths:
        if root.is_file():
            continue
        candidates.extend([root / base_name, root / f"{base_name}.json"])

    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            return json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
    return None


def sklearn_candidate_paths(
    metadata: DatasetMetadata,
    search_paths: Sequence[Path],
) -> list[Path]:
    """Build sklearn source candidates with metadata-preferred paths first."""
    candidate_paths: list[Path] = []
    seen: set[Path] = set()

    sklearn_dist = _first_distribution_by_role(metadata, "sklearn_model")
    if sklearn_dist:
        base_name = sklearn_dist.get_filename()
        for root in search_paths:
            if root.is_file():
                continue
            for suffix in ("", ".py", ".txt"):
                candidate = root / f"{base_name}{suffix}"
                if candidate.exists() and candidate not in seen:
                    seen.add(candidate)
                    candidate_paths.append(candidate)

    for candidate in iter_files_with_extensions(search_paths, "py", "txt"):
        if candidate not in seen:
            seen.add(candidate)
            candidate_paths.append(candidate)

    return candidate_paths


def sources_for_role(
    role: ModelRole,
    metadata: DatasetMetadata,
    search_paths: Sequence[Path],
) -> list[ModelSource]:
    """
    Build ordered integration sources for a resolved model role.

    The ordering matters for fallbacks: metadata-derived filenames are preferred
    before broad extension scans when possible.
    """
    if role == "onnx_model":
        onnx_path = _find_file_by_extension(search_paths, "onnx")
        if not onnx_path:
            raise FileNotFoundError("ONNX file not found in local paths.")
        return [
            ModelSource(
                path=onnx_path,
                metadata=load_sidecar_metadata(
                    metadata,
                    search_paths,
                    "onnx_model",
                    target_format=ONNX_FORMAT,
                ),
                role=role,
            )
        ]

    if role == "huggingface_model":
        local_hf_dir = next(
            (path for path in search_paths if path.is_dir() and (path / "config.json").exists()),
            None,
        )
        return [
            ModelSource(
                path=local_hf_dir,
                model_id=metadata.title if local_hf_dir is None else None,
                metadata=load_sidecar_metadata(
                    metadata,
                    search_paths,
                    "huggingface_model",
                    target_format=HF_FORMAT,
                ),
                role=role,
            )
        ]

    meta = load_sidecar_metadata(metadata, search_paths, "sklearn_model") or {}
    candidates = sklearn_candidate_paths(metadata, search_paths)
    if not candidates:
        raise FileNotFoundError(
            "No sklearn model source found. Expected a Python script implementing SKLearnModel."
        )
    return [ModelSource(path=path, metadata=meta, role=role) for path in candidates]


def build_load_options(
    backend: BackendName,
    model_dir: Union[str, Path],
    token: Optional[str],
    device_map: Union[str, Dict[str, Any]],
    dtype: str,
    trust_remote_code: bool,
    load_task_specific_head: bool,
    onnx_providers: Optional[list[str]],
) -> LoadOptions:
    """Build backend-specific loading options from user parameters."""
    if backend is BackendName.HUGGINGFACE:
        return LoadOptions(
            token=token,
            cache_dir=model_dir,
            device_map=device_map,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            load_task_specific_head=load_task_specific_head,
        )
    if backend is BackendName.ONNX:
        return LoadOptions(onnx_providers=onnx_providers)
    return LoadOptions()


def build_model_load_plan(
    metadata: DatasetMetadata,
    search_paths: Sequence[Path],
    model_dir: Union[str, Path],
    token: Optional[str],
    device_map: Union[str, Dict[str, Any]],
    dtype: str,
    trust_remote_code: bool,
    load_task_specific_head: bool,
    onnx_providers: Optional[list[str]],
) -> ModelLoadPlan:
    """
    Resolve backend, source candidates, and options for one load call.

    Raises:
        ValueError: If model role detection is ambiguous or impossible.
        FileNotFoundError: If required artifacts for a resolved role are missing.
    """
    role = detect_model_role(metadata, search_paths)
    backend = ROLE_TO_BACKEND[role]
    return ModelLoadPlan(
        backend=backend,
        sources=sources_for_role(role, metadata, search_paths),
        options=build_load_options(
            backend=backend,
            model_dir=model_dir,
            token=token,
            device_map=device_map,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            load_task_specific_head=load_task_specific_head,
            onnx_providers=onnx_providers,
        ),
    )
