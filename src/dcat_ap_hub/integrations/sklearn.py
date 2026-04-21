"""Scikit-learn-style scripted backend integration."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import inspect
import sys
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from dcat_ap_hub.integrations.models import (
    BackendName,
    LoadedModel,
    LoadOptions,
    ModelLoader,
    ModelSource,
)


class SKLearnModel(ABC):
    """Protocol-like abstract base class expected from scripted sklearn models."""

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the model to the training data."""
        ...

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> Any:
        """Make predictions using the fitted model."""
        ...


class SklearnLoader(ModelLoader):
    """ModelLoader implementation for sklearn-style scripted models."""

    backend = BackendName.SKLEARN

    @staticmethod
    def _is_sklearn_source_file(path: Path) -> bool:
        """Return whether a path can contain scripted sklearn model code."""
        return path.is_file() and path.suffix.lower() in {".py", ".txt"}

    @classmethod
    def _iter_source_files(cls, path: Path) -> list[Path]:
        """Resolve candidate source files from a file or directory input path."""
        if cls._is_sklearn_source_file(path):
            return [path]
        if path.is_dir():
            return sorted(
                candidate
                for candidate in path.iterdir()
                if cls._is_sklearn_source_file(candidate)
            )
        return []

    @staticmethod
    def _load_source_model(source_file: Path, class_name: str | None = None) -> SKLearnModel:
        """Dynamically import a source file and instantiate a ``SKLearnModel`` subclass."""
        unique_module_name = f"dcat_sklearn_model_{source_file.stem}_{uuid.uuid4().hex[:8]}"
        if source_file.suffix == ".txt":
            loader = importlib.machinery.SourceFileLoader(unique_module_name, str(source_file))
            spec = importlib.util.spec_from_loader(unique_module_name, loader)
        else:
            spec = importlib.util.spec_from_file_location(unique_module_name, source_file)
        if not spec or not spec.loader:
            raise ImportError(f"Could not load module from {source_file}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[unique_module_name] = module
        try:
            spec.loader.exec_module(module)
            selected = None
            for _, obj in inspect.getmembers(module, inspect.isclass):
                if not issubclass(obj, SKLearnModel) or obj is SKLearnModel:
                    continue
                if class_name and obj.__name__ != class_name:
                    continue
                selected = obj
                break
            if not selected:
                hint = f" named '{class_name}'" if class_name else ""
                raise AttributeError(
                    f"No class{hint} inheriting SKLearnModel found in '{source_file.name}'."
                )
            return selected()
        finally:
            sys.modules.pop(unique_module_name, None)

    def can_load(self, source: ModelSource) -> bool:
        """Return whether source likely contains scripted sklearn model artifacts."""
        role = source.role or ""
        if role in {"sklearn", "sklearn_model"} and source.path is not None:
            return True
        if source.path is None:
            return False
        if source.path.is_dir() and (source.path / "config.json").exists():
            return False
        return bool(self._iter_source_files(source.path))

    def load(self, source: ModelSource, options: LoadOptions) -> LoadedModel:
        """Load and instantiate the first valid sklearn-style source candidate."""
        if source.path is None:
            raise ValueError("Sklearn loader requires a filesystem path.")

        candidates = self._iter_source_files(source.path)
        if not candidates:
            raise FileNotFoundError(f"No sklearn source files found in '{source.path}'.")

        errors: list[str] = []
        for candidate in candidates:
            try:
                model = self._load_source_model(
                    source_file=candidate,
                    class_name=options.sklearn_class_name,
                )
                return LoadedModel(
                    backend=self.backend,
                    model=model,
                    adapter=None,
                    metadata=source.metadata or {},
                )
            except Exception as e:
                errors.append(f"{candidate.name}: {e}")

        raise RuntimeError(
            "Failed to load sklearn model from available candidates. "
            + " | ".join(errors)
        )
