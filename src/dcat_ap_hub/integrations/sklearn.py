"""Scikit-learn-like model loading integration."""

import importlib.machinery
import importlib.util
import inspect
import sys
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from dcat_ap_hub.integrations.models import (
    BackendName,
    LoadedModel,
    LoadOptions,
    ModelLoader,
    ModelSource,
)


class SKLearnModel(ABC):
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

    def can_load(self, source: ModelSource) -> bool:
        role = source.role or ""
        if role in {"sklearn", "sklearn_model"}:
            return True
        if source.path is None:
            return False
        if source.path.is_file() and source.path.suffix.lower() in {".py", ".txt"}:
            return True
        if source.path.is_dir() and (
            any(source.path.glob("*.py")) or any(source.path.glob("*.txt"))
        ):
            return True
        return False

    def load(self, source: ModelSource, options: LoadOptions) -> LoadedModel:
        if source.path is None:
            raise ValueError("Sklearn loader requires a filesystem path.")

        path = source.path
        candidates: list[Path] = []
        if path.is_file():
            candidates = [path]
        else:
            for pattern in ("*.py", "*.txt"):
                candidates.extend(sorted(path.glob(pattern)))

        if not candidates:
            raise FileNotFoundError(f"No sklearn source files found in '{path}'.")

        errors: list[str] = []
        for candidate in candidates:
            try:
                model = self._load_sklearn_model(
                    model_path=candidate, class_name=options.sklearn_class_name
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

    def _load_sklearn_model(
        self, model_path: Union[str, Path], class_name: Optional[str] = None
    ) -> SKLearnModel:
        """
        Load a sklearn model by dynamically importing a Python module and
        instantiating a class that inherits from SKLearnModel.
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"SKLearn model file not found at: {path}")

        # Python module path containing SKLearnModel subclass
        if path.suffix != ".py" and path.suffix != ".txt":
            raise ValueError(
                f"Unsupported sklearn model source '{path.name}'. Expected a .py or .txt module."
            )

        unique_mod_name = f"dcat_sklearn_model_{path.stem}_{uuid.uuid4().hex[:8]}"

        # Non-.py scripts (e.g. .txt) need an explicit source loader.
        if path.suffix == ".txt":
            loader = importlib.machinery.SourceFileLoader(unique_mod_name, str(path))
            spec = importlib.util.spec_from_loader(unique_mod_name, loader)
        else:
            spec = importlib.util.spec_from_file_location(unique_mod_name, path)
        if not spec or not spec.loader:
            raise ImportError(f"Could not load module from {path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[unique_mod_name] = module
        try:
            spec.loader.exec_module(module)

            selected_class = None
            for _, obj in inspect.getmembers(module, inspect.isclass):
                if not issubclass(obj, SKLearnModel) or obj is SKLearnModel:
                    continue
                if class_name and obj.__name__ != class_name:
                    continue
                selected_class = obj
                break

            if not selected_class:
                hint = f" named '{class_name}'" if class_name else ""
                raise AttributeError(
                    f"No class{hint} inheriting SKLearnModel found in '{path.name}'."
                )

            return selected_class()
        finally:
            sys.modules.pop(unique_mod_name, None)


def load_sklearn_model(
    model_path: Union[str, Path], class_name: Optional[str] = None
) -> SKLearnModel:
    """Backward-compatible API for sklearn scripted model loading."""
    loader = SklearnLoader()
    return loader._load_sklearn_model(model_path=model_path, class_name=class_name)
