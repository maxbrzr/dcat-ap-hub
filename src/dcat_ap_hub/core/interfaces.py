from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List

import numpy as np


class DataProcessor(ABC):
    """
    Abstract base class for all dataset processors.
    Users must inherit from this and implement the process method.
    """

    @abstractmethod
    def process(self, input_files: List[Path], output_dir: Path) -> None:
        """
        Core logic to transform input files.
        :param input_files: List of paths to the raw data files.
        :param output_dir: Directory where the parsed results must be saved.
        """
        pass


class SKLearnModel(ABC):
    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fit the model to the training data.
        """
        pass

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> Any:
        """
        Make predictions using the fitted model.
        """
        pass
