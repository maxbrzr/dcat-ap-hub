from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

PROCESSOR_PROFILE_URI = "http://example.org/profiles/Processor"


class BaseProcessor(ABC):
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
