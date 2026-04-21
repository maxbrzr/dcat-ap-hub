"""Dynamic processor loading and execution utilities."""

import importlib.util
import inspect
import sys
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence


class DataProcessor(ABC):
    """
    Base class for dataset processors referenced in metadata resources.

    Custom processor scripts should subclass this type and implement
    ``process(...)``.
    """

    @abstractmethod
    def process(self, input_files: Sequence[Path], output_dir: Path) -> None:
        """
        Transform raw input files and write processed outputs.

        Args:
            input_files: Paths to raw dataset files to be consumed.
            output_dir: Destination directory for processed outputs.
        """
        ...


def apply_processor_logic(
    parser_path: Path,
    input_paths: Sequence[Path],
    output_dir: Path,
    verbose: bool = False,
) -> None:
    """
    Dynamically load a processor module and execute its ``DataProcessor`` class.

    The loader uses a unique transient module name to avoid collisions when
    processing multiple datasets in the same interpreter session.
    """
    if verbose:
        print(f"Loading processor module: {parser_path.name}")

    unique_mod_name = None

    try:
        # Use unique module names to prevent collisions between processor runs.
        unique_mod_name = f"dcat_processor_{parser_path.stem}_{uuid.uuid4().hex[:8]}"

        spec = importlib.util.spec_from_file_location(unique_mod_name, parser_path)
        if not spec or not spec.loader:
            raise ImportError(f"Could not load spec from {parser_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[unique_mod_name] = module
        spec.loader.exec_module(module)

        processor_class = None

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, DataProcessor) and obj is not DataProcessor:
                processor_class = obj
                break

        if not processor_class:
            raise AttributeError(
                f"Module '{parser_path.name}' must contain a class inheriting DataProcessor."
            )

        if verbose:
            print(f"Found processor class: {processor_class.__name__}")
            print(f"Processing {len(input_paths)} files...")

        processor_instance = processor_class()
        processor_instance.process(input_paths, output_dir)

        if verbose:
            print(f"Success! Output saved to: {output_dir}")

    except Exception as e:
        # Remove transient module registration on failure to avoid stale state.
        if unique_mod_name and unique_mod_name in sys.modules:
            del sys.modules[unique_mod_name]
        raise RuntimeError(f"Processor execution failed: {e}") from e
