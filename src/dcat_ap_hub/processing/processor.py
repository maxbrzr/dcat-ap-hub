"""Dynamic processor loading and execution utilities."""

import importlib.machinery
import importlib.util
import inspect
import sys
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from types import ModuleType
from typing import Callable


class DataProcessor(ABC):
    """
    Optional base class for dataset processors referenced in metadata resources.

    Processor scripts do not need to import dcat-ap-hub. A script is accepted
    when it provides, in order of precedence:

    1. a module-level function ``process(data_dir, output_dir)``;
    2. a class inheriting ``DataProcessor``.
    """

    @abstractmethod
    def process(self, data_dir: Path, output_dir: Path) -> None:
        """
        Turn the raw files in ``data_dir`` into the AI-ready split.

        Args:
            data_dir: The dataset folder. Besides the raw files (possibly in
                sub-folders extracted from archives) it holds the processor
                script, ``dcat-metadata.jsonld``, and ``output_dir``; a
                processor reads only the raw files it needs.
            output_dir: Directory to write ``train.csv`` and ``test.csv`` to.
        """
        ...


def apply_processor_logic(
    processor_path: Path,
    data_dir: Path,
    output_dir: Path,
    verbose: bool = False,
) -> None:
    """
    Load a processor script and run it on ``data_dir``.

    The loader uses a unique transient module name to avoid collisions when
    processing multiple datasets in the same interpreter session.
    """
    if verbose:
        print(f"Loading processor module: {processor_path.name}")

    unique_mod_name = None

    try:
        # Use unique module names to prevent collisions between processor runs.
        unique_mod_name = f"dcat_processor_{processor_path.stem}_{uuid.uuid4().hex[:8]}"

        # An explicit loader also accepts scripts stored without a ``.py`` suffix.
        loader = importlib.machinery.SourceFileLoader(unique_mod_name, str(processor_path))
        spec = importlib.util.spec_from_file_location(
            unique_mod_name, processor_path, loader=loader
        )
        if not spec or not spec.loader:
            raise ImportError(f"Could not load spec from {processor_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[unique_mod_name] = module
        spec.loader.exec_module(module)

        name, process = find_processor(module)

        if verbose:
            print(f"Found processor: {name}")

        process(data_dir, output_dir)

        if verbose:
            print(f"Success! Output saved to: {output_dir}")

    except Exception as e:
        # Remove transient module registration on failure to avoid stale state.
        if unique_mod_name and unique_mod_name in sys.modules:
            del sys.modules[unique_mod_name]
        raise RuntimeError(f"Processor execution failed: {e}") from e


def find_processor(module: ModuleType) -> tuple[str, Callable[[Path, Path], None]]:
    """
    Return a display name and callable for the processor defined in ``module``.

    See :class:`DataProcessor` for the accepted forms and their precedence.
    Helper classes that happen to have a ``process`` method are ignored.
    """
    function = getattr(module, "process", None)
    if inspect.isfunction(function):
        return "process()", function

    for _, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, DataProcessor) and obj is not DataProcessor:
            return obj.__name__, obj().process

    raise AttributeError(
        f"Module '{module.__name__}' must define a module-level "
        "process(data_dir, output_dir) function or a DataProcessor subclass."
    )
