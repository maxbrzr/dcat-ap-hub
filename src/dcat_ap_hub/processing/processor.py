"""Dynamic processor loading and execution utilities."""

import importlib.util
import inspect
import sys
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from types import ModuleType
from typing import Callable, Protocol, Sequence, runtime_checkable


@runtime_checkable
class ProcessorLike(Protocol):
    """Structural interface every processor satisfies."""

    def process(self, input_files: Sequence[Path], output_dir: Path) -> None: ...


class DataProcessor(ABC):
    """
    Optional base class for dataset processors referenced in metadata resources.

    Processor scripts do not need to import dcat-ap-hub. A script is accepted
    when it provides, in order of precedence:

    1. a class inheriting ``DataProcessor``;
    2. a class defined in the script with a ``process(input_files, output_dir)``
       method;
    3. a module-level function ``process(input_files, output_dir)``.
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

        name, process = find_processor(module)

        if verbose:
            print(f"Found processor: {name}")
            print(f"Processing {len(input_paths)} files...")

        process(list(input_paths), output_dir)

        if verbose:
            print(f"Success! Output saved to: {output_dir}")

    except Exception as e:
        # Remove transient module registration on failure to avoid stale state.
        if unique_mod_name and unique_mod_name in sys.modules:
            del sys.modules[unique_mod_name]
        raise RuntimeError(f"Processor execution failed: {e}") from e


def find_processor(
    module: ModuleType,
) -> tuple[str, Callable[[Sequence[Path], Path], None]]:
    """
    Return a display name and callable for the processor defined in ``module``.

    See :class:`DataProcessor` for the accepted forms and their precedence.
    Classes imported into the script from elsewhere are ignored.
    """
    classes = [
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if obj.__module__ == module.__name__ or issubclass(obj, DataProcessor)
    ]

    for obj in classes:
        if issubclass(obj, DataProcessor) and obj is not DataProcessor:
            return obj.__name__, obj().process

    for obj in classes:
        if obj.__module__ == module.__name__ and callable(getattr(obj, "process", None)):
            return obj.__name__, obj().process

    function = getattr(module, "process", None)
    if inspect.isfunction(function):
        return "process()", function

    raise AttributeError(
        f"Module '{module.__name__}' must define a module-level "
        "process(input_files, output_dir) function or a class with a process "
        "method (optionally inheriting DataProcessor)."
    )
