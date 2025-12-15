import sys
import uuid
import inspect
import importlib.util
from pathlib import Path
from typing import List

from dcat_ap_hub.core.interfaces import BaseProcessor


def apply_processor_logic(
    parser_path: Path, input_paths: List[Path], output_dir: Path, verbose: bool = False
) -> None:
    """
    Dynamically loads a Python module and executes the BaseProcessor implementation.
    """
    if verbose:
        print(f"Loading processor module: {parser_path.name}")

    # Initialize variable outside try block to avoid UnboundLocalError in except block
    unique_mod_name = None

    try:
        # Generate a unique name to prevent collisions if multiple datasets are loaded
        # e.g., "dcat_processor_traffic_data_12345"
        unique_mod_name = f"dcat_processor_{parser_path.stem}_{uuid.uuid4().hex[:8]}"

        # 1. Load the module dynamically
        spec = importlib.util.spec_from_file_location(unique_mod_name, parser_path)
        if not spec or not spec.loader:
            raise ImportError(f"Could not load spec from {parser_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[unique_mod_name] = module  # Register with unique name
        spec.loader.exec_module(module)

        # 2. Inspect module to find the Processor class
        processor_class = None

        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Check for inheritance
            if issubclass(obj, BaseProcessor) and obj is not BaseProcessor:
                processor_class = obj
                break

        if not processor_class:
            raise AttributeError(
                f"Module '{parser_path.name}' must contain a class that inherits from 'BaseProcessor'."
            )

        # 3. Instantiate and Execute
        if verbose:
            print(f"Found processor class: {processor_class.__name__}")
            print(f"Processing {len(input_paths)} files...")

        # Instantiate
        processor_instance = processor_class()
        # Run
        processor_instance.process(input_paths, output_dir)

        if verbose:
            print(f"Success! Output saved to: {output_dir}")

    except Exception as e:
        # Clean up the module from sys.modules to prevent memory leaks
        # Only try to delete if unique_mod_name was actually assigned
        if unique_mod_name and unique_mod_name in sys.modules:
            del sys.modules[unique_mod_name]
        raise RuntimeError(f"Processor execution failed: {e}") from e
