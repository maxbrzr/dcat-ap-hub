"""Small utility helpers used by optional processing integrations."""

import importlib
import subprocess
import sys
from typing import Any


def install_package(package: str) -> None:
    """Install a package into the active Python environment via pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def from_import_or_install(package: str, module_path: str, symbol: str) -> Any:
    """
    Import ``symbol`` from ``module_path``, installing ``package`` if needed.

    This helper is intentionally simple and best suited for interactive or
    notebook workflows where auto-install behavior is acceptable.

    Examples:
      from_import_or_install("tabpfn", "tabpfn", "TabPFNRegressor")
      from_import_or_install("tabpfn", "tabpfn.constants", "ModelVersion")
    """
    try:
        module = importlib.import_module(module_path)
    except ImportError:
        install_package(package)
        module = importlib.import_module(module_path)

    try:
        return getattr(module, symbol)
    except AttributeError as e:
        raise ImportError(f"Module '{module_path}' has no attribute '{symbol}'") from e
