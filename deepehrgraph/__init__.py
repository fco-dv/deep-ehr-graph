"""Top-level package for DeepEHRGraph."""
from __future__ import annotations

import importlib.metadata

__version__ = importlib.metadata.version("deepehrgraph")


def print_version() -> None:
    """Prints the version of the package."""
    print(__version__)
