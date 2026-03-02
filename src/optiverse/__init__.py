# src/optiverse/__init__.py
from __future__ import annotations

# Keep the package import ultra-lightweight
try:
    # Python 3.8+: importlib.metadata is in stdlib
    from importlib.metadata import version, PackageNotFoundError  # type: ignore
    try:
        __version__ = version("optiverse")
    except PackageNotFoundError:
        __version__ = "0.0.0+dev"
except Exception:
    __version__ = "0.0.0+dev"

__all__ = ["__version__"]