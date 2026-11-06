"""Compatibility import path for loading format recipe files.

The loader lives in :mod:`aivia.formats.catalog`; this module re-exports it so
callers that group recipe loading under a ``loader`` name can use either path.
"""

from aivia.formats.catalog import *  # noqa: F403
from aivia.formats.catalog import __all__ as __all__
