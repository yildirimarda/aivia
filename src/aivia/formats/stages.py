"""Stage registry compatibility module.

The registry lives in :mod:`aivia.formats.registry`; this module re-exports it
so callers that group the stage contracts with the stage types can use either
import path.
"""

from aivia.formats.registry import *  # noqa: F403
from aivia.formats.registry import __all__ as __all__
