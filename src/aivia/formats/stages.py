"""Stage registry and built-in stage compatibility module.

The registry lives in :mod:`aivia.formats.registry` and the input topic stage in
:mod:`aivia.formats.input_topic`; this module re-exports both so callers that
group stage contracts with the stage types can use either import path.
"""

from aivia.formats.input_topic import *  # noqa: F403
from aivia.formats.input_topic import __all__ as _input_topic_all
from aivia.formats.registry import *  # noqa: F403
from aivia.formats.registry import __all__ as _registry_all

__all__ = [*_registry_all, *_input_topic_all]
