"""Compatibility module for the input topic stage.

The stage lives in :mod:`aivia.formats.input_topic`; this module re-exports it
for callers that reach the input stage through the shorter import path.
"""

from aivia.formats.input_topic import *  # noqa: F403
from aivia.formats.input_topic import __all__ as __all__
