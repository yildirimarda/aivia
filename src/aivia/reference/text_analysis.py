"""Compatibility import for the text reference analyser."""

from aivia.reference.analyze_text import *  # noqa: F403
from aivia.reference.analyze_text import __all__ as _analyze_text_all

__all__ = [*_analyze_text_all]
