"""Compatibility import for the video reference analyser."""

from aivia.reference.analyze_video import *  # noqa: F403
from aivia.reference.analyze_video import __all__ as _analyze_video_all

__all__ = [*_analyze_video_all]
