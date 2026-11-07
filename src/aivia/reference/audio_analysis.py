"""Compatibility import for the audio reference analyser."""

from aivia.reference.analyze_audio import *  # noqa: F403
from aivia.reference.analyze_audio import __all__ as _analyze_audio_all

__all__ = [*_analyze_audio_all]
