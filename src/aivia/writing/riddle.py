"""The riddle writing mode, re-exported from the shared humour module."""

from aivia.writing.humor import *  # noqa: F403
from aivia.writing.humor import __all__ as _humor_all

__all__ = [
    name
    for name in _humor_all
    if name.startswith("Riddle")
    or name.startswith("build_riddle")
    or name
    in {
        "coerce_riddle",
        "extract_riddle",
        "normalise_riddle",
        "normalize_riddle",
        "parse_riddle",
        "riddle_script",
        "validate_riddle",
    }
]
