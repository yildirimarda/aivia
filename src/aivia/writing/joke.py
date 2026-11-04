"""The joke writing mode, re-exported from the shared humour module."""

from aivia.writing.humor import *  # noqa: F403
from aivia.writing.humor import __all__ as _humor_all

__all__ = [
    name
    for name in _humor_all
    if name.startswith("Joke")
    or name.startswith("build_joke")
    or name
    in {
        "coerce_joke",
        "extract_joke",
        "joke_script",
        "normalise_joke",
        "normalize_joke",
        "parse_joke",
        "validate_joke",
    }
]
