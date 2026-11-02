"""Language direction primitives shared by rendering and language helpers.

The full language resolver lives in :mod:`aivia.llm.languages`, which also
detects languages and reads their configured settings. Rendering must stay
importable without the LLM package, so the small direction table lives here and
the resolver re-exports it instead of owning a second copy.
"""

from __future__ import annotations

import re
from enum import StrEnum

__all__ = [
    "RTL_CODES",
    "TextDirection",
    "is_rtl_language",
    "language_direction",
]


class TextDirection(StrEnum):
    """The writing direction of a language."""

    LTR = "ltr"
    RTL = "rtl"


RTL_CODES = frozenset(
    {
        "ar",
        "arc",
        "ckb",
        "dv",
        "fa",
        "ha",
        "he",
        "khw",
        "ks",
        "ku",
        "ps",
        "sd",
        "syr",
        "ug",
        "ur",
        "yi",
    }
)
"""ISO 639 codes written right to left by default."""

_CODE = re.compile(r"[a-z]{2,3}")


def _primary_code(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("language code must be a string")
    text = value.strip()
    if not text:
        raise ValueError("language code must not be empty")
    text = text.split("@", 1)[0].split("(", 1)[0].strip()
    primary = re.split(r"[-\s]+", text.replace("_", "-"), maxsplit=1)[0]
    primary = primary.split(".", 1)[0].casefold()
    if not _CODE.fullmatch(primary):
        raise ValueError(f"not a language code: {value!r}")
    return primary


def language_direction(value: str) -> TextDirection:
    """Return the default writing direction for a language code."""
    return TextDirection.RTL if _primary_code(value) in RTL_CODES else TextDirection.LTR


def is_rtl_language(value: str) -> bool:
    """Whether a language is written right to left by default."""
    return language_direction(value) is TextDirection.RTL
