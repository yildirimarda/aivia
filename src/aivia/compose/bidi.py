"""Resolve and apply the writing direction of composition text.

A text run is stored in logical order, the order a reader types it. A renderer
must draw it in visual order, which differs when right-to-left runs are mixed
with left-to-right ones. The Unicode bidirectional algorithm supplies that
visual order; this module keeps the call sites simple and adds the project's
per-language direction defaults on top.

The direction of a layer is chosen in this order: an explicit direction, then
the layer language, then the composition language, then the first strong
character in the text. A language with no configured default falls back to the
Unicode default, left to right.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable

from bidi.algorithm import get_display  # type: ignore[import-untyped]

from aivia.core.languages import TextDirection, language_direction

__all__ = [
    "TextDirection",
    "first_strong_direction",
    "has_rtl_characters",
    "reorder_text",
    "resolve_direction",
    "shape_text",
]

_RTL_TYPES = frozenset({"R", "AL"})
_RTL_STRONG_OR_NUMBER = frozenset({"R", "AL", "AN"})


def _coerce_direction(value: TextDirection | str | None) -> TextDirection | None:
    if value is None:
        return None
    if isinstance(value, TextDirection):
        return value
    if isinstance(value, str):
        key = value.strip().casefold()
        if key in ("", "auto"):
            return None
        try:
            return TextDirection(key)
        except ValueError as exc:
            raise ValueError(f"unsupported text direction {value!r}") from exc
    raise ValueError(f"unsupported text direction {value!r}")


def _bidi_types(text: str) -> Iterable[str]:
    return (unicodedata.bidirectional(character) for character in text)


def has_rtl_characters(text: str) -> bool:
    """Whether *text* contains a strong right-to-left or Arabic number run."""
    return any(kind in _RTL_STRONG_OR_NUMBER for kind in _bidi_types(text))


def first_strong_direction(text: str) -> TextDirection:
    """Return the direction of the first strong character, defaulting to LTR."""
    for kind in _bidi_types(text):
        if kind in _RTL_TYPES:
            return TextDirection.RTL
        if kind == "L":
            return TextDirection.LTR
    return TextDirection.LTR


def resolve_direction(
    text: str,
    *,
    language: str = "",
    direction: TextDirection | str | None = None,
) -> TextDirection:
    """Resolve the paragraph direction for *text*.

    An explicit *direction* wins. Otherwise a non-empty *language* supplies its
    configured default, and only then does the text's first strong character
    decide. The caller passes the composition language when a layer has none.
    """
    explicit = _coerce_direction(direction)
    if explicit is not None:
        return explicit
    if language.strip():
        try:
            return language_direction(language)
        except ValueError as exc:
            raise ValueError(f"unsupported language {language!r}") from exc
    return first_strong_direction(text)


def reorder_text(
    text: str,
    *,
    direction: TextDirection | str | None = None,
    language: str = "",
) -> str:
    """Return *text* in visual order, one paragraph at a time.

    Newlines are paragraph separators and are preserved. A line that stays
    left to right is returned unchanged so existing Latin output is byte for
    byte identical; mixed or right-to-left lines are reordered by the Unicode
    bidirectional algorithm.
    """
    if not text:
        return text
    lines: list[str] = []
    for line in text.split("\n"):
        resolved = resolve_direction(line, language=language, direction=direction)
        if resolved is TextDirection.LTR and not has_rtl_characters(line):
            lines.append(line)
            continue
        base = "R" if resolved is TextDirection.RTL else "L"
        lines.append(get_display(line, base_dir=base))
    return "\n".join(lines)


shape_text = reorder_text
"""Alias for :func:`reorder_text`, named after the shaping step it feeds."""
