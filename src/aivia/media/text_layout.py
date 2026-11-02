"""Language-aware line breaking and font glyph coverage for subtitles.

Subtitle text is wrapped before it reaches a player or a renderer. The break
rules themselves live in :mod:`aivia.core.text_breaks` so composition rendering
can share them without importing the media package. This module re-exports
those helpers and adds glyph coverage checks against a real font file.

Glyph coverage is checked against a real font file. A character is considered
missing when its rendered mask is identical to the font's ``.notdef`` glyph, so
a bundled font can be validated before text is burned into a video.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

from PIL import ImageFont

from aivia.core.errors import ValidationError
from aivia.core.paths import package_fonts_dir
from aivia.core.text_breaks import (
    DEFAULT_LINE_BREAK_LANGUAGE,
    LineBreakRules,
    break_lines,
    hyphenate_word,
    line_break_opportunities,
    split_syllables,
    wrap_text,
)

__all__ = [
    "DEFAULT_LINE_BREAK_LANGUAGE",
    "LineBreakRules",
    "break_lines",
    "bundled_font_path",
    "font_covers_text",
    "hyphenate_word",
    "line_break_opportunities",
    "missing_glyphs",
    "split_syllables",
    "wrap_subtitle_text",
    "wrap_text",
]

_MISSING_GLYPH = "\U0010ffff"

_FONT_STYLES = {
    "regular": "Regular",
    "bold": "Bold",
    "italic": "Italic",
    "bolditalic": "BoldItalic",
    "bold_italic": "BoldItalic",
    "bold-italic": "BoldItalic",
    "bold italic": "BoldItalic",
}


wrap_subtitle_text = wrap_text
"""Alias for :func:`wrap_text` used by subtitle callers."""


def bundled_font_path(family: str = "Lato", style: str = "regular") -> Path:
    """Return the path of a bundled font family member.

    The family names a directory under the packaged fonts and the style names
    one of ``regular``, ``bold``, ``italic`` or ``bolditalic``.
    """
    if not isinstance(family, str) or not family.strip():
        raise ValidationError(
            "font family must be non-empty", details={"field": "family"}
        )
    if not isinstance(style, str) or not style.strip():
        raise ValidationError(
            "font style must be non-empty", details={"field": "style"}
        )
    style_key = style.strip().lower()
    suffix = _FONT_STYLES.get(style_key) or _FONT_STYLES.get(
        style_key.replace(" ", "_")
    )
    if suffix is None:
        allowed = ", ".join(sorted({"regular", "bold", "italic", "bolditalic"}))
        raise ValidationError(
            f"unsupported bundled font style {style!r}; expected one of {allowed}",
            details={"field": "style"},
        )
    root = package_fonts_dir()
    family_dir = next(
        (
            candidate
            for candidate in root.iterdir()
            if candidate.is_dir() and candidate.name.lower() == family.strip().lower()
        ),
        None,
    )
    expected = f"{family.strip()}-{suffix}".lower()
    if family_dir is not None:
        for candidate in family_dir.glob("*.ttf"):
            if candidate.stem.lower() == expected:
                return candidate
    raise ValidationError(
        f"bundled font {family!r} ({style}) is not available",
        details={"field": "font"},
    )


def _load_font(font: str | Path, size: int) -> ImageFont.FreeTypeFont:
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValidationError(
            "font size must be a positive integer",
            details={"field": "size", "value": size},
        )
    path = Path(font)
    if not path.is_file():
        raise ValidationError(
            f"font file does not exist: {path}",
            details={"field": "font"},
        )
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError as exc:
        raise ValidationError(
            f"cannot load font file: {path}",
            details={"field": "font"},
        ) from exc


def _glyph_signature(
    font: ImageFont.FreeTypeFont,
    character: str,
) -> tuple[tuple[int, int], bytes]:
    mask = font.getmask(character)
    return mask.size, bytes(mask)


def missing_glyphs(
    font: str | Path,
    text: str,
    *,
    size: int = 32,
) -> list[str]:
    """Return the distinct characters in *text* the font cannot render.

    Whitespace, control and format characters are ignored. A character is
    missing when its mask is identical to the font's ``.notdef`` glyph, which
    is what Pillow draws for an unmapped code point.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    loaded = _load_font(font, size)
    notdef = _glyph_signature(loaded, _MISSING_GLYPH)
    missing: list[str] = []
    seen: set[str] = set()
    for character in text:
        if character in seen:
            continue
        seen.add(character)
        if character.isspace() or unicodedata.category(character) in {"Cc", "Cf"}:
            continue
        if _glyph_signature(loaded, character) == notdef:
            missing.append(character)
    return missing


def font_covers_text(
    font: str | Path,
    text: str,
    *,
    size: int = 32,
) -> bool:
    """Return whether *font* has a glyph for every character in *text*."""
    return not missing_glyphs(font, text, size=size)
