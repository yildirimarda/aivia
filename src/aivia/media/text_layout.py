"""Language-aware line breaking and font glyph coverage for subtitles.

Subtitle text is wrapped before it reaches a player or a renderer. Breaking a
line inside a word is only safe when the language defines where a word may be
split:

* English wraps at spaces and after explicit hyphens and soft hyphens;
* German wraps the same way, which keeps compound words such as ``E-Mail``
  and source supplied soft hyphens intact;
* Turkish additionally hyphenates at syllable boundaries, using the Turkish
  vowel set and the rule that the last consonant before the next vowel starts
  the next syllable (``Tür-ki-ye``, ``Al-man-ca``).

The rules are deliberately conservative. No dictionary is consulted, so a word
without an explicit or rule-derived break point is never split except as a last
resort when it cannot fit on a line at all. A soft hyphen becomes a visible
hyphen only when the line actually breaks there.

Glyph coverage is checked against a real font file. A character is considered
missing when its rendered mask is identical to the font's ``.notdef`` glyph, so
a bundled font can be validated before text is burned into a video.
"""

from __future__ import annotations

import unicodedata
from itertools import pairwise
from pathlib import Path

from PIL import ImageFont

from aivia.core.errors import ValidationError
from aivia.core.paths import package_fonts_dir

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

DEFAULT_LINE_BREAK_LANGUAGE = "en"
"""The language used when a caller does not name one."""

_SOFT_HYPHEN = "\u00ad"
_BREAKING_HYPHENS = frozenset({"-", "\u2010"})
_NON_BREAKING_HYPHEN = "\u2011"
_APOSTROPHES = frozenset({"'", "\u2019", "\u02bc"})
_MISSING_GLYPH = "\U0010ffff"

_TURKISH_VOWELS = frozenset("aeıioöuüAEIİOÖUÜ")  # noqa: RUF001
_ENGLISH_VOWELS = frozenset("aeiouyAEIOUY")

_FONT_STYLES = {
    "regular": "Regular",
    "bold": "Bold",
    "italic": "Italic",
    "bolditalic": "BoldItalic",
    "bold_italic": "BoldItalic",
    "bold-italic": "BoldItalic",
    "bold italic": "BoldItalic",
}


class LineBreakRules:
    """The break opportunities a language permits inside a word."""

    __slots__ = ("code", "syllables", "vowels")

    def __init__(
        self,
        code: str,
        *,
        vowels: frozenset[str] = frozenset(),
        syllables: bool = False,
    ) -> None:
        self.code = code
        self.vowels = vowels
        self.syllables = syllables

    def boundaries(self, word: str) -> list[int]:
        """Return indices where a new syllable starts in *word*."""
        if not self.syllables or any(char in _APOSTROPHES for char in word):
            return []
        vowels = [index for index, char in enumerate(word) if char in self.vowels]
        boundaries: list[int] = []
        for left, right in pairwise(vowels):
            cluster = right - left - 1
            boundary = left + 1 if cluster == 0 else right - 1
            if 0 < boundary < len(word):
                boundaries.append(boundary)
        return sorted(set(boundaries))


_RULES: dict[str, LineBreakRules] = {
    "en": LineBreakRules("en", vowels=_ENGLISH_VOWELS),
    "de": LineBreakRules("de"),
    "tr": LineBreakRules("tr", vowels=_TURKISH_VOWELS, syllables=True),
}


def _language_key(language: str) -> str:
    if not isinstance(language, str):
        raise TypeError("language must be a string")
    key = language.strip().replace("_", "-").lower().split("-", 1)[0]
    return key or DEFAULT_LINE_BREAK_LANGUAGE


def _rules_for(language: str) -> LineBreakRules:
    key = _language_key(language)
    return _RULES.get(key, LineBreakRules(key))


def _validate_max_chars(max_chars: int) -> int:
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars <= 0:
        raise ValidationError(
            "line length must be a positive integer",
            details={"field": "max_chars", "value": max_chars},
        )
    return max_chars


def line_break_opportunities(word: str, language: str = "") -> list[int]:
    """Return the indices in *word* where a line may break before the next char.

    An explicit hyphen stays at the end of the line. A soft hyphen is a break
    opportunity only and is removed when the word fits without breaking. The
    result never splits a combining mark from its base character.
    """
    if not isinstance(word, str):
        raise TypeError("word must be a string")
    if not word:
        return []
    rules = _rules_for(language)
    candidates: set[int] = set()
    for index, char in enumerate(word[:-1]):
        if char in _BREAKING_HYPHENS or char == _SOFT_HYPHEN:
            candidates.add(index + 1)
        elif char == _NON_BREAKING_HYPHEN:
            continue
    candidates.update(rules.boundaries(word))
    return sorted(
        index
        for index in candidates
        if 0 < index < len(word)
        and word[index - 1] not in _APOSTROPHES
        and word[index] not in _APOSTROPHES
        and not unicodedata.combining(word[index])
    )


def split_syllables(word: str, language: str = "tr") -> list[str]:
    """Split *word* into syllables for languages with syllable line breaking.

    Languages without syllable rules return the word unchanged. Apostrophes
    suppress syllable breaks so a proper noun and its suffix stay together.
    """
    if not isinstance(word, str):
        raise TypeError("word must be a string")
    rules = _rules_for(language)
    boundaries = rules.boundaries(word)
    if not boundaries:
        return [word]
    pieces: list[str] = []
    start = 0
    for boundary in boundaries:
        pieces.append(word[start:boundary])
        start = boundary
    pieces.append(word[start:])
    return pieces


def hyphenate_word(
    word: str,
    *,
    max_chars: int,
    language: str = "",
) -> list[str]:
    """Split *word* into hyphenated fragments no longer than *max_chars*.

    A fragment ends with an explicit hyphen, a soft hyphen turned visible, or
    an inserted hyphen at a syllable boundary. If the language offers no break
    point before the limit, the word is split at the limit as a last resort.
    """
    limit = _validate_max_chars(max_chars)
    if not isinstance(word, str):
        raise TypeError("word must be a string")
    if not word:
        return [""]
    visible = word.replace(_SOFT_HYPHEN, "")
    if len(visible) <= limit:
        return [visible]

    rules = _rules_for(language)
    pieces: list[str] = []
    remaining = word
    while len(remaining.replace(_SOFT_HYPHEN, "")) > limit:
        fits = [
            boundary
            for boundary in line_break_opportunities(remaining, rules.code)
            if boundary <= limit
        ]
        visible_length = len(remaining.replace(_SOFT_HYPHEN, ""))
        if fits:
            cut = max(fits)
            if visible_length - cut == 1 and len(fits) > 1:
                cut = fits[-2]
            head = remaining[:cut].replace(_SOFT_HYPHEN, "-")
            if not head.endswith(("-", "\u2010")):
                head += "-"
        else:
            cut = limit
            while cut < len(remaining) and unicodedata.combining(remaining[cut]):
                cut += 1
            if visible_length - cut == 1 and cut > 1:
                cut -= 1
            if cut < len(remaining) and remaining[cut] == _SOFT_HYPHEN:
                remaining = remaining[:cut] + remaining[cut + 1 :]
                head = remaining[:cut]
            else:
                head = remaining[:cut] + "-"
        pieces.append(head)
        remaining = remaining[cut:]
    tail = remaining.replace(_SOFT_HYPHEN, "")
    if tail:
        pieces.append(tail)
    return pieces


def break_lines(
    text: str,
    *,
    max_chars: int,
    language: str = "",
) -> list[str]:
    """Wrap *text* into lines no longer than *max_chars* for *language*.

    Existing newlines are hard line breaks. Words longer than the limit are
    hyphenated using the language rules, or split at the limit when the
    language permits no earlier break.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    limit = _validate_max_chars(max_chars)
    rules = _rules_for(language)
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines: list[str] = []

    for paragraph in normalized.split("\n"):
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = ""
        for word in words:
            visible = word.replace(_SOFT_HYPHEN, "")
            fragments = (
                hyphenate_word(word, max_chars=limit, language=rules.code)
                if len(visible) > limit
                else [visible]
            )
            for index, fragment in enumerate(fragments):
                if not current:
                    current = fragment
                elif len(current) + 1 + len(fragment) <= limit:
                    current = f"{current} {fragment}"
                else:
                    lines.append(current)
                    current = fragment
                if index < len(fragments) - 1:
                    lines.append(current)
                    current = ""
        if current:
            lines.append(current)
    return lines


def wrap_text(
    text: str,
    *,
    max_chars: int,
    language: str = "",
) -> str:
    """Return *text* wrapped into lines joined with newline characters."""
    return "\n".join(break_lines(text, max_chars=max_chars, language=language))


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
