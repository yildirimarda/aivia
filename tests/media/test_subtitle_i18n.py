"""Per-language subtitle line breaking and bundled font glyph coverage."""

from __future__ import annotations

from pathlib import Path

import pytest

from aivia.core.errors import ValidationError
from aivia.media.subtitles import (
    DEFAULT_LINE_BREAK_LANGUAGE,
    SubtitleCue,
    break_lines,
    bundled_font_path,
    font_covers_text,
    hyphenate_word,
    line_break_opportunities,
    missing_glyphs,
    split_syllables,
    wrap_text,
)

TURKISH_ALPHABET = "abcçdefgğhıijklmnoöprsştuüvyzABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZâÂîÎûÛ"  # noqa: RUF001
GERMAN_ALPHABET = "abcdefghijklmnopqrstuvwxyzäöüßABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜẞ"


def test_bundled_lato_covers_turkish_and_german_characters() -> None:
    text = TURKISH_ALPHABET + GERMAN_ALPHABET

    for style in ("regular", "bold", "italic", "bolditalic"):
        font = bundled_font_path("Lato", style)
        assert missing_glyphs(font, text) == [], style
        assert font_covers_text(font, text), style


def test_glyph_coverage_reports_characters_the_font_cannot_render() -> None:
    font = bundled_font_path()

    assert missing_glyphs(font, "Türkçe Deutsch") == []
    assert missing_glyphs(font, "Türkçe \u2603") == ["\u2603"]


def test_english_breaks_at_spaces_and_explicit_hyphens_only() -> None:
    assert DEFAULT_LINE_BREAK_LANGUAGE == "en"
    assert break_lines("Hello brave new world", max_chars=12) == [
        "Hello brave",
        "new world",
    ]
    assert break_lines("E-Mail clients work", max_chars=10, language="en") == [
        "E-Mail",
        "clients",
        "work",
    ]
    assert line_break_opportunities("E-Mail", "en") == [2]
    assert line_break_opportunities("well-known", "en") == [5]
    assert split_syllables("Türkiye", "en") == ["Türkiye"]


def test_turkish_syllables_are_used_as_line_break_points() -> None:
    assert split_syllables("Türkiye", "tr") == ["Tür", "ki", "ye"]
    assert split_syllables("Almanca", "tr") == ["Al", "man", "ca"]
    assert split_syllables("Türkçe", "tr") == ["Türk", "çe"]
    assert split_syllables("İstanbul", "tr") == ["İs", "tan", "bul"]
    assert split_syllables("kitaplar", "tr") == ["ki", "tap", "lar"]
    assert split_syllables("öğretmen", "tr") == ["öğ", "ret", "men"]
    assert split_syllables("saat", "tr") == ["sa", "at"]

    assert hyphenate_word("Türkiye", max_chars=4, language="tr") == ["Tür-", "kiye"]
    assert break_lines("Almanca güzel", max_chars=4, language="tr") == [
        "Al-",
        "man-",
        "ca",
        "gü-",
        "zel",
    ]


def test_turkish_apostrophes_suppress_syllable_breaks() -> None:
    assert split_syllables("Türkiye'de", "tr") == ["Türkiye'de"]
    assert line_break_opportunities("Türkiye'de", "tr") == []
    assert break_lines("Türkiye'de güzel bir gün", max_chars=10, language="tr") == [
        "Türkiye'de",
        "güzel bir",
        "gün",
    ]


def test_german_compounds_break_at_hyphens_and_soft_hyphens() -> None:
    assert split_syllables("Türkiye", "de") == ["Türkiye"]
    assert line_break_opportunities("E-Mail-Adresse", "de") == [2, 7]
    assert hyphenate_word("E-Mail-Adresse", max_chars=8, language="de") == [
        "E-Mail-",
        "Adresse",
    ]
    assert break_lines("Silben\u00adtrennung", max_chars=8, language="de") == [
        "Silben-",
        "trennung",
    ]
    assert wrap_text("Silben\u00adtrennung", max_chars=20, language="de") == (
        "Silbentrennung"
    )
    assert line_break_opportunities("E\u2011Mail", "de") == []


def test_unknown_languages_fall_back_to_space_and_hyphen_breaks() -> None:
    assert break_lines("Guten Tag", max_chars=5, language="xx") == ["Guten", "Tag"]
    assert break_lines("a-b", max_chars=2, language="xx") == ["a-", "b"]


def test_wrapped_lines_never_exceed_the_requested_length() -> None:
    for language in ("en", "de", "tr"):
        lines = break_lines("Türkiye güzel bir gün", max_chars=6, language=language)
        assert lines
        assert all(len(line) <= 6 for line in lines), language


def test_subtitle_cues_expose_language_aware_wrapping() -> None:
    cue = SubtitleCue(
        start_seconds=0.0,
        end_seconds=2.0,
        text="Türkiye güzel bir gün",
    )

    assert cue.wrapped_lines(max_chars=10, language="tr") == [
        "Türkiye",
        "güzel bir",
        "gün",
    ]
    assert cue.wrap(max_chars=10, language="tr") == "Türkiye\ngüzel bir\ngün"
    assert cue.lines == ["Türkiye güzel bir gün"]


def test_invalid_line_break_and_font_inputs_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        break_lines("text", max_chars=0)
    with pytest.raises(ValidationError):
        hyphenate_word("word", max_chars=-1)
    with pytest.raises(ValidationError):
        bundled_font_path(style="missing")
    with pytest.raises(ValidationError):
        missing_glyphs(tmp_path / "missing.ttf", "text")
    with pytest.raises(TypeError):
        break_lines("text", max_chars=4, language=1)  # type: ignore[arg-type]
