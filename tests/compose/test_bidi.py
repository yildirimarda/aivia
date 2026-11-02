"""Mixed left-to-right and right-to-left text shaping."""

from __future__ import annotations

import os
import unicodedata
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageChops, ImageFont
from pydantic import ValidationError

from aivia.compose import (
    Composition,
    FontManager,
    PillowStillRenderer,
    TextDirection,
    TextLayer,
    first_strong_direction,
    has_rtl_characters,
    layout_text,
    reorder_text,
    resolve_direction,
    shape_arabic,
    shape_text,
)

_GOLDEN_ENV = "AIVA_COMPARE_GOLDENS"
_ARABIC_GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "bidi" / "arabic.png"
_ARABIC_TEXT = "مرحبا بالعالم — Hello 2026"
_ARABIC_FONT = "Amiri"


def _composition(
    layers: list[dict[str, Any]],
    *,
    width: int = 260,
    height: int = 80,
    language: str = "",
) -> Composition:
    return Composition.model_validate(
        {
            "canvas": {"width": width, "height": height, "background": "#000000"},
            "duration_seconds": 3.0,
            "language": language,
            "layers": layers,
        }
    )


def _render(
    layers: list[dict[str, Any]],
    **kwargs: Any,
) -> Image.Image:
    return PillowStillRenderer().render_still(_composition(layers, **kwargs))


def _difference(left: Image.Image, right: Image.Image) -> bool:
    diff = ImageChops.difference(left.convert("RGB"), right.convert("RGB"))
    return diff.getbbox() is not None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello world", TextDirection.LTR),
        ("123 456", TextDirection.LTR),
        ("", TextDirection.LTR),
        ("!!!", TextDirection.LTR),
        ("\u05e9\u05dc\u05d5\u05dd", TextDirection.RTL),
        ("\u0645\u0631\u062d\u0628\u0627", TextDirection.RTL),
        ("123 \u05e9\u05dc\u05d5\u05dd", TextDirection.RTL),
    ],
)
def test_first_strong_character_sets_the_default_direction(
    text: str,
    expected: TextDirection,
) -> None:
    assert first_strong_direction(text) is expected


def test_a_language_supplies_its_direction_default() -> None:
    assert resolve_direction("Hello", language="en") is TextDirection.LTR
    assert resolve_direction("Hallo", language="de_DE") is TextDirection.LTR
    assert resolve_direction("Merhaba", language="tr") is TextDirection.LTR
    assert resolve_direction("Hello", language="ar") is TextDirection.RTL
    assert resolve_direction("Hello", language="he") is TextDirection.RTL
    assert resolve_direction("Hello", language="fa_IR") is TextDirection.RTL


def test_an_explicit_direction_wins_over_language_and_text() -> None:
    assert (
        resolve_direction("Hello", language="ar", direction="ltr") is TextDirection.LTR
    )
    assert (
        resolve_direction(
            "\u05e9\u05dc\u05d5\u05dd",
            language="en",
            direction=TextDirection.RTL,
        )
        is TextDirection.RTL
    )
    assert resolve_direction("Hello", direction="auto") is TextDirection.LTR


def test_an_unknown_direction_or_language_is_rejected() -> None:
    with pytest.raises(ValueError):
        resolve_direction("Hello", direction="sideways")
    with pytest.raises(ValueError):
        resolve_direction("Hello", language="!!!")


def test_rtl_characters_are_detected_but_latin_and_numbers_are_not() -> None:
    assert has_rtl_characters("\u05e9\u05dc\u05d5\u05dd") is True
    assert has_rtl_characters("\u0645\u0631\u062d\u0628\u0627") is True
    assert has_rtl_characters("\u0661\u0662\u0663") is True
    assert has_rtl_characters("Hello 123") is False


@pytest.mark.parametrize(
    ("text", "direction", "expected"),
    [
        ("Hello world", TextDirection.LTR, "Hello world"),
        (
            "Hello \u05e9\u05dc\u05d5\u05dd world",
            TextDirection.LTR,
            "Hello \u05dd\u05d5\u05dc\u05e9 world",
        ),
        (
            "Hello \u05e9\u05dc\u05d5\u05dd world",
            TextDirection.RTL,
            "world \u05dd\u05d5\u05dc\u05e9 Hello",
        ),
        (
            "abc 123 \u05e9\u05dc\u05d5\u05dd 456 xyz",
            TextDirection.LTR,
            "abc 123 456 \u05dd\u05d5\u05dc\u05e9 xyz",
        ),
        (
            "abc 123 \u05e9\u05dc\u05d5\u05dd 456 xyz",
            TextDirection.RTL,
            "xyz 456 \u05dd\u05d5\u05dc\u05e9 abc 123",
        ),
        (
            "2026 \u05e9\u05dc\u05d5\u05dd 42",
            TextDirection.RTL,
            "42 \u05dd\u05d5\u05dc\u05e9 2026",
        ),
    ],
)
def test_mixed_text_is_reordered_for_its_paragraph_direction(
    text: str,
    direction: TextDirection,
    expected: str,
) -> None:
    assert reorder_text(text, direction=direction) == expected
    assert shape_text(text, direction=direction) == expected


def test_each_line_is_shaped_as_its_own_paragraph() -> None:
    text = "one\ntwo \u05e9\u05dc\u05d5\u05dd"
    assert reorder_text(text, direction=TextDirection.RTL) == (
        "one\n\u05dd\u05d5\u05dc\u05e9 two"
    )
    assert reorder_text(text, language="ar") == "one\n\u05dd\u05d5\u05dc\u05e9 two"


def test_text_layer_carries_language_and_direction_defaults() -> None:
    arabic = TextLayer(text="Hello", language=" ar ")
    auto = TextLayer(text="\u05e9\u05dc\u05d5\u05dd")
    overridden = TextLayer(text="Hello", language="ar", direction=TextDirection.LTR)

    assert arabic.language == "ar"
    assert arabic.direction is None
    assert arabic.text_direction is TextDirection.RTL
    assert auto.text_direction is TextDirection.RTL
    assert overridden.direction is TextDirection.LTR
    assert overridden.text_direction is TextDirection.LTR


def test_text_layer_accepts_direction_aliases_and_rejects_unknown_values() -> None:
    assert (
        TextLayer.model_validate({"text": "Hi", "dir": "rtl"}).direction
        is TextDirection.RTL
    )
    assert (
        TextLayer.model_validate({"text": "Hi", "writing_direction": "LTR"}).direction
        is TextDirection.LTR
    )
    with pytest.raises(ValidationError):
        TextLayer.model_validate({"text": "Hi", "direction": "sideways"})
    with pytest.raises(ValidationError):
        TextLayer.model_validate({"text": "Hi", "language": 42})


def test_layout_shapes_mixed_lines_into_visual_order() -> None:
    ltr = layout_text(
        "Hello \u05e9\u05dc\u05d5\u05dd",
        max_width=500,
        max_height=200,
        font_size=20,
        direction=TextDirection.LTR,
    )
    rtl = layout_text(
        "Hello \u05e9\u05dc\u05d5\u05dd",
        max_width=500,
        max_height=200,
        font_size=20,
        direction=TextDirection.RTL,
    )

    assert ltr.lines == ("Hello \u05dd\u05d5\u05dc\u05e9",)
    assert rtl.lines == ("\u05dd\u05d5\u05dc\u05e9 Hello",)


def test_wrapping_keeps_words_whole_and_shapes_every_line() -> None:
    layout = layout_text(
        "alpha \u05d1\u05d2\u05d3 delta \u05d4\u05d5\u05d6 gamma",
        max_width=90,
        max_height=400,
        font_size=20,
        direction=TextDirection.RTL,
    )

    assert len(layout.lines) > 1
    assert all("\u05d1\u05d2\u05d3" not in line for line in layout.lines)
    assert any("\u05d3\u05d2\u05d1" in line for line in layout.lines)
    assert all("\u05d4\u05d5\u05d6" not in line for line in layout.lines)
    assert any("\u05d6\u05d5\u05d4" in line for line in layout.lines)


def test_renderer_uses_basic_layout_so_pillow_does_not_reshape_bidi_text() -> None:
    from aivia.compose.fonts import font_manager
    from aivia.compose.render import _load_font

    font = _load_font(str(font_manager().resolve(None)), 24)

    assert font.layout_engine is ImageFont.Layout.BASIC


def test_renderer_uses_a_layer_language_default() -> None:
    explicit = _render(
        [
            {
                "type": "text",
                "text": "Hello \u05e9\u05dc\u05d5\u05dd",
                "font_size": 24,
                "direction": "rtl",
            }
        ]
    )
    from_language = _render(
        [
            {
                "type": "text",
                "text": "Hello \u05e9\u05dc\u05d5\u05dd",
                "font_size": 24,
                "language": "ar",
            }
        ]
    )
    ltr = _render(
        [
            {
                "type": "text",
                "text": "Hello \u05e9\u05dc\u05d5\u05dd",
                "font_size": 24,
                "direction": "ltr",
            }
        ]
    )

    assert not _difference(explicit, from_language)
    assert _difference(explicit, ltr)


def test_renderer_uses_the_composition_language_when_the_layer_has_none() -> None:
    explicit = _render(
        [
            {
                "type": "text",
                "text": "Hello \u05e9\u05dc\u05d5\u05dd",
                "font_size": 24,
                "direction": "rtl",
            }
        ]
    )
    from_composition = _render(
        [{"type": "text", "text": "Hello \u05e9\u05dc\u05d5\u05dd", "font_size": 24}],
        language="ar",
    )
    layer_override = _render(
        [
            {
                "type": "text",
                "text": "Hello \u05e9\u05dc\u05d5\u05dd",
                "font_size": 24,
                "language": "en",
            }
        ],
        language="ar",
    )

    assert not _difference(explicit, from_composition)
    assert _difference(explicit, layer_override)


def test_renderer_accepts_direction_aliases_and_rejects_unknown_values() -> None:
    explicit = _render(
        [
            {
                "type": "text",
                "text": "Hello \u05e9\u05dc\u05d5\u05dd",
                "font_size": 24,
                "direction": "rtl",
            }
        ]
    )
    alias = _render(
        [
            {
                "type": "text",
                "text": "Hello \u05e9\u05dc\u05d5\u05dd",
                "font_size": 24,
                "dir": "rtl",
            }
        ]
    )

    assert not _difference(explicit, alias)
    with pytest.raises(Exception, match="direction"):
        _render(
            [
                {
                    "type": "text",
                    "text": "Hello",
                    "direction": "sideways",
                }
            ]
        )


def test_arabic_letters_are_joined_into_presentation_forms() -> None:
    assert shape_arabic("سلام") == "\ufeb3\ufefc\ufee1"
    assert shape_arabic("لا") == "\ufefb"


def test_arabic_shaping_leaves_other_scripts_and_empty_text_unchanged() -> None:
    assert shape_arabic("") == ""
    assert shape_arabic("Hello world") == "Hello world"
    assert shape_arabic("שלום") == "שלום"


def test_arabic_shaping_preserves_harakat_instead_of_dropping_them() -> None:
    original = "مَرْحَبًا"
    shaped = shape_arabic(original)
    marks = {character for character in original if unicodedata.combining(character)}

    assert marks
    assert marks <= set(shaped)
    assert not (set(original) - marks) & set(shaped)


def test_arabic_joining_happens_before_the_visual_reorder() -> None:
    shaped = "Hello \ufe8e\ufe92\ufea3\ufeae\ufee3 world"

    assert shape_text("Hello مرحبا world") == shaped
    assert reorder_text("Hello مرحبا world") == shaped
    assert shape_text("Hello مرحبا world", direction="rtl") == (
        "world \ufe8e\ufe92\ufea3\ufeae\ufee3 Hello"
    )


def test_layout_shapes_arabic_words_into_presentation_forms() -> None:
    layout = layout_text(
        "مرحبا بالعالم",
        max_width=500,
        max_height=200,
        font_size=40,
        font_family=_ARABIC_FONT,
        language="ar",
    )

    assert layout.lines == (
        "\ufee2\ufedf\ufe8e\ufecc\ufedf\ufe8e\ufe91 \ufe8e\ufe92\ufea3\ufeae\ufee3",
    )
    assert not any(
        "\u0600" <= character <= "\u06ff" for line in layout.lines for character in line
    )


def _missing_from_font(font: ImageFont.FreeTypeFont, text: str) -> list[str]:
    def signature(character: str) -> tuple[tuple[int, int], bytes]:
        mask = font.getmask(character)
        return mask.size, bytes(mask)

    notdef = signature("\U0010ffff")
    return [
        character
        for character in text
        if not character.isspace() and signature(character) == notdef
    ]


def test_the_bundled_rtl_font_covers_every_shaped_arabic_glyph() -> None:
    font = ImageFont.truetype(
        str(FontManager().resolve(_ARABIC_FONT)),
        40,
        layout_engine=ImageFont.Layout.BASIC,
    )
    shaped = shape_text(_ARABIC_TEXT, language="ar")

    assert _missing_from_font(font, shaped) == []
    assert _missing_from_font(font, "مرحبا") == []


def test_the_renderer_draws_joined_arabic_with_the_bundled_font() -> None:
    image = _render(
        [
            {
                "type": "text",
                "text": _ARABIC_TEXT,
                "font_size": 40,
                "color": "#f2f2f2",
                "x": 16,
                "y": 16,
                "width": 608,
                "height": 128,
                "vertical_align": "middle",
                "wrap": False,
            }
        ],
        width=640,
        height=160,
        language="ar",
    )

    ink = ImageChops.difference(
        image.convert("RGB"),
        Image.new("RGB", image.size, (0, 0, 0)),
    )
    assert ink.getbbox() is not None
    layout = layout_text(
        _ARABIC_TEXT,
        max_width=608,
        max_height=128,
        font_family=_ARABIC_FONT,
        font_size=40,
        wrap=False,
        language="ar",
    )
    assert layout.lines == (
        "Hello 2026 — \ufee2\ufedf\ufe8e\ufecc\ufedf\ufe8e\ufe91 "
        "\ufe8e\ufe92\ufea3\ufeae\ufee3",
    )


def test_arabic_text_uses_the_rtl_bundled_font_without_an_explicit_family() -> None:
    automatic = _render(
        [
            {
                "type": "text",
                "text": _ARABIC_TEXT,
                "font_size": 40,
                "color": "#f2f2f2",
                "x": 16,
                "y": 16,
                "width": 608,
                "height": 128,
                "vertical_align": "middle",
                "wrap": False,
            }
        ],
        width=640,
        height=160,
        language="ar",
    )
    explicit = _render(
        [
            {
                "type": "text",
                "text": _ARABIC_TEXT,
                "font_family": _ARABIC_FONT,
                "font_size": 40,
                "color": "#f2f2f2",
                "x": 16,
                "y": 16,
                "width": 608,
                "height": 128,
                "vertical_align": "middle",
                "wrap": False,
            }
        ],
        width=640,
        height=160,
        language="ar",
    )

    assert not _difference(automatic, explicit)


@pytest.mark.golden
def test_the_arabic_render_matches_the_golden_image() -> None:
    if os.environ.get(_GOLDEN_ENV) != "1":
        pytest.skip(f"set {_GOLDEN_ENV}=1 to compare golden renders")

    rendered = _render(
        [
            {
                "type": "text",
                "text": _ARABIC_TEXT,
                "font_size": 40,
                "color": "#f2f2f2",
                "x": 16,
                "y": 16,
                "width": 608,
                "height": 128,
                "vertical_align": "middle",
                "wrap": False,
            }
        ],
        width=640,
        height=160,
        language="ar",
    )
    expected = Image.open(_ARABIC_GOLDEN).convert("RGBA")

    assert rendered.size == expected.size
    assert ImageChops.difference(rendered, expected).getbbox() is None
