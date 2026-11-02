"""Mixed left-to-right and right-to-left text shaping."""

from __future__ import annotations

from typing import Any

import pytest
from PIL import Image, ImageChops, ImageFont
from pydantic import ValidationError

from aivia.compose import (
    Composition,
    PillowStillRenderer,
    TextDirection,
    TextLayer,
    first_strong_direction,
    has_rtl_characters,
    layout_text,
    reorder_text,
    resolve_direction,
    shape_text,
)


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
