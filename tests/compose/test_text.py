"""Text layers: typography, wrapping, shrink to fit and ellipsis."""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageChops
from pydantic import ValidationError

from aivia.compose import (
    Canvas,
    Composition,
    Layer,
    LayerType,
    PillowStillRenderer,
    RenderError,
    TextAlign,
    TextLayer,
    TextOverflow,
    TextStyle,
    TextWeight,
    VerticalAlign,
    layout_text,
)
from aivia.core.config import RenderConfig
from aivia.core.paths import package_fonts_dir

_FONT_SUFFIXES = ("Regular", "Bold", "Italic", "BoldItalic")


def _composition(
    layers: list[dict[str, Any]],
    *,
    width: int = 200,
    height: int = 60,
    background: str = "#000000",
    duration_seconds: float = 3.0,
    validate: bool = True,
) -> Composition:
    values = {
        "canvas": {
            "width": width,
            "height": height,
            "background": background,
        },
        "duration_seconds": duration_seconds,
        "layers": layers,
    }
    if validate:
        return Composition.model_validate(values)
    return Composition.model_construct(
        canvas=Canvas.model_validate(values["canvas"]),
        duration_seconds=duration_seconds,
        layers=[Layer.model_validate(layer) for layer in layers],
    )


def _render(
    layers: list[dict[str, Any]],
    *,
    validate: bool = True,
    **kwargs: Any,
) -> Image.Image:
    return PillowStillRenderer().render_still(
        _composition(layers, validate=validate, **kwargs)
    )


def _bbox(image: Image.Image) -> tuple[int, int, int, int] | None:
    rgb = image.convert("RGB")
    background = Image.new("RGB", rgb.size, (0, 0, 0))
    return ImageChops.difference(rgb, background).getbbox()


def _has_color(image: Image.Image, color: tuple[int, int, int]) -> bool:
    rgb = image.convert("RGB")
    return color in {pixel for _, pixel in rgb.getcolors(rgb.width * rgb.height) or []}


def _fragment_width(*fragments: str, font_size: int = 24) -> int:
    from aivia.compose.fonts import FontManager
    from aivia.compose.render import _load_font

    font = _load_font(str(FontManager().resolve(None)), font_size)
    return max(math.ceil(font.getlength(fragment)) for fragment in fragments) + 1


def _copy_lato_family(root: Path, name: str = "Custom") -> Path:
    source = package_fonts_dir() / "lato"
    directory = root / name
    directory.mkdir(parents=True)
    for suffix in _FONT_SUFFIXES:
        shutil.copyfile(
            source / f"Lato-{suffix}.ttf",
            directory / f"{name}-{suffix}.ttf",
        )
    return directory


def test_text_layer_carries_full_typography_control() -> None:
    layer = TextLayer.model_validate(
        {
            "content": "Hello world",
            "font": " Lato ",
            "size": 32,
            "text_color": " #ffffff ",
            "alignment": "centre",
            "valign": "bottom",
            "weight": "bold",
            "style": "italic",
            "line_spacing": 1.5,
            "tracking": 2,
            "word_wrap": False,
            "lines": 3,
            "shrink": True,
            "min_size": 10,
            "text_overflow": "truncate",
            "truncation": "...",
            "outline": " #000000 ",
            "outline_width": 2,
        }
    )

    assert layer.type is LayerType.TEXT
    assert layer.text == "Hello world"
    assert layer.font_family == "Lato"
    assert layer.font_size == 32.0
    assert layer.color == "#ffffff"
    assert layer.align is TextAlign.CENTER
    assert layer.vertical_align is VerticalAlign.BOTTOM
    assert layer.font_weight is TextWeight.BOLD
    assert layer.font_style is TextStyle.ITALIC
    assert layer.line_height == 1.5
    assert layer.letter_spacing == 2.0
    assert layer.wrap is False
    assert layer.max_lines == 3
    assert layer.shrink_to_fit is True
    assert layer.min_font_size == 10.0
    assert layer.overflow is TextOverflow.ELLIPSIS
    assert layer.ellipsis == "..."
    assert layer.stroke == "#000000"
    assert layer.stroke_width == 2.0


def test_text_layer_defaults_and_alias_values() -> None:
    layer = TextLayer(text="Hi")

    assert layer.font_family == ""
    assert layer.font_size == 48.0
    assert layer.color == "#ffffff"
    assert layer.align is TextAlign.LEFT
    assert layer.vertical_align is VerticalAlign.TOP
    assert layer.font_weight is TextWeight.REGULAR
    assert layer.font_style is TextStyle.NORMAL
    assert layer.line_height == 1.2
    assert layer.letter_spacing == 0.0
    assert layer.wrap is True
    assert layer.max_lines is None
    assert layer.shrink_to_fit is False
    assert layer.min_font_size == 8.0
    assert layer.overflow is TextOverflow.ELLIPSIS
    assert layer.ellipsis == "\u2026"
    assert layer.stroke is None
    assert layer.stroke_width == 0.0

    assert (
        TextLayer.model_validate({"text": "Hi", "font_weight": 700}).font_weight
        is TextWeight.BOLD
    )
    assert (
        TextLayer.model_validate({"text": "Hi", "font_weight": 100}).font_weight
        is TextWeight.THIN
    )
    assert (
        TextLayer.model_validate({"text": "Hi", "font_weight": "normal"}).font_weight
        is TextWeight.REGULAR
    )
    assert (
        TextLayer.model_validate({"text": "Hi", "align": "end"}).align
        is TextAlign.RIGHT
    )
    assert (
        TextLayer.model_validate(
            {"text": "Hi", "vertical_align": "center"}
        ).vertical_align
        is VerticalAlign.MIDDLE
    )
    assert (
        TextLayer.model_validate({"text": "Hi", "style": "oblique"}).font_style
        is TextStyle.ITALIC
    )
    assert (
        TextLayer.model_validate({"text": "Hi", "overflow": "truncate"}).overflow
        is TextOverflow.ELLIPSIS
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"text": "Hi", "font_size": 0},
        {"text": "Hi", "font_size": -1},
        {"text": "Hi", "font_size": math.inf},
        {"text": "Hi", "line_height": 0},
        {"text": "Hi", "line_height": math.nan},
        {"text": "Hi", "letter_spacing": math.inf},
        {"text": "Hi", "max_lines": 0},
        {"text": "Hi", "max_lines": 1.5},
        {"text": "Hi", "shrink": True, "font_size": 10, "min_size": 20},
        {"text": "Hi", "overflow": "scroll"},
        {"text": "Hi", "ellipsis": ""},
        {"text": "Hi", "wrap": "yes"},
        {"text": "Hi", "stroke_width": -1},
        {"text": "Hi", "font_weight": "ultra"},
        {"text": "Hi", "font_weight": 0},
        {"text": "Hi", "font_weight": 1001},
        {"text": "Hi", "font_style": "slanted"},
        {"text": "Hi", "vertical_align": "sideways"},
        {"text": "Hi", "stroke": 42},
        {"text": "Hi", "align": "justify"},
    ],
)
def test_text_layer_rejects_invalid_typography(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        TextLayer.model_validate(payload)


def test_layout_wraps_words_and_long_tokens_within_the_box() -> None:
    wrapped = layout_text(
        "one two three four five six",
        max_width=80,
        max_height=400,
        font_size=20,
    )
    long_word = layout_text(
        "supercalifragilisticexpialidocious",
        max_width=40,
        max_height=400,
        font_size=20,
    )

    assert len(wrapped.lines) > 1
    assert wrapped.width <= 80
    assert wrapped.truncated is False
    assert len(long_word.lines) > 1
    assert long_word.width <= 40


@pytest.mark.parametrize(
    ("text", "language", "fragments", "expected"),
    [
        ("E-Mail-Adresse", "de", ("E-Mail-", "Adresse"), ("E-Mail-", "Adresse")),
        ("Türkiye", "tr", ("Tür-", "kiye"), ("Tür-", "kiye")),
        (
            "Silben\u00adtrennung",
            "de",
            ("Silben-", "trennung"),
            ("Silben-", "trennung"),
        ),
    ],
)
def test_long_words_break_at_language_rule_points(
    text: str,
    language: str,
    fragments: tuple[str, ...],
    expected: tuple[str, ...],
) -> None:
    width = _fragment_width(*fragments)
    layout = layout_text(
        text,
        max_width=width,
        max_height=400,
        font_size=24,
        language=language,
    )

    assert layout.lines == expected
    assert layout.width <= width


def test_soft_hyphens_are_invisible_when_the_word_fits() -> None:
    layout = layout_text(
        "Silben\u00adtrennung",
        max_width=500,
        max_height=400,
        font_size=24,
        language="de",
    )

    assert layout.lines == ("Silbentrennung",)


def test_non_breaking_hyphens_are_not_break_points() -> None:
    layout = layout_text(
        "E\u2011Mail",
        max_width=_fragment_width("E\u2011Mail"),
        max_height=400,
        font_size=24,
        language="de",
    )

    assert layout.lines == ("E\u2011Mail",)


def test_newlines_are_hard_breaks_and_wrap_can_be_disabled() -> None:
    layout = layout_text(
        "one\ntwo three\nfour",
        max_width=500,
        max_height=500,
        font_size=20,
        wrap=False,
    )
    clipped = layout_text(
        "one two three",
        max_width=30,
        max_height=200,
        font_size=20,
        wrap=False,
        overflow=TextOverflow.CLIP,
    )

    assert layout.lines == ("one", "two three", "four")
    assert clipped.lines == ("one two three",)
    assert clipped.truncated is True


def test_shrink_to_fit_reduces_the_font_until_the_text_fits() -> None:
    text = "one two three four five six seven eight"
    fixed = layout_text(
        text,
        max_width=100,
        max_height=60,
        font_size=40,
        min_font_size=8,
    )
    shrunk = layout_text(
        text,
        max_width=100,
        max_height=60,
        font_size=40,
        min_font_size=8,
        shrink_to_fit=True,
    )

    assert shrunk.font_size < fixed.font_size
    assert len(shrunk.lines) > len(fixed.lines)
    assert shrunk.width <= 100
    assert shrunk.height <= 60
    assert shrunk.truncated is False


def test_ellipsis_marks_lines_that_do_not_fit() -> None:
    ellipsized = layout_text(
        "one two three four five six",
        max_width=120,
        max_height=100,
        font_size=20,
        max_lines=2,
    )
    clipped = layout_text(
        "one two three four five six",
        max_width=120,
        max_height=100,
        font_size=20,
        max_lines=2,
        overflow=TextOverflow.CLIP,
    )
    custom = layout_text(
        "one two three four five six",
        max_width=120,
        max_height=100,
        font_size=20,
        max_lines=2,
        ellipsis="...",
    )
    fitting = layout_text(
        "one two",
        max_width=500,
        max_height=100,
        font_size=20,
        max_lines=2,
    )

    assert len(ellipsized.lines) == 2
    assert ellipsized.lines[-1].endswith("\u2026")
    assert ellipsized.truncated is True
    assert clipped.lines[-1].endswith("\u2026") is False
    assert custom.lines[-1].endswith("...")
    assert fitting.truncated is False
    assert not fitting.lines[-1].endswith("\u2026")


def test_letter_spacing_line_height_and_weight_change_the_layout() -> None:
    plain = layout_text("Hello", max_width=500, max_height=100, font_size=32)
    spaced = layout_text(
        "Hello",
        max_width=500,
        max_height=100,
        font_size=32,
        letter_spacing=5,
    )
    compact = layout_text(
        "Hello\nWorld",
        max_width=500,
        max_height=200,
        font_size=32,
        line_height=1.0,
    )
    tall = layout_text(
        "Hello\nWorld",
        max_width=500,
        max_height=200,
        font_size=32,
        line_height=2.0,
    )
    bold = layout_text(
        "Hello",
        max_width=500,
        max_height=100,
        font_size=32,
        font_weight=TextWeight.BOLD,
    )

    assert spaced.width > plain.width
    assert tall.height > compact.height
    assert bold.width >= plain.width


def test_text_layers_are_supported_by_the_still_renderer() -> None:
    assert LayerType.TEXT in PillowStillRenderer.supported_layer_types


def test_a_text_layer_draws_its_colour_inside_its_box() -> None:
    image = _render(
        [
            {
                "type": "text",
                "text": "Hello",
                "font_size": 32,
                "color": "#ff0000",
                "x": 10,
                "y": 10,
                "width": 120,
                "height": 50,
            }
        ],
        width=160,
        height=80,
    )

    box = _bbox(image)
    assert box is not None
    left, top, right, bottom = box
    assert 10 <= left < right <= 130
    assert 10 <= top < bottom <= 60
    assert _has_color(image, (255, 0, 0))


@pytest.mark.parametrize("align", ["left", "center", "right"])
def test_horizontal_alignment_positions_the_text_block(align: str) -> None:
    image = _render(
        [{"type": "text", "text": "Hello", "font_size": 32, "align": align}]
    )

    box = _bbox(image)
    assert box is not None
    left, _, right, _ = box
    if align == "left":
        assert left < 10
    elif align == "center":
        assert abs((left + right) / 2.0 - 100.0) <= 5
    else:
        assert right > 190


@pytest.mark.parametrize("vertical", ["top", "middle", "bottom"])
def test_vertical_alignment_positions_the_text_block(vertical: str) -> None:
    image = _render(
        [
            {
                "type": "text",
                "text": "Hello",
                "font_size": 32,
                "vertical_align": vertical,
            }
        ]
    )

    box = _bbox(image)
    assert box is not None
    _, top, _, bottom = box
    if vertical == "top":
        assert top < 15
    elif vertical == "middle":
        assert abs((top + bottom) / 2.0 - 30.0) <= 3
    else:
        assert bottom > 45


def test_wrapping_adds_lines_in_the_rendered_frame() -> None:
    wrapped = _render(
        [
            {
                "type": "text",
                "text": "one two three four five six",
                "font_size": 20,
                "width": 80,
                "height": 200,
            }
        ],
        width=100,
        height=220,
    )
    single = _render(
        [
            {
                "type": "text",
                "text": "one two three four five six",
                "font_size": 20,
                "width": 80,
                "height": 200,
                "wrap": False,
            }
        ],
        width=100,
        height=220,
    )

    wrapped_box = _bbox(wrapped)
    single_box = _bbox(single)
    assert wrapped_box is not None and single_box is not None
    assert (wrapped_box[3] - wrapped_box[1]) > (single_box[3] - single_box[1])


def test_text_colour_and_stroke_are_rendered() -> None:
    image = _render(
        [
            {
                "type": "text",
                "text": "Hello",
                "font_size": 32,
                "color": "#ff0000",
                "stroke": "#ffffff",
                "stroke_width": 2,
            }
        ],
        width=160,
        height=80,
    )

    assert _has_color(image, (255, 0, 0))
    assert _has_color(image, (255, 255, 255))


def test_text_follows_position_keyframes() -> None:
    composition = _composition(
        [
            {
                "type": "text",
                "text": "Hello",
                "font_size": 20,
                "color": "#ff0000",
                "keyframes": {
                    "position": [
                        {"time": 0, "value": (0, 0)},
                        {"time": 2, "value": (40, 0)},
                    ]
                },
            }
        ],
        width=120,
        height=40,
    )
    renderer = PillowStillRenderer()

    start = _bbox(renderer.render_still(composition, 0.0))
    end = _bbox(renderer.render_still(composition, 2.0))

    assert start is not None and end is not None
    assert end[0] > start[0] + 20


def test_text_default_size_comes_from_render_config() -> None:
    config = RenderConfig.from_mapping({"font_size": 24})
    layer = {"type": "text", "text": "Hello", "color": "#ff0000"}

    configured = PillowStillRenderer(render=config).render_still(_composition([layer]))
    explicit = _render([{**layer, "font_size": 24}])

    assert configured.tobytes() == explicit.tobytes()


def test_text_default_family_comes_from_render_config(tmp_path: Path) -> None:
    _copy_lato_family(tmp_path)
    config = RenderConfig.from_mapping(
        {"default_font": "Custom", "fonts_dir": str(tmp_path)}
    )
    renderer = PillowStillRenderer(render=config)
    layer = {"type": "text", "text": "Hello", "color": "#ff0000"}

    configured = renderer.render_still(_composition([layer]))
    explicit = renderer.render_still(_composition([{**layer, "font_family": "Custom"}]))

    assert configured.tobytes() == explicit.tobytes()


def test_explicit_text_style_overrides_render_config(tmp_path: Path) -> None:
    _copy_lato_family(tmp_path)
    _copy_lato_family(tmp_path, "Lato")
    config = RenderConfig.from_mapping(
        {
            "default_font": "Custom",
            "fonts_dir": str(tmp_path),
            "font_size": 24,
        }
    )
    layer = {
        "type": "text",
        "text": "Hello",
        "color": "#ff0000",
        "font_family": "Lato",
        "font_size": 32,
    }

    configured = PillowStillRenderer(render=config).render_still(_composition([layer]))
    explicit = _render([layer])

    assert configured.tobytes() == explicit.tobytes()


@pytest.mark.parametrize(
    "layer",
    [
        {"type": "text", "text": "Hi", "color": "not-a-colour"},
        {"type": "text", "text": "Hi", "font_family": "Missing"},
        {"type": "text", "text": "Hi", "align": "justify"},
        {"type": "text", "text": "Hi", "font_size": 0},
        {"type": "text", "text": "Hi", "max_lines": 0},
        {"type": "text", "text": "Hi", "line_height": 0},
        {"type": "text", "text": "Hi", "overflow": "scroll"},
        {"type": "text", "text": "Hi", "font_weight": "ultra"},
        {"type": "text", "text": "Hi", "stroke_width": -1},
        {"type": "text", "text": "   "},
    ],
)
def test_invalid_text_layers_raise_a_render_error(layer: dict[str, Any]) -> None:
    with pytest.raises(RenderError):
        _render([layer], validate=False)
