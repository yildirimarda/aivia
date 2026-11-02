"""Sticker and logo layers: nine anchors, margins and an optional drop shadow."""

from __future__ import annotations

import math
from typing import Any

import pytest
from PIL import Image
from pydantic import ValidationError

from aivia.compose import (
    Anchor,
    Canvas,
    Composition,
    FitMode,
    Layer,
    LayerType,
    LogoLayer,
    PillowStillRenderer,
    RenderError,
    StickerLayer,
)

CANVAS = (100, 80)
STICKER_SIZE = (20, 10)
RED = (255, 0, 0, 255)
BLUE = (0, 0, 255, 255)
BLACK = (0, 0, 0, 255)


def _composition(
    layers: list[dict[str, Any]],
    *,
    width: int = CANVAS[0],
    height: int = CANVAS[1],
    background: str = "#0000ff",
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


def _solid(
    color: tuple[int, int, int, int],
    size: tuple[int, int] = STICKER_SIZE,
) -> Image.Image:
    return Image.new("RGBA", size, color)


def _pixel(image: Image.Image, x: int, y: int) -> tuple[int, int, int, int]:
    value = image.getpixel((x, y))
    assert isinstance(value, tuple)
    return value[0], value[1], value[2], value[3]


def _assert_color(
    image: Image.Image,
    x: int,
    y: int,
    expected: tuple[int, int, int, int],
    *,
    tolerance: int = 0,
) -> None:
    actual = _pixel(image, x, y)
    for channel, target in zip(actual, expected, strict=True):
        assert abs(channel - target) <= tolerance, (actual, expected)


def _render(
    layers: list[dict[str, Any]],
    *,
    assets: dict[str, Image.Image] | None = None,
    validate: bool = True,
    **kwargs: Any,
) -> Image.Image:
    return PillowStillRenderer(assets).render_still(
        _composition(layers, validate=validate, **kwargs)
    )


def test_sticker_layer_defaults_to_center_without_a_shadow() -> None:
    layer = StickerLayer(source="sticker.png")

    assert layer.type is LayerType.STICKER
    assert layer.anchor is Anchor.CENTER
    assert layer.margin == 0.0
    assert layer.margin_x == 0.0
    assert layer.margin_y == 0.0
    assert layer.width is None
    assert layer.height is None
    assert layer.scale == 1.0
    assert layer.fit is FitMode.CONTAIN
    assert layer.shadow is False
    assert layer.shadow_color == "#000000"
    assert layer.shadow_offset == (4.0, 4.0)
    assert layer.shadow_blur == 0.0
    assert layer.shadow_opacity == 0.5


def test_logo_layer_accepts_anchor_margin_and_shadow_aliases() -> None:
    layer = LogoLayer.model_validate(
        {
            "asset_id": " logo.png ",
            "corner": "bottom-right",
            "margins": [12, 18],
            "w": 80,
            "h": 40,
            "zoom": 0.5,
            "mode": "contain",
            "drop_shadow": True,
            "shadow_colour": " #112233 ",
            "shadow_position": (-3, 4),
            "shadow_softness": 2,
            "shadow_alpha": 0.75,
        }
    )

    assert layer.source == "logo.png"
    assert layer.anchor is Anchor.BOTTOM_RIGHT
    assert layer.margin == (12.0, 18.0)
    assert layer.margin_x == 12.0
    assert layer.margin_y == 18.0
    assert layer.width == 80
    assert layer.height == 40
    assert layer.scale == 0.5
    assert layer.fit is FitMode.CONTAIN
    assert layer.shadow is True
    assert layer.shadow_color == "#112233"
    assert layer.shadow_offset == (-3.0, 4.0)
    assert layer.shadow_blur == 2.0
    assert layer.shadow_opacity == 0.75


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("top-left", Anchor.TOP_LEFT),
        ("top", Anchor.TOP_CENTER),
        ("top right", Anchor.TOP_RIGHT),
        ("left", Anchor.MIDDLE_LEFT),
        ("centre", Anchor.CENTER),
        ("middle", Anchor.CENTER),
        ("right", Anchor.MIDDLE_RIGHT),
        ("bottom-left", Anchor.BOTTOM_LEFT),
        ("south", Anchor.BOTTOM_CENTER),
        ("south-east", Anchor.BOTTOM_RIGHT),
        ("northwest", Anchor.TOP_LEFT),
        ("NE", Anchor.TOP_RIGHT),
    ],
)
def test_anchor_accepts_nine_positions_and_common_aliases(
    value: str, expected: Anchor
) -> None:
    assert Anchor(value) is expected
    assert StickerLayer.model_validate({"source": "a.png", "anchor": value}).anchor is (
        expected
    )


@pytest.mark.parametrize("value", ["middle-centre", "north-west-ish", 42, True])
def test_anchor_rejects_unknown_values(value: Any) -> None:
    with pytest.raises(ValidationError):
        StickerLayer.model_validate({"source": "a.png", "anchor": value})


@pytest.mark.parametrize(
    "payload",
    [
        {"margin": -1},
        {"margin": [1, -1]},
        {"margin": [1, 2, 3]},
        {"margin": True},
        {"margin": math.inf},
        {"width": 0},
        {"height": -2},
        {"width": 10.5},
        {"scale": 0},
        {"scale": math.nan},
        {"fit": "squish"},
        {"shadow": "yes"},
        {"shadow_color": " "},
        {"shadow_color": 12},
        {"shadow_offset": [1]},
        {"shadow_offset": [1, math.inf]},
        {"shadow_blur": -1},
        {"shadow_opacity": 1.5},
        {"shadow_opacity": -0.1},
        {"unknown": True},
    ],
)
def test_sticker_layer_rejects_invalid_placement_and_shadow_fields(
    payload: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        StickerLayer.model_validate({"source": "a.png", **payload})


def test_sticker_and_logo_layers_are_supported_by_the_still_renderer() -> None:
    assert LayerType.STICKER in PillowStillRenderer.supported_layer_types
    assert LayerType.LOGO in PillowStillRenderer.supported_layer_types


@pytest.mark.parametrize("layer_type", ["sticker", "logo"])
@pytest.mark.parametrize(
    ("anchor", "expected"),
    [
        ("top_left", (5, 5)),
        ("top_center", (40, 5)),
        ("top_right", (75, 5)),
        ("middle_left", (5, 35)),
        ("center", (40, 35)),
        ("middle_right", (75, 35)),
        ("bottom_left", (5, 65)),
        ("bottom_center", (40, 65)),
        ("bottom_right", (75, 65)),
    ],
)
def test_every_anchor_pins_the_overlay_inside_its_margin(
    layer_type: str, anchor: str, expected: tuple[int, int]
) -> None:
    image = _render(
        [
            {
                "type": layer_type,
                "source": "badge.png",
                "anchor": anchor,
                "margin": 5,
                "width": STICKER_SIZE[0],
                "height": STICKER_SIZE[1],
            }
        ],
        assets={"badge.png": _solid(RED)},
    )

    x, y = expected
    _assert_color(image, x, y, RED)
    _assert_color(image, x + STICKER_SIZE[0] - 1, y + STICKER_SIZE[1] - 1, RED)
    if x > 0:
        _assert_color(image, x - 1, y, BLUE)
    if y > 0:
        _assert_color(image, x, y - 1, BLUE)


@pytest.mark.parametrize(
    ("margin", "expected"),
    [
        (6, (6, 6)),
        ([3, 7], (3, 7)),
        ((3, 7), (3, 7)),
    ],
)
def test_scalar_and_pair_margins_offset_the_anchor(
    margin: Any, expected: tuple[int, int]
) -> None:
    image = _render(
        [
            {
                "type": "sticker",
                "source": "badge.png",
                "anchor": "top_left",
                "margin": margin,
                "width": STICKER_SIZE[0],
                "height": STICKER_SIZE[1],
            }
        ],
        assets={"badge.png": _solid(RED)},
    )

    x, y = expected
    _assert_color(image, x, y, RED)
    if x > 0:
        _assert_color(image, x - 1, y, BLUE)
    if y > 0:
        _assert_color(image, x, y - 1, BLUE)


def test_an_overlay_keeps_its_natural_size_when_no_size_is_given() -> None:
    image = _render(
        [
            {
                "type": "logo",
                "source": "logo.png",
                "anchor": "top_left",
            }
        ],
        assets={"logo.png": _solid(RED, (12, 8))},
    )

    _assert_color(image, 0, 0, RED)
    _assert_color(image, 11, 7, RED)
    _assert_color(image, 12, 0, BLUE)
    _assert_color(image, 0, 8, BLUE)


def test_width_height_and_scale_size_the_overlay() -> None:
    by_width = _render(
        [
            {
                "type": "sticker",
                "source": "badge.png",
                "anchor": "top_left",
                "width": 40,
            }
        ],
        assets={"badge.png": _solid(RED)},
    )
    _assert_color(by_width, 39, 19, RED)
    _assert_color(by_width, 40, 20, BLUE)

    by_height = _render(
        [
            {
                "type": "sticker",
                "source": "badge.png",
                "anchor": "top_left",
                "height": 30,
            }
        ],
        assets={"badge.png": _solid(RED)},
    )
    _assert_color(by_height, 59, 29, RED)
    _assert_color(by_height, 60, 30, BLUE)

    scaled = _render(
        [
            {
                "type": "sticker",
                "source": "badge.png",
                "anchor": "top_left",
                "scale": 0.5,
            }
        ],
        assets={"badge.png": _solid(RED)},
    )
    _assert_color(scaled, 9, 4, RED)
    _assert_color(scaled, 10, 5, BLUE)


def test_contain_preserves_aspect_and_stretch_fills_the_box() -> None:
    contained = _render(
        [
            {
                "type": "logo",
                "source": "logo.png",
                "anchor": "center",
                "width": 40,
                "height": 40,
            }
        ],
        assets={"logo.png": _solid(RED)},
    )
    _assert_color(contained, 30, 30, RED)
    _assert_color(contained, 69, 49, RED)
    _assert_color(contained, 30, 29, BLUE)
    _assert_color(contained, 30, 50, BLUE)

    stretched = _render(
        [
            {
                "type": "logo",
                "source": "logo.png",
                "anchor": "center",
                "width": 40,
                "height": 40,
                "fit": "stretch",
            }
        ],
        assets={"logo.png": _solid(RED)},
    )
    _assert_color(stretched, 30, 20, RED)
    _assert_color(stretched, 69, 59, RED)


def test_no_shadow_leaves_the_background_untouched() -> None:
    image = _render(
        [
            {
                "type": "sticker",
                "source": "badge.png",
                "anchor": "top_left",
                "margin": 10,
                "width": STICKER_SIZE[0],
                "height": STICKER_SIZE[1],
            }
        ],
        assets={"badge.png": _solid(RED)},
    )

    _assert_color(image, 11, 11, RED)
    _assert_color(image, 33, 13, BLUE)


def test_an_optional_drop_shadow_is_offset_behind_the_overlay() -> None:
    image = _render(
        [
            {
                "type": "sticker",
                "source": "badge.png",
                "anchor": "top_left",
                "margin": 10,
                "width": STICKER_SIZE[0],
                "height": STICKER_SIZE[1],
                "shadow": True,
                "shadow_color": "#000000",
                "shadow_offset": [5, 0],
                "shadow_blur": 0,
                "shadow_opacity": 1,
            }
        ],
        assets={"badge.png": _solid(RED)},
    )

    _assert_color(image, 11, 11, RED)
    _assert_color(image, 31, 13, BLACK)
    _assert_color(image, 34, 13, BLACK)
    _assert_color(image, 36, 13, BLUE)
    _assert_color(image, 13, 21, BLUE)


def test_a_shadow_can_be_softened_and_semi_transparent() -> None:
    image = _render(
        [
            {
                "type": "logo",
                "source": "logo.png",
                "anchor": "top_left",
                "margin": 10,
                "width": STICKER_SIZE[0],
                "height": STICKER_SIZE[1],
                "shadow": True,
                "shadow_offset": [0, 0],
                "shadow_blur": 4,
                "shadow_opacity": 0.5,
            }
        ],
        assets={"logo.png": _solid(RED)},
    )

    red, green, blue, _ = _pixel(image, 31, 13)
    assert red < 255
    assert blue > 0
    assert green == 0


@pytest.mark.parametrize(
    "payload",
    [
        {"anchor": "middle-centre"},
        {"margin": -1},
        {"width": 0},
        {"height": -1},
        {"scale": 0},
        {"fit": "squish"},
        {"shadow": "yes"},
        {"shadow": True, "shadow_offset": [1]},
        {"shadow": True, "shadow_blur": -1},
        {"shadow": True, "shadow_opacity": 2},
        {"shadow": True, "shadow_color": "not-a-colour"},
    ],
)
def test_the_renderer_rejects_invalid_overlay_placement(
    payload: dict[str, Any],
) -> None:
    with pytest.raises(RenderError):
        _render(
            [
                {
                    "type": "sticker",
                    "source": "badge.png",
                    **payload,
                }
            ],
            assets={"badge.png": _solid(RED)},
            validate=False,
        )


def test_a_supported_overlay_without_a_source_fails_clearly() -> None:
    with pytest.raises(RenderError, match="has no source"):
        _render([{"type": "logo", "anchor": "center"}], validate=False)

    with pytest.raises(RenderError, match="no asset registered"):
        _render(
            [{"type": "sticker", "source": "missing.png"}],
            assets={"badge.png": _solid(RED)},
        )
