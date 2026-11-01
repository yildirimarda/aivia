"""Masks, blend modes and basic filters for image layers."""

from __future__ import annotations

from typing import Any

import pytest
from PIL import Image, ImageChops
from pydantic import ValidationError

from aivia.compose import (
    BlendMode,
    Composition,
    FilterType,
    FitMode,
    ImageFilter,
    ImageLayer,
    ImageMask,
    MaskMode,
    PillowStillRenderer,
    RenderError,
)


def _composition(**overrides: Any) -> Composition:
    values: dict[str, Any] = {
        "canvas": "2x2",
        "duration_seconds": 1.0,
        "fps": 30,
    }
    values.update(overrides)
    return Composition.model_validate(values)


def _solid(
    color: tuple[int, int, int, int],
    size: tuple[int, int] = (2, 2),
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
    tolerance: int = 1,
) -> None:
    actual = _pixel(image, x, y)
    for channel, target in zip(actual, expected, strict=True):
        assert abs(channel - target) <= tolerance, (actual, expected)


def _render(
    layer: dict[str, Any],
    assets: dict[str, Image.Image],
    *,
    canvas: str = "2x2",
    background: str = "#000000",
) -> Image.Image:
    composition = _composition(
        canvas={"resolution": canvas, "background": background},
        layers=[layer],
    )
    return PillowStillRenderer(assets).render_still(composition)


def test_blend_mode_enum_accepts_aliases_and_rejects_unknown_modes() -> None:
    assert BlendMode("source-over") is BlendMode.NORMAL
    assert BlendMode("src_over") is BlendMode.NORMAL
    assert BlendMode("plus") is BlendMode.ADD
    assert BlendMode("lighter") is BlendMode.LIGHTEN
    assert BlendMode("soft-light") is BlendMode.SOFT_LIGHT

    with pytest.raises(ValueError):
        BlendMode("colour-burn")


def test_blend_mode_aliases_and_mask_and_filters_validate_on_image_layers() -> None:
    layer = ImageLayer.model_validate(
        {
            "source": "cover.png",
            "blend": "multiply",
            "mask": "mask.png",
            "filter": {"grayscale": True, "blur": 2},
        }
    )

    assert layer.blend_mode is BlendMode.MULTIPLY
    assert layer.mask is not None
    assert layer.mask.source == "mask.png"
    assert layer.mask.mode is MaskMode.ALPHA
    assert [spec.type for spec in layer.filters] == [
        FilterType.GRAYSCALE,
        FilterType.BLUR,
    ]
    assert layer.filters[1].radius == 2.0

    flat = ImageLayer.model_validate(
        {
            "source": "cover.png",
            "blending_mode": "screen",
            "mask_source": "mask.png",
            "mask_mode": "luminance",
            "mask_invert": True,
            "mask_feather": 1,
        }
    )
    assert flat.blend_mode is BlendMode.SCREEN
    assert flat.mask is not None
    assert flat.mask.source == "mask.png"
    assert flat.mask.mode is MaskMode.LUMINANCE
    assert flat.mask.invert is True
    assert flat.mask.feather == 1.0

    with pytest.raises(ValidationError):
        ImageLayer.model_validate({"source": "cover.png", "blend": "colour-burn"})
    with pytest.raises(ValidationError):
        ImageLayer.model_validate({"source": "cover.png", "filter": {"zoom": 2}})
    with pytest.raises(ValidationError):
        ImageLayer.model_validate({"source": "cover.png", "mask": ""})


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (BlendMode.NORMAL, (200, 100, 50, 255)),
        (BlendMode.MULTIPLY, (78, 58, 39, 255)),
        (BlendMode.SCREEN, (222, 192, 211, 255)),
        (BlendMode.OVERLAY, (157, 127, 167, 255)),
        (BlendMode.ADD, (255, 250, 250, 255)),
        (BlendMode.SUBTRACT, (0, 50, 150, 255)),
        (BlendMode.DARKEN, (100, 100, 50, 255)),
        (BlendMode.LIGHTEN, (200, 150, 200, 255)),
        (BlendMode.DIFFERENCE, (100, 50, 150, 255)),
        (BlendMode.HARD_LIGHT, (188, 118, 78, 255)),
        (BlendMode.SOFT_LIGHT, (134, 136, 173, 255)),
    ],
)
def test_blend_modes_combine_two_solid_colours_numerically(
    mode: BlendMode,
    expected: tuple[int, int, int, int],
) -> None:
    image = _render(
        {
            "type": "image",
            "source": "top.png",
            "fit": "stretch",
            "blend_mode": mode,
        },
        {"top.png": _solid((200, 100, 50, 255))},
        background="#6496c8",
    )

    _assert_color(image, 0, 0, expected)
    _assert_color(image, 1, 1, expected)


def test_blend_modes_respect_the_layer_alpha_channel() -> None:
    image = _render(
        {
            "type": "image",
            "source": "top.png",
            "fit": "stretch",
            "blend": "multiply",
        },
        {"top.png": _solid((200, 100, 50, 128))},
        background="#6496c8",
    )

    _assert_color(image, 0, 0, (89, 104, 119, 255))


def test_a_luminance_mask_limits_visibility_and_can_be_inverted() -> None:
    mask = Image.new("L", (2, 2), 0)
    mask.paste(255, (0, 0, 1, 2))
    assets = {
        "red.png": _solid((255, 0, 0, 255)),
        "mask.png": mask,
    }

    image = _render(
        {
            "type": "image",
            "source": "red.png",
            "fit": "stretch",
            "mask": {"source": "mask.png", "mode": "luminance"},
        },
        assets,
    )
    _assert_color(image, 0, 0, (255, 0, 0, 255))
    _assert_color(image, 0, 1, (255, 0, 0, 255))
    _assert_color(image, 1, 0, (0, 0, 0, 255))
    _assert_color(image, 1, 1, (0, 0, 0, 255))

    inverted = _render(
        {
            "type": "image",
            "source": "red.png",
            "fit": "stretch",
            "mask": {
                "source": "mask.png",
                "mode": "luminance",
                "invert": True,
            },
        },
        assets,
    )
    _assert_color(inverted, 0, 0, (0, 0, 0, 255))
    _assert_color(inverted, 1, 0, (255, 0, 0, 255))


def test_an_alpha_mask_uses_the_mask_alpha_channel() -> None:
    mask = Image.new("RGBA", (2, 2), (255, 255, 255, 0))
    mask.paste((255, 255, 255, 255), (0, 0, 1, 2))

    image = _render(
        {
            "type": "image",
            "source": "red.png",
            "fit": "stretch",
            "mask": "mask.png",
        },
        {"red.png": _solid((255, 0, 0, 255)), "mask.png": mask},
    )

    _assert_color(image, 0, 0, (255, 0, 0, 255))
    _assert_color(image, 1, 0, (0, 0, 0, 255))


def test_a_mask_is_scaled_to_the_layer_box() -> None:
    image = _render(
        {
            "type": "image",
            "source": "red.png",
            "fit": "stretch",
            "mask": {"source": "mask.png", "mode": "luminance"},
        },
        {
            "red.png": _solid((255, 0, 0, 255)),
            "mask.png": Image.new("L", (1, 1), 128),
        },
    )

    _assert_color(image, 0, 0, (128, 0, 0, 255))
    _assert_color(image, 1, 1, (128, 0, 0, 255))


def test_mask_feather_softens_the_alpha_edge() -> None:
    mask = Image.new("L", (4, 1), 0)
    mask.paste(255, (0, 0, 2, 1))

    image = _render(
        {
            "type": "image",
            "source": "red.png",
            "fit": "stretch",
            "mask": {
                "source": "mask.png",
                "mode": "luminance",
                "feather": 1,
            },
        },
        {"red.png": _solid((255, 0, 0, 255)), "mask.png": mask},
        canvas="4x1",
    )

    red = [_pixel(image, x, 0)[0] for x in range(4)]
    assert all(0 < value < 255 for value in red), red
    assert red == sorted(red, reverse=True), red


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ({"type": "grayscale"}, (76, 76, 76, 255)),
        ({"type": "invert"}, (0, 255, 255, 255)),
        ({"type": "brightness", "amount": 0.5}, (127, 0, 0, 255)),
        ({"type": "contrast", "amount": 0}, (76, 76, 76, 255)),
        ({"type": "saturation", "amount": 0}, (76, 76, 76, 255)),
    ],
)
def test_basic_filters_change_pixels_with_expected_numeric_results(
    spec: dict[str, Any],
    expected: tuple[int, int, int, int],
) -> None:
    image = _render(
        {
            "type": "image",
            "source": "red.png",
            "fit": "stretch",
            "filters": [spec],
        },
        {"red.png": _solid((255, 0, 0, 255))},
        canvas="1x1",
    )

    _assert_color(image, 0, 0, expected, tolerance=0)


def test_filter_shorthand_runs_in_declared_order() -> None:
    brightness_then_invert = _render(
        {
            "type": "image",
            "source": "red.png",
            "fit": "stretch",
            "filters": {"brightness": 0.5, "invert": True},
        },
        {"red.png": _solid((255, 0, 0, 255))},
        canvas="1x1",
    )
    invert_then_brightness = _render(
        {
            "type": "image",
            "source": "red.png",
            "fit": "stretch",
            "filters": [
                {"type": "invert"},
                {"type": "brightness", "amount": 0.5},
            ],
        },
        {"red.png": _solid((255, 0, 0, 255))},
        canvas="1x1",
    )

    _assert_color(brightness_then_invert, 0, 0, (128, 255, 255, 255), tolerance=0)
    _assert_color(invert_then_brightness, 0, 0, (0, 127, 127, 255), tolerance=0)


def test_blur_and_sharpen_change_neighbouring_pixels() -> None:
    edge = Image.new("RGBA", (5, 1), (0, 0, 0, 255))
    edge.paste((255, 255, 255, 255), (3, 0, 5, 1))
    gradient = Image.new("RGBA", (5, 1))
    for x in range(5):
        gradient.putpixel((x, 0), (x * 40, x * 40, x * 40, 255))

    sharpened = _render(
        {
            "type": "image",
            "source": "gradient.png",
            "fit": "stretch",
            "filters": [{"type": "sharpen", "amount": 1}],
        },
        {"gradient.png": gradient},
        canvas="5x1",
    )
    blurred = _render(
        {
            "type": "image",
            "source": "edge.png",
            "fit": "stretch",
            "filters": {"blur": 1},
        },
        {"edge.png": edge},
        canvas="5x1",
    )
    plain = _render(
        {"type": "image", "source": "edge.png", "fit": "stretch"},
        {"edge.png": edge},
        canvas="5x1",
    )

    assert _pixel(sharpened, 4, 0)[0] > _pixel(gradient, 4, 0)[0]
    assert (
        ImageChops.difference(blurred.convert("RGB"), plain.convert("RGB")).getbbox()
        is not None
    )
    assert (
        _pixel(blurred, 0, 0)[0] < _pixel(blurred, 2, 0)[0] < _pixel(blurred, 4, 0)[0]
    )


def test_filters_and_masks_are_deterministic_and_repeatable() -> None:
    mask = Image.new("L", (2, 2), 0)
    mask.paste(255, (0, 0, 1, 2))
    layer = {
        "type": "image",
        "source": "red.png",
        "fit": "stretch",
        "blend": "screen",
        "mask": {"source": "mask.png", "mode": "luminance", "feather": 0.5},
        "filters": [{"type": "brightness", "amount": 0.5}],
    }
    assets = {"red.png": _solid((255, 0, 0, 255)), "mask.png": mask}
    composition = _composition(
        canvas={"resolution": "2x2", "background": "#102030"},
        layers=[layer],
    )

    first = PillowStillRenderer(assets).render_still(composition)
    second = PillowStillRenderer(assets).render_still(composition)

    assert (
        ImageChops.difference(first.convert("RGB"), second.convert("RGB")).getbbox()
        is None
    )


def test_invalid_blend_or_filter_values_raise_a_render_error() -> None:
    composition = _composition(
        layers=[{"type": "image", "source": "red.png", "fit": "stretch"}]
    )
    extra = composition.layers[0].__pydantic_extra__
    assert extra is not None
    extra["blend_mode"] = "colour-burn"

    with pytest.raises(RenderError):
        PillowStillRenderer({"red.png": _solid((255, 0, 0, 255))}).render_still(
            composition
        )


def test_mask_and_filter_models_expose_shorthands_and_defaults() -> None:
    mask = ImageMask.model_validate("mask.png")
    assert mask.source == "mask.png"
    assert mask.mode is MaskMode.ALPHA
    assert mask.fit is FitMode.STRETCH
    assert mask.feather == 0.0
    assert mask.invert is False

    filter_spec = ImageFilter.model_validate("blur")
    assert filter_spec.type is FilterType.BLUR
    assert filter_spec.amount == 1.0
    assert filter_spec.radius is None
