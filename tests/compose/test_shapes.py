"""Shape layers: rectangles, rounded rectangles, circles and lines."""

from __future__ import annotations

import math
from typing import Any

import pytest
from PIL import Image
from pydantic import ValidationError

from aivia.compose import (
    Canvas,
    Composition,
    Gradient,
    GradientStop,
    GradientType,
    Layer,
    LayerType,
    PillowStillRenderer,
    RenderError,
    ShapeLayer,
    ShapeType,
)


def _composition(
    layers: list[dict[str, Any]],
    *,
    width: int = 10,
    height: int = 10,
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
    validate: bool = True,
    **kwargs: Any,
) -> Image.Image:
    return PillowStillRenderer().render_still(
        _composition(layers, validate=validate, **kwargs)
    )


def test_shape_layer_carries_fill_stroke_radius_and_gradient() -> None:
    layer = ShapeLayer.model_validate(
        {
            "shape": "rounded-rect",
            "fill_color": " #ff0000 ",
            "border": "#ffffff",
            "border_width": 3,
            "corner_radius": 8,
            "gradient": {"from": "#ff0000", "to": "#0000ff", "angle": 90},
        }
    )

    assert layer.type is LayerType.SHAPE
    assert layer.shape is ShapeType.ROUNDED_RECTANGLE
    assert layer.fill == "#ff0000"
    assert layer.stroke == "#ffffff"
    assert layer.stroke_width == 3.0
    assert layer.radius == 8.0
    assert isinstance(layer.gradient, Gradient)
    assert layer.gradient.angle == 90.0


def test_shape_layer_defaults_and_aliases() -> None:
    layer = ShapeLayer.model_validate({"form": "rect", "color": "#123456"})

    assert layer.shape is ShapeType.RECTANGLE
    assert layer.fill == "#123456"
    assert layer.stroke is None
    assert layer.stroke_width == 1.0
    assert layer.radius == 0.0
    assert layer.gradient is None


def test_shape_layer_accepts_a_gradient_as_its_fill() -> None:
    layer = ShapeLayer.model_validate(
        {
            "shape": "circle",
            "fill": {"start_color": "#ff0000", "end_color": "#0000ff"},
        }
    )

    assert isinstance(layer.fill, Gradient)
    assert [stop.color for stop in layer.fill.stops] == ["#ff0000", "#0000ff"]


@pytest.mark.parametrize(
    "payload",
    [
        {"shape": "triangle"},
        {"shape": "rectangle", "stroke_width": -1},
        {"shape": "rectangle", "stroke_width": math.inf},
        {"shape": "rectangle", "radius": -1},
        {"shape": "rectangle", "radius": math.nan},
        {"shape": "rectangle", "stroke": 42},
        {
            "shape": "rectangle",
            "fill": {"start": "#fff", "end": "#000"},
            "gradient": {"start": "#fff", "end": "#000"},
        },
    ],
)
def test_shape_layer_rejects_invalid_style(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ShapeLayer.model_validate(payload)


def test_gradient_accepts_shorthands_and_ordered_stops() -> None:
    shorthand = Gradient.model_validate({"from": "#ff0000", "to": "#0000ff"})
    colors = Gradient.model_validate({"colors": ["#ff0000", "#00ff00", "#0000ff"]})
    strings = Gradient.model_validate({"stops": ["#ff0000", "#0000ff"]})
    mappings = Gradient.model_validate(
        {"stops": [{"color": "#ff0000"}, {"color": "#0000ff"}]}
    )

    assert shorthand.type is GradientType.LINEAR
    assert [(stop.color, stop.offset) for stop in shorthand.stops] == [
        ("#ff0000", 0.0),
        ("#0000ff", 1.0),
    ]
    assert [stop.offset for stop in colors.stops] == [0.0, 0.5, 1.0]
    assert [stop.offset for stop in strings.stops] == [0.0, 1.0]
    assert [stop.offset for stop in mappings.stops] == [0.0, 1.0]


@pytest.mark.parametrize(
    "payload",
    [
        {"stops": [{"color": "#fff", "offset": 0.0}]},
        {"stops": ["#fff", ""]},
        {
            "stops": [
                {"color": "#fff", "offset": 1.0},
                {"color": "#000", "offset": 0.0},
            ]
        },
        {"start": "#fff", "end": "#000", "angle": math.inf},
        {"stops": ["#fff", "#000"], "unknown": True},
    ],
)
def test_gradient_rejects_invalid_stops_and_angles(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Gradient.model_validate(payload)


def test_gradient_stop_aliases_and_validation() -> None:
    stop = GradientStop.model_validate({"colour": " #abcdef ", "position": 0.25})

    assert stop.color == "#abcdef"
    assert stop.offset == 0.25

    with pytest.raises(ValidationError):
        GradientStop.model_validate({"color": " ", "offset": 0.5})
    with pytest.raises(ValidationError):
        GradientStop.model_validate({"color": "#fff", "offset": 1.5})


def test_shape_layers_are_supported_by_the_still_renderer() -> None:
    assert LayerType.SHAPE in PillowStillRenderer.supported_layer_types


def test_a_rectangle_fills_only_its_geometry() -> None:
    image = _render(
        [
            {
                "id": "box",
                "type": "shape",
                "shape": "rectangle",
                "x": 2,
                "y": 2,
                "width": 4,
                "height": 3,
                "fill": "#ff0000",
            }
        ]
    )

    _assert_color(image, 2, 2, (255, 0, 0, 255))
    _assert_color(image, 5, 4, (255, 0, 0, 255))
    _assert_color(image, 1, 1, (0, 0, 0, 255))
    _assert_color(image, 6, 2, (0, 0, 0, 255))
    _assert_color(image, 2, 5, (0, 0, 0, 255))


def test_a_rectangle_draws_a_stroke_inside_its_geometry() -> None:
    image = _render(
        [
            {
                "type": "shape",
                "shape": "rect",
                "x": 2,
                "y": 2,
                "width": 4,
                "height": 4,
                "fill": "#ff0000",
                "stroke": "#ffffff",
                "stroke_width": 1,
            }
        ]
    )

    _assert_color(image, 2, 2, (255, 255, 255, 255))
    _assert_color(image, 5, 5, (255, 255, 255, 255))
    _assert_color(image, 3, 3, (255, 0, 0, 255))
    _assert_color(image, 1, 1, (0, 0, 0, 255))


def test_a_rounded_rectangle_clears_its_corners() -> None:
    image = _render(
        [
            {
                "type": "shape",
                "shape": "rounded_rectangle",
                "x": 2,
                "y": 2,
                "width": 6,
                "height": 6,
                "radius": 2,
                "fill": "#ff0000",
            }
        ]
    )

    _assert_color(image, 2, 2, (0, 0, 0, 255))
    _assert_color(image, 7, 7, (0, 0, 0, 255))
    _assert_color(image, 4, 4, (255, 0, 0, 255))
    _assert_color(image, 4, 2, (255, 0, 0, 255))


def test_a_circle_fills_an_ellipse_inside_its_box() -> None:
    image = _render(
        [
            {
                "type": "shape",
                "shape": "oval",
                "x": 2,
                "y": 2,
                "width": 6,
                "height": 6,
                "fill": "#00ff00",
            }
        ]
    )

    _assert_color(image, 2, 2, (0, 0, 0, 255))
    _assert_color(image, 4, 4, (0, 255, 0, 255))
    _assert_color(image, 4, 2, (0, 255, 0, 255))
    _assert_color(image, 7, 7, (0, 0, 0, 255))


def test_a_line_runs_from_the_top_left_to_the_bottom_right() -> None:
    image = _render(
        [
            {
                "type": "shape",
                "shape": "line",
                "x": 1,
                "y": 1,
                "width": 6,
                "height": 6,
                "stroke": "#ffffff",
                "stroke_width": 1,
            }
        ],
        width=8,
        height=8,
    )

    _assert_color(image, 1, 1, (255, 255, 255, 255))
    _assert_color(image, 3, 3, (255, 255, 255, 255))
    _assert_color(image, 6, 6, (255, 255, 255, 255))
    _assert_color(image, 1, 3, (0, 0, 0, 255))


def test_a_line_uses_its_fill_when_no_stroke_is_set() -> None:
    image = _render(
        [
            {
                "type": "shape",
                "shape": "line",
                "width": 4,
                "height": 4,
                "fill": "#0000ff",
                "stroke_width": 2,
            }
        ],
        width=4,
        height=4,
    )

    _assert_color(image, 1, 1, (0, 0, 255, 255))
    _assert_color(image, 2, 2, (0, 0, 255, 255))


@pytest.mark.parametrize(
    ("gradient", "first", "last"),
    [
        (
            {"start": "#ff0000", "end": "#0000ff"},
            (255, 0, 0, 255),
            (0, 0, 255, 255),
        ),
        (
            {"colors": ["#ff0000", "#00ff00", "#0000ff"]},
            (255, 0, 0, 255),
            (0, 0, 255, 255),
        ),
    ],
)
def test_a_rectangle_fills_with_a_linear_gradient(
    gradient: dict[str, Any],
    first: tuple[int, int, int, int],
    last: tuple[int, int, int, int],
) -> None:
    image = _render(
        [
            {
                "type": "shape",
                "shape": "rectangle",
                "gradient": gradient,
            }
        ],
        width=6,
        height=4,
    )

    _assert_color(image, 0, 0, first)
    _assert_color(image, 5, 0, last)


def test_a_gradient_angle_runs_top_to_bottom() -> None:
    image = _render(
        [
            {
                "type": "shape",
                "shape": "rectangle",
                "gradient": {"start": "#ff0000", "end": "#0000ff", "angle": 90},
            }
        ],
        width=4,
        height=4,
    )

    _assert_color(image, 0, 0, (255, 0, 0, 255))
    _assert_color(image, 0, 3, (0, 0, 255, 255))


def test_a_gradient_can_be_the_fill_of_a_circle() -> None:
    image = _render(
        [
            {
                "type": "shape",
                "shape": "circle",
                "x": 2,
                "y": 2,
                "width": 5,
                "height": 5,
                "fill": {"start": "#ff0000", "end": "#0000ff"},
            }
        ]
    )

    _assert_color(image, 2, 2, (0, 0, 0, 255))
    _assert_color(image, 4, 4, (127, 0, 128, 255), tolerance=1)


def test_shape_opacity_and_z_order_follow_the_layer_envelope() -> None:
    image = _render(
        [
            {
                "id": "back",
                "type": "shape",
                "shape": "rectangle",
                "fill": "#ff0000",
                "z": 0,
            },
            {
                "id": "front",
                "type": "shape",
                "shape": "rectangle",
                "x": 0,
                "y": 0,
                "width": 3,
                "height": 3,
                "fill": "#0000ff",
                "opacity": 0.5,
                "z": 1,
            },
        ]
    )

    _assert_color(image, 0, 0, (128, 0, 128, 255), tolerance=1)
    _assert_color(image, 5, 5, (255, 0, 0, 255))


def test_shape_geometry_follows_position_keyframes() -> None:
    composition = _composition(
        [
            {
                "type": "shape",
                "shape": "rectangle",
                "width": 2,
                "height": 2,
                "fill": "#ff0000",
                "keyframes": {
                    "position": [
                        {"time": 0, "value": (0, 0)},
                        {"time": 2, "value": (4, 2)},
                    ]
                },
            }
        ],
        duration_seconds=3.0,
    )
    renderer = PillowStillRenderer()

    start = renderer.render_still(composition, 0.0)
    middle = renderer.render_still(composition, 1.0)
    end = renderer.render_still(composition, 2.0)

    _assert_color(start, 0, 0, (255, 0, 0, 255))
    _assert_color(middle, 0, 0, (0, 0, 0, 255))
    _assert_color(middle, 2, 1, (255, 0, 0, 255))
    _assert_color(end, 4, 2, (255, 0, 0, 255))
    _assert_color(end, 0, 0, (0, 0, 0, 255))


@pytest.mark.parametrize(
    "layer",
    [
        {
            "type": "shape",
            "shape": "rectangle",
            "fill": "not-a-colour",
        },
        {
            "type": "shape",
            "shape": "rectangle",
            "gradient": {"stops": ["not-a-colour", "#000000"]},
        },
        {
            "type": "shape",
            "shape": "triangle",
        },
    ],
)
def test_invalid_shape_styles_raise_a_render_error(layer: dict[str, Any]) -> None:
    with pytest.raises(RenderError):
        _render([layer], validate=False)
