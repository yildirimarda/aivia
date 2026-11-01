"""Concrete layer types: image, video, text, shape, sticker and logo."""

from __future__ import annotations

import math
from typing import Any

import pytest
from pydantic import ValidationError

from aivia.compose import (
    AnyLayer,
    FitMode,
    ImageLayer,
    Layer,
    LayerAdapter,
    LayerType,
    LayerUnion,
    LogoLayer,
    ShapeLayer,
    ShapeType,
    StickerLayer,
    TextAlign,
    TextLayer,
    VideoLayer,
    parse_layer,
)


def test_every_concrete_layer_has_the_expected_type_and_fields() -> None:
    image = ImageLayer(source="cover.png")
    video = VideoLayer(source="clip.mp4")
    text = TextLayer(text="Hello")
    shape = ShapeLayer(shape=ShapeType.RECTANGLE)
    sticker = StickerLayer(source="sticker.png")
    logo = LogoLayer(source="logo.svg")

    assert [layer.type for layer in (image, video, text, shape, sticker, logo)] == list(
        LayerType
    )
    assert isinstance(image, Layer)
    assert image.source == "cover.png"
    assert video.source == "clip.mp4"
    assert text.text == "Hello"
    assert shape.shape is ShapeType.RECTANGLE
    assert sticker.source == "sticker.png"
    assert logo.source == "logo.svg"


@pytest.mark.parametrize(
    ("layer_class", "payload", "expected"),
    [
        (ImageLayer, "cover.png", {"source": "cover.png"}),
        (VideoLayer, "clip.mp4", {"source": "clip.mp4"}),
        (TextLayer, "Hello", {"text": "Hello"}),
        (ShapeLayer, "circle", {"shape": "circle"}),
        (StickerLayer, "sticker.png", {"source": "sticker.png"}),
        (LogoLayer, "logo.svg", {"source": "logo.svg"}),
    ],
)
def test_concrete_layers_accept_a_string_shorthand(
    layer_class: type[Layer], payload: str, expected: dict[str, Any]
) -> None:
    layer = layer_class.model_validate(payload)

    for field, value in expected.items():
        assert getattr(layer, field) == value
    assert layer.type is layer_class.model_fields["type"].default


@pytest.mark.parametrize(
    ("layer_class", "payload"),
    [
        (ImageLayer, {}),
        (ImageLayer, {"source": ""}),
        (ImageLayer, {"source": "   "}),
        (ImageLayer, {"source": 42}),
        (VideoLayer, {}),
        (VideoLayer, {"source": ""}),
        (TextLayer, {}),
        (TextLayer, {"text": ""}),
        (TextLayer, {"text": "   "}),
        (TextLayer, {"text": 42}),
        (ShapeLayer, {}),
        (ShapeLayer, {"shape": "triangle"}),
        (StickerLayer, {}),
        (StickerLayer, {"source": ""}),
        (LogoLayer, {}),
        (LogoLayer, {"source": ""}),
    ],
)
def test_concrete_layers_reject_missing_or_invalid_required_fields(
    layer_class: type[Layer], payload: dict[str, Any]
) -> None:
    with pytest.raises(ValidationError):
        layer_class.model_validate(payload)


@pytest.mark.parametrize(
    ("layer_class", "payload"),
    [
        (ImageLayer, {"type": "video", "source": "cover.png"}),
        (VideoLayer, {"type": "image", "source": "clip.mp4"}),
        (TextLayer, {"type": "shape", "text": "Hello"}),
        (ShapeLayer, {"type": "text", "shape": "circle"}),
        (StickerLayer, {"type": "logo", "source": "sticker.png"}),
        (LogoLayer, {"type": "sticker", "source": "logo.svg"}),
    ],
)
def test_concrete_layers_reject_a_mismatched_type(
    layer_class: type[Layer], payload: dict[str, Any]
) -> None:
    with pytest.raises(ValidationError):
        layer_class.model_validate(payload)


def test_concrete_layers_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ImageLayer.model_validate({"source": "cover.png", "unknown": True})


@pytest.mark.parametrize(
    "layer_class",
    [ImageLayer, VideoLayer, TextLayer, ShapeLayer, StickerLayer, LogoLayer],
)
def test_concrete_layers_keep_the_common_envelope_validation(
    layer_class: type[Layer],
) -> None:
    required = {
        ImageLayer: {"source": "cover.png"},
        VideoLayer: {"source": "clip.mp4"},
        TextLayer: {"text": "Hello"},
        ShapeLayer: {"shape": "circle"},
        StickerLayer: {"source": "sticker.png"},
        LogoLayer: {"source": "logo.svg"},
    }[layer_class]

    layer = layer_class.model_validate(
        {**required, "id": " layer ", "start": 1, "duration": 2, "z": 3}
    )
    assert layer.id == "layer"
    assert layer.start == 1.0
    assert layer.duration == 2.0
    assert layer.end == 3.0
    assert layer.z == 3

    with pytest.raises(ValidationError):
        layer_class.model_validate({**required, "opacity": 1.5})
    with pytest.raises(ValidationError):
        layer_class.model_validate({**required, "start": -1})
    with pytest.raises(ValidationError):
        layer_class.model_validate({**required, "start": 2, "end": 1})


def test_image_layer_validates_and_aliases_fit_modes() -> None:
    assert ImageLayer(source="a.png").fit is FitMode.COVER
    assert (
        ImageLayer.model_validate({"source": "a.png", "fit": "contain"}).fit
        is FitMode.CONTAIN
    )
    assert (
        ImageLayer.model_validate({"source": "a.png", "mode": "crop"}).fit
        is FitMode.COVER
    )
    assert (
        ImageLayer.model_validate({"source": "a.png", "fit": "fill"}).fit
        is FitMode.STRETCH
    )
    assert (
        ImageLayer.model_validate({"source": "a.png", "fit": "actual"}).fit
        is FitMode.NONE
    )
    assert ImageLayer.model_validate({"asset_id": " a.png "}).source == "a.png"

    with pytest.raises(ValidationError):
        ImageLayer.model_validate({"source": "a.png", "fit": "squish"})


def test_video_layer_validates_trim_loop_audio_and_speed() -> None:
    layer = VideoLayer.model_validate(
        {
            "path": "clip.mp4",
            "trim_start": 1,
            "trim_end": 4,
            "loop": True,
            "mute": True,
            "speed": 2,
        }
    )

    assert layer.source == "clip.mp4"
    assert layer.trim_start_seconds == 1.0
    assert layer.trim_end_seconds == 4.0
    assert layer.loop is True
    assert layer.muted is True
    assert layer.playback_rate == 2.0
    assert layer.fit is FitMode.COVER

    for payload in (
        {"source": "clip.mp4", "trim_start": -1},
        {"source": "clip.mp4", "trim_end": -1},
        {"source": "clip.mp4", "trim_start": 4, "trim_end": 4},
        {"source": "clip.mp4", "trim_start": 5, "trim_end": 4},
        {"source": "clip.mp4", "trim_start": math.inf},
        {"source": "clip.mp4", "playback_rate": 0},
        {"source": "clip.mp4", "playback_rate": -1},
        {"source": "clip.mp4", "playback_rate": math.nan},
    ):
        with pytest.raises(ValidationError):
            VideoLayer.model_validate(payload)


def test_text_layer_validates_content_typography_and_alignment() -> None:
    layer = TextLayer.model_validate(
        {
            "content": "  Hello  ",
            "font": " Inter ",
            "size": 64,
            "text_color": " #fff ",
            "alignment": "centre",
        }
    )

    assert layer.text == "  Hello  "
    assert layer.font_family == "Inter"
    assert layer.font_size == 64.0
    assert layer.color == "#fff"
    assert layer.align is TextAlign.CENTER
    assert (
        TextLayer.model_validate({"text": "Hi", "align": "end"}).align
        is TextAlign.RIGHT
    )

    for payload in (
        {"text": "Hi", "font_size": 0},
        {"text": "Hi", "font_size": -1},
        {"text": "Hi", "font_size": math.inf},
        {"text": "Hi", "align": "justify"},
        {"text": "Hi", "font_family": 1},
        {"text": "Hi", "color": None},
    ):
        with pytest.raises(ValidationError):
            TextLayer.model_validate(payload)


def test_shape_layer_validates_and_aliases_shape_kinds() -> None:
    assert ShapeLayer.model_validate({"shape": "rect"}).shape is ShapeType.RECTANGLE
    assert (
        ShapeLayer.model_validate({"shape": "rounded-rect"}).shape
        is ShapeType.ROUNDED_RECTANGLE
    )
    assert ShapeLayer.model_validate({"shape": "oval"}).shape is ShapeType.CIRCLE
    assert ShapeLayer.model_validate({"shape": "line_segment"}).shape is ShapeType.LINE
    assert ShapeLayer.model_validate({"form": "square"}).shape is ShapeType.RECTANGLE


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"type": "image", "source": "cover.png"}, ImageLayer),
        ({"kind": "video", "source": "clip.mp4"}, VideoLayer),
        ({"type": "text", "text": "Hello"}, TextLayer),
        ({"type": "shape", "shape": "circle"}, ShapeLayer),
        ({"type": "sticker", "source": "sticker.png"}, StickerLayer),
        ({"type": "logo", "source": "logo.svg"}, LogoLayer),
    ],
)
def test_parse_layer_dispatches_to_the_concrete_class(
    payload: dict[str, Any], expected: type[Layer]
) -> None:
    layer = parse_layer(payload)

    assert isinstance(layer, expected)
    assert LayerAdapter.validate_python(payload) == layer


def test_parse_layer_rejects_an_unknown_or_missing_type() -> None:
    with pytest.raises(ValidationError):
        parse_layer({"type": "audio", "source": "voice.mp3"})
    with pytest.raises(ValidationError):
        parse_layer({"source": "cover.png"})


def test_any_layer_union_and_aliases_expose_every_concrete_class() -> None:
    assert LayerUnion is AnyLayer
    assert ImageLayer in AnyLayer.__args__
    assert VideoLayer in AnyLayer.__args__
    assert TextLayer in AnyLayer.__args__
    assert ShapeLayer in AnyLayer.__args__
    assert StickerLayer in AnyLayer.__args__
    assert LogoLayer in AnyLayer.__args__


@pytest.mark.parametrize(
    ("layer_class", "payload"),
    [
        (ImageLayer, {"source": "cover.png", "fit": "contain"}),
        (
            VideoLayer,
            {"source": "clip.mp4", "trim_start": 1, "trim_end": 2, "loop": True},
        ),
        (TextLayer, {"text": "Hello", "font_size": 32, "align": "center"}),
        (ShapeLayer, {"shape": "rounded_rectangle"}),
        (StickerLayer, {"source": "sticker.png"}),
        (LogoLayer, {"source": "logo.svg"}),
    ],
)
def test_concrete_layers_round_trip_through_json(
    layer_class: type[Layer], payload: dict[str, Any]
) -> None:
    layer = layer_class.model_validate(payload)

    restored = layer_class.model_validate(layer.model_dump(mode="json"))

    assert restored == layer
