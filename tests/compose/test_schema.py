"""Composition schema: canvas, timing, layers, audio, effects and version."""

from __future__ import annotations

import math
from typing import Any

import pytest
from pydantic import ValidationError

from aivia.compose import (
    COMPOSITION_SCHEMA_VERSION,
    CURRENT_COMPOSITION_SCHEMA_VERSION,
    AudioTrack,
    AudioTrackRole,
    AudioTrackSpec,
    BaseLayer,
    Canvas,
    CanvasConfig,
    CanvasSpec,
    Composition,
    CompositionConfig,
    CompositionSchema,
    CompositionSpec,
    Effect,
    EffectConfig,
    EffectSpec,
    Layer,
    LayerConfig,
    LayerSpec,
    LayerType,
    TrackRole,
)


def _canvas() -> dict[str, int]:
    return {"width": 1080, "height": 1920}


def _composition(**overrides: Any) -> Composition:
    values: dict[str, Any] = {
        "canvas": _canvas(),
        "duration_seconds": 10.0,
        "fps": 30,
    }
    values.update(overrides)
    return Composition.model_validate(values)


def test_composition_carries_canvas_duration_fps_layers_audio_and_effects() -> None:
    composition = Composition.model_validate(
        {
            "id": "composition_demo",
            "schema_version": 1,
            "canvas": "1080x1920",
            "duration": 12.0,
            "frame_rate": 25,
            "layers": [
                {
                    "id": "background",
                    "type": "image",
                    "source": "asset_background",
                    "start": 0,
                    "end": 12.0,
                },
                {"id": "title", "kind": "text", "start": 1, "duration": 3},
            ],
            "tracks": [
                {
                    "id": "music",
                    "source": "music.mp3",
                    "role": "music",
                    "volume_db": -3,
                    "fade_in": 0.5,
                }
            ],
            "effects": [
                "fade",
                {
                    "id": "vignette",
                    "type": "vignette",
                    "start": 2,
                    "end": 8,
                    "params": {"strength": 0.4},
                },
            ],
        }
    )

    assert composition.id == "composition_demo"
    assert composition.schema_version == 1
    assert isinstance(composition.canvas, Canvas)
    assert composition.canvas.size == (1080, 1920)
    assert composition.duration == 12.0
    assert composition.fps == 25
    assert composition.frame_rate == 25
    assert composition.frame_count == 300
    assert [layer.id for layer in composition.layers] == ["background", "title"]
    assert composition.layers[0].type is LayerType.IMAGE
    assert composition.layers[0].model_extra == {"source": "asset_background"}
    assert composition.layers[1].end == 4.0
    assert composition.audio_tracks == composition.tracks
    assert composition.audio_tracks[0].role is AudioTrackRole.MUSIC
    assert composition.audio_tracks[0].volume == -3.0
    assert [effect.type for effect in composition.effects] == ["fade", "vignette"]
    assert composition.effects[1].params == {"strength": 0.4}


def test_composition_defaults_are_empty_and_versioned() -> None:
    composition = Composition(
        canvas=Canvas(width=1920, height=1080), duration_seconds=5
    )

    assert composition.id == ""
    assert composition.schema_version == CURRENT_COMPOSITION_SCHEMA_VERSION
    assert composition.schema_version == COMPOSITION_SCHEMA_VERSION == 1
    assert composition.version == 1
    assert composition.fps == 30
    assert composition.layers == []
    assert composition.audio_tracks == []
    assert composition.audio == []
    assert composition.effects == []
    assert composition.layer_count == 0
    assert composition.frame_count == 150
    assert composition.width == 1920
    assert composition.height == 1080
    assert composition.aspect_ratio == pytest.approx(16 / 9)


def test_composition_round_trips_through_json() -> None:
    composition = _composition(
        id="composition_round_trip",
        layers=[{"id": "logo", "type": "logo", "start": 1, "duration": 2}],
        audio_tracks=[{"id": "voice", "source": "voice.mp3", "role": "voice"}],
        effects=[{"id": "blur", "type": "blur"}],
    )

    payload = composition.model_dump(mode="json")
    restored = Composition.model_validate(payload)

    assert restored == composition
    assert restored.model_dump(mode="json") == payload


def test_composition_rejects_unknown_top_level_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        _composition(unknown_field=True)


def test_composition_rejects_invalid_schema_version() -> None:
    with pytest.raises(ValidationError):
        _composition(schema_version=0)


def test_composition_keeps_an_explicit_schema_version() -> None:
    composition = _composition(schema_version=2)

    assert composition.schema_version == 2


def test_canvas_accepts_shorthand_and_reports_geometry() -> None:
    from_string = Canvas.model_validate("1080x1920")
    from_resolution = Canvas.model_validate({"resolution": "1920x1080"})
    from_pair = Canvas.model_validate([1280, 720])
    from_dimensions = Canvas.model_validate(
        {"size": "1080x1350", "background": " #ffffff "}
    )

    assert from_string.size == (1080, 1920)
    assert from_string.resolution == "1080x1920"
    assert from_string.aspect_ratio == pytest.approx(9 / 16)
    assert from_string.pixels == 1080 * 1920
    assert from_resolution.size == (1920, 1080)
    assert from_pair.size == (1280, 720)
    assert from_dimensions.size == (1080, 1350)
    assert from_dimensions.background == "#ffffff"


@pytest.mark.parametrize(
    "payload",
    [
        {"width": 0, "height": 1920},
        {"width": 1080, "height": -1},
        {"width": True, "height": 1920},
        {"width": 10.5, "height": 1920},
        "not-a-size",
        "1080x",
        [1080],
    ],
)
def test_canvas_rejects_invalid_dimensions(payload: object) -> None:
    with pytest.raises(ValidationError):
        Canvas.model_validate(payload)


@pytest.mark.parametrize("duration", [0, -1, math.inf, math.nan, True, "ten"])
def test_composition_rejects_an_invalid_duration(duration: object) -> None:
    with pytest.raises(ValidationError):
        _composition(duration_seconds=duration)


@pytest.mark.parametrize("fps", [0, -1, 29.97, True])
def test_composition_rejects_an_invalid_frame_rate(fps: object) -> None:
    with pytest.raises(ValidationError):
        _composition(fps=fps)


def test_layer_supports_every_declared_type() -> None:
    layers = [Layer(type=kind) for kind in LayerType]

    assert {layer.type for layer in layers} == set(LayerType)
    assert {kind.value for kind in LayerType} == {
        "image",
        "video",
        "text",
        "shape",
        "sticker",
        "logo",
    }


def test_layer_accepts_shorthand_and_derives_timing() -> None:
    layer = Layer.model_validate("text")

    assert layer.type is LayerType.TEXT
    assert layer.start == 0.0
    assert layer.duration is None
    assert layer.end is None

    timed = Layer.model_validate({"type": "image", "start": 2, "duration": 3, "z": 4})
    assert timed.start_seconds == 2.0
    assert timed.duration == 3.0
    assert timed.end == 5.0
    assert timed.z == 4


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "image", "start": 5, "end": 5},
        {"type": "image", "start": 5, "end": 4},
        {"type": "image", "start": 0, "duration": 2, "end": 3},
        {"type": "image", "opacity": -0.1},
        {"type": "image", "opacity": 1.1},
        {"type": "image", "start": math.inf},
        {"type": "image", "start": -1},
        {"type": "image", "z": 1.5},
        {"type": "not-a-layer"},
    ],
)
def test_layer_rejects_invalid_values(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Layer.model_validate(payload)


def test_audio_track_supports_placement_gain_fades_and_ducking() -> None:
    voice = AudioTrack.model_validate(
        {
            "id": "voice",
            "source": "voice.mp3",
            "role": "voice",
            "start": 1.5,
            "duration": 8,
            "volume": 2,
            "fade_in": 0.25,
            "fade_out": 0.5,
        }
    )
    music = AudioTrack.model_validate("music.mp3")

    assert voice.source == "voice.mp3"
    assert voice.role is AudioTrackRole.VOICE
    assert voice.start == 1.5
    assert voice.duration == 8.0
    assert voice.volume == 2.0
    assert voice.is_voice is True
    assert voice.should_duck is False
    assert music.source == "music.mp3"
    assert music.role is AudioTrackRole.MUSIC
    assert music.should_duck is True
    assert TrackRole is AudioTrackRole


@pytest.mark.parametrize(
    "payload",
    [
        {"source": ""},
        {"source": "music.mp3", "start": -1},
        {"source": "music.mp3", "duration": 0},
        {"source": "music.mp3", "fade_in": -1},
        {"source": "music.mp3", "volume_db": math.inf},
        {"source": "music.mp3", "duration": 1, "fade_out": 2},
        {"source": "music.mp3", "role": "unknown"},
    ],
)
def test_audio_track_rejects_invalid_values(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        AudioTrack.model_validate(payload)


def test_effect_accepts_shorthand_and_a_time_window() -> None:
    shorthand = Effect.model_validate("fade")
    windowed = Effect.model_validate(
        {
            "id": "blur",
            "type": "blur",
            "start": 1,
            "end": 3,
            "params": {"radius": 8},
        }
    )

    assert shorthand.type == "fade"
    assert shorthand.name == "fade"
    assert windowed.kind == "blur"
    assert windowed.start == 1.0
    assert windowed.end == 3.0
    assert windowed.params == {"radius": 8}


@pytest.mark.parametrize(
    "payload",
    [
        {"type": ""},
        {"type": "fade", "start": 3, "end": 3},
        {"type": "fade", "start": 3, "end": 2},
        {"type": "fade", "start": math.nan},
    ],
)
def test_effect_rejects_invalid_values(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Effect.model_validate(payload)


def test_composition_looks_up_objects_by_id() -> None:
    composition = _composition(
        layers=[{"id": "title", "type": "text"}],
        audio_tracks=[{"id": "voice", "source": "voice.mp3"}],
        effects=[{"id": "fade", "type": "fade"}],
    )

    assert composition.layer("title") is composition.layers[0]
    assert composition.track("voice") is composition.audio_tracks[0]
    assert composition.effect("fade") is composition.effects[0]
    assert composition.layer("missing") is None
    assert composition.track("missing") is None
    assert composition.effect("missing") is None


def test_public_aliases_point_at_the_same_schema() -> None:
    assert CanvasSpec is Canvas
    assert CanvasConfig is Canvas
    assert LayerSpec is Layer
    assert LayerConfig is Layer
    assert BaseLayer is Layer
    assert AudioTrackSpec is AudioTrack
    assert EffectSpec is Effect
    assert EffectConfig is Effect
    assert CompositionSpec is Composition
    assert CompositionConfig is Composition
    assert CompositionSchema is Composition
