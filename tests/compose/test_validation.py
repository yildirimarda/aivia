"""Composition validation: every cross-object error in one pass."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aivia.compose import (
    Composition,
    CompositionIssue,
    CompositionIssues,
    CompositionValidationError,
    FitMode,
    ImageLayer,
    LogoLayer,
    ShapeLayer,
    StickerLayer,
    TextLayer,
    VideoLayer,
    validate_composition,
    validate_composition_or_raise,
)


def _composition(**overrides: Any) -> Composition:
    values: dict[str, Any] = {
        "canvas": "1080x1920",
        "duration_seconds": 10.0,
        "fps": 30,
    }
    values.update(overrides)
    return Composition.model_validate(values)


def _codes(issues: CompositionIssues) -> list[str]:
    return [issue.code for issue in issues]


def test_a_valid_composition_has_no_issues() -> None:
    composition = _composition(
        layers=[
            {"id": "background", "type": "image", "source": "bg.png", "duration": 10},
            {
                "id": "title",
                "type": "text",
                "text": "Hello",
                "start": 1,
                "duration": 3,
                "keyframes": {
                    "opacity": [
                        {"time": 0, "value": 0},
                        {"time": 1, "value": 1},
                    ]
                },
            },
        ],
        audio_tracks=[
            {"id": "voice", "source": "voice.mp3", "role": "voice", "duration": 8}
        ],
        effects=[{"id": "fade", "type": "fade", "start": 0, "end": 1}],
    )

    issues = validate_composition(composition, assets={"bg.png", "voice.mp3"})

    assert issues == []
    assert issues.ok is True
    assert issues.errors == []
    assert isinstance(issues, CompositionIssues)


def test_duplicate_ids_are_reported_for_layers_tracks_and_effects() -> None:
    composition = _composition(
        layers=[
            {"id": "same", "type": "text", "text": "one"},
            {"id": "same", "type": "text", "text": "two"},
            {"id": "same", "type": "text", "text": "three"},
        ],
        audio_tracks=[
            {"id": "audio", "source": "a.mp3"},
            {"id": "audio", "source": "b.mp3"},
        ],
        effects=[
            {"id": "effect", "type": "fade"},
            {"id": "effect", "type": "blur"},
        ],
    )

    issues = validate_composition(composition)

    assert _codes(issues) == ["duplicate_id"] * 4
    assert [issue.path for issue in issues] == [
        "layers[1].id",
        "layers[2].id",
        "audio_tracks[1].id",
        "effects[1].id",
    ]
    assert "duplicate layer id 'same'" in str(issues[0])
    assert "duplicate layer id 'same'" in str(issues[1])
    assert "duplicate audio track id 'audio'" in str(issues[2])
    assert "duplicate effect id 'effect'" in str(issues[3])


def test_timings_outside_the_composition_duration_are_all_reported() -> None:
    composition = _composition(
        layers=[
            {"id": "starts_late", "type": "text", "text": "x", "start": 10},
            {
                "id": "ends_late",
                "type": "text",
                "text": "x",
                "start": 8,
                "duration": 5,
            },
        ],
        audio_tracks=[
            {"id": "voice", "source": "voice.mp3", "start": 9, "duration": 4}
        ],
        effects=[
            {"id": "late_effect", "type": "fade", "start": 9, "end": 11},
            {"id": "starts_after", "type": "blur", "start": 10},
        ],
    )

    issues = validate_composition(composition)

    assert _codes(issues) == ["timing", "timing", "timing", "timing", "timing"]
    assert [issue.path for issue in issues] == [
        "layers[0]",
        "layers[1]",
        "audio_tracks[0]",
        "effects[0]",
        "effects[1]",
    ]
    assert "at or after the composition duration" in str(issues[0])
    assert "after the composition duration" in str(issues[1])
    assert "audio track ends at 13s" in str(issues[2])
    assert "effect ends at 11s" in str(issues[3])


def test_missing_assets_are_reported_for_layers_and_audio_tracks() -> None:
    composition = _composition(
        layers=[
            {"id": "bg", "type": "image", "source": "missing.png"},
            {"id": "logo", "type": "logo", "source": "present.svg"},
        ],
        audio_tracks=[{"id": "voice", "source": "voice.mp3"}],
    )

    issues = validate_composition(composition, assets={"present.svg"})

    assert _codes(issues) == ["missing_asset", "missing_asset"]
    assert [issue.path for issue in issues] == [
        "layers[0].source",
        "audio_tracks[0].source",
    ]
    assert "missing.png" in str(issues[0])
    assert "voice.mp3" in str(issues[1])


def test_asset_existence_can_use_a_callable_mapping_or_directory(
    tmp_path: Path,
) -> None:
    composition = _composition(
        layers=[{"id": "bg", "type": "image", "source": "bg.png"}]
    )

    assert validate_composition(composition, asset_exists=lambda source: False)
    assert not validate_composition(composition, asset_exists=lambda source: True)
    assert not validate_composition(composition, assets={"bg.png": "asset"})

    (tmp_path / "bg.png").write_bytes(b"image")
    assert not validate_composition(composition, assets=tmp_path)


def test_asset_existence_is_skipped_without_a_resolver() -> None:
    composition = _composition(
        layers=[{"id": "bg", "type": "image", "source": "missing.png"}],
        audio_tracks=[{"id": "voice", "source": "missing.mp3"}],
    )

    assert validate_composition(composition) == []


def test_unsorted_keyframes_and_out_of_range_values_are_all_reported() -> None:
    composition = _composition(
        layers=[
            {
                "id": "title",
                "type": "text",
                "text": "Hello",
                "keyframes": {
                    "opacity": [
                        {"time": 2, "value": 0.5},
                        {"time": 1, "value": 2},
                    ],
                    "scale": [
                        {"time": 0, "value": -1},
                        {"time": 11, "value": 1},
                    ],
                },
            }
        ]
    )

    issues = validate_composition(composition)

    assert _codes(issues) == [
        "keyframe_order",
        "keyframe_range",
        "keyframe_range",
        "keyframe_time",
    ]
    assert [issue.path for issue in issues] == [
        "layers[0].keyframes.opacity[1].time",
        "layers[0].keyframes.opacity[1].value",
        "layers[0].keyframes.scale[0].value",
        "layers[0].keyframes.scale[1].time",
    ]
    assert "strictly ascending" in str(issues[0])
    assert "above the maximum 1" in str(issues[1])
    assert "below the minimum 0" in str(issues[2])
    assert "after the composition duration" in str(issues[3])


def test_keyframe_times_cannot_outlive_an_explicit_layer_end() -> None:
    composition = _composition(
        layers=[
            {
                "id": "title",
                "type": "text",
                "text": "Hello",
                "start": 1,
                "end": 3,
                "keyframes": {
                    "opacity": [
                        {"time": 1, "value": 0},
                        {"time": 3, "value": 1},
                    ]
                },
                "scale_keyframes": [(1, 1), (4, 2)],
            }
        ]
    )

    issues = validate_composition(composition)

    assert _codes(issues) == ["keyframe_time"]
    assert [issue.path for issue in issues] == ["layers[0].scale_keyframes[1].time"]
    assert "after the layer window ending at 3s" in str(issues[0])


def test_keyframe_times_cannot_outlive_a_layer_duration() -> None:
    composition = _composition(
        layers=[
            {
                "id": "title",
                "type": "text",
                "text": "Hello",
                "start": 2,
                "duration": 2,
                "keyframes": {
                    "position": [
                        {"at": 2, "value": [0, 0]},
                        {"at": 4.5, "value": [10, 20]},
                    ]
                },
            }
        ]
    )

    issues = validate_composition(composition)

    assert _codes(issues) == ["keyframe_time"]
    assert [issue.path for issue in issues] == ["layers[0].keyframes.position[1].time"]
    assert "after the layer window ending at 4s" in str(issues[0])


def test_keyframes_accept_pairs_and_property_specific_containers() -> None:
    composition = _composition(
        layers=[
            {
                "id": "title",
                "type": "text",
                "text": "Hello",
                "opacity_keyframes": [(0, 0.0), (1, 1.0)],
                "animations": {
                    "position": [
                        {"at": 0, "value": [0, 0]},
                        {"at": 2, "value": [10, 20]},
                    ]
                },
            }
        ]
    )

    assert validate_composition(composition) == []


def test_invalid_keyframe_shapes_and_values_are_reported() -> None:
    composition = _composition(
        layers=[
            {
                "id": "title",
                "type": "text",
                "text": "Hello",
                "keyframes": {
                    "opacity": "not-a-list",
                    "rotation": [{"value": 1}, {"time": "soon", "value": "fast"}],
                    "scale": [{"time": 1, "value": [1, 2]}],
                },
            }
        ]
    )

    issues = validate_composition(composition)

    assert "keyframe" in _codes(issues)
    assert "keyframe_time" in _codes(issues)
    assert "keyframe_range" in _codes(issues)
    assert any("keyframes must be a list" in str(issue) for issue in issues)
    assert any("keyframe needs a time" in str(issue) for issue in issues)
    assert any(
        "keyframe time must be a finite number" in str(issue) for issue in issues
    )
    assert any("must be a single number" in str(issue) for issue in issues)


def test_composition_layers_are_parsed_into_their_concrete_classes() -> None:
    composition = _composition(
        layers=[
            {"id": "bg", "type": "image", "source": "bg.png", "fit": "contain"},
            {"id": "clip", "type": "video", "source": "clip.mp4", "loop": True},
            {"id": "title", "type": "text", "text": "Hello", "font_size": 32},
            {"id": "badge", "type": "shape", "shape": "circle"},
            {"id": "sticker", "type": "sticker", "source": "sticker.png"},
            {"id": "logo", "type": "logo", "source": "logo.svg"},
        ]
    )

    layers = composition.layers
    assert isinstance(layers[0], ImageLayer)
    assert isinstance(layers[1], VideoLayer)
    assert isinstance(layers[2], TextLayer)
    assert isinstance(layers[3], ShapeLayer)
    assert isinstance(layers[4], StickerLayer)
    assert isinstance(layers[5], LogoLayer)
    assert layers[0].fit is FitMode.CONTAIN
    assert layers[1].loop is True
    assert layers[2].font_size == 32.0
    assert layers[4].source == "sticker.png"


@pytest.mark.parametrize(
    ("layer", "path"),
    [
        ({"type": "image"}, "layers[0].source"),
        ({"type": "image", "source": "bg.png", "fit": "squish"}, "layers[0].fit"),
        (
            {"type": "video", "source": "clip.mp4", "speed": 0},
            "layers[0].speed",
        ),
        ({"type": "text", "text": "Hi", "font_size": 0}, "layers[0].font_size"),
        ({"type": "text"}, "layers[0].text"),
        ({"type": "shape", "shape": "triangle"}, "layers[0].shape"),
        (
            {"type": "sticker", "source": "s.png", "anchor": "diagonal"},
            "layers[0].anchor",
        ),
        ({"type": "logo", "source": "l.svg", "opacity": 2}, "layers[0].opacity"),
    ],
)
def test_composition_validates_type_specific_layer_fields(
    layer: dict[str, Any], path: str
) -> None:
    issues = validate_composition(
        {
            "canvas": "1080x1920",
            "duration_seconds": 10.0,
            "fps": 30,
            "layers": [layer],
        }
    )

    assert _codes(issues) == ["schema"]
    assert issues[0].path == path


def test_composition_keeps_unmodelled_layer_fields_as_extras() -> None:
    composition = _composition(
        layers=[
            {
                "id": "bg",
                "type": "image",
                "source": "bg.png",
                "x": 12,
                "glow": True,
                "keyframes": {"opacity": [(0, 0), (1, 1)]},
            }
        ]
    )

    layer = composition.layers[0]
    assert isinstance(layer, ImageLayer)
    assert layer.x == 12
    assert layer.model_extra == {
        "source": "bg.png",
        "glow": True,
    }
    assert layer.keyframes is not None
    assert layer.model_dump()["keyframes"] == {"opacity": [(0, 0), (1, 1)]}


def test_schema_errors_are_returned_with_paths_instead_of_raising() -> None:
    issues = validate_composition(
        {
            "duration_seconds": 0,
            "fps": 0,
            "layers": [{"type": "not-a-layer"}],
        }
    )

    assert _codes(issues) == ["schema"] * len(issues)
    assert len(issues) >= 4
    paths = {issue.path for issue in issues}
    assert "canvas" in paths
    assert "duration_seconds" in paths
    assert "fps" in paths
    assert "layers[0].type" in paths


def test_validate_or_raise_reports_every_issue_on_the_error() -> None:
    composition = _composition(
        layers=[
            {"id": "same", "type": "text", "text": "one"},
            {"id": "same", "type": "text", "text": "two"},
        ],
        audio_tracks=[
            {"id": "voice", "source": "missing.mp3", "start": 9, "duration": 5}
        ],
    )

    with pytest.raises(CompositionValidationError) as raised:
        validate_composition_or_raise(composition, assets=set())

    error = raised.value
    assert len(error.issues) == 3
    assert all(isinstance(issue, CompositionIssue) for issue in error.issues)
    assert "composition is invalid" in str(error)
    assert [item["code"] for item in error.details["errors"]] == [
        "duplicate_id",
        "timing",
        "missing_asset",
    ]

    issues = validate_composition(composition, assets=set())
    with pytest.raises(CompositionValidationError):
        issues.raise_for_errors()


def test_validate_or_raise_returns_the_loaded_composition() -> None:
    payload = {"canvas": "1080x1920", "duration_seconds": 5.0}

    composition = validate_composition_or_raise(payload)

    assert isinstance(composition, Composition)
    assert composition.duration_seconds == 5.0
