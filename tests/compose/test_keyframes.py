"""Keyframes and easing for position, scale, rotation and opacity."""

from __future__ import annotations

import math
from typing import Any

import pytest
from pydantic import ValidationError

from aivia.compose import (
    EASING_FUNCTIONS,
    AnimatedProperty,
    Composition,
    Easing,
    Keyframe,
    KeyframeAnimation,
    KeyframeTrack,
    Layer,
    LayerKeyframes,
    apply_easing,
    ease_in,
    keyframe_track,
    keyframes_for_layer,
    linear,
    sample_layer,
    validate_composition,
)


def _close(actual: Any, expected: float) -> bool:
    assert isinstance(actual, float)
    return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)


@pytest.mark.parametrize("easing", list(Easing))
def test_every_easing_starts_at_zero_and_ends_at_one(easing: Easing) -> None:
    assert _close(apply_easing(easing, 0.0), 0.0)
    assert _close(apply_easing(easing, 1.0), 1.0)
    assert easing in EASING_FUNCTIONS


@pytest.mark.parametrize(
    ("easing", "expected"),
    [
        (Easing.LINEAR, 0.5),
        (Easing.EASE_IN, 0.25),
        (Easing.EASE_OUT, 0.75),
        (Easing.EASE_IN_OUT, 0.5),
        (Easing.EASE_IN_QUAD, 0.25),
        (Easing.EASE_OUT_QUAD, 0.75),
        (Easing.EASE_IN_CUBIC, 0.125),
        (Easing.EASE_OUT_CUBIC, 0.875),
        (Easing.SMOOTHSTEP, 0.5),
        (Easing.EASE_IN_SINE, 1 - math.cos(math.pi / 4)),
        (Easing.EASE_OUT_SINE, math.sin(math.pi / 4)),
    ],
)
def test_named_easings_have_known_midpoints(easing: Easing, expected: float) -> None:
    assert _close(apply_easing(easing, 0.5), expected)


def test_easing_accepts_aliases_strings_and_custom_callables() -> None:
    assert Easing("ease-in-out") is Easing.EASE_IN_OUT
    assert Easing("easeIn") is Easing.EASE_IN
    assert Easing("smooth") is Easing.SMOOTHSTEP
    assert apply_easing("ease-out", 0.5) == apply_easing(Easing.EASE_OUT, 0.5)
    assert apply_easing(lambda progress: progress / 2, 1.0) == 0.5


def test_easing_clamps_progress_and_rejects_bad_values() -> None:
    assert apply_easing(Easing.LINEAR, -2.0) == 0.0
    assert apply_easing(Easing.LINEAR, 4.0) == 1.0
    with pytest.raises(ValueError):
        apply_easing(Easing.LINEAR, math.nan)
    with pytest.raises(TypeError):
        apply_easing(Easing.LINEAR, "soon")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Easing("wobble")


def test_step_easing_holds_until_the_end() -> None:
    assert Easing.STEP.apply(0.0) == 0.0
    assert Easing.STEP.apply(0.99) == 0.0
    assert Easing.STEP.apply(1.0) == 1.0


def test_keyframe_accepts_aliases_and_a_pair_value() -> None:
    frame = Keyframe.model_validate({"at": 1.5, "to": [10, 20], "ease": "ease-in"})

    assert frame.time_seconds == frame.time == frame.at == 1.5
    assert frame.value == (10.0, 20.0)
    assert frame.easing is Easing.EASE_IN


def test_keyframe_rejects_non_finite_times_and_values() -> None:
    with pytest.raises(ValidationError):
        Keyframe.model_validate({"time": math.inf, "value": 1})
    with pytest.raises(ValidationError):
        Keyframe.model_validate({"time": 0, "value": math.nan})
    with pytest.raises(ValidationError):
        Keyframe.model_validate({"time": 0, "value": [1, 2, 3]})
    with pytest.raises(ValidationError):
        Keyframe.model_validate({"time": 0, "value": "fast"})


def test_scalar_track_interpolates_and_holds_outside_the_range() -> None:
    track = KeyframeTrack.model_validate(
        {
            "property": "opacity",
            "keyframes": [
                {"time": 0, "value": 0.0},
                {"time": 1, "value": 1.0},
            ],
        }
    )

    assert track.sample(-1) == 0.0
    assert track.sample(0) == 0.0
    assert _close(track.sample(0.25), 0.25)
    assert _close(track.sample(0.5), 0.5)
    assert track.sample(1) == 1.0
    assert track.sample(10) == 1.0
    assert track.at(0.5) == track.value_at(0.5) == track(0.5)


def test_pair_track_interpolates_each_component() -> None:
    track = keyframe_track(
        AnimatedProperty.POSITION,
        [(0, (0, 100)), (2, (40, 0))],
    )

    assert track.sample(0) == (0.0, 100.0)
    assert track.sample(1) == (20.0, 50.0)
    assert track.sample(2) == (40.0, 0.0)


def test_the_starting_keyframe_easing_controls_its_segment() -> None:
    track = keyframe_track(
        "opacity",
        [
            {"time": 0, "value": 0, "easing": "ease-in"},
            {"time": 1, "value": 1, "easing": "step"},
            {"time": 2, "value": 0},
        ],
    )

    assert _close(track.sample(0.5), 0.25)
    assert track.sample(1.5) == 1.0
    assert track.sample(2) == 0.0


def test_scale_accepts_a_scalar_or_a_pair() -> None:
    scalar = keyframe_track("scale", [(0, 1), (1, 2)])
    pair = keyframe_track("scale", [(0, (1, 2)), (1, (3, 4))])

    assert scalar.sample(0.5) == 1.5
    assert pair.sample(0.5) == (2.0, 3.0)


def test_tracks_reject_unsorted_times_and_inconsistent_shapes() -> None:
    with pytest.raises(ValidationError, match="strictly ascending"):
        keyframe_track("opacity", [(1, 1), (0, 0)])
    with pytest.raises(ValidationError, match="strictly ascending"):
        keyframe_track("opacity", [(0, 1), (0, 0)])
    with pytest.raises(ValidationError, match="same value shape"):
        keyframe_track("scale", [(0, 1), (1, (2, 2))])


@pytest.mark.parametrize(
    ("property_name", "frames"),
    [
        ("position", [(0, 1), (1, 2)]),
        ("position", [(0, (1, 2, 3)), (1, (1, 2, 3))]),
        ("rotation", [(0, (0, 0)), (1, (1, 1))]),
        ("opacity", [(0, (0, 0)), (1, (1, 1))]),
        ("opacity", [(0, -0.1), (1, 1)]),
        ("opacity", [(0, 0), (1, 1.1)]),
        ("scale", [(0, -1), (1, 1)]),
        ("scale", [(0, (1, -2)), (1, (1, 2))]),
    ],
)
def test_property_specific_value_validation(
    property_name: str, frames: list[Any]
) -> None:
    with pytest.raises(ValidationError):
        keyframe_track(property_name, frames)


def test_a_mapping_of_times_is_accepted_as_keyframes() -> None:
    track = KeyframeTrack.model_validate(
        {
            "property": "rotation",
            "keyframes": {
                0: 0,
                1: {"value": 90, "easing": "step"},
                2: 180,
            },
        }
    )

    assert track.times == (0.0, 1.0, 2.0)
    assert track.sample(0.5) == 45.0
    assert track.sample(1.5) == 90.0
    assert track.sample(2) == 180.0


def test_layer_keyframes_read_grouped_and_per_property_spellings() -> None:
    layer = Layer.model_validate(
        {
            "type": "text",
            "text": "Hello",
            "keyframes": {
                "position": [(0, (0, 0)), (1, (10, 20))],
                "opacity": [(0, 0), (1, 1)],
            },
            "rotation_animation": [(0, 0), (2, 360)],
            "scale_keyframes": [(0, 1), (1, 2)],
        }
    )

    animation = keyframes_for_layer(layer)

    assert isinstance(animation, LayerKeyframes)
    assert isinstance(animation, KeyframeAnimation)
    assert set(animation.tracks) == set(AnimatedProperty)
    assert animation.position is not None
    assert animation.sample(0.5) == {
        AnimatedProperty.POSITION: (5.0, 10.0),
        AnimatedProperty.OPACITY: 0.5,
        AnimatedProperty.ROTATION: 90.0,
        AnimatedProperty.SCALE: 1.5,
    }


def test_sample_layer_returns_only_declared_properties() -> None:
    layer = Layer.model_validate(
        {"type": "text", "text": "Hello", "opacity_keyframes": [(0, 0), (1, 1)]}
    )

    assert sample_layer(layer, 0.5) == {AnimatedProperty.OPACITY: 0.5}
    assert keyframes_for_layer(Layer.model_validate({"type": "text"})).sample(1.0) == {}


def test_layer_keyframes_can_be_built_from_raw_mappings() -> None:
    animation = LayerKeyframes.model_validate(
        {
            "position": {"keyframes": [(0, [0, 0]), (1, [100, 50])]},
            "opacity": [(0, 0), (1, 1)],
        }
    )

    assert animation.sample_property("position", 0.5) == (50.0, 25.0)
    assert animation.sample_property(AnimatedProperty.OPACITY, 0.5) == 0.5
    assert animation.value("rotation", 0.5) is None
    assert "scale" not in animation
    with pytest.raises(KeyError):
        animation["scale"]


def test_animation_rejects_a_track_for_the_wrong_property() -> None:
    with pytest.raises(ValidationError):
        LayerKeyframes.model_validate(
            {
                "opacity": {
                    "property": "rotation",
                    "keyframes": [(0, 0), (1, 1)],
                }
            }
        )


def test_public_easing_functions_match_their_named_curves() -> None:
    assert linear(0.5) == 0.5
    assert ease_in(0.5) == 0.25
    assert ease_in(2.0) == 1.0


def test_layer_keyframes_accept_a_grouped_mapping_and_property_aliases() -> None:
    animation = LayerKeyframes.model_validate(
        {"keyframes": {"alpha": [(0, 0), (1, 1)], "translate": [(0, [0, 0])]}}
    )

    assert animation.opacity is not None
    assert animation.position is not None
    assert animation.sample_property("opacity", 0.5) == 0.5


def test_composition_validation_reports_an_unknown_easing() -> None:
    composition = Composition.model_validate(
        {
            "canvas": "1080x1920",
            "duration_seconds": 2,
            "layers": [
                {
                    "type": "text",
                    "text": "Hello",
                    "keyframes": {
                        "opacity": [
                            {"time": 0, "value": 0, "easing": "wobble"},
                            {"time": 1, "value": 1},
                        ]
                    },
                }
            ],
        }
    )

    issues = validate_composition(composition)

    assert [issue.code for issue in issues] == ["keyframe_easing"]
    assert "wobble" in str(issues[0])
