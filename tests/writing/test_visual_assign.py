"""Every beat gets a visual type and AI video never exceeds the configured cap."""

from __future__ import annotations

import pytest

from aivia.core.models.script import Beat, Script, VisualType
from aivia.writing import (
    DEFAULT_MAX_AI_VIDEO_BEATS,
    DEFAULT_VISUAL_STRATEGY,
    VisualAssigner,
    VisualStrategy,
    assign_visual_types,
    normalise_visual_strategy,
    visual_types,
)


def _script(count: int = 5) -> Script:
    return Script(
        title="Bees",
        beats=[Beat(index=index, text=f"Fact {index + 1}.") for index in range(count)],
    )


def _values(script: Script) -> list[str]:
    return [beat.visual_type for beat in script.beats]


def test_the_default_hybrid_assignment_caps_ai_video_beats() -> None:
    script = _script()

    result = assign_visual_types(script)

    assert DEFAULT_VISUAL_STRATEGY == "hybrid"
    assert DEFAULT_MAX_AI_VIDEO_BEATS == 2
    assert _values(result) == [
        VisualType.VIDEO.value,
        VisualType.VIDEO.value,
        VisualType.IMAGE.value,
        VisualType.IMAGE.value,
        VisualType.IMAGE.value,
    ]
    assert all(beat.visual_type for beat in result.beats)


@pytest.mark.parametrize(
    ("limit", "videos"),
    [
        (0, 0),
        (1, 1),
        (2, 2),
        (5, 5),
        (99, 5),
    ],
)
def test_the_video_limit_is_a_hard_cap(limit: int, videos: int) -> None:
    result = assign_visual_types(
        _script(),
        strategy="video",
        max_ai_video_beats=limit,
    )

    values = _values(result)
    assert values.count(VisualType.VIDEO.value) == videos
    assert values[videos:] == [VisualType.IMAGE.value] * (len(values) - videos)


def test_the_cap_is_across_a_sequence_not_per_beat() -> None:
    beats = _script(4).beats

    result = assign_visual_types(beats, strategy="video", max_ai_video_beats=1)

    assert [beat.visual_type for beat in result] == [
        VisualType.VIDEO.value,
        VisualType.IMAGE.value,
        VisualType.IMAGE.value,
        VisualType.IMAGE.value,
    ]


@pytest.mark.parametrize(
    ("strategy", "visual"),
    [
        ("image", VisualType.IMAGE),
        ("ai_image", VisualType.IMAGE),
        ("stock", VisualType.STOCK),
        ("still", VisualType.STILL),
        ("text", VisualType.TEXT),
        ("none", VisualType.NONE),
    ],
)
def test_fixed_strategies_assign_every_beat(
    strategy: str,
    visual: VisualType,
) -> None:
    result = assign_visual_types(
        _script(),
        strategy=strategy,
        max_ai_video_beats=0,
    )

    assert _values(result) == [visual.value] * len(result.beats)


@pytest.mark.parametrize("strategy", ["video", "ai_video", "videos", "video_gen"])
def test_video_aliases_share_the_same_cap(strategy: str) -> None:
    result = assign_visual_types(
        _script(3),
        strategy=strategy,
        max_ai_video_beats=1,
    )

    assert _values(result) == [
        VisualType.VIDEO.value,
        VisualType.IMAGE.value,
        VisualType.IMAGE.value,
    ]


def test_existing_types_are_kept_but_video_is_downgraded_beyond_the_cap() -> None:
    script = Script(
        beats=[
            Beat(text="One.", visual_type="stock"),
            Beat(text="Two.", visual_type="video"),
            Beat(text="Three.", visual_type="ai_video"),
            Beat(text="Four."),
        ]
    )

    result = assign_visual_types(
        script,
        strategy="existing",
        max_ai_video_beats=1,
    )

    assert _values(result) == [
        VisualType.STOCK.value,
        VisualType.VIDEO.value,
        VisualType.IMAGE.value,
        VisualType.IMAGE.value,
    ]


def test_assignment_is_deterministic_idempotent_and_does_not_mutate() -> None:
    script = _script()

    first = assign_visual_types(script)
    second = assign_visual_types(script)
    again = assign_visual_types(first)

    assert first == second == again
    assert _values(script) == [""] * len(script.beats)


def test_a_single_beat_and_an_empty_script_are_supported() -> None:
    beat = assign_visual_types(Beat(text="One."))

    assert beat.visual_type == VisualType.VIDEO.value

    empty = assign_visual_types(Script())
    assert empty.beats == []


def test_the_assigner_exposes_the_types_without_changing_the_script() -> None:
    script = _script(3)
    assigner = VisualAssigner(strategy="video", max_ai_video_beats=1)

    assert assigner.types_for(script) == [
        VisualType.VIDEO.value,
        VisualType.IMAGE.value,
        VisualType.IMAGE.value,
    ]
    assert visual_types(script, strategy="video", max_ai_video_beats=1) == [
        VisualType.VIDEO.value,
        VisualType.IMAGE.value,
        VisualType.IMAGE.value,
    ]
    assert _values(script) == [""] * len(script.beats)


@pytest.mark.parametrize(
    ("value", "strategy"),
    [
        ("hybrid", VisualStrategy.HYBRID),
        ("HYBRID", VisualStrategy.HYBRID),
        ("ai-video", VisualStrategy.VIDEO),
        ("provided", VisualStrategy.EXISTING),
        (None, VisualStrategy.HYBRID),
    ],
)
def test_strategy_normalisation(
    value: str | None,
    strategy: VisualStrategy,
) -> None:
    assert normalise_visual_strategy(value) is strategy


def test_an_unknown_strategy_fails_with_the_allowed_names() -> None:
    with pytest.raises(ValueError, match="unknown visual strategy"):
        assign_visual_types(_script(), strategy="hologram")


@pytest.mark.parametrize("limit", [-1, 1.5, True, "2"])
def test_the_video_limit_must_be_a_non_negative_integer(limit: object) -> None:
    with pytest.raises(ValueError, match="max_ai_video_beats"):
        assign_visual_types(  # type: ignore[call-overload]
            _script(),
            max_ai_video_beats=limit,
        )


def test_an_unsupported_value_is_rejected() -> None:
    with pytest.raises(TypeError):
        assign_visual_types("not a script")  # type: ignore[arg-type]
