"""Each beat becomes one concrete, deterministic visual request."""

from __future__ import annotations

import pytest

from aivia.core.models.script import (
    Beat,
    ImagePrompt,
    MediaKind,
    MotionPrompt,
    Script,
    VisualType,
)
from aivia.visuals import (
    DEFAULT_MAX_AI_VIDEO_BEATS,
    DEFAULT_VISUAL_STRATEGY,
    VisualPlan,
    VisualRequest,
    VisualStrategyResolver,
    build_visual_plan,
    resolve_visual_request,
    resolve_visual_requests,
    resolve_visual_strategy,
    visual_requests,
)


def _script(count: int = 5) -> Script:
    return Script(
        title="Bees",
        beats=[
            Beat(
                index=index,
                text=f"Fact {index + 1}.",
                media_prompts=[
                    ImagePrompt(
                        prompt=f"Still prompt {index + 1}.",
                        negative_prompt="no text",
                        style="documentary",
                        aspect_ratio="9:16",
                        seed=10 + index,
                    ),
                    MotionPrompt(
                        prompt=f"Motion prompt {index + 1}.",
                        duration_seconds=2.5,
                    ),
                ],
            )
            for index in range(count)
        ],
    )


def _types(requests: list[VisualRequest]) -> list[str]:
    return [request.visual_type.value for request in requests]


def test_the_default_hybrid_resolves_one_request_per_beat() -> None:
    requests = resolve_visual_requests(_script())

    assert DEFAULT_VISUAL_STRATEGY == "hybrid"
    assert DEFAULT_MAX_AI_VIDEO_BEATS == 2
    assert len(requests) == 5
    assert _types(requests) == [
        VisualType.VIDEO.value,
        VisualType.VIDEO.value,
        VisualType.IMAGE.value,
        VisualType.IMAGE.value,
        VisualType.IMAGE.value,
    ]
    assert [request.index for request in requests] == [0, 1, 2, 3, 4]
    assert all(request.strategy == "hybrid" for request in requests)


def test_each_request_uses_the_prompt_that_matches_its_kind() -> None:
    requests = resolve_visual_requests(_script(3), max_ai_video_beats=1)

    assert requests[0].kind is MediaKind.VIDEO
    assert requests[0].prompt == "Motion prompt 1."
    assert requests[0].duration_seconds == 2.5
    assert requests[0].aspect_ratio == "9:16"
    assert requests[0].seed == 10

    assert requests[1].kind is MediaKind.IMAGE
    assert requests[1].prompt == "Still prompt 2."
    assert requests[1].negative_prompt == "no text"
    assert requests[1].style == "documentary"
    assert requests[1].seed == 11


def test_the_video_cap_is_hard_across_a_sequence_and_a_script() -> None:
    beats = _script(4).beats

    from_beats = resolve_visual_requests(
        beats,
        strategy="video",
        max_ai_video_beats=1,
    )
    from_script = resolve_visual_requests(
        Script(beats=beats),
        strategy="video",
        max_ai_video_beats=1,
    )

    assert _types(from_beats) == [
        VisualType.VIDEO.value,
        VisualType.IMAGE.value,
        VisualType.IMAGE.value,
        VisualType.IMAGE.value,
    ]
    assert from_beats == from_script


@pytest.mark.parametrize(
    ("strategy", "visual", "kind"),
    [
        ("image", VisualType.IMAGE, MediaKind.IMAGE),
        ("ai_image", VisualType.IMAGE, MediaKind.IMAGE),
        ("stock", VisualType.STOCK, MediaKind.STOCK),
        ("still", VisualType.STILL, MediaKind.STILL),
        ("text", VisualType.TEXT, MediaKind.IMAGE),
        ("none", VisualType.NONE, MediaKind.IMAGE),
    ],
)
def test_fixed_strategies_resolve_every_beat(
    strategy: str,
    visual: VisualType,
    kind: MediaKind,
) -> None:
    requests = resolve_visual_requests(
        _script(3),
        strategy=strategy,
        max_ai_video_beats=0,
    )

    assert _types(requests) == [visual.value] * 3
    assert [request.kind for request in requests] == [kind] * 3


def test_stock_requests_carry_a_search_query() -> None:
    requests = resolve_visual_requests(_script(2), strategy="stock")

    assert all(request.is_stock for request in requests)
    assert [request.query for request in requests] == [
        "Still prompt 1.",
        "Still prompt 2.",
    ]
    assert all(request.prompt == request.query for request in requests)


def test_existing_types_are_kept_but_video_is_capped() -> None:
    script = Script(
        beats=[
            Beat(text="One.", visual_type="stock"),
            Beat(text="Two.", visual_type="video"),
            Beat(text="Three.", visual_type="ai_video"),
            Beat(text="Four."),
        ]
    )

    requests = resolve_visual_requests(
        script,
        strategy="existing",
        max_ai_video_beats=1,
    )

    assert _types(requests) == [
        VisualType.STOCK.value,
        VisualType.VIDEO.value,
        VisualType.IMAGE.value,
        VisualType.IMAGE.value,
    ]
    assert requests[0].query == "One."


def test_missing_prompts_fall_back_to_the_beat_text() -> None:
    requests = resolve_visual_requests(
        [Beat(text="A bee lands on a flower.")],
        strategy="image",
    )

    assert requests[0].prompt == "A bee lands on a flower."
    assert requests[0].query == ""


def test_caller_fallbacks_only_fill_missing_aspect_and_duration() -> None:
    bare = resolve_visual_requests(
        [Beat(text="One.")],
        strategy="image",
        aspect_ratio="16:9",
        duration_seconds=4.0,
    )
    explicit = resolve_visual_requests(
        [Beat(text="One.", duration_seconds=2.0)],
        strategy="image",
        aspect_ratio="16:9",
        duration_seconds=4.0,
    )

    assert bare[0].aspect_ratio == "16:9"
    assert bare[0].duration_seconds == 4.0
    assert explicit[0].duration_seconds == 2.0


def test_a_single_beat_resolves_to_one_request() -> None:
    beat = Beat(
        text="One.",
        media_prompts=[ImagePrompt(prompt="A still.")],
    )

    request = resolve_visual_request(beat, strategy="image")
    dispatched = resolve_visual_strategy(beat, strategy="image")

    assert request.visual_type is VisualType.IMAGE
    assert request.prompt == "A still."
    assert dispatched == request


def test_resolution_is_deterministic_idempotent_and_does_not_mutate() -> None:
    script = _script()

    first = resolve_visual_requests(script)
    second = resolve_visual_requests(script)
    again = resolve_visual_requests(script)

    assert first == second == again
    assert [beat.visual_type for beat in script.beats] == [""] * len(script.beats)


def test_the_resolver_exposes_requests_types_and_a_plan() -> None:
    script = _script(3)
    resolver = VisualStrategyResolver(strategy="video", max_ai_video_beats=1)

    assert resolver.types_for(script) == [
        VisualType.VIDEO.value,
        VisualType.IMAGE.value,
        VisualType.IMAGE.value,
    ]
    assert _types(resolver.requests_for(script)) == [
        VisualType.VIDEO.value,
        VisualType.IMAGE.value,
        VisualType.IMAGE.value,
    ]
    assert resolver.resolve(script) == resolver.requests_for(script)
    assert resolver.resolve(script.beats[0]) == resolver.resolve_beat(script.beats[0])

    plan = build_visual_plan(script, strategy="video", max_ai_video_beats=1)
    assert isinstance(plan, VisualPlan)
    assert plan.requests == resolver.requests_for(script)
    assert plan.strategy.value == "video"
    assert plan.max_ai_video_beats == 1


def test_the_request_serializes_canonically() -> None:
    request = resolve_visual_requests(_script(1))[0]

    data = request.to_dict()
    assert data["visual_type"] == "video"
    assert data["kind"] == "video"
    assert request.to_json() == request.to_json()
    assert '"prompt":"Motion prompt 1."' in request.to_json()


def test_alias_functions_return_the_same_requests() -> None:
    script = _script(2)

    assert visual_requests(script) == resolve_visual_requests(script)


def test_an_unknown_strategy_fails_with_the_allowed_names() -> None:
    with pytest.raises(ValueError, match="unknown visual strategy"):
        resolve_visual_requests(_script(), strategy="hologram")


@pytest.mark.parametrize("limit", [-1, 1.5, True, "2"])
def test_the_video_limit_must_be_a_non_negative_integer(limit: object) -> None:
    with pytest.raises(ValueError, match="max_ai_video_beats"):
        resolve_visual_requests(
            _script(),
            max_ai_video_beats=limit,  # type: ignore[arg-type]
        )


def test_an_unsupported_value_is_rejected() -> None:
    with pytest.raises(TypeError):
        resolve_visual_requests("not a script")  # type: ignore[arg-type]
