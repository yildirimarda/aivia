"""Reproducing a reference structure matches scene count and durations within tolerance.

A reference analysis carries a measured structure of ordered scenes, each with a
duration. Reproducing it generates one clip per scene, probes each clip's actual
duration and compares scene count and durations against the reference within a
tolerance, regenerating once on deviation.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

import pytest

from aivia.core.errors import ValidationError
from aivia.media.video_service import VideoOutcome
from aivia.providers.protocols.video import VideoRequest, VideoResponse, VideoResult
from aivia.reference import (
    DEFAULT_STRUCTURE_MATCH_TOLERANCE_SECONDS,
    ReferenceAnalysis,
    SceneSpec,
    StructureReproductionResult,
    StructureSpec,
    reproduce_structure,
)


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine)


class FakeGenerator:
    """Generates one clip per scene with durations scheduled per generation."""

    def __init__(
        self,
        scene_count: int,
        durations_per_generation: list[list[float]],
    ) -> None:
        self.scene_count = scene_count
        self.durations_per_generation = durations_per_generation
        self.calls: list[VideoRequest] = []
        self.call_count = 0

    async def __call__(self, request: VideoRequest, token: Any) -> VideoOutcome:
        self.call_count += 1
        self.calls.append(request)
        generation = (self.call_count - 1) // self.scene_count
        index_in_generation = (self.call_count - 1) % self.scene_count
        schedules = self.durations_per_generation
        duration = schedules[min(generation, len(schedules) - 1)][index_in_generation]
        result = VideoResult(data=b"fake-clip", duration_seconds=duration)
        response = VideoResponse(videos=[result], model="model", provider="provider")
        return VideoOutcome(request=request, response=response)


async def probe_duration(outcome: VideoOutcome) -> float | None:
    return outcome.response.videos[0].duration_seconds


def _outcome(duration: float) -> VideoOutcome:
    result = VideoResult(data=b"fake-clip", duration_seconds=duration)
    response = VideoResponse(videos=[result], model="model", provider="provider")
    return VideoOutcome(request=VideoRequest(prompt="unused"), response=response)


async def _fixed_outcome(request: VideoRequest, token: Any) -> VideoOutcome:
    return _outcome(1.0)


def _reference(scene_durations: list[float | None]) -> ReferenceAnalysis:
    scenes = [
        SceneSpec(
            index=index,
            description=f"scene {index}",
            duration_seconds=duration,
        )
        for index, duration in enumerate(scene_durations)
    ]
    return ReferenceAnalysis(structure=StructureSpec(scenes=scenes))


def _request() -> VideoRequest:
    return VideoRequest(prompt="A cinematic scene")


def test_reproduces_matching_structure_on_first_generation() -> None:
    generator = FakeGenerator(3, [[2.0, 2.0, 1.0]])
    result = _run(
        reproduce_structure(
            reference=_reference([2.0, 2.0, 1.0]),
            request=_request(),
            generate_one=generator,
            probe_duration=probe_duration,
        )
    )
    assert isinstance(result, StructureReproductionResult)
    assert result.matched is True
    assert result.regenerations == 0
    assert result.deviation is None
    assert result.scene_count == 3
    assert [scene.duration_seconds for scene in result.scenes] == [2.0, 2.0, 1.0]


def test_duration_exactly_at_tolerance_is_a_match() -> None:
    generator = FakeGenerator(1, [[2.5]])
    result = _run(
        reproduce_structure(
            reference=_reference([2.0]),
            request=_request(),
            generate_one=generator,
            probe_duration=probe_duration,
            tolerance_seconds=0.5,
        )
    )
    assert result.matched is True
    assert result.regenerations == 0


def test_duration_beyond_tolerance_is_a_deviation() -> None:
    generator = FakeGenerator(1, [[3.0]])
    result = _run(
        reproduce_structure(
            reference=_reference([2.0]),
            request=_request(),
            generate_one=generator,
            probe_duration=probe_duration,
            tolerance_seconds=0.5,
            max_regenerations=0,
        )
    )
    assert result.matched is False
    assert result.regenerations == 0
    assert result.deviation is not None
    assert "0.500s" in result.deviation


def test_deviation_regenerates_once_and_matches() -> None:
    generator = FakeGenerator(2, [[1.0, 1.0], [2.0, 3.0]])
    result = _run(
        reproduce_structure(
            reference=_reference([2.0, 3.0]),
            request=_request(),
            generate_one=generator,
            probe_duration=probe_duration,
        )
    )
    assert result.matched is True
    assert result.regenerations == 1
    assert result.deviation is None
    assert result.scene_count == 2


def test_deviation_persists_after_regeneration() -> None:
    generator = FakeGenerator(2, [[1.0, 1.0], [1.0, 1.0]])
    result = _run(
        reproduce_structure(
            reference=_reference([2.0, 3.0]),
            request=_request(),
            generate_one=generator,
            probe_duration=probe_duration,
        )
    )
    assert result.matched is False
    assert result.regenerations == 1
    assert result.deviation is not None
    assert result.scene_count == 2


def test_zero_tolerance_requires_exact_duration() -> None:
    generator = FakeGenerator(1, [[2.0001], [2.0]])
    result = _run(
        reproduce_structure(
            reference=_reference([2.0]),
            request=_request(),
            generate_one=generator,
            probe_duration=probe_duration,
            tolerance_seconds=0.0,
        )
    )
    assert result.matched is True
    assert result.regenerations == 1


def test_max_regenerations_zero_does_not_regenerate() -> None:
    generator = FakeGenerator(1, [[3.0]])
    result = _run(
        reproduce_structure(
            reference=_reference([2.0]),
            request=_request(),
            generate_one=generator,
            probe_duration=probe_duration,
            max_regenerations=0,
        )
    )
    assert result.matched is False
    assert result.regenerations == 0
    assert generator.call_count == 1


def test_empty_reference_matches_without_generating() -> None:
    generator = FakeGenerator(0, [[]])
    result = _run(
        reproduce_structure(
            reference=_reference([]),
            request=_request(),
            generate_one=generator,
            probe_duration=probe_duration,
        )
    )
    assert result.matched is True
    assert result.regenerations == 0
    assert result.scene_count == 0
    assert generator.call_count == 0


def test_scenes_without_duration_are_counted_but_not_duration_matched() -> None:
    generator = FakeGenerator(2, [[1.0, 1.0]])
    result = _run(
        reproduce_structure(
            reference=_reference([None, None]),
            request=_request(),
            generate_one=generator,
            probe_duration=probe_duration,
        )
    )
    assert result.matched is True
    assert result.regenerations == 0
    assert result.scene_count == 2


def test_screens_each_scene_prompt_and_passes_token() -> None:
    screened: list[str] = []
    tokens: list[Any] = []

    async def generate_one(request: VideoRequest, token: Any) -> VideoOutcome:
        tokens.append(token)
        result = VideoResult(data=b"x", duration_seconds=2.0)
        response = VideoResponse(videos=[result], model="m", provider="p")
        return VideoOutcome(request=request, response=response)

    async def screen(prompt: str) -> str:
        screened.append(prompt)
        return "screened-token"

    _run(
        reproduce_structure(
            reference=_reference([2.0, 2.0]),
            request=_request(),
            generate_one=generate_one,
            probe_duration=probe_duration,
            screen_prompt=screen,
            token="raw-token",
        )
    )
    assert tokens == ["screened-token", "screened-token"]
    assert any("scene 1" in prompt for prompt in screened)
    assert any("scene 2" in prompt for prompt in screened)


def test_rejects_none_reference() -> None:
    with pytest.raises(ValidationError):
        _run(
            reproduce_structure(
                reference=None,
                request=_request(),
                generate_one=_fixed_outcome,
                probe_duration=probe_duration,
            )
        )


def test_rejects_negative_tolerance() -> None:
    with pytest.raises(ValueError):
        _run(
            reproduce_structure(
                reference=_reference([2.0]),
                request=_request(),
                generate_one=_fixed_outcome,
                probe_duration=probe_duration,
                tolerance_seconds=-1.0,
            )
        )


def test_default_tolerance_is_positive() -> None:
    assert DEFAULT_STRUCTURE_MATCH_TOLERANCE_SECONDS > 0.0
