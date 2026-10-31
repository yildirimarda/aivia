"""Video requests are clamped, warned and optionally split into priced clips."""

from __future__ import annotations

import pytest

from aivia.core.config.capability import CapabilityChainConfig
from aivia.core.errors import InvalidRequestError
from aivia.providers.media import (
    AdaptationAction,
    MediaConstraints,
    UnadaptableRequestError,
)
from aivia.providers.protocols.video import VideoRequest
from aivia.providers.video import (
    VideoAdaptation,
    adapt_video_request,
    split_duration,
)


def test_video_request_is_clamped_with_recorded_warnings() -> None:
    constraints = MediaConstraints(
        min_duration_seconds=5.0,
        max_duration_seconds=10.0,
        max_count=2,
    )
    request = VideoRequest(prompt="a slow pan", duration_seconds=15.0, count=3)

    result = adapt_video_request(request, constraints)

    assert isinstance(result, VideoAdaptation)
    assert result.split is False
    assert result.clip_count == 1
    assert result.request.duration_seconds == 10.0
    assert result.request.count == 2
    assert result.request.prompt == "a slow pan"
    assert [warning.action for warning in result.warnings] == [
        AdaptationAction.CLAMPED_DURATION,
        AdaptationAction.CLAMPED_COUNT,
    ]
    assert result.warnings[0].requested == 15.0
    assert result.warnings[0].applied == 10.0
    assert request.duration_seconds == 15.0
    assert request.count == 3


def test_short_duration_is_raised_to_the_provider_minimum() -> None:
    result = adapt_video_request(
        VideoRequest(prompt="a slow pan", duration_seconds=1.0),
        MediaConstraints(min_duration_seconds=5.0, max_duration_seconds=10.0),
    )

    assert result.request.duration_seconds == 5.0
    assert result.warnings[0].action is AdaptationAction.CLAMPED_DURATION


def test_a_long_request_is_split_only_when_enabled() -> None:
    constraints = MediaConstraints(min_duration_seconds=4.0, max_duration_seconds=10.0)
    request = VideoRequest(prompt="a slow pan", duration_seconds=25.0)

    clamped = adapt_video_request(request, constraints)
    assert clamped.split is False
    assert clamped.request.duration_seconds == 10.0
    assert clamped.warnings[0].action is AdaptationAction.CLAMPED_DURATION

    split = adapt_video_request(request, constraints, split=True)
    assert split.split is True
    assert split.clip_count == 3
    assert [clip.duration_seconds for clip in split.clips] == pytest.approx(
        [25.0 / 3.0] * 3
    )
    assert split.total_duration_seconds == pytest.approx(25.0)
    assert all(
        clip.duration_seconds is not None and clip.duration_seconds <= 10.0
        for clip in split.clips
    )
    assert all(clip.prompt == "a slow pan" for clip in split.clips)
    warning = split.warnings[-1]
    assert warning.action is AdaptationAction.SPLIT_DURATION
    assert warning.option == "duration_seconds"
    assert warning.requested == 25.0
    assert warning.applied == pytest.approx((25.0 / 3.0,) * 3)
    assert "3 clips" in warning.message


def test_split_uses_the_tightest_cap_and_records_combined_cost() -> None:
    constraints = MediaConstraints(max_duration_seconds=20.0)
    config = CapabilityChainConfig(
        max_clip_seconds=8.0,
        split_long_requests=True,
    )
    request = VideoRequest(prompt="a wave", duration_seconds=20.0, count=2)

    result = adapt_video_request(
        request,
        constraints,
        config=config,
        cost_per_second=0.01,
        cost_per_clip=0.05,
    )

    assert result.clip_count == 3
    assert all(clip.duration_seconds is not None for clip in result.clips)
    assert all(
        clip.duration_seconds is not None and clip.duration_seconds <= 8.0
        for clip in result.clips
    )
    assert result.video_count == 6
    expected = (0.05 * 6) + (0.01 * 20.0 * 2)
    assert result.combined_cost_usd == pytest.approx(expected)
    assert result.cost_usd == result.combined_cost_usd
    assert result.to_dict()["clips"] == 3
    assert result.to_dict()["videos"] == 6
    assert result.to_dict()["combined_cost_usd"] == pytest.approx(expected)


def test_explicit_split_arguments_win_over_the_capability_config() -> None:
    config = CapabilityChainConfig(
        max_clip_seconds=5.0,
        split_long_requests=False,
    )
    constraints = MediaConstraints(max_duration_seconds=20.0)
    request = VideoRequest(prompt="a wave", duration_seconds=12.0)

    result = adapt_video_request(
        request,
        constraints,
        config=config,
        split=True,
        max_clip_seconds=6.0,
    )

    assert result.split is True
    assert result.clip_count == 2
    assert all(
        clip.duration_seconds is not None and clip.duration_seconds <= 6.0
        for clip in result.clips
    )


def test_a_request_within_the_cap_is_not_split_and_has_no_warning() -> None:
    result = adapt_video_request(
        VideoRequest(prompt="a wave", duration_seconds=8.0),
        MediaConstraints(max_duration_seconds=10.0),
        split=True,
    )

    assert result.split is False
    assert result.request.duration_seconds == 8.0
    assert result.changed is False


def test_split_without_a_cap_falls_back_to_clamping() -> None:
    result = adapt_video_request(
        VideoRequest(prompt="a wave", duration_seconds=8.0),
        MediaConstraints(),
        split=True,
    )

    assert result.split is False
    assert result.request.duration_seconds == 8.0
    assert result.changed is False


def test_split_is_unadaptable_when_each_clip_would_be_too_short() -> None:
    with pytest.raises(UnadaptableRequestError) as caught:
        adapt_video_request(
            VideoRequest(prompt="a wave", duration_seconds=12.0),
            MediaConstraints(min_duration_seconds=8.0, max_duration_seconds=10.0),
            split=True,
        )

    assert isinstance(caught.value, InvalidRequestError)
    assert caught.value.details["clips"] == 2
    assert caught.value.details["min_clip_seconds"] == 8.0


def test_split_duration_distributes_evenly_without_losing_time() -> None:
    assert split_duration(25.0, max_clip_seconds=10.0) == pytest.approx(
        (25.0 / 3.0,) * 3
    )
    assert split_duration(10.0, max_clip_seconds=10.0) == (10.0,)
    assert split_duration(5.0, max_clip_seconds=10.0, min_clip_seconds=5.0) == (5.0,)

    with pytest.raises(ValueError, match="positive"):
        split_duration(0.0, max_clip_seconds=10.0)
    with pytest.raises(ValueError, match="positive"):
        split_duration(10.0, max_clip_seconds=0.0)
    with pytest.raises(UnadaptableRequestError, match="exceeds the per-clip cap"):
        split_duration(10.0, max_clip_seconds=5.0, min_clip_seconds=8.0)


def test_costs_must_be_finite_and_non_negative() -> None:
    request = VideoRequest(prompt="a wave", duration_seconds=5.0)
    constraints = MediaConstraints()

    with pytest.raises(ValueError, match="non-negative"):
        adapt_video_request(request, constraints, cost_per_second=-0.01)
    with pytest.raises(ValueError, match="non-negative"):
        adapt_video_request(request, constraints, cost_per_clip=float("nan"))
