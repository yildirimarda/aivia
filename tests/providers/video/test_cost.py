"""Video cost calculation resolves every price dimension into linked events."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import Engine

from aivia.core.cost import CostTracker
from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.errors import ConfigError
from aivia.media import VideoAudioPricing, VideoPricing
from aivia.providers.protocols.video import (
    VideoRequest,
    VideoResponse,
    VideoResult,
)
from aivia.providers.video import (
    CLIP_UNIT,
    VIDEO_OPERATION,
    VIDEO_UNIT,
    VideoCostCalculator,
    video_cost_event,
    video_cost_events,
)

_CLIP_URL = "https://example.com/media/clip.mp4"


@pytest.fixture
def engine() -> Iterator[Engine]:
    created = create_database_engine(url="sqlite+pysqlite:///:memory:")
    upgrade_database(engine=created)
    try:
        yield created
    finally:
        created.dispose()


@pytest.fixture
def tracker(engine: Engine) -> CostTracker:
    return CostTracker(engine)


def _request(**values: Any) -> VideoRequest:
    return VideoRequest(prompt="a slow pan", **values)


def _pricing(**values: Any) -> VideoPricing:
    return VideoPricing.model_validate(values)


def test_per_second_rate_bills_every_generated_second() -> None:
    calculator = VideoCostCalculator(
        _pricing(per_second=0.10, pricing_version="2026-11")
    )

    event = calculator.cost_event(
        _request(duration_seconds=5.0, count=2),
        provider_name="fal",
        model="video-model",
        run_id="run_1",
    )

    assert event.provider == "fal"
    assert event.model == "video-model"
    assert event.operation == VIDEO_OPERATION
    assert event.amount_usd == pytest.approx(1.0)
    assert event.units == pytest.approx(10.0)
    assert event.unit_kind == VIDEO_UNIT
    assert event.run_id == "run_1"
    assert event.pricing_version == "2026-11"
    assert event.estimated is False
    assert event.metadata["video_count"] == 2
    assert event.metadata["billable_seconds"] == pytest.approx(10.0)
    assert event.metadata["per_second_usd"] == pytest.approx(0.10)
    assert event.metadata["price_source"] == "pricing"
    assert event.metadata["audio"] is False
    assert "tier" not in event.metadata


def test_per_clip_rate_bills_each_generated_clip() -> None:
    calculator = VideoCostCalculator(_pricing(per_clip=0.25))

    event = calculator.cost_event(_request(count=3), provider_name="fal")

    assert event.amount_usd == pytest.approx(0.75)
    assert event.units == pytest.approx(3.0)
    assert event.unit_kind == CLIP_UNIT
    assert event.metadata["video_count"] == 3
    assert event.metadata["per_clip_usd"] == pytest.approx(0.25)


def test_tier_and_audio_prices_resolve_into_one_event() -> None:
    pricing = _pricing(
        per_second=0.10,
        per_clip=0.50,
        resolution_tiers={
            "1080p": {"max_pixels": 1920 * 1080, "per_second": 0.20},
        },
        audio=VideoAudioPricing(per_second=0.02, per_clip=0.05),
    )
    calculator = VideoCostCalculator(pricing)

    event = calculator.cost_event(
        _request(
            width=1080,
            height=1920,
            duration_seconds=4.0,
            generate_audio=True,
        ),
        provider_name="fal",
    )

    assert event.metadata["tier"] == "1080p"
    assert event.metadata["resolution"] == [1080, 1920]
    assert event.metadata["per_second_usd"] == pytest.approx(0.20)
    assert event.metadata["base_cost_usd"] == pytest.approx(1.30)
    assert event.metadata["audio_cost_usd"] == pytest.approx(0.13)
    assert event.amount_usd == pytest.approx(1.43)
    assert event.units == pytest.approx(4.0)
    assert event.unit_kind == VIDEO_UNIT
    assert event.metadata["audio"] is True


def test_a_response_prices_the_clips_the_provider_returned() -> None:
    calculator = VideoCostCalculator(
        _pricing(
            per_second=0.05,
            per_clip=0.10,
            audio=VideoAudioPricing(per_second=0.01),
        )
    )
    request = _request(duration_seconds=2.0)
    response = VideoResponse(
        videos=[
            VideoResult(
                url=_CLIP_URL,
                duration_seconds=6.0,
                width=1280,
                height=720,
                has_audio=True,
            )
        ],
        model="video-model",
        provider="fal",
        request_id="req_1",
    )

    event = calculator.cost_event(request, response=response, run_id="run_1")

    assert event.provider == "fal"
    assert event.model == "video-model"
    assert event.metadata["request_id"] == "req_1"
    assert event.metadata["video_count"] == 1
    assert event.metadata["duration_seconds"] == pytest.approx(6.0)
    assert event.amount_usd == pytest.approx(0.46)
    assert event.units == pytest.approx(6.0)
    assert event.unit_kind == VIDEO_UNIT


def test_split_clips_charge_one_per_clip_fee_per_provider_call() -> None:
    calculator = VideoCostCalculator(_pricing(per_second=0.01, per_clip=0.05))
    requests = [
        _request(duration_seconds=5.0),
        _request(duration_seconds=5.0),
        _request(duration_seconds=5.0),
    ]

    event = calculator.cost_event(requests=requests, provider_name="fal")

    assert event.metadata["clips"] == 3
    assert event.metadata["video_count"] == 3
    assert event.metadata["billable_seconds"] == pytest.approx(15.0)
    assert event.amount_usd == pytest.approx(0.30)
    assert event.units == pytest.approx(15.0)
    assert event.unit_kind == VIDEO_UNIT


def test_an_actual_event_is_linked_to_the_estimate_it_replaces(
    tracker: CostTracker,
) -> None:
    calculator = VideoCostCalculator(_pricing(per_second=0.10, per_clip=0.05))
    request = _request(duration_seconds=10.0)

    estimate = calculator.record_estimate(
        request,
        tracker,
        provider_name="fal",
        model="video-model",
        run_id="run_1",
    )
    actual = calculator.record_actual(
        request,
        tracker,
        provider_name="fal",
        model="video-model",
        run_id="run_1",
    )

    assert estimate.estimated is True
    assert actual.estimated is False
    assert actual.replaces == estimate.id
    assert actual.metadata["estimate_event_id"] == estimate.id
    assert tracker.estimated_total(run_id="run_1") == pytest.approx(0.0)
    assert tracker.actual_total(run_id="run_1") == pytest.approx(1.05)
    assert tracker.total(run_id="run_1") == pytest.approx(1.05)


def test_the_convenience_helpers_build_the_same_events() -> None:
    pricing = _pricing(per_second=0.10)
    request = _request(duration_seconds=2.0)

    event = video_cost_event(
        request,
        pricing=pricing,
        provider="fal",
        pricing_version="2026-12",
    )
    events = video_cost_events(
        [request, request],
        pricing=pricing,
        provider_name="fal",
    )

    assert event.amount_usd == pytest.approx(0.20)
    assert event.pricing_version == "2026-12"
    assert len(events) == 2
    assert all(item.amount_usd == pytest.approx(0.20) for item in events)


def test_invalid_requests_durations_and_tiers_are_rejected() -> None:
    calculator = VideoCostCalculator(_pricing(per_second=0.10))

    with pytest.raises(TypeError, match="VideoRequest"):
        calculator.cost_event("not a request")
    with pytest.raises(ConfigError, match="duration_seconds"):
        calculator.cost_event(_request(), duration_seconds=-1.0)
    with pytest.raises(ConfigError, match="unknown video resolution tier"):
        calculator.cost_event(_request(), tier="8k")
    with pytest.raises(ValueError, match="at least one"):
        calculator.cost_event(requests=[])
