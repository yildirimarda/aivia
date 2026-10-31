"""Estimating video cost from per-second, per-clip, tier and audio rates."""

from __future__ import annotations

from typing import Any

import pytest

from aivia.core.errors import ConfigError
from aivia.media import (
    ResolutionTier,
    VideoAudioPricing,
    VideoCostEstimate,
    VideoPricing,
    VideoPricingEstimator,
    VideoResolutionTier,
    estimate_video_adaptation,
    estimate_video_cost,
    estimate_video_requests,
)
from aivia.providers.media import MediaConstraints
from aivia.providers.protocols.video import VideoRequest
from aivia.providers.video import adapt_video_request


def _request(**values: Any) -> VideoRequest:
    return VideoRequest(prompt="a slow pan", **values)


def _pricing(**values: Any) -> VideoPricing:
    return VideoPricing.model_validate(values)


def test_per_second_price_scales_with_duration_and_variations() -> None:
    pricing = _pricing(per_second=0.10)
    request = _request(duration_seconds=5.0, count=2)

    estimate = estimate_video_cost(request, pricing)

    assert isinstance(estimate, VideoCostEstimate)
    assert estimate.amount_usd == pytest.approx(1.0)
    assert estimate.base_cost_usd == pytest.approx(1.0)
    assert estimate.audio_cost_usd == 0.0
    assert estimate.per_second_usd == pytest.approx(0.10)
    assert estimate.per_clip_usd == 0.0
    assert estimate.duration_seconds == pytest.approx(5.0)
    assert estimate.billable_seconds == pytest.approx(10.0)
    assert estimate.videos == 2
    assert estimate.clips == 1
    assert estimate.tier is None
    assert estimate.audio is False
    assert estimate.currency == "USD"
    assert estimate.total_usd == estimate.amount_usd


def test_per_clip_price_applies_to_every_generated_video() -> None:
    pricing = _pricing(per_clip=0.25)

    estimate = estimate_video_cost(_request(count=3), pricing)

    assert estimate.amount_usd == pytest.approx(0.75)
    assert estimate.videos == 3
    assert estimate.per_clip_usd == pytest.approx(0.25)


def test_per_second_and_per_clip_prices_combine() -> None:
    pricing = _pricing(per_second=0.02, per_clip=0.50)

    estimate = estimate_video_cost(
        _request(duration_seconds=10.0, count=2),
        pricing,
    )

    assert estimate.base_cost_usd == pytest.approx(1.40)
    assert estimate.amount_usd == pytest.approx(1.40)
    assert estimate.to_dict()["amount_usd"] == pytest.approx(1.40)
    assert estimate.to_dict()["videos"] == 2


def test_resolution_tiers_select_the_smallest_matching_rate() -> None:
    pricing = _pricing(
        per_second=0.10,
        resolution_tiers={
            "480p": {"max_pixels": 640 * 480, "per_second": 0.04},
            "720p": {"max_pixels": 1280 * 720, "per_second": 0.06},
            "1080p": {"max_pixels": 1920 * 1080, "per_second": 0.10},
            "4k": {"max_pixels": 3840 * 2160, "per_second": 0.25},
        },
    )

    cases = (
        (640, 360, "480p", 0.04),
        (1280, 720, "720p", 0.06),
        (1080, 1920, "1080p", 0.10),
        (3840, 2160, "4k", 0.25),
    )
    for width, height, tier, per_second in cases:
        estimate = estimate_video_cost(
            _request(width=width, height=height, duration_seconds=2.0),
            pricing,
        )
        assert estimate.tier == tier
        assert estimate.per_second_usd == pytest.approx(per_second)
        assert estimate.amount_usd == pytest.approx(per_second * 2.0)
        assert estimate.resolution == (width, height)


def test_a_frame_above_every_tier_uses_the_largest_tier() -> None:
    pricing = _pricing(
        per_second=0.10,
        resolution_tiers={
            "720p": {"per_second": 0.06},
            "4k": {"per_second": 0.25},
        },
    )

    estimate = estimate_video_cost(
        _request(width=7680, height=4320, duration_seconds=1.0),
        pricing,
    )

    assert estimate.tier == "4k"
    assert estimate.amount_usd == pytest.approx(0.25)


def test_tier_names_derive_pixel_budgets_and_respect_orientation() -> None:
    pricing = _pricing(
        per_second=0.10,
        resolution_tiers={"720p": {}, "1080p": {}},
    )

    portrait = estimate_video_cost(
        _request(width=1080, height=1920, duration_seconds=1.0),
        pricing,
    )
    landscape = estimate_video_cost(
        _request(width=1920, height=1080, duration_seconds=1.0),
        pricing,
    )

    assert portrait.tier == "1080p"
    assert landscape.tier == "1080p"
    assert pricing.tier("720p").max_pixels == 1280 * 720
    assert pricing.tier("1080p").max_pixels == 1920 * 1080


def test_tier_multiplier_and_explicit_rates_override_the_base() -> None:
    pricing = _pricing(
        per_second=0.50,
        per_clip=0.50,
        resolution_tiers={
            "1080p": VideoResolutionTier(
                max_pixels=1920 * 1080,
                per_second=0.02,
                per_clip=0.10,
                multiplier=2.0,
            )
        },
    )

    estimate = estimate_video_cost(
        _request(width=1080, height=1920, duration_seconds=5.0, count=2),
        pricing,
    )

    assert estimate.per_second_usd == pytest.approx(0.04)
    assert estimate.per_clip_usd == pytest.approx(0.20)
    assert estimate.amount_usd == pytest.approx((0.04 * 5.0 + 0.20) * 2)


def test_audio_is_charged_only_when_the_request_asks_for_it() -> None:
    pricing = _pricing(
        per_second=0.10,
        audio=VideoAudioPricing(per_second=0.02, per_clip=0.05),
    )

    silent = estimate_video_cost(
        _request(duration_seconds=10.0),
        pricing,
    )
    sounding = estimate_video_cost(
        _request(duration_seconds=10.0, generate_audio=True),
        pricing,
    )

    assert silent.audio is False
    assert silent.audio_cost_usd == 0.0
    assert silent.amount_usd == pytest.approx(1.0)
    assert sounding.audio is True
    assert sounding.audio_cost_usd == pytest.approx(0.25)
    assert sounding.audio_per_second_usd == pytest.approx(0.02)
    assert sounding.audio_per_clip_usd == pytest.approx(0.05)
    assert sounding.amount_usd == pytest.approx(1.25)


def test_audio_multiplier_scales_the_surcharge() -> None:
    pricing = _pricing(
        per_second=0.10,
        audio=VideoAudioPricing(
            per_second=0.02,
            per_clip=0.05,
            multiplier=2.0,
        ),
    )

    estimate = estimate_video_cost(
        _request(duration_seconds=10.0, generate_audio=True),
        pricing,
    )

    assert estimate.audio_cost_usd == pytest.approx(0.50)
    assert estimate.amount_usd == pytest.approx(1.50)


def test_prompt_audio_uses_its_own_option_price() -> None:
    pricing = _pricing(
        per_second=0.10,
        audio=VideoAudioPricing(per_second=0.01),
        audio_options={
            "prompt": VideoAudioPricing(per_second=0.05, per_clip=0.20),
        },
    )

    native = estimate_video_cost(
        _request(duration_seconds=4.0, generate_audio=True),
        pricing,
    )
    prompted = estimate_video_cost(
        _request(
            duration_seconds=4.0,
            generate_audio=True,
            audio_prompt="a quiet jazz trio",
        ),
        pricing,
    )

    assert native.audio_cost_usd == pytest.approx(0.04)
    assert prompted.audio_cost_usd == pytest.approx(0.40)
    assert prompted.audio_per_second_usd == pytest.approx(0.05)
    assert prompted.audio_per_clip_usd == pytest.approx(0.20)


def test_a_tier_can_override_the_audio_price() -> None:
    pricing = _pricing(
        per_second=0.10,
        audio=VideoAudioPricing(per_second=0.01),
        resolution_tiers={
            "1080p": VideoResolutionTier(
                max_pixels=1920 * 1080,
                audio=VideoAudioPricing(per_second=0.10),
            )
        },
    )

    estimate = estimate_video_cost(
        _request(
            width=1080,
            height=1920,
            duration_seconds=2.0,
            generate_audio=True,
        ),
        pricing,
    )

    assert estimate.tier == "1080p"
    assert estimate.audio_per_second_usd == pytest.approx(0.10)
    assert estimate.audio_cost_usd == pytest.approx(0.20)
    assert estimate.amount_usd == pytest.approx(0.40)


def test_the_default_tier_is_used_without_dimensions() -> None:
    pricing = _pricing(
        per_second=0.10,
        default_resolution_tier="1080p",
        resolution_tiers={
            "720p": {"per_second": 0.05},
            "1080p": {"per_second": 0.20},
        },
    )

    estimate = estimate_video_cost(_request(duration_seconds=2.0), pricing)

    assert estimate.tier == "1080p"
    assert estimate.amount_usd == pytest.approx(0.40)


def test_an_explicit_tier_wins_and_an_unknown_tier_is_rejected() -> None:
    pricing = _pricing(
        per_second=0.10,
        resolution_tiers={
            "720p": {"per_second": 0.05},
            "4k": {"per_second": 0.30},
        },
    )

    estimate = estimate_video_cost(
        _request(width=640, height=360, duration_seconds=1.0),
        pricing,
        tier="4k",
    )

    assert estimate.tier == "4k"
    assert estimate.amount_usd == pytest.approx(0.30)
    with pytest.raises(ConfigError, match="unknown video resolution tier"):
        estimate_video_cost(_request(), pricing, tier="8k")


def test_a_short_side_bound_matches_portrait_and_landscape() -> None:
    pricing = _pricing(
        per_second=0.10,
        resolution_tiers={
            "1080p": ResolutionTier(max_short_side=1080, per_second=0.20),
            "4k": ResolutionTier(max_short_side=2160, per_second=0.40),
        },
    )

    portrait = estimate_video_cost(
        _request(width=1080, height=1920, duration_seconds=1.0),
        pricing,
    )
    landscape = estimate_video_cost(
        _request(width=1920, height=1080, duration_seconds=1.0),
        pricing,
    )

    assert portrait.tier == "1080p"
    assert landscape.tier == "1080p"
    assert portrait.amount_usd == pytest.approx(0.20)


def test_split_clips_are_summed_with_one_clip_fee_per_call() -> None:
    pricing = _pricing(per_second=0.01, per_clip=0.05)
    requests = [
        _request(duration_seconds=25.0 / 3.0, count=2),
        _request(duration_seconds=25.0 / 3.0, count=2),
        _request(duration_seconds=25.0 / 3.0, count=2),
    ]

    estimate = estimate_video_requests(requests, pricing)

    assert estimate.clips == 3
    assert estimate.videos == 6
    assert estimate.duration_seconds == pytest.approx(25.0)
    assert estimate.billable_seconds == pytest.approx(50.0)
    assert estimate.amount_usd == pytest.approx((0.01 * 25.0 + 0.05 * 3) * 2)
    assert estimate.to_dict()["clips"] == 3


def test_the_estimator_matches_the_adaptation_combined_cost() -> None:
    adaptation = adapt_video_request(
        _request(duration_seconds=20.0, count=2),
        MediaConstraints(max_duration_seconds=8.0),
        split=True,
        cost_per_second=0.01,
        cost_per_clip=0.05,
    )
    pricing = _pricing(per_second=0.01, per_clip=0.05)

    estimate = estimate_video_adaptation(adaptation, pricing)

    assert estimate.clips == adaptation.clip_count
    assert estimate.videos == adaptation.video_count
    assert estimate.duration_seconds == pytest.approx(adaptation.total_duration_seconds)
    assert estimate.amount_usd == pytest.approx(adaptation.combined_cost_usd)


def test_missing_duration_only_charges_the_per_clip_price() -> None:
    pricing = _pricing(per_second=0.10, per_clip=0.20)

    estimate = estimate_video_cost(_request(), pricing)

    assert estimate.duration_seconds == 0.0
    assert estimate.amount_usd == pytest.approx(0.20)


def test_currency_and_pricing_version_are_copied_onto_the_estimate() -> None:
    pricing = _pricing(
        per_second=0.10,
        currency="eur",
        pricing_version="2026-10",
    )

    estimate = estimate_video_cost(
        _request(duration_seconds=1.0),
        pricing,
    )

    assert estimate.currency == "EUR"
    assert estimate.pricing_version == "2026-10"


def test_a_mapping_is_accepted_as_pricing_input() -> None:
    estimator = VideoPricingEstimator(
        {
            "per_second": 0.10,
            "resolution_tiers": {"720p": {"per_second": 0.05}},
        }
    )

    estimate = estimator.estimate(
        _request(width=1280, height=720, duration_seconds=2.0)
    )

    assert estimator.pricing.tier("720p").max_pixels == 1280 * 720
    assert estimate.tier == "720p"
    assert estimate.amount_usd == pytest.approx(0.10)


def test_invalid_rates_and_unknown_default_tiers_are_rejected() -> None:
    with pytest.raises(ValueError):
        _pricing(per_second=-0.01)
    with pytest.raises(ValueError):
        _pricing(per_clip=float("nan"))
    with pytest.raises(ValueError):
        VideoAudioPricing(per_second=-1.0)
    with pytest.raises(ValueError):
        VideoResolutionTier(max_pixels=1920 * 1080, multiplier=-1.0)
    with pytest.raises(ValueError, match="unknown tier"):
        _pricing(
            default_resolution_tier="missing",
            resolution_tiers={"1080p": {}},
        )


def test_an_estimate_without_requests_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        estimate_video_requests([], _pricing(per_clip=1.0))
