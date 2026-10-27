"""Capability chain configuration: strategy, chain, pricing and clip limits."""

from __future__ import annotations

import pytest
from pydantic import Field

from aivia.core.config import (
    CapabilityChainConfig,
    CapabilityConfig,
    CapabilityStrategy,
    ChainStrategy,
    ConfigModel,
)
from aivia.core.errors import ConfigError


def test_capability_defaults_leave_the_chain_empty_and_the_clip_uncapped() -> None:
    capability = CapabilityChainConfig.from_mapping({})

    assert capability.strategy == CapabilityStrategy.FIRST_AVAILABLE
    assert capability.strategy.value == "first_available"
    assert capability.chain == []
    assert capability.pricing_overrides == {}
    assert capability.max_clip_seconds == 0.0
    assert capability.split_long_requests is False


def test_capability_accepts_strategy_chain_pricing_and_clip_controls() -> None:
    capability = CapabilityChainConfig.from_mapping(
        {
            "strategy": "cheapest",
            "chain": ["fal", "replicate", "local"],
            "pricing_overrides": {"fal": 0.05, "replicate": 0.04},
            "max_clip_seconds": 8,
            "split_long_requests": True,
        }
    )

    assert capability.strategy == CapabilityStrategy.CHEAPEST
    assert capability.chain == ["fal", "replicate", "local"]
    assert capability.pricing_overrides == {"fal": 0.05, "replicate": 0.04}
    assert capability.max_clip_seconds == 8.0
    assert capability.split_long_requests is True


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("first_available", CapabilityStrategy.FIRST_AVAILABLE),
        ("cheapest", CapabilityStrategy.CHEAPEST),
        ("best_quality", CapabilityStrategy.BEST_QUALITY),
        ("round_robin", CapabilityStrategy.ROUND_ROBIN),
        ("weighted_ab", CapabilityStrategy.WEIGHTED_AB),
        (CapabilityStrategy.CHEAPEST, CapabilityStrategy.CHEAPEST),
    ],
)
def test_every_capability_strategy_is_accepted(
    value: str | CapabilityStrategy, expected: CapabilityStrategy
) -> None:
    capability = CapabilityChainConfig.from_mapping({"strategy": value})

    assert capability.strategy == expected


def test_unknown_strategy_names_the_strategy_field() -> None:
    with pytest.raises(ConfigError) as caught:
        CapabilityChainConfig.from_mapping({"strategy": "random"})

    assert "strategy" in str(caught.value)


def test_capability_strategy_aliases_point_at_the_same_enum() -> None:
    assert ChainStrategy is CapabilityStrategy
    assert CapabilityConfig is CapabilityChainConfig


def test_capability_accepts_the_long_and_short_field_aliases() -> None:
    capability = CapabilityChainConfig.from_mapping(
        {
            "provider_chain": ["fal", "local"],
            "chain_strategy": "best_quality",
            "pricing": {"fal": 0.1},
            "per_clip_cap": 6,
            "split": True,
        }
    )

    assert capability.chain == ["fal", "local"]
    assert capability.strategy == CapabilityStrategy.BEST_QUALITY
    assert capability.pricing_overrides == {"fal": 0.1}
    assert capability.per_clip_cap == 6.0
    assert capability.split is True


def test_canonical_fields_win_over_aliases() -> None:
    capability = CapabilityChainConfig.from_mapping(
        {
            "strategy": "cheapest",
            "chain_strategy": "round_robin",
            "chain": ["fal"],
            "providers": ["local"],
        }
    )

    assert capability.strategy == CapabilityStrategy.CHEAPEST
    assert capability.chain == ["fal"]


@pytest.mark.parametrize("entry", ["", "   "])
def test_chain_entries_must_be_non_empty_provider_names(entry: str) -> None:
    with pytest.raises(ConfigError) as caught:
        CapabilityChainConfig.from_mapping({"chain": [entry]})

    assert "chain" in str(caught.value)


def test_chain_entries_are_stripped_and_keep_their_order() -> None:
    capability = CapabilityChainConfig.from_mapping(
        {"chain": [" fal ", "replicate", " local "]}
    )

    assert capability.chain == ["fal", "replicate", "local"]


def test_duplicate_chain_entries_are_rejected() -> None:
    with pytest.raises(ConfigError) as caught:
        CapabilityChainConfig.from_mapping({"chain": ["fal", "fal"]})

    assert "duplicate provider in chain: fal" in str(caught.value)


@pytest.mark.parametrize("price", [-0.01, float("inf"), float("-inf"), float("nan")])
def test_pricing_overrides_must_be_finite_and_non_negative(price: float) -> None:
    with pytest.raises(ConfigError) as caught:
        CapabilityChainConfig.from_mapping({"pricing_overrides": {"fal": price}})

    assert "pricing_overrides" in str(caught.value)


def test_pricing_override_keys_must_not_be_empty() -> None:
    with pytest.raises(ConfigError) as caught:
        CapabilityChainConfig.from_mapping({"pricing_overrides": {" ": 1}})

    assert "pricing_overrides" in str(caught.value)


def test_pricing_override_keys_are_stripped_and_prices_are_floats() -> None:
    capability = CapabilityChainConfig.from_mapping(
        {"pricing_overrides": {" fal ": 2, "local": 0}}
    )

    assert capability.pricing_overrides == {"fal": 2.0, "local": 0.0}


@pytest.mark.parametrize("value", [-1, -0.5])
def test_max_clip_seconds_must_not_be_negative(value: float) -> None:
    with pytest.raises(ConfigError) as caught:
        CapabilityChainConfig.from_mapping({"max_clip_seconds": value})

    assert "max_clip_seconds" in str(caught.value)


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_max_clip_seconds_must_be_finite(value: float) -> None:
    with pytest.raises(ConfigError) as caught:
        CapabilityChainConfig.from_mapping({"max_clip_seconds": value})

    assert "max_clip_seconds" in str(caught.value)


def test_zero_max_clip_seconds_means_no_cap() -> None:
    capability = CapabilityChainConfig.from_mapping({"max_clip_seconds": 0})

    assert capability.max_clip_seconds == 0.0
    assert capability.per_clip_cap == 0.0


def test_unknown_capability_keys_report_the_full_path() -> None:
    with pytest.raises(ConfigError) as caught:
        CapabilityChainConfig.from_mapping({"retries": 3})

    assert "retries" in str(caught.value)


def test_capability_values_are_interpolated_from_the_environment() -> None:
    capability = CapabilityChainConfig.from_mapping(
        {"chain": ["${AIVA_IMAGE_PROVIDER}", "local"]},
        environ={"AIVA_IMAGE_PROVIDER": "fal"},
    )

    assert capability.chain == ["fal", "local"]


def test_capability_defaults_are_independent_between_instances() -> None:
    first = CapabilityChainConfig.from_mapping({})
    second = CapabilityChainConfig.from_mapping({})

    first.chain.append("fal")
    first.pricing_overrides["fal"] = 1.0

    assert second.chain == []
    assert second.pricing_overrides == {}


def test_the_same_capability_model_is_reusable_under_several_sections() -> None:
    class MediaConfig(ConfigModel):
        image_gen: CapabilityChainConfig = Field(default_factory=CapabilityChainConfig)
        video_gen: CapabilityChainConfig = Field(default_factory=CapabilityChainConfig)

    media = MediaConfig.from_mapping(
        {
            "image_gen": {"chain": ["fal"], "strategy": "cheapest"},
            "video_gen": {
                "chain": ["replicate"],
                "max_clip_seconds": 5,
                "split_long_requests": True,
            },
        }
    )

    assert media.image_gen.chain == ["fal"]
    assert media.image_gen.max_clip_seconds == 0.0
    assert media.video_gen.chain == ["replicate"]
    assert media.video_gen.split_long_requests is True
