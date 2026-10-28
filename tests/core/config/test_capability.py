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
    assert capability.weights == {}
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
    payload: dict[str, object] = {"strategy": value}
    if expected is CapabilityStrategy.WEIGHTED_AB:
        payload["weights"] = {"fal": 1.0}
    capability = CapabilityChainConfig.from_mapping(payload)

    assert capability.strategy == expected


def test_unknown_strategy_names_the_strategy_field() -> None:
    with pytest.raises(ConfigError) as caught:
        CapabilityChainConfig.from_mapping({"strategy": "random"})

    assert "strategy" in str(caught.value)


def test_capability_strategy_aliases_point_at_the_same_enum() -> None:
    assert ChainStrategy is CapabilityStrategy
    assert CapabilityConfig is CapabilityChainConfig


def test_weighted_ab_accepts_per_provider_weights() -> None:
    capability = CapabilityChainConfig.from_mapping(
        {
            "strategy": "weighted_ab",
            "chain": ["fal", "replicate", "local"],
            "weights": {" fal ": 3, "replicate": 2, "local": 1},
        }
    )

    assert capability.strategy == CapabilityStrategy.WEIGHTED_AB
    assert capability.weights == {"fal": 3.0, "replicate": 2.0, "local": 1.0}
    assert capability.provider_weights == capability.weights
    assert capability.weight_sum == 6.0


def test_weights_are_optional_for_every_other_strategy() -> None:
    capability = CapabilityChainConfig.from_mapping(
        {"strategy": "cheapest", "chain": ["fal", "local"]}
    )

    assert capability.weights == {}


def test_non_weighted_strategies_also_accept_weights() -> None:
    capability = CapabilityChainConfig.from_mapping(
        {"strategy": "round_robin", "chain": ["fal"], "weights": {"fal": 0.5}}
    )

    assert capability.weights == {"fal": 0.5}


@pytest.mark.parametrize("weights", [None, {}])
def test_weighted_ab_requires_per_provider_weights(
    weights: dict[str, float] | None,
) -> None:
    payload: dict[str, object] = {
        "strategy": "weighted_ab",
        "chain": ["fal", "local"],
    }
    if weights is not None:
        payload["weights"] = weights

    with pytest.raises(ConfigError) as caught:
        CapabilityChainConfig.from_mapping(payload)

    assert "weighted_ab" in str(caught.value)
    assert "weights" in str(caught.value)


def test_weights_accept_the_provider_weights_alias() -> None:
    capability = CapabilityChainConfig.from_mapping(
        {
            "strategy": "weighted_ab",
            "chain": ["fal", "local"],
            "provider_weights": {"fal": 0.25, "local": 0.75},
        }
    )

    assert capability.weights == {"fal": 0.25, "local": 0.75}
    assert capability.provider_weights == {"fal": 0.25, "local": 0.75}


def test_weights_must_reference_providers_in_the_chain() -> None:
    with pytest.raises(ConfigError) as caught:
        CapabilityChainConfig.from_mapping(
            {
                "strategy": "weighted_ab",
                "chain": ["fal"],
                "weights": {"fal": 1.0, "replicate": 1.0},
            }
        )

    assert "weights" in str(caught.value)
    assert "replicate" in str(caught.value)


@pytest.mark.parametrize("weight", [-0.01, float("inf"), float("-inf"), float("nan")])
def test_weights_must_be_finite_and_non_negative(weight: float) -> None:
    with pytest.raises(ConfigError) as caught:
        CapabilityChainConfig.from_mapping(
            {
                "strategy": "weighted_ab",
                "chain": ["fal"],
                "weights": {"fal": weight},
            }
        )

    assert "weights" in str(caught.value)


def test_weight_keys_must_not_be_empty() -> None:
    with pytest.raises(ConfigError) as caught:
        CapabilityChainConfig.from_mapping(
            {
                "strategy": "weighted_ab",
                "chain": ["fal"],
                "weights": {" ": 1.0},
            }
        )

    assert "weights" in str(caught.value)


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
            "weights": {"fal": 1.0},
            "provider_weights": {"local": 1.0},
        }
    )

    assert capability.strategy == CapabilityStrategy.CHEAPEST
    assert capability.chain == ["fal"]
    assert capability.weights == {"fal": 1.0}


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
    first.weights["fal"] = 1.0
    first.pricing_overrides["fal"] = 1.0

    assert second.chain == []
    assert second.weights == {}
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
