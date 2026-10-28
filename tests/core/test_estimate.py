"""Estimator selection honours capability chains and router strategies."""

from __future__ import annotations

import pytest

from aivia.core.config import CapabilityChainConfig, CapabilityStrategy
from aivia.core.config.formats import FormatRecipeConfig
from aivia.core.errors import ConfigError
from aivia.core.estimate import Estimator
from aivia.core.models.provider import ProviderMeta, ProviderPrice
from aivia.core.router import MissingQualityRankWarning


def provider(
    name: str,
    *,
    capability: str = "llm",
    price: float | None = None,
    quality_rank: int | None = None,
) -> ProviderMeta:
    return ProviderMeta(
        name=name,
        capabilities=[capability],
        pricing={} if price is None else {capability: ProviderPrice(amount_usd=price)},
        quality_rank=quality_rank,
    )


def recipe(*stages: dict[str, object]) -> FormatRecipeConfig:
    return FormatRecipeConfig.model_validate({"stages": list(stages)})


def test_first_available_walks_the_configured_chain_not_the_price() -> None:
    providers = {
        "dear": provider("dear", price=0.50),
        "cheap": provider("cheap", price=0.01),
    }

    result = Estimator(
        providers,
        config=CapabilityChainConfig(
            strategy=CapabilityStrategy.FIRST_AVAILABLE,
            chain=["dear", "cheap"],
        ),
    ).estimate("sample", recipe({"stage": "write"}))

    stage = result.stages[0]
    assert stage.provider == "dear"
    assert stage.amount_usd == pytest.approx(0.50)


def test_cheapest_only_considers_the_configured_chain() -> None:
    providers = {
        "outside": provider("outside", price=0.001),
        "dear": provider("dear", price=0.50),
        "cheap": provider("cheap", price=0.05),
    }

    result = Estimator(
        providers,
        config=CapabilityChainConfig(
            strategy=CapabilityStrategy.CHEAPEST,
            chain=["dear", "cheap"],
        ),
    ).estimate("sample", recipe({"stage": "write"}))

    assert result.stages[0].provider == "cheap"
    assert result.stages[0].amount_usd == pytest.approx(0.05)


def test_cheapest_uses_chain_order_as_the_tie_breaker() -> None:
    providers = {
        "first": provider("first", price=0.05),
        "second": provider("second", price=0.05),
    }

    result = Estimator(
        providers,
        config=CapabilityChainConfig(
            strategy=CapabilityStrategy.CHEAPEST,
            chain=["second", "first"],
        ),
    ).estimate("sample", recipe({"stage": "write"}))

    assert result.stages[0].provider == "second"


def test_per_capability_chains_select_independently() -> None:
    providers = {
        "text": provider("text", capability="llm", price=0.10),
        "speech": provider("speech", capability="tts", price=0.20),
        "image": provider("image", capability="image_gen", price=0.30),
    }
    configs = {
        "llm": CapabilityChainConfig(
            strategy=CapabilityStrategy.FIRST_AVAILABLE,
            chain=["text"],
        ),
        "tts": CapabilityChainConfig(
            strategy=CapabilityStrategy.FIRST_AVAILABLE,
            chain=["speech"],
        ),
        "image_gen": CapabilityChainConfig(
            strategy=CapabilityStrategy.FIRST_AVAILABLE,
            chain=["image"],
        ),
    }

    result = Estimator(providers, configs=configs).estimate(
        "sample",
        recipe(
            {"stage": "write"},
            {"stage": "voice"},
            {"stage": "visuals", "options": {"strategy": "image"}},
        ),
    )

    assert [stage.provider for stage in result.stages] == [
        "text",
        "speech",
        "image",
    ]
    assert result.total_usd == pytest.approx(0.60)


def test_config_pricing_overrides_are_used_by_selection_and_the_estimate() -> None:
    providers = {
        "metadata": provider("metadata", price=0.01),
        "overridden": provider("overridden", price=0.50),
    }

    result = Estimator(
        providers,
        config=CapabilityChainConfig(
            strategy=CapabilityStrategy.CHEAPEST,
            chain=["metadata", "overridden"],
            pricing_overrides={"metadata": 0.75, "overridden": 0.02},
        ),
    ).estimate("sample", recipe({"stage": "write"}))

    assert result.stages[0].provider == "overridden"
    assert result.stages[0].amount_usd == pytest.approx(0.02)


def test_best_quality_picks_the_highest_rank_and_warns_for_unranked() -> None:
    providers = {
        "good": provider("good", price=0.01, quality_rank=2),
        "best": provider("best", price=0.50, quality_rank=5),
        "unranked": provider("unranked", price=0.001),
    }

    with pytest.warns(MissingQualityRankWarning):
        result = Estimator(
            providers,
            config=CapabilityChainConfig(
                strategy=CapabilityStrategy.BEST_QUALITY,
                chain=["good", "best", "unranked"],
            ),
        ).estimate("sample", recipe({"stage": "write"}))

    assert result.stages[0].provider == "best"
    assert result.stages[0].amount_usd == pytest.approx(0.50)


def test_round_robin_rotates_between_stages_of_the_same_capability() -> None:
    providers = {
        "first": provider("first", price=0.01),
        "second": provider("second", price=0.02),
    }

    result = Estimator(
        providers,
        config=CapabilityChainConfig(
            strategy=CapabilityStrategy.ROUND_ROBIN,
            chain=["first", "second"],
        ),
    ).estimate(
        "sample",
        recipe(
            {"stage": "write"},
            {"stage": "rewrite"},
            {"stage": "review"},
        ),
    )

    assert [stage.provider for stage in result.stages] == [
        "first",
        "second",
        "first",
    ]


def test_weighted_ab_is_reproducible_with_a_seed() -> None:
    providers = {
        "heavy": provider("heavy", price=0.10),
        "light": provider("light", price=0.20),
    }
    config = CapabilityChainConfig(
        strategy=CapabilityStrategy.WEIGHTED_AB,
        chain=["heavy", "light"],
        weights={"heavy": 9.0, "light": 1.0},
    )

    first = Estimator(providers, config=config, seed=7).estimate(
        "sample",
        recipe({"stage": "write"}, {"stage": "rewrite"}),
    )
    second = Estimator(providers, config=config, seed=7).estimate(
        "sample",
        recipe({"stage": "write"}, {"stage": "rewrite"}),
    )

    assert [stage.provider for stage in first.stages] == [
        stage.provider for stage in second.stages
    ]
    assert {stage.provider for stage in first.stages} <= {"heavy", "light"}


def test_an_unknown_provider_in_the_chain_is_a_config_error() -> None:
    estimator = Estimator(
        {"known": provider("known", price=0.01)},
        config=CapabilityChainConfig(
            strategy=CapabilityStrategy.FIRST_AVAILABLE,
            chain=["missing"],
        ),
    )

    with pytest.raises(ConfigError, match="unknown providers"):
        estimator.estimate("sample", recipe({"stage": "write"}))


def test_without_a_strategy_the_estimator_still_prefers_the_cheapest() -> None:
    providers = {
        "dear": provider("dear", price=0.20),
        "cheap": provider("cheap", price=0.10),
    }

    result = Estimator(providers).estimate("sample", recipe({"stage": "write"}))

    assert result.stages[0].provider == "cheap"
