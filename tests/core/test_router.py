"""Router selection: first available and cheapest provider selection."""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from aivia.core.config import CapabilityChainConfig, CapabilityStrategy
from aivia.core.errors import ConfigError, ProviderExhaustedError
from aivia.core.models import ProviderMeta, ProviderPrice
from aivia.core.router import ProviderRouter, Router, RouteSelection, RouteSkip
from aivia.providers import Provider


class FakeProvider(Provider):
    """A provider with configurable capabilities, prices and a recorded name."""

    def __init__(
        self,
        name: str,
        capabilities: list[str],
        pricing: dict[str, Any] | float | None = None,
    ) -> None:
        super().__init__(
            ProviderMeta.model_validate(
                {
                    "name": name,
                    "capabilities": capabilities,
                    "pricing": {} if pricing is None else pricing,
                }
            )
        )


def make_router(
    *providers: FakeProvider,
    chain: list[str] | None = None,
    strategy: CapabilityStrategy = CapabilityStrategy.FIRST_AVAILABLE,
    pricing_overrides: dict[str, float] | None = None,
) -> Router:
    return Router(
        providers={provider.name: provider for provider in providers},
        config=CapabilityChainConfig(
            strategy=strategy,
            chain=chain or [],
            pricing_overrides=pricing_overrides or {},
        ),
    )


def test_first_available_picks_the_first_chain_provider() -> None:
    first = FakeProvider("first", ["image_gen"])
    second = FakeProvider("second", ["image_gen"])
    router = make_router(first, second, chain=["first", "second"])

    assert router.select("image_gen") is first


def test_chain_order_wins_over_pool_order() -> None:
    first = FakeProvider("first", ["image_gen"])
    second = FakeProvider("second", ["image_gen"])
    router = make_router(first, second, chain=["second", "first"])

    assert router.select("image_gen") is second


def test_providers_without_the_capability_are_skipped() -> None:
    text_only = FakeProvider("text", ["llm"])
    image = FakeProvider("image", ["image_gen"])
    router = make_router(text_only, image, chain=["text", "image"])

    assert router.select("image_gen") is image


def test_every_required_capability_must_be_declared() -> None:
    image_only = FakeProvider("image", ["image_gen"])
    editor = FakeProvider("editor", ["image_gen", "image_edit"])
    router = make_router(image_only, editor, chain=["image", "editor"])

    assert router.select("image_gen", required=["image_edit"]) is editor
    assert [
        p.name for p in router.candidates("image_gen", required=["image_edit"])
    ] == ["editor"]


def test_the_capability_itself_is_always_required() -> None:
    unrelated = FakeProvider("unrelated", ["llm"])
    router = make_router(unrelated)

    with pytest.raises(ProviderExhaustedError):
        router.select("image_gen")


def test_a_closed_provider_is_not_available() -> None:
    closed = FakeProvider("closed", ["image_gen"])
    open_provider = FakeProvider("open", ["image_gen"])
    router = make_router(closed, open_provider, chain=["closed", "open"])
    asyncio.run(closed.close())

    assert router.select("image_gen") is open_provider
    assert [p.name for p in router.candidates("image_gen")] == ["open"]


def test_an_empty_chain_uses_provider_insertion_order() -> None:
    first = FakeProvider("first", ["image_gen"])
    second = FakeProvider("second", ["image_gen"])
    router = make_router(first, second)

    assert router.select("image_gen") is first


def test_per_capability_chains_select_independently() -> None:
    text = FakeProvider("text", ["llm"])
    image = FakeProvider("image", ["image_gen"])
    router = Router(
        providers=[text, image],
        configs={
            "llm": CapabilityChainConfig(chain=["text", "image"]),
            "image_gen": CapabilityChainConfig(chain=["image", "text"]),
        },
    )

    assert router.select("llm") is text
    assert router.select("image_gen") is image


def test_chain_and_strategy_can_be_passed_without_a_config_model() -> None:
    first = FakeProvider("first", ["image_gen"])
    second = FakeProvider("second", ["image_gen"])
    router = Router(
        providers=[first, second],
        strategy="first_available",
        chain=["second", "first"],
    )

    assert router.select("image_gen") is second


def test_cheapest_picks_the_lowest_priced_available_provider() -> None:
    pricey = FakeProvider("pricey", ["image_gen"], pricing={"image_gen": 0.10})
    cheap = FakeProvider("cheap", ["image_gen"], pricing={"image_gen": 0.01})
    router = make_router(
        pricey,
        cheap,
        chain=["pricey", "cheap"],
        strategy=CapabilityStrategy.CHEAPEST,
    )

    assert router.select("image_gen") is cheap


def test_cheapest_uses_chain_order_as_the_tie_breaker() -> None:
    first = FakeProvider("first", ["image_gen"], pricing={"image_gen": 0.05})
    second = FakeProvider("second", ["image_gen"], pricing={"image_gen": 0.05})
    router = make_router(
        first,
        second,
        chain=["second", "first"],
        strategy=CapabilityStrategy.CHEAPEST,
    )

    assert router.select("image_gen") is second


def test_pricing_overrides_beat_provider_metadata() -> None:
    metadata_cheap = FakeProvider(
        "metadata_cheap", ["image_gen"], pricing={"image_gen": 0.01}
    )
    overridden = FakeProvider("overridden", ["image_gen"], pricing={"image_gen": 0.50})
    router = make_router(
        metadata_cheap,
        overridden,
        chain=["metadata_cheap", "overridden"],
        strategy=CapabilityStrategy.CHEAPEST,
        pricing_overrides={"metadata_cheap": 0.75, "overridden": 0.02},
    )

    assert router.select("image_gen") is overridden


def test_pricing_override_uses_the_configured_provider_name() -> None:
    aliased = FakeProvider("metadata_name", ["image_gen"], pricing={"image_gen": 0.50})
    other = FakeProvider("other", ["image_gen"], pricing={"image_gen": 0.10})
    router = Router(
        providers={"alias": aliased, "other": other},
        config=CapabilityChainConfig(
            strategy=CapabilityStrategy.CHEAPEST,
            chain=["alias", "other"],
            pricing_overrides={"alias": 0.01},
        ),
    )

    assert router.select("image_gen") is aliased


def test_a_zero_price_override_marks_a_provider_free() -> None:
    free = FakeProvider("free", ["image_gen"], pricing={"image_gen": 0.50})
    paid = FakeProvider("paid", ["image_gen"], pricing={"image_gen": 0.01})
    router = make_router(
        paid,
        free,
        chain=["paid", "free"],
        strategy=CapabilityStrategy.CHEAPEST,
        pricing_overrides={"free": 0.0},
    )

    assert router.select("image_gen") is free


def test_cheapest_only_considers_open_providers_with_every_capability() -> None:
    closed = FakeProvider("closed", ["image_gen"], pricing={"image_gen": 0.001})
    wrong = FakeProvider("wrong", ["llm"], pricing={"default": 0.001})
    cheap = FakeProvider("cheap", ["image_gen"], pricing={"image_gen": 0.20})
    expensive = FakeProvider("expensive", ["image_gen"], pricing={"image_gen": 0.50})
    router = make_router(
        closed,
        wrong,
        expensive,
        cheap,
        chain=["closed", "wrong", "expensive", "cheap"],
        strategy=CapabilityStrategy.CHEAPEST,
    )
    asyncio.run(closed.close())

    selection = router.select_with_record("image_gen")

    assert selection.provider is cheap
    assert selection.skipped == (
        RouteSkip(provider="closed", reason="closed"),
        RouteSkip(
            provider="wrong",
            reason="missing_capabilities",
            missing=("image_gen",),
        ),
    )


def test_cheapest_uses_default_pricing_when_the_capability_has_no_price() -> None:
    default_priced = FakeProvider(
        "default_priced", ["image_gen"], pricing={"default": 0.02}
    )
    explicit = FakeProvider("explicit", ["image_gen"], pricing={"image_gen": 0.03})
    router = make_router(
        explicit,
        default_priced,
        chain=["explicit", "default_priced"],
        strategy=CapabilityStrategy.CHEAPEST,
    )

    assert router.select("image_gen") is default_priced


def test_cheapest_normalises_metadata_prices_to_one_unit() -> None:
    bulk = FakeProvider(
        "bulk",
        ["image_gen"],
        pricing={"image_gen": ProviderPrice(amount_usd=0.10, units_per_price=100)},
    )
    single = FakeProvider("single", ["image_gen"], pricing={"image_gen": 0.01})
    router = make_router(
        single,
        bulk,
        chain=["single", "bulk"],
        strategy=CapabilityStrategy.CHEAPEST,
    )

    assert router.select("image_gen") is bulk


def test_providers_without_a_price_sort_after_priced_candidates() -> None:
    unpriced = FakeProvider("unpriced", ["image_gen"])
    priced = FakeProvider("priced", ["image_gen"], pricing={"image_gen": 5.0})
    router = make_router(
        unpriced,
        priced,
        chain=["unpriced", "priced"],
        strategy=CapabilityStrategy.CHEAPEST,
    )

    assert router.select("image_gen") is priced


def test_cheapest_falls_back_to_chain_order_when_no_prices_are_known() -> None:
    first = FakeProvider("first", ["image_gen"])
    second = FakeProvider("second", ["image_gen"])
    router = make_router(
        first,
        second,
        chain=["second", "first"],
        strategy=CapabilityStrategy.CHEAPEST,
    )

    assert router.select("image_gen") is second


def test_cheapest_can_be_requested_for_a_single_call() -> None:
    pricey = FakeProvider("pricey", ["image_gen"], pricing={"image_gen": 0.10})
    cheap = FakeProvider("cheap", ["image_gen"], pricing={"image_gen": 0.01})
    router = make_router(pricey, cheap, chain=["pricey", "cheap"])

    selection = router.select_with_record(
        "image_gen", strategy=CapabilityStrategy.CHEAPEST
    )

    assert selection.provider is cheap
    assert selection.strategy is CapabilityStrategy.CHEAPEST
    assert selection.chain == ("pricey", "cheap")
    assert selection.skipped == ()
    assert router.last_selection is selection


def test_a_chain_naming_an_unknown_provider_is_a_configuration_error() -> None:
    image = FakeProvider("image", ["image_gen"])
    router = make_router(image, chain=["missing"])

    with pytest.raises(ConfigError, match="unknown providers: missing") as caught:
        router.select("image_gen")

    assert caught.value.details["unknown"] == ["missing"]
    assert caught.value.details["available"] == ["image"]


def test_exhaustion_reports_every_skipped_provider() -> None:
    text = FakeProvider("text", ["llm"])
    closed = FakeProvider("closed", ["image_gen"])
    router = make_router(text, closed, chain=["text", "closed"])
    asyncio.run(closed.close())

    with pytest.raises(ProviderExhaustedError) as caught:
        router.select("image_gen")

    details = caught.value.details
    assert details["capability"] == "image_gen"
    assert details["required"] == ["image_gen"]
    assert details["chain"] == ["text", "closed"]
    assert details["skipped"] == [
        {
            "provider": "text",
            "reason": "missing_capabilities",
            "missing": ["image_gen"],
        },
        {"provider": "closed", "reason": "closed"},
    ]


def test_selection_is_recorded_with_the_walked_chain() -> None:
    text = FakeProvider("text", ["llm"])
    image = FakeProvider("image", ["image_gen", "image_edit"])
    router = make_router(text, image, chain=["text", "image"])

    selection = router.select_with_record("image_gen", required=["image_edit"])

    assert isinstance(selection, RouteSelection)
    assert selection.provider is image
    assert selection.name == "image"
    assert selection.capability == "image_gen"
    assert selection.strategy is CapabilityStrategy.FIRST_AVAILABLE
    assert selection.required == ("image_gen", "image_edit")
    assert selection.chain == ("text", "image")
    assert selection.skipped == (
        RouteSkip(
            provider="text",
            reason="missing_capabilities",
            missing=("image_gen", "image_edit"),
        ),
    )
    assert router.last_selection is selection


def test_an_unimplemented_strategy_fails_clearly() -> None:
    image = FakeProvider("image", ["image_gen"])
    router = make_router(image, strategy=CapabilityStrategy.BEST_QUALITY)

    with pytest.raises(ConfigError, match="not implemented: best_quality") as caught:
        router.select("image_gen")

    assert caught.value.details["implemented"] == ["first_available", "cheapest"]


def test_unknown_strategy_names_are_rejected() -> None:
    image = FakeProvider("image", ["image_gen"])
    router = make_router(image)

    with pytest.raises(ConfigError, match="unknown router strategy"):
        router.select("image_gen", strategy="random")


def test_router_aliases_and_provider_names() -> None:
    image = FakeProvider("image", ["image_gen"])
    router = Router(providers=[image])

    assert ProviderRouter is Router
    assert router.provider_names == ("image",)
    assert router.select_provider("image_gen") is image
    assert router.available("image_gen") == (image,)


def test_provider_instances_must_have_unique_names() -> None:
    one = FakeProvider("same", ["image_gen"])
    two = FakeProvider("same", ["image_gen"])

    with pytest.raises(ConfigError, match="duplicate provider name: same"):
        Router(providers=[one, two])


def test_provider_instances_must_have_a_name() -> None:
    unnamed = FakeProvider("", ["image_gen"])

    with pytest.raises(ConfigError, match="non-empty name"):
        Router(providers=[unnamed])


def test_provider_pool_rejects_non_providers() -> None:
    with pytest.raises(TypeError, match="Provider instances"):
        Router(providers=cast(Any, {"bad": object()}))
