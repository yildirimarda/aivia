"""Router selection: first available provider honouring required capabilities."""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from aivia.core.config import CapabilityChainConfig, CapabilityStrategy
from aivia.core.errors import ConfigError, ProviderExhaustedError
from aivia.core.models import ProviderMeta
from aivia.core.router import ProviderRouter, Router, RouteSelection, RouteSkip
from aivia.providers import Provider


class FakeProvider(Provider):
    """A provider with configurable capabilities and a recorded name."""

    def __init__(self, name: str, capabilities: list[str]) -> None:
        super().__init__(ProviderMeta(name=name, capabilities=capabilities))


def make_router(
    *providers: FakeProvider,
    chain: list[str] | None = None,
    strategy: CapabilityStrategy = CapabilityStrategy.FIRST_AVAILABLE,
) -> Router:
    return Router(
        providers={provider.name: provider for provider in providers},
        config=CapabilityChainConfig(strategy=strategy, chain=chain or []),
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
    router = make_router(image, strategy=CapabilityStrategy.CHEAPEST)

    with pytest.raises(ConfigError, match="not implemented: cheapest") as caught:
        router.select("image_gen")

    assert caught.value.details["implemented"] == ["first_available"]


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
