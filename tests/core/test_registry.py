"""Provider registry contract: registration, lookup, loading and failures."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest

from aivia.core.errors import ConfigError
from aivia.core.models import ProviderMeta
from aivia.providers import Provider
from aivia.providers.registry import (
    ProviderRegistry,
    provider_registry,
    registry,
)


class AlphaProvider(Provider):
    default_metadata = ProviderMeta(name="alpha", capabilities=["llm"])


class BetaProvider(Provider):
    default_metadata = ProviderMeta(name="beta", capabilities=["image_gen"])


@pytest.fixture
def custom_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    module = ModuleType("aivia_test_custom_providers")

    class CustomProvider(Provider):
        default_metadata = ProviderMeta(name="custom", capabilities=["tts"])

    class NotAProvider:
        pass

    stub = cast(Any, module)
    stub.CustomProvider = CustomProvider
    stub.NotAProvider = NotAProvider
    stub.Outer = SimpleNamespace(Inner=CustomProvider)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module

def test_registered_providers_are_listed_in_sorted_order() -> None:
    providers = ProviderRegistry()
    providers.register("zulu", AlphaProvider)
    providers.register("alpha", BetaProvider)

    assert providers.names() == ("alpha", "zulu")
    assert providers.keys() == ("alpha", "zulu")
    assert [entry.name for entry in providers] == ["alpha", "zulu"]
    assert [entry.provider for entry in providers.registrations()] == [
        BetaProvider,
        AlphaProvider,
    ]
    assert "alpha" in providers
    assert "missing" not in providers
    assert 1 not in providers
    assert len(providers) == 2


def test_a_provider_can_register_itself_with_the_decorator() -> None:
    providers = ProviderRegistry()

    @providers.register("custom", module="custom_module")
    class RegisteredProvider(Provider):
        pass

    entry = providers.registrations()[0]

    assert providers.get("custom") is RegisteredProvider
    assert entry.provider is RegisteredProvider
    assert entry.module == "custom_module"


def test_register_provider_is_an_explicit_alias() -> None:
    providers = ProviderRegistry()

    entry = providers.register_provider("alpha", AlphaProvider, module="builtin")

    assert entry == providers.registrations()[0]
    assert entry.name == "alpha"
    assert entry.provider is AlphaProvider


def test_registered_names_are_stripped() -> None:
    providers = ProviderRegistry()

    providers.register("  alpha  ", AlphaProvider)

    assert providers.get("alpha") is AlphaProvider
    assert " alpha " in providers
    assert providers.names() == ("alpha",)


def test_duplicate_names_fail_with_the_type_and_the_two_classes() -> None:
    providers = ProviderRegistry()
    providers.register("alpha", AlphaProvider)

    with pytest.raises(ConfigError, match="already registered: alpha") as caught:
        providers.register("alpha", BetaProvider)

    assert caught.value.details["type"] == "alpha"
    assert caught.value.details["existing"] == "AlphaProvider"
    assert caught.value.details["new"] == "BetaProvider"


def test_the_same_class_cannot_claim_one_name_twice() -> None:
    providers = ProviderRegistry()
    providers.register("alpha", AlphaProvider)

    with pytest.raises(ConfigError, match="already registered: alpha"):
        providers.register("alpha", AlphaProvider)


@pytest.mark.parametrize("name", ["", " ", "two words"])
def test_malformed_provider_types_are_rejected(name: str) -> None:
    providers = ProviderRegistry()

    with pytest.raises(ConfigError, match="provider type"):
        providers.register(name, AlphaProvider)


def test_provider_types_must_be_strings() -> None:
    providers = ProviderRegistry()

    with pytest.raises(TypeError, match="provider type must be a string"):
        providers.register(cast(Any, 1), AlphaProvider)


def test_a_registered_type_must_be_a_provider_subclass() -> None:
    providers = ProviderRegistry()

    with pytest.raises(TypeError, match="Provider subclass"):
        providers.register("bad", cast(Any, dict))


def test_unknown_name_lookup_fails_clearly() -> None:
    providers = ProviderRegistry()
    providers.register("alpha", AlphaProvider)

    with pytest.raises(KeyError, match="unknown provider: missing"):
        providers.get("missing")


def test_resolve_returns_a_registered_provider_by_name() -> None:
    providers = ProviderRegistry()
    providers.register("alpha", AlphaProvider)

    assert providers.resolve("alpha") is AlphaProvider


def test_resolve_loads_a_dotted_path_with_a_colon(
    custom_module: ModuleType,
) -> None:
    providers = ProviderRegistry()

    loaded = providers.resolve(f"{custom_module.__name__}:CustomProvider")

    assert loaded.__name__ == "CustomProvider"
    assert issubclass(loaded, Provider)


def test_resolve_loads_a_dotted_path_with_a_dot(custom_module: ModuleType) -> None:
    providers = ProviderRegistry()

    loaded = providers.resolve(f"{custom_module.__name__}.CustomProvider")

    assert loaded.__name__ == "CustomProvider"


def test_resolve_loads_a_nested_attribute(custom_module: ModuleType) -> None:
    providers = ProviderRegistry()

    loaded = providers.resolve(f"{custom_module.__name__}:Outer.Inner")

    assert loaded.__name__ == "CustomProvider"


def test_a_loaded_path_is_cached(custom_module: ModuleType) -> None:
    providers = ProviderRegistry()
    path = f"{custom_module.__name__}:CustomProvider"

    first = providers.load(path)
    second = providers.resolve(path)

    assert first is second


def test_unknown_type_fails_with_the_registered_names() -> None:
    providers = ProviderRegistry()
    providers.register("zulu", AlphaProvider)
    providers.register("alpha", BetaProvider)

    with pytest.raises(ConfigError) as caught:
        providers.resolve("missing")

    message = str(caught.value)
    assert "unknown provider type: missing" in message
    assert "registered: alpha, zulu" in message
    assert caught.value.details["type"] == "missing"
    assert caught.value.details["available"] == ["alpha", "zulu"]


def test_an_unknown_type_names_none_when_the_registry_is_empty() -> None:
    providers = ProviderRegistry()

    with pytest.raises(ConfigError, match="registered: none"):
        providers.resolve("missing")


def test_a_missing_module_fails_with_the_path() -> None:
    providers = ProviderRegistry()
    path = "definitely_missing_aivia_provider_module:Provider"

    with pytest.raises(ConfigError, match="cannot load provider type") as caught:
        providers.load(path)

    assert caught.value.details["type"] == path
    assert (
        caught.value.details["module"]
        == "definitely_missing_aivia_provider_module"
    )


def test_a_missing_attribute_fails_with_the_module_and_attribute(
    custom_module: ModuleType,
) -> None:
    providers = ProviderRegistry()
    path = f"{custom_module.__name__}:Missing"

    with pytest.raises(ConfigError, match="has no attribute") as caught:
        providers.load(path)

    assert caught.value.details["module"] == custom_module.__name__
    assert caught.value.details["attribute"] == "Missing"


def test_a_dotted_path_must_resolve_to_a_provider_subclass(
    custom_module: ModuleType,
) -> None:
    providers = ProviderRegistry()

    with pytest.raises(ConfigError, match="must be a Provider subclass"):
        providers.load(f"{custom_module.__name__}:NotAProvider")


@pytest.mark.parametrize("path", [":Class", "module:", ".Class", "module."])
def test_malformed_dotted_paths_are_rejected(path: str) -> None:
    providers = ProviderRegistry()

    with pytest.raises(ConfigError, match="invalid provider type path"):
        providers.load(path)


def test_resolve_rejects_a_non_string_type() -> None:
    providers = ProviderRegistry()

    with pytest.raises(TypeError, match="provider type must be a string"):
        providers.resolve(cast(Any, 1))


def test_the_shared_registry_is_exposed_under_both_names() -> None:
    assert registry is provider_registry
    assert isinstance(registry, ProviderRegistry)
