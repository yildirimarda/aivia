"""Provider factory: build router instances from provider config and secrets."""

from __future__ import annotations

from typing import Any, cast

import pytest

from aivia.core.config import ProviderConfig, ProvidersConfig
from aivia.core.errors import ConfigError
from aivia.core.models import ProviderMeta
from aivia.core.router import Router
from aivia.core.secrets import MissingSecretError, SecretsLoader
from aivia.providers import (
    Provider,
    ProviderRegistry,
    build_provider,
    build_provider_instances,
    build_providers,
    build_providers_from_config,
    create_provider,
    create_providers,
    provider_from_config,
    provider_instances_from_config,
    providers_from_config,
)
from aivia.providers.mock import MockProvider
from aivia.providers.provider_factory import ProviderFactory
from aivia.providers.registry import ProviderRegistry as RegistryClass


class CapturingProvider(Provider):
    """A fake adapter that records the constructor arguments the factory passes."""

    default_metadata = ProviderMeta(name="capturing", capabilities=["llm"])

    def __init__(
        self,
        *,
        token: str,
        region: str = "",
        queue: bool = False,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.token = token
        self.region = region
        self.queue = queue


def _registry() -> ProviderRegistry:
    registry = ProviderRegistry()
    registry.register("capturing", CapturingProvider)
    return registry


def _entry(**values: Any) -> ProviderConfig:
    return ProviderConfig.from_mapping(values)


def test_build_provider_passes_options_and_resolves_secret_refs() -> None:
    entry = _entry(
        type="capturing",
        options={"region": "eu", "queue": True},
        secret_refs={"token": "CAPTURE_TOKEN"},
    )
    secrets = SecretsLoader(environ={"CAPTURE_TOKEN": "resolved-secret"})

    provider = build_provider(
        entry,
        name="fast",
        registry=_registry(),
        secrets=secrets,
    )

    assert isinstance(provider, CapturingProvider)
    assert provider.name == "fast"
    assert provider.token == "resolved-secret"
    assert provider.region == "eu"
    assert provider.queue is True


def test_build_providers_preserves_names_and_skips_disabled_entries() -> None:
    config = ProvidersConfig.from_mapping(
        {
            "fast": {
                "type": "capturing",
                "options": {"region": "eu"},
                "secret_refs": {"token": "CAPTURE_TOKEN"},
            },
            "slow": {
                "type": "capturing",
                "options": {"region": "us"},
                "secret_refs": {"token": "CAPTURE_TOKEN"},
            },
            "off": {"type": "capturing", "enabled": False},
        }
    )

    providers = build_providers(
        config,
        registry=_registry(),
        secrets=SecretsLoader(environ={"CAPTURE_TOKEN": "secret"}),
    )

    assert tuple(providers) == ("fast", "slow")
    assert providers["fast"].name == "fast"
    assert providers["slow"].name == "slow"
    assert cast(CapturingProvider, providers["fast"]).region == "eu"
    assert cast(CapturingProvider, providers["slow"]).region == "us"


def test_build_providers_accepts_a_raw_mapping_or_no_section() -> None:
    config = {
        "capturing": {
            "type": "capturing",
            "secret_refs": {"token": "CAPTURE_TOKEN"},
        }
    }

    providers = build_providers(
        config,
        registry=_registry(),
        secrets=SecretsLoader(environ={"CAPTURE_TOKEN": "secret"}),
    )

    assert tuple(providers) == ("capturing",)
    assert build_providers(None, registry=_registry()) == {}


def test_a_resolved_secret_wins_over_a_plaintext_option() -> None:
    entry = _entry(
        type="capturing",
        options={"token": "plaintext"},
        secret_refs={"token": "CAPTURE_TOKEN"},
    )

    provider = build_provider(
        entry,
        name="fast",
        registry=_registry(),
        secrets=SecretsLoader(environ={"CAPTURE_TOKEN": "resolved"}),
    )

    assert cast(CapturingProvider, provider).token == "resolved"


def test_a_missing_secret_fails_before_the_provider_is_created() -> None:
    entry = _entry(type="capturing", secret_refs={"token": "MISSING_TOKEN"})

    with pytest.raises(MissingSecretError) as caught:
        build_provider(
            entry,
            name="fast",
            registry=_registry(),
            secrets=SecretsLoader(environ={}),
        )

    assert "MISSING_TOKEN" in str(caught.value)
    assert "plaintext" not in str(caught.value)


def test_an_unknown_provider_type_names_the_type_and_registered_types() -> None:
    with pytest.raises(ConfigError) as caught:
        build_provider(
            _entry(type="not_registered"),
            name="fast",
            registry=_registry(),
        )

    assert "not_registered" in str(caught.value)
    assert "capturing" in str(caught.value)


def test_a_dotted_path_loads_a_provider_without_registering_it() -> None:
    provider = build_provider(
        _entry(type="aivia.providers.mock:MockProvider", options={"seed": 3}),
        name="loaded",
        registry=ProviderRegistry(),
    )

    assert isinstance(provider, MockProvider)
    assert provider.name == "loaded"
    assert provider._seed == 3


def test_built_providers_wire_directly_into_a_router_chain() -> None:
    config = ProvidersConfig.from_mapping(
        {
            "text": {"type": "mock", "options": {"cost_usd": 0.0}},
            "backup": {"type": "mock"},
        }
    )

    providers = build_providers(config)
    router = Router(providers=providers, chain=["text", "backup"])

    assert router.select("llm").name == "text"


def test_build_provider_accepts_the_name_before_the_config() -> None:
    entry = _entry(type="capturing", secret_refs={"token": "CAPTURE_TOKEN"})

    provider = build_provider(
        "fast",
        entry,
        registry=_registry(),
        secrets=SecretsLoader(environ={"CAPTURE_TOKEN": "secret"}),
    )

    assert provider.name == "fast"
    assert cast(CapturingProvider, provider).token == "secret"


def test_building_does_not_mutate_the_configured_options() -> None:
    entry = _entry(
        type="capturing",
        options={"region": "eu"},
        secret_refs={"token": "CAPTURE_TOKEN"},
    )

    build_provider(
        entry,
        name="fast",
        registry=_registry(),
        secrets=SecretsLoader(environ={"CAPTURE_TOKEN": "secret"}),
    )

    assert entry.options == {"region": "eu"}
    assert entry.secret_refs == {"token": "CAPTURE_TOKEN"}


def test_the_factory_aliases_build_the_same_instances() -> None:
    entry = _entry(type="capturing", secret_refs={"token": "CAPTURE_TOKEN"})
    secrets = SecretsLoader(environ={"CAPTURE_TOKEN": "secret"})
    registry = _registry()

    assert create_provider is build_provider
    assert provider_from_config is build_provider
    assert create_providers is build_providers
    assert providers_from_config is build_providers
    assert build_provider_instances is build_providers
    assert provider_instances_from_config is build_providers
    assert build_providers_from_config is build_providers
    assert (
        create_provider(entry, name="fast", registry=registry, secrets=secrets).name
        == "fast"
    )


def test_a_bound_factory_reuses_its_registry_secrets_and_options() -> None:
    entry = _entry(type="capturing", secret_refs={"token": "CAPTURE_TOKEN"})
    factory = ProviderFactory(
        _registry(),
        secrets=SecretsLoader(environ={"CAPTURE_TOKEN": "secret"}),
        options={"queue": True},
    )

    provider = factory.build(entry, name="fast")
    providers = factory({"fast": entry})

    assert cast(CapturingProvider, provider).queue is True
    assert tuple(providers) == ("fast",)
    assert cast(CapturingProvider, providers["fast"]).token == "secret"


def test_invalid_factory_arguments_are_rejected() -> None:
    with pytest.raises(TypeError, match="ProviderConfig"):
        build_provider(cast(Any, object()))
    with pytest.raises(TypeError, match="ProvidersConfig"):
        build_providers(cast(Any, object()))
    with pytest.raises(TypeError, match="ProviderRegistry"):
        build_providers({}, registry=cast(Any, object()))
    with pytest.raises(TypeError, match="SecretsLoader"):
        build_providers({}, secrets=cast(Any, object()))
    with pytest.raises(TypeError, match="options"):
        build_providers({}, options=cast(Any, object()))


def test_the_config_helpers_build_the_same_instances() -> None:
    entry = _entry(type="capturing", secret_refs={"token": "CAPTURE_TOKEN"})
    config = ProvidersConfig.from_mapping({"fast": entry})
    secrets = SecretsLoader(environ={"CAPTURE_TOKEN": "secret"})

    providers = config.build(registry=_registry(), secrets=secrets)
    provider = entry.build("fast", registry=_registry(), secrets=secrets)

    assert tuple(providers) == ("fast",)
    assert providers["fast"].name == "fast"
    assert provider.name == "fast"
    built = config.build_providers(registry=_registry(), secrets=secrets)
    assert built["fast"].name == "fast"


def test_a_provider_name_must_be_non_empty() -> None:
    with pytest.raises(ConfigError, match="provider name"):
        build_provider(
            _entry(type="capturing", secret_refs={"token": "CAPTURE_TOKEN"}),
            name="   ",
            registry=_registry(),
            secrets=SecretsLoader(environ={"CAPTURE_TOKEN": "secret"}),
        )


def test_registry_alias_stays_importable() -> None:
    assert RegistryClass is ProviderRegistry
