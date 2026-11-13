"""Run and estimate routers are built through the provider factory so configured
options and secret references apply to every provider instance."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import pytest

from aivia.core.config import ProviderConfig, ProvidersConfig
from aivia.core.config.formats import FormatRecipeConfig
from aivia.core.errors import ConfigError
from aivia.core.estimate import Estimator
from aivia.core.models.provider import ProviderMeta, ProviderPrice
from aivia.core.router import Router
from aivia.core.secrets import MissingSecretError, SecretsLoader
from aivia.providers import (
    Provider,
    ProviderRegistry,
    build_provider,
    build_providers,
)
from aivia.providers.provider_factory import ProviderFactory


class RecordingProvider(Provider):
    """A fake adapter that records the constructor arguments the factory passes."""

    default_metadata = ProviderMeta(
        name="recording",
        capabilities=["llm"],
        pricing={"llm": ProviderPrice(amount_usd=0.01)},
    )

    def __init__(
        self,
        *,
        token: str = "",
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
    registry.register("recording", RecordingProvider)
    return registry


def _entry(**values: Any) -> ProviderConfig:
    return ProviderConfig.from_mapping(values)


def _providers(
    entries: dict[str, ProviderConfig],
    *,
    secrets: SecretsLoader | None = None,
    options: Mapping[str, Any] | None = None,
) -> dict[str, Provider]:
    return build_providers(
        entries,
        registry=_registry(),
        secrets=secrets,
        options=options,
    )


def test_run_router_applies_configured_options_and_secrets() -> None:
    config = ProvidersConfig.from_mapping(
        {
            "fast": {
                "type": "recording",
                "options": {"region": "eu", "queue": True},
                "secret_refs": {"token": "RECORDING_TOKEN"},
            },
        }
    )

    providers = build_providers(
        config,
        registry=_registry(),
        secrets=SecretsLoader(environ={"RECORDING_TOKEN": "resolved-secret"}),
    )

    router = Router(providers=providers)
    provider = router.select("llm")

    assert isinstance(provider, RecordingProvider)
    assert provider.name == "fast"
    assert provider.token == "resolved-secret"
    assert provider.region == "eu"
    assert provider.queue is True


def test_estimate_router_applies_configured_options_and_secrets() -> None:
    providers = _providers(
        {
            "fast": _entry(
                type="recording",
                options={"region": "eu", "queue": True},
                secret_refs={"token": "RECORDING_TOKEN"},
            )
        },
        secrets=SecretsLoader(environ={"RECORDING_TOKEN": "resolved-secret"}),
    )

    recipe = FormatRecipeConfig.model_validate({"stages": [{"stage": "write"}]})
    result = Estimator(providers).estimate("sample", recipe)

    stage = result.stages[0]
    assert stage.provider == "fast"
    assert stage.amount_usd == pytest.approx(0.01)

    live = cast(RecordingProvider, providers["fast"])
    assert live.token == "resolved-secret"
    assert live.region == "eu"
    assert live.queue is True


def test_bound_factory_defaults_apply_to_the_estimate() -> None:
    factory = ProviderFactory(
        _registry(),
        secrets=SecretsLoader(environ={"RECORDING_TOKEN": "factory-secret"}),
        options={"queue": True, "region": "eu"},
    )

    providers = factory(
        {"fast": _entry(type="recording", secret_refs={"token": "RECORDING_TOKEN"})}
    )

    recipe = FormatRecipeConfig.model_validate({"stages": [{"stage": "write"}]})
    result = Estimator(providers).estimate("sample", recipe)

    assert result.stages[0].provider == "fast"
    live = cast(RecordingProvider, providers["fast"])
    assert live.token == "factory-secret"
    assert live.region == "eu"
    assert live.queue is True


def test_a_missing_secret_fails_before_the_estimate_is_built() -> None:
    entry = _entry(type="recording", secret_refs={"token": "MISSING_TOKEN"})

    with pytest.raises(MissingSecretError):
        _providers({"fast": entry}, secrets=SecretsLoader(environ={}))


def test_a_resolved_secret_wins_over_a_plaintext_option_in_the_estimate() -> None:
    providers = _providers(
        {
            "fast": _entry(
                type="recording",
                options={"token": "plaintext", "region": "eu"},
                secret_refs={"token": "RECORDING_TOKEN"},
            )
        },
        secrets=SecretsLoader(environ={"RECORDING_TOKEN": "resolved"}),
    )

    recipe = FormatRecipeConfig.model_validate({"stages": [{"stage": "write"}]})
    result = Estimator(providers).estimate("sample", recipe)

    assert result.stages[0].provider == "fast"
    assert cast(RecordingProvider, providers["fast"]).token == "resolved"


def test_building_through_factory_does_not_mutate_configured_options() -> None:
    entry = _entry(
        type="recording",
        options={"region": "eu"},
        secret_refs={"token": "RECORDING_TOKEN"},
    )

    _providers(
        {"fast": entry},
        secrets=SecretsLoader(environ={"RECORDING_TOKEN": "secret"}),
    )

    assert entry.options == {"region": "eu"}
    assert entry.secret_refs == {"token": "RECORDING_TOKEN"}


def test_build_provider_accepts_the_name_before_the_config() -> None:
    entry = _entry(type="recording", secret_refs={"token": "RECORDING_TOKEN"})

    provider = build_provider(
        "fast",
        entry,
        registry=_registry(),
        secrets=SecretsLoader(environ={"RECORDING_TOKEN": "secret"}),
    )

    assert isinstance(provider, RecordingProvider)
    assert provider.name == "fast"
    assert provider.token == "secret"


def test_an_unknown_provider_type_names_the_type_and_registered_types() -> None:
    with pytest.raises(ConfigError) as caught:
        build_provider(
            _entry(type="not_registered"),
            name="fast",
            registry=_registry(),
        )

    assert "not_registered" in str(caught.value)
    assert "recording" in str(caught.value)
