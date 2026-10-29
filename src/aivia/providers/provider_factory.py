"""Build provider instances from the ``providers`` configuration section.

A capability chain names provider instances, but the router needs live objects.
This factory turns each configured entry into one adapter instance: ``type``
selects a registered provider or a dotted import path, ``options`` are passed to
its constructor unchanged, and every ``secret_refs`` entry resolves a secret
through :class:`~aivia.core.secrets.SecretsLoader` before construction. Secrets
are referenced by name in configuration and never stored in the config file.

The configured name becomes the provider's name, so a capability chain can use
the same name it declares without any hand wiring. Disabled entries are skipped.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aivia.core.config.providers import ProviderConfig, ProvidersConfig
from aivia.core.errors import ConfigError
from aivia.core.secrets import SecretsLoader
from aivia.providers.base import Provider
from aivia.providers.registry import ProviderRegistry
from aivia.providers.registry import registry as provider_registry

__all__ = [
    "ProviderFactory",
    "build_provider",
    "build_provider_instances",
    "build_providers",
    "build_providers_from_config",
    "create_provider",
    "create_providers",
    "provider_from_config",
    "provider_instances_from_config",
    "providers_from_config",
]


def build_provider(
    config: ProviderConfig | Mapping[str, Any] | str | None = None,
    name: str | ProviderConfig | Mapping[str, Any] | None = None,
    *,
    registry: ProviderRegistry | None = None,
    secrets: SecretsLoader | None = None,
    options: Mapping[str, Any] | None = None,
    **overrides: Any,
) -> Provider:
    """Build one provider from *config*.

    *config* is a validated :class:`ProviderConfig`, a mapping such as one
    ``providers.<name>`` entry, or a bare type string. The first two positional
    arguments may be given as ``(config, name)`` or ``(name, config)``. *options*
    and *overrides* are applied after the configured options, and a resolved
    secret always wins over a plaintext value for the same option.

    *registry* defaults to the shared provider registry. *secrets* defaults to
    the environment/file/keyring loader when a secret reference is declared.
    """
    config, name = _split_config_and_name(config, name)
    entry = _provider_config(config)
    source = _registry(registry)
    loader = _secrets_loader(secrets)
    values = _provider_options(entry, options, overrides, loader)
    if name is not None:
        values["name"] = _name(name)
    elif "name" not in values and isinstance(config, str):
        values["name"] = _name(config)

    provider_type = source.resolve(entry.type)
    return provider_type(**values)


def build_providers(
    config: ProvidersConfig | Mapping[str, Any] | None = None,
    *,
    registry: ProviderRegistry | None = None,
    secrets: SecretsLoader | None = None,
    options: Mapping[str, Any] | None = None,
    **overrides: Any,
) -> dict[str, Provider]:
    """Build one provider per enabled entry in *config*, keyed by configured name.

    A missing section produces an empty mapping. Disabled entries are skipped so
    a capability chain that still names one never receives a live adapter. The
    section is never mutated.
    """
    section = _providers_config(config)
    source = _registry(registry)
    loader = _secrets_loader(secrets)
    common = _mapping(options, "options")
    providers: dict[str, Provider] = {}
    for name, entry in section.enabled_providers().items():
        values = _provider_options(entry, common, overrides, loader)
        values["name"] = _name(name)
        provider_type = source.resolve(entry.type)
        providers[name] = provider_type(**values)
    return providers


class ProviderFactory:
    """A reusable builder bound to one registry, secret loader and defaults."""

    def __init__(
        self,
        registry: ProviderRegistry | None = None,
        *,
        secrets: SecretsLoader | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> None:
        self._registry = _registry(registry)
        self._secrets = _secrets_loader(secrets)
        self._options = _mapping(options, "options")

    def build(
        self,
        config: ProviderConfig | Mapping[str, Any] | str | None = None,
        name: str | None = None,
        **overrides: Any,
    ) -> Provider:
        """Build one provider with this factory's defaults."""
        return build_provider(
            config,
            name,
            registry=self._registry,
            secrets=self._secrets,
            options=self._options,
            **overrides,
        )

    def build_all(
        self,
        config: ProvidersConfig | Mapping[str, Any] | None = None,
        **overrides: Any,
    ) -> dict[str, Provider]:
        """Build every enabled provider with this factory's defaults."""
        return build_providers(
            config,
            registry=self._registry,
            secrets=self._secrets,
            options=self._options,
            **overrides,
        )

    def __call__(
        self,
        config: ProvidersConfig | Mapping[str, Any] | None = None,
        **overrides: Any,
    ) -> dict[str, Provider]:
        """Alias for :meth:`build_all`."""
        return self.build_all(config, **overrides)


def _split_config_and_name(
    config: ProviderConfig | Mapping[str, Any] | str | None,
    name: str | ProviderConfig | Mapping[str, Any] | None,
) -> tuple[ProviderConfig | Mapping[str, Any] | str | None, str | None]:
    """Accept ``(config, name)`` and ``(name, config)`` positional orders."""
    if (
        isinstance(config, str)
        and not isinstance(name, str)
        and isinstance(name, (ProviderConfig, Mapping))
    ):
        return name, config
    return config, name if isinstance(name, str) else None


def _provider_config(
    value: ProviderConfig | Mapping[str, Any] | str | None,
) -> ProviderConfig:
    if isinstance(value, ProviderConfig):
        return value
    if isinstance(value, str):
        return ProviderConfig(type=value)
    if isinstance(value, Mapping):
        return ProviderConfig.from_mapping(value)
    raise TypeError("provider config must be a ProviderConfig, mapping or type string")


def _providers_config(
    value: ProvidersConfig | Mapping[str, Any] | None,
) -> ProvidersConfig:
    if value is None:
        return ProvidersConfig.from_mapping({})
    if isinstance(value, ProvidersConfig):
        return value
    if isinstance(value, Mapping):
        return ProvidersConfig.from_mapping(value)
    raise TypeError("providers config must be a ProvidersConfig or mapping")


def _provider_options(
    entry: ProviderConfig,
    options: Mapping[str, Any] | None,
    overrides: Mapping[str, Any],
    secrets: SecretsLoader | None,
) -> dict[str, Any]:
    values = dict(entry.options)
    values.update(_mapping(options, "options"))
    values.update(overrides)
    if entry.secret_refs:
        loader = secrets if secrets is not None else SecretsLoader()
        for option, reference in entry.secret_refs.items():
            values[option] = loader.get(reference)
    return values


def _registry(value: ProviderRegistry | None) -> ProviderRegistry:
    if value is None:
        return provider_registry
    if not isinstance(value, ProviderRegistry):
        raise TypeError("registry must be a ProviderRegistry")
    return value


def _secrets_loader(value: SecretsLoader | None) -> SecretsLoader | None:
    if value is not None and not isinstance(value, SecretsLoader):
        raise TypeError("secrets must be a SecretsLoader")
    return value


def _mapping(value: Mapping[str, Any] | None, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return dict(value)


def _name(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError("provider name must be a non-empty string")
    return value.strip()


create_provider = build_provider
"""Alias for :func:`build_provider`."""

provider_from_config = build_provider
"""Alias for :func:`build_provider`."""

create_providers = build_providers
"""Alias for :func:`build_providers`."""

providers_from_config = build_providers
"""Alias for :func:`build_providers`."""

build_provider_instances = build_providers
"""Alias for :func:`build_providers`."""

provider_instances_from_config = build_providers
"""Alias for :func:`build_providers`."""

build_providers_from_config = build_providers
"""Alias for :func:`build_providers`."""
