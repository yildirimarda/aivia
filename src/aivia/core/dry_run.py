"""Dry-run provider substitution.

A dry run is a safety boundary, not a hint: it must not spend money or open an
outbound connection even when the configuration names a real provider. The
reliable way to guarantee that is to replace the provider section before any
provider instance or router chain is built. Every configured name keeps its
position and enabled state, but its type becomes the registered all-in-one
mock and its real options and secret references are dropped.

The original configuration is never mutated, so a dry-run copy can never leak
mock types into a config snapshot or a later real run.
"""

from __future__ import annotations

from typing import Any

from aivia.core.config.providers import ProviderConfig, ProvidersConfig
from aivia.providers.base import Provider
from aivia.providers.mock import register_mock_providers
from aivia.providers.registry import ProviderRegistry
from aivia.providers.registry import registry as provider_registry

__all__ = [
    "DRY_RUN_PROVIDER_TYPE",
    "apply_dry_run",
    "dry_run_provider_config",
    "dry_run_providers",
]

DRY_RUN_PROVIDER_TYPE = "mock"
"""The registered provider type used for every provider during a dry run."""


def _mock_entry() -> ProviderConfig:
    return ProviderConfig(type=DRY_RUN_PROVIDER_TYPE)


def dry_run_provider_config(
    section: ProvidersConfig | None,
) -> ProvidersConfig:
    """Return a copy of *section* whose active entries use the mock type.

    A missing or empty section becomes a single mock provider so a fresh
    installation cannot fall back to a real provider that happens to be
    registered globally. Disabled entries keep their configured type because
    no caller may build them.
    """
    if section is None or not section.providers:
        return ProvidersConfig(providers={"mock": _mock_entry()})

    entries: dict[str, ProviderConfig] = {}
    for name, entry in section.providers.items():
        if entry.enabled:
            entries[name] = _mock_entry()
        else:
            entries[name] = entry.model_copy(deep=True)
    return ProvidersConfig(enabled=section.enabled, providers=entries)


def dry_run_providers(
    section: ProvidersConfig | None,
    *,
    registry: ProviderRegistry | None = None,
) -> dict[str, Provider]:
    """Return one mock instance for every provider a dry run may build.

    The configured names are preserved, so capability chains and router
    selection keep working unchanged. *registry* defaults to the shared
    provider registry; the mock types are registered on it idempotently.
    """
    source = provider_registry if registry is None else registry
    register_mock_providers(source)
    config = dry_run_provider_config(section)
    provider_type = source.resolve(DRY_RUN_PROVIDER_TYPE)
    return {name: provider_type(name=name) for name in config.enabled_providers()}


def apply_dry_run(config: Any) -> Any:
    """Return *config* with its provider section swapped for mocks.

    The input is returned unchanged when it has no provider section, so this is
    safe to call for any resolved configuration object.
    """
    section = getattr(config, "providers", None)
    if not isinstance(section, ProvidersConfig):
        return config
    copy = getattr(config, "model_copy", None)
    if not callable(copy):
        return config
    return copy(update={"providers": dry_run_provider_config(section)})
