"""Build delivery providers from the ``delivery`` configuration section.

The orders config describes *what* to deliver (folder, link, email, webhook)
and the fallback chain; this factory turns each configured provider into a live
adapter. Credentials are referenced by name in configuration and resolved
through :class:`~aivia.core.secrets.SecretsLoader` before construction, never
read from the config file itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aivia.core.config.http import HttpClientConfig
from aivia.core.config.orders import (
    DeliveryConfig,
    DeliveryProviderConfig,
)
from aivia.core.errors import ConfigError
from aivia.core.http import HttpClient
from aivia.core.secrets import SecretsLoader
from aivia.providers.delivery.email import EmailDeliveryProvider
from aivia.providers.delivery.folder import LocalFolderDeliveryProvider
from aivia.providers.delivery.link import ExpiringStorageLinkDeliveryProvider
from aivia.providers.delivery.webhook import SignedWebhookDeliveryProvider
from aivia.providers.protocols.storage import StorageProvider

__all__ = [
    "build_delivery_provider",
    "build_delivery_providers",
    "create_delivery_provider",
    "create_delivery_providers",
    "delivery_provider_from_config",
    "delivery_providers_from_config",
]


def _resolve_secret(loader: SecretsLoader, reference: str, label: str) -> str:
    if not reference or not reference.strip():
        return ""
    return loader.get(reference.strip())


def build_delivery_provider(
    config: DeliveryProviderConfig | Mapping[str, Any] | str,
    *,
    name: str | None = None,
    storage: StorageProvider | None = None,
    secrets: SecretsLoader | None = None,
    http_client: HttpClient | None = None,
    http_config: HttpClientConfig | None = None,
) -> Any:
    """Build one delivery provider from *config*.

    *config* may be a validated :class:`DeliveryProviderConfig`, a mapping, or a
    provider type string such as ``folder``. A type string builds a provider
    with defaults only, so callers should normally pass a mapping or config.
    """
    loader = secrets if secrets is not None else SecretsLoader()
    if isinstance(config, DeliveryProviderConfig):
        provider = config
        values = dict(provider.model_dump(exclude_unset=True))
        resolved_type = provider.type
    elif isinstance(config, Mapping):
        values = dict(config)
        resolved_type = str(values.get("type", "")).strip().lower()
    elif isinstance(config, str):
        resolved_type = config.strip().lower()
        values = {}
    else:
        raise TypeError(
            "delivery provider config must be a DeliveryProviderConfig, "
            f"mapping or type string, got {type(config).__name__}"
        )
    provider_name = name if name is not None else str(values.get("name", resolved_type))

    if resolved_type in {
        "folder",
        "local",
        "file",
        "filesystem",
        "local_folder",
    }:
        return LocalFolderDeliveryProvider(
            path=str(values.get("path", "deliveries")),
            name=provider_name,
            http_config=http_config,
        )

    if resolved_type in {
        "link",
        "storage_link",
        "signed_link",
        "expiring_link",
        "expiring_storage_link",
    }:
        if storage is None:
            raise ConfigError(
                "link delivery provider requires a storage backend",
                details={"provider": provider_name, "type": resolved_type},
            )
        return ExpiringStorageLinkDeliveryProvider(
            storage=storage,
            expires_seconds=int(values.get("expires_seconds", 604_800)),
            name=provider_name,
            http_config=http_config,
        )

    if resolved_type in {
        "email",
        "mail",
        "smtp",
    }:
        return EmailDeliveryProvider(
            smtp_host=str(values.get("smtp_host", "")),
            smtp_port=int(values.get("smtp_port", 587)),
            from_address=str(values.get("from_address", "")),
            subject=str(values.get("subject", "Your personalized card")),
            username=_resolve_secret(
                loader, str(values.get("username_ref", "")), "username"
            ),
            password=_resolve_secret(
                loader, str(values.get("password_ref", "")), "password"
            ),
            timeout_seconds=float(values.get("timeout_seconds", 30.0)),
            name=provider_name,
            http_config=http_config,
        )

    if resolved_type in {
        "webhook",
        "signed_webhook",
        "http",
    }:
        return SignedWebhookDeliveryProvider(
            url=str(values.get("url", "")),
            secret=_resolve_secret(
                loader, str(values.get("secret_ref", "")), "webhook secret"
            ),
            allowlist=list(values.get("allowlist", [])),
            signature_header=str(values.get("signature_header", "X-Aivia-Signature")),
            http_client=http_client or HttpClient(http_config or HttpClientConfig()),
            name=provider_name,
            http_config=http_config,
        )

    raise ConfigError(
        f"unknown delivery provider type: {resolved_type!r}",
        details={"provider": provider_name, "type": resolved_type},
    )


def build_delivery_providers(
    config: DeliveryConfig | Mapping[str, Any] | None = None,
    *,
    storage: StorageProvider | None = None,
    secrets: SecretsLoader | None = None,
    http_client: HttpClient | None = None,
    http_config: HttpClientConfig | None = None,
) -> dict[str, Any]:
    """Build every enabled delivery provider from *config*, keyed by name.

    A missing or empty section produces an empty mapping. Disabled providers
    are skipped.
    """
    if config is None:
        delivery = DeliveryConfig.from_mapping({})
    elif isinstance(config, DeliveryConfig):
        delivery = config
    elif isinstance(config, Mapping):
        delivery = DeliveryConfig.from_mapping(config)
    else:
        raise TypeError(
            "delivery config must be a DeliveryConfig, mapping or None, "
            f"got {type(config).__name__}"
        )
    if not delivery.enabled:
        return {}
    providers: dict[str, Any] = {}
    for name, provider in delivery.providers.items():
        if not provider.enabled:
            continue
        providers[name] = build_delivery_provider(
            provider,
            name=name,
            storage=storage,
            secrets=secrets,
            http_client=http_client,
            http_config=http_config,
        )
    return providers


create_delivery_provider = build_delivery_provider
"""Alias for :func:`build_delivery_provider`."""
delivery_provider_from_config = build_delivery_provider
"""Alias for :func:`build_delivery_provider`."""

create_delivery_providers = build_delivery_providers
"""Alias for :func:`build_delivery_providers`."""
delivery_providers_from_config = build_delivery_providers
"""Alias for :func:`build_delivery_providers`."""
