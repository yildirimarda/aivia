"""Build the configured storage provider from the ``storage`` config section.

The media service should not branch on the backend name. This factory owns that
choice: a ``local`` section produces a filesystem provider rooted below the
user data directory, while an ``s3`` section produces an S3 compatible provider
whose credential references are resolved through :class:`SecretsLoader`.

Credentials are referenced by name in configuration and are never read from or
written to the config file itself. A missing reference fails before any network
client is created.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from aivia.core.config.http import HttpClientConfig
from aivia.core.config.ops import StorageConfig
from aivia.core.errors import ConfigError
from aivia.core.secrets import SecretsLoader
from aivia.providers.protocols.storage import StorageProvider
from aivia.providers.storage_local import LocalStorageProvider
from aivia.providers.storage_s3 import S3StorageProvider

__all__ = [
    "build_storage_provider",
    "create_storage_provider",
    "storage_provider_from_config",
]


def build_storage_provider(
    config: StorageConfig | Mapping[str, Any] | None = None,
    *,
    data_dir: str | Path | None = None,
    secrets: SecretsLoader | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    clock: Callable[[], datetime] | None = None,
    http_config: HttpClientConfig | None = None,
    name: str | None = None,
    health_timeout_seconds: float | None = None,
    close_timeout_seconds: float | None = None,
) -> StorageProvider:
    """Build a storage provider for *config*.

    *config* may be a validated :class:`StorageConfig`, a mapping such as the
    raw ``storage`` section, or ``None`` to use the local defaults. ``data_dir``
    anchors a relative ``local_dir``; without it the user data directory is
    used. ``secrets`` resolves S3 credential references and defaults to the
    environment/file/keyring loader.

    ``transport``, ``clock``, ``http_config`` and the timeout arguments are
    forwarded to the provider. The S3-only options are ignored for a local
    backend so callers can pass one set of options regardless of backend.
    """
    storage = _storage_config(config)
    if secrets is not None and not isinstance(secrets, SecretsLoader):
        raise TypeError("secrets must be a SecretsLoader")
    if transport is not None and not isinstance(transport, httpx.AsyncBaseTransport):
        raise TypeError("transport must be an httpx.AsyncBaseTransport")
    if clock is not None and not callable(clock):
        raise TypeError("clock must be callable")

    common: dict[str, Any] = {
        "name": name,
        "health_timeout_seconds": health_timeout_seconds,
        "close_timeout_seconds": close_timeout_seconds,
        "http_config": http_config,
    }

    if storage.backend == "local":
        return LocalStorageProvider(config=storage, data_dir=data_dir, **common)

    if storage.backend == "s3":
        loader = secrets or SecretsLoader()
        access_key = _resolve_credential(loader, storage.access_key_ref, "access key")
        secret_key = _resolve_credential(loader, storage.secret_key_ref, "secret key")
        return S3StorageProvider(
            config=storage,
            access_key=access_key,
            secret_key=secret_key,
            transport=transport,
            clock=clock,
            **common,
        )

    raise ConfigError(
        f"unknown storage backend: {storage.backend!r}",
        details={"backend": storage.backend},
    )


def _storage_config(
    config: StorageConfig | Mapping[str, Any] | None,
) -> StorageConfig:
    if config is None:
        return StorageConfig.from_mapping({})
    if isinstance(config, StorageConfig):
        return config
    if isinstance(config, Mapping):
        return StorageConfig.from_mapping(config)
    raise TypeError("config must be a StorageConfig or mapping")


def _resolve_credential(
    loader: SecretsLoader,
    reference: str | None,
    label: str,
) -> str:
    if reference is None or not reference.strip():
        raise ConfigError(
            f"s3 storage requires a {label} reference",
            details={"field": f"{label.replace(' ', '_')}_ref"},
        )
    return loader.get(reference.strip())


create_storage_provider = build_storage_provider
"""Alias for :func:`build_storage_provider`."""

storage_provider_from_config = build_storage_provider
"""Alias for :func:`build_storage_provider`."""
