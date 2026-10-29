"""Storage factory: choose local or S3 from config and resolve credentials."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from aivia.core.config import StorageConfig
from aivia.core.errors import ConfigError
from aivia.core.secrets import MissingSecretError, SecretsLoader
from aivia.providers import (
    LocalStorageProvider,
    S3StorageProvider,
    build_storage_provider,
    create_storage_provider,
    storage_provider_from_config,
)
from aivia.providers.protocols import StoragePutRequest

ACCESS_KEY = "AKIDEXAMPLE"
SECRET_KEY = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY"
REGION = "eu-central-1"
BUCKET = "media"
ENDPOINT = "https://s3.example.test"


class MockS3:
    """An in-memory S3 endpoint sufficient for a factory round trip."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if request.method == "HEAD":
            status = 200 if path in self.objects else 404
            return httpx.Response(status)
        if request.method == "PUT":
            self.objects[path] = request.content
            return httpx.Response(200)
        if request.method == "GET":
            data = self.objects.get(path)
            if data is None:
                return httpx.Response(404)
            return httpx.Response(200, content=data)
        return httpx.Response(405)


def _s3_config(**overrides: Any) -> StorageConfig:
    values: dict[str, Any] = {
        "backend": "s3",
        "bucket": BUCKET,
        "endpoint": ENDPOINT,
        "region": REGION,
        "access_key_ref": "AIVIA_S3_ACCESS_KEY",
        "secret_key_ref": "AIVIA_S3_SECRET_KEY",
    }
    values.update(overrides)
    return StorageConfig.from_mapping(values)


def _secrets(**overrides: str) -> SecretsLoader:
    values = {
        "AIVIA_S3_ACCESS_KEY": ACCESS_KEY,
        "AIVIA_S3_SECRET_KEY": SECRET_KEY,
    }
    values.update(overrides)
    return SecretsLoader(environ=values)


def test_local_config_builds_a_provider_below_the_data_dir(tmp_path: Path) -> None:
    config = StorageConfig.from_mapping({"backend": "local", "local_dir": "media"})

    provider = build_storage_provider(config, data_dir=tmp_path)

    assert isinstance(provider, LocalStorageProvider)
    assert provider.root == (tmp_path / "media").resolve()
    stored = asyncio.run(
        provider.put(StoragePutRequest(data=b"hello", key="outputs/hello.txt"))
    )
    assert asyncio.run(provider.get(stored.key)) == b"hello"


def test_local_config_accepts_a_mapping_and_an_absolute_directory(
    tmp_path: Path,
) -> None:
    provider = build_storage_provider(
        {"backend": "local", "local_dir": str(tmp_path / "absolute")}
    )

    assert isinstance(provider, LocalStorageProvider)
    assert provider.root == (tmp_path / "absolute").resolve()


def test_s3_config_builds_a_provider_with_resolved_credentials() -> None:
    mock = MockS3()
    config = _s3_config(prefix="runs", presign_seconds=120)

    provider = build_storage_provider(
        config,
        secrets=_secrets(),
        transport=httpx.MockTransport(mock),
    )

    assert isinstance(provider, S3StorageProvider)
    assert provider.bucket == BUCKET
    assert provider.endpoint == ENDPOINT
    assert provider.region == REGION
    assert provider.prefix == "runs"
    assert provider.presign_seconds == 120

    stored = asyncio.run(
        provider.put(StoragePutRequest(data=b"hello", key="outputs/hello.txt"))
    )
    assert asyncio.run(provider.get(stored.key)) == b"hello"
    assert "/media/runs/outputs/hello.txt" in mock.objects
    authorization = mock.requests[0].headers["authorization"]
    assert f"Credential={ACCESS_KEY}/" in authorization
    assert SECRET_KEY not in authorization


def test_a_missing_s3_secret_fails_before_building_a_client(tmp_path: Path) -> None:
    secrets_file = tmp_path / "secrets.env"
    secrets_file.write_text("", encoding="utf-8")
    loader = SecretsLoader(secrets_file=secrets_file, environ={})

    with pytest.raises(MissingSecretError) as caught:
        build_storage_provider(_s3_config(), secrets=loader)

    assert "AIVIA_S3_ACCESS_KEY" in str(caught.value)
    assert ACCESS_KEY not in str(caught.value)


def test_factory_aliases_build_the_same_provider(tmp_path: Path) -> None:
    config = StorageConfig.from_mapping({"backend": "local", "local_dir": "media"})

    assert create_storage_provider is build_storage_provider
    assert storage_provider_from_config is build_storage_provider
    assert isinstance(
        create_storage_provider(config, data_dir=tmp_path), LocalStorageProvider
    )


def test_factory_rejects_a_non_config_value() -> None:
    with pytest.raises(TypeError, match="StorageConfig"):
        build_storage_provider(cast(Any, object()))


def test_factory_rejects_an_unknown_backend() -> None:
    config = StorageConfig.model_construct(backend="ftp")

    with pytest.raises(ConfigError, match="unknown storage backend"):
        build_storage_provider(config)
