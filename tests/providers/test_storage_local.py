"""Local storage: content addressed keys, dedupe and file URL resolution."""

from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import pytest

from aivia.core.config import StorageConfig
from aivia.core.errors import ConfigError, InvalidRequestError
from aivia.providers import (
    LocalStorage,
    LocalStorageProvider,
    ProviderRegistry,
    provider_registry,
)
from aivia.providers.protocols import (
    CAPABILITY_STORAGE,
    StorageObject,
    StorageProvider,
    StoragePutRequest,
)


def put(provider: LocalStorageProvider, data: bytes, **kwargs: Any) -> StorageObject:
    return asyncio.run(provider.put(StoragePutRequest(data=data, **kwargs)))


def test_local_storage_declares_the_storage_capability(tmp_path: Path) -> None:
    provider = LocalStorageProvider(tmp_path)

    assert isinstance(provider, StorageProvider)
    assert provider.supports(CAPABILITY_STORAGE)
    assert provider.name == "local"
    assert provider.root == tmp_path.resolve()
    assert provider.directory == provider.root
    assert LocalStorage is LocalStorageProvider
    assert provider_registry.resolve("local") is LocalStorageProvider


def test_put_derives_a_content_addressed_path(tmp_path: Path) -> None:
    provider = LocalStorageProvider(tmp_path)
    data = b"hello world"
    digest = hashlib.sha256(data).hexdigest()

    stored = put(provider, data)

    assert stored.key == f"objects/{digest[:2]}/{digest}"
    assert stored.sha256 == digest
    assert stored.size_bytes == len(data)
    assert stored.created_at is not None
    assert (tmp_path / stored.key).read_bytes() == data
    assert stored.url == (tmp_path / stored.key).resolve().as_uri()


def test_put_dedupes_identical_bytes_without_rewriting(tmp_path: Path) -> None:
    provider = LocalStorageProvider(tmp_path)
    first = put(provider, b"same bytes")

    path = tmp_path / first.key
    old_mtime = 1_000_000_000_000_000_000
    os.utime(path, ns=(old_mtime, old_mtime))

    second = put(provider, b"same bytes")

    assert second.key == first.key
    assert second.sha256 == first.sha256
    assert path.stat().st_mtime_ns == old_mtime
    assert len([item for item in tmp_path.rglob("*") if item.is_file()]) == 1


def test_a_supplied_key_is_kept_and_resolved_to_a_file_url(tmp_path: Path) -> None:
    provider = LocalStorageProvider(tmp_path)
    stored = put(
        provider,
        b"hello",
        key="  outputs/hello.txt  ",
        content_type="text/plain",
        metadata={"run": "run-1"},
    )

    assert stored.key == "outputs/hello.txt"
    assert stored.content_type == "text/plain"
    assert stored.metadata == {"run": "run-1"}
    assert (tmp_path / "outputs" / "hello.txt").read_bytes() == b"hello"

    url = provider.url("outputs/hello.txt")
    parsed = urlsplit(url)
    assert parsed.scheme == "file"
    assert Path(unquote(parsed.path)) == (tmp_path / "outputs" / "hello.txt").resolve()
    assert provider.url("outputs/hello.txt", expires_seconds=60) == url


def test_put_refuses_to_clobber_a_key_unless_overwrite_is_set(
    tmp_path: Path,
) -> None:
    provider = LocalStorageProvider(tmp_path)
    put(provider, b"first", key="fixed.txt")

    with pytest.raises(InvalidRequestError, match="already exists"):
        put(provider, b"second", key="fixed.txt")

    replaced = put(provider, b"second", key="fixed.txt", overwrite=True)
    assert asyncio.run(provider.get("fixed.txt")) == b"second"
    assert replaced.sha256 == hashlib.sha256(b"second").hexdigest()


def test_get_stat_and_delete_follow_the_storage_contract(tmp_path: Path) -> None:
    provider = LocalStorageProvider(tmp_path)
    stored = put(provider, b"payload")

    assert asyncio.run(provider.get(stored.key)) == b"payload"
    assert asyncio.run(provider.stat(stored.key)) == stored
    assert asyncio.run(provider.delete(stored.key)) is True
    assert asyncio.run(provider.delete(stored.key)) is False
    assert asyncio.run(provider.stat(stored.key)) is None
    with pytest.raises(KeyError):
        asyncio.run(provider.get(stored.key))


def test_a_new_provider_instance_reads_and_stats_the_same_objects(
    tmp_path: Path,
) -> None:
    first = LocalStorageProvider(tmp_path)
    stored = put(first, b"persisted", key="nested/file.txt")

    second = LocalStorageProvider(directory=tmp_path)

    assert asyncio.run(second.get(stored.key)) == b"persisted"
    restated = asyncio.run(second.stat(stored.key))
    assert restated is not None
    assert restated.key == stored.key
    assert restated.sha256 == stored.sha256
    assert restated.size_bytes == stored.size_bytes
    assert restated.content_type == "text/plain"


def test_unsafe_keys_cannot_escape_the_storage_root(tmp_path: Path) -> None:
    provider = LocalStorageProvider(tmp_path / "root")

    for key in ("../escape.txt", "/etc/passwd", "C:\\windows\\system32", "a/../../b"):
        with pytest.raises(InvalidRequestError):
            put(provider, b"x", key=key)

    with pytest.raises(InvalidRequestError):
        provider.url("../../etc/passwd")


def test_a_relative_local_dir_resolves_below_the_data_dir(tmp_path: Path) -> None:
    config = StorageConfig.from_mapping({"backend": "local", "local_dir": "media"})

    provider = LocalStorageProvider(config=config, data_dir=tmp_path)

    assert provider.root == (tmp_path / "media").resolve()


def test_the_local_provider_rejects_an_s3_configuration(tmp_path: Path) -> None:
    config = StorageConfig.from_mapping(
        {
            "backend": "s3",
            "bucket": "media",
            "endpoint": "https://s3.example.test",
            "region": "eu-central-1",
            "access_key_ref": "AIVIA_S3_ACCESS_KEY",
            "secret_key_ref": "AIVIA_S3_SECRET_KEY",
        }
    )

    with pytest.raises(ConfigError, match="backend"):
        LocalStorageProvider(config=config, data_dir=tmp_path)


def test_the_provider_is_registered_once() -> None:
    registry = ProviderRegistry()
    registry.register("local", LocalStorageProvider)

    assert registry.resolve("local") is LocalStorageProvider
