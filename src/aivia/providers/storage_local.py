"""Local filesystem storage with content addressed keys and dedupe.

The provider keeps every blob under one root directory. A put without an
explicit key derives a content addressed key from the SHA-256 digest, so the
same bytes always land on the same path. A supplied key is kept so a caller can
pin stable paths for final outputs. Putting identical bytes under an existing
key is a no-op that returns the original record; different bytes under the same
key require ``overwrite``.

URLs are ``file://`` links to the resolved object. Local files do not expire,
so an ``expires_seconds`` argument is accepted for protocol compatibility but
does not change the URL.
"""

from __future__ import annotations

import hashlib
import mimetypes
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aivia.core.config.http import HttpClientConfig
from aivia.core.config.ops import StorageConfig
from aivia.core.errors import ConfigError, InvalidRequestError
from aivia.core.models.provider import ProviderMeta, ProviderPrice
from aivia.providers.base import Provider
from aivia.providers.protocols.storage import (
    CAPABILITY_STORAGE,
    StorageObject,
    StoragePutRequest,
)
from aivia.providers.registry import register_provider

__all__ = [
    "LocalStorage",
    "LocalStorageProvider",
]

_DEFAULT_METADATA = ProviderMeta(
    name="local",
    capabilities=[CAPABILITY_STORAGE],
    pricing={CAPABILITY_STORAGE: ProviderPrice(amount_usd=0.0)},
)

_KEY_SEPARATOR = "/"
_CONTENT_PREFIX = "objects"


def _utc_from_ns(timestamp_ns: int) -> datetime:
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=UTC)


def _normalise_key(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("storage key must be a string")
    text = value.strip().replace("\\", _KEY_SEPARATOR)
    if not text:
        raise InvalidRequestError("storage key must be non-empty")
    if "\x00" in text:
        raise InvalidRequestError("storage key must not contain NUL")
    if text.startswith(_KEY_SEPARATOR):
        raise InvalidRequestError(f"storage key must be relative: {value!r}")

    parts: list[str] = []
    for part in text.split(_KEY_SEPARATOR):
        if part in {"", "."}:
            continue
        if part == "..":
            raise InvalidRequestError(f"storage key must not traverse: {value!r}")
        if ":" in part:
            raise InvalidRequestError(f"storage key must not name a drive: {value!r}")
        parts.append(part)
    if not parts:
        raise InvalidRequestError("storage key must be non-empty")
    return _KEY_SEPARATOR.join(parts)


def _content_key(digest: str) -> str:
    return f"{_CONTENT_PREFIX}/{digest[:2]}/{digest}"


def _guess_content_type(key: str) -> str | None:
    guessed, _ = mimetypes.guess_type(key)
    return guessed


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class LocalStorageProvider(Provider):
    """A storage provider rooted at a local directory.

    ``root`` may be a path, an existing :class:`StorageConfig`, or ``None`` to
    use ``<user data dir>/media``. ``directory`` and ``base_dir`` are accepted
    as aliases so a caller can name the root whichever way reads best. When a
    relative root is supplied with a ``StorageConfig``, it is resolved below
    ``data_dir`` when that is given, otherwise below the user data directory.
    """

    default_metadata = _DEFAULT_METADATA

    def __init__(
        self,
        root: str | Path | StorageConfig | None = None,
        *,
        directory: str | Path | None = None,
        base_dir: str | Path | None = None,
        config: StorageConfig | None = None,
        data_dir: str | Path | None = None,
        name: str | None = None,
        health_timeout_seconds: float | None = None,
        close_timeout_seconds: float | None = None,
        http_config: HttpClientConfig | None = None,
    ) -> None:
        super().__init__(
            name=name,
            health_timeout_seconds=health_timeout_seconds,
            close_timeout_seconds=close_timeout_seconds,
            http_config=http_config,
        )
        if isinstance(root, StorageConfig):
            if config is not None:
                raise ValueError("storage config supplied twice")
            config = root
            root = None
        if config is not None and not isinstance(config, StorageConfig):
            raise TypeError("config must be a StorageConfig")
        if config is not None and config.backend != "local":
            raise ConfigError(
                "local storage provider requires storage.backend='local'",
                details={"backend": config.backend},
            )

        selected = root if root is not None else directory
        if selected is None:
            selected = base_dir
        if selected is None and config is not None:
            selected = config.local_dir
        if selected is None:
            selected = "media"

        if not isinstance(selected, (str, Path)):
            raise TypeError("storage root must be a path or string")
        path = Path(selected).expanduser()
        if not path.is_absolute():
            if data_dir is not None:
                path = Path(data_dir).expanduser() / path
            elif config is not None:
                from aivia.core import paths

                path = paths.data_dir() / path
        self._root = path.resolve()
        self._cache: dict[str, tuple[StorageObject, int, int]] = {}

    @property
    def root(self) -> Path:
        """The absolute directory every object is stored under."""
        return self._root

    @property
    def directory(self) -> Path:
        """Alias for :attr:`root`."""
        return self._root

    @property
    def base_dir(self) -> Path:
        """Alias for :attr:`root`."""
        return self._root

    @property
    def local_dir(self) -> Path:
        """Alias for :attr:`root`."""
        return self._root

    @staticmethod
    def content_key(data: bytes) -> str:
        """Return the content addressed key for *data*."""
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("data must be bytes")
        digest = hashlib.sha256(bytes(data)).hexdigest()
        return _content_key(digest)

    def _path_for(self, key: str) -> tuple[str, Path]:
        normalised = _normalise_key(key)
        candidate = (self._root / normalised).resolve()
        if candidate != self._root and self._root not in candidate.parents:
            raise InvalidRequestError(
                f"storage key escapes the storage root: {key!r}",
                details={"key": key},
            )
        return normalised, candidate

    def _cached(self, key: str, path: Path) -> StorageObject | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        stored, mtime_ns, size = entry
        try:
            stat = path.stat()
        except OSError:
            return None
        if stat.st_mtime_ns != mtime_ns or stat.st_size != size:
            return None
        return stored

    def _object_for(
        self,
        key: str,
        path: Path,
        *,
        digest: str,
        content_type: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> StorageObject:
        stat = path.stat()
        stored = StorageObject(
            key=key,
            size_bytes=stat.st_size,
            content_type=content_type,
            sha256=digest,
            url=self.url(key),
            created_at=_utc_from_ns(stat.st_mtime_ns),
            metadata=dict(metadata or {}),
        )
        self._cache[key] = (stored, stat.st_mtime_ns, stat.st_size)
        return stored

    async def put(self, request: StoragePutRequest) -> StorageObject:
        """Store *request*, deduping identical bytes under the same key."""
        if not isinstance(request, StoragePutRequest):
            raise TypeError("request must be a StoragePutRequest")
        digest = hashlib.sha256(request.data).hexdigest()
        key = _normalise_key(request.key) if request.key is not None else None
        if key is None:
            key = _content_key(digest)
        _, path = self._path_for(key)

        existing = self._cached(key, path)
        if existing is not None and existing.sha256 == digest:
            return existing
        if path.exists():
            if path.is_dir():
                raise InvalidRequestError(f"storage key names a directory: {key}")
            existing_digest = _sha256_file(path)
            if existing_digest == digest:
                return self._object_for(
                    key,
                    path,
                    digest=digest,
                    content_type=request.content_type or _guess_content_type(key),
                )
            if not request.overwrite:
                raise InvalidRequestError(
                    f"storage key already exists: {key}",
                    details={"key": key},
                )

        content_type = request.content_type or _guess_content_type(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            dir=path.parent,
            prefix=".aivia-storage-",
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(request.data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

        return self._object_for(
            key,
            path,
            digest=digest,
            content_type=content_type,
            metadata=request.metadata,
        )

    async def get(self, key: str) -> bytes:
        """Read *key*, raising :class:`KeyError` when it does not exist."""
        normalised, path = self._path_for(key)
        try:
            return path.read_bytes()
        except (FileNotFoundError, IsADirectoryError):
            raise KeyError(normalised) from None

    async def stat(self, key: str) -> StorageObject | None:
        """Return the record for *key*, or ``None`` when it is missing."""
        normalised, path = self._path_for(key)
        if not path.is_file():
            return None
        cached = self._cached(normalised, path)
        if cached is not None:
            return cached
        return self._object_for(
            normalised,
            path,
            digest=_sha256_file(path),
            content_type=_guess_content_type(normalised),
        )

    async def delete(self, key: str) -> bool:
        """Delete *key* and report whether a file was removed."""
        normalised, path = self._path_for(key)
        self._cache.pop(normalised, None)
        try:
            path.unlink()
        except (FileNotFoundError, IsADirectoryError):
            return False
        return True

    def url(self, key: str, *, expires_seconds: int | None = None) -> str:
        """Return a ``file://`` URL for *key*.

        Local files never expire, so *expires_seconds* is validated but does
        not alter the URL.
        """
        if expires_seconds is not None:
            if isinstance(expires_seconds, bool) or not isinstance(
                expires_seconds, int
            ):
                raise TypeError("expires_seconds must be an integer")
            if expires_seconds <= 0:
                raise ValueError("expires_seconds must be positive")
        _, path = self._path_for(key)
        return path.as_uri()


LocalStorage = LocalStorageProvider
"""Alias for :class:`LocalStorageProvider`."""


register_provider("local", LocalStorageProvider, module=__name__)
