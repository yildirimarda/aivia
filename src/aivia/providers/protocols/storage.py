"""Typed storage protocol and content addressed object models.

Storage keeps generated media and intermediates. A provider puts bytes and
returns an object record with a key, size, checksum and resolved URL. The local
and S3 providers share this boundary so the media service never knows which
backend is configured.

A put request may leave the key unset, in which case the provider derives a
content addressed key from the bytes. A supplied key is kept so callers can pin
stable paths for final outputs.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlsplit

from pydantic import Field, field_validator

from aivia.providers.protocols.base import (
    CapabilityProvider,
    ProtocolModel,
    ensure_utc,
)

__all__ = [
    "CAPABILITY_STORAGE",
    "StorageObject",
    "StorageProvider",
    "StoragePutRequest",
]

CAPABILITY_STORAGE = "storage"
"""The capability name a storage provider declares."""


def _strip(value: Any) -> Any:
    return value.strip() if isinstance(value, str) else value


def _strip_optional(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return value


def _validate_location(value: str) -> str:
    text = value.strip()
    parsed = urlsplit(text)
    if parsed.scheme.lower() not in {"http", "https", "file"}:
        raise ValueError("storage url must be http, https or file")
    if parsed.scheme.lower() in {"http", "https"} and not parsed.hostname:
        raise ValueError("storage url must name a host")
    return text


class StoragePutRequest(ProtocolModel):
    """A validated request to store one blob.

    ``key`` is optional because a content addressed backend can derive one.
    ``overwrite`` lets a caller replace a pinned key; the default refuses to
    clobber an existing object.
    """

    data: bytes
    key: str | None = None
    content_type: str | None = None
    overwrite: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("data")
    @classmethod
    def _validate_data(cls, value: bytes) -> bytes:
        if not value:
            raise ValueError("storage data must be non-empty")
        return value

    @field_validator("key", "content_type", mode="before")
    @classmethod
    def _strip_optional_text(cls, value: Any) -> Any:
        return _strip_optional(value)

    @property
    def size_bytes(self) -> int:
        """The number of bytes to store."""
        return len(self.data)


class StorageObject(ProtocolModel):
    """One stored blob and the metadata needed to serve it.

    ``sha256`` is the hex digest of the bytes. ``url`` may be a presigned link,
    a local file URL or a public URL, and ``expires_at`` says when it stops
    working.
    """

    key: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    content_type: str | None = None
    sha256: str | None = None
    url: str | None = None
    expires_at: datetime | None = None
    created_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("key", mode="before")
    @classmethod
    def _strip_key(cls, value: Any) -> Any:
        return _strip(value)

    @field_validator("content_type", "sha256", mode="before")
    @classmethod
    def _strip_optional_text(cls, value: Any) -> Any:
        return _strip_optional(value)

    @field_validator("sha256")
    @classmethod
    def _validate_sha256(cls, value: str | None) -> str | None:
        if value is None:
            return None
        digest = value.lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("sha256 must be a 64 character hex digest")
        return digest

    @field_validator("url")
    @classmethod
    def _validate_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_location(value)

    @field_validator("expires_at", "created_at")
    @classmethod
    def _normalize_timestamp(cls, value: datetime | None) -> datetime | None:
        return None if value is None else ensure_utc(value)


@runtime_checkable
class StorageProvider(CapabilityProvider, Protocol):
    """A provider that serves the ``storage`` capability."""

    async def put(self, request: StoragePutRequest) -> StorageObject:
        """Store *request* and return the resulting object record."""

    async def get(self, key: str) -> bytes:
        """Read the bytes stored under *key*."""

    async def stat(self, key: str) -> StorageObject | None:
        """Return the object record for *key*, or ``None`` when missing."""

    async def delete(self, key: str) -> bool:
        """Delete *key* and return whether an object was removed."""

    def url(self, key: str, *, expires_seconds: int | None = None) -> str:
        """Resolve *key* to a URL, optionally expiring after some seconds."""
