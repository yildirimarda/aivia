"""Shared pieces for the capability protocols.

Capability protocols describe the typed boundary between the router and a
provider adapter. They are structural: a concrete adapter does not need to
inherit from a protocol class, it only has to expose the members the protocol
names. Every adapter still subclasses :class:`aivia.providers.base.Provider`
for metadata, health and shutdown.

Requests and responses are immutable pydantic models with ``extra="forbid"``.
Forbidding unknown fields keeps a provider from silently accepting a request
option it does not understand, while immutability makes it safe to log and
cache a request without a later stage changing it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol, runtime_checkable
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict

from aivia.core.models.provider import ProviderMeta
from aivia.providers.base import ProviderHealth

__all__ = [
    "CapabilityProvider",
    "ProtocolModel",
    "ensure_utc",
    "normalize_domain",
    "validate_http_url",
]


class ProtocolModel(BaseModel):
    """Base for typed protocol requests and responses.

    Unknown fields are rejected so a typo cannot silently turn into a no-op
    request option. Instances are frozen because requests and responses cross
    provider boundaries and may be recorded, cached or logged.
    """

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
        frozen=True,
    )


@runtime_checkable
class CapabilityProvider(Protocol):
    """The lifecycle and metadata every capability provider exposes.

    The router depends only on this surface plus the capability specific
    method, so a test fake and a real adapter are interchangeable.
    """

    @property
    def name(self) -> str:
        """The provider name from its metadata."""

    @property
    def metadata(self) -> ProviderMeta:
        """The validated metadata for this provider."""

    def supports(self, capability: str) -> bool:
        """Return whether this provider declares *capability*."""

    async def health(self) -> ProviderHealth:
        """Return the provider's current health."""

    async def close(self) -> None:
        """Release resources held by this provider."""


def ensure_utc(value: datetime) -> datetime:
    """Normalize a datetime to timezone-aware UTC.

    Naive values are treated as UTC because every timestamp in the system is
    stored in UTC; aware values are converted.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def validate_http_url(value: str, *, field: str = "url") -> str:
    """Validate and return an absolute, credential-free http(s) URL."""
    text = value.strip()
    if not text:
        raise ValueError(f"{field} must be non-empty")
    parsed = urlsplit(text)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{field} must be an absolute http(s) URL: {text!r}")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{field} must not contain credentials")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError(f"{field} has an invalid port: {text!r}") from exc
    return text


def normalize_domain(value: str) -> str:
    """Normalize a search domain filter to a bare lower-case host name."""
    if not isinstance(value, str):
        raise ValueError("domain must be a string")
    text = value.strip().lower()
    if not text:
        raise ValueError("domain must be non-empty")
    if "://" in text:
        host = urlsplit(text).hostname
        if not host:
            raise ValueError(f"domain must name a host: {value!r}")
        text = host
    elif any(separator in text for separator in ("/", "?", "#", "@", ":")):
        raise ValueError(f"domain must be a bare host name: {value!r}")
    if any(character.isspace() for character in text):
        raise ValueError(f"domain must be a bare host name: {value!r}")
    if text.startswith("www."):
        text = text[4:]
    if not text:
        raise ValueError(f"domain must name a host: {value!r}")
    return text
