"""Typed publisher protocol, post request and result.

A publisher sends a finished output to one platform account. Formatting,
account selection and attestation policy happen before the call; the adapter
only publishes the validated request and returns the platform identifiers.

Media may be inline bytes, a URL or a local path. Exactly one source is set so
an adapter never has to choose between two competing locations.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol, Self, runtime_checkable

from pydantic import Field, field_validator, model_validator

from aivia.core.models.io import Attestation
from aivia.providers.protocols.base import (
    CapabilityProvider,
    ProtocolModel,
    ensure_utc,
    validate_http_url,
)

__all__ = [
    "CAPABILITY_PUBLISH",
    "PostVisibility",
    "PublishMedia",
    "PublishRequest",
    "PublishResult",
    "Publisher",
]

CAPABILITY_PUBLISH = "publish"
"""The capability name a publisher declares."""


class PostVisibility(StrEnum):
    """Who may see a published post."""

    PUBLIC = "public"
    UNLISTED = "unlisted"
    PRIVATE = "private"
    FOLLOWERS = "followers"


def _strip(value: Any) -> Any:
    return value.strip() if isinstance(value, str) else value


def _strip_optional(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return value


class PublishMedia(ProtocolModel):
    """One image, video or audio file attached to a post."""

    data: bytes | None = None
    url: str | None = None
    path: str | None = None
    media_type: str | None = None
    filename: str | None = None
    alt_text: str | None = None

    @field_validator("data")
    @classmethod
    def _validate_data(cls, value: bytes | None) -> bytes | None:
        if value is not None and not value:
            raise ValueError("publish media data must be non-empty")
        return value

    @field_validator("url")
    @classmethod
    def _validate_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_http_url(value, field="publish media url")

    @field_validator("path", "media_type", "filename", "alt_text", mode="before")
    @classmethod
    def _strip_optional_text(cls, value: Any) -> Any:
        return _strip_optional(value)

    @field_validator("media_type")
    @classmethod
    def _validate_media_type(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = value.lower()
        if not text.startswith(("image/", "video/", "audio/")):
            raise ValueError("media_type must be an image, video or audio type")
        return text

    @model_validator(mode="after")
    def _validate_source(self) -> Self:
        sources = (self.data is not None, self.url is not None, self.path is not None)
        if sum(sources) != 1:
            raise ValueError("publish media needs exactly one of data, url or path")
        return self

    @property
    def size_bytes(self) -> int | None:
        """The size of inline data, or ``None`` for a remote or local file."""
        if self.data is None:
            return None
        return len(self.data)


class PublishRequest(ProtocolModel):
    """A validated post ready for one platform account.

    ``idempotency_key`` lets an adapter avoid posting twice after a timeout.
    ``scheduled_for`` is a UTC instant; timezone policy belongs to the caller.
    """

    platform: str = Field(min_length=1)
    account: str = ""
    profile: str = ""
    language: str = ""
    format: str = ""
    title: str = ""
    caption: str = ""
    hashtags: list[str] = Field(default_factory=list)
    mentions: list[str] = Field(default_factory=list)
    media: list[PublishMedia] = Field(default_factory=list)
    visibility: PostVisibility = PostVisibility.PRIVATE
    scheduled_for: datetime | None = None
    idempotency_key: str = ""
    dry_run: bool = False
    disclosure_text: str = ""
    attestation: Attestation | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("platform", mode="before")
    @classmethod
    def _strip_platform(cls, value: Any) -> Any:
        return _strip(value)

    @field_validator(
        "account",
        "profile",
        "language",
        "format",
        "title",
        "caption",
        "idempotency_key",
        "disclosure_text",
        mode="before",
    )
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _strip(value)

    @field_validator("hashtags", "mentions")
    @classmethod
    def _normalize_tags(cls, value: list[str]) -> list[str]:
        tags: list[str] = []
        for raw in value:
            tag = _strip(raw)
            if not tag:
                raise ValueError("hashtags and mentions must be non-empty")
            if tag not in tags:
                tags.append(tag)
        return tags

    @field_validator("scheduled_for")
    @classmethod
    def _normalize_timestamp(cls, value: datetime | None) -> datetime | None:
        return None if value is None else ensure_utc(value)

    @property
    def has_media(self) -> bool:
        """Whether the post carries media."""
        return bool(self.media)


class PublishResult(ProtocolModel):
    """The platform identifiers returned after a publish attempt."""

    platform: str = ""
    account: str = ""
    provider: str = ""
    remote_id: str = ""
    url: str | None = None
    published_at: datetime | None = None
    dry_run: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("platform", "account", "provider", "remote_id", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _strip(value)

    @field_validator("url")
    @classmethod
    def _validate_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_http_url(value, field="published post url")

    @field_validator("published_at")
    @classmethod
    def _normalize_timestamp(cls, value: datetime | None) -> datetime | None:
        return None if value is None else ensure_utc(value)

    @property
    def is_live(self) -> bool:
        """Whether the result names a real remote post."""
        return bool(self.remote_id) and not self.dry_run


@runtime_checkable
class Publisher(CapabilityProvider, Protocol):
    """A provider that serves the ``publish`` capability."""

    async def publish(self, request: PublishRequest) -> PublishResult:
        """Publish *request* and return the platform identifiers."""
