"""Typed notifier protocol, message models and receipt.

A notifier delivers a message to a human channel: a chat, a webhook, an email
or the console. It is a transport, not a policy engine: the caller decides who
to notify, when and what buttons to offer, while the adapter only sends the
validated message and reports what the channel acknowledged.

A message may carry text, media or both. Buttons are either a URL or a callback
payload, never both, so an adapter never has to guess which one to send.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol, Self, runtime_checkable

from pydantic import Field, field_validator, model_validator

from aivia.providers.protocols.base import (
    CapabilityProvider,
    ProtocolModel,
    ensure_utc,
    validate_http_url,
)

__all__ = [
    "CAPABILITY_NOTIFY",
    "NotificationButton",
    "NotificationLevel",
    "NotificationMedia",
    "NotificationMessage",
    "NotificationReceipt",
    "Notifier",
]

CAPABILITY_NOTIFY = "notify"
"""The capability name a notifier declares."""


class NotificationLevel(StrEnum):
    """How urgent a notification is."""

    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


def _strip(value: Any) -> Any:
    return value.strip() if isinstance(value, str) else value


def _strip_optional(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return value


class NotificationButton(ProtocolModel):
    """One inline button attached to a notification.

    ``url`` opens a link while ``callback_data`` asks the channel to send the
    payload back. Exactly one is set.
    """

    label: str = Field(min_length=1)
    url: str | None = None
    callback_data: str | None = None

    @field_validator("label", mode="before")
    @classmethod
    def _strip_label(cls, value: Any) -> Any:
        return _strip(value)

    @field_validator("url")
    @classmethod
    def _validate_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_http_url(value, field="button url")

    @field_validator("callback_data", mode="before")
    @classmethod
    def _strip_callback(cls, value: Any) -> Any:
        return _strip_optional(value)

    @model_validator(mode="after")
    def _validate_action(self) -> Self:
        if (self.url is None) == (self.callback_data is None):
            raise ValueError("button needs exactly one of url or callback_data")
        return self


class NotificationMedia(ProtocolModel):
    """One image, video or audio attachment supplied inline, by URL or path."""

    data: bytes | None = None
    url: str | None = None
    path: str | None = None
    media_type: str | None = None
    filename: str | None = None
    caption: str | None = None

    @field_validator("data")
    @classmethod
    def _validate_data(cls, value: bytes | None) -> bytes | None:
        if value is not None and not value:
            raise ValueError("notification media data must be non-empty")
        return value

    @field_validator("url")
    @classmethod
    def _validate_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_http_url(value, field="notification media url")

    @field_validator("path", "media_type", "filename", "caption", mode="before")
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
            raise ValueError(
                "notification media needs exactly one of data, url or path"
            )
        return self

    @property
    def size_bytes(self) -> int | None:
        """The size of inline data, or ``None`` for a remote or local file."""
        if self.data is None:
            return None
        return len(self.data)


class NotificationMessage(ProtocolModel):
    """A validated message sent to a human channel.

    ``text`` and ``media`` are both optional but at least one must be present.
    ``buttons`` is a flat list; an adapter is free to arrange it into rows.
    """

    title: str = ""
    text: str = ""
    level: NotificationLevel = NotificationLevel.INFO
    media: list[NotificationMedia] = Field(default_factory=list)
    buttons: list[NotificationButton] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("title", "text", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _strip(value)

    @model_validator(mode="after")
    def _validate_content(self) -> Self:
        if not (self.text or self.title or self.media):
            raise ValueError("notification needs text, title or media")
        return self

    @property
    def has_media(self) -> bool:
        """Whether the message carries an attachment."""
        return bool(self.media)

    @property
    def has_buttons(self) -> bool:
        """Whether the message offers inline buttons."""
        return bool(self.buttons)


class NotificationReceipt(ProtocolModel):
    """What a channel acknowledged after accepting a message."""

    message_id: str = ""
    channel: str = ""
    provider: str = ""
    delivered: bool = True
    delivered_at: datetime | None = None
    url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("message_id", "channel", "provider", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _strip(value)

    @field_validator("delivered_at")
    @classmethod
    def _normalize_timestamp(cls, value: datetime | None) -> datetime | None:
        return None if value is None else ensure_utc(value)

    @field_validator("url")
    @classmethod
    def _validate_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_http_url(value, field="receipt url")


@runtime_checkable
class Notifier(CapabilityProvider, Protocol):
    """A provider that serves the ``notify`` capability."""

    async def send(self, message: NotificationMessage) -> NotificationReceipt:
        """Send *message* and return the channel acknowledgement."""
