"""Typed delivery protocol, request and receipt.

A delivery hands finished files to an order recipient through a folder, an
expiring storage link, email or a signed webhook. The provider only moves the
artifacts and reports the destination it used; retries, scheduling and the
fallback chain belong to the delivery service.

Artifacts may be inline bytes, a URL or a local path. Exactly one source is set
so an adapter never has to choose between two competing locations.
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
    "CAPABILITY_DELIVERY",
    "DeliveryArtifact",
    "DeliveryMethod",
    "DeliveryProvider",
    "DeliveryReceipt",
    "DeliveryRequest",
]

CAPABILITY_DELIVERY = "delivery"
"""The capability name a delivery provider declares."""


class DeliveryMethod(StrEnum):
    """The transport a delivery provider uses."""

    FOLDER = "folder"
    LINK = "link"
    EMAIL = "email"
    WEBHOOK = "webhook"
    WEB_CARD = "web_card"


def _strip(value: Any) -> Any:
    return value.strip() if isinstance(value, str) else value


def _strip_optional(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return value


class DeliveryArtifact(ProtocolModel):
    """One finished file handed to a recipient."""

    data: bytes | None = None
    url: str | None = None
    path: str | None = None
    media_type: str | None = None
    filename: str | None = None

    @field_validator("data")
    @classmethod
    def _validate_data(cls, value: bytes | None) -> bytes | None:
        if value is not None and not value:
            raise ValueError("delivery artifact data must be non-empty")
        return value

    @field_validator("url")
    @classmethod
    def _validate_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_http_url(value, field="delivery artifact url")

    @field_validator("path", "media_type", "filename", mode="before")
    @classmethod
    def _strip_optional_text(cls, value: Any) -> Any:
        return _strip_optional(value)

    @model_validator(mode="after")
    def _validate_source(self) -> Self:
        sources = (self.data is not None, self.url is not None, self.path is not None)
        if sum(sources) != 1:
            raise ValueError("delivery artifact needs exactly one of data, url or path")
        return self

    @property
    def size_bytes(self) -> int | None:
        """The size of inline data, or ``None`` for a remote or local file."""
        if self.data is None:
            return None
        return len(self.data)


class DeliveryRequest(ProtocolModel):
    """A validated request to deliver an order's finished output.

    ``target`` is the destination the method needs: a folder, URL, email
    address or webhook endpoint. ``idempotency_key`` prevents a retry from
    delivering twice.
    """

    method: DeliveryMethod
    target: str = Field(min_length=1)
    order_id: str = ""
    run_id: str = ""
    subject: str = ""
    message: str = ""
    artifacts: list[DeliveryArtifact] = Field(default_factory=list)
    expires_seconds: int | None = Field(default=None, gt=0)
    scheduled_for: datetime | None = None
    idempotency_key: str = ""
    attestation: Attestation | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("target", mode="before")
    @classmethod
    def _strip_target(cls, value: Any) -> Any:
        return _strip(value)

    @field_validator(
        "order_id",
        "run_id",
        "subject",
        "message",
        "idempotency_key",
        mode="before",
    )
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _strip(value)

    @field_validator("scheduled_for")
    @classmethod
    def _normalize_timestamp(cls, value: datetime | None) -> datetime | None:
        return None if value is None else ensure_utc(value)

    @property
    def has_artifacts(self) -> bool:
        """Whether the request carries at least one file."""
        return bool(self.artifacts)


class DeliveryReceipt(ProtocolModel):
    """What a delivery provider acknowledged after accepting the output."""

    method: DeliveryMethod
    reference: str = ""
    target: str = ""
    provider: str = ""
    delivered: bool = True
    delivered_at: datetime | None = None
    url: str | None = None
    expires_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("reference", "target", "provider", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _strip(value)

    @field_validator("url")
    @classmethod
    def _validate_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_http_url(value, field="delivery url")

    @field_validator("delivered_at", "expires_at")
    @classmethod
    def _normalize_timestamp(cls, value: datetime | None) -> datetime | None:
        return None if value is None else ensure_utc(value)


@runtime_checkable
class DeliveryProvider(CapabilityProvider, Protocol):
    """A provider that serves the ``delivery`` capability."""

    async def deliver(self, request: DeliveryRequest) -> DeliveryReceipt:
        """Deliver *request* and return the recipient acknowledgement."""
