"""Typed stock media protocol, request and response models.

A stock provider searches a licensed library for images or video. It is a
search capability, not a generation one: the request carries a query and
filters, and every result records the creator, license and attribution needed
to publish it legally.

Results always carry an absolute download URL; preview and page URLs are
optional. A result never carries inline bytes because the media service
downloads and validates the chosen asset itself.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol, Self, runtime_checkable

from pydantic import Field, field_validator, model_validator

from aivia.providers.protocols.base import (
    CapabilityProvider,
    ProtocolModel,
    validate_http_url,
)
from aivia.providers.protocols.media import (
    ImageFormat,
    MediaDimensions,
    VideoFormat,
)

__all__ = [
    "CAPABILITY_STOCK",
    "StockAsset",
    "StockMediaKind",
    "StockOrientation",
    "StockProvider",
    "StockRequest",
    "StockResponse",
    "StockResult",
]

CAPABILITY_STOCK = "stock"
"""The capability name a stock media provider declares."""


class StockMediaKind(StrEnum):
    """The kind of media a stock search returns."""

    IMAGE = "image"
    VIDEO = "video"


class StockOrientation(StrEnum):
    """The orientation filter a stock search may apply."""

    LANDSCAPE = "landscape"
    PORTRAIT = "portrait"
    SQUARE = "square"


def _strip(value: Any) -> Any:
    return value.strip() if isinstance(value, str) else value


def _strip_optional(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return value


class StockRequest(ProtocolModel):
    """A validated stock media search.

    ``per_page`` and ``page`` page through results. ``min_width`` and
    ``min_height`` filter by pixel size, while the duration bounds only apply
    to video. A duration range must be ordered.
    """

    query: str = Field(min_length=1)
    kind: StockMediaKind = StockMediaKind.IMAGE
    orientation: StockOrientation | None = None
    min_width: int | None = Field(default=None, gt=0)
    min_height: int | None = Field(default=None, gt=0)
    min_duration_seconds: float | None = Field(default=None, ge=0.0)
    max_duration_seconds: float | None = Field(default=None, gt=0.0)
    color: str | None = None
    language: str | None = None
    safe_search: bool = True
    per_page: int = Field(default=20, ge=1, le=200)
    page: int = Field(default=1, ge=1)
    timeout_seconds: float | None = Field(default=None, gt=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("query", mode="before")
    @classmethod
    def _validate_query(cls, value: Any) -> Any:
        return _strip(value)

    @field_validator("color", "language", mode="before")
    @classmethod
    def _strip_optional_text(cls, value: Any) -> Any:
        return _strip_optional(value)

    @model_validator(mode="after")
    def _validate_duration_range(self) -> Self:
        if (
            self.min_duration_seconds is not None
            and self.max_duration_seconds is not None
            and self.min_duration_seconds > self.max_duration_seconds
        ):
            raise ValueError(
                "min_duration_seconds must not exceed max_duration_seconds"
            )
        return self


class StockAsset(ProtocolModel):
    """One licensed stock image or video and its required attribution.

    ``url`` is the downloadable asset, while ``preview_url`` and ``page_url``
    point at a smaller copy and the provider page. ``license`` and
    ``license_url`` name the terms and ``attribution`` holds the exact credit
    line a publisher must show.
    """

    id: str = Field(min_length=1)
    provider: str = ""
    kind: StockMediaKind = StockMediaKind.IMAGE
    url: str
    preview_url: str | None = None
    page_url: str | None = None
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    duration_seconds: float | None = Field(default=None, ge=0.0)
    format: ImageFormat | VideoFormat | None = None
    creator: str | None = None
    creator_url: str | None = None
    license: str | None = None
    license_url: str | None = None
    attribution: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "provider", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _strip(value)

    @field_validator("url")
    @classmethod
    def _validate_url(cls, value: str) -> str:
        return validate_http_url(value, field="stock asset url")

    @field_validator("preview_url", "page_url", "creator_url", "license_url")
    @classmethod
    def _validate_optional_urls(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_http_url(value, field="stock asset url")

    @field_validator(
        "creator",
        "license",
        "attribution",
        mode="before",
    )
    @classmethod
    def _strip_optional_text(cls, value: Any) -> Any:
        return _strip_optional(value)

    @model_validator(mode="after")
    def _validate_size(self) -> Self:
        if (self.width is None) != (self.height is None):
            raise ValueError("stock asset width and height must be set together")
        return self

    @property
    def dimensions(self) -> MediaDimensions | None:
        """The pixel size when the provider reported both dimensions."""
        if self.width is None or self.height is None:
            return None
        return MediaDimensions(width=self.width, height=self.height)

    @property
    def is_video(self) -> bool:
        """Whether the asset is a video."""
        return self.kind is StockMediaKind.VIDEO

    @property
    def has_attribution(self) -> bool:
        """Whether the asset carries a credit line."""
        return bool(self.attribution)


StockResult = StockAsset
"""Alias for :class:`StockAsset` matching the search result naming."""


class StockResponse(ProtocolModel):
    """A validated stock media search response from one provider."""

    query: str = Field(min_length=1)
    assets: list[StockAsset] = Field(default_factory=list)
    provider: str = ""
    total_results: int | None = Field(default=None, ge=0)
    page: int | None = Field(default=None, ge=1)
    next_page: int | None = Field(default=None, ge=1)
    request_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("query", mode="before")
    @classmethod
    def _validate_query(cls, value: Any) -> Any:
        return _strip(value)

    @field_validator("provider", mode="before")
    @classmethod
    def _strip_provider(cls, value: Any) -> Any:
        return _strip(value)

    @field_validator("request_id", mode="before")
    @classmethod
    def _strip_request_id(cls, value: Any) -> Any:
        return _strip_optional(value)

    @property
    def urls(self) -> list[str]:
        """Every download URL in response order."""
        return [asset.url for asset in self.assets]

    @property
    def count(self) -> int:
        """The number of returned assets."""
        return len(self.assets)

    @property
    def first(self) -> StockAsset | None:
        """The first asset, or ``None`` when the search found nothing."""
        return self.assets[0] if self.assets else None


@runtime_checkable
class StockProvider(CapabilityProvider, Protocol):
    """A provider that serves the ``stock`` capability."""

    async def search(self, request: StockRequest) -> StockResponse:
        """Search *request* and return validated licensed assets."""
