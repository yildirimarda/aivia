"""Typed image generation protocol, request and response models.

An image provider turns a validated prompt and explicit output options into one
or more images. Prompt enhancement, size adaptation, variation policy and cost
accounting happen before or after the call, never inside an adapter.

An image result carries either inline bytes or a URL, never both, so the media
service can validate and store it without knowing which shape the provider
prefers.
"""

from __future__ import annotations

import re
from typing import Any, Protocol, Self, runtime_checkable

from pydantic import Field, field_validator, model_validator

from aivia.providers.protocols.base import (
    CapabilityProvider,
    ProtocolModel,
    validate_http_url,
)
from aivia.providers.protocols.media import ImageFormat, MediaDimensions, MediaInput

__all__ = [
    "CAPABILITY_IMAGE_GEN",
    "CAPABILITY_IMAGE_INPUT",
    "ImageProvider",
    "ImageRequest",
    "ImageResponse",
    "ImageResult",
]

CAPABILITY_IMAGE_GEN = "image_gen"
"""The capability name an image generation provider declares."""

CAPABILITY_IMAGE_INPUT = "image_input"
"""The capability a provider declares when it accepts reference images."""

_ASPECT_RATIO = re.compile(r"^(?P<width>\d+)\s*:\s*(?P<height>\d+)$")


def _strip(value: Any) -> Any:
    return value.strip() if isinstance(value, str) else value


def _strip_optional(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return value


def normalize_aspect_ratio(value: str) -> str:
    """Normalize an aspect ratio to ``WIDTH:HEIGHT`` with positive terms."""
    text = value.strip()
    parsed = _ASPECT_RATIO.fullmatch(text)
    if parsed is None:
        raise ValueError("aspect_ratio must be WIDTH:HEIGHT, for example 16:9")
    width = int(parsed.group("width"))
    height = int(parsed.group("height"))
    if width <= 0 or height <= 0:
        raise ValueError("aspect ratio terms must be positive")
    return f"{width}:{height}"


class ImageRequest(ProtocolModel):
    """A validated request to generate one or more images.

    ``width`` and ``height`` describe an exact pixel size and must be supplied
    together. ``aspect_ratio`` is an alternative when the caller only cares
    about the shape. ``count`` is the number of variations requested, and
    ``reference_images`` carries optional visual references for providers that
    support them.
    """

    prompt: str = Field(min_length=1)
    negative_prompt: str | None = None
    model: str | None = None
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    aspect_ratio: str | None = None
    format: ImageFormat = ImageFormat.PNG
    quality: str | None = None
    style: str | None = None
    seed: int | None = None
    count: int = Field(default=1, ge=1, le=8)
    reference_images: list[MediaInput] = Field(default_factory=list)
    timeout_seconds: float | None = Field(default=None, gt=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("prompt", mode="before")
    @classmethod
    def _validate_prompt(cls, value: Any) -> Any:
        return _strip(value)

    @field_validator(
        "negative_prompt",
        "model",
        "quality",
        "style",
        mode="before",
    )
    @classmethod
    def _strip_optional_text(cls, value: Any) -> Any:
        return _strip_optional(value)

    @field_validator("aspect_ratio", mode="before")
    @classmethod
    def _normalize_aspect_ratio(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip()
            return normalize_aspect_ratio(text) if text else None
        return value

    @model_validator(mode="after")
    def _validate_size(self) -> Self:
        if (self.width is None) != (self.height is None):
            raise ValueError("image width and height must be set together")
        return self


class ImageResult(ProtocolModel):
    """One generated image with its format, size and provenance."""

    data: bytes | None = None
    url: str | None = None
    format: ImageFormat = ImageFormat.PNG
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    seed: int | None = None
    revised_prompt: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("data")
    @classmethod
    def _validate_data(cls, value: bytes | None) -> bytes | None:
        if value is not None and not value:
            raise ValueError("image data must be non-empty")
        return value

    @field_validator("url")
    @classmethod
    def _validate_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_http_url(value, field="image url")

    @field_validator("revised_prompt", mode="before")
    @classmethod
    def _strip_revised_prompt(cls, value: Any) -> Any:
        return _strip_optional(value)

    @model_validator(mode="after")
    def _validate_source_and_size(self) -> Self:
        sources = (self.data is not None, self.url is not None)
        if sum(sources) != 1:
            raise ValueError("image result needs exactly one of data or url")
        if (self.width is None) != (self.height is None):
            raise ValueError("image width and height must be set together")
        return self

    @property
    def is_inline(self) -> bool:
        """Whether the image carries inline bytes rather than a URL."""
        return self.data is not None

    @property
    def size_bytes(self) -> int | None:
        """The size of inline data, or ``None`` for a remote image."""
        if self.data is None:
            return None
        return len(self.data)

    @property
    def dimensions(self) -> MediaDimensions | None:
        """The pixel size when the provider reported both dimensions."""
        if self.width is None or self.height is None:
            return None
        return MediaDimensions(width=self.width, height=self.height)


class ImageResponse(ProtocolModel):
    """A validated image generation response from one provider."""

    images: list[ImageResult] = Field(min_length=1)
    model: str = ""
    provider: str = ""
    request_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("model", "provider", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _strip(value)

    @field_validator("request_id", mode="before")
    @classmethod
    def _strip_request_id(cls, value: Any) -> Any:
        return _strip_optional(value)

    @property
    def first(self) -> ImageResult:
        """The first generated image."""
        return self.images[0]

    @property
    def count(self) -> int:
        """The number of generated images."""
        return len(self.images)


@runtime_checkable
class ImageProvider(CapabilityProvider, Protocol):
    """A provider that serves the ``image_gen`` capability."""

    async def generate(self, request: ImageRequest) -> ImageResponse:
        """Generate *request* and return validated images."""
