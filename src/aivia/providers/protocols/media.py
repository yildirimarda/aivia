"""Shared value objects for the media generation protocols.

Image, image edit, video, music, sfx and stock all describe media in terms of a
container format, pixel dimensions and an input that is either inline bytes or
a URL. Keeping those models here means an image produced by one protocol can be
fed to another without an adapter inventing a second shape.

Sizes are pixels, durations are seconds and every URL is an absolute http(s)
URL. A media input carries exactly one source so a provider never has to guess
whether inline bytes or a remote location wins.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Self

from pydantic import Field, field_validator, model_validator

from aivia.providers.protocols.base import ProtocolModel, validate_http_url

__all__ = [
    "ImageFormat",
    "MediaDimensions",
    "MediaInput",
    "VideoFormat",
]


class ImageFormat(StrEnum):
    """A container or encoding for generated or edited images."""

    PNG = "png"
    JPEG = "jpeg"
    WEBP = "webp"
    GIF = "gif"
    AVIF = "avif"
    TIFF = "tiff"


class VideoFormat(StrEnum):
    """A container or encoding for generated video."""

    MP4 = "mp4"
    WEBM = "webm"
    MOV = "mov"
    MKV = "mkv"
    GIF = "gif"


class MediaDimensions(ProtocolModel):
    """A validated pixel size with the usual orientation helpers."""

    width: int = Field(gt=0)
    height: int = Field(gt=0)

    @property
    def aspect_ratio(self) -> float:
        """The width divided by the height."""
        return self.width / self.height

    @property
    def pixels(self) -> int:
        """The total number of pixels."""
        return self.width * self.height

    @property
    def is_landscape(self) -> bool:
        """Whether the media is wider than it is tall."""
        return self.width > self.height

    @property
    def is_portrait(self) -> bool:
        """Whether the media is taller than it is wide."""
        return self.width < self.height

    @property
    def is_square(self) -> bool:
        """Whether the media has equal width and height."""
        return self.width == self.height


class MediaInput(ProtocolModel):
    """An image, frame or mask supplied to a provider.

    ``data`` holds inline bytes and ``url`` points at a remote file; exactly one
    must be set. ``media_type`` is optional but, when given, must be a media
    MIME type so a provider never receives a mistyped file as an image.
    """

    data: bytes | None = None
    url: str | None = None
    media_type: str | None = None
    filename: str | None = None

    @field_validator("data")
    @classmethod
    def _validate_data(cls, value: bytes | None) -> bytes | None:
        if value is not None and not value:
            raise ValueError("media data must be non-empty")
        return value

    @field_validator("url")
    @classmethod
    def _validate_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_http_url(value, field="media url")

    @field_validator("media_type", mode="before")
    @classmethod
    def _normalize_media_type(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip().lower()
            return text or None
        return value

    @field_validator("media_type")
    @classmethod
    def _validate_media_type(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.startswith(("image/", "video/", "audio/")):
            raise ValueError("media_type must be an image, video or audio type")
        return value

    @field_validator("filename", mode="before")
    @classmethod
    def _strip_filename(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip()
            return text or None
        return value

    @model_validator(mode="after")
    def _validate_source(self) -> Self:
        sources = (self.data is not None, self.url is not None)
        if sum(sources) != 1:
            raise ValueError("media input needs exactly one of data or url")
        return self

    @property
    def is_inline(self) -> bool:
        """Whether the input carries inline bytes rather than a URL."""
        return self.data is not None

    @property
    def size_bytes(self) -> int | None:
        """The size of inline data, or ``None`` for a remote input."""
        if self.data is None:
            return None
        return len(self.data)
