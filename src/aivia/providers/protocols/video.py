"""Typed video generation protocol, request and response models.

A video provider turns a validated prompt into one or more clips. The request
carries the output duration, size, frame rate and audio intent explicitly so
the media service can estimate cost and adapt unsupported values before the
call.

Optional first and last frames use :class:`MediaInput`. A provider that cannot
consume a frame must reject the request instead of silently ignoring it.
"""

from __future__ import annotations

from typing import Any, Protocol, Self, runtime_checkable

from pydantic import Field, field_validator, model_validator

from aivia.providers.protocols.base import (
    CapabilityProvider,
    ProtocolModel,
    validate_http_url,
)
from aivia.providers.protocols.image import normalize_aspect_ratio
from aivia.providers.protocols.media import MediaDimensions, MediaInput, VideoFormat

__all__ = [
    "CAPABILITY_VIDEO_GEN",
    "VideoProvider",
    "VideoRequest",
    "VideoResponse",
    "VideoResult",
]

CAPABILITY_VIDEO_GEN = "video_gen"
"""The capability name a video generation provider declares."""


def _strip(value: Any) -> Any:
    return value.strip() if isinstance(value, str) else value


def _strip_optional(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return value


class VideoRequest(ProtocolModel):
    """A validated request to generate one or more video clips.

    ``width`` and ``height`` must be supplied together; ``aspect_ratio`` is the
    alternative when an exact size is not required. ``generate_audio`` asks for
    a native soundtrack and ``audio_prompt`` describes it when the provider
    supports audio direction.
    """

    prompt: str = Field(min_length=1)
    negative_prompt: str | None = None
    model: str | None = None
    first_frame: MediaInput | None = None
    last_frame: MediaInput | None = None
    duration_seconds: float | None = Field(default=None, gt=0.0)
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    aspect_ratio: str | None = None
    fps: int | None = Field(default=None, gt=0, le=120)
    format: VideoFormat = VideoFormat.MP4
    seed: int | None = None
    count: int = Field(default=1, ge=1, le=4)
    generate_audio: bool = False
    audio_prompt: str | None = None
    timeout_seconds: float | None = Field(default=None, gt=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("prompt", mode="before")
    @classmethod
    def _validate_prompt(cls, value: Any) -> Any:
        return _strip(value)

    @field_validator(
        "negative_prompt",
        "model",
        "audio_prompt",
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
    def _validate_size_and_audio(self) -> Self:
        if (self.width is None) != (self.height is None):
            raise ValueError("video width and height must be set together")
        if self.audio_prompt and not self.generate_audio:
            raise ValueError("audio_prompt requires generate_audio")
        return self


class VideoResult(ProtocolModel):
    """One generated video clip with its format, size and audio state."""

    data: bytes | None = None
    url: str | None = None
    format: VideoFormat = VideoFormat.MP4
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    duration_seconds: float | None = Field(default=None, gt=0.0)
    fps: int | None = Field(default=None, gt=0, le=120)
    has_audio: bool = False
    seed: int | None = None
    thumbnail_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("data")
    @classmethod
    def _validate_data(cls, value: bytes | None) -> bytes | None:
        if value is not None and not value:
            raise ValueError("video data must be non-empty")
        return value

    @field_validator("url", "thumbnail_url")
    @classmethod
    def _validate_urls(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_http_url(value, field="video url")

    @model_validator(mode="after")
    def _validate_source_and_size(self) -> Self:
        sources = (self.data is not None, self.url is not None)
        if sum(sources) != 1:
            raise ValueError("video result needs exactly one of data or url")
        if (self.width is None) != (self.height is None):
            raise ValueError("video width and height must be set together")
        return self

    @property
    def is_inline(self) -> bool:
        """Whether the clip carries inline bytes rather than a URL."""
        return self.data is not None

    @property
    def size_bytes(self) -> int | None:
        """The size of inline data, or ``None`` for a remote clip."""
        if self.data is None:
            return None
        return len(self.data)

    @property
    def dimensions(self) -> MediaDimensions | None:
        """The pixel size when the provider reported both dimensions."""
        if self.width is None or self.height is None:
            return None
        return MediaDimensions(width=self.width, height=self.height)


class VideoResponse(ProtocolModel):
    """A validated video generation response from one provider."""

    videos: list[VideoResult] = Field(min_length=1)
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
    def first(self) -> VideoResult:
        """The first generated clip."""
        return self.videos[0]

    @property
    def count(self) -> int:
        """The number of generated clips."""
        return len(self.videos)


@runtime_checkable
class VideoProvider(CapabilityProvider, Protocol):
    """A provider that serves the ``video_gen`` capability."""

    async def generate(self, request: VideoRequest) -> VideoResponse:
        """Generate *request* and return validated video clips."""
