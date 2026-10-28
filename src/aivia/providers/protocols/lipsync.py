"""Typed lip sync protocol, request and result models.

A lip sync provider animates a character's mouth from an audio track. The
source may be a still image, a video or a sprite sheet; a sprite sheet is the
no-AI fallback path, where the adapter swaps mouth frames in time with the
audio and adds blinking and idle motion.

The audio is always supplied. A source or a sprite sheet must be present, and
the request rejects an audio-only source so a provider never receives a file it
cannot animate.
"""

from __future__ import annotations

from typing import Any, Protocol, Self, runtime_checkable

from pydantic import Field, field_validator, model_validator

from aivia.providers.protocols.base import (
    CapabilityProvider,
    ProtocolModel,
    validate_http_url,
)
from aivia.providers.protocols.media import MediaDimensions, MediaInput, VideoFormat

__all__ = [
    "CAPABILITY_LIPSYNC",
    "LipSyncProvider",
    "LipSyncRequest",
    "LipSyncResponse",
    "LipSyncResult",
    "LipSyncSpriteSheet",
]

CAPABILITY_LIPSYNC = "lipsync"
"""The capability name a lip sync provider declares."""


class LipSyncSpriteSheet(ProtocolModel):
    """A grid of character frames used by the no-AI animation fallback.

    ``mouth_frames`` maps a mouth shape or viseme name to a frame index. Every
    index must fall inside ``frame_count`` so an adapter never seeks past the
    sheet.
    """

    image: MediaInput
    frame_width: int = Field(gt=0)
    frame_height: int = Field(gt=0)
    frame_count: int = Field(ge=1)
    fps: int | None = Field(default=None, gt=0, le=120)
    mouth_frames: dict[str, int] = Field(default_factory=dict)
    blink_frame: int | None = Field(default=None, ge=0)
    idle_frames: list[int] = Field(default_factory=list)

    @field_validator("image")
    @classmethod
    def _validate_image(cls, value: MediaInput) -> MediaInput:
        if value.media_type is not None and not value.media_type.startswith("image/"):
            raise ValueError("sprite sheet image must be an image")
        return value

    @field_validator("mouth_frames")
    @classmethod
    def _validate_mouth_frames(cls, value: dict[str, int]) -> dict[str, int]:
        frames: dict[str, int] = {}
        for raw_key, index in value.items():
            key = raw_key.strip()
            if not key:
                raise ValueError("mouth frame names must be non-empty")
            if index < 0:
                raise ValueError("mouth frame index must be non-negative")
            frames[key] = index
        return frames

    @model_validator(mode="after")
    def _validate_frame_indices(self) -> Self:
        indices = list(self.mouth_frames.values()) + list(self.idle_frames)
        if self.blink_frame is not None:
            indices.append(self.blink_frame)
        if any(index >= self.frame_count for index in indices):
            raise ValueError("frame index must be less than frame_count")
        return self


class LipSyncRequest(ProtocolModel):
    """A validated request to animate a character from an audio track."""

    audio: MediaInput
    source: MediaInput | None = None
    sprite_sheet: LipSyncSpriteSheet | None = None
    format: VideoFormat = VideoFormat.MP4
    fps: int = Field(default=30, gt=0, le=120)
    duration_seconds: float | None = Field(default=None, gt=0.0)
    smooth: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("audio")
    @classmethod
    def _validate_audio(cls, value: MediaInput) -> MediaInput:
        if value.media_type is not None and not value.media_type.startswith("audio/"):
            raise ValueError("lip sync audio must be an audio type")
        return value

    @field_validator("source")
    @classmethod
    def _validate_source(cls, value: MediaInput | None) -> MediaInput | None:
        if value is None:
            return None
        if value.media_type is not None and not value.media_type.startswith(
            ("image/", "video/")
        ):
            raise ValueError("lip sync source must be an image or video")
        return value

    @model_validator(mode="after")
    def _validate_visual(self) -> Self:
        if self.source is None and self.sprite_sheet is None:
            raise ValueError("lip sync needs a source or a sprite sheet")
        return self


class LipSyncResult(ProtocolModel):
    """One animated clip carrying inline bytes or a URL, never both."""

    data: bytes | None = None
    url: str | None = None
    format: VideoFormat = VideoFormat.MP4
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    duration_seconds: float | None = Field(default=None, gt=0.0)
    fps: int | None = Field(default=None, gt=0, le=120)
    used_fallback: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("data")
    @classmethod
    def _validate_data(cls, value: bytes | None) -> bytes | None:
        if value is not None and not value:
            raise ValueError("lip sync data must be non-empty")
        return value

    @field_validator("url")
    @classmethod
    def _validate_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_http_url(value, field="lip sync url")

    @model_validator(mode="after")
    def _validate_source_and_size(self) -> Self:
        sources = (self.data is not None, self.url is not None)
        if sum(sources) != 1:
            raise ValueError("lip sync result needs exactly one of data or url")
        if (self.width is None) != (self.height is None):
            raise ValueError("lip sync width and height must be set together")
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


class LipSyncResponse(ProtocolModel):
    """A validated lip sync response from one provider."""

    video: LipSyncResult
    model: str = ""
    provider: str = ""
    request_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("model", "provider", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator("request_id", mode="before")
    @classmethod
    def _strip_request_id(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip()
            return text or None
        return value


@runtime_checkable
class LipSyncProvider(CapabilityProvider, Protocol):
    """A provider that serves the ``lipsync`` capability."""

    async def animate(self, request: LipSyncRequest) -> LipSyncResponse:
        """Animate *request* and return the resulting clip."""
