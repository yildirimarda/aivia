"""Typed music generation protocol, request and response models.

A music provider turns a validated description and optional lyrics into one or
more audio tracks. Vocals, instrumentation, tempo and duration are explicit
because they change both the result and the price.

Tracks reuse :class:`AudioFormat`, so a generated song can be mixed, aligned
and delivered with the same audio helpers as speech and sound effects.
"""

from __future__ import annotations

from typing import Any, Protocol, Self, runtime_checkable

from pydantic import Field, field_validator, model_validator

from aivia.providers.protocols.audio import AudioFormat
from aivia.providers.protocols.base import (
    CapabilityProvider,
    ProtocolModel,
    validate_http_url,
)

__all__ = [
    "CAPABILITY_MUSIC_GEN",
    "MusicProvider",
    "MusicRequest",
    "MusicResponse",
    "MusicTrack",
]

CAPABILITY_MUSIC_GEN = "music_gen"
"""The capability name a music generation provider declares."""


def _strip(value: Any) -> Any:
    return value.strip() if isinstance(value, str) else value


def _strip_optional(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return value


class MusicRequest(ProtocolModel):
    """A validated request to generate music.

    ``prompt`` describes the mood, genre and instrumentation. ``lyrics`` may be
    supplied for a vocal track and must be omitted for an instrumental one.
    ``count`` asks for several candidates so a later stage can prefilter them.
    """

    prompt: str = Field(min_length=1)
    lyrics: str | None = None
    instrumental: bool = False
    model: str | None = None
    duration_seconds: float | None = Field(default=None, gt=0.0)
    format: AudioFormat = AudioFormat.MP3
    sample_rate: int | None = Field(default=None, gt=0)
    tempo_bpm: float | None = Field(default=None, gt=0.0)
    style: str | None = None
    seed: int | None = None
    count: int = Field(default=1, ge=1, le=8)
    timeout_seconds: float | None = Field(default=None, gt=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("prompt", mode="before")
    @classmethod
    def _validate_prompt(cls, value: Any) -> Any:
        return _strip(value)

    @field_validator("lyrics", "model", "style", mode="before")
    @classmethod
    def _strip_optional_text(cls, value: Any) -> Any:
        return _strip_optional(value)

    @model_validator(mode="after")
    def _validate_vocals(self) -> Self:
        if self.instrumental and self.lyrics:
            raise ValueError("an instrumental track must not carry lyrics")
        return self


class MusicTrack(ProtocolModel):
    """One generated music track with its format and provenance."""

    data: bytes | None = None
    url: str | None = None
    format: AudioFormat = AudioFormat.MP3
    duration_seconds: float | None = Field(default=None, ge=0.0)
    sample_rate: int | None = Field(default=None, gt=0)
    channels: int | None = Field(default=None, gt=0)
    seed: int | None = None
    title: str | None = None
    lyrics: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("data")
    @classmethod
    def _validate_data(cls, value: bytes | None) -> bytes | None:
        if value is not None and not value:
            raise ValueError("music data must be non-empty")
        return value

    @field_validator("url")
    @classmethod
    def _validate_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_http_url(value, field="music url")

    @field_validator("title", "lyrics", mode="before")
    @classmethod
    def _strip_optional_text(cls, value: Any) -> Any:
        return _strip_optional(value)

    @model_validator(mode="after")
    def _validate_source(self) -> Self:
        sources = (self.data is not None, self.url is not None)
        if sum(sources) != 1:
            raise ValueError("music track needs exactly one of data or url")
        return self

    @property
    def is_inline(self) -> bool:
        """Whether the track carries inline bytes rather than a URL."""
        return self.data is not None

    @property
    def size_bytes(self) -> int | None:
        """The size of inline data, or ``None`` for a remote track."""
        if self.data is None:
            return None
        return len(self.data)

    @property
    def has_lyrics(self) -> bool:
        """Whether the track carries lyrics."""
        return bool(self.lyrics)


class MusicResponse(ProtocolModel):
    """A validated music generation response from one provider."""

    tracks: list[MusicTrack] = Field(min_length=1)
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
    def first(self) -> MusicTrack:
        """The first generated track."""
        return self.tracks[0]

    @property
    def count(self) -> int:
        """The number of generated tracks."""
        return len(self.tracks)

    @property
    def has_lyrics(self) -> bool:
        """Whether any generated track carries lyrics."""
        return any(track.has_lyrics for track in self.tracks)


@runtime_checkable
class MusicProvider(CapabilityProvider, Protocol):
    """A provider that serves the ``music_gen`` capability."""

    async def generate(self, request: MusicRequest) -> MusicResponse:
        """Generate *request* and return validated music tracks."""
