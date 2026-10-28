"""Typed sound effect protocol, request and response models.

A sound effect provider turns a validated description into one or more short
audio clips. Unlike music, a sound effect is usually described by its source
and action rather than lyrics or tempo, and looping is an explicit request
option.

Clips reuse :class:`AudioFormat` so the audio helpers can process them the same
way as speech and music.
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
    "CAPABILITY_SFX",
    "SFXClip",
    "SFXProvider",
    "SFXRequest",
    "SFXResponse",
]

CAPABILITY_SFX = "sfx"
"""The capability name a sound effect provider declares."""


def _strip(value: Any) -> Any:
    return value.strip() if isinstance(value, str) else value


def _strip_optional(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return value


class SFXRequest(ProtocolModel):
    """A validated request to generate sound effects.

    ``duration_seconds`` is optional because a provider may pick a natural
    length. ``loop`` asks for a seamless loop when the provider supports it,
    and ``count`` asks for several candidates.
    """

    prompt: str = Field(min_length=1)
    model: str | None = None
    duration_seconds: float | None = Field(default=None, gt=0.0)
    format: AudioFormat = AudioFormat.MP3
    sample_rate: int | None = Field(default=None, gt=0)
    loop: bool = False
    seed: int | None = None
    count: int = Field(default=1, ge=1, le=8)
    timeout_seconds: float | None = Field(default=None, gt=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("prompt", mode="before")
    @classmethod
    def _validate_prompt(cls, value: Any) -> Any:
        return _strip(value)

    @field_validator("model", mode="before")
    @classmethod
    def _strip_model(cls, value: Any) -> Any:
        return _strip_optional(value)


class SFXClip(ProtocolModel):
    """One generated sound effect with its format and provenance."""

    data: bytes | None = None
    url: str | None = None
    format: AudioFormat = AudioFormat.MP3
    duration_seconds: float | None = Field(default=None, ge=0.0)
    sample_rate: int | None = Field(default=None, gt=0)
    channels: int | None = Field(default=None, gt=0)
    loop: bool = False
    seed: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("data")
    @classmethod
    def _validate_data(cls, value: bytes | None) -> bytes | None:
        if value is not None and not value:
            raise ValueError("sound effect data must be non-empty")
        return value

    @field_validator("url")
    @classmethod
    def _validate_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_http_url(value, field="sound effect url")

    @model_validator(mode="after")
    def _validate_source(self) -> Self:
        sources = (self.data is not None, self.url is not None)
        if sum(sources) != 1:
            raise ValueError("sound effect clip needs exactly one of data or url")
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


class SFXResponse(ProtocolModel):
    """A validated sound effect response from one provider."""

    clips: list[SFXClip] = Field(min_length=1)
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
    def first(self) -> SFXClip:
        """The first generated clip."""
        return self.clips[0]

    @property
    def count(self) -> int:
        """The number of generated clips."""
        return len(self.clips)


@runtime_checkable
class SFXProvider(CapabilityProvider, Protocol):
    """A provider that serves the ``sfx`` capability."""

    async def generate(self, request: SFXRequest) -> SFXResponse:
        """Generate *request* and return validated sound effect clips."""
