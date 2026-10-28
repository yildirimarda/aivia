"""Typed text to speech protocol, request and response models.

A text to speech provider turns validated text and a resolved voice into an
audio payload. Voice selection, per-language mapping and pronunciation hints
are resolved before the call; the adapter only executes the request.

The response carries the encoded audio bytes, the container format, duration
and, when the provider supports it, word timestamps that later stages can use
for subtitles and alignment.
"""

from __future__ import annotations

from typing import Any, Protocol, Self, runtime_checkable

from pydantic import Field, field_validator, model_validator

from aivia.providers.protocols.audio import AudioFormat, WordTimestamp
from aivia.providers.protocols.base import CapabilityProvider, ProtocolModel

__all__ = [
    "CAPABILITY_TTS",
    "TTSProvider",
    "TTSRequest",
    "TTSResponse",
]

CAPABILITY_TTS = "tts"
"""The capability name a text to speech provider declares."""


class TTSRequest(ProtocolModel):
    """A validated request to synthesize speech.

    ``voice`` is already resolved to a provider voice id. ``pronunciation``
    maps a word or name to a phonetic spelling when the provider supports it.
    """

    text: str = Field(min_length=1)
    voice: str = Field(min_length=1)
    model: str | None = None
    language: str | None = None
    format: AudioFormat = AudioFormat.MP3
    sample_rate: int | None = Field(default=None, gt=0)
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    instructions: str | None = None
    pronunciation: dict[str, str] = Field(default_factory=dict)
    seed: int | None = None
    word_timestamps: bool = False
    timeout_seconds: float | None = Field(default=None, gt=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text", "voice", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("model", "instructions", mode="before")
    @classmethod
    def _strip_optional_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip()
            return text or None
        return value

    @field_validator("language", mode="before")
    @classmethod
    def _normalize_language(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip().lower()
            return text or None
        return value

    @field_validator("pronunciation", mode="before")
    @classmethod
    def _coerce_pronunciation(cls, value: Any) -> Any:
        if value is None:
            return {}
        return value

    @field_validator("pronunciation")
    @classmethod
    def _validate_pronunciation(cls, value: dict[str, str]) -> dict[str, str]:
        result: dict[str, str] = {}
        for key, raw in value.items():
            word = key.strip()
            if not word:
                raise ValueError("pronunciation keys must be non-empty")
            if not isinstance(raw, str):
                raise ValueError("pronunciation values must be strings")
            phonetic = raw.strip()
            if not phonetic:
                raise ValueError("pronunciation values must be non-empty")
            result[word] = phonetic
        return result


class TTSResponse(ProtocolModel):
    """A validated audio payload from one text to speech provider."""

    audio: bytes = Field(min_length=1)
    format: AudioFormat = AudioFormat.MP3
    sample_rate: int | None = Field(default=None, gt=0)
    duration_seconds: float | None = Field(default=None, ge=0.0)
    voice: str = ""
    model: str = ""
    provider: str = ""
    word_timestamps: list[WordTimestamp] = Field(default_factory=list)
    request_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("voice", "model", "provider", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("request_id", mode="before")
    @classmethod
    def _strip_request_id(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip()
            return text or None
        return value

    @model_validator(mode="after")
    def _validate_word_timestamps(self) -> Self:
        starts = [word.start_seconds for word in self.word_timestamps]
        if starts != sorted(starts):
            raise ValueError("word timestamps must be ordered by start_seconds")
        return self

    @property
    def size_bytes(self) -> int:
        """The size of the encoded audio payload."""
        return len(self.audio)

    @property
    def has_word_timestamps(self) -> bool:
        """Whether the provider returned word level timing."""
        return bool(self.word_timestamps)


@runtime_checkable
class TTSProvider(CapabilityProvider, Protocol):
    """A provider that serves the ``tts`` capability."""

    async def synthesize(self, request: TTSRequest) -> TTSResponse:
        """Synthesize *request* and return a validated audio response."""
