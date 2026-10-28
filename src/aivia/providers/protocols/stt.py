"""Typed speech to text protocol, request and response models.

A speech to text provider accepts inline audio or a URL and returns a
transcript with optional phrase segments and word level timestamps. Word
timing is a first class part of the protocol because subtitles, forced
alignment and name verification all depend on it.

The request carries only execution options. Language detection, model choice
and cost accounting happen before or after the call, never inside an adapter.
"""

from __future__ import annotations

from typing import Any, Protocol, Self, runtime_checkable

from pydantic import Field, field_validator, model_validator

from aivia.providers.protocols.audio import TranscriptSegment, WordTimestamp
from aivia.providers.protocols.base import (
    CapabilityProvider,
    ProtocolModel,
    validate_http_url,
)

__all__ = [
    "CAPABILITY_STT",
    "STTProvider",
    "STTRequest",
    "STTResponse",
]

CAPABILITY_STT = "stt"
"""The capability name a speech to text provider declares."""


class STTRequest(ProtocolModel):
    """A validated request to transcribe one audio source.

    Exactly one of ``audio`` and ``audio_url`` must be set. ``media_type`` is
    optional but, when given, must describe audio or video so a provider never
    receives a mistyped file as speech.
    """

    audio: bytes | None = None
    audio_url: str | None = None
    media_type: str | None = None
    filename: str | None = None
    model: str | None = None
    language: str | None = None
    prompt: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=1.0)
    word_timestamps: bool = True
    diarize: bool = False
    timeout_seconds: float | None = Field(default=None, gt=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("audio")
    @classmethod
    def _validate_audio(cls, value: bytes | None) -> bytes | None:
        if value is not None and not value:
            raise ValueError("audio must be non-empty")
        return value

    @field_validator("audio_url")
    @classmethod
    def _validate_audio_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_http_url(value, field="audio url")

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
        if not value.startswith(("audio/", "video/")):
            raise ValueError("media_type must start with 'audio/' or 'video/'")
        return value

    @field_validator(
        "filename",
        "model",
        "prompt",
        mode="before",
    )
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

    @model_validator(mode="after")
    def _validate_source(self) -> Self:
        sources = (self.audio is not None, self.audio_url is not None)
        if sum(sources) != 1:
            raise ValueError(
                "speech to text request needs exactly one of audio or audio_url"
            )
        return self


class STTResponse(ProtocolModel):
    """A validated transcription from one speech to text provider.

    ``words`` holds provider level word timing; when a provider only reports
    timing inside ``segments``, :attr:`all_words` flattens the segments so
    callers do not need to know which shape the adapter returned.
    """

    text: str = ""
    language: str | None = None
    duration_seconds: float | None = Field(default=None, ge=0.0)
    segments: list[TranscriptSegment] = Field(default_factory=list)
    words: list[WordTimestamp] = Field(default_factory=list)
    model: str = ""
    provider: str = ""
    request_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text", "model", "provider", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("language", mode="before")
    @classmethod
    def _normalize_language(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip().lower()
            return text or None
        return value

    @field_validator("request_id", mode="before")
    @classmethod
    def _strip_request_id(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip()
            return text or None
        return value

    @model_validator(mode="after")
    def _validate_order(self) -> Self:
        starts = [word.start_seconds for word in self.words]
        if starts != sorted(starts):
            raise ValueError("word timestamps must be ordered by start_seconds")
        segment_starts = [segment.start_seconds for segment in self.segments]
        if segment_starts != sorted(segment_starts):
            raise ValueError("segments must be ordered by start_seconds")
        return self

    @property
    def all_words(self) -> list[WordTimestamp]:
        """Provider words, or the words flattened from segments."""
        if self.words:
            return list(self.words)
        return [word for segment in self.segments for word in segment.words]

    @property
    def word_count(self) -> int:
        """The number of words with timing information."""
        return len(self.all_words)

    @property
    def has_word_timestamps(self) -> bool:
        """Whether any word level timing is available."""
        return bool(self.all_words)


@runtime_checkable
class STTProvider(CapabilityProvider, Protocol):
    """A provider that serves the ``stt`` capability."""

    async def transcribe(self, request: STTRequest) -> STTResponse:
        """Transcribe *request* and return a validated transcript."""
