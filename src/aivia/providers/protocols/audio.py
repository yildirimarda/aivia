"""Shared audio value objects for the speech protocols.

Text to speech and speech to text both describe audio in terms of a container
format and word level timing. Keeping those models in one place means a
transcript produced by speech to text can be consumed by subtitles and
alignment without an adapter inventing a second shape.

Times are seconds from the start of the audio, never wall-clock timestamps.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Self

from pydantic import Field, field_validator, model_validator

from aivia.providers.protocols.base import ProtocolModel

__all__ = [
    "AudioFormat",
    "TranscriptSegment",
    "WordTimestamp",
]


class AudioFormat(StrEnum):
    """A container or encoding for generated or transcribed audio."""

    MP3 = "mp3"
    WAV = "wav"
    OGG = "ogg"
    OPUS = "opus"
    AAC = "aac"
    FLAC = "flac"
    PCM = "pcm"
    M4A = "m4a"


class WordTimestamp(ProtocolModel):
    """One spoken word with its start and end time in seconds.

    ``confidence`` is optional because not every provider reports one. A
    provider must never emit an end before its start; callers can rely on
    :attr:`duration_seconds` being non-negative.
    """

    word: str = Field(min_length=1)
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(ge=0.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    speaker: str | None = None

    @field_validator("word", "speaker", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip()
            return text or None
        return value

    @model_validator(mode="after")
    def _validate_span(self) -> Self:
        if self.end_seconds < self.start_seconds:
            raise ValueError("word end_seconds must not be before start_seconds")
        return self

    @property
    def duration_seconds(self) -> float:
        """The length of the spoken word in seconds."""
        return self.end_seconds - self.start_seconds

    @property
    def text(self) -> str:
        """Alias for :attr:`word`."""
        return self.word


class TranscriptSegment(ProtocolModel):
    """A contiguous phrase of a transcript and the words inside it.

    Segments are what subtitle and alignment stages consume; ``words`` may be
    empty when a provider only reports phrase level timing.
    """

    text: str = Field(min_length=1)
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(ge=0.0)
    words: list[WordTimestamp] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    speaker: str | None = None

    @field_validator("text", "speaker", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip()
            return text or None
        return value

    @model_validator(mode="after")
    def _validate_span(self) -> Self:
        if self.end_seconds < self.start_seconds:
            raise ValueError("segment end_seconds must not be before start_seconds")
        starts = [word.start_seconds for word in self.words]
        if starts != sorted(starts):
            raise ValueError("segment words must be ordered by start_seconds")
        return self

    @property
    def duration_seconds(self) -> float:
        """The length of the segment in seconds."""
        return self.end_seconds - self.start_seconds
