"""Composition-native subtitle values.

The renderer must stay importable without the media package, so a composition
cannot hold a :class:`aivia.media.subtitles.Subtitle` directly. These models
mirror the parts of that model a composition needs: timed words, cues, a style
mapping and a language. Conversion to the media subtitle model happens lazily,
only when a caller actually burns or exports the track.

Keeping the data here also means the composition JSON schema describes
subtitles without pulling provider or media modules into the render import
graph.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Self, cast

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from aivia.core.config.media import (
    DEFAULT_MEDIA_SUBTITLE_MAX_CHARS,
    DEFAULT_MEDIA_SUBTITLE_MAX_DURATION_SECONDS,
    DEFAULT_MEDIA_SUBTITLE_MAX_GAP_SECONDS,
)

__all__ = [
    "CompositionSubtitle",
    "Subtitle",
    "SubtitleCue",
    "SubtitleTrack",
    "SubtitleWord",
]


class SubtitleWord(BaseModel):
    """One timed word in a subtitle track."""

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        validate_assignment=True,
    )

    word: str = Field(
        min_length=1,
        validation_alias=AliasChoices("word", "text", "value"),
    )
    start_seconds: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices("start_seconds", "start", "start_time"),
    )
    end_seconds: float | None = Field(
        default=None,
        ge=0.0,
        validation_alias=AliasChoices("end_seconds", "end", "end_time"),
    )
    confidence: float | None = None

    @field_validator("word", mode="before")
    @classmethod
    def _strip_word(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _validate_span(self) -> Self:
        end = self.end_seconds
        if end is None:
            end = self.start_seconds
            object.__setattr__(self, "end_seconds", end)
        if end < self.start_seconds:
            raise ValueError("word end_seconds must not be before start_seconds")
        return self

    @property
    def text(self) -> str:
        """Alias for :attr:`word`."""
        return self.word

    @property
    def start(self) -> float:
        """Alias for :attr:`start_seconds`."""
        return self.start_seconds

    @property
    def end(self) -> float:
        """Alias for :attr:`end_seconds`."""
        return self.end_seconds or self.start_seconds


class SubtitleCue(BaseModel):
    """One subtitle cue with an inclusive start and end time."""

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        validate_assignment=True,
    )

    start_seconds: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices("start_seconds", "start", "start_time"),
    )
    end_seconds: float | None = Field(
        default=None,
        ge=0.0,
        validation_alias=AliasChoices("end_seconds", "end", "end_time"),
    )
    text: str = ""
    words: list[SubtitleWord] = Field(default_factory=list)
    speaker: str | None = None

    @field_validator("text", "speaker", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator("words", mode="before")
    @classmethod
    def _coerce_words(cls, value: Any) -> Any:
        return [] if value is None else value

    @model_validator(mode="after")
    def _derive_cue(self) -> Self:
        if not self.text and self.words:
            object.__setattr__(
                self,
                "text",
                " ".join(word.word for word in self.words),
            )
        end = self.end_seconds
        if end is None:
            end = (
                max(word.end_seconds or word.start_seconds for word in self.words)
                if self.words
                else self.start_seconds
            )
            object.__setattr__(self, "end_seconds", end)
        if not self.text:
            raise ValueError("subtitle cue needs text or words")
        if end < self.start_seconds:
            raise ValueError("cue end_seconds must not be before start_seconds")
        return self

    @classmethod
    def from_words(
        cls,
        words: Iterable[Any],
        *,
        start_seconds: float | None = None,
        end_seconds: float | None = None,
        text: str | None = None,
    ) -> Self:
        """Build a cue from timed words, deriving its span and text."""
        items = [SubtitleWord.model_validate(word) for word in words]
        if not items:
            raise ValueError("a cue needs at least one word")
        return cls(
            start_seconds=(
                items[0].start_seconds if start_seconds is None else start_seconds
            ),
            end_seconds=(
                max(word.end_seconds or word.start_seconds for word in items)
                if end_seconds is None
                else end_seconds
            ),
            text=" ".join(word.word for word in items) if text is None else text,
            words=items,
        )

    @property
    def start(self) -> float:
        """Alias for :attr:`start_seconds`."""
        return self.start_seconds

    @property
    def end(self) -> float:
        """Alias for :attr:`end_seconds`."""
        return self.end_seconds or self.start_seconds

    @property
    def duration_seconds(self) -> float:
        """The length of the cue in seconds."""
        return self.end - self.start_seconds


class SubtitleTrack(BaseModel):
    """A subtitle track with cues, timed words, a style and a language."""

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        validate_assignment=True,
    )

    cues: list[SubtitleCue] = Field(default_factory=list)
    words: list[SubtitleWord] = Field(default_factory=list)
    language: str = ""
    style: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _accept_shorthands(cls, data: Any) -> Any:
        if isinstance(data, BaseModel):
            return data.model_dump()
        if isinstance(data, (list, tuple)):
            return {"cues": list(data)}
        if isinstance(data, Mapping):
            values = dict(data)
            if not ({"cues", "words"} & set(values)) and (
                {"start", "start_seconds", "text"} & set(values)
            ):
                return {"cues": [values]}
            return values
        return data

    @field_validator("cues", "words", mode="before")
    @classmethod
    def _coerce_items(cls, value: Any) -> Any:
        return [] if value is None else value

    @field_validator("language", mode="before")
    @classmethod
    def _normalize_language(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _derive_cues(self) -> Self:
        if not self.cues and self.words:
            object.__setattr__(self, "cues", _group_words(self.words))
        if not self.words and self.cues:
            object.__setattr__(
                self,
                "words",
                [word for cue in self.cues for word in cue.words],
            )
        return self

    @classmethod
    def from_words(
        cls,
        words: Iterable[Any],
        *,
        language: str = "",
        style: Any = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Self:
        """Build a track from timed words, grouping them into cues."""
        return cls(
            words=list(words),
            language=language,
            style=style,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_cues(
        cls,
        cues: Iterable[Any],
        *,
        language: str = "",
        style: Any = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Self:
        """Build a track from cues."""
        return cls(
            cues=list(cues),
            language=language,
            style=style,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_transcript(cls, transcript: Any, **kwargs: Any) -> Self:
        """Build a track from a transcript mapping or word sequence."""
        if isinstance(transcript, Mapping):
            words = list(transcript.get("words") or [])
            if not words:
                words = [
                    word
                    for segment in transcript.get("segments") or []
                    for word in (
                        segment.get("words", [])
                        if isinstance(segment, Mapping)
                        else getattr(segment, "words", [])
                    )
                ]
            if "language" not in kwargs and transcript.get("language"):
                kwargs["language"] = transcript["language"]
        else:
            words = list(getattr(transcript, "words", transcript))
        return cls.from_words(words, **kwargs)

    @property
    def text(self) -> str:
        """The concatenated cue text."""
        return "\n".join(cue.text for cue in self.cues)

    @property
    def start_seconds(self) -> float:
        """The earliest cue start, or zero for an empty track."""
        return min((cue.start_seconds for cue in self.cues), default=0.0)

    @property
    def end_seconds(self) -> float:
        """The latest cue end, or zero for an empty track."""
        return max((cue.end for cue in self.cues), default=0.0)

    @property
    def duration_seconds(self) -> float:
        """The span of the whole track in seconds."""
        return self.end_seconds - self.start_seconds

    @property
    def timed_words(self) -> list[SubtitleWord]:
        """Every timed word in cue order."""
        return list(self.words)

    def to_srt(self) -> str:
        """Return this track as an SRT document."""
        return cast(str, self.to_media().to_srt())

    def to_vtt(self) -> str:
        """Return this track as a WebVTT document."""
        return cast(str, self.to_media().to_vtt())

    @property
    def srt(self) -> str:
        """The track as an SRT document."""
        return self.to_srt()

    @property
    def vtt(self) -> str:
        """The track as a WebVTT document."""
        return self.to_vtt()

    def to_media(self) -> Any:
        """Return the media subtitle model, importing it lazily."""
        from aivia.media.subtitles import Subtitle

        return Subtitle.model_validate(
            self.model_dump(by_alias=False, exclude_none=False)
        )

    def to_ass(self, **options: Any) -> str:
        """Render this track as an ASS document through the media model."""
        return cast(str, self.to_media().to_ass(**options))

    def export(self, format: str, **options: Any) -> str:
        """Export this track as SRT, VTT or ASS through the media model."""
        return cast(str, self.to_media().export(format, **options))


Subtitle = SubtitleTrack
"""Alias for :class:`SubtitleTrack`."""

CompositionSubtitle = SubtitleTrack
"""Alias for :class:`SubtitleTrack`."""


def _group_words(words: Iterable[SubtitleWord]) -> list[SubtitleCue]:
    groups: list[list[SubtitleWord]] = []
    current: list[SubtitleWord] = []
    for word in words:
        if current:
            gap = word.start_seconds - (current[-1].end_seconds or 0.0)
            text = " ".join(item.word for item in (*current, word))
            duration = (word.end_seconds or word.start_seconds) - (
                current[0].start_seconds
            )
            if (
                gap > DEFAULT_MEDIA_SUBTITLE_MAX_GAP_SECONDS
                or len(text) > DEFAULT_MEDIA_SUBTITLE_MAX_CHARS
                or duration > DEFAULT_MEDIA_SUBTITLE_MAX_DURATION_SECONDS
            ):
                groups.append(current)
                current = []
        current.append(word)
    if current:
        groups.append(current)
    return [SubtitleCue.from_words(group) for group in groups]


def cues_from_words(words: Iterable[Any]) -> list[SubtitleCue]:
    """Group timed words into cues for composition shorthand input."""
    values = [SubtitleWord.model_validate(word) for word in words]
    return _group_words(values)
