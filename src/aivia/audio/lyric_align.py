"""Align written lyrics to a generated song's transcript.

A song provider returns audio, not text. A speech to text provider can report
what it heard and when, but sung words are not spoken words: vowels stretch,
syllables split across tokens, ad-libs appear between lines and a chorus can
repeat the same words several times. This module joins the written lyric to the
timed transcript without calling a provider itself:

* :func:`align_lyrics` maps every lyric word onto the transcript words and
  groups the result into lines. Matching is ordered, so a repeated chorus is
  assigned to each occurrence in turn rather than always to the first one.
* Case, accents, punctuation, repeated letters and hyphenated syllables are
  folded before comparison. A sung ``"lo-o-ove"`` therefore still matches
  ``"love"`` without lowering the line's coverage.
* Every word, extra transcript word, fuzzy substitution and untimed line is
  reported as a :class:`LyricMismatch` with its line context, so a caller can
  warn, retry or fall back instead of silently shipping bad karaoke timing.

The operation is pure and offline. :func:`align_lyrics_audio` is the only entry
point that calls a provider, and it only transcribes before delegating to the
same deterministic alignment.
"""

from __future__ import annotations

import inspect
import math
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Self

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from aivia.audio.qa import Severity
from aivia.core.config.media import (
    DEFAULT_MEDIA_ALIGNMENT_SIMILARITY_THRESHOLD,
    DEFAULT_MEDIA_LYRIC_DELETION_COST,
    DEFAULT_MEDIA_LYRIC_INSERTION_COST,
    DEFAULT_MEDIA_LYRIC_LINE_THRESHOLD,
    DEFAULT_MEDIA_LYRIC_MATCH_WEIGHT,
    DEFAULT_MEDIA_LYRIC_MERGE_COST,
    DEFAULT_MEDIA_LYRIC_MIN_CONFIDENCE,
    MAX_MEDIA_SIMILARITY,
    MIN_MEDIA_SIMILARITY,
)
from aivia.core.errors import ConfigError, ValidationError
from aivia.providers.protocols.audio import WordTimestamp
from aivia.providers.protocols.stt import STTRequest, STTResponse
from aivia.providers.stt.common import (
    response_duration_seconds,
    segments_from_values,
    words_from_values,
)
from aivia.writing.lyrics import extract_lyrics

__all__ = [
    "DEFAULT_ALIGNMENT_THRESHOLD",
    "DEFAULT_ALIGN_THRESHOLD",
    "DEFAULT_LINE_THRESHOLD",
    "DEFAULT_LYRIC_ALIGNMENT_THRESHOLD",
    "DEFAULT_LYRIC_LINE_THRESHOLD",
    "DEFAULT_LYRIC_THRESHOLD",
    "DEFAULT_MATCH_THRESHOLD",
    "DEFAULT_MAX_MERGE_TOKENS",
    "DEFAULT_MIN_CONFIDENCE",
    "DEFAULT_SIMILARITY_THRESHOLD",
    "LyricAlign",
    "LyricAligner",
    "LyricAlignment",
    "LyricAlignmentLine",
    "LyricAlignmentResult",
    "LyricAlignmentWord",
    "LyricLine",
    "LyricLineMismatch",
    "LyricMismatch",
    "LyricWord",
    "LyricsAlignment",
    "align_lyric",
    "align_lyric_lines",
    "align_lyrics",
    "align_lyrics_audio",
    "align_lyrics_to_audio",
    "align_lyrics_to_song",
    "align_lyrics_to_transcript",
    "align_song",
    "align_song_lyrics",
    "find_lyric_mismatches",
    "find_mismatches",
    "forced_align_lyrics",
    "lyric_alignment",
    "lyric_mismatches",
    "lyric_word_similarity",
    "report_lyric_mismatches",
    "report_mismatches",
]

DEFAULT_LYRIC_ALIGNMENT_THRESHOLD = DEFAULT_MEDIA_ALIGNMENT_SIMILARITY_THRESHOLD
"""Word similarity at or above this pairs a lyric word with a transcript word."""

DEFAULT_LYRIC_THRESHOLD = DEFAULT_LYRIC_ALIGNMENT_THRESHOLD
"""Alias for :data:`DEFAULT_LYRIC_ALIGNMENT_THRESHOLD`."""

DEFAULT_ALIGNMENT_THRESHOLD = DEFAULT_LYRIC_ALIGNMENT_THRESHOLD
"""Alias for :data:`DEFAULT_LYRIC_ALIGNMENT_THRESHOLD`."""

DEFAULT_ALIGN_THRESHOLD = DEFAULT_LYRIC_ALIGNMENT_THRESHOLD
"""Alias for :data:`DEFAULT_LYRIC_ALIGNMENT_THRESHOLD`."""

DEFAULT_MATCH_THRESHOLD = DEFAULT_LYRIC_ALIGNMENT_THRESHOLD
"""Alias for :data:`DEFAULT_LYRIC_ALIGNMENT_THRESHOLD`."""

DEFAULT_SIMILARITY_THRESHOLD = DEFAULT_LYRIC_ALIGNMENT_THRESHOLD
"""Alias for :data:`DEFAULT_LYRIC_ALIGNMENT_THRESHOLD`."""

DEFAULT_LYRIC_LINE_THRESHOLD = DEFAULT_MEDIA_LYRIC_LINE_THRESHOLD
"""Alias for :data:`DEFAULT_LINE_THRESHOLD`."""

DEFAULT_LINE_THRESHOLD = DEFAULT_MEDIA_LYRIC_LINE_THRESHOLD
"""Coverage at or above this marks a line as matched without a line mismatch."""

DEFAULT_MAX_MERGE_TOKENS = 3
"""The most adjacent tokens one sung word may be split into or out of."""

DEFAULT_MIN_CONFIDENCE = DEFAULT_MEDIA_LYRIC_MIN_CONFIDENCE
"""Provider confidence below this is reported as a low-confidence word."""

_WORD = re.compile(r"[^\W_]+(?:['\u2019][^\W_]+)*", re.UNICODE)
_EPSILON = 1e-9
_MATCH_WEIGHT = DEFAULT_MEDIA_LYRIC_MATCH_WEIGHT
_DELETION_COST = DEFAULT_MEDIA_LYRIC_DELETION_COST
_INSERTION_COST = DEFAULT_MEDIA_LYRIC_INSERTION_COST
_MERGE_COST = DEFAULT_MEDIA_LYRIC_MERGE_COST

_FILLERS = frozenset(
    {
        "ah",
        "aah",
        "ay",
        "da",
        "doo",
        "eh",
        "hey",
        "hmm",
        "la",
        "mm",
        "mmm",
        "na",
        "o",
        "oh",
        "ooh",
        "uh",
        "whoa",
        "woah",
        "ya",
        "yea",
        "yeah",
        "yo",
    }
)
"""Ad-libs a singer may add without changing the written lyric."""


def _fold(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in folded if not unicodedata.combining(char))


def _normalise(text: str) -> str:
    return "".join(char for char in _fold(text) if char.isalnum())


def _sung_key(text: str) -> str:
    return re.sub(r"(.)\1+", r"\1", _normalise(text))


def _tokens(text: str) -> list[str]:
    return [match.group() for match in _WORD.finditer(text)]


def _number(value: Any, *, field: str, non_negative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(
            f"{field} must be a number",
            details={"field": field, "value": repr(value)},
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValidationError(
            f"{field} must be finite",
            details={"field": field, "value": repr(value)},
        )
    if non_negative and number < 0.0:
        raise ValidationError(
            f"{field} must not be negative",
            details={"field": field, "value": value},
        )
    return number


def _threshold(value: Any, *, field: str) -> float:
    number = _number(value, field=field)
    if not 0.0 <= number <= 1.0:
        raise ValidationError(
            f"{field} must be between 0 and 1",
            details={"field": field, "value": value},
        )
    return number


def lyric_word_similarity(left: str, right: str) -> float:
    """Return how closely two lyric or transcript words match in ``[0, 1]``.

    Case, accents and punctuation are folded away. Repeated letters and split
    syllables are then collapsed, so ``"lo-o-ove"`` and ``"loooove"`` both score
    ``1.0`` against ``"love"``; an ordinary near miss such as ``"Smith"`` and
    ``"Smyth"`` keeps its character similarity.
    """
    left_key = _normalise(left)
    right_key = _normalise(right)
    if not left_key or not right_key:
        return 1.0 if left_key == right_key else 0.0
    if left_key == right_key:
        return 1.0
    left_sung = _sung_key(left)
    right_sung = _sung_key(right)
    if left_sung == right_sung:
        return 1.0
    direct = SequenceMatcher(None, left_key, right_key).ratio()
    sung = SequenceMatcher(None, left_sung, right_sung).ratio()
    return max(direct, sung)


@dataclass(frozen=True, slots=True)
class _Token:
    text: str
    norm: str
    key: str
    start: float | None
    end: float | None
    confidence: float | None
    index: int | None


@dataclass(frozen=True, slots=True)
class _Transcript:
    tokens: list[_Token]
    duration: float | None
    language: str | None
    provider: str
    model: str
    response: STTResponse | None


@dataclass(frozen=True, slots=True)
class _LineSpec:
    text: str
    section: str
    index: int
    tokens: list[str]


def _token_from_word(word: WordTimestamp, index: int) -> _Token:
    return _Token(
        text=word.word,
        norm=_normalise(word.word),
        key=_sung_key(word.word),
        start=word.start_seconds,
        end=word.end_seconds,
        confidence=word.confidence,
        index=index,
    )


def _tokens_from_text(text: str) -> list[_Token]:
    return [
        _Token(
            text=token,
            norm=_normalise(token),
            key=_sung_key(token),
            start=None,
            end=None,
            confidence=None,
            index=None,
        )
        for token in _tokens(text)
    ]


def _split_timed_word(word: WordTimestamp, index: int) -> list[_Token]:
    raw_parts = word.word.split()
    if len(raw_parts) <= 1:
        return [_token_from_word(word, index)]
    parts = [part for raw in raw_parts for part in _tokens(raw)]
    if len(parts) <= 1:
        return [_token_from_word(word, index)]
    weights = [max(1, len(_normalise(part))) for part in parts]
    total = sum(weights)
    cursor = word.start_seconds
    tokens: list[_Token] = []
    for offset, (part, weight) in enumerate(zip(parts, weights, strict=True)):
        end = cursor + word.duration_seconds * weight / total
        if offset == len(parts) - 1:
            end = word.end_seconds
        tokens.append(
            _Token(
                text=part,
                norm=_normalise(part),
                key=_sung_key(part),
                start=cursor,
                end=end,
                confidence=word.confidence,
                index=index,
            )
        )
        cursor = end
    return tokens


def _word_models(values: Sequence[Any]) -> list[WordTimestamp]:
    converted: list[WordTimestamp] = []
    for raw in values:
        if isinstance(raw, WordTimestamp):
            converted.append(raw)
            continue
        parsed = words_from_values([raw])
        if not parsed:
            raise ValidationError("transcript word is invalid")
        converted.extend(parsed)
    return sorted(converted, key=lambda word: word.start_seconds)


def _segment_models(values: Sequence[Any]) -> list[Any]:
    converted: list[Any] = []
    for raw in values:
        if hasattr(raw, "start_seconds") and hasattr(raw, "words"):
            converted.append(raw)
        else:
            converted.extend(segments_from_values([raw]))
    return converted


def _tokens_from_words(words: Sequence[Any]) -> list[_Token]:
    tokens: list[_Token] = []
    timed: list[Any] = []
    for raw in words:
        if isinstance(raw, str):
            tokens.extend(_tokens_from_text(raw))
        else:
            timed.append(raw)
    for index, word in enumerate(_word_models(timed)):
        tokens.extend(_split_timed_word(word, index))
    starts = [token.start for token in tokens if token.start is not None]
    if starts != sorted(starts):
        raise ValidationError("transcript words must be ordered by start_seconds")
    return tokens


def _transcript_from_response(response: STTResponse) -> _Transcript:
    words = response.all_words
    return _Transcript(
        tokens=(
            _tokens_from_words(words) if words else _tokens_from_text(response.text)
        ),
        duration=response_duration_seconds(response),
        language=response.language,
        provider=response.provider,
        model=response.model,
        response=response,
    )


def _transcript_from_source(source: Any) -> _Transcript:
    if isinstance(source, STTResponse):
        return _transcript_from_response(source)
    if isinstance(source, WordTimestamp):
        return _Transcript(
            tokens=[_token_from_word(source, 0)],
            duration=source.end_seconds,
            language=None,
            provider="",
            model="",
            response=None,
        )
    if isinstance(source, str):
        return _Transcript(
            tokens=_tokens_from_text(source),
            duration=None,
            language=None,
            provider="",
            model="",
            response=None,
        )
    if isinstance(source, Mapping):
        values = dict(source)
        if "text" in values and not any(key in values for key in ("words", "segments")):
            return _Transcript(
                tokens=_tokens_from_text(str(values["text"])),
                duration=None,
                language=None,
                provider="",
                model="",
                response=None,
            )
        try:
            if "words" in values:
                values["words"] = _word_models(values["words"])
            if "segments" in values:
                values["segments"] = _segment_models(values["segments"])
            response = STTResponse.model_validate(values)
        except Exception as exc:
            raise ValidationError("transcript mapping is invalid") from exc
        return _transcript_from_response(response)
    if hasattr(source, "all_words"):
        words = list(source.all_words)
        text = getattr(source, "text", "") or ""
        return _Transcript(
            tokens=(_tokens_from_words(words) if words else _tokens_from_text(text)),
            duration=getattr(source, "duration_seconds", None),
            language=getattr(source, "language", None),
            provider=getattr(source, "provider", "") or "",
            model=getattr(source, "model", "") or "",
            response=source if isinstance(source, STTResponse) else None,
        )
    if isinstance(source, Iterable) and not isinstance(source, (bytes, bytearray)):
        return _Transcript(
            tokens=_tokens_from_words(list(source)),
            duration=None,
            language=None,
            provider="",
            model="",
            response=None,
        )
    raise ValidationError(
        "transcript must be a speech to text response, words or text",
        details={"type": type(source).__name__},
    )


def _line_specs(lyrics: Any) -> list[_LineSpec]:
    try:
        payload = extract_lyrics(lyrics)
        sections = list(payload.sections)
    except ConfigError:
        sections = []

    specs: list[_LineSpec] = []
    for section in sections:
        for line in section.lines:
            text = line.strip()
            if not text:
                continue
            specs.append(
                _LineSpec(
                    text=text,
                    section=section.name,
                    index=len(specs),
                    tokens=_tokens(text),
                )
            )
    if specs:
        return specs

    if not sections:
        payload_value = getattr(lyrics, "payload", None)
        if isinstance(payload_value, Mapping):
            try:
                parsed_payload = extract_lyrics(payload_value)
            except ConfigError:
                parsed_payload = None
            if parsed_payload is not None:
                for section in parsed_payload.sections:
                    for line in section.lines:
                        text = line.strip()
                        if text:
                            specs.append(
                                _LineSpec(
                                    text=text,
                                    section=section.name,
                                    index=len(specs),
                                    tokens=_tokens(text),
                                )
                            )
                if specs:
                    return specs

    beats = getattr(lyrics, "beats", None)
    if beats is None and isinstance(lyrics, Mapping):
        beats = lyrics.get("beats")
    if beats is not None:
        for beat in beats:
            text = (
                beat.get("text", "")
                if isinstance(beat, Mapping)
                else getattr(beat, "text", "")
            )
            if isinstance(text, str) and text.strip():
                specs.append(
                    _LineSpec(text.strip(), "verse", len(specs), _tokens(text))
                )
    if specs:
        return specs
    raise ValidationError(
        "lyrics must contain at least one non-empty line",
        details={"field": "lyrics"},
    )


def _similarity(left: _Token, right: _Token) -> float:
    if left.key == right.key:
        return 1.0
    return lyric_word_similarity(left.text, right.text)


def _joined_similarity(left: Sequence[_Token], right: Sequence[_Token]) -> float:
    left_text = " ".join(token.text for token in left)
    right_text = " ".join(token.text for token in right)
    return lyric_word_similarity(left_text, right_text)


def _align_tokens(
    lyric: Sequence[_Token],
    transcript: Sequence[_Token],
    *,
    threshold: float,
    max_merge_tokens: int,
) -> tuple[dict[int, tuple[int, int, float]], list[int]]:
    count = len(lyric)
    heard = len(transcript)
    negative = float("-inf")
    scores = [[negative] * (heard + 1) for _ in range(count + 1)]
    choices: list[list[tuple[str, int, int, float] | None]] = [
        [None] * (heard + 1) for _ in range(count + 1)
    ]
    scores[0][0] = 0.0
    for i in range(count + 1):
        for j in range(heard + 1):
            if i == 0 and j == 0:
                continue
            best = negative
            choice: tuple[str, int, int, float] | None = None
            if i > 0 and scores[i - 1][j] != negative:
                candidate = scores[i - 1][j] - _DELETION_COST
                if candidate > best + _EPSILON:
                    best = candidate
                    choice = ("lyric_gap", 1, 0, 0.0)
            if j > 0 and scores[i][j - 1] != negative:
                candidate = scores[i][j - 1] - _INSERTION_COST
                if candidate > best + _EPSILON:
                    best = candidate
                    choice = ("transcript_gap", 0, 1, 0.0)
            for lyric_span in range(1, max_merge_tokens + 1):
                if i < lyric_span:
                    break
                for heard_span in range(1, max_merge_tokens + 1):
                    if j < heard_span:
                        break
                    if lyric_span > 1 and heard_span > 1:
                        continue
                    if scores[i - lyric_span][j - heard_span] == negative:
                        continue
                    left = lyric[i - lyric_span : i]
                    right = transcript[j - heard_span : j]
                    similarity = (
                        _similarity(left[0], right[0])
                        if lyric_span == 1 and heard_span == 1
                        else _joined_similarity(left, right)
                    )
                    if similarity < threshold:
                        continue
                    merged = (lyric_span - 1) + (heard_span - 1)
                    candidate = (
                        scores[i - lyric_span][j - heard_span]
                        + _MATCH_WEIGHT * similarity
                        - _MERGE_COST * merged
                    )
                    if candidate > best + _EPSILON:
                        best = candidate
                        choice = ("match", lyric_span, heard_span, similarity)
            if choice is not None:
                scores[i][j] = best
                choices[i][j] = choice

    matches: dict[int, tuple[int, int, float]] = {}
    extras: list[int] = []
    i = count
    j = heard
    while i > 0 or j > 0:
        choice = choices[i][j]
        if choice is None:
            raise ValidationError(
                "lyrics and transcript could not be aligned",
                details={"lyric_words": count, "transcript_words": heard},
            )
        kind, lyric_span, heard_span, similarity = choice
        if kind == "match":
            for offset in range(lyric_span):
                matches[i - lyric_span + offset] = (
                    j - heard_span,
                    j,
                    similarity,
                )
            i -= lyric_span
            j -= heard_span
        elif kind == "lyric_gap":
            i -= 1
        else:
            extras.append(j - 1)
            j -= 1
    extras.reverse()
    return matches, extras


class LyricMismatch(BaseModel):
    """One difference between the written lyric and what was sung."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
    )

    code: str = Field(
        min_length=1,
        validation_alias=AliasChoices("code", "kind", "type", "reason"),
    )
    message: str = ""
    severity: Severity = Severity.WARNING
    line_index: int | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices("line_index", "line", "line_number"),
    )
    word_index: int | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices("word_index", "index", "position"),
    )
    line_word_index: int | None = Field(default=None, ge=0)
    expected: str = ""
    actual: str = ""
    similarity: float | None = Field(default=None, ge=0.0, le=1.0)
    start_seconds: float | None = Field(default=None, ge=0.0)
    end_seconds: float | None = Field(default=None, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("code", "message", "expected", "actual", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("severity", mode="before")
    @classmethod
    def _coerce_severity(cls, value: Any) -> Any:
        if isinstance(value, Severity):
            return value
        if isinstance(value, str):
            return Severity(value.strip().lower())
        return value

    @model_validator(mode="after")
    def _validate_span(self) -> Self:
        if (
            self.start_seconds is not None
            and self.end_seconds is not None
            and self.end_seconds < self.start_seconds
        ):
            raise ValueError("end_seconds must not be before start_seconds")
        return self

    @property
    def line(self) -> int | None:
        """Alias for :attr:`line_index`."""
        return self.line_index

    @property
    def kind(self) -> str:
        """The broad category of this mismatch."""
        if self.code.startswith("missing"):
            return "missing"
        if self.code.startswith("extra"):
            return "extra"
        if self.code.startswith(("mispronounced", "substitution", "wrong")):
            return "substitution"
        if self.code.startswith(("untimed", "timing", "overlap")):
            return "timing"
        if self.code.startswith("low_confidence"):
            return "confidence"
        if self.code.startswith(("weak_line", "line_coverage", "low_coverage")):
            return "line"
        return self.code

    @property
    def category(self) -> str:
        """Alias for :attr:`kind`."""
        return self.kind

    @property
    def type(self) -> str:
        """Alias for :attr:`code`."""
        return self.code

    @property
    def level(self) -> str:
        """The severity value as a string."""
        return self.severity.value

    def to_dict(self) -> dict[str, Any]:
        """Return the mismatch as a JSON-compatible mapping."""
        return {
            "code": self.code,
            "kind": self.kind,
            "severity": self.severity.value,
            "message": self.message,
            "line_index": self.line_index,
            "word_index": self.word_index,
            "line_word_index": self.line_word_index,
            "expected": self.expected,
            "actual": self.actual,
            "similarity": self.similarity,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
        }


class LyricWord(BaseModel):
    """One written lyric word and the sung word it aligned to."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
    )

    word: str = Field(
        min_length=1,
        validation_alias=AliasChoices("word", "text", "token"),
    )
    line_index: int | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices("line_index", "line", "line_number"),
    )
    index: int | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices("index", "word_index", "position"),
    )
    start_seconds: float | None = Field(
        default=None,
        ge=0.0,
        validation_alias=AliasChoices("start_seconds", "start", "start_time"),
    )
    end_seconds: float | None = Field(
        default=None,
        ge=0.0,
        validation_alias=AliasChoices("end_seconds", "end", "end_time"),
    )
    similarity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("similarity", "score", "match_score"),
    )
    matched: bool = Field(
        default=False,
        validation_alias=AliasChoices("matched", "found", "verified"),
    )
    transcript_word: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "transcript_word",
            "matched_word",
            "source_word",
        ),
    )
    transcript_index: int | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices("transcript_index", "heard_index"),
    )
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    interpolated: bool = Field(default=False)
    section: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("word", "transcript_word", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip()
            return text or None
        return value

    @model_validator(mode="after")
    def _validate_span(self) -> Self:
        if (
            self.start_seconds is not None
            and self.end_seconds is not None
            and self.end_seconds < self.start_seconds
        ):
            raise ValueError("end_seconds must not be before start_seconds")
        return self

    @property
    def text(self) -> str:
        """Alias for :attr:`word`."""
        return self.word

    @property
    def start(self) -> float | None:
        """Alias for :attr:`start_seconds`."""
        return self.start_seconds

    @property
    def end(self) -> float | None:
        """Alias for :attr:`end_seconds`."""
        return self.end_seconds

    @property
    def score(self) -> float:
        """Alias for :attr:`similarity`."""
        return self.similarity

    @property
    def matched_word(self) -> str | None:
        """Alias for :attr:`transcript_word`."""
        return self.transcript_word

    @property
    def line(self) -> int | None:
        """Alias for :attr:`line_index`."""
        return self.line_index

    @property
    def aligned(self) -> bool:
        """Whether the word has a start and an end time."""
        return self.start_seconds is not None and self.end_seconds is not None

    @property
    def timed(self) -> bool:
        """Alias for :attr:`aligned`."""
        return self.aligned

    @property
    def exact(self) -> bool:
        """Whether the word matched at full similarity."""
        return self.matched and self.similarity >= MAX_MEDIA_SIMILARITY

    @property
    def source(self) -> str:
        """A short label for how this word received its timing."""
        if self.interpolated:
            return "interpolated"
        if not self.matched:
            return "unmatched"
        return "exact" if self.exact else "fuzzy"

    @property
    def duration_seconds(self) -> float | None:
        """The timed length of the word, or ``None`` when it is untimed."""
        if self.start_seconds is None or self.end_seconds is None:
            return None
        return self.end_seconds - self.start_seconds


class LyricLine(BaseModel):
    """One lyric line with its words and the span it was sung over."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
    )

    text: str = Field(
        min_length=1,
        validation_alias=AliasChoices("text", "line", "lyrics", "content"),
    )
    section: str = ""
    index: int | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices("index", "line_index", "line_number"),
    )
    start_seconds: float | None = Field(default=None, ge=0.0)
    end_seconds: float | None = Field(default=None, ge=0.0)
    words: list[LyricWord] = Field(default_factory=list)
    matched: bool = False
    similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    interpolated: bool = False
    mismatches: list[LyricMismatch] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text", "section", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value

    @model_validator(mode="after")
    def _validate_span(self) -> Self:
        if (
            self.start_seconds is not None
            and self.end_seconds is not None
            and self.end_seconds < self.start_seconds
        ):
            raise ValueError("end_seconds must not be before start_seconds")
        return self

    @property
    def line(self) -> str:
        """Alias for :attr:`text`."""
        return self.text

    @property
    def line_index(self) -> int | None:
        """Alias for :attr:`index`."""
        return self.index

    @property
    def start(self) -> float | None:
        """Alias for :attr:`start_seconds`."""
        return self.start_seconds

    @property
    def end(self) -> float | None:
        """Alias for :attr:`end_seconds`."""
        return self.end_seconds

    @property
    def score(self) -> float:
        """Alias for :attr:`similarity`."""
        return self.similarity

    @property
    def aligned(self) -> bool:
        """Whether the line has a start and an end time."""
        return self.start_seconds is not None and self.end_seconds is not None

    @property
    def timed(self) -> bool:
        """Alias for :attr:`aligned`."""
        return self.aligned

    @property
    def word_count(self) -> int:
        """The number of written words in the line."""
        return len(self.words)

    @property
    def matched_words(self) -> list[LyricWord]:
        """The words found in the transcript."""
        return [word for word in self.words if word.matched]

    @property
    def unmatched_words(self) -> list[LyricWord]:
        """The written words the transcript did not contain."""
        return [word for word in self.words if not word.matched]

    @property
    def missing_words(self) -> list[LyricWord]:
        """Alias for :attr:`unmatched_words`."""
        return self.unmatched_words

    @property
    def coverage(self) -> float:
        """The share of words matched to a transcript word."""
        if not self.words:
            return 1.0
        return len(self.matched_words) / len(self.words)

    @property
    def transcript_text(self) -> str:
        """The transcript words that aligned to this line."""
        return " ".join(
            word.transcript_word
            for word in self.words
            if word.transcript_word is not None
        )

    @property
    def duration_seconds(self) -> float | None:
        """The timed length of the line, or ``None`` when it is untimed."""
        if self.start_seconds is None or self.end_seconds is None:
            return None
        return self.end_seconds - self.start_seconds


class LyricAlignment(BaseModel):
    """The aligned lines, words and mismatches for one lyric and transcript."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )

    lyrics: str = Field(
        validation_alias=AliasChoices("lyrics", "text", "lyric_text", "source"),
    )
    lines: list[LyricLine] = Field(default_factory=list)
    mismatches: list[LyricMismatch] = Field(default_factory=list)
    threshold: float = Field(
        default=DEFAULT_LYRIC_ALIGNMENT_THRESHOLD,
        ge=0.0,
        le=1.0,
    )
    line_threshold: float = Field(default=DEFAULT_LINE_THRESHOLD, ge=0.0, le=1.0)
    language: str | None = None
    duration_seconds: float | None = Field(default=None, ge=0.0)
    provider: str = ""
    model: str = ""
    transcript: STTResponse | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("lyrics", "provider", "model", mode="before")
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

    @property
    def text(self) -> str:
        """Alias for :attr:`lyrics`."""
        return self.lyrics

    @property
    def words(self) -> list[LyricWord]:
        """Every lyric word in line order."""
        return [word for line in self.lines for word in line.words]

    @property
    def tokens(self) -> list[LyricWord]:
        """Alias for :attr:`words`."""
        return self.words

    @property
    def word_count(self) -> int:
        """The number of written lyric words."""
        return len(self.words)

    @property
    def line_count(self) -> int:
        """The number of written lyric lines."""
        return len(self.lines)

    @property
    def matched_count(self) -> int:
        """The number of lyric words found in the transcript."""
        return len(self.matched_words)

    @property
    def unmatched_count(self) -> int:
        """The number of lyric words missing from the transcript."""
        return len(self.unmatched_words)

    @property
    def matched_line_count(self) -> int:
        """The number of lines with at least one matched word."""
        return len(self.matched_lines)

    @property
    def unmatched_line_count(self) -> int:
        """The number of lines with no matched word."""
        return len(self.untimed_lines)

    @property
    def matched_words(self) -> list[LyricWord]:
        """The lyric words found in the transcript."""
        return [word for word in self.words if word.matched]

    @property
    def unmatched_words(self) -> list[LyricWord]:
        """The lyric words missing from the transcript."""
        return [word for word in self.words if not word.matched]

    @property
    def missing_words(self) -> list[LyricWord]:
        """Alias for :attr:`unmatched_words`."""
        return self.unmatched_words

    @property
    def matched_lines(self) -> list[LyricLine]:
        """The lines with at least one matched word."""
        return [line for line in self.lines if line.matched]

    @property
    def untimed_lines(self) -> list[LyricLine]:
        """The lines that received no timing at all."""
        return [line for line in self.lines if not line.aligned]

    @property
    def missing_lines(self) -> list[LyricLine]:
        """Alias for :attr:`untimed_lines`."""
        return self.untimed_lines

    @property
    def aligned(self) -> bool:
        """Whether any line received timing."""
        return any(line.aligned for line in self.lines)

    @property
    def has_timings(self) -> bool:
        """Alias for :attr:`aligned`."""
        return self.aligned

    @property
    def coverage(self) -> float:
        """The share of lyric words matched to a transcript word."""
        if not self.words:
            return 1.0
        return len(self.matched_words) / len(self.words)

    @property
    def word_coverage(self) -> float:
        """Alias for :attr:`coverage`."""
        return self.coverage

    @property
    def line_coverage(self) -> float:
        """The share of lyric lines with at least one matched word."""
        if not self.lines:
            return 1.0
        return len(self.matched_lines) / len(self.lines)

    @property
    def score(self) -> float:
        """Alias for :attr:`coverage`."""
        return self.coverage

    @property
    def mismatch_count(self) -> int:
        """The number of reported mismatches."""
        return len(self.mismatches)

    @property
    def missing(self) -> list[LyricMismatch]:
        """The missing-word mismatches."""
        return [mismatch for mismatch in self.mismatches if mismatch.kind == "missing"]

    @property
    def extra(self) -> list[LyricMismatch]:
        """The extra-word mismatches."""
        return [mismatch for mismatch in self.mismatches if mismatch.kind == "extra"]

    @property
    def substitutions(self) -> list[LyricMismatch]:
        """The fuzzy word substitutions."""
        return [
            mismatch for mismatch in self.mismatches if mismatch.kind == "substitution"
        ]

    @property
    def timing(self) -> list[LyricMismatch]:
        """The untimed-line and overlap mismatches."""
        return [mismatch for mismatch in self.mismatches if mismatch.kind == "timing"]

    @property
    def weak_lines(self) -> list[LyricLine]:
        """The lines whose coverage is below the configured threshold."""
        return [
            line for line in self.lines if line.metadata.get("below_line_threshold")
        ]

    @property
    def line_mismatches(self) -> list[LyricMismatch]:
        """The weak-line mismatches."""
        return [mismatch for mismatch in self.mismatches if mismatch.kind == "line"]

    @property
    def errors(self) -> list[LyricMismatch]:
        """Mismatches severe enough to fail the stage."""
        return [
            mismatch
            for mismatch in self.mismatches
            if mismatch.severity is Severity.ERROR
        ]

    @property
    def warnings(self) -> list[LyricMismatch]:
        """Advisory mismatches."""
        return [
            mismatch
            for mismatch in self.mismatches
            if mismatch.severity is Severity.WARNING
        ]

    @property
    def ok(self) -> bool:
        """Whether the alignment found no mismatch at all."""
        return not self.mismatches

    @property
    def passed(self) -> bool:
        """Whether the alignment found no error-level mismatch."""
        return not self.errors

    def by_code(self, code: str) -> list[LyricMismatch]:
        """Return mismatches with *code*, accepting a broad kind too."""
        aliases = {
            "missing": "missing_word",
            "missing_line": "untimed_line",
            "line_missing": "untimed_line",
            "extra": "extra_word",
            "substitution": "mispronounced_word",
            "mispronunciation": "mispronounced_word",
            "untimed": "untimed_line",
            "overlap": "timing_overlap",
            "confidence": "low_confidence_word",
            "low_confidence": "low_confidence_word",
            "line": "weak_line",
            "weak": "weak_line",
            "line_coverage": "weak_line",
            "low_coverage": "weak_line",
        }
        wanted = aliases.get(code, code)
        return [
            mismatch
            for mismatch in self.mismatches
            if mismatch.code == wanted or mismatch.kind == code
        ]

    @property
    def word_timestamps(self) -> list[WordTimestamp]:
        """The timed words as the shared protocol model."""
        return [
            WordTimestamp(
                word=word.word,
                start_seconds=word.start_seconds,
                end_seconds=word.end_seconds,
                confidence=word.confidence,
            )
            for word in self.words
            if word.start_seconds is not None and word.end_seconds is not None
        ]

    @property
    def timed_words(self) -> list[WordTimestamp]:
        """Alias for :attr:`word_timestamps`."""
        return self.word_timestamps

    @property
    def line_timestamps(self) -> list[WordTimestamp]:
        """The timed lines as the shared protocol model."""
        return [
            WordTimestamp(
                word=line.text,
                start_seconds=line.start_seconds,
                end_seconds=line.end_seconds,
            )
            for line in self.lines
            if line.start_seconds is not None and line.end_seconds is not None
        ]

    @property
    def line_timings(self) -> list[WordTimestamp]:
        """Alias for :attr:`line_timestamps`."""
        return self.line_timestamps

    @property
    def word_timings(self) -> list[WordTimestamp]:
        """Alias for :attr:`word_timestamps`."""
        return self.word_timestamps

    def to_word_timestamps(self) -> list[WordTimestamp]:
        """Return :attr:`word_timestamps`."""
        return self.word_timestamps

    def as_word_timestamps(self) -> list[WordTimestamp]:
        """Alias for :meth:`to_word_timestamps`."""
        return self.word_timestamps

    def to_dict(self) -> dict[str, Any]:
        """Return the alignment as a JSON-compatible mapping."""
        return {
            "lyrics": self.lyrics,
            "lines": [line.model_dump(mode="json") for line in self.lines],
            "mismatches": [mismatch.to_dict() for mismatch in self.mismatches],
            "threshold": self.threshold,
            "line_threshold": self.line_threshold,
            "language": self.language,
            "duration_seconds": self.duration_seconds,
            "provider": self.provider,
            "model": self.model,
            "coverage": self.coverage,
            "line_coverage": self.line_coverage,
            "ok": self.ok,
            "passed": self.passed,
            "metadata": dict(self.metadata),
        }

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.lines)

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, index: int) -> LyricLine:
        return self.lines[index]

    def __bool__(self) -> bool:
        return bool(self.lines)


LyricAlignmentResult = LyricAlignment
"""Alias for :class:`LyricAlignment`."""

LyricsAlignment = LyricAlignment
"""Alias for :class:`LyricAlignment`."""

LyricAlignmentLine = LyricLine
"""Alias for :class:`LyricLine`."""

LyricAlignmentWord = LyricWord
"""Alias for :class:`LyricWord`."""

LyricLineMismatch = LyricMismatch
"""Alias for :class:`LyricMismatch`."""


def _interpolate(words: list[LyricWord], *, duration: float | None) -> None:
    anchors = [
        index
        for index, word in enumerate(words)
        if word.start_seconds is not None and word.end_seconds is not None
    ]
    if not anchors:
        if duration is None or not words:
            return
        total = sum(max(1, len(word.word)) for word in words)
        cursor = 0.0
        for word in words:
            share = duration * max(1, len(word.word)) / total
            word.start_seconds = cursor
            word.end_seconds = cursor + share
            word.interpolated = True
            cursor += share
        return

    runs: list[tuple[int, int]] = []
    previous = -1
    for anchor in anchors:
        if anchor > previous + 1:
            runs.append((previous + 1, anchor))
        previous = anchor
    if previous + 1 < len(words):
        runs.append((previous + 1, len(words)))

    for start_index, end_index in runs:
        previous_word = words[start_index - 1] if start_index > 0 else None
        next_word = words[end_index] if end_index < len(words) else None
        start = previous_word.end_seconds if previous_word is not None else 0.0
        if next_word is not None:
            end = next_word.start_seconds
        elif duration is not None:
            end = duration
        elif previous_word is not None and previous_word.end_seconds is not None:
            end = previous_word.end_seconds
        else:
            continue
        if start is None:
            start = 0.0
        if end is None or end < start:
            end = start
        run = words[start_index:end_index]
        total = sum(max(1, len(word.word)) for word in run)
        cursor = start
        for word in run:
            share = (end - start) * max(1, len(word.word)) / total
            word.start_seconds = cursor
            word.end_seconds = cursor + share
            word.interpolated = True
            cursor += share


def _line_span(line: LyricLine) -> None:
    starts = [
        word.start_seconds for word in line.words if word.start_seconds is not None
    ]
    ends = [word.end_seconds for word in line.words if word.end_seconds is not None]
    if starts:
        line.start_seconds = min(starts)
    if ends:
        line.end_seconds = max(ends)
    line.interpolated = bool(line.words) and all(
        word.interpolated for word in line.words
    )


def _mismatch(
    code: str,
    message: str,
    *,
    severity: Severity = Severity.WARNING,
    line_index: int | None = None,
    word_index: int | None = None,
    line_word_index: int | None = None,
    expected: str = "",
    actual: str = "",
    similarity: float | None = None,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> LyricMismatch:
    return LyricMismatch(
        code=code,
        message=message,
        severity=severity,
        line_index=line_index,
        word_index=word_index,
        line_word_index=line_word_index,
        expected=expected,
        actual=actual,
        similarity=similarity,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        metadata=metadata or {},
    )


def align_lyrics(
    lyrics: Any,
    transcript: Any = None,
    *,
    words: Any = None,
    threshold: float = DEFAULT_LYRIC_ALIGNMENT_THRESHOLD,
    line_threshold: float = DEFAULT_LINE_THRESHOLD,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_similarity: float | None = None,
    min_score: float | None = None,
    interpolate: bool = False,
    max_merge_tokens: int = DEFAULT_MAX_MERGE_TOKENS,
    report_extra: bool = True,
    report_fuzzy: bool = True,
    ignore_fillers: bool = True,
    language: str | None = None,
) -> LyricAlignment:
    """Align *lyrics* to the timed *transcript* of a generated song.

    *lyrics* may be lyric text with ``[Verse]`` headers, a sequence of lines, a
    :class:`~aivia.writing.schema.LyricsPayload`, a script payload or a mapping
    with ``sections``/``lines``. *transcript* may be a
    :class:`~aivia.providers.protocols.stt.STTResponse`, a word list, a mapping
    with ``words``/``segments`` or plain text. ``words`` is an alternative
    keyword for the transcript and ``min_similarity``/``min_score`` alias
    *threshold*.

    Matching is ordered and tolerant of sung text: repeated letters, split
    syllables, case, accents and punctuation are folded, and adjacent tokens can
    merge. Extra transcript words and missing lyric words do not shift the
    timing of later lines. Set *interpolate* to distribute a gap across words
    that were not heard, *report_extra* or *report_fuzzy* to silence those
    mismatch classes, and *ignore_fillers* to ignore ad-libs such as ``oh``.
    Words with a provider confidence below *min_confidence* are reported too.
    """
    if transcript is None:
        transcript = words
    if transcript is None:
        raise ValidationError(
            "a transcript or word list is required",
            details={"field": "transcript"},
        )
    if min_similarity is not None:
        threshold = min_similarity
    if min_score is not None:
        threshold = min_score
    score_threshold = _threshold(threshold, field="threshold")
    line_score = _threshold(line_threshold, field="line_threshold")
    confidence_threshold = _threshold(min_confidence, field="min_confidence")
    if not isinstance(interpolate, bool):
        raise TypeError("interpolate must be a bool")
    if not isinstance(report_extra, bool):
        raise TypeError("report_extra must be a bool")
    if not isinstance(report_fuzzy, bool):
        raise TypeError("report_fuzzy must be a bool")
    if not isinstance(ignore_fillers, bool):
        raise TypeError("ignore_fillers must be a bool")
    if isinstance(max_merge_tokens, bool) or not isinstance(max_merge_tokens, int):
        raise TypeError("max_merge_tokens must be an integer")
    if max_merge_tokens < 1:
        raise ValidationError(
            "max_merge_tokens must be positive",
            details={"field": "max_merge_tokens", "value": max_merge_tokens},
        )

    specs = _line_specs(lyrics)
    parsed = _transcript_from_source(transcript)
    lyric_tokens: list[_Token] = []
    for spec in specs:
        for token in spec.tokens:
            lyric_tokens.append(
                _Token(
                    text=token,
                    norm=_normalise(token),
                    key=_sung_key(token),
                    start=None,
                    end=None,
                    confidence=None,
                    index=len(lyric_tokens),
                )
            )

    matches, extras = _align_tokens(
        lyric_tokens,
        parsed.tokens,
        threshold=score_threshold,
        max_merge_tokens=max_merge_tokens,
    )

    aligned_words: list[LyricWord] = []
    cursor = 0
    for spec in specs:
        for token in spec.tokens:
            match = matches.get(cursor)
            if match is None:
                aligned_words.append(
                    LyricWord(
                        word=token,
                        line_index=spec.index,
                        index=cursor,
                        similarity=MIN_MEDIA_SIMILARITY,
                        matched=False,
                        section=spec.section,
                    )
                )
                cursor += 1
                continue
            heard_start, heard_end, similarity = match
            heard_tokens = parsed.tokens[heard_start:heard_end]
            aligned_words.append(
                LyricWord(
                    word=token,
                    line_index=spec.index,
                    index=cursor,
                    start_seconds=heard_tokens[0].start,
                    end_seconds=heard_tokens[-1].end,
                    similarity=similarity,
                    matched=True,
                    transcript_word=" ".join(item.text for item in heard_tokens),
                    transcript_index=heard_tokens[0].index,
                    confidence=heard_tokens[0].confidence,
                    section=spec.section,
                    metadata={"merged": len(heard_tokens) > 1},
                )
            )
            cursor += 1

    duration = parsed.duration
    if duration is None:
        ends = [word.end_seconds for word in aligned_words if word.end_seconds]
        duration = max(ends) if ends else None
    if interpolate:
        _interpolate(aligned_words, duration=duration)

    lines: list[LyricLine] = []
    for spec in specs:
        line_words = [word for word in aligned_words if word.line_index == spec.index]
        line = LyricLine(
            text=spec.text,
            section=spec.section,
            index=spec.index,
            words=line_words,
            matched=any(word.matched for word in line_words),
            similarity=(
                sum(word.similarity if word.matched else 0.0 for word in line_words)
                / len(line_words)
                if line_words
                else 1.0
            ),
        )
        _line_span(line)
        lines.append(line)

    mismatches: list[LyricMismatch] = []
    for line in lines:
        for line_word_index, word in enumerate(line.words):
            if not word.matched:
                mismatches.append(
                    _mismatch(
                        "missing_word",
                        f"line {line.index + 1 if line.index is not None else 1} "
                        f"is missing the word {word.word!r}",
                        line_index=line.index,
                        word_index=word.index,
                        line_word_index=line_word_index,
                        expected=word.word,
                    )
                )
            elif report_fuzzy and word.similarity < MAX_MEDIA_SIMILARITY - _EPSILON:
                mismatches.append(
                    _mismatch(
                        "mispronounced_word",
                        f"line {line.index + 1 if line.index is not None else 1} "
                        f"sang {word.transcript_word!r} for {word.word!r}",
                        severity=Severity.INFO,
                        line_index=line.index,
                        word_index=word.index,
                        line_word_index=line_word_index,
                        expected=word.word,
                        actual=word.transcript_word or "",
                        similarity=word.similarity,
                        start_seconds=word.start_seconds,
                        end_seconds=word.end_seconds,
                    )
                )
            if (
                word.matched
                and word.confidence is not None
                and word.confidence < confidence_threshold - _EPSILON
            ):
                mismatches.append(
                    _mismatch(
                        "low_confidence_word",
                        f"line {line.index + 1 if line.index is not None else 1} "
                        f"heard {word.transcript_word!r} with confidence "
                        f"{word.confidence:.2f}",
                        severity=Severity.INFO,
                        line_index=line.index,
                        word_index=word.index,
                        line_word_index=line_word_index,
                        expected=word.word,
                        actual=word.transcript_word or "",
                        similarity=word.similarity,
                        start_seconds=word.start_seconds,
                        end_seconds=word.end_seconds,
                        metadata={
                            "confidence": word.confidence,
                            "min_confidence": confidence_threshold,
                        },
                    )
                )

    for heard_index in extras:
        heard_token = parsed.tokens[heard_index]
        if ignore_fillers and heard_token.key in _FILLERS:
            continue
        if not report_extra:
            continue
        extra_line: int | None = None
        for word in aligned_words:
            if (
                word.transcript_index is not None
                and word.transcript_index <= heard_index
            ):
                extra_line = word.line_index
        mismatches.append(
            _mismatch(
                "extra_word",
                f"heard {heard_token.text!r} which is not in the lyric",
                severity=Severity.INFO,
                line_index=extra_line,
                word_index=heard_index,
                actual=heard_token.text,
                start_seconds=heard_token.start,
                end_seconds=heard_token.end,
                metadata={"transcript_index": heard_index},
            )
        )

    previous_end: float | None = None
    for line in lines:
        if line.start_seconds is None or line.end_seconds is None:
            mismatches.append(
                _mismatch(
                    "untimed_line",
                    f"line {line.index + 1 if line.index is not None else 1} "
                    "could not be timed",
                    line_index=line.index,
                    expected=line.text,
                    similarity=line.similarity,
                )
            )
            continue
        if previous_end is not None and line.start_seconds < previous_end - _EPSILON:
            mismatches.append(
                _mismatch(
                    "timing_overlap",
                    f"line {line.index + 1 if line.index is not None else 1} "
                    "starts before the previous line ends",
                    severity=Severity.ERROR,
                    line_index=line.index,
                    expected=line.text,
                    start_seconds=line.start_seconds,
                    end_seconds=line.end_seconds,
                    metadata={"previous_end_seconds": previous_end},
                )
            )
        previous_end = line.end_seconds
        if line.matched and line.coverage < line_score - _EPSILON:
            line.metadata["below_line_threshold"] = True
            line.metadata["coverage"] = line.coverage
            mismatches.append(
                _mismatch(
                    "weak_line",
                    f"line {line.index + 1 if line.index is not None else 1} "
                    f"matched only {line.coverage:.0%} of its words",
                    severity=Severity.INFO,
                    line_index=line.index,
                    expected=line.text,
                    similarity=line.similarity,
                    start_seconds=line.start_seconds,
                    end_seconds=line.end_seconds,
                    metadata={
                        "coverage": line.coverage,
                        "line_threshold": line_score,
                    },
                )
            )

    mismatches.sort(
        key=lambda item: (
            -1 if item.line_index is None else item.line_index,
            -1 if item.word_index is None else item.word_index,
            item.code,
        )
    )
    for line in lines:
        line.mismatches = [
            mismatch for mismatch in mismatches if mismatch.line_index == line.index
        ]

    return LyricAlignment(
        lyrics="\n".join(spec.text for spec in specs),
        lines=lines,
        mismatches=mismatches,
        threshold=score_threshold,
        line_threshold=line_score,
        language=(
            language.strip().lower()
            if language and language.strip()
            else parsed.language
        ),
        duration_seconds=duration,
        provider=parsed.provider,
        model=parsed.model,
        transcript=parsed.response,
        metadata={
            "lines": len(lines),
            "words": len(aligned_words),
            "matched_words": len([word for word in aligned_words if word.matched]),
            "unmatched_words": len(
                [word for word in aligned_words if not word.matched]
            ),
            "extra_words": len(extras),
            "matched_lines": len([line for line in lines if line.matched]),
            "untimed_lines": len([line for line in lines if not line.aligned]),
            "mismatches": len(mismatches),
            "missing_words": len(
                [item for item in mismatches if item.kind == "missing"]
            ),
            "extra_word_mismatches": len(
                [item for item in mismatches if item.kind == "extra"]
            ),
            "substitutions": len(
                [item for item in mismatches if item.kind == "substitution"]
            ),
            "line_coverage": (
                len([line for line in lines if line.matched]) / len(lines)
                if lines
                else 1.0
            ),
        },
    )


align_song = align_lyrics
"""Alias for :func:`align_lyrics`."""

align_lyric = align_lyrics
"""Alias for :func:`align_lyrics`."""

align_song_lyrics = align_lyrics
"""Alias for :func:`align_lyrics`."""

align_lyric_lines = align_lyrics
"""Alias for :func:`align_lyrics`."""

align_lyrics_to_song = align_lyrics
"""Alias for :func:`align_lyrics`."""

align_lyrics_to_transcript = align_lyrics
"""Alias for :func:`align_lyrics`."""

lyric_alignment = align_lyrics
"""Alias for :func:`align_lyrics`."""

forced_align_lyrics = align_lyrics
"""Alias for :func:`align_lyrics`."""


async def align_lyrics_audio(
    lyrics: Any,
    audio: bytes,
    provider: Any,
    *,
    request: STTRequest | None = None,
    threshold: float = DEFAULT_LYRIC_ALIGNMENT_THRESHOLD,
    line_threshold: float = DEFAULT_LINE_THRESHOLD,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    interpolate: bool = False,
    max_merge_tokens: int = DEFAULT_MAX_MERGE_TOKENS,
    report_extra: bool = True,
    report_fuzzy: bool = True,
    ignore_fillers: bool = True,
    language: str | None = None,
    model: str | None = None,
    prompt: str | None = None,
    timeout_seconds: float | None = None,
) -> LyricAlignment:
    """Transcribe *audio* with *provider* and align *lyrics* to the result.

    The provider is only called for transcription. Alignment itself is the
    same deterministic operation as :func:`align_lyrics`.
    """
    if not isinstance(audio, (bytes, bytearray)) or not audio:
        raise ValidationError(
            "audio must be non-empty bytes",
            details={"field": "audio"},
        )
    transcribe = getattr(provider, "transcribe", None)
    if not callable(transcribe):
        raise TypeError("provider must expose an async transcribe method")
    if request is not None and not isinstance(request, STTRequest):
        raise TypeError("request must be a STTRequest or None")
    if request is None:
        prepared = STTRequest(
            audio=bytes(audio),
            language=language,
            model=model,
            prompt=prompt,
            timeout_seconds=timeout_seconds,
            word_timestamps=True,
        )
    else:
        prepared = request.model_copy(
            update={
                "audio": bytes(audio),
                "audio_url": None,
                "word_timestamps": True,
            }
        )
    result = transcribe(prepared)
    if inspect.isawaitable(result):
        result = await result
    if not isinstance(result, STTResponse):
        raise ValidationError(
            "speech to text provider returned an invalid response",
            details={"provider": getattr(provider, "name", "")},
        )
    return align_lyrics(
        lyrics,
        result,
        threshold=threshold,
        line_threshold=line_threshold,
        min_confidence=min_confidence,
        interpolate=interpolate,
        max_merge_tokens=max_merge_tokens,
        report_extra=report_extra,
        report_fuzzy=report_fuzzy,
        ignore_fillers=ignore_fillers,
        language=language,
    )


align_lyrics_to_audio = align_lyrics_audio
"""Alias for :func:`align_lyrics_audio`."""


def find_lyric_mismatches(
    lyrics: Any,
    transcript: Any = None,
    *,
    words: Any = None,
    **options: Any,
) -> list[LyricMismatch]:
    """Return only the mismatches from :func:`align_lyrics`."""
    return align_lyrics(lyrics, transcript, words=words, **options).mismatches


find_mismatches = find_lyric_mismatches
"""Alias for :func:`find_lyric_mismatches`."""

lyric_mismatches = find_lyric_mismatches
"""Alias for :func:`find_lyric_mismatches`."""

report_lyric_mismatches = find_lyric_mismatches
"""Alias for :func:`find_lyric_mismatches`."""

report_mismatches = find_lyric_mismatches
"""Alias for :func:`find_lyric_mismatches`."""


class LyricAligner:
    """A reusable lyric aligner with one threshold and reporting policy."""

    def __init__(
        self,
        threshold: float = DEFAULT_LYRIC_ALIGNMENT_THRESHOLD,
        *,
        line_threshold: float = DEFAULT_LINE_THRESHOLD,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        interpolate: bool = False,
        max_merge_tokens: int = DEFAULT_MAX_MERGE_TOKENS,
        report_extra: bool = True,
        report_fuzzy: bool = True,
        ignore_fillers: bool = True,
    ) -> None:
        self._threshold = _threshold(threshold, field="threshold")
        self._line_threshold = _threshold(line_threshold, field="line_threshold")
        self._min_confidence = _threshold(min_confidence, field="min_confidence")
        if not isinstance(interpolate, bool):
            raise TypeError("interpolate must be a bool")
        if not isinstance(report_extra, bool):
            raise TypeError("report_extra must be a bool")
        if not isinstance(report_fuzzy, bool):
            raise TypeError("report_fuzzy must be a bool")
        if not isinstance(ignore_fillers, bool):
            raise TypeError("ignore_fillers must be a bool")
        if isinstance(max_merge_tokens, bool) or not isinstance(max_merge_tokens, int):
            raise TypeError("max_merge_tokens must be an integer")
        if max_merge_tokens < 1:
            raise ValidationError(
                "max_merge_tokens must be positive",
                details={"field": "max_merge_tokens", "value": max_merge_tokens},
            )
        self._interpolate = interpolate
        self._max_merge_tokens = max_merge_tokens
        self._report_extra = report_extra
        self._report_fuzzy = report_fuzzy
        self._ignore_fillers = ignore_fillers

    @property
    def threshold(self) -> float:
        """The similarity needed to pair a lyric word."""
        return self._threshold

    @property
    def line_threshold(self) -> float:
        """The line coverage needed to avoid a weak-line note."""
        return self._line_threshold

    @property
    def min_confidence(self) -> float:
        """The provider confidence below which a word is reported."""
        return self._min_confidence

    @property
    def interpolate(self) -> bool:
        """Whether unmatched words receive distributed timing."""
        return self._interpolate

    def align(self, lyrics: Any, transcript: Any) -> LyricAlignment:
        """Align *lyrics* to *transcript*."""
        return align_lyrics(
            lyrics,
            transcript,
            threshold=self._threshold,
            line_threshold=self._line_threshold,
            min_confidence=self._min_confidence,
            interpolate=self._interpolate,
            max_merge_tokens=self._max_merge_tokens,
            report_extra=self._report_extra,
            report_fuzzy=self._report_fuzzy,
            ignore_fillers=self._ignore_fillers,
        )

    def __call__(self, lyrics: Any, transcript: Any) -> LyricAlignment:
        return self.align(lyrics, transcript)


LyricAlign = LyricAligner
"""Alias for :class:`LyricAligner`."""
