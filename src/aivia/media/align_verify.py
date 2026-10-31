"""Forced alignment of known text to audio and fuzzy term verification.

A speech to text provider reports what it heard and when it heard it. A script
already knows what *should* have been said. This module joins the two without
calling a provider itself:

* :func:`align_text` maps every word of a known script onto the timed words of
  a transcript. Exact words are paired first, then near matches above a
  configurable threshold. Words that were not spoken stay untimed rather than
  receiving an invented timestamp; :func:`align_text` can interpolate them
  explicitly when a caller needs a complete timing list.
* :func:`verify_terms` checks names, brands and other expected terms against a
  transcript with a deterministic fuzzy match. A term is verified when its
  best window scores at or above the configured threshold.

Both operations are pure and offline. They never call a provider and never
spend money, so they are safe to run on every generated voice-over. A provider
call is only made by :func:`align_audio`, which transcribes first and then
delegates to :func:`align_text`.
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

from aivia.core.config.media import (
    DEFAULT_MEDIA_ALIGNMENT_SIMILARITY_THRESHOLD,
    MAX_MEDIA_SIMILARITY,
    MIN_MEDIA_SIMILARITY,
)
from aivia.core.config.orders import NameVerifyConfig
from aivia.core.errors import ValidationError
from aivia.providers.protocols.audio import WordTimestamp
from aivia.providers.protocols.stt import STTRequest, STTResponse
from aivia.providers.stt.common import (
    response_duration_seconds,
    segments_from_values,
    words_from_values,
)

__all__ = [
    "DEFAULT_ALIGNMENT_THRESHOLD",
    "DEFAULT_ALIGN_THRESHOLD",
    "DEFAULT_FUZZY_THRESHOLD",
    "DEFAULT_MATCH_THRESHOLD",
    "DEFAULT_SIMILARITY_THRESHOLD",
    "DEFAULT_TERM_THRESHOLD",
    "DEFAULT_VERIFY_THRESHOLD",
    "AlignedToken",
    "AlignedWord",
    "Alignment",
    "AlignmentResult",
    "AlignmentWord",
    "ForcedAligner",
    "ForcedAlignment",
    "TermMatch",
    "TermResult",
    "TermVerification",
    "TermVerifier",
    "VerificationResult",
    "align",
    "align_audio",
    "align_known_text",
    "align_text",
    "align_text_to_audio",
    "align_text_to_transcript",
    "align_transcript",
    "align_word_timestamps",
    "align_words",
    "aligned_word_timestamps",
    "check_terms",
    "find_term",
    "forced_align",
    "forced_align_text",
    "fuzzy_match",
    "fuzzy_match_terms",
    "match_term",
    "match_terms",
    "phrase_similarity",
    "similarity",
    "term_match",
    "term_matches",
    "term_similarity",
    "terms_match",
    "to_word_timestamps",
    "verification_threshold",
    "verify_name",
    "verify_names",
    "verify_term",
    "verify_terms",
    "verify_transcript",
    "word_similarity",
]

DEFAULT_ALIGNMENT_THRESHOLD = DEFAULT_MEDIA_ALIGNMENT_SIMILARITY_THRESHOLD
"""Word similarity at or above this pairs a script word with a transcript word."""

DEFAULT_ALIGN_THRESHOLD = DEFAULT_ALIGNMENT_THRESHOLD
"""Alias for :data:`DEFAULT_ALIGNMENT_THRESHOLD`."""

DEFAULT_TERM_THRESHOLD = NameVerifyConfig().threshold
"""Fuzzy score at or above this verifies an expected term."""

DEFAULT_VERIFY_THRESHOLD = DEFAULT_TERM_THRESHOLD
"""Alias for :data:`DEFAULT_TERM_THRESHOLD`."""

DEFAULT_MATCH_THRESHOLD = DEFAULT_ALIGNMENT_THRESHOLD
"""Alias for :data:`DEFAULT_ALIGNMENT_THRESHOLD`."""

DEFAULT_SIMILARITY_THRESHOLD = DEFAULT_ALIGNMENT_THRESHOLD
"""Alias for :data:`DEFAULT_ALIGNMENT_THRESHOLD`."""

DEFAULT_FUZZY_THRESHOLD = DEFAULT_TERM_THRESHOLD
"""Alias for :data:`DEFAULT_TERM_THRESHOLD`."""

_WORD = re.compile(r"[^\W_]+(?:['\u2019][^\W_]+)*", re.UNICODE)
"""A word, optionally with an internal apostrophe, but without punctuation."""


def _fold(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in folded if not unicodedata.combining(char))


def _normalize(text: str) -> str:
    return "".join(char for char in _fold(text) if char.isalnum())


def _tokens(text: str) -> list[str]:
    return [match.group() for match in _WORD.finditer(text)]


def _require_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    text = value.strip()
    if not text:
        raise ValidationError(
            f"{field} must be non-empty",
            details={"field": field},
        )
    return text


def _string(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    return value


def _validate_threshold(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(
            f"{field} must be a number between 0 and 1",
            details={"field": field, "value": value},
        )
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValidationError(
            f"{field} must be between 0 and 1",
            details={"field": field, "value": value},
        )
    return number


def _similarity(left: str, right: str) -> float:
    left_key = _normalize(left)
    right_key = _normalize(right)
    if not left_key or not right_key:
        return 1.0 if left_key == right_key else 0.0
    if left_key == right_key:
        return 1.0
    return SequenceMatcher(None, left_key, right_key).ratio()


def word_similarity(left: str, right: str) -> float:
    """Return the normalized character similarity of two words in ``[0, 1]``.

    Case, accents and punctuation are folded away, so ``"Café"`` and ``"cafe"``
    score ``1.0`` and ``"Smyth"``/``"Smith"`` score above the default threshold.
    Two empty words are identical; one empty word is unrelated.
    """
    return _similarity(
        _string(left, field="left"),
        _string(right, field="right"),
    )


def phrase_similarity(left: str, right: str) -> float:
    """Return the normalized character similarity of two phrases in ``[0, 1]``."""
    return _similarity(
        _string(left, field="left"),
        _string(right, field="right"),
    )


def term_similarity(term: str, candidate: str) -> float:
    """Return how closely *candidate* matches the expected *term*."""
    return phrase_similarity(term, candidate)


similarity = word_similarity
"""Alias for :func:`word_similarity`."""


@dataclass(frozen=True)
class _Token:
    text: str
    norm: str
    start: float | None
    end: float | None
    confidence: float | None
    speaker: str | None
    index: int | None


@dataclass(frozen=True)
class _Transcript:
    tokens: list[_Token]
    duration: float | None
    language: str | None
    provider: str
    model: str
    response: STTResponse | None


def _token_from_word(word: WordTimestamp, index: int) -> _Token:
    return _Token(
        text=word.word,
        norm=_normalize(word.word),
        start=word.start_seconds,
        end=word.end_seconds,
        confidence=word.confidence,
        speaker=word.speaker,
        index=index,
    )


def _token_from_text(text: str) -> _Token:
    return _Token(
        text=text,
        norm=_normalize(text),
        start=None,
        end=None,
        confidence=None,
        speaker=None,
        index=None,
    )


def _tokens_from_text(text: str) -> list[_Token]:
    return [_token_from_text(token) for token in _tokens(text)]


def _tokens_from_words(words: Sequence[Any]) -> list[_Token]:
    tokens: list[_Token] = []
    for index, raw in enumerate(words):
        if isinstance(raw, WordTimestamp):
            word = raw
        elif isinstance(raw, Mapping):
            converted = words_from_values([raw])
            if not converted:
                raise ValidationError(
                    "transcript word is invalid",
                    details={"index": index},
                )
            word = converted[0]
        elif isinstance(raw, str):
            tokens.extend(_tokens_from_text(raw))
            continue
        else:
            raise ValidationError(
                "transcript words must be WordTimestamp values or mappings",
                details={"index": index},
            )
        tokens.append(_token_from_word(word, index))
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
                values["words"] = words_from_values(values["words"])
            if "segments" in values:
                values["segments"] = segments_from_values(values["segments"])
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


class AlignedWord(BaseModel):
    """One known script word and the transcript word it aligned to.

    ``start_seconds`` and ``end_seconds`` are ``None`` when the word was not
    found in the transcript and no interpolation was requested. ``matched`` is
    true only for a transcript word; interpolated words are timed but not
    matched.
    """

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
    )

    word: str = Field(
        min_length=1,
        validation_alias=AliasChoices("word", "text", "token"),
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
        validation_alias=AliasChoices(
            "similarity",
            "score",
            "match_score",
        ),
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
        validation_alias=AliasChoices("transcript_index", "index"),
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("confidence", "probability"),
    )
    interpolated: bool = Field(
        default=False,
        validation_alias=AliasChoices("interpolated", "estimated"),
    )
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
    def matched_text(self) -> str | None:
        """Alias for :attr:`transcript_word`."""
        return self.transcript_word

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
        """Whether the word matched the transcript exactly."""
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


class AlignmentResult(BaseModel):
    """The aligned words for one known script.

    ``words`` holds every script word in order. Filter with
    :attr:`aligned_words` for the words with timing, or :attr:`unmatched` for
    the words the transcript did not contain. :attr:`word_timestamps` returns
    the protocol model consumed by subtitles.
    """

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )

    text: str = Field(
        validation_alias=AliasChoices("text", "script", "known_text"),
    )
    words: list[AlignedWord] = Field(default_factory=list)
    threshold: float = Field(
        default=DEFAULT_ALIGNMENT_THRESHOLD,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("threshold", "min_similarity"),
    )
    language: str | None = None
    duration_seconds: float | None = Field(default=None, ge=0.0)
    provider: str = ""
    model: str = ""
    transcript: STTResponse | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text", "provider", "model", mode="before")
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
    def tokens(self) -> list[AlignedWord]:
        """Alias for :attr:`words`."""
        return self.words

    @property
    def word_count(self) -> int:
        """The number of script words."""
        return len(self.words)

    @property
    def matched_count(self) -> int:
        """The number of script words found in the transcript."""
        return len(self.matched_words)

    @property
    def unmatched_count(self) -> int:
        """The number of script words missing from the transcript."""
        return len(self.unmatched)

    @property
    def aligned(self) -> bool:
        """Whether any script word received timing."""
        return self.has_timings

    @property
    def aligned_words(self) -> list[AlignedWord]:
        """The words that have both a start and an end time."""
        return [word for word in self.words if word.aligned]

    @property
    def matched_words(self) -> list[AlignedWord]:
        """The words found in the transcript."""
        return [word for word in self.words if word.matched]

    @property
    def unmatched(self) -> list[AlignedWord]:
        """The script words the transcript did not contain."""
        return [word for word in self.words if not word.matched]

    @property
    def missing(self) -> list[AlignedWord]:
        """Alias for :attr:`unmatched`."""
        return self.unmatched

    @property
    def unmatched_words(self) -> list[AlignedWord]:
        """Alias for :attr:`unmatched`."""
        return self.unmatched

    @property
    def coverage(self) -> float:
        """The share of script words matched to a transcript word."""
        if not self.words:
            return 1.0
        return len(self.matched_words) / len(self.words)

    @property
    def score(self) -> float:
        """Alias for :attr:`coverage`."""
        return self.coverage

    @property
    def has_timings(self) -> bool:
        """Whether any script word received timing."""
        return bool(self.aligned_words)

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

    def to_word_timestamps(self) -> list[WordTimestamp]:
        """Return :attr:`word_timestamps`."""
        return self.word_timestamps

    def as_word_timestamps(self) -> list[WordTimestamp]:
        """Alias for :meth:`to_word_timestamps`."""
        return self.word_timestamps

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.words)

    def __len__(self) -> int:
        return len(self.words)

    def __getitem__(self, index: int) -> AlignedWord:
        return self.words[index]

    def __bool__(self) -> bool:
        return bool(self.words)


def _pair_tokens(
    known: list[_Token],
    transcript: list[_Token],
    *,
    threshold: float,
) -> dict[int, tuple[int, float]]:
    pairs: dict[int, tuple[int, float]] = {}
    matcher = SequenceMatcher(
        None,
        [token.norm for token in known],
        [token.norm for token in transcript],
        autojunk=False,
    )
    for tag, left_start, left_end, right_start, right_end in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(left_end - left_start):
                pairs[left_start + offset] = (
                    right_start + offset,
                    1.0,
                )
            continue
        if tag != "replace":
            continue
        candidates: list[tuple[float, int, int]] = []
        for left_index in range(left_start, left_end):
            for right_index in range(right_start, right_end):
                score = _similarity(
                    known[left_index].text,
                    transcript[right_index].text,
                )
                if score >= threshold:
                    candidates.append((score, left_index, right_index))
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        used_left: set[int] = set()
        used_right: set[int] = set()
        for score, left_index, right_index in candidates:
            if left_index in used_left or right_index in used_right:
                continue
            pairs[left_index] = (right_index, score)
            used_left.add(left_index)
            used_right.add(right_index)
    return pairs


def _interpolate(
    words: list[AlignedWord],
    *,
    duration: float | None,
) -> None:
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


def align_text(
    text: str,
    transcript: Any = None,
    *,
    words: Any = None,
    threshold: float = DEFAULT_ALIGNMENT_THRESHOLD,
    min_similarity: float | None = None,
    min_score: float | None = None,
    interpolate: bool = False,
) -> AlignmentResult:
    """Align every word of *text* to the timed words in *transcript*.

    *transcript* may be an :class:`~aivia.providers.protocols.stt.STTResponse`,
    a sequence of :class:`~aivia.providers.protocols.audio.WordTimestamp`
    values, a mapping with ``words``/``segments``, or plain text. Plain text has
    no timing, so alignment still reports which words matched but leaves every
    timestamp empty. ``words`` is accepted as an alternative keyword for the
    transcript, and ``min_similarity``/``min_score`` alias *threshold*.

    Exact and fuzzy matches are paired; a script word that is not spoken stays
    untimed unless *interpolate* is true. Interpolation distributes a gap
    between neighbouring anchors by word length, never calls a provider and is
    marked on each word.
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
    known_text = _require_text(text, field="text")
    score_threshold = _validate_threshold(threshold, field="threshold")
    if not isinstance(interpolate, bool):
        raise TypeError("interpolate must be a bool")
    parsed = _transcript_from_source(transcript)
    known = _tokens_from_text(known_text)
    pairs = _pair_tokens(known, parsed.tokens, threshold=score_threshold)

    aligned: list[AlignedWord] = []
    for index, token in enumerate(known):
        pair = pairs.get(index)
        if pair is None:
            aligned.append(
                AlignedWord(
                    word=token.text,
                    similarity=MIN_MEDIA_SIMILARITY,
                    matched=False,
                )
            )
            continue
        right_index, score = pair
        other = parsed.tokens[right_index]
        aligned.append(
            AlignedWord(
                word=token.text,
                start_seconds=other.start,
                end_seconds=other.end,
                similarity=score,
                matched=True,
                transcript_word=other.text,
                transcript_index=other.index,
                confidence=other.confidence,
            )
        )

    duration = parsed.duration
    if duration is None:
        ends = [word.end_seconds for word in aligned if word.end_seconds is not None]
        duration = max(ends) if ends else None
    if interpolate:
        _interpolate(aligned, duration=duration)

    return AlignmentResult(
        text=known_text,
        words=aligned,
        threshold=score_threshold,
        language=parsed.language,
        duration_seconds=duration,
        provider=parsed.provider,
        model=parsed.model,
        transcript=parsed.response,
        metadata={
            "matched": len([word for word in aligned if word.matched]),
            "unmatched": len([word for word in aligned if not word.matched]),
        },
    )


align_transcript = align_text
"""Alias for :func:`align_text`."""

align_words = align_text
"""Alias for :func:`align_text`."""


def aligned_word_timestamps(
    text: str,
    transcript: Any = None,
    *,
    words: Any = None,
    threshold: float = DEFAULT_ALIGNMENT_THRESHOLD,
    min_similarity: float | None = None,
    min_score: float | None = None,
    interpolate: bool = False,
) -> list[WordTimestamp]:
    """Return only the timed words from :func:`align_text`."""
    return align_text(
        text,
        transcript,
        words=words,
        threshold=threshold,
        min_similarity=min_similarity,
        min_score=min_score,
        interpolate=interpolate,
    ).word_timestamps


align_word_timestamps = aligned_word_timestamps
"""Alias for :func:`aligned_word_timestamps`."""

to_word_timestamps = aligned_word_timestamps
"""Alias for :func:`aligned_word_timestamps`."""


async def align_audio(
    text: str,
    audio: bytes,
    provider: Any,
    *,
    request: STTRequest | None = None,
    threshold: float = DEFAULT_ALIGNMENT_THRESHOLD,
    interpolate: bool = False,
    language: str | None = None,
    model: str | None = None,
    prompt: str | None = None,
    timeout_seconds: float | None = None,
) -> AlignmentResult:
    """Transcribe *audio* with *provider* and align it to the known *text*.

    The provider is only called for transcription. Alignment itself is the
    same deterministic operation as :func:`align_text`, so the returned words
    never depend on which speech to text provider produced the timing.
    """
    known_text = _require_text(text, field="text")
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
    return align_text(
        known_text,
        result,
        threshold=threshold,
        interpolate=interpolate,
    )


def forced_align(
    text: str,
    source: Any = None,
    *,
    transcript: Any = None,
    words: Any = None,
    provider: Any = None,
    request: STTRequest | None = None,
    threshold: float = DEFAULT_ALIGNMENT_THRESHOLD,
    interpolate: bool = False,
    language: str | None = None,
    model: str | None = None,
    prompt: str | None = None,
    timeout_seconds: float | None = None,
) -> AlignmentResult | Any:
    """Align known text to a transcript, or transcribe audio first.

    Without *provider* this is a synchronous :func:`align_text` over
    *transcript*, *words* or *source*. With a provider and raw audio it returns
    the awaitable from :func:`align_audio`.
    """
    chosen = transcript if transcript is not None else words
    if chosen is None:
        chosen = source
    if provider is not None and isinstance(chosen, (bytes, bytearray)):
        return align_audio(
            text,
            bytes(chosen),
            provider,
            request=request,
            threshold=threshold,
            interpolate=interpolate,
            language=language,
            model=model,
            prompt=prompt,
            timeout_seconds=timeout_seconds,
        )
    if chosen is None:
        raise ValidationError(
            "a transcript, words or audio source is required",
            details={"field": "transcript"},
        )
    return align_text(
        text,
        chosen,
        threshold=threshold,
        interpolate=interpolate,
    )


align = forced_align
"""Alias for :func:`forced_align`."""

align_text_to_audio = forced_align
"""Alias for :func:`forced_align`."""

align_known_text = align_text
"""Alias for :func:`align_text`."""

align_text_to_transcript = align_text
"""Alias for :func:`align_text`."""

forced_align_text = align_text
"""Alias for :func:`align_text`."""


class ForcedAligner:
    """A reusable aligner with one threshold and interpolation policy."""

    def __init__(
        self,
        threshold: float = DEFAULT_ALIGNMENT_THRESHOLD,
        *,
        interpolate: bool = False,
    ) -> None:
        self._threshold = _validate_threshold(threshold, field="threshold")
        if not isinstance(interpolate, bool):
            raise TypeError("interpolate must be a bool")
        self._interpolate = interpolate

    @property
    def threshold(self) -> float:
        """The similarity needed to pair a word."""
        return self._threshold

    @property
    def interpolate(self) -> bool:
        """Whether unmatched words receive distributed timing."""
        return self._interpolate

    def align(self, text: str, transcript: Any) -> AlignmentResult:
        """Align *text* to *transcript*."""
        return align_text(
            text,
            transcript,
            threshold=self._threshold,
            interpolate=self._interpolate,
        )

    def __call__(self, text: str, transcript: Any) -> AlignmentResult:
        return self.align(text, transcript)


class TermMatch(BaseModel):
    """The best fuzzy match for one expected term in a transcript."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
    )

    term: str = Field(
        min_length=1,
        validation_alias=AliasChoices("term", "text", "expected"),
    )
    matched: bool = Field(
        default=False,
        validation_alias=AliasChoices("matched", "verified", "found"),
    )
    score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("score", "similarity", "match_score"),
    )
    matched_text: str = Field(
        default="",
        validation_alias=AliasChoices(
            "matched_text",
            "candidate",
            "matched_word",
            "heard",
        ),
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
    index: int | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices("index", "transcript_index"),
    )
    words: list[WordTimestamp] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("term", "matched_text", mode="before")
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
    def text(self) -> str:
        """Alias for :attr:`term`."""
        return self.term

    @property
    def verified(self) -> bool:
        """Alias for :attr:`matched`."""
        return self.matched

    @property
    def similarity(self) -> float:
        """Alias for :attr:`score`."""
        return self.score

    @property
    def matched_word(self) -> str:
        """Alias for :attr:`matched_text`."""
        return self.matched_text

    @property
    def start(self) -> float | None:
        """Alias for :attr:`start_seconds`."""
        return self.start_seconds

    @property
    def end(self) -> float | None:
        """Alias for :attr:`end_seconds`."""
        return self.end_seconds

    @property
    def duration_seconds(self) -> float | None:
        """The timed span of the match, or ``None`` when untimed."""
        if self.start_seconds is None or self.end_seconds is None:
            return None
        return self.end_seconds - self.start_seconds

    def __bool__(self) -> bool:
        return self.matched

    def __eq__(self, other: object) -> bool:
        if isinstance(other, bool):
            return self.matched is other
        return super().__eq__(other)

    def __hash__(self) -> int:
        return hash((self.term, self.matched, self.score, self.matched_text))


class TermVerification(BaseModel):
    """The result of verifying every expected term against a transcript."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    terms: list[TermMatch] = Field(default_factory=list)
    threshold: float = Field(
        default=DEFAULT_TERM_THRESHOLD,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("threshold", "min_similarity"),
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def results(self) -> list[TermMatch]:
        """Alias for :attr:`terms`."""
        return self.terms

    @property
    def matches(self) -> list[TermMatch]:
        """The terms found in the transcript."""
        return [match for match in self.terms if match.matched]

    @property
    def missing(self) -> list[TermMatch]:
        """The terms not found in the transcript."""
        return [match for match in self.terms if not match.matched]

    @property
    def failed(self) -> list[TermMatch]:
        """Alias for :attr:`missing`."""
        return self.missing

    @property
    def matched_terms(self) -> tuple[str, ...]:
        """The matched terms in input order."""
        return tuple(match.term for match in self.matches)

    @property
    def missing_terms(self) -> tuple[str, ...]:
        """The missing terms in input order."""
        return tuple(match.term for match in self.missing)

    @property
    def all_matched(self) -> bool:
        """Whether every expected term was found."""
        return all(match.matched for match in self.terms)

    @property
    def verified(self) -> bool:
        """Alias for :attr:`all_matched`."""
        return self.all_matched

    @property
    def passed(self) -> bool:
        """Alias for :attr:`all_matched`."""
        return self.all_matched

    @property
    def failed_terms(self) -> tuple[str, ...]:
        """Alias for :attr:`missing_terms`."""
        return self.missing_terms

    @property
    def score(self) -> float:
        """The lowest term score, or ``1.0`` when there are no terms."""
        if not self.terms:
            return 1.0
        return min(match.score for match in self.terms)

    @property
    def scores(self) -> dict[str, float]:
        """Map each term to its best fuzzy score."""
        return {match.term: match.score for match in self.terms}

    @property
    def by_term(self) -> dict[str, TermMatch]:
        """Map each term to its :class:`TermMatch`."""
        return {match.term: match for match in self.terms}

    def as_mapping(self) -> dict[str, bool]:
        """Return the verification as a term-to-boolean mapping."""
        return {match.term: match.matched for match in self.terms}

    def keys(self) -> tuple[str, ...]:
        """The verified terms in input order."""
        return tuple(match.term for match in self.terms)

    def values(self) -> tuple[TermMatch, ...]:
        """The term matches in input order."""
        return tuple(self.terms)

    def items(self) -> tuple[tuple[str, TermMatch], ...]:
        """The term and match pairs in input order."""
        return tuple((match.term, match) for match in self.terms)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.terms)

    def __len__(self) -> int:
        return len(self.terms)

    def __getitem__(self, index: int | str) -> TermMatch:
        if isinstance(index, str):
            return self.by_term[index]
        return self.terms[index]

    def __contains__(self, term: object) -> bool:
        return isinstance(term, str) and term in self.by_term

    def __bool__(self) -> bool:
        return self.all_matched

    def __eq__(self, other: object) -> bool:
        if isinstance(other, bool):
            return self.all_matched is other
        return super().__eq__(other)

    def __hash__(self) -> int:
        return hash((tuple(self.terms), self.threshold))


def _term_tokens(term: str) -> list[_Token]:
    return _tokens_from_text(term)


def _match_term(
    term: str,
    parsed: _Transcript,
    *,
    threshold: float,
) -> TermMatch:
    expected = _term_tokens(term)
    if not expected or not parsed.tokens:
        return TermMatch(term=term, score=MIN_MEDIA_SIMILARITY, matched=False)

    size = len(expected)
    best_score = MIN_MEDIA_SIMILARITY
    best_start: int | None = None
    if len(parsed.tokens) < size:
        windows = [parsed.tokens]
        starts = [0]
    else:
        windows = [
            parsed.tokens[index : index + size]
            for index in range(len(parsed.tokens) - size + 1)
        ]
        starts = list(range(len(parsed.tokens) - size + 1))

    expected_key = " ".join(token.norm for token in expected)
    for start, window in zip(starts, windows, strict=True):
        candidate_key = " ".join(token.norm for token in window)
        if expected_key == candidate_key:
            score = MAX_MEDIA_SIMILARITY
        else:
            score = SequenceMatcher(None, expected_key, candidate_key).ratio()
        if score > best_score:
            best_score = score
            best_start = start
            if score >= MAX_MEDIA_SIMILARITY:
                break

    if best_start is None:
        return TermMatch(term=term, score=best_score, matched=False)

    window = windows[starts.index(best_start)]
    timed = [
        token for token in window if token.start is not None and token.end is not None
    ]
    words = [
        WordTimestamp(
            word=token.text,
            start_seconds=token.start,
            end_seconds=token.end,
            confidence=token.confidence,
        )
        for token in timed
        if token.start is not None and token.end is not None
    ]
    return TermMatch(
        term=term,
        matched=best_score >= threshold,
        score=best_score,
        matched_text=" ".join(token.text for token in window),
        start_seconds=timed[0].start if timed else None,
        end_seconds=timed[-1].end if timed else None,
        index=window[0].index if window and window[0].index is not None else None,
        words=words,
    )


def verify_term(
    term: str,
    transcript: Any = None,
    *,
    text: str | None = None,
    threshold: float | None = None,
    config: Any = None,
    min_similarity: float | None = None,
    min_score: float | None = None,
) -> TermMatch:
    """Return the best fuzzy match for one expected *term*."""
    if transcript is None:
        transcript = text
    if transcript is None:
        raise ValidationError(
            "a transcript is required",
            details={"field": "transcript"},
        )
    if min_similarity is not None:
        threshold = min_similarity
    if min_score is not None:
        threshold = min_score
    expected = _require_text(term, field="term")
    score_threshold = (
        verification_threshold(config)
        if threshold is None
        else _validate_threshold(threshold, field="threshold")
    )
    parsed = _transcript_from_source(transcript)
    return _match_term(expected, parsed, threshold=score_threshold)


term_match = verify_term
"""Alias for :func:`verify_term`."""

match_term = verify_term
"""Alias for :func:`verify_term`."""

find_term = verify_term
"""Alias for :func:`verify_term`."""

fuzzy_match = verify_term
"""Alias for :func:`verify_term`."""


def term_matches(
    term: str,
    transcript: Any = None,
    *,
    text: str | None = None,
    threshold: float | None = None,
    config: Any = None,
    min_similarity: float | None = None,
    min_score: float | None = None,
) -> bool:
    """Return whether *term* is present in *transcript* above the threshold."""
    return verify_term(
        term,
        transcript,
        text=text,
        threshold=threshold,
        config=config,
        min_similarity=min_similarity,
        min_score=min_score,
    ).matched


def verify_terms(
    terms: str | Sequence[str],
    transcript: Any = None,
    *,
    text: str | None = None,
    threshold: float | None = None,
    config: Any = None,
    min_similarity: float | None = None,
    min_score: float | None = None,
    require_all: bool = False,
) -> TermVerification:
    """Verify every expected term against *transcript*.

    A single string is treated as one term. *require_all* does not change the
    returned object; it raises :class:`~aivia.core.errors.ValidationError` when
    a term is missing, which lets a caller gate a paid retry in one line.
    """
    if transcript is None:
        transcript = text
    if transcript is None:
        raise ValidationError(
            "a transcript is required",
            details={"field": "transcript"},
        )
    if min_similarity is not None:
        threshold = min_similarity
    if min_score is not None:
        threshold = min_score
    if isinstance(terms, str):
        expected = [_require_text(terms, field="term")]
    elif isinstance(terms, Sequence) and not isinstance(terms, (bytes, bytearray)):
        expected = [_require_text(term, field="term") for term in terms]
    else:
        raise TypeError("terms must be a string or a sequence of strings")
    if not isinstance(require_all, bool):
        raise TypeError("require_all must be a bool")
    score_threshold = (
        verification_threshold(config)
        if threshold is None
        else _validate_threshold(threshold, field="threshold")
    )
    parsed = _transcript_from_source(transcript)
    verification = TermVerification(
        terms=[
            _match_term(term, parsed, threshold=score_threshold) for term in expected
        ],
        threshold=score_threshold,
        metadata={"provider": parsed.provider, "model": parsed.model},
    )
    if require_all and not verification.all_matched:
        raise ValidationError(
            "one or more expected terms were not found",
            details={"missing": list(verification.missing_terms)},
        )
    return verification


verify_transcript = verify_terms
"""Alias for :func:`verify_terms`."""


def terms_match(
    terms: str | Sequence[str],
    transcript: Any = None,
    *,
    text: str | None = None,
    threshold: float | None = None,
    config: Any = None,
    min_similarity: float | None = None,
    min_score: float | None = None,
) -> dict[str, bool]:
    """Return a term-to-boolean mapping for *terms*."""
    verification = verify_terms(
        terms,
        transcript,
        text=text,
        threshold=threshold,
        config=config,
        min_similarity=min_similarity,
        min_score=min_score,
    )
    return verification.as_mapping()


def check_terms(
    terms: str | Sequence[str],
    transcript: Any = None,
    *,
    text: str | None = None,
    threshold: float | None = None,
    config: Any = None,
    min_similarity: float | None = None,
    min_score: float | None = None,
) -> bool:
    """Return whether every expected term was found."""
    return verify_terms(
        terms,
        transcript,
        text=text,
        threshold=threshold,
        config=config,
        min_similarity=min_similarity,
        min_score=min_score,
    ).all_matched


class TermVerifier:
    """A reusable verifier with one threshold, optionally from config."""

    def __init__(
        self,
        threshold: float | None = None,
        *,
        config: Any = None,
    ) -> None:
        self._threshold = (
            verification_threshold(config)
            if threshold is None
            else _validate_threshold(threshold, field="threshold")
        )

    @property
    def threshold(self) -> float:
        """The fuzzy score needed to verify a term."""
        return self._threshold

    def verify(
        self,
        terms: str | Sequence[str],
        transcript: Any,
        *,
        require_all: bool = False,
    ) -> TermVerification:
        """Verify *terms* against *transcript*."""
        return verify_terms(
            terms,
            transcript,
            threshold=self._threshold,
            require_all=require_all,
        )

    def term(self, term: str, transcript: Any) -> TermMatch:
        """Verify one term against *transcript*."""
        return verify_term(term, transcript, threshold=self._threshold)

    def score(self, term: str, transcript: Any) -> float:
        """Return the best score for one term without applying the threshold."""
        return verify_term(term, transcript, threshold=MIN_MEDIA_SIMILARITY).score

    def matches(self, term: str, transcript: Any) -> bool:
        """Return whether one term is verified."""
        return self.term(term, transcript).matched

    def __call__(
        self,
        terms: str | Sequence[str],
        transcript: Any,
        *,
        require_all: bool = False,
    ) -> TermVerification:
        return self.verify(terms, transcript, require_all=require_all)


def _config_value(config: Any, *path: str) -> Any:
    current = config
    for key in path:
        if isinstance(current, Mapping):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
        if current is None:
            return None
    return current


def verification_threshold(config: Any = None) -> float:
    """Resolve the fuzzy term threshold from configuration.

    Accepts a :class:`~aivia.core.config.orders.NameVerifyConfig`, an
    ``OrdersConfig``, or a mapping such as ``{"orders": {"name_verify":
    {"threshold": 0.9}}}``. A missing value falls back to the configured
    default. Invalid values raise :class:`~aivia.core.errors.ValidationError`.
    """
    if config is None:
        return DEFAULT_TERM_THRESHOLD
    if isinstance(config, NameVerifyConfig):
        return _validate_threshold(config.threshold, field="threshold")
    if isinstance(config, (int, float)) and not isinstance(config, bool):
        return _validate_threshold(config, field="threshold")
    if isinstance(config, Mapping):
        direct = config.get("threshold")
        if direct is not None:
            return _validate_threshold(direct, field="threshold")
        for path in (
            ("orders", "name_verify", "threshold"),
            ("orders", "name_verification", "threshold"),
            ("name_verify", "threshold"),
            ("name_verification", "threshold"),
            ("orders", "name_verify_threshold"),
            ("name_verify_threshold",),
        ):
            value = _config_value(config, *path)
            if value is not None:
                return _validate_threshold(value, field="threshold")
        return DEFAULT_TERM_THRESHOLD
    orders = getattr(config, "orders", config)
    verify = getattr(orders, "name_verify", None)
    if verify is None:
        verify = getattr(orders, "name_verification", None)
    if verify is not None:
        return _validate_threshold(
            getattr(verify, "threshold", DEFAULT_TERM_THRESHOLD),
            field="threshold",
        )
    value = getattr(orders, "name_verify_threshold", None)
    if value is not None:
        return _validate_threshold(value, field="threshold")
    return DEFAULT_TERM_THRESHOLD


verify_name = verify_term
"""Alias for :func:`verify_term`."""

verify_names = verify_terms
"""Alias for :func:`verify_terms`."""

match_terms = verify_terms
"""Alias for :func:`verify_terms`."""

fuzzy_match_terms = verify_terms
"""Alias for :func:`verify_terms`."""

Alignment = AlignmentResult
"""Alias for :class:`AlignmentResult`."""

ForcedAlignment = AlignmentResult
"""Alias for :class:`AlignmentResult`."""

AlignmentWord = AlignedWord
"""Alias for :class:`AlignedWord`."""

AlignedToken = AlignedWord
"""Alias for :class:`AlignedWord`."""

VerificationResult = TermVerification
"""Alias for :class:`TermVerification`."""

TermResult = TermMatch
"""Alias for :class:`TermMatch`."""


align_known_text_to_audio = forced_align
"""Alias for :func:`forced_align`."""

align_text_to_words = align_text
"""Alias for :func:`align_text`."""

verify_known_terms = verify_terms
"""Alias for :func:`verify_terms`."""

verify_expected_terms = verify_terms
"""Alias for :func:`verify_terms`."""

fuzzy_verify_terms = verify_terms
"""Alias for :func:`verify_terms`."""

term_verification = verify_terms
"""Alias for :func:`verify_terms`."""

FuzzyTermVerifier = TermVerifier
"""Alias for :class:`TermVerifier`."""

TextAligner = ForcedAligner
"""Alias for :class:`ForcedAligner`."""
