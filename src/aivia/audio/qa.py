"""Deterministic quality checks for synthesized speech.

Voice-over is synthesized, measured and timed before it is composed, but a
measured duration does not prove the audio is usable. This module reads the
voice segments of a timeline, or plain segment values, and reports four kinds
of problem without calling a provider or running a binary:

* **silence** -- a segment is mostly silent, starts or ends with a long pause,
  or contains an unexpectedly long internal gap. Explicit silence spans can be
  supplied by an analyser; otherwise the gaps are inferred from word timings.
* **duplication** -- a script repeats a word or sentence, or two segments speak
  exactly the same text in the same language.
* **truncation** -- the audio ends before the expected text, word timings run
  past the segment end, the duration is far shorter than the text needs, or the
  expected text ends with an ellipsis that may hide a cut.
* **mispronunciation** -- a transcript differs from the expected words, a
  word has low provider confidence, or an expected term is not one of its
  accepted pronunciations.

Every finding carries a severity so a caller can warn, retry or fail the stage.
The checks are pure: the same segments and thresholds always produce the same
findings in the same order.
"""

from __future__ import annotations

import itertools
import math
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from enum import StrEnum
from typing import Any

from aivia.compose.timeline import Timeline
from aivia.core.config import media as media_config

_TIME_TOLERANCE_SECONDS = media_config.AUDIO_QA_TIME_TOLERANCE_SECONDS
DEFAULT_EXPECTED_WORDS_PER_SECOND = (
    media_config.DEFAULT_AUDIO_QA_EXPECTED_WORDS_PER_SECOND
)
DEFAULT_MAX_INTERNAL_SILENCE_SECONDS = (
    media_config.DEFAULT_AUDIO_QA_MAX_INTERNAL_SILENCE_SECONDS
)
DEFAULT_MAX_LEADING_SILENCE_SECONDS = (
    media_config.DEFAULT_AUDIO_QA_MAX_LEADING_SILENCE_SECONDS
)
DEFAULT_MAX_MISPRONUNCIATION_RATIO = (
    media_config.DEFAULT_AUDIO_QA_MAX_MISPRONUNCIATION_RATIO
)
DEFAULT_MAX_SILENCE_RATIO = media_config.DEFAULT_AUDIO_QA_MAX_SILENCE_RATIO
DEFAULT_MAX_TRAILING_SILENCE_SECONDS = (
    media_config.DEFAULT_AUDIO_QA_MAX_TRAILING_SILENCE_SECONDS
)
DEFAULT_MIN_DURATION_RATIO = media_config.DEFAULT_AUDIO_QA_MIN_DURATION_RATIO
DEFAULT_MIN_WORD_CONFIDENCE = media_config.DEFAULT_AUDIO_QA_MIN_WORD_CONFIDENCE
DEFAULT_SILENT_RATIO = media_config.DEFAULT_AUDIO_QA_SILENT_RATIO
_MAX_CONFIDENCE = media_config.MAX_AUDIO_QA_CONFIDENCE

__all__ = [
    "DEFAULT_EXPECTED_WORDS_PER_SECOND",
    "DEFAULT_MAX_INTERNAL_SILENCE_SECONDS",
    "DEFAULT_MAX_LEADING_SILENCE_SECONDS",
    "DEFAULT_MAX_MISPRONUNCIATION_RATIO",
    "DEFAULT_MAX_SILENCE_RATIO",
    "DEFAULT_MAX_TRAILING_SILENCE_SECONDS",
    "DEFAULT_MIN_DURATION_RATIO",
    "DEFAULT_MIN_WORD_CONFIDENCE",
    "DEFAULT_SILENT_RATIO",
    "AudioQA",
    "AudioQAFinding",
    "AudioQAReport",
    "AudioQAResult",
    "AudioQualityChecker",
    "AudioSegment",
    "Severity",
    "SilenceSpan",
    "TimedWord",
    "audio_qa",
    "check_audio",
    "check_audio_quality",
    "lint_audio_quality",
    "qa_audio",
    "segments_from_timeline",
]

_EPSILON = 1e-9
_TOKEN = re.compile(r"[^\W_]+(?:['\u2019][^\W_]+)*", re.UNICODE)
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+", re.UNICODE)
_ELLIPSIS = re.compile(r"(?:\u2026|\.{3,})\s*$", re.UNICODE)
_BEAT_INDEX = re.compile(r"(?:beat[_-]?)?(\d+)$", re.IGNORECASE)


class Severity(StrEnum):
    """How strongly a finding should affect the pipeline."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


def _number(value: Any, name: str, *, non_negative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if non_negative and number < 0.0:
        raise ValueError(f"{name} must not be negative")
    return number


def _optional_number(
    value: Any,
    name: str,
    *,
    non_negative: bool = False,
) -> float | None:
    if value is None:
        return None
    return _number(value, name, non_negative=non_negative)


def _first_text(value: Mapping[str, Any], names: Sequence[str]) -> str:
    for name in names:
        raw = value.get(name)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return ""


def _first_text_attr(value: Any, names: Sequence[str]) -> str:
    for name in names:
        raw = getattr(value, name, None)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return ""


def _first_number(
    value: Mapping[str, Any],
    names: Sequence[str],
    default: float = 0.0,
) -> float:
    for name in names:
        raw = value.get(name)
        if raw is not None:
            return _number(raw, name)
    return default


def _optional_number_attr(value: Any, names: Sequence[str]) -> float | None:
    for name in names:
        raw = getattr(value, name, None)
        if raw is not None:
            return _number(raw, names[0])
    return None


def _normalise_token(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _word_tokens(text: str) -> list[str]:
    return [match.group(0) for match in _TOKEN.finditer(text)]


def _tokens(text: str) -> list[str]:
    return [_normalise_token(token) for token in _word_tokens(text)]


def _normalise_text(text: str) -> str:
    return " ".join(_tokens(text))


def _repeated_words(text: str) -> list[str]:
    words = _word_tokens(text)
    repeated: list[str] = []
    for previous, current in itertools.pairwise(words):
        if (
            _normalise_token(previous) == _normalise_token(current)
            and current not in repeated
        ):
            repeated.append(current)
    return repeated


def _repeated_sentences(text: str) -> list[str]:
    sentences = [
        sentence.strip() for sentence in _SENTENCE_BREAK.split(text) if sentence.strip()
    ]
    repeated: list[str] = []
    for previous, current in itertools.pairwise(sentences):
        if (
            _normalise_text(previous) == _normalise_text(current)
            and current not in repeated
        ):
            repeated.append(current)
    return repeated


@dataclass(frozen=True, slots=True)
class TimedWord:
    """One spoken word with absolute times on the timeline."""

    word: str
    start_seconds: float = 0.0
    end_seconds: float | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        word = self.word.strip()
        if not word:
            raise ValueError("word must be non-empty")
        start = _number(self.start_seconds, "start_seconds", non_negative=True)
        end = (
            start
            if self.end_seconds is None
            else _number(self.end_seconds, "end_seconds", non_negative=True)
        )
        if end < start:
            raise ValueError("end_seconds must not be before start_seconds")
        confidence = self.confidence
        if confidence is not None:
            confidence = _number(confidence, "confidence", non_negative=True)
            if confidence > _MAX_CONFIDENCE:
                raise ValueError("confidence must be between 0 and 1")
        object.__setattr__(self, "word", word)
        object.__setattr__(self, "start_seconds", start)
        object.__setattr__(self, "end_seconds", end)
        object.__setattr__(self, "confidence", confidence)

    @property
    def text(self) -> str:
        """Alias for :attr:`word`."""
        return self.word

    @property
    def duration_seconds(self) -> float:
        """The length of the word in seconds."""
        return (self.end_seconds or self.start_seconds) - self.start_seconds

    @classmethod
    def from_value(cls, value: Any) -> TimedWord | None:
        """Coerce a word model or mapping, returning ``None`` when empty."""
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            word = _first_text(value, ("word", "text", "value"))
            start = _first_number(value, ("start_seconds", "start", "start_time"))
            end = _optional_number(
                value.get("end_seconds", value.get("end", value.get("end_time"))),
                "end_seconds",
                non_negative=True,
            )
            confidence = _optional_number(
                value.get("confidence"),
                "confidence",
                non_negative=True,
            )
        else:
            word = _first_text_attr(value, ("word", "text", "value"))
            start_value = _optional_number_attr(
                value,
                ("start_seconds", "start", "start_time"),
            )
            start = 0.0 if start_value is None else start_value
            end = _optional_number_attr(
                value,
                ("end_seconds", "end", "end_time"),
            )
            confidence = _optional_number_attr(value, ("confidence",))
        if not word:
            return None
        return cls(
            word=word,
            start_seconds=start,
            end_seconds=end,
            confidence=confidence,
        )


@dataclass(frozen=True, slots=True)
class SilenceSpan:
    """A silent interval in absolute timeline seconds."""

    start_seconds: float
    end_seconds: float

    def __post_init__(self) -> None:
        start = _number(self.start_seconds, "start_seconds", non_negative=True)
        end = _number(self.end_seconds, "end_seconds", non_negative=True)
        if end < start:
            raise ValueError("silence end_seconds must not be before start_seconds")
        object.__setattr__(self, "start_seconds", start)
        object.__setattr__(self, "end_seconds", end)

    @property
    def duration_seconds(self) -> float:
        """The length of the silence in seconds."""
        return self.end_seconds - self.start_seconds

    @classmethod
    def from_value(cls, value: Any) -> SilenceSpan:
        """Coerce a pair, mapping or object into a silence span."""
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            mapped_start = _first_number(
                value,
                ("start_seconds", "start", "start_time", "from_seconds"),
            )
            mapped_end = _first_number(
                value,
                ("end_seconds", "end", "end_time", "to_seconds"),
            )
            return cls(mapped_start, mapped_end)
        if isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        ):
            items = list(value)
            if len(items) != 2:
                raise ValueError("a silence pair needs a start and an end")
            return cls(
                _number(items[0], "start_seconds"),
                _number(items[1], "end_seconds"),
            )
        span_start = _optional_number_attr(
            value,
            ("start_seconds", "start", "start_time"),
        )
        span_end = _optional_number_attr(
            value,
            ("end_seconds", "end", "end_time"),
        )
        if span_start is None or span_end is None:
            raise TypeError("silence span must have a start and an end")
        return cls(span_start, span_end)


def _coerce_words(value: Any) -> tuple[TimedWord, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Iterable):
        raise TypeError("words must be an iterable of word values")
    words = [word for item in value if (word := TimedWord.from_value(item)) is not None]
    return tuple(sorted(words, key=lambda item: item.start_seconds))


def _coerce_silences(value: Any) -> tuple[SilenceSpan, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Iterable):
        raise TypeError("silences must be an iterable of spans")
    spans = [SilenceSpan.from_value(item) for item in value]
    return tuple(sorted(spans, key=lambda item: item.start_seconds))


@dataclass(frozen=True, slots=True)
class AudioSegment:
    """One synthesized segment and everything needed to check it.

    Times are absolute on the timeline. When a segment is checked on its own,
    ``start_seconds`` defaults to zero and word and silence times are relative
    to that start. ``text`` is the expected script; ``transcript`` is what a
    speech recognizer heard. Word timings may be used when no transcript is
    available.
    """

    id: str = ""
    text: str = ""
    language: str = ""
    beat: str = ""
    speaker: str = ""
    voice: str = ""
    start_seconds: float = 0.0
    end_seconds: float | None = None
    duration_seconds: float | None = None
    words: tuple[TimedWord, ...] = ()
    transcript: str = ""
    transcript_words: tuple[TimedWord, ...] = ()
    silences: tuple[SilenceSpan, ...] = ()

    def __post_init__(self) -> None:
        start = _number(self.start_seconds, "start_seconds", non_negative=True)
        end = _optional_number(
            self.end_seconds,
            "end_seconds",
            non_negative=True,
        )
        duration = _optional_number(
            self.duration_seconds,
            "duration_seconds",
            non_negative=True,
        )
        words = _coerce_words(self.words)
        transcript_words = _coerce_words(self.transcript_words)
        silences = _coerce_silences(self.silences)
        if end is None and duration is not None:
            end = start + duration
        if end is not None and end < start:
            raise ValueError("end_seconds must not be before start_seconds")
        if duration is None and end is not None:
            duration = end - start
        if end is None:
            timed = [*words, *transcript_words]
            if timed:
                end = max((word.end_seconds or word.start_seconds) for word in timed)
                duration = max(0.0, end - start)
        object.__setattr__(self, "start_seconds", start)
        object.__setattr__(self, "end_seconds", end)
        object.__setattr__(self, "duration_seconds", duration)
        object.__setattr__(self, "words", words)
        object.__setattr__(self, "transcript_words", transcript_words)
        object.__setattr__(self, "silences", silences)

    @property
    def label(self) -> str:
        """A stable name used in findings."""
        return self.id or self.beat or self.language or "audio"

    @classmethod
    def from_value(cls, value: Any) -> AudioSegment:
        """Coerce a mapping, timeline voice or existing segment."""
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(
                id=_first_text(value, ("id", "voice_id", "name")),
                text=_first_text(
                    value,
                    ("text", "expected_text", "script", "narration", "line"),
                ),
                language=_first_text(value, ("language", "lang", "locale")),
                beat=_first_text(value, ("beat", "beat_id", "segment")),
                speaker=_first_text(value, ("speaker", "who", "character")),
                voice=_first_text(value, ("voice", "voice_id", "voice_name")),
                start_seconds=_first_number(
                    value,
                    ("start_seconds", "start", "at", "offset_seconds"),
                ),
                end_seconds=_optional_number(
                    value.get("end_seconds", value.get("end", value.get("stop"))),
                    "end_seconds",
                    non_negative=True,
                ),
                duration_seconds=_optional_number(
                    value.get(
                        "duration_seconds",
                        value.get("duration", value.get("length_seconds")),
                    ),
                    "duration_seconds",
                    non_negative=True,
                ),
                words=_coerce_words(
                    value.get(
                        "words",
                        value.get("word_timings", value.get("timed_words")),
                    )
                ),
                transcript=_first_text(
                    value,
                    ("transcript", "transcription", "heard_text", "spoken_text"),
                ),
                transcript_words=_coerce_words(
                    value.get(
                        "transcript_words",
                        value.get("transcription_words", value.get("heard_words")),
                    )
                ),
                silences=_coerce_silences(
                    value.get(
                        "silences",
                        value.get("silence_spans", value.get("gaps")),
                    )
                ),
            )
        return cls(
            id=_first_text_attr(value, ("id", "voice_id", "name")),
            text=_first_text_attr(
                value,
                ("text", "expected_text", "narration", "line"),
            ),
            language=_first_text_attr(value, ("language", "lang", "locale")),
            beat=_first_text_attr(value, ("beat", "beat_id", "segment")),
            speaker=_first_text_attr(value, ("speaker", "who", "character")),
            voice=_first_text_attr(value, ("voice", "voice_id", "voice_name")),
            start_seconds=(
                _optional_number_attr(value, ("start_seconds", "start", "at")) or 0.0
            ),
            end_seconds=_optional_number_attr(value, ("end_seconds", "end", "stop")),
            duration_seconds=_optional_number_attr(
                value,
                ("duration_seconds", "duration", "length_seconds"),
            ),
            words=_coerce_words(
                getattr(value, "words", None) or getattr(value, "word_timings", None)
            ),
            transcript=_first_text_attr(
                value,
                ("transcript", "transcription", "heard_text"),
            ),
            transcript_words=_coerce_words(
                getattr(value, "transcript_words", None)
                or getattr(value, "heard_words", None)
            ),
            silences=_coerce_silences(
                getattr(value, "silences", None)
                or getattr(value, "silence_spans", None)
            ),
        )


type SegmentInput = AudioSegment | Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class AudioQAFinding:
    """One problem found in a voice segment."""

    code: str
    severity: Severity
    message: str
    language: str = ""
    beat: str = ""
    voice: str = ""
    start_seconds: float | None = None
    end_seconds: float | None = None
    expected: str = ""
    actual: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.severity, Severity):
            object.__setattr__(self, "severity", Severity(self.severity))

    def to_dict(self) -> dict[str, Any]:
        """Return the finding as a JSON-compatible mapping."""
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "language": self.language,
            "beat": self.beat,
            "voice": self.voice,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "expected": self.expected,
            "actual": self.actual,
        }


class AudioQAReport(list[AudioQAFinding]):
    """Every finding from one pass, with the checked segment count attached."""

    def __init__(
        self,
        iterable: Iterable[AudioQAFinding] = (),
        *,
        segments: int = 0,
        languages: Sequence[str] = (),
    ) -> None:
        super().__init__(iterable)
        self.segments = segments
        self.languages = tuple(dict.fromkeys(languages))

    @property
    def findings(self) -> list[AudioQAFinding]:
        """The findings as a plain list."""
        return list(self)

    @property
    def ok(self) -> bool:
        """Whether the pass found nothing at all."""
        return not self

    @property
    def passed(self) -> bool:
        """Whether the pass found no errors."""
        return not self.errors

    @property
    def errors(self) -> list[AudioQAFinding]:
        """Findings severe enough to fail the stage."""
        return [finding for finding in self if finding.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[AudioQAFinding]:
        """Advisory findings."""
        return [finding for finding in self if finding.severity is Severity.WARNING]

    @property
    def infos(self) -> list[AudioQAFinding]:
        """Informational findings."""
        return [finding for finding in self if finding.severity is Severity.INFO]

    def by_code(self, code: str) -> list[AudioQAFinding]:
        """Return the findings with *code*."""
        return [finding for finding in self if finding.code == code]

    @property
    def silence(self) -> list[AudioQAFinding]:
        """The silence findings."""
        return self.by_code("silence")

    @property
    def duplication(self) -> list[AudioQAFinding]:
        """The duplication findings."""
        return self.by_code("duplication")

    @property
    def truncation(self) -> list[AudioQAFinding]:
        """The truncation findings."""
        return self.by_code("truncation")

    @property
    def mispronunciation(self) -> list[AudioQAFinding]:
        """The mispronunciation findings."""
        return self.by_code("mispronunciation")

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a JSON-compatible mapping."""
        return {
            "ok": self.ok,
            "passed": self.passed,
            "segments": self.segments,
            "languages": list(self.languages),
            "findings": [finding.to_dict() for finding in self],
        }


AudioQAResult = AudioQAReport
"""Alias for :class:`AudioQAReport`."""


def _ratio(value: Any, name: str) -> float:
    number = _number(value, name, non_negative=True)
    if number > 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return number


def _positive(value: Any, name: str) -> float:
    number = _number(value, name, non_negative=True)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _pronunciation_map(
    value: Mapping[str, Any] | None,
) -> dict[str, frozenset[str]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("pronunciations must be a mapping")
    result: dict[str, frozenset[str]] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str) or not raw_key.strip():
            raise TypeError("pronunciation keys must be non-empty strings")
        accepted: list[str] = []
        if isinstance(raw_value, str):
            accepted.append(raw_value)
        elif isinstance(raw_value, Sequence) and not isinstance(
            raw_value,
            (str, bytes, bytearray),
        ):
            accepted.extend(str(item) for item in raw_value)
        else:
            raise TypeError("pronunciation values must be a string or a sequence")
        tokens = frozenset(
            token for item in (raw_key, *accepted) for token in _tokens(item)
        )
        result[_normalise_text(raw_key)] = tokens
    return result


def _segment_list(value: Any) -> list[AudioSegment]:
    if isinstance(value, (AudioSegment, Mapping)):
        return [AudioSegment.from_value(value)]
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Iterable):
        raise TypeError("segments must be a segment or an iterable of segments")
    return [AudioSegment.from_value(item) for item in value]


class AudioQualityChecker:
    """Run the four audio quality checks with configurable thresholds."""

    def __init__(
        self,
        *,
        max_internal_silence_seconds: float = DEFAULT_MAX_INTERNAL_SILENCE_SECONDS,
        max_leading_silence_seconds: float = DEFAULT_MAX_LEADING_SILENCE_SECONDS,
        max_trailing_silence_seconds: float = DEFAULT_MAX_TRAILING_SILENCE_SECONDS,
        max_silence_ratio: float = DEFAULT_MAX_SILENCE_RATIO,
        silent_ratio: float = DEFAULT_SILENT_RATIO,
        min_word_confidence: float = DEFAULT_MIN_WORD_CONFIDENCE,
        max_mispronunciation_ratio: float = DEFAULT_MAX_MISPRONUNCIATION_RATIO,
        expected_words_per_second: float = DEFAULT_EXPECTED_WORDS_PER_SECOND,
        min_duration_ratio: float = DEFAULT_MIN_DURATION_RATIO,
        pronunciations: Mapping[str, Any] | None = None,
    ) -> None:
        self.max_internal_silence_seconds = _number(
            max_internal_silence_seconds,
            "max_internal_silence_seconds",
            non_negative=True,
        )
        self.max_leading_silence_seconds = _number(
            max_leading_silence_seconds,
            "max_leading_silence_seconds",
            non_negative=True,
        )
        self.max_trailing_silence_seconds = _number(
            max_trailing_silence_seconds,
            "max_trailing_silence_seconds",
            non_negative=True,
        )
        self.max_silence_ratio = _ratio(max_silence_ratio, "max_silence_ratio")
        self.silent_ratio = _ratio(silent_ratio, "silent_ratio")
        self.min_word_confidence = _ratio(
            min_word_confidence,
            "min_word_confidence",
        )
        self.max_mispronunciation_ratio = _ratio(
            max_mispronunciation_ratio,
            "max_mispronunciation_ratio",
        )
        self.expected_words_per_second = _positive(
            expected_words_per_second,
            "expected_words_per_second",
        )
        self.min_duration_ratio = _ratio(min_duration_ratio, "min_duration_ratio")
        self.pronunciations = _pronunciation_map(pronunciations)

    def _finding(
        self,
        segment: AudioSegment,
        code: str,
        severity: Severity,
        message: str,
        *,
        expected: str = "",
        actual: str = "",
    ) -> AudioQAFinding:
        return AudioQAFinding(
            code=code,
            severity=severity,
            message=message,
            language=segment.language,
            beat=segment.beat,
            voice=segment.label,
            start_seconds=segment.start_seconds,
            end_seconds=segment.end_seconds,
            expected=expected,
            actual=actual,
        )

    def _inferred_silences(self, segment: AudioSegment) -> list[SilenceSpan]:
        words = [word for word in segment.words if word.end_seconds is not None]
        if not words:
            return []
        start = segment.start_seconds
        end = segment.end_seconds
        spans: list[SilenceSpan] = []
        cursor = start
        for word in words:
            word_start = word.start_seconds
            word_end = word.end_seconds or word_start
            if word_start - cursor > _EPSILON:
                spans.append(SilenceSpan(cursor, word_start))
            cursor = max(cursor, word_end)
        if end is not None and end - cursor > _EPSILON:
            spans.append(SilenceSpan(cursor, end))
        return spans

    def _silence_findings(self, segment: AudioSegment) -> list[AudioQAFinding]:
        start = segment.start_seconds
        end = segment.end_seconds
        spans = list(segment.silences) or self._inferred_silences(segment)
        if end is not None:
            spans = [
                SilenceSpan(
                    max(start, span.start_seconds),
                    min(end, span.end_seconds),
                )
                for span in spans
                if span.end_seconds > start and span.start_seconds < end
            ]
            spans = [span for span in spans if span.duration_seconds > _EPSILON]
        if not spans:
            return []

        findings: list[AudioQAFinding] = []
        duration = segment.duration_seconds
        if duration:
            total = sum(span.duration_seconds for span in spans)
            ratio = min(1.0, total / duration)
            if ratio >= self.silent_ratio:
                findings.append(
                    self._finding(
                        segment,
                        "silence",
                        Severity.ERROR,
                        f"{segment.label} is almost entirely silent "
                        f"({ratio:.0%} of {duration:.2f}s)",
                    )
                )
            elif ratio > self.max_silence_ratio:
                findings.append(
                    self._finding(
                        segment,
                        "silence",
                        Severity.WARNING,
                        f"{segment.label} is {ratio:.0%} silent, above the "
                        f"{self.max_silence_ratio:.0%} limit",
                    )
                )

        leading = (
            spans[0].end_seconds - start
            if spans and spans[0].start_seconds <= start + _EPSILON
            else 0.0
        )
        if leading > self.max_leading_silence_seconds:
            findings.append(
                self._finding(
                    segment,
                    "silence",
                    Severity.WARNING,
                    f"{segment.label} starts with {leading:.2f}s of silence",
                )
            )
        if end is not None:
            trailing = (
                end - spans[-1].start_seconds
                if spans and spans[-1].end_seconds >= end - _EPSILON
                else 0.0
            )
            if trailing > self.max_trailing_silence_seconds:
                findings.append(
                    self._finding(
                        segment,
                        "silence",
                        Severity.WARNING,
                        f"{segment.label} ends with {trailing:.2f}s of silence",
                    )
                )
        for span in spans:
            touches_start = span.start_seconds <= start + _EPSILON
            touches_end = end is not None and span.end_seconds >= end - _EPSILON
            if touches_start or touches_end:
                continue
            if span.duration_seconds > self.max_internal_silence_seconds:
                findings.append(
                    self._finding(
                        segment,
                        "silence",
                        Severity.WARNING,
                        f"{segment.label} has a {span.duration_seconds:.2f}s "
                        "internal pause",
                    )
                )
        return findings

    def _duplication_findings(
        self,
        segments: Sequence[AudioSegment],
    ) -> list[AudioQAFinding]:
        findings: list[AudioQAFinding] = []
        seen: dict[str, AudioSegment] = {}
        for segment in segments:
            text = segment.text or segment.transcript
            for word in _repeated_words(text):
                findings.append(
                    self._finding(
                        segment,
                        "duplication",
                        Severity.WARNING,
                        f"{segment.label} repeats the word {word!r}",
                        actual=word,
                    )
                )
            for sentence in _repeated_sentences(text):
                findings.append(
                    self._finding(
                        segment,
                        "duplication",
                        Severity.WARNING,
                        f"{segment.label} repeats the sentence {sentence!r}",
                        actual=sentence,
                    )
                )
            key = _normalise_text(text)
            if not key:
                continue
            previous = seen.get(key)
            if previous is not None and (
                previous.language == segment.language
                or not previous.language
                or not segment.language
            ):
                findings.append(
                    self._finding(
                        segment,
                        "duplication",
                        Severity.WARNING,
                        f"{segment.label} repeats the text of {previous.label}",
                        expected=previous.text,
                        actual=text,
                    )
                )
            else:
                seen[key] = segment
        return findings

    def _truncation_findings(self, segment: AudioSegment) -> list[AudioQAFinding]:
        findings: list[AudioQAFinding] = []
        expected = _tokens(segment.text)
        if segment.text and _ELLIPSIS.search(segment.text):
            findings.append(
                self._finding(
                    segment,
                    "truncation",
                    Severity.WARNING,
                    f"{segment.label} ends with an ellipsis, which can hide a cut",
                    expected=segment.text,
                )
            )
        if not expected:
            return findings

        transcript_tokens = _tokens(segment.transcript)
        observed = transcript_tokens or [
            _normalise_token(word.word) for word in segment.words
        ]
        if transcript_tokens:
            matcher = SequenceMatcher(None, expected, transcript_tokens, autojunk=False)
            blocks = [block for block in matcher.get_matching_blocks() if block.size]
            if blocks:
                last = blocks[-1]
                observed_end = last.b + last.size
                expected_end = last.a + last.size
                missing = len(expected) - expected_end
                if observed_end == len(transcript_tokens) and missing > 0:
                    findings.append(
                        self._finding(
                            segment,
                            "truncation",
                            Severity.ERROR,
                            f"{segment.label} ends after {len(transcript_tokens)} "
                            f"of {len(expected)} expected words",
                            expected=" ".join(expected),
                            actual=" ".join(transcript_tokens),
                        )
                    )

        if segment.end_seconds is not None and segment.words:
            last_end = max(
                (word.end_seconds or word.start_seconds) for word in segment.words
            )
            if last_end > segment.end_seconds + _TIME_TOLERANCE_SECONDS:
                findings.append(
                    self._finding(
                        segment,
                        "truncation",
                        Severity.ERROR,
                        f"{segment.label} word timing runs to {last_end:.2f}s, "
                        f"past the {segment.end_seconds:.2f}s segment end",
                        actual=f"{last_end:.2f}s",
                    )
                )

        duration = segment.duration_seconds
        if duration and not observed:
            expected_duration = len(expected) / self.expected_words_per_second
            if duration < expected_duration * self.min_duration_ratio:
                findings.append(
                    self._finding(
                        segment,
                        "truncation",
                        Severity.ERROR,
                        f"{segment.label} lasts {duration:.2f}s but "
                        f"{len(expected)} words need about "
                        f"{expected_duration:.2f}s",
                        expected=f"{expected_duration:.2f}s",
                        actual=f"{duration:.2f}s",
                    )
                )
        return findings

    def _accepted(
        self,
        token: str,
        actual: str,
        pronunciations: Mapping[str, frozenset[str]],
    ) -> bool:
        accepted = pronunciations.get(token)
        return accepted is not None and actual in accepted

    def _mispronunciation_findings(
        self,
        segment: AudioSegment,
    ) -> list[AudioQAFinding]:
        findings: list[AudioQAFinding] = []
        expected = _tokens(segment.text)
        transcript = _tokens(segment.transcript)
        if transcript:
            if not expected:
                return findings
            matcher = SequenceMatcher(None, expected, transcript, autojunk=False)
            issues: list[tuple[str, str]] = []
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == "equal":
                    continue
                if tag == "replace":
                    overlap = min(i2 - i1, j2 - j1)
                    for offset in range(overlap):
                        expected_word = expected[i1 + offset]
                        actual_word = transcript[j1 + offset]
                        if self._accepted(
                            expected_word,
                            actual_word,
                            self.pronunciations,
                        ):
                            continue
                        issues.append((expected_word, actual_word))
                    for index in range(i1 + overlap, i2):
                        issues.append((expected[index], ""))
                    for index in range(j1 + overlap, j2):
                        issues.append(("", transcript[index]))
                elif tag == "delete":
                    if i2 == len(expected) and j2 == len(transcript):
                        continue
                    for index in range(i1, i2):
                        issues.append((expected[index], ""))
                elif tag == "insert":
                    for index in range(j1, j2):
                        issues.append(("", transcript[index]))
            ratio = len(issues) / max(1, len(expected))
            severity = (
                Severity.ERROR
                if ratio > self.max_mispronunciation_ratio
                else Severity.WARNING
            )
            for expected_word, actual_word in issues:
                if not actual_word:
                    message = (
                        f"{segment.label} drops the expected word {expected_word!r}"
                    )
                elif not expected_word:
                    message = (
                        f"{segment.label} adds the unexpected word {actual_word!r}"
                    )
                else:
                    message = (
                        f"{segment.label} may mispronounce {expected_word!r} "
                        f"as {actual_word!r}"
                    )
                findings.append(
                    self._finding(
                        segment,
                        "mispronunciation",
                        severity,
                        message,
                        expected=expected_word,
                        actual=actual_word,
                    )
                )
            return findings

        for word in segment.words:
            confidence = word.confidence
            if confidence is not None and confidence < self.min_word_confidence:
                findings.append(
                    self._finding(
                        segment,
                        "mispronunciation",
                        Severity.WARNING,
                        f"{segment.label} has low confidence "
                        f"({confidence:.2f}) on {word.word!r}",
                        expected=word.word,
                    )
                )
        return findings

    def check_silence(
        self,
        segments: SegmentInput | Sequence[SegmentInput],
    ) -> list[AudioQAFinding]:
        """Return the silence findings for one segment or a sequence."""
        return [
            finding
            for segment in _segment_list(segments)
            for finding in self._silence_findings(segment)
        ]

    def check_duplication(
        self,
        segments: SegmentInput | Sequence[SegmentInput],
    ) -> list[AudioQAFinding]:
        """Return the duplication findings for one segment or a sequence."""
        return self._duplication_findings(_segment_list(segments))

    def check_truncation(
        self,
        segments: SegmentInput | Sequence[SegmentInput],
    ) -> list[AudioQAFinding]:
        """Return the truncation findings for one segment or a sequence."""
        return [
            finding
            for segment in _segment_list(segments)
            for finding in self._truncation_findings(segment)
        ]

    def check_mispronunciation(
        self,
        segments: SegmentInput | Sequence[SegmentInput],
    ) -> list[AudioQAFinding]:
        """Return the mispronunciation findings for one segment or a sequence."""
        return [
            finding
            for segment in _segment_list(segments)
            for finding in self._mispronunciation_findings(segment)
        ]

    def check(
        self,
        segments: SegmentInput | Sequence[SegmentInput],
    ) -> AudioQAReport:
        """Run all four checks and return one ordered report."""
        values = _segment_list(segments)
        findings = [
            *self._silence_findings_all(values),
            *self._duplication_findings(values),
            *[
                finding
                for segment in values
                for finding in self._truncation_findings(segment)
            ],
            *[
                finding
                for segment in values
                for finding in self._mispronunciation_findings(segment)
            ],
        ]
        return AudioQAReport(
            findings,
            segments=len(values),
            languages=[segment.language for segment in values if segment.language],
        )

    def _silence_findings_all(
        self,
        segments: Sequence[AudioSegment],
    ) -> list[AudioQAFinding]:
        return [
            finding
            for segment in segments
            for finding in self._silence_findings(segment)
        ]

    __call__ = check


AudioQA = AudioQualityChecker
"""Alias for :class:`AudioQualityChecker`."""


def _timeline_value(value: Any) -> Timeline:
    if isinstance(value, Timeline):
        return value
    if isinstance(value, Mapping):
        return Timeline.model_validate(dict(value))
    if hasattr(value, "languages"):
        return Timeline.model_validate(
            {
                "languages": getattr(value, "languages", []),
                "beats": getattr(value, "beats", []),
            }
        )
    raise TypeError("timeline must be a Timeline, mapping or timeline-like object")


def _script_map(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "scripts"):
        value = value.scripts
    if isinstance(value, Mapping) and isinstance(value.get("scripts"), Mapping):
        value = value["scripts"]
    if not isinstance(value, Mapping):
        raise TypeError("scripts must be a mapping of language to script")
    return {
        str(key).replace("_", "-").casefold(): script for key, script in value.items()
    }


def _text_of(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        return _first_text(value, ("text", "narration", "line", "content"))
    return _first_text_attr(value, ("text", "narration", "line", "content"))


def _script_beat(script: Any, beat_id: str) -> Any:
    match = _BEAT_INDEX.search(beat_id)
    if match is None:
        return None
    index = int(match.group(1))
    beats = getattr(script, "beats", None)
    if beats is None and isinstance(script, Mapping):
        beats = script.get("beats", [])
    if not isinstance(beats, Sequence) or index >= len(beats):
        return None
    return beats[index]


def _beat_units(beat: Any) -> list[tuple[str, str]]:
    dialogue = getattr(beat, "dialogue", None)
    if dialogue is None and isinstance(beat, Mapping):
        dialogue = beat.get("dialogue", [])
    lines: list[tuple[str, str]] = []
    if isinstance(dialogue, Sequence):
        for line in dialogue:
            text = _first_text_attr(line, ("text", "line", "content"))
            if not text and isinstance(line, Mapping):
                text = _first_text(line, ("text", "line", "content"))
            speaker = _first_text_attr(line, ("speaker", "character", "role", "who"))
            if not speaker and isinstance(line, Mapping):
                speaker = _first_text(line, ("speaker", "character", "role", "who"))
            if text:
                lines.append((text, speaker or "narrator"))
    narration = _text_of(beat)
    if not narration:
        return lines
    if not lines:
        return [(narration, "narrator")]
    target = _normalise_text(narration)
    joined = _normalise_text(" ".join(text for text, _speaker in lines))
    if target == joined or any(_normalise_text(text) == target for text, _ in lines):
        return lines
    return [(narration, "narrator"), *lines]


def _dialogue_units(script: Any) -> list[tuple[str, str]]:
    lines = getattr(script, "dialogue_lines", None)
    if lines is None and isinstance(script, Mapping):
        lines = script.get("dialogue_lines", script.get("dialogue", []))
    units: list[tuple[str, str]] = []
    if isinstance(lines, Sequence):
        for line in lines:
            text = _first_text_attr(line, ("text", "line", "content"))
            if not text and isinstance(line, Mapping):
                text = _first_text(line, ("text", "line", "content"))
            speaker = _first_text_attr(line, ("speaker", "character", "role", "who"))
            if not speaker and isinstance(line, Mapping):
                speaker = _first_text(line, ("speaker", "character", "role", "who"))
            if text:
                units.append((text, speaker or "narrator"))
    return units


def _full_script_text(script: Any) -> str:
    hook = _first_text_attr(script, ("hook",))
    if not hook and isinstance(script, Mapping):
        hook = _first_text(script, ("hook",))
    outro = _first_text_attr(script, ("outro",))
    if not outro and isinstance(script, Mapping):
        outro = _first_text(script, ("outro",))
    beats = getattr(script, "beats", None)
    if beats is None and isinstance(script, Mapping):
        beats = script.get("beats", [])
    parts: list[str] = []
    if hook:
        parts.append(hook)
    if isinstance(beats, Sequence):
        for beat in beats:
            text = _text_of(beat)
            if text:
                parts.append(text)
    if outro:
        parts.append(outro)
    if not parts:
        parts.extend(text for text, _speaker in _dialogue_units(script))
    return "\n\n".join(parts)


def _expected_text(
    script: Any,
    voice: Any,
    occurrence: int,
    *,
    beat_text: str = "",
) -> str:
    if script is None:
        return beat_text
    beat_id = _first_text_attr(voice, ("beat", "beat_id", "segment"))
    if not beat_id and isinstance(voice, Mapping):
        beat_id = _first_text(voice, ("beat", "beat_id", "segment"))
    speaker = _first_text_attr(voice, ("speaker", "who", "character"))
    if not speaker and isinstance(voice, Mapping):
        speaker = _first_text(voice, ("speaker", "who", "character"))
    if beat_id:
        beat = _script_beat(script, beat_id)
        if beat is not None:
            units = _beat_units(beat)
            if speaker:
                matching = [
                    text
                    for text, unit_speaker in units
                    if unit_speaker.casefold() == speaker.casefold()
                ]
            else:
                matching = [text for text, _speaker in units]
            if matching:
                return matching[min(occurrence, len(matching) - 1)]
            return _text_of(beat) or beat_text
    voice_id = _first_text_attr(voice, ("id", "voice_id", "name"))
    if not voice_id and isinstance(voice, Mapping):
        voice_id = _first_text(voice, ("id", "voice_id", "name"))
    if voice_id.casefold().endswith("_script"):
        return _full_script_text(script)
    units = _dialogue_units(script)
    if speaker:
        matching = [
            text
            for text, unit_speaker in units
            if unit_speaker.casefold() == speaker.casefold()
        ]
    else:
        matching = [text for text, _speaker in units]
    if matching:
        return matching[min(occurrence, len(matching) - 1)]
    return _full_script_text(script)


def _lookup_extra(
    value: Mapping[Any, Any] | Sequence[Any] | None,
    segment: AudioSegment,
    index: int,
) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        keys: list[Any] = [segment.id, segment.beat, segment.language, index]
        keys.extend(
            [
                (segment.language, segment.beat),
                (segment.language, segment.id),
            ]
        )
        for key in keys:
            if key is not None and key != "" and key in value:
                return value[key]
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value[index] if index < len(value) else None
    return None


def _transcript_value(value: Any) -> tuple[str, tuple[TimedWord, ...]]:
    if isinstance(value, str):
        return value.strip(), ()
    if isinstance(value, Mapping):
        text = _first_text(value, ("text", "transcript", "transcription"))
        words = _coerce_words(
            value.get("words", value.get("all_words", value.get("word_timestamps")))
        )
        return text, words
    text = _first_text_attr(value, ("text", "transcript", "transcription"))
    words = _coerce_words(
        getattr(value, "all_words", None) or getattr(value, "words", None)
    )
    return text, words


def _silence_value(value: Any) -> tuple[SilenceSpan, ...]:
    if isinstance(value, Mapping):
        value = value.get(
            "silences",
            value.get("silence_spans", value.get("spans", value.get("gaps"))),
        )
    return _coerce_silences(value)


def _apply_extras(
    segments: Sequence[AudioSegment],
    transcripts: Mapping[Any, Any] | Sequence[Any] | None,
    silences: Mapping[Any, Any] | Sequence[Any] | None,
) -> list[AudioSegment]:
    result: list[AudioSegment] = []
    for index, segment in enumerate(segments):
        updates: dict[str, Any] = {}
        transcript = _lookup_extra(transcripts, segment, index)
        if transcript is not None:
            text, words = _transcript_value(transcript)
            updates["transcript"] = text
            updates["transcript_words"] = words
        silence = _lookup_extra(silences, segment, index)
        if silence is not None:
            updates["silences"] = _silence_value(silence)
        result.append(replace(segment, **updates) if updates else segment)
    return result


def segments_from_timeline(
    timeline: Any,
    *,
    scripts: Any = None,
    transcripts: Mapping[Any, Any] | Sequence[Any] | None = None,
    silences: Mapping[Any, Any] | Sequence[Any] | None = None,
) -> list[AudioSegment]:
    """Build checkable segments from a timeline and optional scripts."""
    value = _timeline_value(timeline)
    script_map = _script_map(scripts)
    beat_text = {beat.id: beat.text for beat in value.beats if beat.id and beat.text}
    occurrences: dict[tuple[str, str, str], int] = {}
    segments: list[AudioSegment] = []
    for language in value.languages:
        script = script_map.get(language.language.casefold())
        for voice in language.voices:
            if not voice.enabled:
                continue
            key = (language.language, voice.beat, voice.speaker.casefold())
            occurrence = occurrences.get(key, 0)
            occurrences[key] = occurrence + 1
            text = _expected_text(
                script,
                voice,
                occurrence,
                beat_text=beat_text.get(voice.beat, ""),
            )
            start = voice.start_seconds or 0.0
            end = voice.end_seconds
            if end is None and voice.duration_seconds is not None:
                end = start + voice.duration_seconds
            segment = AudioSegment(
                id=voice.id,
                text=text,
                language=language.language,
                beat=voice.beat,
                speaker=voice.speaker,
                voice=voice.voice,
                start_seconds=start,
                end_seconds=end,
                duration_seconds=voice.duration_seconds,
                words=_coerce_words(voice.words),
            )
            segments.append(segment)
    return _apply_extras(segments, transcripts, silences)


def check_audio_quality(
    segments: Any,
    *,
    scripts: Any = None,
    transcripts: Mapping[Any, Any] | Sequence[Any] | None = None,
    silences: Mapping[Any, Any] | Sequence[Any] | None = None,
    pronunciations: Mapping[str, Any] | None = None,
    **thresholds: Any,
) -> AudioQAReport:
    """Check audio quality from a timeline or a sequence of segments.

    A :class:`~aivia.compose.timeline.Timeline` (or its mapping form) is
    expanded with :func:`segments_from_timeline`; anything else is treated as a
    segment or an iterable of segments. ``transcripts`` and ``silences`` may be
    keyed by voice id, beat, language or index when the input is a plain
    sequence, and by the same keys when a timeline is given.
    """
    checker = AudioQualityChecker(pronunciations=pronunciations, **thresholds)
    if isinstance(segments, Timeline) or (
        isinstance(segments, Mapping) and "languages" in segments
    ):
        values = segments_from_timeline(
            segments,
            scripts=scripts,
            transcripts=transcripts,
            silences=silences,
        )
    else:
        values = _apply_extras(_segment_list(segments), transcripts, silences)
    return checker.check(values)


audio_qa = check_audio_quality
check_audio = check_audio_quality
lint_audio_quality = check_audio_quality
qa_audio = check_audio_quality
