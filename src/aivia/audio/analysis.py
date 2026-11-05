"""Report tempo, a silence map and loudness for any audio file.

The audio stages synthesize, select and mix, but callers still need a small,
serializable description of an arbitrary recording before deciding what to do
with it. The reference analyser needs a tempo and a silence map, the music
selector needs integrated loudness, and a quality pass needs to know where the
quiet stretches are. This module is that utility.

It does not run a second audio pipeline. The single ffmpeg and ffprobe
implementation lives in :mod:`aivia.media.audio_mix`; here it is wrapped in an
immutable report whose silence map is explicit and whose fields are aliases for
the names callers use. Every runner stays injectable, so the offline suite
never needs the real binaries.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from aivia.audio.qa import SilenceSpan
from aivia.core.config.media import (
    DEFAULT_AUDIO_ANALYSIS_MAX_BPM,
    DEFAULT_AUDIO_ANALYSIS_MIN_BPM,
    DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS,
    DEFAULT_MEDIA_FFPROBE_TIMEOUT_SECONDS,
    DEFAULT_MUSIC_SILENCE_MIN_SECONDS,
    DEFAULT_MUSIC_SILENCE_NOISE_DB,
)
from aivia.core.media_probe import FFprobeRunner
from aivia.media import audio_mix as _media
from aivia.providers.protocols.audio import AudioFormat

__all__ = [
    "DEFAULT_ANALYSIS_MAX_BPM",
    "DEFAULT_ANALYSIS_MIN_BPM",
    "DEFAULT_ANALYSIS_SILENCE_MIN_SECONDS",
    "DEFAULT_ANALYSIS_SILENCE_NOISE_DB",
    "AudioAnalysis",
    "AudioAnalysisReport",
    "AudioAnalyzer",
    "SilenceMap",
    "analyze_audio",
    "analyze_audio_file",
    "parse_audio_analysis",
]

DEFAULT_ANALYSIS_MIN_BPM = DEFAULT_AUDIO_ANALYSIS_MIN_BPM
"""The lowest tempo, in BPM, the estimator will report."""

DEFAULT_ANALYSIS_MAX_BPM = DEFAULT_AUDIO_ANALYSIS_MAX_BPM
"""The highest tempo, in BPM, the estimator will report."""

DEFAULT_ANALYSIS_SILENCE_NOISE_DB = DEFAULT_MUSIC_SILENCE_NOISE_DB
"""Audio below this level is treated as silence."""

DEFAULT_ANALYSIS_SILENCE_MIN_SECONDS = DEFAULT_MUSIC_SILENCE_MIN_SECONDS
"""A quiet stretch must last this long to enter the silence map."""


def _number(
    value: Any,
    *,
    label: str,
    non_negative: bool = False,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    if non_negative and number < 0.0:
        raise ValueError(f"{label} must not be negative")
    if positive and number <= 0.0:
        raise ValueError(f"{label} must be positive")
    return number


def _optional_number(
    value: Any,
    *,
    label: str,
    non_negative: bool = False,
    positive: bool = False,
) -> float | None:
    if value is None:
        return None
    return _number(
        value,
        label=label,
        non_negative=non_negative,
        positive=positive,
    )


def _ratio(value: Any, *, label: str) -> float:
    number = _number(value, label=label, non_negative=True)
    if number > 1.0:
        raise ValueError(f"{label} must be between 0 and 1")
    return number


@dataclass(frozen=True, slots=True)
class SilenceMap:
    """Every silent interval in a recording, in absolute seconds.

    The map keeps the measured duration so it can report the silent share
    without the caller remembering which file the intervals came from. Spans
    are sorted, clamped to the duration and exposed both as spans and as plain
    ``(start, end)`` pairs.
    """

    spans: tuple[SilenceSpan, ...] = ()
    duration_seconds: float = 0.0

    def __post_init__(self) -> None:
        duration = _number(
            self.duration_seconds,
            label="duration_seconds",
            non_negative=True,
        )
        spans: list[SilenceSpan] = []
        for value in self.spans:
            span = SilenceSpan.from_value(value)
            start = min(span.start_seconds, duration)
            end = min(span.end_seconds, duration)
            if end > start:
                spans.append(SilenceSpan(start, end))
        spans.sort(key=lambda item: item.start_seconds)
        object.__setattr__(self, "duration_seconds", duration)
        object.__setattr__(self, "spans", tuple(spans))

    @classmethod
    def from_intervals(
        cls,
        intervals: Sequence[Sequence[float] | SilenceSpan],
        *,
        duration_seconds: float = 0.0,
    ) -> SilenceMap:
        """Build a map from ``(start, end)`` pairs or silence spans."""
        return cls(
            spans=tuple(SilenceSpan.from_value(item) for item in intervals),
            duration_seconds=duration_seconds,
        )

    @property
    def intervals(self) -> tuple[tuple[float, float], ...]:
        """The spans as plain ``(start, end)`` pairs."""
        return tuple(
            (span.start_seconds, span.end_seconds) for span in self.spans
        )

    @property
    def total_seconds(self) -> float:
        """The summed length of every silent interval."""
        return math.fsum(span.duration_seconds for span in self.spans)

    @property
    def seconds(self) -> float:
        """Alias for :attr:`total_seconds`."""
        return self.total_seconds

    @property
    def ratio(self) -> float:
        """The silent share of the recording, between 0 and 1."""
        if not self.duration_seconds:
            return 0.0
        return min(1.0, self.total_seconds / self.duration_seconds)

    @property
    def is_empty(self) -> bool:
        """Whether no silence was measured."""
        return not self.spans

    @property
    def has_silence(self) -> bool:
        """Alias for ``not is_empty``."""
        return bool(self.spans)

    def contains(self, seconds: float) -> bool:
        """Whether *seconds* falls inside a silent interval."""
        moment = _number(seconds, label="seconds", non_negative=True)
        return any(
            span.start_seconds <= moment < span.end_seconds for span in self.spans
        )

    def to_list(self) -> list[list[float]]:
        """Return a JSON-serializable list of interval pairs."""
        return [[start, end] for start, end in self.intervals]

    def to_dict(self) -> dict[str, Any]:
        """Return the map as a JSON-compatible mapping."""
        return {
            "duration_seconds": self.duration_seconds,
            "intervals": self.to_list(),
            "total_seconds": self.total_seconds,
            "ratio": self.ratio,
        }

    def __iter__(self) -> Iterator[SilenceSpan]:
        return iter(self.spans)

    def __len__(self) -> int:
        return len(self.spans)

    def __bool__(self) -> bool:
        return bool(self.spans)


@dataclass(frozen=True, slots=True)
class AudioAnalysis:
    """Measured tempo, silence map and loudness of one audio file."""

    duration_seconds: float = 0.0
    tempo_bpm: float | None = None
    silence_seconds: float = 0.0
    silence_ratio: float = 0.0
    silence_intervals: tuple[tuple[float, float], ...] = ()
    integrated_loudness_lufs: float | None = None
    true_peak_db: float | None = None
    peak_db: float | None = None
    mean_volume_db: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        duration = _number(
            self.duration_seconds,
            label="duration_seconds",
            non_negative=True,
        )
        silence_seconds = _number(
            self.silence_seconds,
            label="silence_seconds",
            non_negative=True,
        )
        silence_ratio = _ratio(self.silence_ratio, label="silence_ratio")
        intervals = tuple(
            (
                SilenceSpan.from_value(item).start_seconds,
                SilenceSpan.from_value(item).end_seconds,
            )
            for item in self.silence_intervals
        )
        if not silence_seconds and intervals:
            silence_seconds = math.fsum(end - start for start, end in intervals)
        if not silence_ratio and silence_seconds and duration:
            silence_ratio = min(1.0, silence_seconds / duration)
        object.__setattr__(self, "duration_seconds", duration)
        object.__setattr__(
            self,
            "tempo_bpm",
            _optional_number(self.tempo_bpm, label="tempo_bpm", positive=True),
        )
        object.__setattr__(
            self,
            "silence_seconds",
            min(silence_seconds, duration) if duration else silence_seconds,
        )
        object.__setattr__(self, "silence_ratio", silence_ratio)
        object.__setattr__(self, "silence_intervals", intervals)
        object.__setattr__(
            self,
            "integrated_loudness_lufs",
            _optional_number(
                self.integrated_loudness_lufs,
                label="integrated_loudness_lufs",
            ),
        )
        object.__setattr__(
            self,
            "true_peak_db",
            _optional_number(self.true_peak_db, label="true_peak_db"),
        )
        object.__setattr__(
            self,
            "peak_db",
            _optional_number(self.peak_db, label="peak_db"),
        )
        object.__setattr__(
            self,
            "mean_volume_db",
            _optional_number(self.mean_volume_db, label="mean_volume_db"),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def duration(self) -> float:
        """Alias for :attr:`duration_seconds`."""
        return self.duration_seconds

    @property
    def tempo(self) -> float | None:
        """Alias for :attr:`tempo_bpm`."""
        return self.tempo_bpm

    @property
    def bpm(self) -> float | None:
        """Alias for :attr:`tempo_bpm`."""
        return self.tempo_bpm

    @property
    def silence(self) -> float:
        """Alias for :attr:`silence_seconds`."""
        return self.silence_seconds

    @property
    def silence_map(self) -> SilenceMap:
        """The measured silent intervals as a :class:`SilenceMap`."""
        return SilenceMap(
            spans=tuple(
                SilenceSpan(start, end) for start, end in self.silence_intervals
            ),
            duration_seconds=self.duration_seconds,
        )

    @property
    def silences(self) -> tuple[SilenceSpan, ...]:
        """Alias for the spans inside :attr:`silence_map`."""
        return self.silence_map.spans

    @property
    def silence_spans(self) -> tuple[SilenceSpan, ...]:
        """Alias for :attr:`silences`."""
        return self.silences

    @property
    def loudness_lufs(self) -> float | None:
        """Alias for :attr:`integrated_loudness_lufs`."""
        return self.integrated_loudness_lufs

    @property
    def lufs(self) -> float | None:
        """Alias for :attr:`integrated_loudness_lufs`."""
        return self.integrated_loudness_lufs

    @property
    def loudness(self) -> float | None:
        """Alias for :attr:`integrated_loudness_lufs`."""
        return self.integrated_loudness_lufs

    @property
    def peak(self) -> float | None:
        """The highest true peak or sample peak in dBFS."""
        return self.true_peak_db if self.true_peak_db is not None else self.peak_db

    @property
    def is_silent(self) -> bool:
        """Whether any silence was measured."""
        return bool(self.silence_seconds)

    @property
    def has_silence(self) -> bool:
        """Alias for :attr:`is_silent`."""
        return self.is_silent

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation without raw audio."""
        return {
            "duration_seconds": self.duration_seconds,
            "tempo_bpm": self.tempo_bpm,
            "silence_seconds": self.silence_seconds,
            "silence_ratio": self.silence_ratio,
            "silence_intervals": [list(item) for item in self.silence_intervals],
            "silence_map": self.silence_map.to_dict(),
            "integrated_loudness_lufs": self.integrated_loudness_lufs,
            "true_peak_db": self.true_peak_db,
            "peak_db": self.peak_db,
            "mean_volume_db": self.mean_volume_db,
            "metadata": dict(self.metadata),
        }


AudioAnalysisReport = AudioAnalysis
"""Alias for :class:`AudioAnalysis` used when a report is passed around."""


def _from_media(analysis: _media.AudioAnalysis) -> AudioAnalysis:
    return AudioAnalysis(
        duration_seconds=analysis.duration_seconds,
        tempo_bpm=analysis.tempo_bpm,
        silence_seconds=analysis.silence_seconds,
        silence_ratio=analysis.silence_ratio,
        silence_intervals=analysis.silence_intervals,
        integrated_loudness_lufs=analysis.integrated_loudness_lufs,
        true_peak_db=analysis.true_peak_db,
        peak_db=analysis.peak_db,
        mean_volume_db=analysis.mean_volume_db,
        metadata=dict(analysis.metadata),
    )


def parse_audio_analysis(
    text: str,
    duration_seconds: float,
    *,
    tempo_bpm: float | None = None,
    min_bpm: float = DEFAULT_ANALYSIS_MIN_BPM,
    max_bpm: float = DEFAULT_ANALYSIS_MAX_BPM,
) -> AudioAnalysis:
    """Parse ffmpeg silence, EBU R128, astats and tempo output."""
    return _from_media(
        _media.parse_audio_analysis(
            text,
            duration_seconds,
            tempo_bpm=tempo_bpm,
            min_bpm=min_bpm,
            max_bpm=max_bpm,
        )
    )


def analyze_audio(
    source: _media.AudioSource,
    *,
    duration_seconds: float | None = None,
    format: str | AudioFormat | None = None,
    tempo_bpm: float | None = None,
    min_bpm: float = DEFAULT_ANALYSIS_MIN_BPM,
    max_bpm: float = DEFAULT_ANALYSIS_MAX_BPM,
    silence_noise_db: float = DEFAULT_ANALYSIS_SILENCE_NOISE_DB,
    silence_min_seconds: float = DEFAULT_ANALYSIS_SILENCE_MIN_SECONDS,
    detect_tempo: bool = True,
    decode_tempo: bool = True,
    tempo_sample_rate: int = 8_000,
    ffmpeg_runner: _media.FFmpegRunner | None = None,
    ffprobe_runner: FFprobeRunner | None = None,
    ffmpeg_executable: str = "ffmpeg",
    ffprobe_executable: str = "ffprobe",
    timeout_seconds: float = DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS,
    ffprobe_timeout_seconds: float = DEFAULT_MEDIA_FFPROBE_TIMEOUT_SECONDS,
) -> AudioAnalysis:
    """Measure duration, tempo, silence and loudness of *source*.

    ``source`` may be a path or encoded bytes. An explicit ``tempo_bpm`` skips
    tempo detection. Otherwise a tempo reported by ffmpeg is used, then the
    loudness envelope is scored, and finally a short mono PCM decode is
    analysed. A failed optional decode leaves the tempo unset rather than
    failing the whole analysis.
    """
    return _from_media(
        _media.analyze_audio(
            source,
            duration_seconds=duration_seconds,
            format=format,
            tempo_bpm=tempo_bpm,
            min_bpm=min_bpm,
            max_bpm=max_bpm,
            silence_noise_db=silence_noise_db,
            silence_min_seconds=silence_min_seconds,
            detect_tempo=detect_tempo,
            decode_tempo=decode_tempo,
            tempo_sample_rate=tempo_sample_rate,
            ffmpeg_runner=ffmpeg_runner,
            ffprobe_runner=ffprobe_runner,
            ffmpeg_executable=ffmpeg_executable,
            ffprobe_executable=ffprobe_executable,
            timeout_seconds=timeout_seconds,
            ffprobe_timeout_seconds=ffprobe_timeout_seconds,
        )
    )


analyze_audio_file = analyze_audio
"""Alias for :func:`analyze_audio` for callers that pass a file path."""


@dataclass(frozen=True, slots=True)
class AudioAnalyzer:
    """An analysis facade with the silence and tempo defaults bound."""

    silence_noise_db: float = DEFAULT_ANALYSIS_SILENCE_NOISE_DB
    silence_min_seconds: float = DEFAULT_ANALYSIS_SILENCE_MIN_SECONDS
    min_bpm: float = DEFAULT_ANALYSIS_MIN_BPM
    max_bpm: float = DEFAULT_ANALYSIS_MAX_BPM
    detect_tempo: bool = True
    decode_tempo: bool = True

    def __post_init__(self) -> None:
        noise = _number(self.silence_noise_db, label="silence_noise_db")
        minimum = _number(
            self.silence_min_seconds,
            label="silence_min_seconds",
            non_negative=True,
        )
        low = _number(self.min_bpm, label="min_bpm", positive=True)
        high = _number(self.max_bpm, label="max_bpm", positive=True)
        if high <= low:
            raise ValueError("max_bpm must be greater than min_bpm")
        object.__setattr__(self, "silence_noise_db", noise)
        object.__setattr__(self, "silence_min_seconds", minimum)
        object.__setattr__(self, "min_bpm", low)
        object.__setattr__(self, "max_bpm", high)

    def analyze(
        self,
        source: _media.AudioSource,
        **overrides: Any,
    ) -> AudioAnalysis:
        """Analyse *source* using this instance's configured defaults."""
        options: dict[str, Any] = {
            "silence_noise_db": self.silence_noise_db,
            "silence_min_seconds": self.silence_min_seconds,
            "min_bpm": self.min_bpm,
            "max_bpm": self.max_bpm,
            "detect_tempo": self.detect_tempo,
            "decode_tempo": self.decode_tempo,
        }
        options.update(overrides)
        return analyze_audio(source, **options)
