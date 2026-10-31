"""Generate several music candidates and keep the best usable one.

A music provider can return a track that is the wrong length, mostly silent,
clipped or far too quiet. Asking for several candidates and rejecting the bad
ones deterministically is much cheaper than another paid generation, so this
module analyses every candidate before any later stage spends time on it.

The analysis runs ffmpeg and ffprobe with argv lists and no shell. Both runners
are injectable, which keeps the offline suite free of the real binaries. The
filter itself is pure code: duration bounds, silence ratio, clipping ratio and
integrated loudness are compared with :class:`MusicCandidateConfig`. The chosen
index is recorded on :class:`MusicCandidateSelection` so a run can reproduce
which candidate was used.

Generation accepts any object with an async ``generate`` method, including a
provider in a unit test and the media service in normal operation. The helper
does not own caching, budget or licensing; those stay in the music service.
"""

from __future__ import annotations

import asyncio
import inspect
import math
import re
import subprocess
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from aivia.core.config.media import (
    DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS,
    DEFAULT_MEDIA_FFPROBE_TIMEOUT_SECONDS,
    DEFAULT_MUSIC_CLIPPING_PEAK_DB,
    DEFAULT_MUSIC_DURATION_TOLERANCE_SECONDS,
    DEFAULT_MUSIC_MAX_CLIPPING_RATIO,
    DEFAULT_MUSIC_MAX_LOUDNESS_LUFS,
    DEFAULT_MUSIC_MAX_SILENCE_RATIO,
    DEFAULT_MUSIC_MIN_DURATION_SECONDS,
    DEFAULT_MUSIC_MIN_LOUDNESS_LUFS,
    DEFAULT_MUSIC_SILENCE_MIN_SECONDS,
    DEFAULT_MUSIC_SILENCE_NOISE_DB,
)
from aivia.core.errors import ValidationError
from aivia.core.media_probe import FFprobeRunner, sniff_media
from aivia.media.audio_utils import measure_audio_duration
from aivia.providers.protocols.audio import AudioFormat
from aivia.providers.protocols.music import MusicRequest, MusicResponse, MusicTrack

__all__ = [
    "DEFAULT_MUSIC_CLIPPING_PEAK_DB",
    "DEFAULT_MUSIC_DURATION_TOLERANCE_SECONDS",
    "DEFAULT_MUSIC_MAX_CLIPPING_RATIO",
    "DEFAULT_MUSIC_MAX_LOUDNESS_LUFS",
    "DEFAULT_MUSIC_MAX_SILENCE_RATIO",
    "DEFAULT_MUSIC_MIN_DURATION_SECONDS",
    "DEFAULT_MUSIC_MIN_LOUDNESS_LUFS",
    "DEFAULT_MUSIC_SILENCE_MIN_SECONDS",
    "DEFAULT_MUSIC_SILENCE_NOISE_DB",
    "MusicCandidate",
    "MusicCandidateAnalysis",
    "MusicCandidateConfig",
    "MusicCandidateError",
    "MusicCandidateFilter",
    "MusicCandidateGenerator",
    "MusicCandidatePrefilter",
    "MusicCandidateResult",
    "MusicCandidateSelection",
    "MusicCandidateSelector",
    "MusicSource",
    "analyze_music_candidate",
    "filter_music_candidates",
    "generate_music_candidates",
    "generate_music_candidates_sync",
    "music_candidate_config",
    "parse_music_analysis",
    "prefilter_music_candidates",
    "score_music_candidate",
    "select_music_candidate",
    "select_music_candidates",
    "select_music_track",
]

type MusicSource = str | Path | bytes
"""A music file path or the raw bytes of an encoded track."""

FFmpegRunner = Callable[[Sequence[str]], Any]
"""A callable that runs an ffmpeg argv and returns its output."""

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_SILENCE_START = re.compile(rf"silence_start:\s*({_NUMBER})")
_SILENCE_END = re.compile(
    rf"silence_end:\s*({_NUMBER})(?:\s*\|\s*silence_duration:\s*({_NUMBER}))?"
)
_LOUDNESS = re.compile(rf"\bI:\s*({_NUMBER})\s*LUFS")
_TRUE_PEAK = re.compile(rf"\bPeak:\s*({_NUMBER})\s*dBFS")
_PEAK_LEVEL = re.compile(rf"Peak level dB:\s*({_NUMBER})")
_MAX_VOLUME = re.compile(rf"max_volume:\s*({_NUMBER})\s*dB")
_MEAN_VOLUME = re.compile(rf"mean_volume:\s*({_NUMBER})\s*dB")
_PEAK_COUNT = re.compile(r"Peak count:\s*(\d+)")
_SAMPLE_COUNT = re.compile(r"Number of samples:\s*(\d+)")


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


class MusicCandidateConfig(BaseModel):
    """Thresholds applied before a generated music candidate is accepted.

    A target duration overrides the explicit minimum and maximum bounds and
    accepts candidates within ``duration_tolerance_seconds`` of the request.
    Loudness is integrated LUFS from ffmpeg's EBU R128 meter. ``prefer``
    controls whether the first accepted candidate wins or the highest scoring
    one does; the chosen index is recorded either way.
    """

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
        populate_by_name=True,
    )

    min_duration_seconds: float = Field(
        default=DEFAULT_MUSIC_MIN_DURATION_SECONDS,
        ge=0.0,
        validation_alias=AliasChoices(
            "min_duration_seconds",
            "min_duration",
            "minimum_duration_seconds",
            "duration_min_seconds",
        ),
    )
    max_duration_seconds: float | None = Field(
        default=None,
        ge=0.0,
        validation_alias=AliasChoices(
            "max_duration_seconds",
            "max_duration",
            "maximum_duration_seconds",
            "duration_max_seconds",
        ),
    )
    target_duration_seconds: float | None = Field(
        default=None,
        gt=0.0,
        validation_alias=AliasChoices(
            "target_duration_seconds",
            "target_duration",
            "requested_duration_seconds",
            "duration_seconds",
        ),
    )
    duration_tolerance_seconds: float = Field(
        default=DEFAULT_MUSIC_DURATION_TOLERANCE_SECONDS,
        ge=0.0,
        validation_alias=AliasChoices(
            "duration_tolerance_seconds",
            "duration_tolerance",
            "tolerance_seconds",
        ),
    )
    max_silence_ratio: float = Field(
        default=DEFAULT_MUSIC_MAX_SILENCE_RATIO,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices(
            "max_silence_ratio",
            "silence_ratio",
            "max_silence",
            "max_silence_fraction",
        ),
    )
    max_clipping_ratio: float = Field(
        default=DEFAULT_MUSIC_MAX_CLIPPING_RATIO,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices(
            "max_clipping_ratio",
            "clipping_ratio",
            "max_clipping",
            "max_clipping_fraction",
        ),
    )
    min_loudness_lufs: float | None = Field(
        default=DEFAULT_MUSIC_MIN_LOUDNESS_LUFS,
        validation_alias=AliasChoices(
            "min_loudness_lufs",
            "min_lufs",
            "min_loudness",
        ),
    )
    max_loudness_lufs: float | None = Field(
        default=DEFAULT_MUSIC_MAX_LOUDNESS_LUFS,
        validation_alias=AliasChoices(
            "max_loudness_lufs",
            "max_lufs",
            "max_loudness",
        ),
    )
    silence_noise_db: float = Field(
        default=DEFAULT_MUSIC_SILENCE_NOISE_DB,
        validation_alias=AliasChoices(
            "silence_noise_db",
            "silence_noise",
            "silence_threshold_db",
        ),
    )
    silence_min_seconds: float = Field(
        default=DEFAULT_MUSIC_SILENCE_MIN_SECONDS,
        ge=0.0,
        validation_alias=AliasChoices(
            "silence_min_seconds",
            "min_silence_seconds",
            "silence_duration_seconds",
        ),
    )
    clipping_peak_db: float = Field(
        default=DEFAULT_MUSIC_CLIPPING_PEAK_DB,
        validation_alias=AliasChoices(
            "clipping_peak_db",
            "clipping_db",
            "clip_peak_db",
        ),
    )
    require_loudness: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "require_loudness",
            "require_lufs",
        ),
    )
    prefer: Literal["first", "quality"] = Field(
        default="first",
        validation_alias=AliasChoices(
            "prefer",
            "strategy",
            "selection",
        ),
    )

    @field_validator(
        "min_duration_seconds",
        "max_duration_seconds",
        "target_duration_seconds",
        "duration_tolerance_seconds",
        "max_silence_ratio",
        "max_clipping_ratio",
        "silence_noise_db",
        "silence_min_seconds",
        "clipping_peak_db",
        "min_loudness_lufs",
        "max_loudness_lufs",
        mode="before",
    )
    @classmethod
    def _validate_number(cls, value: Any) -> Any:
        if value is None:
            return None
        return _finite(value, label="music candidate threshold")

    @model_validator(mode="after")
    def _validate_ranges(self) -> MusicCandidateConfig:
        if (
            self.max_duration_seconds is not None
            and self.min_duration_seconds > self.max_duration_seconds
        ):
            raise ValueError(
                "min_duration_seconds must not exceed max_duration_seconds"
            )
        if (
            self.min_loudness_lufs is not None
            and self.max_loudness_lufs is not None
            and self.min_loudness_lufs > self.max_loudness_lufs
        ):
            raise ValueError("min_loudness_lufs must not exceed max_loudness_lufs")
        return self

    @property
    def duration_bounds(self) -> tuple[float, float | None]:
        """The effective minimum and maximum duration in seconds."""
        if self.target_duration_seconds is not None:
            tolerance = self.duration_tolerance_seconds
            return (
                max(0.0, self.target_duration_seconds - tolerance),
                self.target_duration_seconds + tolerance,
            )
        return self.min_duration_seconds, self.max_duration_seconds

    @property
    def loudness_bounds(self) -> tuple[float | None, float | None]:
        """The effective integrated loudness bounds in LUFS."""
        return self.min_loudness_lufs, self.max_loudness_lufs


def music_candidate_config(
    value: MusicCandidateConfig | Mapping[str, Any] | BaseModel | None = None,
    **overrides: Any,
) -> MusicCandidateConfig:
    """Resolve *value* and non-``None`` *overrides* into a config object."""
    if value is None:
        values: dict[str, Any] = {}
    elif isinstance(value, (MusicCandidateConfig, BaseModel)):
        values = value.model_dump()
    elif isinstance(value, Mapping):
        raw = dict(value)
        nested = raw.get("music_candidates") or raw.get("music")
        values = dict(nested) if isinstance(nested, Mapping) else raw
    else:
        raise TypeError("music candidate config must be a mapping or model")
    for key, item in overrides.items():
        if item is not None:
            values[key] = item
    return MusicCandidateConfig.model_validate(values)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _last_float(pattern: re.Pattern[str], text: str) -> float | None:
    matches = pattern.findall(text)
    if not matches:
        return None
    return _as_float(matches[-1])


def _last_int(pattern: re.Pattern[str], text: str) -> int | None:
    matches = pattern.findall(text)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except (TypeError, ValueError):
        return None


def _max_float(pattern: re.Pattern[str], text: str) -> float | None:
    values = [_as_float(item) for item in pattern.findall(text)]
    present = [item for item in values if item is not None]
    return max(present) if present else None


def _silence_intervals(text: str, duration: float) -> tuple[tuple[float, float], ...]:
    intervals: list[tuple[float, float]] = []
    started: float | None = None
    for line in text.splitlines():
        start = _SILENCE_START.search(line)
        if start is not None:
            started = _as_float(start.group(1))
            continue
        end = _SILENCE_END.search(line)
        if end is None or started is None:
            continue
        finish = _as_float(end.group(1))
        if finish is None:
            continue
        reported = _as_float(end.group(2))
        length = reported if reported is not None else finish - started
        if length < 0:
            length = 0.0
        intervals.append((started, started + length))
        started = None
    if started is not None and duration > started:
        intervals.append((started, duration))
    return tuple(intervals)


def parse_music_analysis(
    text: str,
    duration_seconds: float,
    *,
    index: int = 0,
    config: MusicCandidateConfig | Mapping[str, Any] | None = None,
) -> MusicCandidateAnalysis:
    """Parse ffmpeg silence, EBU R128 and astats output into one analysis."""
    active = music_candidate_config(config)
    duration = max(0.0, _finite(duration_seconds, label="duration_seconds"))
    intervals = _silence_intervals(text, duration)
    silence_seconds = math.fsum(end - start for start, end in intervals)
    silence_ratio = silence_seconds / duration if duration else 0.0

    integrated = _last_float(_LOUDNESS, text)
    true_peak = _last_float(_TRUE_PEAK, text)
    peak_level = _max_float(_PEAK_LEVEL, text)
    max_volume = _last_float(_MAX_VOLUME, text)
    mean_volume = _last_float(_MEAN_VOLUME, text)
    peak_count = _last_int(_PEAK_COUNT, text)
    sample_count = _last_int(_SAMPLE_COUNT, text)

    if integrated is None:
        integrated = mean_volume
    if true_peak is None:
        true_peak = max_volume
    if peak_level is None:
        peak_level = max_volume

    clipped_share = 0.0
    if peak_count is not None and sample_count and sample_count > 0:
        at_full_scale = peak_level is None or peak_level >= active.clipping_peak_db
        if at_full_scale:
            clipped_share = min(1.0, peak_count / sample_count)

    return MusicCandidateAnalysis(
        index=index,
        duration_seconds=duration,
        silence_seconds=min(silence_seconds, duration),
        silence_ratio=min(1.0, max(0.0, silence_ratio)),
        silence_intervals=intervals,
        clipping_ratio=clipped_share,
        clipped_samples=peak_count or 0,
        total_samples=sample_count or 0,
        integrated_loudness_lufs=integrated,
        true_peak_db=true_peak,
        peak_db=peak_level,
        mean_volume_db=mean_volume,
        metadata={"silence_noise_db": active.silence_noise_db},
    )


@dataclass(frozen=True, slots=True)
class MusicCandidateAnalysis:
    """Measured properties of one music candidate."""

    index: int = 0
    duration_seconds: float = 0.0
    silence_seconds: float = 0.0
    silence_ratio: float = 0.0
    silence_intervals: tuple[tuple[float, float], ...] = ()
    clipping_ratio: float = 0.0
    clipped_samples: int = 0
    total_samples: int = 0
    integrated_loudness_lufs: float | None = None
    true_peak_db: float | None = None
    peak_db: float | None = None
    mean_volume_db: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.silence_ratio and self.silence_seconds and self.duration_seconds:
            object.__setattr__(
                self,
                "silence_ratio",
                min(1.0, self.silence_seconds / self.duration_seconds),
            )
        if not self.clipping_ratio and self.clipped_samples and self.total_samples:
            object.__setattr__(
                self,
                "clipping_ratio",
                min(1.0, self.clipped_samples / self.total_samples),
            )

    @property
    def duration(self) -> float:
        """Alias for :attr:`duration_seconds`."""
        return self.duration_seconds

    @property
    def silence(self) -> float:
        """Alias for :attr:`silence_seconds`."""
        return self.silence_seconds

    @property
    def clipping(self) -> float:
        """Alias for :attr:`clipping_ratio`."""
        return self.clipping_ratio

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
        """Whether the candidate contains any measured silence."""
        return bool(self.silence_seconds)

    @property
    def is_clipped(self) -> bool:
        """Whether any measured sample reached full scale."""
        return bool(self.clipping_ratio)

    @property
    def has_clipping(self) -> bool:
        """Alias for :attr:`is_clipped`."""
        return self.is_clipped

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation without raw audio."""
        return {
            "index": self.index,
            "duration_seconds": self.duration_seconds,
            "silence_seconds": self.silence_seconds,
            "silence_ratio": self.silence_ratio,
            "silence_intervals": [list(item) for item in self.silence_intervals],
            "clipping_ratio": self.clipping_ratio,
            "clipped_samples": self.clipped_samples,
            "total_samples": self.total_samples,
            "integrated_loudness_lufs": self.integrated_loudness_lufs,
            "true_peak_db": self.true_peak_db,
            "peak_db": self.peak_db,
            "mean_volume_db": self.mean_volume_db,
        }


@dataclass(frozen=True, slots=True)
class MusicCandidate:
    """One analysed candidate and the deterministic filter verdict."""

    analysis: MusicCandidateAnalysis
    track: MusicTrack | None = None
    passed: bool = True
    reasons: tuple[str, ...] = ()
    score: float = 1.0

    @property
    def index(self) -> int:
        """The candidate's position in the generated response."""
        return self.analysis.index

    @property
    def duration_seconds(self) -> float:
        """Alias for the measured duration."""
        return self.analysis.duration_seconds

    @property
    def silence_ratio(self) -> float:
        """Alias for the measured silence ratio."""
        return self.analysis.silence_ratio

    @property
    def clipping_ratio(self) -> float:
        """Alias for the measured clipping ratio."""
        return self.analysis.clipping_ratio

    @property
    def integrated_loudness_lufs(self) -> float | None:
        """Alias for the measured integrated loudness."""
        return self.analysis.integrated_loudness_lufs

    @property
    def loudness_lufs(self) -> float | None:
        """Alias for the measured integrated loudness."""
        return self.analysis.integrated_loudness_lufs

    @property
    def lufs(self) -> float | None:
        """Alias for the measured integrated loudness."""
        return self.analysis.integrated_loudness_lufs

    @property
    def accepted(self) -> bool:
        """Alias for :attr:`passed`."""
        return self.passed

    @property
    def rejected(self) -> bool:
        """Whether at least one filter rejected this candidate."""
        return not self.passed

    @property
    def reason(self) -> str:
        """The rejection reasons joined into one string."""
        return "; ".join(self.reasons)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for logs."""
        return {
            "index": self.index,
            "passed": self.passed,
            "reasons": list(self.reasons),
            "score": self.score,
            "analysis": self.analysis.to_dict(),
            "seed": None if self.track is None else self.track.seed,
            "title": None if self.track is None else self.track.title,
        }


@dataclass(frozen=True, slots=True)
class MusicCandidateSelection:
    """Every candidate, the accepted ones and the recorded chosen index."""

    candidates: tuple[MusicCandidate, ...] = ()
    chosen_index: int | None = None
    config: MusicCandidateConfig | None = None
    response: MusicResponse | None = None

    @property
    def chosen(self) -> MusicCandidate | None:
        """The accepted candidate at :attr:`chosen_index`."""
        if self.chosen_index is None:
            return None
        for candidate in self.candidates:
            if candidate.index == self.chosen_index:
                return candidate
        return None

    @property
    def selected(self) -> MusicCandidate | None:
        """Alias for :attr:`chosen`."""
        return self.chosen

    @property
    def selected_index(self) -> int | None:
        """Alias for :attr:`chosen_index`."""
        return self.chosen_index

    @property
    def index(self) -> int | None:
        """Alias for :attr:`chosen_index`."""
        return self.chosen_index

    @property
    def track(self) -> MusicTrack | None:
        """The chosen track, or ``None`` when every candidate was rejected."""
        candidate = self.chosen
        return None if candidate is None else candidate.track

    @property
    def analysis(self) -> MusicCandidateAnalysis | None:
        """The analysis of the chosen candidate."""
        candidate = self.chosen
        return None if candidate is None else candidate.analysis

    @property
    def accepted(self) -> tuple[MusicCandidate, ...]:
        """Every candidate that passed all filters in generation order."""
        return tuple(candidate for candidate in self.candidates if candidate.passed)

    @property
    def passing(self) -> tuple[MusicCandidate, ...]:
        """Alias for :attr:`accepted`."""
        return self.accepted

    @property
    def passed(self) -> bool:
        """Whether a candidate was chosen."""
        return self.ok

    @property
    def rejected(self) -> tuple[MusicCandidate, ...]:
        """Every candidate that failed at least one filter."""
        return tuple(candidate for candidate in self.candidates if not candidate.passed)

    @property
    def dropped(self) -> tuple[MusicCandidate, ...]:
        """Alias for :attr:`rejected`."""
        return self.rejected

    @property
    def best_index(self) -> int | None:
        """The highest scoring accepted candidate, if any."""
        accepted = self.accepted
        if not accepted:
            return None
        return max(accepted, key=lambda item: (item.score, -item.index)).index

    @property
    def ok(self) -> bool:
        """Whether a candidate was chosen."""
        return self.chosen_index is not None

    @property
    def failed(self) -> bool:
        """Whether no candidate was accepted."""
        return self.chosen_index is None

    @property
    def reason(self) -> str:
        """The first rejection reason, or an empty string when accepted."""
        if self.ok:
            return ""
        for candidate in self.candidates:
            if candidate.reasons:
                return candidate.reason
        return "no music candidates"

    def require(self) -> MusicCandidateSelection:
        """Return this selection, raising when every candidate was rejected."""
        if not self.ok:
            raise MusicCandidateError(
                "no music candidate passed the prefilter",
                details={
                    "candidates": [
                        candidate.to_dict() for candidate in self.candidates
                    ],
                },
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for logs."""
        return {
            "chosen_index": self.chosen_index,
            "best_index": self.best_index,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


class MusicCandidateError(ValidationError):
    """No generated music candidate passed the deterministic prefilter."""

    code = "music_candidate"


MusicCandidateResult = MusicCandidateSelection
"""Alias for :class:`MusicCandidateSelection`."""


def _format_value(value: AudioFormat | str | None) -> str | None:
    if value is None:
        return None
    text = value.value if isinstance(value, AudioFormat) else str(value)
    return text.lower().lstrip(".") or None


@contextmanager
def _materialized(
    source: MusicSource,
    *,
    format: AudioFormat | str | None = None,
) -> Iterator[Path]:
    if isinstance(source, (bytes, bytearray, memoryview)):
        payload = bytes(source)
        if not payload:
            raise ValidationError("music candidate payload is empty")
        suffix = _format_value(format)
        if suffix is None:
            signature = sniff_media(payload)
            suffix = signature.extension if signature is not None else "audio"
        with tempfile.TemporaryDirectory(prefix="aivia-music-") as directory:
            path = Path(directory) / f"source.{suffix}"
            path.write_bytes(payload)
            yield path
        return
    path = Path(source)
    if not path.is_file():
        raise ValidationError(
            f"music candidate does not exist: {path}",
            details={"path": str(path)},
        )
    yield path


def _output_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _run_ffmpeg(
    argv: Sequence[str],
    *,
    runner: FFmpegRunner | None,
    timeout_seconds: float,
) -> subprocess.CompletedProcess[Any]:
    if runner is not None:
        result = runner(argv)
        if isinstance(result, subprocess.CompletedProcess):
            return result
        return subprocess.CompletedProcess(list(argv), 0, stdout=result, stderr=b"")
    try:
        return subprocess.run(
            list(argv),
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        raise ValidationError(
            f"ffmpeg is not available: {argv[0]}",
            details={"executable": argv[0]},
        ) from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValidationError(
            "ffmpeg could not analyse the music candidate",
            details={"executable": argv[0], "error": type(exc).__name__},
        ) from exc


def _analysis_argv(
    executable: str,
    path: Path,
    config: MusicCandidateConfig,
) -> list[str]:
    filters = ",".join(
        (
            "silencedetect="
            f"noise={config.silence_noise_db:g}dB:"
            f"d={config.silence_min_seconds:g}",
            "ebur128=peak=true",
            "astats=metadata=0:reset=0",
        )
    )
    return [
        executable,
        "-hide_banner",
        "-nostats",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-af",
        filters,
        "-f",
        "null",
        "-",
    ]


def analyze_music_candidate(
    source: MusicSource | MusicTrack,
    *,
    index: int = 0,
    duration_seconds: float | None = None,
    format: AudioFormat | str | None = None,
    config: MusicCandidateConfig | Mapping[str, Any] | None = None,
    ffmpeg_runner: FFmpegRunner | None = None,
    ffprobe_runner: FFprobeRunner | None = None,
    ffmpeg_executable: str = "ffmpeg",
    ffprobe_executable: str = "ffprobe",
    timeout_seconds: float = DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS,
    ffprobe_timeout_seconds: float = DEFAULT_MEDIA_FFPROBE_TIMEOUT_SECONDS,
) -> MusicCandidateAnalysis:
    """Measure duration, silence, clipping and loudness of one candidate.

    A :class:`~aivia.providers.protocols.music.MusicTrack` must carry inline
    bytes; remote tracks are downloaded by the music service before analysis.
    ``duration_seconds`` skips ffprobe when the caller already measured it.
    """
    active = music_candidate_config(config)
    track: MusicTrack | None = None
    if isinstance(source, MusicTrack):
        track = source
        if source.data is None:
            raise ValidationError(
                "music candidate needs inline audio data for analysis",
                details={"index": index, "url": source.url},
            )
        payload: MusicSource = source.data
        if format is None:
            format = source.format
    else:
        payload = source

    with _materialized(payload, format=format) as path:
        duration_source = "measured"
        if duration_seconds is None:
            try:
                duration = measure_audio_duration(
                    path,
                    format=format,
                    ffprobe_executable=ffprobe_executable,
                    ffprobe_runner=ffprobe_runner,
                    timeout_seconds=ffprobe_timeout_seconds,
                )
            except ValidationError:
                if (
                    ffprobe_runner is None
                    and track is not None
                    and track.duration_seconds is not None
                ):
                    duration = _finite(
                        track.duration_seconds,
                        label="duration_seconds",
                    )
                    duration_source = "reported"
                else:
                    raise
        else:
            duration = _finite(duration_seconds, label="duration_seconds")

        argv = _analysis_argv(ffmpeg_executable, path, active)
        completed = _run_ffmpeg(
            argv,
            runner=ffmpeg_runner,
            timeout_seconds=timeout_seconds,
        )
        stderr = _output_text(completed.stderr)
        stdout = _output_text(completed.stdout)
        if completed.returncode != 0:
            raise ValidationError(
                "ffmpeg could not analyse the music candidate",
                details={
                    "returncode": completed.returncode,
                    "stderr": stderr[:500],
                },
            )

    analysis = parse_music_analysis(
        "\n".join((stderr, stdout)),
        duration,
        index=index,
        config=active,
    )
    analysis.metadata["duration_source"] = duration_source
    if track is not None and track.duration_seconds is not None:
        analysis.metadata["reported_duration_seconds"] = track.duration_seconds
    return analysis


def _coerce_analysis(
    value: Any,
    *,
    index: int,
    duration_seconds: float | None = None,
) -> MusicCandidateAnalysis:
    if isinstance(value, MusicCandidateAnalysis):
        if value.index == index:
            return value
        return MusicCandidateAnalysis(
            index=index,
            duration_seconds=value.duration_seconds,
            silence_seconds=value.silence_seconds,
            silence_ratio=value.silence_ratio,
            silence_intervals=value.silence_intervals,
            clipping_ratio=value.clipping_ratio,
            clipped_samples=value.clipped_samples,
            total_samples=value.total_samples,
            integrated_loudness_lufs=value.integrated_loudness_lufs,
            true_peak_db=value.true_peak_db,
            peak_db=value.peak_db,
            mean_volume_db=value.mean_volume_db,
            metadata=dict(value.metadata),
        )
    if isinstance(value, MusicCandidate):
        return _coerce_analysis(value.analysis, index=index)
    if isinstance(value, Mapping):
        duration = value.get("duration_seconds", value.get("duration"))
        if duration is None:
            duration = duration_seconds
        if duration is None:
            raise ValidationError("music candidate analysis needs a duration")
        seconds = _finite(duration, label="duration_seconds")
        silence_seconds = _as_float(value.get("silence_seconds", value.get("silence")))
        silence_ratio = _as_float(value.get("silence_ratio"))
        if silence_ratio is None and silence_seconds is not None and seconds:
            silence_ratio = silence_seconds / seconds
        return MusicCandidateAnalysis(
            index=index,
            duration_seconds=seconds,
            silence_seconds=silence_seconds or 0.0,
            silence_ratio=silence_ratio or 0.0,
            clipping_ratio=_as_float(value.get("clipping_ratio", value.get("clipping")))
            or 0.0,
            clipped_samples=int(value.get("clipped_samples", 0) or 0),
            total_samples=int(value.get("total_samples", 0) or 0),
            integrated_loudness_lufs=_as_float(
                value.get(
                    "integrated_loudness_lufs",
                    value.get(
                        "loudness_lufs",
                        value.get("lufs", value.get("loudness")),
                    ),
                )
            ),
            true_peak_db=_as_float(value.get("true_peak_db", value.get("peak"))),
            peak_db=_as_float(value.get("peak_db")),
            mean_volume_db=_as_float(value.get("mean_volume_db")),
        )
    raise TypeError("analyzer must return a music candidate analysis or mapping")


def _call_analyzer(
    analyzer: Callable[..., Any],
    track: MusicTrack,
    index: int,
) -> Any:
    parameters: Mapping[str, Any]
    try:
        parameters = inspect.signature(analyzer).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "index" in parameters:
        return analyzer(track, index=index)
    return analyzer(track)


def _evaluate(
    analysis: MusicCandidateAnalysis,
    config: MusicCandidateConfig,
) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    minimum, maximum = config.duration_bounds
    if analysis.duration_seconds < minimum:
        reasons.append("duration too short")
    if maximum is not None and analysis.duration_seconds > maximum:
        reasons.append("duration too long")
    if analysis.silence_ratio > config.max_silence_ratio:
        reasons.append("too much silence")
    if analysis.clipping_ratio > config.max_clipping_ratio:
        reasons.append("clipping")
    peak = analysis.true_peak_db
    if peak is None:
        peak = analysis.peak_db
    if (
        peak is not None
        and peak >= config.clipping_peak_db
        and "clipping" not in reasons
    ):
        reasons.append("clipping")
    loudness = analysis.integrated_loudness_lufs
    if loudness is None:
        if config.require_loudness:
            reasons.append("loudness unavailable")
    else:
        if config.min_loudness_lufs is not None and loudness < config.min_loudness_lufs:
            reasons.append("too quiet")
        if config.max_loudness_lufs is not None and loudness > config.max_loudness_lufs:
            reasons.append("too loud")
    return not reasons, tuple(reasons)


def score_music_candidate(
    analysis: MusicCandidateAnalysis,
    config: MusicCandidateConfig | Mapping[str, Any] | None = None,
) -> float:
    """Score one candidate from 0 to 1, rewarding closeness to the targets."""
    active = music_candidate_config(config)
    parts: list[float] = []
    minimum, maximum = active.duration_bounds
    if active.target_duration_seconds is not None:
        tolerance = max(active.duration_tolerance_seconds, 1e-9)
        distance = abs(analysis.duration_seconds - active.target_duration_seconds)
        parts.append(max(0.0, 1.0 - distance / tolerance))
    elif maximum is not None and maximum > minimum:
        midpoint = (minimum + maximum) / 2.0
        half = (maximum - minimum) / 2.0
        parts.append(max(0.0, 1.0 - abs(analysis.duration_seconds - midpoint) / half))
    if active.max_silence_ratio:
        parts.append(max(0.0, 1.0 - analysis.silence_ratio / active.max_silence_ratio))
    if active.max_clipping_ratio:
        parts.append(
            max(0.0, 1.0 - analysis.clipping_ratio / active.max_clipping_ratio)
        )
    loudness = analysis.integrated_loudness_lufs
    low = active.min_loudness_lufs
    high = active.max_loudness_lufs
    if loudness is not None and low is not None and high is not None and high > low:
        midpoint = (low + high) / 2.0
        half = (high - low) / 2.0
        parts.append(max(0.0, 1.0 - abs(loudness - midpoint) / half))
    if not parts:
        return 1.0
    return math.fsum(parts) / len(parts)


def _normalize_candidates(
    value: Any,
) -> tuple[list[Any], MusicResponse | None]:
    if isinstance(value, MusicResponse):
        return list(value.tracks), value
    if isinstance(value, (MusicTrack, MusicCandidate, MusicCandidateAnalysis)):
        return [value], None
    if isinstance(value, Mapping):
        return [value], None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value), None
    raise TypeError("music candidates must be a response, track or sequence")


def select_music_candidate(
    candidates: Any,
    config: MusicCandidateConfig | Mapping[str, Any] | None = None,
    *,
    analyzer: Callable[..., Any] | None = None,
    response: MusicResponse | None = None,
    ffmpeg_runner: FFmpegRunner | None = None,
    ffprobe_runner: FFprobeRunner | None = None,
    ffmpeg_executable: str = "ffmpeg",
    ffprobe_executable: str = "ffprobe",
    timeout_seconds: float = DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS,
    ffprobe_timeout_seconds: float = DEFAULT_MEDIA_FFPROBE_TIMEOUT_SECONDS,
) -> MusicCandidateSelection:
    """Analyse, prefilter and choose one candidate, recording its index."""
    active = music_candidate_config(config)
    items, detected = _normalize_candidates(candidates)
    if response is None:
        response = detected
    selected: list[MusicCandidate] = []
    for index, item in enumerate(items):
        if isinstance(item, MusicCandidate):
            analysis = _coerce_analysis(item.analysis, index=index)
            track = item.track
        elif isinstance(item, MusicCandidateAnalysis):
            analysis = _coerce_analysis(item, index=index)
            track = None
        elif isinstance(item, MusicTrack):
            track = item
            if analyzer is not None:
                result = _call_analyzer(analyzer, item, index)
            else:
                result = analyze_music_candidate(
                    item,
                    index=index,
                    config=active,
                    ffmpeg_runner=ffmpeg_runner,
                    ffprobe_runner=ffprobe_runner,
                    ffmpeg_executable=ffmpeg_executable,
                    ffprobe_executable=ffprobe_executable,
                    timeout_seconds=timeout_seconds,
                    ffprobe_timeout_seconds=ffprobe_timeout_seconds,
                )
            analysis = _coerce_analysis(result, index=index)
        elif isinstance(item, Mapping):
            analysis = _coerce_analysis(item, index=index)
            track = None
        else:
            raise TypeError("each music candidate must be a track, analysis or mapping")
        passed, reasons = _evaluate(analysis, active)
        selected.append(
            MusicCandidate(
                analysis=analysis,
                track=track,
                passed=passed,
                reasons=reasons,
                score=score_music_candidate(analysis, active),
            )
        )

    accepted = [candidate for candidate in selected if candidate.passed]
    chosen_index: int | None = None
    if accepted:
        if active.prefer == "quality":
            chosen = max(accepted, key=lambda item: (item.score, -item.index))
        else:
            chosen = accepted[0]
        chosen_index = chosen.index

    return MusicCandidateSelection(
        candidates=tuple(selected),
        chosen_index=chosen_index,
        config=active,
        response=response,
    )


select_music_candidates = select_music_candidate
select_music_track = select_music_candidate
"""Aliases for :func:`select_music_candidate`."""


def prefilter_music_candidates(
    candidates: Any,
    config: MusicCandidateConfig | Mapping[str, Any] | None = None,
    **values: Any,
) -> tuple[MusicCandidate, ...]:
    """Return only the candidates that pass every deterministic filter."""
    return select_music_candidate(candidates, config, **values).accepted


filter_music_candidates = prefilter_music_candidates
"""Alias for :func:`prefilter_music_candidates`."""


async def _generate(
    generator: Any,
    request: MusicRequest,
) -> MusicResponse:
    method = getattr(generator, "generate", None)
    if method is None and callable(generator):
        method = generator
    if method is None:
        raise TypeError("generator must have a generate method or be callable")
    result = method(request)
    if inspect.isawaitable(result):
        result = await result
    if isinstance(result, MusicResponse):
        return result
    if isinstance(result, MusicTrack):
        return MusicResponse(tracks=[result])
    if isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
        return MusicResponse(tracks=list(result))
    raise TypeError("generator must return a MusicResponse or music tracks")


async def generate_music_candidates(
    generator: Any,
    request: MusicRequest,
    *,
    config: MusicCandidateConfig | Mapping[str, Any] | None = None,
    count: int | None = None,
    analyzer: Callable[..., Any] | None = None,
    ffmpeg_runner: FFmpegRunner | None = None,
    ffprobe_runner: FFprobeRunner | None = None,
    ffmpeg_executable: str = "ffmpeg",
    ffprobe_executable: str = "ffprobe",
    timeout_seconds: float = DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS,
    ffprobe_timeout_seconds: float = DEFAULT_MEDIA_FFPROBE_TIMEOUT_SECONDS,
) -> MusicCandidateSelection:
    """Ask *generator* for several tracks, prefilter them and choose one.

    The generator is any object with an async ``generate`` method, normally the
    music service. When the request names a duration it becomes the target the
    candidate duration is checked against unless the config overrides it.
    """
    if not isinstance(request, MusicRequest):
        raise TypeError("request must be a MusicRequest")
    active = request
    if count is not None:
        active = request.model_copy(update={"count": count})
    settings = music_candidate_config(config)
    if settings.target_duration_seconds is None and active.duration_seconds is not None:
        settings = settings.model_copy(
            update={"target_duration_seconds": active.duration_seconds}
        )
    response = await _generate(generator, active)
    return select_music_candidate(
        response,
        settings,
        analyzer=analyzer,
        ffmpeg_runner=ffmpeg_runner,
        ffprobe_runner=ffprobe_runner,
        ffmpeg_executable=ffmpeg_executable,
        ffprobe_executable=ffprobe_executable,
        timeout_seconds=timeout_seconds,
        ffprobe_timeout_seconds=ffprobe_timeout_seconds,
    )


def generate_music_candidates_sync(
    generator: Any,
    request: MusicRequest,
    **values: Any,
) -> MusicCandidateSelection:
    """Run :func:`generate_music_candidates` from synchronous code."""
    return asyncio.run(generate_music_candidates(generator, request, **values))


class MusicCandidateSelector:
    """Reusable analyzer and prefilter with one set of thresholds."""

    def __init__(
        self,
        config: MusicCandidateConfig | Mapping[str, Any] | None = None,
        *,
        analyzer: Callable[..., Any] | None = None,
        ffmpeg_runner: FFmpegRunner | None = None,
        ffprobe_runner: FFprobeRunner | None = None,
        ffmpeg_executable: str = "ffmpeg",
        ffprobe_executable: str = "ffprobe",
        timeout_seconds: float = DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS,
        ffprobe_timeout_seconds: float = DEFAULT_MEDIA_FFPROBE_TIMEOUT_SECONDS,
    ) -> None:
        self._config = music_candidate_config(config)
        self._analyzer = analyzer
        self._ffmpeg_runner = ffmpeg_runner
        self._ffprobe_runner = ffprobe_runner
        self._ffmpeg_executable = ffmpeg_executable
        self._ffprobe_executable = ffprobe_executable
        self._timeout_seconds = timeout_seconds
        self._ffprobe_timeout_seconds = ffprobe_timeout_seconds

    @property
    def config(self) -> MusicCandidateConfig:
        """The thresholds this selector applies."""
        return self._config

    @property
    def thresholds(self) -> MusicCandidateConfig:
        """Alias for :attr:`config`."""
        return self._config

    def analyze(
        self,
        source: MusicSource | MusicTrack,
        *,
        index: int = 0,
        duration_seconds: float | None = None,
        format: AudioFormat | str | None = None,
    ) -> MusicCandidateAnalysis:
        """Analyze one source with this selector's runners and thresholds."""
        return analyze_music_candidate(
            source,
            index=index,
            duration_seconds=duration_seconds,
            format=format,
            config=self._config,
            ffmpeg_runner=self._ffmpeg_runner,
            ffprobe_runner=self._ffprobe_runner,
            ffmpeg_executable=self._ffmpeg_executable,
            ffprobe_executable=self._ffprobe_executable,
            timeout_seconds=self._timeout_seconds,
            ffprobe_timeout_seconds=self._ffprobe_timeout_seconds,
        )

    def select(self, candidates: Any) -> MusicCandidateSelection:
        """Analyze, prefilter and choose one candidate."""
        return select_music_candidate(
            candidates,
            self._config,
            analyzer=self._analyzer,
            ffmpeg_runner=self._ffmpeg_runner,
            ffprobe_runner=self._ffprobe_runner,
            ffmpeg_executable=self._ffmpeg_executable,
            ffprobe_executable=self._ffprobe_executable,
            timeout_seconds=self._timeout_seconds,
            ffprobe_timeout_seconds=self._ffprobe_timeout_seconds,
        )

    def prefilter(self, candidates: Any) -> tuple[MusicCandidate, ...]:
        """Return only the candidates that pass every filter."""
        return self.select(candidates).accepted

    def filter(self, candidates: Any) -> tuple[MusicCandidate, ...]:
        """Alias for :meth:`prefilter`."""
        return self.prefilter(candidates)

    async def generate(
        self,
        generator: Any,
        request: MusicRequest,
        *,
        count: int | None = None,
    ) -> MusicCandidateSelection:
        """Generate candidates with *generator* and select one."""
        return await generate_music_candidates(
            generator,
            request,
            config=self._config,
            count=count,
            analyzer=self._analyzer,
            ffmpeg_runner=self._ffmpeg_runner,
            ffprobe_runner=self._ffprobe_runner,
            ffmpeg_executable=self._ffmpeg_executable,
            ffprobe_executable=self._ffprobe_executable,
            timeout_seconds=self._timeout_seconds,
            ffprobe_timeout_seconds=self._ffprobe_timeout_seconds,
        )

    def generate_sync(
        self,
        generator: Any,
        request: MusicRequest,
        *,
        count: int | None = None,
    ) -> MusicCandidateSelection:
        """Run :meth:`generate` from synchronous code."""
        return asyncio.run(self.generate(generator, request, count=count))


MusicCandidateFilter = MusicCandidateSelector
"""Alias for :class:`MusicCandidateSelector`."""

MusicCandidatePrefilter = MusicCandidateSelector
"""Alias for :class:`MusicCandidateSelector`."""

MusicCandidateGenerator = MusicCandidateSelector
"""Alias for :class:`MusicCandidateSelector`."""
