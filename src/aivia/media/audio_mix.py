"""Analyse audio and mix several tracks with verified voice ducking.

Generated speech, music and effects arrive as separate files with no shared
timeline. Before they are composed they need a measured tempo, a silence map
and integrated loudness, and then one deterministic mix where a music bed
steps back whenever a voice is present.

The analysis runs ffmpeg with ``silencedetect``, ``ebur128`` and ``astats``.
Tempo is read from an explicit ffmpeg report when one is present and otherwise
estimated from the loudness envelope or from a short PCM decode. Mixing builds
a pure ``filter_complex`` string first, so the graph can be inspected without
running a binary, then writes a new file through a temporary path. Voice
ducking uses ``sidechaincompress`` and is verified from the produced graph and
settings, never assumed from a flag.

Both ffmpeg and ffprobe are invoked with argv lists and no shell. Their runners
are injectable, which keeps the offline suite free of the real binaries.
"""

from __future__ import annotations

import array
import itertools
import math
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
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
    DEFAULT_AUDIO_ANALYSIS_MAX_BPM as DEFAULT_ANALYSIS_MAX_BPM,
)
from aivia.core.config.media import (
    DEFAULT_AUDIO_ANALYSIS_MIN_BPM as DEFAULT_ANALYSIS_MIN_BPM,
)
from aivia.core.config.media import (
    DEFAULT_AUDIO_DUCKING_ATTACK_MS as DEFAULT_DUCKING_ATTACK_MS,
)
from aivia.core.config.media import (
    DEFAULT_AUDIO_DUCKING_RATIO as DEFAULT_DUCKING_RATIO,
)
from aivia.core.config.media import (
    DEFAULT_AUDIO_DUCKING_RELEASE_MS as DEFAULT_DUCKING_RELEASE_MS,
)
from aivia.core.config.media import (
    DEFAULT_AUDIO_DUCKING_THRESHOLD_DB as DEFAULT_DUCKING_THRESHOLD_DB,
)
from aivia.core.config.media import (
    DEFAULT_AUDIO_MIX_CHANNELS as DEFAULT_MIX_CHANNELS,
)
from aivia.core.config.media import (
    DEFAULT_AUDIO_MIX_SAMPLE_RATE as DEFAULT_MIX_SAMPLE_RATE,
)
from aivia.core.config.media import (
    DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS,
    DEFAULT_MEDIA_FFPROBE_TIMEOUT_SECONDS,
    DEFAULT_MUSIC_SILENCE_MIN_SECONDS,
    DEFAULT_MUSIC_SILENCE_NOISE_DB,
)
from aivia.core.errors import ValidationError
from aivia.core.media_probe import FFprobeRunner, sniff_media
from aivia.media.audio_utils import measure_audio_duration
from aivia.providers.protocols.audio import AudioFormat

__all__ = [
    "DEFAULT_ANALYSIS_MAX_BPM",
    "DEFAULT_ANALYSIS_MIN_BPM",
    "DEFAULT_DUCKING_ATTACK_MS",
    "DEFAULT_DUCKING_RATIO",
    "DEFAULT_DUCKING_RELEASE_MS",
    "DEFAULT_DUCKING_THRESHOLD_DB",
    "DEFAULT_MIX_CHANNELS",
    "DEFAULT_MIX_SAMPLE_RATE",
    "AudioAnalysis",
    "AudioMixResult",
    "AudioMixSettings",
    "AudioSource",
    "AudioTrack",
    "AudioTrackRole",
    "DuckingReport",
    "DuckingSettings",
    "DuckingVerification",
    "FFmpegRunner",
    "analyze_audio",
    "build_mix_filtergraph",
    "estimate_tempo",
    "estimate_tempo_from_envelope",
    "mix_audio",
    "mix_tracks",
    "parse_audio_analysis",
    "verify_ducking",
]

type AudioSource = str | Path | bytes
"""An audio path or the raw bytes of an encoded payload."""

FFmpegRunner = Callable[
    [Sequence[str]],
    subprocess.CompletedProcess[Any]
    | bytes
    | bytearray
    | memoryview
    | str
    | Path
    | None,
]
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
_TEMPO = re.compile(
    r"\b(?:tempo(?:_bpm)?|bpm)\s*[:=]\s*(" + _NUMBER + r")",
    re.IGNORECASE,
)
_TEMPO_SUFFIX = re.compile(r"(" + _NUMBER + r")\s*bpm\b", re.IGNORECASE)
_EBUR_FRAME = re.compile(
    r"\bt:\s*(" + _NUMBER + r").*?\bM:\s*(" + _NUMBER + r")",
    re.IGNORECASE,
)
_SIDECHAIN = re.compile(
    r"sidechaincompress=threshold=(" + _NUMBER + r"):ratio=(" + _NUMBER + r")"
    r":attack=(" + _NUMBER + r"):release=(" + _NUMBER + r")"
)

_CODECS: dict[str, str] = {
    "mp3": "libmp3lame",
    "wav": "pcm_s16le",
    "ogg": "libvorbis",
    "opus": "libopus",
    "aac": "aac",
    "flac": "flac",
    "m4a": "aac",
    "pcm": "pcm_s16le",
}


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
    if non_negative and number < 0:
        raise ValueError(f"{label} must be non-negative")
    if positive and number <= 0:
        raise ValueError(f"{label} must be positive")
    return number


def _finite(value: Any, *, label: str) -> float:
    try:
        return _number(value, label=label)
    except (TypeError, ValueError) as exc:
        raise ValidationError(str(exc)) from exc


def _last_float(pattern: re.Pattern[str], text: str) -> float | None:
    matches = pattern.findall(text)
    if not matches:
        return None
    value = matches[-1]
    if isinstance(value, tuple):
        value = value[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _max_float(pattern: re.Pattern[str], text: str) -> float | None:
    values = [float(item) for item in pattern.findall(text) if item]
    return max(values) if values else None


def _silence_intervals(
    text: str,
    duration: float,
) -> tuple[tuple[float, float], ...]:
    intervals: list[tuple[float, float]] = []
    started: float | None = None
    for line in text.splitlines():
        start = _SILENCE_START.search(line)
        if start is not None:
            started = max(0.0, float(start.group(1)))
            continue
        end = _SILENCE_END.search(line)
        if end is None:
            continue
        finish = float(end.group(1))
        if started is None:
            started = 0.0
        reported = end.group(2)
        length = float(reported) if reported is not None else finish - started
        if length < 0:
            length = 0.0
        intervals.append((started, started + length))
        started = None
    if started is not None and duration > started:
        intervals.append((started, duration))
    return tuple(intervals)


def _tempo_from_frames(text: str) -> float | None:
    frames = [
        float(match.group(2)) for match in _EBUR_FRAME.finditer(text) if match.group(2)
    ]
    if len(frames) < 8:
        return None
    return estimate_tempo_from_envelope(frames, frames_per_second=10.0)


def estimate_tempo_from_envelope(
    envelope: Sequence[float],
    *,
    frames_per_second: float = 10.0,
    min_bpm: float = DEFAULT_ANALYSIS_MIN_BPM,
    max_bpm: float = DEFAULT_ANALYSIS_MAX_BPM,
) -> float | None:
    """Estimate a tempo in BPM from a loudness or energy envelope.

    The envelope is sampled at ``frames_per_second``. Positive frame-to-frame
    changes are treated as onsets and scored with normalized autocorrelation.
    A flat or very short envelope returns ``None`` rather than a guess.
    """
    rate = _finite(frames_per_second, label="frames_per_second")
    if rate <= 0:
        raise ValidationError("frames_per_second must be positive")
    low = _finite(min_bpm, label="min_bpm")
    high = _finite(max_bpm, label="max_bpm")
    if low <= 0 or high <= low:
        raise ValidationError("min_bpm and max_bpm must be positive and ordered")
    values = [float(item) for item in envelope]
    if len(values) < 4:
        return None
    onsets = [
        max(0.0, current - previous) for previous, current in itertools.pairwise(values)
    ]
    if not any(onsets):
        return None
    mean = math.fsum(onsets) / len(onsets)
    centered = [value - mean for value in onsets]
    minimum_lag = max(1, math.floor(rate * 60.0 / high))
    maximum_lag = min(len(centered) - 1, math.ceil(rate * 60.0 / low))
    if maximum_lag <= minimum_lag:
        return None
    best_lag = 0
    peak_correlation = 0.0
    for lag in range(minimum_lag, maximum_lag + 1):
        left = centered[:-lag]
        right = centered[lag:]
        numerator = math.fsum(a * b for a, b in zip(left, right, strict=False))
        left_energy = math.fsum(value * value for value in left)
        right_energy = math.fsum(value * value for value in right)
        denominator = math.sqrt(left_energy * right_energy)
        if denominator <= 0:
            continue
        score = numerator / denominator
        if score > peak_correlation:
            peak_correlation = score
            best_lag = lag
    if best_lag == 0 or peak_correlation <= 0:
        return None
    bpm = 60.0 * rate / best_lag
    while bpm < low:
        bpm *= 2.0
    while bpm > high:
        bpm /= 2.0
    return round(bpm, 3)


def estimate_tempo(
    pcm: bytes | bytearray | memoryview,
    *,
    sample_rate: int = 8_000,
    frame_seconds: float = 0.02,
    min_bpm: float = DEFAULT_ANALYSIS_MIN_BPM,
    max_bpm: float = DEFAULT_ANALYSIS_MAX_BPM,
) -> float | None:
    """Estimate tempo from mono signed 16-bit little-endian PCM bytes."""
    if isinstance(pcm, (bytearray, memoryview)):
        payload = bytes(pcm)
    elif isinstance(pcm, bytes):
        payload = pcm
    else:
        raise TypeError("pcm must be bytes-like")
    rate = int(_finite(sample_rate, label="sample_rate"))
    if rate <= 0:
        raise ValidationError("sample_rate must be positive")
    window = max(1, int(rate * _finite(frame_seconds, label="frame_seconds")))
    samples = array.array("h")
    usable = len(payload) - (len(payload) % 2)
    if usable <= 0:
        return None
    samples.frombytes(payload[:usable])
    if sys.byteorder == "big":
        samples.byteswap()
    if len(samples) < window * 4:
        return None
    envelope: list[float] = []
    for start in range(0, len(samples) - window + 1, window):
        block = samples[start : start + window]
        energy = math.fsum(float(value) * float(value) for value in block)
        envelope.append(math.sqrt(energy / len(block)))
    return estimate_tempo_from_envelope(
        envelope,
        frames_per_second=rate / window,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
    )


@dataclass(frozen=True, slots=True)
class AudioAnalysis:
    """Measured properties of one audio file."""

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
        if not self.silence_ratio and self.silence_seconds and self.duration_seconds:
            object.__setattr__(
                self,
                "silence_ratio",
                min(1.0, self.silence_seconds / self.duration_seconds),
            )

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
            "integrated_loudness_lufs": self.integrated_loudness_lufs,
            "true_peak_db": self.true_peak_db,
            "peak_db": self.peak_db,
            "mean_volume_db": self.mean_volume_db,
        }


def parse_audio_analysis(
    text: str,
    duration_seconds: float,
    *,
    tempo_bpm: float | None = None,
    min_bpm: float = DEFAULT_ANALYSIS_MIN_BPM,
    max_bpm: float = DEFAULT_ANALYSIS_MAX_BPM,
) -> AudioAnalysis:
    """Parse ffmpeg silence, EBU R128, astats and tempo output."""
    duration = max(0.0, _finite(duration_seconds, label="duration_seconds"))
    intervals = _silence_intervals(text, duration)
    silence_seconds = math.fsum(end - start for start, end in intervals)
    silence_ratio = silence_seconds / duration if duration else 0.0

    integrated = _last_float(_LOUDNESS, text)
    true_peak = _last_float(_TRUE_PEAK, text)
    peak_level = _max_float(_PEAK_LEVEL, text)
    max_volume = _last_float(_MAX_VOLUME, text)
    mean_volume = _last_float(_MEAN_VOLUME, text)

    if integrated is None:
        integrated = mean_volume
    if true_peak is None:
        true_peak = max_volume
    if peak_level is None:
        peak_level = max_volume

    tempo = tempo_bpm
    source = "provided" if tempo is not None else ""
    if tempo is None:
        tempo = _last_float(_TEMPO, text)
        if tempo is None:
            tempo = _last_float(_TEMPO_SUFFIX, text)
        if tempo is not None:
            source = "reported"
    if tempo is None:
        tempo = _tempo_from_frames(text)
        if tempo is not None:
            source = "estimated"

    return AudioAnalysis(
        duration_seconds=duration,
        tempo_bpm=tempo,
        silence_seconds=min(silence_seconds, duration),
        silence_ratio=min(1.0, max(0.0, silence_ratio)),
        silence_intervals=intervals,
        integrated_loudness_lufs=integrated,
        true_peak_db=true_peak,
        peak_db=peak_level,
        mean_volume_db=mean_volume,
        metadata={"tempo_source": source} if source else {},
    )


@contextmanager
def _materialize(
    source: AudioSource,
    *,
    prefix: str = "aivia-audio-mix-",
) -> Iterator[Path]:
    if isinstance(source, (bytes, bytearray, memoryview)):
        payload = bytes(source)
        if not payload:
            raise ValidationError("audio payload is empty")
        signature = sniff_media(payload)
        extension = signature.extension if signature is not None else "audio"
        with tempfile.TemporaryDirectory(prefix=prefix) as directory:
            path = Path(directory) / f"source.{extension}"
            path.write_bytes(payload)
            yield path
        return
    path = Path(source)
    if not path.is_file():
        raise ValidationError(
            f"audio source does not exist: {path}",
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
            "ffmpeg could not run",
            details={"executable": argv[0], "error": type(exc).__name__},
        ) from exc


def _base(executable: str) -> list[str]:
    return [executable, "-hide_banner", "-loglevel", "error", "-y"]


def _analysis_argv(
    executable: str,
    path: Path,
    *,
    silence_noise_db: float,
    silence_min_seconds: float,
) -> list[str]:
    filters = ",".join(
        (
            f"silencedetect=noise={silence_noise_db:g}dB:d={silence_min_seconds:g}",
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


def _tempo_argv(executable: str, path: Path, sample_rate: int) -> list[str]:
    return [
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "-",
    ]


def analyze_audio(
    source: AudioSource,
    *,
    duration_seconds: float | None = None,
    format: str | AudioFormat | None = None,
    tempo_bpm: float | None = None,
    min_bpm: float = DEFAULT_ANALYSIS_MIN_BPM,
    max_bpm: float = DEFAULT_ANALYSIS_MAX_BPM,
    silence_noise_db: float = DEFAULT_MUSIC_SILENCE_NOISE_DB,
    silence_min_seconds: float = DEFAULT_MUSIC_SILENCE_MIN_SECONDS,
    detect_tempo: bool = True,
    decode_tempo: bool = True,
    tempo_sample_rate: int = 8_000,
    ffmpeg_runner: FFmpegRunner | None = None,
    ffprobe_runner: FFprobeRunner | None = None,
    ffmpeg_executable: str = "ffmpeg",
    ffprobe_executable: str = "ffprobe",
    timeout_seconds: float = DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS,
    ffprobe_timeout_seconds: float = DEFAULT_MEDIA_FFPROBE_TIMEOUT_SECONDS,
) -> AudioAnalysis:
    """Measure duration, tempo, silence and loudness of *source*.

    An explicit ``tempo_bpm`` skips tempo detection. Otherwise a tempo reported
    by ffmpeg is used, then the loudness envelope is scored, and finally a
    short mono PCM decode is analysed. A failed optional decode leaves the
    tempo unset instead of failing the whole analysis.
    """
    noise = _finite(silence_noise_db, label="silence_noise_db")
    minimum_silence = _finite(
        silence_min_seconds,
        label="silence_min_seconds",
    )
    if minimum_silence < 0:
        raise ValidationError("silence_min_seconds must not be negative")
    fmt = format.value if isinstance(format, AudioFormat) else format
    with _materialize(source) as path:
        if duration_seconds is None:
            duration = measure_audio_duration(
                path,
                format=fmt,
                ffprobe_executable=ffprobe_executable,
                ffprobe_runner=ffprobe_runner,
                timeout_seconds=ffprobe_timeout_seconds,
            )
        else:
            duration = _finite(duration_seconds, label="duration_seconds")

        completed = _run_ffmpeg(
            _analysis_argv(
                ffmpeg_executable,
                path,
                silence_noise_db=noise,
                silence_min_seconds=minimum_silence,
            ),
            runner=ffmpeg_runner,
            timeout_seconds=timeout_seconds,
        )
        text = "\n".join(
            (_output_text(completed.stderr), _output_text(completed.stdout))
        )
        if completed.returncode != 0:
            raise ValidationError(
                "ffmpeg could not analyse the audio",
                details={
                    "returncode": completed.returncode,
                    "stderr": text[:500],
                },
            )

        analysis = parse_audio_analysis(
            text,
            duration,
            tempo_bpm=tempo_bpm,
            min_bpm=min_bpm,
            max_bpm=max_bpm,
        )
        if analysis.tempo_bpm is None and detect_tempo and decode_tempo:
            try:
                decoded = _run_ffmpeg(
                    _tempo_argv(
                        ffmpeg_executable,
                        path,
                        int(_finite(tempo_sample_rate, label="tempo_sample_rate")),
                    ),
                    runner=ffmpeg_runner,
                    timeout_seconds=timeout_seconds,
                )
            except ValidationError:
                decoded = None
            if decoded is not None and decoded.returncode == 0:
                raw = decoded.stdout
                if isinstance(raw, (bytes, bytearray, memoryview)):
                    tempo = estimate_tempo(
                        raw,
                        sample_rate=tempo_sample_rate,
                        min_bpm=min_bpm,
                        max_bpm=max_bpm,
                    )
                    if tempo is not None:
                        analysis = AudioAnalysis(
                            duration_seconds=analysis.duration_seconds,
                            tempo_bpm=tempo,
                            silence_seconds=analysis.silence_seconds,
                            silence_ratio=analysis.silence_ratio,
                            silence_intervals=analysis.silence_intervals,
                            integrated_loudness_lufs=(
                                analysis.integrated_loudness_lufs
                            ),
                            true_peak_db=analysis.true_peak_db,
                            peak_db=analysis.peak_db,
                            mean_volume_db=analysis.mean_volume_db,
                            metadata={
                                **analysis.metadata,
                                "tempo_source": "decoded",
                            },
                        )
        return analysis


class AudioTrackRole(StrEnum):
    """How a track participates in ducking."""

    VOICE = "voice"
    MUSIC = "music"
    BACKGROUND = "background"
    SFX = "sfx"
    EFFECTS = "effects"

    @property
    def is_voice(self) -> bool:
        """Whether this role is a voice that drives the sidechain."""
        return self is AudioTrackRole.VOICE


class AudioTrack(BaseModel):
    """One input track with its gain, placement and fades."""

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )

    source: AudioSource = Field(
        validation_alias=AliasChoices(
            "source",
            "path",
            "file",
            "audio",
            "data",
        )
    )
    role: AudioTrackRole = Field(
        default=AudioTrackRole.MUSIC,
        validation_alias=AliasChoices("role", "kind", "type"),
    )
    is_voice: bool | None = Field(
        default=None,
        validation_alias=AliasChoices("is_voice", "voice"),
    )
    duck: bool | None = Field(
        default=None,
        validation_alias=AliasChoices("duck", "ducked", "allow_ducking"),
    )
    volume_db: float = Field(
        default=0.0,
        validation_alias=AliasChoices("volume_db", "volume", "gain_db", "gain"),
    )
    offset_seconds: float = Field(
        default=0.0,
        validation_alias=AliasChoices(
            "offset_seconds",
            "offset",
            "start_seconds",
            "delay_seconds",
            "delay",
        ),
    )
    fade_in_seconds: float = Field(
        default=0.0,
        validation_alias=AliasChoices("fade_in_seconds", "fade_in"),
    )
    fade_out_seconds: float = Field(
        default=0.0,
        validation_alias=AliasChoices("fade_out_seconds", "fade_out"),
    )
    duration_seconds: float | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "duration_seconds",
            "duration",
            "length_seconds",
        ),
    )
    loop: bool = Field(default=False, validation_alias=AliasChoices("loop", "looped"))

    @field_validator("volume_db", mode="before")
    @classmethod
    def _validate_volume(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("volume_db must be a number")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("volume_db must be finite")
        return number

    @field_validator(
        "offset_seconds",
        "fade_in_seconds",
        "fade_out_seconds",
        "duration_seconds",
        mode="before",
    )
    @classmethod
    def _validate_seconds(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("time values must be numbers")
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ValueError("time values must be finite and non-negative")
        return number

    @model_validator(mode="before")
    @classmethod
    def _infer_role(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        explicit = data.get("is_voice", data.get("voice"))
        if explicit is True:
            data.setdefault("role", AudioTrackRole.VOICE)
        elif explicit is False:
            data.setdefault("role", AudioTrackRole.MUSIC)
        return data

    @model_validator(mode="after")
    def _validate_duration(self) -> Self:
        fade_out = self.fade_out_seconds
        if (
            fade_out > 0
            and self.duration_seconds is not None
            and fade_out > self.duration_seconds
        ):
            raise ValueError("fade_out_seconds must not exceed duration_seconds")
        return self

    @property
    def voice(self) -> bool:
        """Whether this track is a voice that drives ducking."""
        if self.is_voice is not None:
            return self.is_voice
        return self.role.is_voice

    @property
    def should_duck(self) -> bool:
        """Whether this non-voice track may be ducked under a voice."""
        if self.voice:
            return False
        if self.duck is not None:
            return self.duck
        return self.role in (
            AudioTrackRole.MUSIC,
            AudioTrackRole.BACKGROUND,
        )

    @property
    def volume(self) -> float:
        """Alias for :attr:`volume_db`."""
        return self.volume_db

    @property
    def offset(self) -> float:
        """Alias for :attr:`offset_seconds`."""
        return self.offset_seconds

    @classmethod
    def from_value(
        cls, value: AudioTrack | Mapping[str, Any] | AudioSource
    ) -> AudioTrack:
        """Coerce a path, bytes, mapping or track into a track."""
        if isinstance(value, cls):
            return value
        if isinstance(value, (str, Path, bytes, bytearray, memoryview)):
            return cls(
                source=bytes(value)
                if isinstance(value, (bytearray, memoryview))
                else value
            )
        if isinstance(value, Mapping):
            return cls.model_validate(dict(value))
        raise TypeError("track must be an AudioTrack, mapping, path or bytes")


class DuckingSettings(BaseModel):
    """Sidechain compression that lowers music while a voice is present."""

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
        populate_by_name=True,
    )

    enabled: bool = Field(default=True, validation_alias=AliasChoices("enabled", "on"))
    threshold_db: float = Field(
        default=DEFAULT_DUCKING_THRESHOLD_DB,
        validation_alias=AliasChoices(
            "threshold_db",
            "threshold",
            "duck_threshold_db",
        ),
    )
    ratio: float = Field(
        default=DEFAULT_DUCKING_RATIO,
        validation_alias=AliasChoices("ratio", "duck_ratio"),
    )
    attack_ms: float = Field(
        default=DEFAULT_DUCKING_ATTACK_MS,
        validation_alias=AliasChoices("attack_ms", "attack"),
    )
    release_ms: float = Field(
        default=DEFAULT_DUCKING_RELEASE_MS,
        validation_alias=AliasChoices("release_ms", "release"),
    )

    @field_validator("threshold_db", "ratio", "attack_ms", "release_ms", mode="before")
    @classmethod
    def _validate_number(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("ducking values must be numbers")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("ducking values must be finite")
        return number

    @model_validator(mode="after")
    def _validate_ranges(self) -> Self:
        ceiling = self.threshold_db
        compression = self.ratio
        if ceiling > 0:
            raise ValueError("threshold_db must not be positive")
        if compression < 1:
            raise ValueError("ratio must be at least 1")
        if self.attack_ms < 0 or self.release_ms < 0:
            raise ValueError("attack_ms and release_ms must not be negative")
        return self

    @property
    def threshold(self) -> float:
        """The sidechain threshold as a linear amplitude."""
        linear = 10.0 ** (float(self.threshold_db) / 20.0)
        return float(min(1.0, max(0.000976563, linear)))

    @classmethod
    def from_value(
        cls,
        value: DuckingSettings | Mapping[str, Any] | None,
    ) -> DuckingSettings:
        """Coerce ``None``, a mapping or an instance into settings."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, BaseModel):
            return cls.model_validate(value.model_dump())
        if isinstance(value, Mapping):
            return cls.model_validate(dict(value))
        raise TypeError("ducking must be DuckingSettings, a mapping or None")


class AudioMixSettings(BaseModel):
    """Output and normalization settings for one mix."""

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
        populate_by_name=True,
    )

    duration_seconds: float | None = Field(
        default=None,
        validation_alias=AliasChoices("duration_seconds", "duration"),
    )
    normalize_loudness: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "normalize_loudness",
            "normalize",
            "normalize_audio",
        ),
    )
    target_lufs: float = Field(
        default=-14.0,
        validation_alias=AliasChoices(
            "target_lufs", "loudness_lufs", "target_loudness"
        ),
    )
    true_peak_db: float = Field(
        default=-1.5,
        validation_alias=AliasChoices("true_peak_db", "true_peak"),
    )
    loudness_range: float = Field(
        default=11.0,
        validation_alias=AliasChoices("loudness_range", "lra"),
    )
    sample_rate: int = Field(
        default=DEFAULT_MIX_SAMPLE_RATE,
        gt=0,
        validation_alias=AliasChoices("sample_rate", "rate"),
    )
    channels: int = Field(
        default=DEFAULT_MIX_CHANNELS,
        gt=0,
        le=8,
        validation_alias=AliasChoices("channels", "channel_count"),
    )
    format: AudioFormat | None = None
    ducking: DuckingSettings = Field(default_factory=DuckingSettings)

    @field_validator("duration_seconds", mode="before")
    @classmethod
    def _validate_duration(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("duration_seconds must be a number")
        number = float(value)
        if not math.isfinite(number) or number <= 0:
            raise ValueError("duration_seconds must be finite and positive")
        return number

    @field_validator(
        "target_lufs",
        "true_peak_db",
        "loudness_range",
        mode="before",
    )
    @classmethod
    def _validate_loudness(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("loudness values must be numbers")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("loudness values must be finite")
        return number

    @model_validator(mode="after")
    def _validate_ranges(self) -> Self:
        if self.true_peak_db > 0:
            raise ValueError("true_peak_db must not be positive")
        dynamic_range = self.loudness_range
        if dynamic_range < 0:
            raise ValueError("loudness_range must not be negative")
        return self

    @property
    def duration(self) -> float | None:
        """Alias for :attr:`duration_seconds`."""
        return self.duration_seconds

    @classmethod
    def from_value(
        cls,
        value: AudioMixSettings | Mapping[str, Any] | None,
    ) -> AudioMixSettings:
        """Coerce ``None``, a mapping or an instance into settings."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, BaseModel):
            return cls.model_validate(value.model_dump())
        if isinstance(value, Mapping):
            return cls.model_validate(dict(value))
        raise TypeError("settings must be AudioMixSettings, a mapping or None")


@dataclass(frozen=True, slots=True)
class DuckingVerification:
    """Whether a mix graph actually applies sidechain ducking."""

    applied: bool = False
    verified: bool = False
    voice_tracks: int = 0
    background_tracks: int = 0
    threshold_db: float | None = None
    ratio: float | None = None
    attack_ms: float | None = None
    release_ms: float | None = None
    reason: str = ""

    def require(self) -> DuckingVerification:
        """Return this verification, raising when ducking is not verified."""
        if not self.verified:
            raise ValidationError(
                f"voice ducking is not verified: {self.reason or 'missing sidechain'}"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for logs."""
        return {
            "applied": self.applied,
            "verified": self.verified,
            "voice_tracks": self.voice_tracks,
            "background_tracks": self.background_tracks,
            "threshold_db": self.threshold_db,
            "ratio": self.ratio,
            "attack_ms": self.attack_ms,
            "release_ms": self.release_ms,
            "reason": self.reason,
        }


DuckingReport = DuckingVerification
"""Alias for :class:`DuckingVerification`."""


@dataclass(frozen=True, slots=True)
class AudioMixResult:
    """The written mix, the graph used and the ducking verification."""

    path: Path
    tracks: tuple[AudioTrack, ...]
    settings: AudioMixSettings
    filtergraph: str
    argv: tuple[str, ...]
    ducking: DuckingVerification

    @property
    def destination(self) -> Path:
        """Alias for :attr:`path`."""
        return self.path

    @property
    def graph(self) -> str:
        """Alias for :attr:`filtergraph`."""
        return self.filtergraph

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for logs."""
        return {
            "path": str(self.path),
            "filtergraph": self.filtergraph,
            "argv": list(self.argv),
            "ducking": self.ducking.to_dict(),
            "tracks": len(self.tracks),
        }


def _format_value(value: AudioFormat | str | None) -> str | None:
    if value is None:
        return None
    text = value.value if isinstance(value, AudioFormat) else str(value)
    return text.lower().lstrip(".") or None


def _suffix(destination: Path, settings: AudioMixSettings) -> str:
    explicit = _format_value(settings.format)
    if explicit is not None:
        return explicit
    suffix = destination.suffix.lstrip(".").lower()
    if not suffix:
        raise ValidationError(
            f"cannot infer an audio format from {destination.name!r}; "
            "pass an explicit format"
        )
    return suffix


def _track_filters(
    track: AudioTrack,
    index: int,
    settings: AudioMixSettings,
    *,
    input_offset: int = 0,
) -> str:
    filters: list[str] = []
    duration = track.duration_seconds
    if duration is not None:
        filters.extend(
            [
                f"atrim=duration={duration:.6f}",
                "asetpts=PTS-STARTPTS",
            ]
        )
    if track.volume_db != 0:
        filters.append(f"volume={track.volume_db:.6f}dB")
    fade_in = track.fade_in_seconds
    fade_out = track.fade_out_seconds
    delay = track.offset_seconds
    if fade_in > 0:
        filters.append(f"afade=t=in:st=0:d={fade_in:.6f}")
    if fade_out > 0:
        if duration is not None:
            start = max(0.0, duration - fade_out)
            filters.append(f"afade=t=out:st={start:.6f}:d={fade_out:.6f}")
        else:
            filters.extend(
                [
                    "areverse",
                    f"afade=t=in:st=0:d={fade_out:.6f}",
                    "areverse",
                ]
            )
    if delay > 0:
        milliseconds = round(delay * 1000.0)
        filters.append(f"adelay={milliseconds}:all=1")
    if not filters:
        filters.append("anull")
    return f"[{index + input_offset}:a]{','.join(filters)}[a{index}]"


def _bus(parts: list[str], labels: Sequence[str], name: str) -> str:
    if len(labels) == 1:
        parts.append(f"[{labels[0]}]anull[{name}]")
        return name
    joined = "".join(f"[{label}]" for label in labels)
    parts.append(
        f"{joined}amix=inputs={len(labels)}:duration=longest:"
        f"dropout_transition=0:normalize=0[{name}]"
    )
    return name


def build_mix_filtergraph(
    tracks: Sequence[AudioTrack | Mapping[str, Any] | AudioSource],
    settings: AudioMixSettings | Mapping[str, Any] | None = None,
    *,
    input_offset: int = 0,
) -> str:
    """Build the deterministic ffmpeg filtergraph for a list of tracks.

    Inputs are addressed in the same order as *tracks*. Voice tracks are mixed
    into a sidechain bus that drives ``sidechaincompress`` on the music bed;
    the voice bus is then mixed back in unchanged. *input_offset* shifts the
    input indices when the audio inputs follow other streams in one command.
    """
    active = AudioMixSettings.from_value(settings)
    values = [AudioTrack.from_value(track) for track in tracks]
    if not values:
        raise ValidationError("a mix needs at least one track")
    for track in values:
        if track.loop and active.duration_seconds is None:
            raise ValidationError("a looping track needs a mix duration")

    parts: list[str] = []
    labels: list[str] = []
    voices: list[str] = []
    backgrounds: list[str] = []
    for index, track in enumerate(values):
        parts.append(_track_filters(track, index, active, input_offset=input_offset))
        label = f"a{index}"
        labels.append(label)
        if track.voice:
            voices.append(label)
        elif track.should_duck:
            backgrounds.append(label)

    ducking = active.ducking
    if ducking.enabled and voices and backgrounds:
        voice_bus = _bus(parts, voices, "voicebus")
        bed = _bus(parts, backgrounds, "bed")
        parts.append(f"[{voice_bus}]asplit=2[voice_sc][voice_out]")
        parts.append(
            f"[{bed}][voice_sc]sidechaincompress="
            f"threshold={ducking.threshold:.6f}:"
            f"ratio={ducking.ratio:g}:"
            f"attack={ducking.attack_ms:g}:"
            f"release={ducking.release_ms:g}:"
            "makeup=1[bed_ducked]"
        )
        parts.append(
            "[bed_ducked][voice_out]amix=inputs=2:duration=longest:"
            "dropout_transition=0:normalize=0[mixed]"
        )
    else:
        _bus(parts, labels, "mixed")

    post: list[str] = []
    if active.duration_seconds is not None:
        post.append(f"atrim=duration={active.duration_seconds:.6f}")
    if active.normalize_loudness:
        post.append(
            f"loudnorm=I={active.target_lufs:.6f}:"
            f"TP={active.true_peak_db:.6f}:"
            f"LRA={active.loudness_range:.6f}"
        )
    if post:
        parts.append(f"[mixed]{','.join(post)}[out]")
    else:
        parts.append("[mixed]anull[out]")
    return ";".join(parts)


def _encode_args(fmt: str, settings: AudioMixSettings) -> list[str]:
    codec = _CODECS.get(fmt)
    arguments: list[str] = ["-vn"]
    if codec is not None:
        arguments.extend(["-c:a", codec])
    arguments.extend(["-ar", str(settings.sample_rate)])
    arguments.extend(["-ac", str(settings.channels)])
    return arguments


def _same_file(source: Path, destination: Path) -> bool:
    if source.resolve() == destination.resolve():
        return True
    try:
        return destination.exists() and os.path.samefile(source, destination)
    except OSError:
        return False


def _check_destination(sources: Sequence[Path], destination: Path) -> None:
    for source in sources:
        if _same_file(source, destination):
            raise ValidationError("destination must not overwrite an input track")


def _run_to_file(
    argv: Sequence[str],
    target: Path,
    *,
    runner: FFmpegRunner | None,
    timeout_seconds: float,
) -> None:
    completed = _run_ffmpeg(argv, runner=runner, timeout_seconds=timeout_seconds)
    if completed.returncode != 0:
        raise ValidationError(
            "ffmpeg could not mix the audio",
            details={
                "returncode": completed.returncode,
                "stderr": _output_text(completed.stderr)[:500],
            },
        )
    if not target.is_file() or target.stat().st_size == 0:
        payload = completed.stdout
        if isinstance(payload, (bytes, bytearray, memoryview)):
            target.write_bytes(bytes(payload))
        elif isinstance(payload, Path) and payload.is_file():
            target.write_bytes(payload.read_bytes())
        elif isinstance(payload, str) and payload:
            target.write_text(payload, encoding="utf-8")
    if not target.is_file() or target.stat().st_size == 0:
        raise ValidationError(
            "ffmpeg produced no mixed audio",
            details={"target": str(target)},
        )


def verify_ducking(
    value: AudioMixResult | str,
    ducking: DuckingSettings | Mapping[str, Any] | None = None,
    *,
    voice_tracks: int | None = None,
    background_tracks: int | None = None,
) -> DuckingVerification:
    """Verify that a graph applies sidechain ducking where it is needed.

    *value* may be a mix result or a raw filtergraph. When voice and background
    tracks coexist and ducking is enabled, the graph must contain a
    ``sidechaincompress`` node and its parameters must match the settings.
    """
    if isinstance(value, AudioMixResult):
        graph = value.filtergraph
        active = (
            value.settings.ducking
            if ducking is None
            else DuckingSettings.from_value(ducking)
        )
        if voice_tracks is None:
            voice_tracks = sum(1 for track in value.tracks if track.voice)
        if background_tracks is None:
            background_tracks = sum(1 for track in value.tracks if track.should_duck)
    else:
        graph = str(value)
        active = DuckingSettings.from_value(ducking)

    voices = 0 if voice_tracks is None else max(0, voice_tracks)
    backgrounds = 0 if background_tracks is None else max(0, background_tracks)
    match = _SIDECHAIN.search(graph)
    applied = match is not None
    if not active.enabled:
        return DuckingVerification(
            applied=applied,
            verified=not applied,
            voice_tracks=voices,
            background_tracks=backgrounds,
            reason="ducking is disabled",
        )
    if not voices or not backgrounds:
        return DuckingVerification(
            applied=applied,
            verified=not applied,
            voice_tracks=voices,
            background_tracks=backgrounds,
            reason="ducking needs both a voice and a background track",
        )
    if match is None:
        return DuckingVerification(
            applied=False,
            verified=False,
            voice_tracks=voices,
            background_tracks=backgrounds,
            reason="sidechaincompress is missing from the graph",
        )
    threshold = float(match.group(1))
    ratio = float(match.group(2))
    attack = float(match.group(3))
    release = float(match.group(4))
    matches = (
        math.isclose(threshold, active.threshold, rel_tol=1e-6, abs_tol=1e-6)
        and math.isclose(ratio, active.ratio, rel_tol=1e-6, abs_tol=1e-6)
        and math.isclose(attack, active.attack_ms, rel_tol=1e-6, abs_tol=1e-6)
        and math.isclose(release, active.release_ms, rel_tol=1e-6, abs_tol=1e-6)
    )
    reason = "" if matches else "sidechain parameters do not match the settings"
    return DuckingVerification(
        applied=True,
        verified=matches,
        voice_tracks=voices,
        background_tracks=backgrounds,
        threshold_db=active.threshold_db,
        ratio=active.ratio,
        attack_ms=active.attack_ms,
        release_ms=active.release_ms,
        reason=reason,
    )


def mix_audio(
    tracks: Sequence[AudioTrack | Mapping[str, Any] | AudioSource],
    destination: str | Path,
    settings: AudioMixSettings | Mapping[str, Any] | None = None,
    *,
    duration_seconds: float | None = None,
    normalize_loudness: bool | None = None,
    target_lufs: float | None = None,
    true_peak_db: float | None = None,
    sample_rate: int | None = None,
    channels: int | None = None,
    output_format: str | AudioFormat | None = None,
    ducking: DuckingSettings | Mapping[str, Any] | None = None,
    ffmpeg_executable: str = "ffmpeg",
    ffmpeg_runner: FFmpegRunner | None = None,
    timeout_seconds: float = DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS,
) -> AudioMixResult:
    """Mix tracks into a new file and verify voice ducking on the graph.

    Each track is materialized in order, so inline bytes work without a caller
    writing a temporary file. The source files are never modified.
    """
    path = Path(destination)
    values = [AudioTrack.from_value(track) for track in tracks]
    if not values:
        raise ValidationError("a mix needs at least one track")

    overrides: dict[str, Any] = {}
    if duration_seconds is not None:
        overrides["duration_seconds"] = duration_seconds
    if normalize_loudness is not None:
        overrides["normalize_loudness"] = normalize_loudness
    if target_lufs is not None:
        overrides["target_lufs"] = target_lufs
        overrides.setdefault("normalize_loudness", True)
    if true_peak_db is not None:
        overrides["true_peak_db"] = true_peak_db
    if sample_rate is not None:
        overrides["sample_rate"] = sample_rate
    if channels is not None:
        overrides["channels"] = channels
    if output_format is not None:
        overrides["format"] = output_format
    if ducking is not None:
        overrides["ducking"] = DuckingSettings.from_value(ducking)
    base = AudioMixSettings.from_value(settings)
    active = base.model_copy(update=overrides) if overrides else base

    graph = build_mix_filtergraph(values, active)
    fmt = _suffix(path, active)
    path.parent.mkdir(parents=True, exist_ok=True)

    with ExitStack() as stack:
        paths = [stack.enter_context(_materialize(track.source)) for track in values]
        _check_destination(paths, path)
        descriptor, name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=f".{fmt}",
        )
        os.close(descriptor)
        temporary = Path(name)
        try:
            argv = _base(ffmpeg_executable)
            for track, source in zip(values, paths, strict=True):
                if track.loop:
                    argv.extend(["-stream_loop", "-1"])
                argv.extend(["-i", str(source)])
            argv.extend(["-filter_complex", graph, "-map", "[out]"])
            argv.extend(_encode_args(fmt, active))
            argv.append(str(temporary))
            _run_to_file(
                argv,
                temporary,
                runner=ffmpeg_runner,
                timeout_seconds=timeout_seconds,
            )
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    verification = verify_ducking(
        graph,
        active.ducking,
        voice_tracks=sum(1 for track in values if track.voice),
        background_tracks=sum(1 for track in values if track.should_duck),
    )
    return AudioMixResult(
        path=path,
        tracks=tuple(values),
        settings=active,
        filtergraph=graph,
        argv=tuple(argv),
        ducking=verification,
    )


mix_tracks = mix_audio
