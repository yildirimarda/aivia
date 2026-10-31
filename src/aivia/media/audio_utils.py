"""Deterministic audio transforms for generated speech.

Speech adapters return encoded bytes with no reliable duration and no guarantee
about loudness. Before a voice-over is stored or composed it is trimmed, faded
and normalized to a target loudness, and its real duration is measured from the
encoded payload rather than trusted from provider metadata.

Every transform writes a new file through a temporary path and never edits the
source. ffmpeg and ffprobe are invoked with argv lists and no shell, and both
runners are injectable so the offline suite needs neither binary.
"""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

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
)
from aivia.core.errors import ValidationError
from aivia.core.media_probe import FFprobeRunner, run_ffprobe, sniff_media
from aivia.providers.protocols.audio import AudioFormat

__all__ = [
    "AudioProcessing",
    "AudioSettings",
    "AudioSource",
    "FFmpegRunner",
    "SpeechSettings",
    "audio_duration",
    "fade_audio",
    "measure_audio_duration",
    "normalize_audio",
    "normalize_loudness",
    "probe_audio_duration",
    "process_audio",
    "trim_audio",
]

type AudioSource = str | Path | bytes
"""An audio path or the raw bytes of an encoded payload."""

FFmpegRunner = Any
"""A callable that runs an ffmpeg argv and returns its completed process."""


class AudioSettings(BaseModel):
    """The post-processing applied to one synthesized voice-over.

    Loudness normalization is off by default because a local synthesizer may
    already produce a suitable level; enable it to target a delivery platform.
    Trimming and fades are expressed in seconds and are independent of it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    normalize_loudness: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "normalize_loudness",
            "normalize",
            "loudness_normalization",
            "normalize_audio",
        ),
    )
    target_lufs: float = Field(
        default=-14.0,
        validation_alias=AliasChoices(
            "target_lufs",
            "loudness_lufs",
            "loudness_target",
            "target_loudness",
        ),
    )
    true_peak_db: float = Field(
        default=-1.5,
        validation_alias=AliasChoices(
            "true_peak_db",
            "true_peak",
            "max_true_peak",
        ),
    )
    loudness_range: float = Field(
        default=11.0,
        validation_alias=AliasChoices(
            "loudness_range",
            "lra",
        ),
    )
    trim_start_seconds: float = Field(
        default=0.0,
        validation_alias=AliasChoices(
            "trim_start_seconds",
            "trim_start",
            "start_seconds",
        ),
    )
    trim_end_seconds: float | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "trim_end_seconds",
            "trim_end",
            "end_seconds",
        ),
    )
    fade_in_seconds: float = Field(
        default=0.0,
        validation_alias=AliasChoices(
            "fade_in_seconds",
            "fade_in",
        ),
    )
    fade_out_seconds: float = Field(
        default=0.0,
        validation_alias=AliasChoices(
            "fade_out_seconds",
            "fade_out",
        ),
    )
    format: AudioFormat | None = None
    sample_rate: int | None = Field(default=None, gt=0)
    channels: int | None = Field(default=None, gt=0, le=8)

    @field_validator(
        "target_lufs",
        "true_peak_db",
        "loudness_range",
        mode="before",
    )
    @classmethod
    def _validate_number(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("loudness values must be numbers")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("loudness values must be finite")
        return number

    @field_validator(
        "trim_start_seconds",
        "trim_end_seconds",
        "fade_in_seconds",
        "fade_out_seconds",
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

    @model_validator(mode="after")
    def _validate_range(self) -> AudioSettings:
        if (
            self.trim_end_seconds is not None
            and self.trim_end_seconds <= self.trim_start_seconds
        ):
            raise ValueError("trim_end_seconds must be greater than trim_start_seconds")
        if self.true_peak_db > 0:
            raise ValueError("true_peak_db must not be positive")
        if _negative(self.loudness_range):
            raise ValueError("loudness_range must not be negative")
        return self

    @property
    def loudness_lufs(self) -> float:
        """Alias for :attr:`target_lufs`."""
        return self.target_lufs

    @property
    def normalize(self) -> bool:
        """Alias for :attr:`normalize_loudness`."""
        return self.normalize_loudness

    @property
    def trim_start(self) -> float:
        """Alias for :attr:`trim_start_seconds`."""
        return self.trim_start_seconds

    @property
    def trim_end(self) -> float | None:
        """Alias for :attr:`trim_end_seconds`."""
        return self.trim_end_seconds

    @property
    def fade_in(self) -> float:
        """Alias for :attr:`fade_in_seconds`."""
        return self.fade_in_seconds

    @property
    def fade_out(self) -> float:
        """Alias for :attr:`fade_out_seconds`."""
        return self.fade_out_seconds

    @property
    def has_processing(self) -> bool:
        """Whether applying this settings object changes the audio."""
        return bool(
            self.normalize_loudness
            or _positive(self.trim_start_seconds)
            or self.trim_end_seconds is not None
            or _positive(self.fade_in_seconds)
            or _positive(self.fade_out_seconds)
            or self.sample_rate is not None
            or self.channels is not None
            or self.format is not None
        )

    def cache_parameters(self) -> dict[str, Any]:
        """Return the settings as a stable cache-key parameter mapping."""
        return {
            "normalize_loudness": self.normalize_loudness,
            "target_lufs": self.target_lufs,
            "true_peak_db": self.true_peak_db,
            "loudness_range": self.loudness_range,
            "trim_start_seconds": self.trim_start_seconds,
            "trim_end_seconds": self.trim_end_seconds,
            "fade_in_seconds": self.fade_in_seconds,
            "fade_out_seconds": self.fade_out_seconds,
            "format": self.format.value if self.format is not None else None,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
        }

    @classmethod
    def from_value(cls, value: object | None) -> AudioSettings:
        """Coerce ``None``, a mapping or an instance into settings."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, BaseModel):
            return cls.model_validate(value.model_dump())
        if isinstance(value, Mapping):
            return cls.model_validate(dict(value))
        raise TypeError("settings must be an AudioSettings, mapping or None")


SpeechSettings = AudioSettings
"""Alias for :class:`AudioSettings` used by the speech service."""

AudioProcessing = AudioSettings
"""Alias for :class:`AudioSettings`."""


def _positive(value: float) -> bool:
    return value > 0


def _negative(value: float) -> bool:
    return value < 0


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


def _suffix(value: str | Path | None, fallback: str = "audio") -> str:
    if value is None:
        return fallback
    suffix = Path(value).suffix.lstrip(".").lower()
    return suffix or fallback


@contextmanager
def _materialize(
    source: AudioSource,
    *,
    prefix: str = "aivia-audio-",
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
        return subprocess.CompletedProcess(
            list(argv),
            0,
            stdout=result,
            stderr=b"",
        )
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


def _materialize_result(result: Any, target: Path) -> None:
    if result is None:
        return
    if isinstance(result, (bytes, bytearray, memoryview)):
        target.write_bytes(bytes(result))
        return
    if isinstance(result, Path):
        if result.is_file() and result != target:
            target.write_bytes(result.read_bytes())
        return
    if isinstance(result, str):
        candidate = Path(result)
        if candidate.is_file() and candidate != target:
            target.write_bytes(candidate.read_bytes())
        else:
            target.write_text(result, encoding="utf-8")


def _run_to_file(
    argv: Sequence[str],
    target: Path,
    *,
    runner: FFmpegRunner | None,
    timeout_seconds: float,
    failure: str,
) -> Path:
    completed = _run_ffmpeg(argv, runner=runner, timeout_seconds=timeout_seconds)
    if completed.returncode != 0:
        raise ValidationError(
            failure,
            details={
                "returncode": completed.returncode,
                "stderr": _output_text(completed.stderr)[:500],
            },
        )
    if not target.is_file() or target.stat().st_size == 0:
        _materialize_result(completed.stdout, target)
    if not target.is_file() or target.stat().st_size == 0:
        raise ValidationError(
            f"{failure}: ffmpeg produced no output",
            details={"target": str(target)},
        )
    return target


def _base(executable: str) -> list[str]:
    return [executable, "-hide_banner", "-loglevel", "error", "-y"]


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


def _encode_args(fmt: str, settings: AudioSettings) -> list[str]:
    codec = _CODECS.get(fmt)
    arguments: list[str] = ["-vn"]
    if codec is not None:
        arguments.extend(["-c:a", codec])
    if settings.sample_rate is not None:
        arguments.extend(["-ar", str(settings.sample_rate)])
    if settings.channels is not None:
        arguments.extend(["-ac", str(settings.channels)])
    return arguments


def _render(
    source: AudioSource,
    destination: str | Path,
    filters: Sequence[str],
    settings: AudioSettings,
    *,
    output_format: str | AudioFormat | None,
    ffmpeg_executable: str,
    ffmpeg_runner: FFmpegRunner | None,
    timeout_seconds: float,
    failure: str,
) -> Path:
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    fmt = (
        (
            output_format.value
            if isinstance(output_format, AudioFormat)
            else (output_format or _suffix(target, "wav"))
        )
        .lower()
        .lstrip(".")
    )
    with _materialize(source) as active:
        descriptor, name = tempfile.mkstemp(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=f".{fmt}",
        )
        os.close(descriptor)
        temporary = Path(name)
        try:
            argv = _base(ffmpeg_executable)
            argv.extend(["-i", str(active)])
            if filters:
                argv.extend(["-af", ",".join(filters)])
            argv.extend(_encode_args(fmt, settings))
            argv.append(str(temporary))
            _run_to_file(
                argv,
                temporary,
                runner=ffmpeg_runner,
                timeout_seconds=timeout_seconds,
                failure=failure,
            )
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    return target


def _trim_filters(
    start_seconds: float,
    end_seconds: float | None,
    duration_seconds: float | None,
) -> list[str]:
    start = _number(start_seconds, label="start_seconds", non_negative=True)
    end: float | None = None
    if end_seconds is not None and duration_seconds is not None:
        raise ValidationError("pass either end_seconds or duration_seconds, not both")
    if end_seconds is not None:
        end = _number(end_seconds, label="end_seconds", positive=True)
        if end <= start:
            raise ValidationError("end_seconds must be greater than start_seconds")
    elif duration_seconds is not None:
        length = _number(duration_seconds, label="duration_seconds", positive=True)
        end = start + length
    if end is None and start == 0.0:
        return []
    expression = f"atrim=start={start:.6f}"
    if end is not None:
        expression += f":end={end:.6f}"
    return [expression, "asetpts=PTS-STARTPTS"]


def _fade_filters(
    fade_in_seconds: float,
    fade_out_seconds: float,
    *,
    duration_seconds: float | None,
) -> list[str]:
    fade_in = _number(fade_in_seconds, label="fade_in_seconds", non_negative=True)
    fade_out = _number(fade_out_seconds, label="fade_out_seconds", non_negative=True)
    filters: list[str] = []
    if fade_in > 0:
        filters.append(f"afade=t=in:st=0:d={fade_in:.6f}")
    if fade_out > 0:
        if duration_seconds is not None:
            length = _number(
                duration_seconds,
                label="duration_seconds",
                positive=True,
            )
            start = max(0.0, length - fade_out)
            filters.append(f"afade=t=out:st={start:.6f}:d={fade_out:.6f}")
        else:
            filters.append(f"areverse,afade=t=in:st=0:d={fade_out:.6f},areverse")
    return filters


def _loudness_filter(settings: AudioSettings) -> list[str]:
    if not settings.normalize_loudness:
        return []
    return [
        "loudnorm="
        f"I={settings.target_lufs:.6f}:"
        f"TP={settings.true_peak_db:.6f}:"
        f"LRA={settings.loudness_range:.6f}"
    ]


def measure_audio_duration(
    source: AudioSource,
    *,
    format: str | AudioFormat | None = None,
    ffprobe_executable: str = "ffprobe",
    ffprobe_runner: FFprobeRunner | None = None,
    timeout_seconds: float = DEFAULT_MEDIA_FFPROBE_TIMEOUT_SECONDS,
) -> float:
    """Measure the duration of *source* with ffprobe.

    Inline bytes are written to a temporary file whose suffix comes from the
    sniffed container, so ffprobe sees the same extension a stored asset has.
    """
    fmt = format.value if isinstance(format, AudioFormat) else format
    with _materialize(source) as active:
        target = active
        temporary: Path | None = None
        if fmt is not None:
            suffix = str(fmt).lower().lstrip(".")
            if suffix and active.suffix.lower().lstrip(".") != suffix:
                with tempfile.TemporaryDirectory(
                    prefix="aivia-audio-probe-"
                ) as directory:
                    temporary = Path(directory) / f"source.{suffix}"
                    temporary.write_bytes(active.read_bytes())
                    target = temporary
        info = run_ffprobe(
            target,
            executable=ffprobe_executable,
            timeout_seconds=timeout_seconds,
            runner=ffprobe_runner,
        )
    duration = info.duration_seconds
    if duration is None:
        raise ValidationError("ffprobe reported no audio duration")
    return float(duration)


audio_duration = measure_audio_duration
probe_audio_duration = measure_audio_duration


def normalize_audio(
    source: AudioSource,
    destination: str | Path,
    *,
    target_lufs: float = -14.0,
    true_peak_db: float = -1.5,
    loudness_range: float = 11.0,
    settings: AudioSettings | None = None,
    output_format: str | AudioFormat | None = None,
    ffmpeg_executable: str = "ffmpeg",
    ffmpeg_runner: FFmpegRunner | None = None,
    timeout_seconds: float = DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS,
) -> Path:
    """Write *source* normalized to *target_lufs* and a true peak ceiling."""
    active = settings or AudioSettings(
        normalize_loudness=True,
        target_lufs=target_lufs,
        true_peak_db=true_peak_db,
        loudness_range=loudness_range,
    )
    if not active.normalize_loudness:
        active = active.model_copy(update={"normalize_loudness": True})
    return _render(
        source,
        destination,
        _loudness_filter(active),
        active,
        output_format=output_format,
        ffmpeg_executable=ffmpeg_executable,
        ffmpeg_runner=ffmpeg_runner,
        timeout_seconds=timeout_seconds,
        failure="ffmpeg could not normalize the audio",
    )


normalize_loudness = normalize_audio


def trim_audio(
    source: AudioSource,
    destination: str | Path,
    *,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
    duration_seconds: float | None = None,
    settings: AudioSettings | None = None,
    output_format: str | AudioFormat | None = None,
    ffmpeg_executable: str = "ffmpeg",
    ffmpeg_runner: FFmpegRunner | None = None,
    timeout_seconds: float = DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS,
) -> Path:
    """Write the requested section of *source* without editing the original."""
    active = settings or AudioSettings()
    filters = _trim_filters(start_seconds, end_seconds, duration_seconds)
    return _render(
        source,
        destination,
        filters,
        active,
        output_format=output_format,
        ffmpeg_executable=ffmpeg_executable,
        ffmpeg_runner=ffmpeg_runner,
        timeout_seconds=timeout_seconds,
        failure="ffmpeg could not trim the audio",
    )


def fade_audio(
    source: AudioSource,
    destination: str | Path,
    *,
    fade_in_seconds: float = 0.0,
    fade_out_seconds: float = 0.0,
    duration_seconds: float | None = None,
    settings: AudioSettings | None = None,
    output_format: str | AudioFormat | None = None,
    ffmpeg_executable: str = "ffmpeg",
    ffmpeg_runner: FFmpegRunner | None = None,
    timeout_seconds: float = DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS,
) -> Path:
    """Write *source* with a linear fade in and/or fade out."""
    active = settings or AudioSettings()
    filters = _fade_filters(
        fade_in_seconds,
        fade_out_seconds,
        duration_seconds=duration_seconds,
    )
    return _render(
        source,
        destination,
        filters,
        active,
        output_format=output_format,
        ffmpeg_executable=ffmpeg_executable,
        ffmpeg_runner=ffmpeg_runner,
        timeout_seconds=timeout_seconds,
        failure="ffmpeg could not fade the audio",
    )


def process_audio(
    source: AudioSource,
    destination: str | Path,
    settings: AudioSettings | None = None,
    *,
    duration_seconds: float | None = None,
    output_format: str | AudioFormat | None = None,
    ffmpeg_executable: str = "ffmpeg",
    ffmpeg_runner: FFmpegRunner | None = None,
    timeout_seconds: float = DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS,
) -> Path:
    """Apply trim, fades and loudness normalization in one ffmpeg pass.

    *duration_seconds* is the source length when known. It is only needed to
    place a fade out; without it the fade out is built with ``areverse`` so the
    transform stays deterministic.
    """
    active = AudioSettings.from_value(settings)
    filters = _trim_filters(
        active.trim_start_seconds,
        active.trim_end_seconds,
        None,
    )
    effective = duration_seconds
    if effective is not None:
        effective = max(0.0, effective - active.trim_start_seconds)
    if active.trim_end_seconds is not None:
        effective = max(
            0.0,
            active.trim_end_seconds - active.trim_start_seconds,
        )
    filters.extend(
        _fade_filters(
            active.fade_in_seconds,
            active.fade_out_seconds,
            duration_seconds=effective,
        )
    )
    filters.extend(_loudness_filter(active))
    return _render(
        source,
        destination,
        filters,
        active,
        output_format=output_format or active.format,
        ffmpeg_executable=ffmpeg_executable,
        ffmpeg_runner=ffmpeg_runner,
        timeout_seconds=timeout_seconds,
        failure="ffmpeg could not process the audio",
    )
