"""Deterministic video transforms that never edit the source clip.

Generated and user-supplied videos arrive in arbitrary containers, sizes,
frame rates and with or without an audio track. Delivery and composition need a
small, predictable set of operations before a clip is stored or published:

* trim to a time range without re-reading the whole source;
* change playback speed while keeping audio in sync;
* concatenate clips, optionally with a crossfade between them;
* extract a still frame or the audio track;
* turn a clip into a seamless loop;
* encode GIF, animated WebP or MP4 outputs, reducing quality and size until a
  byte cap is met.

Every operation writes a new file through a temporary path and refuses to use
the source as its destination, so an in-place edit cannot destroy the original.
ffmpeg is invoked with an argv list and no shell. Both the ffmpeg and ffprobe
runners are injectable, which keeps the offline suite free of the real binary.
"""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
import warnings
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from aivia.core.config.media import (
    DEFAULT_MEDIA_ANIMATED_FRAME_RATE as DEFAULT_MEDIA_ANIMATED_FRAME_RATE,
)
from aivia.core.config.media import (
    DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS as DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS,
)
from aivia.core.config.media import (
    DEFAULT_MEDIA_MIN_ANIMATED_FRAME_RATE as DEFAULT_MEDIA_MIN_ANIMATED_FRAME_RATE,
)
from aivia.core.errors import ValidationError
from aivia.core.media_probe import FFprobeInfo, FFprobeRunner, run_ffprobe, sniff_media

__all__ = [
    "DEFAULT_ANIMATED_FRAME_RATE",
    "DEFAULT_AUDIO_BITRATE_KBPS",
    "DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS",
    "DEFAULT_VIDEO_CRF",
    "DEFAULT_VIDEO_QUALITY",
    "AudioDroppedWarning",
    "FFmpegRunner",
    "VideoAudioDroppedWarning",
    "VideoAudioWarning",
    "VideoOutputFormat",
    "VideoSizeError",
    "VideoSource",
    "change_speed",
    "concat_videos",
    "concatenate_videos",
    "convert_video",
    "export_video",
    "extract_audio",
    "extract_frame",
    "make_seamless_loop",
    "seamless_loop",
    "speed_video",
    "trim",
    "trim_video",
]

type VideoSource = str | Path | bytes
"""A video path or the raw bytes of an encoded video."""

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
"""A callable that runs an ffmpeg argv and returns its completed process."""

DEFAULT_ANIMATED_FRAME_RATE = DEFAULT_MEDIA_ANIMATED_FRAME_RATE
"""Default frame rate for GIF and animated WebP output."""

DEFAULT_AUDIO_BITRATE_KBPS = 192
"""Default audio bitrate for re-encoded tracks, in kilobits per second."""

DEFAULT_VIDEO_CRF = 23
"""Default constant rate factor for H.264 output."""

DEFAULT_VIDEO_QUALITY = 80
"""Default quality for animated WebP and GIF palettes."""


class VideoOutputFormat(StrEnum):
    """Containers the utilities can write."""

    MP4 = "mp4"
    WEBP = "webp"
    GIF = "gif"

    @classmethod
    def parse(cls, value: str | VideoOutputFormat) -> VideoOutputFormat:
        """Return a format from a name, alias or file suffix."""
        if isinstance(value, cls):
            return value
        key = value.strip().lower().lstrip(".")
        aliases = {
            "m4v": cls.MP4,
            "webp_anim": cls.WEBP,
        }
        if key in aliases:
            return aliases[key]
        try:
            return cls(key)
        except ValueError as exc:
            allowed = ", ".join(item.value for item in cls)
            raise ValidationError(
                f"unsupported video format {value!r}; expected one of {allowed}"
            ) from exc

    @classmethod
    def from_path(cls, path: Path) -> VideoOutputFormat:
        """Return the format implied by a destination file suffix."""
        if not path.suffix:
            raise ValidationError(
                f"cannot infer a video format from {path.name!r}; "
                "pass an explicit format"
            )
        return cls.parse(path.suffix)

    @property
    def extension(self) -> str:
        """The file suffix without a leading dot."""
        return self.value

    @property
    def supports_audio(self) -> bool:
        """Whether the container can carry an audio track."""
        return self is VideoOutputFormat.MP4


class VideoSizeError(ValidationError):
    """No encode attempt fit the requested byte cap."""

    code = "video_size"


class VideoAudioWarning(UserWarning):
    """Audio was requested but the source has no audio track."""


AudioDroppedWarning = VideoAudioWarning
"""Alias for :class:`VideoAudioWarning`."""

VideoAudioDroppedWarning = VideoAudioWarning
"""Alias for :class:`VideoAudioWarning`."""


@dataclass(frozen=True, slots=True)
class _Attempt:
    """One encode attempt in a size-capped ladder."""

    crf: int
    quality: int
    fps: float | None
    scale: float = 1.0


def _number(
    value: float,
    *,
    label: str,
    positive: bool = False,
    non_negative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValidationError(f"{label} must be finite")
    if positive and number <= 0.0:
        raise ValidationError(f"{label} must be positive")
    if non_negative and number < 0.0:
        raise ValidationError(f"{label} must not be negative")
    return number


def _positive_int(value: int, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValidationError(f"{label} must be a positive integer")
    return value


def _integer(value: int, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{label} must be an integer")
    return value


def _crf(value: int | None) -> int:
    if value is None:
        return DEFAULT_VIDEO_CRF
    checked = _integer(value, label="crf")
    if not 0 <= checked <= 51:
        raise ValidationError("crf must be between 0 and 51")
    return checked


def _quality(value: int | None) -> int:
    if value is None:
        return DEFAULT_VIDEO_QUALITY
    checked = _integer(value, label="quality")
    if not 1 <= checked <= 100:
        raise ValidationError("quality must be between 1 and 100")
    return checked


def _audio_bitrate(value: int) -> int:
    return _positive_int(value, label="audio_bitrate_kbps")


def _same_file(source: Path, destination: Path) -> bool:
    if source.resolve() == destination.resolve():
        return True
    try:
        return destination.exists() and os.path.samefile(source, destination)
    except OSError:
        return False


def _check_destination(
    sources: Sequence[VideoSource],
    destination: Path,
) -> None:
    for source in sources:
        if isinstance(source, (str, Path)) and _same_file(Path(source), destination):
            raise ValidationError("destination must not overwrite the source video")


@contextmanager
def _materialize(
    source: VideoSource,
    *,
    prefix: str = "aivia-video-",
) -> Iterator[Path]:
    if isinstance(source, (bytes, bytearray, memoryview)):
        payload = bytes(source)
        if not payload:
            raise ValidationError("video source is empty")
        with tempfile.TemporaryDirectory(prefix=prefix) as directory:
            signature = sniff_media(payload)
            suffix = f".{signature.extension}" if signature is not None else ".mp4"
            path = Path(directory) / f"source{suffix}"
            path.write_bytes(payload)
            yield path
        return
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise ValidationError(
                f"video source does not exist: {path}",
                details={"path": str(path)},
            )
        yield path
        return
    raise TypeError("video source must be bytes, a path or a VideoResult")


@contextmanager
def _temp_output(destination: Path) -> Iterator[Path]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix or ".tmp"
    descriptor, name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=suffix,
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        yield temporary
    finally:
        temporary.unlink(missing_ok=True)


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


def _unique(values: Sequence[Any]) -> list[Any]:
    seen: set[Any] = set()
    result: list[Any] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _attempts(
    fmt: VideoOutputFormat,
    *,
    crf: int,
    quality: int,
    fps: float | None,
    capped: bool,
) -> list[_Attempt]:
    if fmt is VideoOutputFormat.MP4:
        default_fps = fps
        if not capped:
            return [_Attempt(crf=crf, quality=quality, fps=default_fps)]
        crfs = _unique([min(51, crf + step) for step in (0, 4, 8, 12, 16, 20)])
        attempts = [
            _Attempt(crf=value, quality=quality, fps=default_fps) for value in crfs
        ]
        attempts.extend(
            _Attempt(crf=crfs[-1], quality=quality, fps=default_fps, scale=scale)
            for scale in (0.75, 0.5, 0.35)
        )
        return attempts

    frame_rate = fps if fps is not None else DEFAULT_ANIMATED_FRAME_RATE
    if not capped:
        return [_Attempt(crf=crf, quality=quality, fps=frame_rate)]
    qualities = _unique([max(1, quality - step) for step in (0, 20, 40, 55, 65)])
    attempts = [_Attempt(crf=crf, quality=value, fps=frame_rate) for value in qualities]
    attempts.extend(
        _Attempt(crf=crf, quality=qualities[0], fps=frame_rate, scale=scale)
        for scale in (0.75, 0.5, 0.35)
    )
    if frame_rate > DEFAULT_MEDIA_MIN_ANIMATED_FRAME_RATE:
        attempts.append(_Attempt(crf=crf, quality=qualities[0], fps=frame_rate / 2.0))
    return attempts


def _scale_filter(
    width: int | None,
    height: int | None,
    scale: float,
) -> str | None:
    if scale == 1.0 and width is None and height is None:
        return None

    def scaled(value: int) -> str:
        return f"trunc({value}*{scale:.6f}/2)*2"

    if width is not None and height is not None:
        return f"scale={scaled(width)}:{scaled(height)}:flags=lanczos"
    if width is not None:
        return f"scale={scaled(width)}:-2:flags=lanczos"
    if height is not None:
        return f"scale=-2:{scaled(height)}:flags=lanczos"
    return f"scale=trunc(iw*{scale:.6f}/2)*2:trunc(ih*{scale:.6f}/2)*2:flags=lanczos"


def _post_filters(
    attempt: _Attempt,
    *,
    width: int | None,
    height: int | None,
) -> list[str]:
    filters: list[str] = []
    scale = _scale_filter(width, height, attempt.scale)
    if scale is not None:
        filters.append(scale)
    if attempt.fps is not None:
        filters.append(f"fps={attempt.fps:.6f}")
    return filters


def _gif_colors(quality: int) -> int:
    return max(2, min(256, round(256 * quality / 100)))


def _finalize_graph(
    parts: list[str],
    *,
    video_label: str,
    audio_label: str | None,
    fmt: VideoOutputFormat,
    quality: int,
) -> list[str]:
    if fmt is VideoOutputFormat.GIF:
        parts.append(
            f"[{video_label}]split[s0][s1];"
            f"[s0]palettegen=max_colors={_gif_colors(quality)}[p];"
            f"[s1][p]paletteuse=dither=bayer[b]"
        )
        video_label = "b"
    args = ["-filter_complex", ";".join(parts), "-map", f"[{video_label}]"]
    if audio_label is not None:
        args.extend(["-map", f"[{audio_label}]"])
    return args


def _simple_graph(
    fmt: VideoOutputFormat,
    filters: Sequence[str],
    *,
    quality: int,
) -> list[str]:
    chain = ",".join(filters)
    parts = [f"[0:v]{chain or 'null'}[v]"]
    return _finalize_graph(
        parts,
        video_label="v",
        audio_label=None,
        fmt=fmt,
        quality=quality,
    )


def _encode_args(
    fmt: VideoOutputFormat,
    attempt: _Attempt,
    *,
    audio: bool,
    loop: int,
    audio_bitrate_kbps: int,
) -> list[str]:
    if fmt is VideoOutputFormat.MP4:
        args = [
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(attempt.crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ]
        if audio:
            args.extend(["-c:a", "aac", "-b:a", f"{audio_bitrate_kbps}k"])
        else:
            args.append("-an")
        args.extend(["-f", "mp4"])
        return args
    if fmt is VideoOutputFormat.WEBP:
        return [
            "-c:v",
            "libwebp_anim",
            "-loop",
            str(loop),
            "-quality",
            str(attempt.quality),
            "-compression_level",
            "6",
            "-an",
            "-f",
            "webp",
        ]
    return ["-loop", str(loop), "-an", "-f", "gif"]


def _default_attempt(fmt: VideoOutputFormat) -> _Attempt:
    return _Attempt(
        crf=DEFAULT_VIDEO_CRF,
        quality=DEFAULT_VIDEO_QUALITY,
        fps=None if fmt is VideoOutputFormat.MP4 else DEFAULT_ANIMATED_FRAME_RATE,
    )


def _write_atomic(
    destination: Path,
    build_argv: Callable[[Path], Sequence[str]],
    *,
    runner: FFmpegRunner | None,
    timeout_seconds: float,
    failure: str,
) -> Path:
    with _temp_output(destination) as temporary:
        _run_to_file(
            build_argv(temporary),
            temporary,
            runner=runner,
            timeout_seconds=timeout_seconds,
            failure=failure,
        )
        os.replace(temporary, destination)
    return destination


def trim_video(
    source: VideoSource,
    destination: str | Path,
    *,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
    duration_seconds: float | None = None,
    output_format: str | VideoOutputFormat | None = None,
    ffmpeg_executable: str = "ffmpeg",
    ffmpeg_runner: FFmpegRunner | None = None,
    timeout_seconds: float = DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS,
) -> Path:
    """Write the section of *source* from *start_seconds* to the requested end.

    ``end_seconds`` and ``duration_seconds`` are mutually exclusive; when both
    are omitted the clip runs to the end of the source.
    """
    path = Path(destination)
    fmt = (
        VideoOutputFormat.from_path(path)
        if output_format is None
        else VideoOutputFormat.parse(output_format)
    )
    start = _number(start_seconds, label="start_seconds", non_negative=True)
    if end_seconds is not None and duration_seconds is not None:
        raise ValidationError("pass either end_seconds or duration_seconds, not both")
    length: float | None = None
    if end_seconds is not None:
        end = _number(end_seconds, label="end_seconds", positive=True)
        if end <= start:
            raise ValidationError("end_seconds must be greater than start_seconds")
        length = end - start
    elif duration_seconds is not None:
        length = _number(duration_seconds, label="duration_seconds", positive=True)
    _check_destination([source], path)
    attempt = _default_attempt(fmt)
    audio = fmt.supports_audio

    with _materialize(source, prefix="aivia-video-trim-") as active:

        def build(temporary: Path) -> list[str]:
            argv = _base(ffmpeg_executable)
            argv.extend(["-i", str(active), "-ss", f"{start:.6f}"])
            if length is not None:
                argv.extend(["-t", f"{length:.6f}"])
            argv.extend(_simple_graph(fmt, [], quality=attempt.quality))
            if audio:
                argv.extend(["-map", "0:a:0?"])
            argv.extend(
                _encode_args(
                    fmt,
                    attempt,
                    audio=audio,
                    loop=0,
                    audio_bitrate_kbps=DEFAULT_AUDIO_BITRATE_KBPS,
                )
            )
            argv.append(str(temporary))
            return argv

        return _write_atomic(
            path,
            build,
            runner=ffmpeg_runner,
            timeout_seconds=timeout_seconds,
            failure="ffmpeg could not trim the video",
        )


def _atempo_chain(speed: float) -> str:
    factors: list[float] = []
    remaining = speed
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    factors.append(remaining)
    return ",".join(f"atempo={factor:.6f}" for factor in factors)


def _has_audio(
    path: Path,
    *,
    executable: str,
    runner: FFprobeRunner | None,
    timeout_seconds: float,
) -> bool:
    info = run_ffprobe(
        path,
        executable=executable,
        timeout_seconds=timeout_seconds,
        runner=runner,
    )
    return bool(info.audio_streams)


def change_speed(
    source: VideoSource,
    destination: str | Path,
    speed: float,
    *,
    keep_audio: bool = True,
    output_format: str | VideoOutputFormat | None = None,
    ffmpeg_executable: str = "ffmpeg",
    ffmpeg_runner: FFmpegRunner | None = None,
    ffprobe_executable: str = "ffprobe",
    ffprobe_runner: FFprobeRunner | None = None,
    timeout_seconds: float = DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS,
) -> Path:
    """Change playback speed; ``2.0`` doubles and ``0.5`` halves it.

    Audio is stretched with an ``atempo`` chain so it stays in sync. Set
    ``keep_audio`` to false to drop the track instead. A source without an
    audio track is encoded silently and emits :class:`VideoAudioWarning`.
    """
    path = Path(destination)
    fmt = (
        VideoOutputFormat.from_path(path)
        if output_format is None
        else VideoOutputFormat.parse(output_format)
    )
    factor = _number(speed, label="speed", positive=True)
    _check_destination([source], path)
    attempt = _default_attempt(fmt)
    wants_audio = keep_audio and fmt.supports_audio
    filters = [f"setpts=(PTS-STARTPTS)/{factor:.6f}"]
    filters.extend(_post_filters(attempt, width=None, height=None))

    with _materialize(source, prefix="aivia-video-speed-") as active:
        audio = wants_audio
        if audio and not _has_audio(
            active,
            executable=ffprobe_executable,
            runner=ffprobe_runner,
            timeout_seconds=timeout_seconds,
        ):
            audio = False
            warnings.warn(
                "source has no audio track; audio was dropped",
                VideoAudioWarning,
                stacklevel=2,
            )

        def build(temporary: Path) -> list[str]:
            argv = _base(ffmpeg_executable)
            argv.extend(["-i", str(active)])
            argv.extend(_simple_graph(fmt, filters, quality=attempt.quality))
            if audio:
                argv.extend(
                    [
                        "-map",
                        "0:a:0?",
                        "-af",
                        _atempo_chain(factor),
                    ]
                )
            argv.extend(
                _encode_args(
                    fmt,
                    attempt,
                    audio=audio,
                    loop=0,
                    audio_bitrate_kbps=DEFAULT_AUDIO_BITRATE_KBPS,
                )
            )
            argv.append(str(temporary))
            return argv

        return _write_atomic(
            path,
            build,
            runner=ffmpeg_runner,
            timeout_seconds=timeout_seconds,
            failure="ffmpeg could not change the video speed",
        )


def _concat_entry(path: Path) -> str:
    escaped = str(path.resolve()).replace("'", "'\\''")
    return f"file '{escaped}'"


def _frame_rate(value: str | None) -> float | None:
    if not value:
        return None
    numerator, _, denominator = value.partition("/")
    try:
        rate = float(numerator)
        if denominator:
            rate /= float(denominator)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if not math.isfinite(rate) or rate <= 0:
        return None
    return rate


def _even_dimension(value: int) -> int:
    return max(2, value - value % 2)


def _video_geometry(
    info: FFprobeInfo,
) -> tuple[int | None, int | None, float | None, str | None]:
    stream = info.primary_video
    if stream is None:
        return None, None, None, None
    width = stream.width if stream.width and stream.width > 0 else None
    height = stream.height if stream.height and stream.height > 0 else None
    if width is None or height is None:
        width = height = None
    return width, height, _frame_rate(stream.frame_rate), stream.pixel_format


def _normalize_video_filter(
    geometry: tuple[int | None, int | None, float | None, str | None],
    target: tuple[int | None, int | None, float | None, str | None],
) -> str:
    width, height, frame_rate, pixel_format = geometry
    target_width, target_height, target_rate, target_pixel_format = target
    filters: list[str] = []
    if (
        target_width is not None
        and target_height is not None
        and (width != target_width or height != target_height)
    ):
        filters.extend(
            [
                f"scale={target_width}:{target_height}:"
                "force_original_aspect_ratio=decrease:flags=lanczos",
                f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2:color=black",
                "setsar=1",
            ]
        )
    if target_rate is not None and (
        frame_rate is None or abs(frame_rate - target_rate) > 1e-3
    ):
        filters.append(f"fps={target_rate:.6f}")
    if target_pixel_format is not None and pixel_format != target_pixel_format:
        filters.append(f"format=pix_fmts={target_pixel_format}")
    return ",".join(filters)


_AUDIO_CHANNEL_LAYOUTS = {
    1: "mono",
    2: "stereo",
    3: "2.1",
    4: "4.0",
    5: "5.0",
    6: "5.1",
    7: "6.1",
    8: "7.1",
}


def _audio_geometry(info: FFprobeInfo) -> tuple[int | None, int | None]:
    stream = info.audio_streams[0] if info.audio_streams else None
    if stream is None:
        return None, None
    sample_rate = (
        stream.sample_rate if stream.sample_rate and stream.sample_rate > 0 else None
    )
    channels = stream.channels if stream.channels and stream.channels > 0 else None
    return sample_rate, channels


def _normalize_audio_filter(
    geometry: tuple[int | None, int | None],
    target: tuple[int | None, int | None],
) -> str:
    sample_rate, channels = geometry
    target_rate, target_channels = target
    filters: list[str] = []
    if target_rate is not None and (sample_rate is None or sample_rate != target_rate):
        filters.append(f"aresample={target_rate}")
    if target_channels is not None and (
        channels is None or channels != target_channels
    ):
        layout = _AUDIO_CHANNEL_LAYOUTS.get(target_channels)
        if layout is not None:
            filters.append(f"aformat=channel_layouts={layout}")
    return ",".join(filters)


def _crossfade_graph(
    durations: Sequence[float],
    crossfade: float,
    *,
    with_audio: bool,
    fmt: VideoOutputFormat,
    quality: int,
    post_filters: Sequence[str],
    input_filters: Sequence[str] | None = None,
    audio_input_filters: Sequence[str] | None = None,
) -> list[str]:
    parts: list[str] = []
    normalizers = list(input_filters or [])
    if len(normalizers) < len(durations):
        normalizers.extend([""] * (len(durations) - len(normalizers)))
    for index, normalizer in enumerate(normalizers):
        if normalizer:
            parts.append(f"[{index}:v]{normalizer}[vn{index}]")
    video_label = "vn0" if normalizers[0] else "0:v"
    audio_normalizers = list(audio_input_filters or []) if with_audio else []
    if len(audio_normalizers) < len(durations):
        audio_normalizers.extend([""] * (len(durations) - len(audio_normalizers)))
    if with_audio:
        for index, normalizer in enumerate(audio_normalizers):
            if normalizer:
                parts.append(f"[{index}:a]{normalizer}[an{index}]")
    audio_label: str | None = None
    if with_audio:
        audio_label = "an0" if audio_normalizers[0] else "0:a"
    offset = 0.0
    for index in range(1, len(durations)):
        offset += durations[index - 1] - crossfade
        next_video = f"v{index}"
        source_video = f"vn{index}" if normalizers[index] else f"{index}:v"
        parts.append(
            f"[{video_label}][{source_video}]"
            f"xfade=transition=fade:duration={crossfade:.6f}:"
            f"offset={offset:.6f}[{next_video}]"
        )
        video_label = next_video
        if audio_label is not None:
            next_audio = f"a{index}"
            source_audio = f"an{index}" if audio_normalizers[index] else f"{index}:a"
            parts.append(
                f"[{audio_label}][{source_audio}]"
                f"acrossfade=d={crossfade:.6f}[{next_audio}]"
            )
            audio_label = next_audio
    if post_filters:
        next_video = "vpost"
        parts.append(f"[{video_label}]{','.join(post_filters)}[{next_video}]")
        video_label = next_video
    return _finalize_graph(
        parts,
        video_label=video_label,
        audio_label=audio_label,
        fmt=fmt,
        quality=quality,
    )


def _normalized_concat_graph(
    *,
    with_audio: bool,
    fmt: VideoOutputFormat,
    quality: int,
    post_filters: Sequence[str],
    input_filters: Sequence[str],
    audio_input_filters: Sequence[str],
) -> list[str]:
    parts: list[str] = []
    video_inputs: list[str] = []
    for index, normalizer in enumerate(input_filters):
        if normalizer:
            parts.append(f"[{index}:v]{normalizer}[vn{index}]")
            video_inputs.append(f"[vn{index}]")
        else:
            video_inputs.append(f"[{index}:v]")
    video_label = "v"
    parts.append(
        f"{''.join(video_inputs)}concat=n={len(input_filters)}:v=1:a=0[{video_label}]"
    )
    audio_label: str | None = None
    if with_audio:
        normalizers = list(audio_input_filters)
        if len(normalizers) < len(input_filters):
            normalizers.extend([""] * (len(input_filters) - len(normalizers)))
        audio_inputs: list[str] = []
        for index, normalizer in enumerate(normalizers):
            if normalizer:
                parts.append(f"[{index}:a]{normalizer}[an{index}]")
                audio_inputs.append(f"[an{index}]")
            else:
                audio_inputs.append(f"[{index}:a]")
        audio_label = "a"
        parts.append(
            f"{''.join(audio_inputs)}concat=n={len(normalizers)}:v=0:a=1[{audio_label}]"
        )
    if post_filters:
        parts.append(f"[{video_label}]{','.join(post_filters)}[vpost]")
        video_label = "vpost"
    return _finalize_graph(
        parts,
        video_label=video_label,
        audio_label=audio_label,
        fmt=fmt,
        quality=quality,
    )


def _reference_geometry(
    geometries: Sequence[tuple[int | None, int | None, float | None, str | None]],
    reference: int,
) -> tuple[int | None, int | None, float | None, str | None]:
    width, height, frame_rate, pixel_format = geometries[reference]
    if width is None or height is None:
        width, height = next(
            (
                (candidate_width, candidate_height)
                for candidate_width, candidate_height, _, _ in geometries
                if candidate_width is not None and candidate_height is not None
            ),
            (None, None),
        )
    if frame_rate is None:
        frame_rate = next(
            (candidate for _, _, candidate, _ in geometries if candidate is not None),
            None,
        )
    if pixel_format is None:
        pixel_format = next(
            (candidate for _, _, _, candidate in geometries if candidate is not None),
            None,
        )
    return width, height, frame_rate, pixel_format


def _reference_audio(
    geometries: Sequence[tuple[int | None, int | None]],
    reference: int,
) -> tuple[int | None, int | None]:
    sample_rate, channels = geometries[reference]
    if sample_rate is None:
        sample_rate = next(
            (candidate for candidate, _ in geometries if candidate is not None),
            None,
        )
    if channels is None:
        channels = next(
            (candidate for _, candidate in geometries if candidate is not None),
            None,
        )
    return sample_rate, channels


def concatenate_videos(
    sources: Sequence[VideoSource],
    destination: str | Path,
    *,
    crossfade_seconds: float = 0.0,
    normalize: bool = False,
    reference: int = 0,
    output_format: str | VideoOutputFormat | None = None,
    ffmpeg_executable: str = "ffmpeg",
    ffmpeg_runner: FFmpegRunner | None = None,
    ffprobe_executable: str = "ffprobe",
    ffprobe_runner: FFprobeRunner | None = None,
    timeout_seconds: float = DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS,
) -> Path:
    """Join two or more clips in order, optionally crossfading between them.

    A zero crossfade uses ffmpeg's concat demuxer, which keeps the source
    codecs. Passing ``normalize=True`` instead re-encodes every clip through a
    concat filter so mixed geometry, pixel formats and audio layouts join
    cleanly; ``reference`` selects the clip whose format wins. A positive
    crossfade probes each clip's duration, requires the fade to be shorter than
    every clip and builds an ``xfade`` chain. Clips whose dimensions or frame
    rate differ from the reference clip that reports them are scaled, padded
    and resampled to that canonical geometry so ``xfade`` gets matching inputs.
    Clips whose pixel format differs from the reference clip that reports one
    are converted so every ``xfade`` input matches. Audio is kept when every
    clip has an audio track; clips whose sample rate or channel layout differ
    from the reference clip that reports them are resampled and remapped so
    ``acrossfade`` gets matching inputs.
    """
    path = Path(destination)
    fmt = (
        VideoOutputFormat.from_path(path)
        if output_format is None
        else VideoOutputFormat.parse(output_format)
    )
    if not isinstance(sources, Sequence) or isinstance(
        sources, (str, bytes, bytearray, memoryview)
    ):
        raise TypeError("sources must be a sequence of video sources")
    if len(sources) < 2:
        raise ValidationError("concatenation needs at least two video sources")
    crossfade = _number(
        crossfade_seconds,
        label="crossfade_seconds",
        non_negative=True,
    )
    if not isinstance(normalize, bool):
        raise TypeError("normalize must be a boolean")
    if isinstance(reference, bool) or not isinstance(reference, int):
        raise TypeError("reference must be an integer index")
    if reference < 0 or reference >= len(sources):
        raise ValidationError("reference must point at one of the concatenated clips")
    _check_destination(sources, path)
    attempt = _default_attempt(fmt)
    post_filters = _post_filters(attempt, width=None, height=None)

    with ExitStack() as stack:
        active = [
            stack.enter_context(_materialize(item, prefix="aivia-video-concat-"))
            for item in sources
        ]
        if crossfade == 0.0 and not normalize:
            list_path = stack.enter_context(
                tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".txt",
                    prefix="aivia-concat-",
                    delete=False,
                    encoding="utf-8",
                )
            )
            temporary_list = Path(list_path.name)
            stack.callback(temporary_list.unlink, missing_ok=True)
            list_path.write("\n".join(_concat_entry(item) for item in active))
            list_path.write("\n")
            list_path.close()

            def build(temporary: Path) -> list[str]:
                argv = _base(ffmpeg_executable)
                argv.extend(
                    [
                        "-f",
                        "concat",
                        "-safe",
                        "0",
                        "-i",
                        str(temporary_list),
                    ]
                )
                argv.extend(_simple_graph(fmt, post_filters, quality=attempt.quality))
                if fmt.supports_audio:
                    argv.extend(["-map", "0:a:0?"])
                argv.extend(
                    _encode_args(
                        fmt,
                        attempt,
                        audio=fmt.supports_audio,
                        loop=0,
                        audio_bitrate_kbps=DEFAULT_AUDIO_BITRATE_KBPS,
                    )
                )
                argv.append(str(temporary))
                return argv

            return _write_atomic(
                path,
                build,
                runner=ffmpeg_runner,
                timeout_seconds=timeout_seconds,
                failure="ffmpeg could not concatenate the videos",
            )

        infos = [
            run_ffprobe(
                item,
                executable=ffprobe_executable,
                timeout_seconds=timeout_seconds,
                runner=ffprobe_runner,
            )
            for item in active
        ]
        durations: list[float] = []
        if crossfade > 0.0:
            for info in infos:
                clip_length = info.duration_seconds
                if clip_length is None or clip_length <= 0:
                    raise ValidationError(
                        "could not determine a clip duration for the crossfade"
                    )
                durations.append(clip_length)
            if crossfade >= min(durations):
                raise ValidationError(
                    "crossfade_seconds must be shorter than every clip"
                )
        with_audio = all(info.audio_streams for info in infos)
        geometries = [_video_geometry(info) for info in infos]
        target_width, target_height, target_rate, target_pixel_format = (
            _reference_geometry(geometries, reference)
        )
        input_filters: list[str] = []
        if (
            (target_width is not None and target_height is not None)
            or target_rate is not None
            or target_pixel_format is not None
        ):
            target = (
                _even_dimension(target_width) if target_width is not None else None,
                _even_dimension(target_height) if target_height is not None else None,
                target_rate,
                target_pixel_format,
            )
            input_filters = [
                _normalize_video_filter(geometry, target) for geometry in geometries
            ]
        else:
            input_filters = [""] * len(geometries)
        audio_geometries = [_audio_geometry(info) for info in infos]
        target_audio_rate, target_audio_channels = _reference_audio(
            audio_geometries, reference
        )
        audio_input_filters: list[str] = []
        if with_audio and (
            target_audio_rate is not None or target_audio_channels is not None
        ):
            target_audio = (target_audio_rate, target_audio_channels)
            audio_input_filters = [
                _normalize_audio_filter(geometry, target_audio)
                for geometry in audio_geometries
            ]
        keep_audio = with_audio and fmt.supports_audio
        if crossfade == 0.0:
            graph = _normalized_concat_graph(
                with_audio=keep_audio,
                fmt=fmt,
                quality=attempt.quality,
                post_filters=post_filters,
                input_filters=input_filters,
                audio_input_filters=audio_input_filters,
            )
        else:
            graph = _crossfade_graph(
                durations,
                crossfade,
                with_audio=keep_audio,
                fmt=fmt,
                quality=attempt.quality,
                post_filters=post_filters,
                input_filters=input_filters,
                audio_input_filters=audio_input_filters,
            )

        def build_normalized(temporary: Path) -> list[str]:
            argv = _base(ffmpeg_executable)
            for item in active:
                argv.extend(["-i", str(item)])
            argv.extend(graph)
            argv.extend(
                _encode_args(
                    fmt,
                    attempt,
                    audio=keep_audio,
                    loop=0,
                    audio_bitrate_kbps=DEFAULT_AUDIO_BITRATE_KBPS,
                )
            )
            argv.append(str(temporary))
            return argv

        return _write_atomic(
            path,
            build_normalized,
            runner=ffmpeg_runner,
            timeout_seconds=timeout_seconds,
            failure=(
                "ffmpeg could not crossfade the videos"
                if crossfade > 0.0
                else "ffmpeg could not concatenate the videos"
            ),
        )


def extract_frame(
    source: VideoSource,
    destination: str | Path,
    *,
    timestamp_seconds: float = 0.0,
    ffmpeg_executable: str = "ffmpeg",
    ffmpeg_runner: FFmpegRunner | None = None,
    timeout_seconds: float = DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS,
) -> Path:
    """Extract one still frame at *timestamp_seconds* into *destination*."""
    path = Path(destination)
    if not path.suffix:
        raise ValidationError(
            f"cannot infer an image format from {path.name!r}; add a suffix"
        )
    timestamp = _number(
        timestamp_seconds,
        label="timestamp_seconds",
        non_negative=True,
    )
    _check_destination([source], path)

    with _materialize(source, prefix="aivia-video-frame-") as active:

        def build(temporary: Path) -> list[str]:
            argv = _base(ffmpeg_executable)
            argv.extend(
                [
                    "-ss",
                    f"{timestamp:.6f}",
                    "-i",
                    str(active),
                    "-frames:v",
                    "1",
                    "-an",
                    "-map",
                    "0:v:0",
                ]
            )
            argv.append(str(temporary))
            return argv

        return _write_atomic(
            path,
            build,
            runner=ffmpeg_runner,
            timeout_seconds=timeout_seconds,
            failure="ffmpeg could not extract a frame",
        )


_AUDIO_CODECS: dict[str, str] = {
    ".mp3": "libmp3lame",
    ".m4a": "aac",
    ".aac": "aac",
    ".wav": "pcm_s16le",
    ".flac": "flac",
    ".ogg": "libvorbis",
    ".opus": "libopus",
    ".webm": "libopus",
}


def _audio_codec(path: Path) -> str:
    try:
        return _AUDIO_CODECS[path.suffix.lower()]
    except KeyError:
        allowed = ", ".join(sorted(_AUDIO_CODECS))
        raise ValidationError(
            f"unsupported audio format {path.suffix or '<none>'!r}; "
            f"expected one of {allowed}"
        ) from None


def extract_audio(
    source: VideoSource,
    destination: str | Path,
    *,
    bitrate_kbps: int = DEFAULT_AUDIO_BITRATE_KBPS,
    ffmpeg_executable: str = "ffmpeg",
    ffmpeg_runner: FFmpegRunner | None = None,
    timeout_seconds: float = DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS,
) -> Path:
    """Extract the first audio track, choosing the codec from the suffix."""
    path = Path(destination)
    codec = _audio_codec(path)
    bitrate = _audio_bitrate(bitrate_kbps)
    _check_destination([source], path)

    with _materialize(source, prefix="aivia-video-audio-") as active:

        def build(temporary: Path) -> list[str]:
            argv = _base(ffmpeg_executable)
            argv.extend(
                [
                    "-i",
                    str(active),
                    "-vn",
                    "-map",
                    "0:a:0",
                    "-c:a",
                    codec,
                ]
            )
            if codec not in {"pcm_s16le", "flac"}:
                argv.extend(["-b:a", f"{bitrate}k"])
            argv.append(str(temporary))
            return argv

        return _write_atomic(
            path,
            build,
            runner=ffmpeg_runner,
            timeout_seconds=timeout_seconds,
            failure="ffmpeg could not extract the audio track",
        )


def make_seamless_loop(
    source: VideoSource,
    destination: str | Path,
    *,
    keep_audio: bool = False,
    output_format: str | VideoOutputFormat | None = None,
    ffmpeg_executable: str = "ffmpeg",
    ffmpeg_runner: FFmpegRunner | None = None,
    ffprobe_executable: str = "ffprobe",
    ffprobe_runner: FFprobeRunner | None = None,
    timeout_seconds: float = DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS,
) -> Path:
    """Turn a clip into a palindrome loop whose ends meet exactly.

    The clip is followed by a reversed copy, so the last frame of the forward
    pass matches the first frame of the reverse pass and the last frame of the
    reverse pass matches the first frame of the forward pass. Audio is dropped
    unless ``keep_audio`` is set, in which case it is reversed as well. A source
    without an audio track is looped silently and emits
    :class:`VideoAudioWarning`.
    """
    path = Path(destination)
    fmt = (
        VideoOutputFormat.from_path(path)
        if output_format is None
        else VideoOutputFormat.parse(output_format)
    )
    _check_destination([source], path)
    attempt = _default_attempt(fmt)
    wants_audio = keep_audio and fmt.supports_audio
    post_filters = _post_filters(attempt, width=None, height=None)

    with _materialize(source, prefix="aivia-video-loop-") as active:
        audio = wants_audio
        if audio and not _has_audio(
            active,
            executable=ffprobe_executable,
            runner=ffprobe_runner,
            timeout_seconds=timeout_seconds,
        ):
            audio = False
            warnings.warn(
                "source has no audio track; audio was dropped",
                VideoAudioWarning,
                stacklevel=2,
            )
        parts = [
            "[0:v]split=2[forward][reverse];"
            "[reverse]reverse[reversed];"
            "[forward][reversed]concat=n=2:v=1:a=0[v]"
        ]
        video_label = "v"
        audio_label: str | None = None
        if audio:
            parts.append(
                "[0:a]asplit=2[af][ar];[ar]areverse[arr];[af][arr]concat=n=2:v=0:a=1[a]"
            )
            audio_label = "a"
        if post_filters:
            parts.append(f"[{video_label}]{','.join(post_filters)}[vpost]")
            video_label = "vpost"
        graph = _finalize_graph(
            parts,
            video_label=video_label,
            audio_label=audio_label,
            fmt=fmt,
            quality=attempt.quality,
        )

        def build(temporary: Path) -> list[str]:
            argv = _base(ffmpeg_executable)
            argv.extend(["-i", str(active)])
            argv.extend(graph)
            argv.extend(
                _encode_args(
                    fmt,
                    attempt,
                    audio=audio,
                    loop=0,
                    audio_bitrate_kbps=DEFAULT_AUDIO_BITRATE_KBPS,
                )
            )
            argv.append(str(temporary))
            return argv

        return _write_atomic(
            path,
            build,
            runner=ffmpeg_runner,
            timeout_seconds=timeout_seconds,
            failure="ffmpeg could not create a seamless loop",
        )


def convert_video(
    source: VideoSource,
    destination: str | Path,
    *,
    output_format: str | VideoOutputFormat | None = None,
    max_bytes: int | None = None,
    fps: float | None = None,
    width: int | None = None,
    height: int | None = None,
    loop: int = 0,
    quality: int | None = None,
    crf: int | None = None,
    keep_audio: bool = True,
    audio_bitrate_kbps: int = DEFAULT_AUDIO_BITRATE_KBPS,
    ffmpeg_executable: str = "ffmpeg",
    ffmpeg_runner: FFmpegRunner | None = None,
    timeout_seconds: float = DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS,
) -> Path:
    """Encode *source* as MP4, animated WebP or GIF, optionally under a cap.

    When ``max_bytes`` is set, quality is lowered and then the frame size and
    rate are reduced until the written file fits. If no attempt fits,
    :class:`VideoSizeError` is raised and the destination is left untouched.
    GIF and animated WebP never carry audio; ``loop`` follows ffmpeg's GIF and
    WebP convention where ``0`` means loop forever.
    """
    path = Path(destination)
    fmt = (
        VideoOutputFormat.from_path(path)
        if output_format is None
        else VideoOutputFormat.parse(output_format)
    )
    cap = _positive_int(max_bytes, label="max_bytes") if max_bytes is not None else None
    fps_value = _number(fps, label="fps", positive=True) if fps is not None else None
    width_value = _positive_int(width, label="width") if width is not None else None
    height_value = _positive_int(height, label="height") if height is not None else None
    loop_value = _integer(loop, label="loop")
    base_quality = _quality(quality)
    base_crf = _crf(crf)
    bitrate = _audio_bitrate(audio_bitrate_kbps)
    audio = keep_audio and fmt.supports_audio
    _check_destination([source], path)

    with (
        _materialize(source, prefix="aivia-video-convert-") as active,
        _temp_output(path) as temporary,
    ):
        last_size: int | None = None
        attempts = _attempts(
            fmt,
            crf=base_crf,
            quality=base_quality,
            fps=fps_value,
            capped=cap is not None,
        )
        for attempt in attempts:
            filters = _post_filters(
                attempt,
                width=width_value,
                height=height_value,
            )
            argv = _base(ffmpeg_executable)
            argv.extend(["-i", str(active)])
            argv.extend(_simple_graph(fmt, filters, quality=attempt.quality))
            if audio:
                argv.extend(["-map", "0:a:0?"])
            argv.extend(
                _encode_args(
                    fmt,
                    attempt,
                    audio=audio,
                    loop=loop_value,
                    audio_bitrate_kbps=bitrate,
                )
            )
            argv.append(str(temporary))
            _run_to_file(
                argv,
                temporary,
                runner=ffmpeg_runner,
                timeout_seconds=timeout_seconds,
                failure="ffmpeg could not convert the video",
            )
            size = temporary.stat().st_size
            if cap is None or size <= cap:
                os.replace(temporary, path)
                return path
            last_size = size
            temporary.unlink()
        raise VideoSizeError(
            f"could not encode the video below {cap} bytes",
            details={
                "max_bytes": cap,
                "last_size_bytes": last_size,
                "attempts": len(attempts),
            },
        )


trim = trim_video
speed_video = change_speed
concat_videos = concatenate_videos
seamless_loop = make_seamless_loop
export_video = convert_video
