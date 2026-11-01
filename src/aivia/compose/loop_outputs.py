"""Render a composition to a seamless loop and verify the seam.

The deterministic video renderer produces the clip. :func:`make_seamless_loop`
turns that clip into a palindrome, so the final frame returns to the first one.
GIF and WebP outputs are encoded from the palindrome with the requested loop
count and byte cap. The written file is then inspected: its first and last
frames are compared with a normalized mean absolute difference and the render
fails when the seam is not similar enough.
"""

from __future__ import annotations

import math
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image, ImageChops, ImageStat

from aivia.core.config.media import (
    DEFAULT_MEDIA_ANIMATED_FRAME_RATE,
    DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS,
    DEFAULT_MEDIA_FFPROBE_TIMEOUT_SECONDS,
    DEFAULT_MEDIA_LOOP_SIMILARITY_THRESHOLD,
)
from aivia.core.errors import ValidationError
from aivia.media.video_utils import (
    DEFAULT_VIDEO_QUALITY,
    VideoOutputFormat,
    VideoSizeError,
    convert_video,
    extract_frame,
    make_seamless_loop,
)

from .errors import RenderError
from .schema import Composition
from .video import FFmpegRunner, VideoRenderResult, render_video

if TYPE_CHECKING:
    from aivia.core.media_probe import FFprobeRunner

__all__ = [
    "DEFAULT_LOOP_SIMILARITY_THRESHOLD",
    "LOOP_RENDERER_VERSION",
    "LoopOutput",
    "LoopOutputError",
    "LoopOutputFormat",
    "LoopRenderResult",
    "LoopSeamError",
    "LoopSizeError",
    "SeamlessLoopOutput",
    "export_loop",
    "export_seamless",
    "export_seamless_loop",
    "frame_similarity",
    "measure_loop_seam",
    "render_loop",
    "render_seamless",
    "render_seamless_loop",
    "seam_similarity",
]

LOOP_RENDERER_VERSION = "1"
"""The version of the seamless loop output pipeline."""

DEFAULT_LOOP_SIMILARITY_THRESHOLD = DEFAULT_MEDIA_LOOP_SIMILARITY_THRESHOLD
"""First and last frames at or above this similarity form a seamless seam."""

_LOOP_FORMATS = frozenset(
    {
        VideoOutputFormat.MP4,
        VideoOutputFormat.GIF,
        VideoOutputFormat.WEBP,
    }
)
_ANIMATED_FORMATS = frozenset({VideoOutputFormat.GIF, VideoOutputFormat.WEBP})
_FIRST_FRAME_TIMESTAMP = 0.0


class LoopOutputError(RenderError):
    """A composition could not be rendered as a seamless loop."""

    code = "loop_output"


class LoopSeamError(LoopOutputError):
    """The first and last frames of the loop are not similar enough."""

    code = "loop_seam"


class LoopSizeError(VideoSizeError, LoopOutputError):
    """No loop encode attempt fit the requested byte cap."""

    code = "loop_size"


@dataclass(frozen=True, slots=True, eq=False)
class LoopRenderResult:
    """A seamless loop, the video it came from and its seam measurement."""

    output: Path
    format: VideoOutputFormat
    loop: int
    size_bytes: int
    max_bytes: int | None
    fps: float
    quality: int
    seam_similarity: float
    seam_threshold: float
    video: VideoRenderResult

    @property
    def path(self) -> Path:
        """Alias for :attr:`output`."""
        return self.output

    @property
    def size(self) -> int:
        """Alias for :attr:`size_bytes`."""
        return self.size_bytes

    @property
    def bytes(self) -> int:
        """Alias for :attr:`size_bytes`."""
        return self.size_bytes

    @property
    def seamless(self) -> bool:
        """Whether the measured seam met the configured threshold."""
        return self.seam_similarity >= self.seam_threshold

    @property
    def container_loop(self) -> int:
        """The loop value written into the container."""
        return _container_loop(self.format, self.loop)

    @property
    def command(self) -> Any:
        """The intermediate video render command."""
        return self.video.command

    @property
    def argv(self) -> Sequence[str]:
        """The argv of the intermediate video render."""
        return self.video.command.argv

    @property
    def probe(self) -> Any:
        """The ffprobe result of the intermediate video render."""
        return self.video.probe

    def __fspath__(self) -> str:
        return str(self.output)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.output, name)

    def __str__(self) -> str:
        return str(self.output)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (str, Path)):
            return self.output == Path(other)
        if not isinstance(other, LoopRenderResult):
            return NotImplemented
        return (
            self.output == other.output
            and self.format is other.format
            and self.loop == other.loop
            and self.size_bytes == other.size_bytes
            and self.max_bytes == other.max_bytes
            and self.fps == other.fps
            and self.quality == other.quality
            and self.seam_similarity == other.seam_similarity
            and self.seam_threshold == other.seam_threshold
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.output,
                self.format,
                self.loop,
                self.size_bytes,
                self.seam_similarity,
            )
        )


def _format(
    output: Path,
    output_format: str | VideoOutputFormat | None,
) -> VideoOutputFormat:
    if output_format is not None:
        try:
            fmt = VideoOutputFormat.parse(output_format)
        except ValidationError as exc:
            raise LoopOutputError(
                f"unsupported loop format {output_format!r}; expected mp4, gif or webp"
            ) from exc
        if fmt not in _LOOP_FORMATS:
            raise LoopOutputError(
                f"unsupported loop format {fmt.value!r}; expected mp4, gif or webp"
            )
        return fmt
    if not output.suffix:
        raise LoopOutputError(
            "cannot infer a loop format; use a .mp4, .gif or .webp output or pass "
            "output_format"
        )
    try:
        fmt = VideoOutputFormat.parse(output.suffix)
    except ValidationError as exc:
        raise LoopOutputError(
            f"unsupported loop format {output.suffix.lstrip('.').lower()!r}; "
            "expected mp4, gif or webp"
        ) from exc
    if fmt not in _LOOP_FORMATS:
        raise LoopOutputError(
            f"unsupported loop format {fmt.value!r}; expected mp4, gif or webp"
        )
    return fmt


def _loop(value: int | bool) -> int:
    if isinstance(value, bool):
        loop = 0 if value else -1
    elif not isinstance(value, int):
        raise LoopOutputError("loop must be an integer or boolean")
    else:
        loop = value
    if loop < -1:
        raise LoopOutputError("loop must be -1, 0 or a positive integer")
    if loop > 65535:
        raise LoopOutputError("loop must not exceed 65535")
    return loop


def _container_loop(fmt: VideoOutputFormat, loop: int) -> int:
    if fmt is VideoOutputFormat.WEBP and loop < 0:
        return 1
    return loop


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LoopOutputError(f"{name} must be a positive integer")
    return value


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LoopOutputError(f"{name} must be a positive number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise LoopOutputError(f"{name} must be a positive number")
    return number


def _quality(value: int | None) -> int:
    if value is None:
        return DEFAULT_VIDEO_QUALITY
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 100:
        raise LoopOutputError("quality must be an integer between 1 and 100")
    return value


def _threshold(value: float | None) -> float:
    if value is None:
        return DEFAULT_LOOP_SIMILARITY_THRESHOLD
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LoopOutputError("seam threshold must be a number between 0 and 1")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise LoopOutputError("seam threshold must be a number between 0 and 1")
    return number


def _rgb(value: Image.Image | str | Path) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB").copy()
    with Image.open(Path(value)) as opened:
        return opened.convert("RGB").copy()


def frame_similarity(
    first: Image.Image | str | Path,
    second: Image.Image | str | Path,
) -> float:
    """Return how similar two frames are in ``[0.0, 1.0]``.

    ``1.0`` means the frames are pixel-identical and ``0.0`` means the mean
    absolute difference across all colour channels is the full 255 levels.
    """
    left = _rgb(first)
    right = _rgb(second)
    if left.size != right.size:
        raise LoopOutputError(
            "frames must have the same size",
            details={"first": left.size, "second": right.size},
        )
    statistics = ImageStat.Stat(ImageChops.difference(left, right))
    mean = sum(statistics.mean) / len(statistics.mean)
    return max(0.0, min(1.0, 1.0 - mean / 255.0))


seam_similarity = frame_similarity


def _rate(value: str | None) -> float | None:
    if not value:
        return None
    numerator, _, denominator = value.partition("/")
    try:
        top = float(numerator)
        bottom = float(denominator) if denominator else 1.0
    except ValueError:
        return None
    if bottom == 0.0:
        return None
    return top / bottom


def _animated_seam(path: Path) -> float:
    with Image.open(path) as image:
        frames = int(getattr(image, "n_frames", 1))
        image.seek(0)
        first = image.convert("RGB").copy()
        image.seek(frames - 1)
        last = image.convert("RGB").copy()
    return frame_similarity(first, last)


def _extract_last_frame(
    path: Path,
    target: Path,
    *,
    timestamp: float,
    rate: float,
    ffmpeg_executable: str,
    timeout_seconds: float,
    ffmpeg_runner: FFmpegRunner | None,
) -> None:
    """Extract the last decodable frame, stepping back if seeking overshoots."""
    error: ValidationError | None = None
    seen: set[float] = set()
    for offset in (0.0, 0.5, 1.0, 2.0):
        candidate = max(_FIRST_FRAME_TIMESTAMP, timestamp - offset / rate)
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            extract_frame(
                path,
                target,
                timestamp_seconds=candidate,
                ffmpeg_executable=ffmpeg_executable,
                ffmpeg_runner=ffmpeg_runner,
                timeout_seconds=timeout_seconds,
            )
            return
        except ValidationError as exc:
            error = exc
    if error is not None:
        raise error
    raise LoopOutputError("the loop output has no decodable last frame")


def _video_seam(
    path: Path,
    *,
    fps: float | None,
    ffmpeg_executable: str,
    ffprobe_executable: str,
    timeout_seconds: float,
    ffprobe_timeout_seconds: float,
    ffmpeg_runner: FFmpegRunner | None,
    ffprobe_runner: FFprobeRunner | None,
) -> float:
    from aivia.core.media_probe import run_ffprobe

    probe = run_ffprobe(
        path,
        executable=ffprobe_executable,
        timeout_seconds=ffprobe_timeout_seconds,
        runner=ffprobe_runner,
    )
    length = probe.duration_seconds
    if length is None or length <= 0.0:
        raise LoopOutputError("the loop output has no measurable duration")
    stream = probe.primary_video
    rate = fps
    if rate is None and stream is not None:
        rate = _rate(stream.frame_rate)
    if rate is None or rate <= 0.0:
        rate = DEFAULT_MEDIA_ANIMATED_FRAME_RATE
    frame_count = max(1, round(length * rate))
    last_timestamp = max(_FIRST_FRAME_TIMESTAMP, (frame_count - 1) / rate)
    with tempfile.TemporaryDirectory(prefix="aivia-loop-seam-") as directory:
        first = Path(directory) / "first.png"
        last = Path(directory) / "last.png"
        extract_frame(
            path,
            first,
            timestamp_seconds=_FIRST_FRAME_TIMESTAMP,
            ffmpeg_executable=ffmpeg_executable,
            ffmpeg_runner=ffmpeg_runner,
            timeout_seconds=timeout_seconds,
        )
        _extract_last_frame(
            path,
            last,
            timestamp=last_timestamp,
            rate=rate,
            ffmpeg_executable=ffmpeg_executable,
            timeout_seconds=timeout_seconds,
            ffmpeg_runner=ffmpeg_runner,
        )
        return frame_similarity(first, last)


def measure_loop_seam(
    output: str | Path,
    *,
    output_format: str | VideoOutputFormat | None = None,
    fps: float | None = None,
    ffmpeg_executable: str = "ffmpeg",
    ffprobe_executable: str = "ffprobe",
    timeout_seconds: float = DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS,
    ffprobe_timeout_seconds: float = DEFAULT_MEDIA_FFPROBE_TIMEOUT_SECONDS,
    ffmpeg_runner: FFmpegRunner | None = None,
    ffprobe_runner: FFprobeRunner | None = None,
) -> float:
    """Measure the first to last frame similarity of a loop output."""
    path = Path(output)
    if not path.is_file():
        raise LoopOutputError(
            "loop output does not exist",
            details={"path": str(path)},
        )
    fmt = _format(path, output_format)
    if fmt in _ANIMATED_FORMATS:
        return _animated_seam(path)
    return _video_seam(
        path,
        fps=fps,
        ffmpeg_executable=ffmpeg_executable,
        ffprobe_executable=ffprobe_executable,
        timeout_seconds=timeout_seconds,
        ffprobe_timeout_seconds=ffprobe_timeout_seconds,
        ffmpeg_runner=ffmpeg_runner,
        ffprobe_runner=ffprobe_runner,
    )


def _result_fps(
    fmt: VideoOutputFormat,
    fps: float | None,
    video: VideoRenderResult,
) -> float:
    if fps is not None:
        return fps
    if fmt is VideoOutputFormat.MP4:
        stream = video.probe.primary_video
        rate = _rate(stream.frame_rate) if stream is not None else None
        if rate is not None:
            return rate
    return float(DEFAULT_MEDIA_ANIMATED_FRAME_RATE)


def render_loop(
    composition: Composition | Mapping[str, Any],
    output: str | Path = "out.mp4",
    *,
    output_format: str | VideoOutputFormat | None = None,
    loop: int | bool = 0,
    max_bytes: int | None = None,
    fps: float | None = None,
    quality: int | None = None,
    seam_threshold: float | None = None,
    keep_audio: bool = False,
    ffmpeg_executable: str = "ffmpeg",
    ffprobe_executable: str = "ffprobe",
    timeout_seconds: float = DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS,
    ffprobe_timeout_seconds: float = DEFAULT_MEDIA_FFPROBE_TIMEOUT_SECONDS,
    ffmpeg_runner: FFmpegRunner | None = None,
    ffprobe_runner: FFprobeRunner | None = None,
    **render_options: Any,
) -> LoopRenderResult:
    """Render *composition* to a seamless loop and verify its seam.

    The clip is rendered deterministically and then turned into a palindrome,
    so the last frame is the first frame played backwards. ``loop`` follows the
    GIF and WebP convention where ``0`` loops forever, ``-1`` plays once and a
    positive integer repeats that many times.

    When ``max_bytes`` is set, quality is lowered and then the frame size and
    rate are reduced until the file fits. If no attempt fits,
    :class:`LoopSizeError` is raised and *output* is left untouched. If the
    first and last frames do not reach ``seam_threshold``,
    :class:`LoopSeamError` is raised and the rejected output is removed.
    """
    destination = Path(output)
    fmt = _format(destination, output_format)
    loop_value = _loop(loop)
    container_loop = _container_loop(fmt, loop_value)
    cap = None if max_bytes is None else _positive_int(max_bytes, "max_bytes")
    fps_value = None if fps is None else _positive_number(fps, "fps")
    quality_value = _quality(quality)
    threshold_value = _threshold(seam_threshold)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=f".{destination.stem}-loop-",
        dir=destination.parent,
    ) as directory:
        source = Path(directory) / f"{destination.stem or 'loop'}.mp4"
        video = render_video(
            composition,
            source,
            ffmpeg_executable=ffmpeg_executable,
            ffprobe_executable=ffprobe_executable,
            timeout_seconds=timeout_seconds,
            ffprobe_timeout_seconds=ffprobe_timeout_seconds,
            ffmpeg_runner=ffmpeg_runner,
            ffprobe_runner=ffprobe_runner,
            **render_options,
        )
        if not isinstance(video, VideoRenderResult):
            raise LoopOutputError("the intermediate render was not a video")
        looped = Path(directory) / "looped.mp4"
        make_seamless_loop(
            video.output,
            looped,
            keep_audio=keep_audio,
            output_format=VideoOutputFormat.MP4,
            ffmpeg_executable=ffmpeg_executable,
            ffmpeg_runner=ffmpeg_runner,
            ffprobe_executable=ffprobe_executable,
            ffprobe_runner=ffprobe_runner,
            timeout_seconds=timeout_seconds,
        )
        try:
            convert_video(
                looped,
                destination,
                output_format=fmt,
                max_bytes=cap,
                fps=fps_value,
                loop=container_loop,
                quality=quality_value,
                keep_audio=keep_audio,
                ffmpeg_executable=ffmpeg_executable,
                ffmpeg_runner=ffmpeg_runner,
                timeout_seconds=timeout_seconds,
            )
        except VideoSizeError as exc:
            raise LoopSizeError(exc.message, details=exc.details) from exc
        try:
            similarity = measure_loop_seam(
                destination,
                output_format=fmt,
                fps=fps_value,
                ffmpeg_executable=ffmpeg_executable,
                ffprobe_executable=ffprobe_executable,
                timeout_seconds=timeout_seconds,
                ffprobe_timeout_seconds=ffprobe_timeout_seconds,
                ffmpeg_runner=ffmpeg_runner,
                ffprobe_runner=ffprobe_runner,
            )
        except ValidationError as exc:
            destination.unlink(missing_ok=True)
            raise LoopOutputError(
                "the loop output could not be inspected",
                details={"error": exc.message},
            ) from exc
        if similarity < threshold_value:
            destination.unlink(missing_ok=True)
            raise LoopSeamError(
                "the loop seam is not similar enough",
                details={
                    "similarity": similarity,
                    "threshold": threshold_value,
                },
            )

    return LoopRenderResult(
        output=destination,
        format=fmt,
        loop=loop_value,
        size_bytes=destination.stat().st_size,
        max_bytes=cap,
        fps=_result_fps(fmt, fps_value, video),
        quality=quality_value,
        seam_similarity=similarity,
        seam_threshold=threshold_value,
        video=video,
    )


render_seamless_loop = render_loop
render_seamless = render_loop
export_loop = render_loop
export_seamless_loop = render_loop
export_seamless = render_loop

LoopOutput = LoopRenderResult
LoopOutputFormat = VideoOutputFormat
SeamlessLoopOutput = LoopRenderResult
