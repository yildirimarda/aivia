"""Render a composition to an optimized animated GIF or WebP file.

The video renderer produces a deterministic intermediate MP4. A second,
size-aware encode then turns it into an animated image: GIF uses a generated
palette and animated WebP uses the animated WebP encoder. Both honour the loop
flag and, when a byte cap is supplied, lower quality and then resolution until
the output fits. The intermediate video is written to a temporary directory
beside the destination so a failed encode never leaves a partial animation.
"""

from __future__ import annotations

import math
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aivia.core.config.media import (
    DEFAULT_MEDIA_ANIMATED_FRAME_RATE,
    DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS,
    DEFAULT_MEDIA_FFPROBE_TIMEOUT_SECONDS,
)
from aivia.core.errors import ValidationError
from aivia.media.video_utils import (
    DEFAULT_VIDEO_QUALITY,
    VideoOutputFormat,
    VideoSizeError,
    convert_video,
)

from .errors import RenderError
from .schema import Composition
from .video import FFmpegRunner, VideoRenderResult, render_video

if TYPE_CHECKING:
    from aivia.core.media_probe import FFprobeRunner

__all__ = [
    "ANIMATED_RENDERER_VERSION",
    "AnimatedOutput",
    "AnimatedOutputError",
    "AnimatedOutputFormat",
    "AnimatedRenderResult",
    "AnimatedSizeError",
    "AnimationOutput",
    "AnimationRenderResult",
    "export_animated",
    "export_animation",
    "export_gif",
    "export_webp",
    "render_animated",
    "render_animated_output",
    "render_animation",
    "render_gif",
    "render_webp",
]

ANIMATED_RENDERER_VERSION = "1"
"""The version of the animated image output pipeline."""

AnimatedOutputFormat = VideoOutputFormat
"""The containers an animated render can write: GIF and WebP."""

_ANIMATED_FORMATS = frozenset({VideoOutputFormat.GIF, VideoOutputFormat.WEBP})


class AnimatedOutputError(RenderError):
    """A composition could not be rendered as an animated image."""

    code = "animated_output"


class AnimatedSizeError(VideoSizeError, AnimatedOutputError):
    """No animated encode attempt fit the requested byte cap."""

    code = "animated_size"


@dataclass(frozen=True, slots=True, eq=False)
class AnimatedRenderResult:
    """An animated image, the video it came from and its final settings."""

    output: Path
    format: VideoOutputFormat
    loop: int
    size_bytes: int
    max_bytes: int | None
    fps: float
    quality: int
    video: VideoRenderResult

    @property
    def path(self) -> Path:
        """Alias for :attr:`output`."""
        return self.output

    @property
    def container_loop(self) -> int:
        """The loop value written into the container."""
        return _container_loop(self.format, self.loop)

    @property
    def size(self) -> int:
        """Alias for :attr:`size_bytes`."""
        return self.size_bytes

    @property
    def bytes(self) -> int:
        """Alias for :attr:`size_bytes`."""
        return self.size_bytes

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
        if not isinstance(other, AnimatedRenderResult):
            return NotImplemented
        return (
            self.output == other.output
            and self.format is other.format
            and self.loop == other.loop
            and self.size_bytes == other.size_bytes
            and self.max_bytes == other.max_bytes
            and self.fps == other.fps
            and self.quality == other.quality
        )

    def __hash__(self) -> int:
        return hash((self.output, self.format, self.loop, self.size_bytes))


def _format(
    output: Path,
    output_format: str | VideoOutputFormat | None,
) -> VideoOutputFormat:
    if output_format is not None:
        try:
            fmt = VideoOutputFormat.parse(output_format)
        except ValidationError as exc:
            raise AnimatedOutputError(str(exc)) from exc
        if fmt not in _ANIMATED_FORMATS:
            raise AnimatedOutputError(
                f"unsupported animated format {fmt.value!r}; expected gif or webp"
            )
        return fmt
    try:
        fmt = VideoOutputFormat.from_path(output)
    except ValidationError as exc:
        raise AnimatedOutputError(
            "cannot infer an animated format; use a .gif or .webp output or pass "
            "output_format"
        ) from exc
    if fmt not in _ANIMATED_FORMATS:
        raise AnimatedOutputError(
            f"unsupported animated format {fmt.value!r}; expected gif or webp"
        )
    return fmt


def _loop(value: int | bool) -> int:
    if isinstance(value, bool):
        loop = 0 if value else -1
    elif not isinstance(value, int):
        raise AnimatedOutputError("loop must be an integer or boolean")
    else:
        loop = value
    if loop < -1:
        raise AnimatedOutputError("loop must be -1, 0 or a positive integer")
    if loop > 65535:
        raise AnimatedOutputError("loop must not exceed 65535")
    return loop


def _container_loop(fmt: VideoOutputFormat, loop: int) -> int:
    if fmt is VideoOutputFormat.WEBP and loop < 0:
        return 1
    return loop


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AnimatedOutputError(f"{name} must be a positive integer")
    return value


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnimatedOutputError(f"{name} must be a positive number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise AnimatedOutputError(f"{name} must be a positive number")
    return number


def _quality(value: int | None) -> int:
    if value is None:
        return DEFAULT_VIDEO_QUALITY
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 100:
        raise AnimatedOutputError("quality must be an integer between 1 and 100")
    return value


def render_animated(
    composition: Composition | Mapping[str, Any],
    output: str | Path = "out.gif",
    *,
    output_format: str | VideoOutputFormat | None = None,
    loop: int | bool = 0,
    max_bytes: int | None = None,
    fps: float | None = None,
    quality: int | None = None,
    ffmpeg_executable: str = "ffmpeg",
    ffprobe_executable: str = "ffprobe",
    timeout_seconds: float = DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS,
    ffprobe_timeout_seconds: float = DEFAULT_MEDIA_FFPROBE_TIMEOUT_SECONDS,
    ffmpeg_runner: FFmpegRunner | None = None,
    ffprobe_runner: FFprobeRunner | None = None,
    **render_options: Any,
) -> AnimatedRenderResult:
    """Render *composition* to an optimized GIF or animated WebP file.

    ``loop`` follows the GIF and WebP convention where ``0`` loops forever,
    ``-1`` plays once and a positive integer repeats that many times. A boolean
    is accepted as a convenience (``True`` means forever, ``False`` means once).

    When ``max_bytes`` is set, quality is lowered and then the frame size and
    rate are reduced until the file fits. If no attempt fits,
    :class:`AnimatedSizeError` is raised and *output* is left untouched.
    """
    destination = Path(output)
    fmt = _format(destination, output_format)
    loop_value = _loop(loop)
    container_loop = _container_loop(fmt, loop_value)
    cap = None if max_bytes is None else _positive_int(max_bytes, "max_bytes")
    fps_value = None if fps is None else _positive_number(fps, "fps")
    quality_value = _quality(quality)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=f".{destination.stem}-anim-",
        dir=destination.parent,
    ) as directory:
        source = Path(directory) / f"{destination.stem or 'animation'}.mp4"
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
            raise AnimatedOutputError("the intermediate render was not a video")
        try:
            convert_video(
                video.output,
                destination,
                output_format=fmt,
                max_bytes=cap,
                fps=fps_value,
                loop=container_loop,
                quality=quality_value,
                keep_audio=False,
                ffmpeg_executable=ffmpeg_executable,
                ffmpeg_runner=ffmpeg_runner,
                timeout_seconds=timeout_seconds,
            )
        except VideoSizeError as exc:
            raise AnimatedSizeError(exc.message, details=exc.details) from exc

    return AnimatedRenderResult(
        output=destination,
        format=fmt,
        loop=loop_value,
        size_bytes=destination.stat().st_size,
        max_bytes=cap,
        fps=fps_value if fps_value is not None else DEFAULT_MEDIA_ANIMATED_FRAME_RATE,
        quality=quality_value,
        video=video,
    )


def render_gif(
    composition: Composition | Mapping[str, Any],
    output: str | Path = "out.gif",
    **options: Any,
) -> AnimatedRenderResult:
    """Render *composition* to an optimized GIF."""
    return render_animated(
        composition,
        output,
        output_format=VideoOutputFormat.GIF,
        **options,
    )


def render_webp(
    composition: Composition | Mapping[str, Any],
    output: str | Path = "out.webp",
    **options: Any,
) -> AnimatedRenderResult:
    """Render *composition* to an optimized animated WebP."""
    return render_animated(
        composition,
        output,
        output_format=VideoOutputFormat.WEBP,
        **options,
    )


render_animation = render_animated
render_animated_output = render_animated
export_animation = render_animated
export_animated = render_animated
export_gif = render_gif
export_webp = render_webp

AnimatedOutput = AnimatedRenderResult
AnimationOutput = AnimatedRenderResult
AnimationRenderResult = AnimatedRenderResult
