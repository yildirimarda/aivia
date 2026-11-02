"""Render a fast preview at reduced resolution and frame rate.

A preview keeps the composition's exact duration, layers and audio, but renders
fewer frames and scales the finished frame down before encoding. Every layer is
composited once per preview frame, so the result matches the full render apart
from size and temporal resolution while doing measurably less work.

The settings are derived from the source composition: a *scale* below one
reduces both the canvas and the frame rate, and optional caps keep a preview
small even for a very large composition. The full output profile, when one is
supplied, is applied before the preview is derived so platform overrides are
honoured.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aivia.core.config.media import (
    DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS,
    DEFAULT_MEDIA_FFPROBE_TIMEOUT_SECONDS,
    DEFAULT_MEDIA_PREVIEW_MAX_FPS,
    DEFAULT_MEDIA_PREVIEW_SCALE,
)

from .errors import RenderError
from .output_profiles import (
    OutputProfile,
    apply_output_profile,
    ffmpeg_options,
    resolve_output_profile,
)
from .schema import Composition
from .video import FFmpegRunner, VideoRenderResult, render_video

if TYPE_CHECKING:
    from aivia.core.media_probe import FFprobeRunner

__all__ = [
    "DEFAULT_PREVIEW_MAX_FPS",
    "DEFAULT_PREVIEW_SCALE",
    "PREVIEW_RENDERER_VERSION",
    "PreviewError",
    "PreviewRenderResult",
    "PreviewSettings",
    "preview_settings",
    "render_preview",
]

PREVIEW_RENDERER_VERSION = "1"
"""The version of the preview render pipeline."""

DEFAULT_PREVIEW_SCALE = DEFAULT_MEDIA_PREVIEW_SCALE
"""The default fraction applied to both canvas dimensions and frame rate."""

DEFAULT_PREVIEW_MAX_FPS = DEFAULT_MEDIA_PREVIEW_MAX_FPS
"""The frame rate a preview never exceeds unless a caller asks for more."""

_MIN_PREVIEW_DIMENSION = 2
"""The smallest even canvas dimension an encoded preview can use."""

_AUDIO_SUFFIXES = frozenset(
    {".mp3", ".m4a", ".wav", ".flac", ".ogg", ".oga", ".opus", ".aac"}
)
_ANIMATED_SUFFIXES = frozenset({".gif", ".webp"})


class PreviewError(RenderError):
    """A preview could not be derived or rendered."""

    code = "preview"


def _scale_number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PreviewError("scale must be a number")
    number = float(value)
    if not math.isfinite(number) or not 0.0 < number <= 1.0:
        raise PreviewError("scale must be greater than zero and at most one")
    return number


def _optional_integer(value: Any, name: str, *, minimum: int = 1) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise PreviewError(f"{name} must be a positive integer")
    if value < minimum:
        raise PreviewError(f"{name} must be at least {minimum}")
    return value


def _even(value: int) -> int:
    return max(_MIN_PREVIEW_DIMENSION, value - value % 2)


@dataclass(frozen=True, slots=True)
class PreviewSettings:
    """The concrete size and frame rate of one preview render."""

    width: int
    height: int
    fps: int
    source_width: int
    source_height: int
    source_fps: int
    scale: float

    def __post_init__(self) -> None:
        for name in (
            "width",
            "height",
            "fps",
            "source_width",
            "source_height",
            "source_fps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise PreviewError(f"{name} must be a positive integer")
        _scale_number(self.scale)
        if self.width > self.source_width or self.height > self.source_height:
            raise PreviewError("preview dimensions must not exceed the source")
        if self.fps > self.source_fps:
            raise PreviewError("preview frame rate must not exceed the source")

    @property
    def resolution(self) -> str:
        """The preview as a ``WIDTHxHEIGHT`` string."""
        return f"{self.width}x{self.height}"

    @property
    def frame_rate(self) -> int:
        """Alias for :attr:`fps`."""
        return self.fps

    @property
    def pixel_ratio(self) -> float:
        """The preview pixels divided by the source pixels."""
        return (self.width * self.height) / (
            self.source_width * self.source_height
        )

    @property
    def frame_ratio(self) -> float:
        """The preview frame count divided by the source frame count."""
        return self.fps / self.source_fps

    @property
    def work_ratio(self) -> float:
        """The relative compositing work of this preview."""
        return self.pixel_ratio * self.frame_ratio

    @property
    def is_reduced(self) -> bool:
        """Whether the preview is smaller or slower than the source."""
        return (
            self.width < self.source_width
            or self.height < self.source_height
            or self.fps < self.source_fps
        )


def preview_settings(
    composition: Composition | Mapping[str, Any],
    *,
    scale: float = DEFAULT_PREVIEW_SCALE,
    max_width: int | None = None,
    max_height: int | None = None,
    max_fps: int | None = DEFAULT_PREVIEW_MAX_FPS,
) -> PreviewSettings:
    """Derive the reduced size and frame rate for *composition*.

    *scale* is applied to both axes and to the frame rate. *max_width*,
    *max_height* and *max_fps* cap the result without ever enlarging it. The
    preview dimensions are even so every common video encoder accepts them.
    """
    value = (
        composition
        if isinstance(composition, Composition)
        else Composition.model_validate(dict(composition))
    )
    factor = _scale_number(scale)
    limit_width = _optional_integer(
        max_width,
        "max_width",
        minimum=_MIN_PREVIEW_DIMENSION,
    )
    limit_height = _optional_integer(
        max_height,
        "max_height",
        minimum=_MIN_PREVIEW_DIMENSION,
    )
    limit_fps = _optional_integer(max_fps, "max_fps")

    width = _even(round(value.canvas.width * factor))
    height = _even(round(value.canvas.height * factor))
    if limit_width is not None:
        width = min(width, _even(limit_width))
    if limit_height is not None:
        height = min(height, _even(limit_height))
    width = min(width, value.canvas.width)
    height = min(height, value.canvas.height)

    fps = max(1, round(value.fps * factor))
    if limit_fps is not None:
        fps = min(fps, limit_fps)
    fps = min(fps, value.fps)

    return PreviewSettings(
        width=width,
        height=height,
        fps=fps,
        source_width=value.canvas.width,
        source_height=value.canvas.height,
        source_fps=value.fps,
        scale=factor,
    )


@dataclass(frozen=True, slots=True)
class PreviewRenderResult:
    """A rendered preview, its settings and the underlying video result."""

    output: Path
    settings: PreviewSettings
    video: VideoRenderResult

    @property
    def path(self) -> Path:
        """Alias for :attr:`output`."""
        return self.output

    @property
    def width(self) -> int:
        """The preview width in pixels."""
        return self.settings.width

    @property
    def height(self) -> int:
        """The preview height in pixels."""
        return self.settings.height

    @property
    def fps(self) -> int:
        """The preview frame rate."""
        return self.settings.fps

    @property
    def command(self) -> Any:
        """The ffmpeg command that produced the preview."""
        return self.video.command

    @property
    def probe(self) -> Any:
        """The ffprobe result of the rendered preview."""
        return self.video.probe

    def __fspath__(self) -> str:
        return str(self.output)

    def __str__(self) -> str:
        return str(self.output)


def _full_composition(
    composition: Composition | Mapping[str, Any],
    options: dict[str, Any],
) -> Composition:
    value = (
        composition
        if isinstance(composition, Composition)
        else Composition.model_validate(dict(composition))
    )
    output_profile = options.pop("output_profile", None)
    config = options.pop("config", options.pop("video", None))
    platform = options.pop("platform", None)
    profile_name = options.pop("profile", None)
    if output_profile is None and config is None:
        return value
    resolved = (
        output_profile
        if isinstance(output_profile, OutputProfile)
        else (
            OutputProfile.from_mapping(output_profile)
            if output_profile is not None
            else resolve_output_profile(
                config,
                platform=platform,
                profile=profile_name,
            )
        )
    )
    for name, option in ffmpeg_options(resolved).items():
        options.setdefault(name, option)
    return apply_output_profile(value, resolved)


def render_preview(
    composition: Composition | Mapping[str, Any],
    output: str | Path = "preview.mp4",
    *,
    scale: float = DEFAULT_PREVIEW_SCALE,
    max_width: int | None = None,
    max_height: int | None = None,
    max_fps: int | None = DEFAULT_PREVIEW_MAX_FPS,
    ffmpeg_executable: str = "ffmpeg",
    ffprobe_executable: str = "ffprobe",
    timeout_seconds: float = DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS,
    ffprobe_timeout_seconds: float = DEFAULT_MEDIA_FFPROBE_TIMEOUT_SECONDS,
    ffmpeg_runner: FFmpegRunner | None = None,
    ffprobe_runner: FFprobeRunner | None = None,
    **command_options: Any,
) -> PreviewRenderResult:
    """Render *composition* to a reduced preview video at *output*.

    The full output profile is applied first when *output_profile*, *config*,
    *video*, *platform* or *profile* is supplied. The remaining
    ``command_options`` are forwarded to
    :func:`~aivia.compose.ffmpeg_cmd.build_ffmpeg_command`. Animated image and
    audio destinations are rejected because a preview is a video.
    """
    suffix = Path(output).suffix.lower()
    if suffix in _ANIMATED_SUFFIXES or suffix in _AUDIO_SUFFIXES:
        raise PreviewError("a preview renders to a video container")

    options = dict(command_options)
    value = _full_composition(composition, options)
    settings = preview_settings(
        value,
        scale=scale,
        max_width=max_width,
        max_height=max_height,
        max_fps=max_fps,
    )
    data = value.model_dump(by_alias=False)
    data["fps"] = settings.fps
    preview = Composition.model_validate(data)

    result = render_video(
        preview,
        output,
        ffmpeg_executable=ffmpeg_executable,
        ffprobe_executable=ffprobe_executable,
        timeout_seconds=timeout_seconds,
        ffprobe_timeout_seconds=ffprobe_timeout_seconds,
        ffmpeg_runner=ffmpeg_runner,
        ffprobe_runner=ffprobe_runner,
        output_width=settings.width,
        output_height=settings.height,
        **options,
    )
    if not isinstance(result, VideoRenderResult):
        raise PreviewError("a preview must render to a video")
    return PreviewRenderResult(output=result.output, settings=settings, video=result)
