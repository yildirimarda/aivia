"""Render a fast preview at reduced resolution and frame rate.

A preview keeps the composition's exact duration, layers and audio, but renders
fewer frames on a smaller canvas. The canvas and every piece of absolute layer
geometry are scaled before compositing, so no layer is ever rendered at the full
source resolution and then downscaled. The result matches the full render apart
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
from .schema import Composition, FilterType, LayerType
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
    "preview_composition",
    "preview_settings",
    "render_preview",
    "scale_composition",
    "scaled_composition",
]

PREVIEW_RENDERER_VERSION = "2"
"""The version of the preview render pipeline."""

DEFAULT_PREVIEW_SCALE = DEFAULT_MEDIA_PREVIEW_SCALE
"""The default fraction applied to both canvas dimensions and frame rate."""

DEFAULT_PREVIEW_MAX_FPS = DEFAULT_MEDIA_PREVIEW_MAX_FPS
"""The frame rate a preview never exceeds unless a caller asks for more."""

_MIN_PREVIEW_DIMENSION = 2
"""The smallest even canvas dimension an encoded preview can use."""

_MIN_PREVIEW_FRAME_RATE = 1
"""The smallest frame rate a preview can use."""

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
        return (self.width * self.height) / (self.source_width * self.source_height)

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


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _scaled_length(value: Any, factor: float) -> Any:
    number = _number(value)
    return value if number is None else number * factor


def _scaled_axis(
    value: Any,
    factor: float,
    *,
    minimum: int | None = None,
) -> Any:
    number = _number(value)
    if number is None:
        return value
    scaled = round(number * factor)
    if minimum is not None:
        scaled = max(minimum, scaled)
    return scaled


def _scaled_geometry_pair(
    value: Any,
    x_factor: float,
    y_factor: float,
    *,
    minimum: int | None = None,
) -> Any:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return value
    x = _number(value[0])
    y = _number(value[1])
    if x is None or y is None:
        return value
    first = round(x * x_factor)
    second = round(y * y_factor)
    if minimum is not None:
        first = max(minimum, first)
        second = max(minimum, second)
    return (first, second)


def _scaled_length_pair(
    value: Any,
    x_factor: float,
    y_factor: float,
) -> Any:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return value
    x = _number(value[0])
    y = _number(value[1])
    if x is None or y is None:
        return value
    return (x * x_factor, y * y_factor)


def _scaled_position_frames(
    frames: Any,
    x_factor: float,
    y_factor: float,
) -> Any:
    if not isinstance(frames, (list, tuple)):
        return frames
    scaled: list[Any] = []
    for frame in frames:
        if not isinstance(frame, Mapping):
            scaled.append(frame)
            continue
        item = dict(frame)
        if "value" in item:
            item["value"] = _scaled_length_pair(
                item["value"],
                x_factor,
                y_factor,
            )
        scaled.append(item)
    return scaled


def _scaled_position_track(
    track: Any,
    x_factor: float,
    y_factor: float,
) -> Any:
    if isinstance(track, Mapping):
        data = dict(track)
        for name in ("keyframes", "frames", "keys", "values"):
            if name in data:
                data[name] = _scaled_position_frames(
                    data[name],
                    x_factor,
                    y_factor,
                )
        return data
    return _scaled_position_frames(track, x_factor, y_factor)


def _scaled_motion(
    motion: Any,
    x_factor: float,
    y_factor: float,
) -> Any:
    if not isinstance(motion, Mapping):
        return motion
    data = dict(motion)
    for name in ("start_pan", "end_pan", "drift"):
        if name in data:
            data[name] = _scaled_length_pair(
                data[name],
                x_factor,
                y_factor,
            )
    return data


def _scaled_animation(
    animation: Any,
    x_factor: float,
    y_factor: float,
    length_factor: float,
) -> Any:
    if not isinstance(animation, Mapping):
        return animation
    data = dict(animation)
    if "distance" in data:
        data["distance"] = _scaled_length(data["distance"], length_factor)
    if "offset" in data:
        data["offset"] = _scaled_length_pair(
            data["offset"],
            x_factor,
            y_factor,
        )
    return data


def _scaled_placement(
    placement: Any,
    x_factor: float,
    y_factor: float,
    length_factor: float,
) -> Any:
    if not isinstance(placement, Mapping):
        return placement
    data = dict(placement)
    for name in ("padding", "face_padding"):
        if name in data:
            data[name] = _scaled_length(data[name], length_factor)
    margin = data.get("margin")
    if isinstance(margin, (tuple, list)):
        data["margin"] = _scaled_length_pair(margin, x_factor, y_factor)
    elif margin is not None:
        data["margin"] = _scaled_length(margin, length_factor)
    step = data.get("step")
    if step is not None:
        data["step"] = _scaled_axis(step, length_factor, minimum=1)
    faces = data.get("faces")
    if isinstance(faces, (list, tuple)):
        scaled_faces: list[Any] = []
        for face in faces:
            if not isinstance(face, Mapping):
                scaled_faces.append(face)
                continue
            item = dict(face)
            for name, factor in (
                ("x", x_factor),
                ("y", y_factor),
                ("width", x_factor),
                ("height", y_factor),
            ):
                if name in item:
                    item[name] = _scaled_length(item[name], factor)
            scaled_faces.append(item)
        data["faces"] = scaled_faces
    return data


def _scaled_filters(filters: Any, length_factor: float) -> Any:
    if not isinstance(filters, (list, tuple)):
        return filters
    scaled: list[Any] = []
    for item in filters:
        if not isinstance(item, Mapping):
            scaled.append(item)
            continue
        data = dict(item)
        kind = data.get("type")
        if data.get("radius") is not None:
            data["radius"] = _scaled_length(data["radius"], length_factor)
        elif (
            kind in (FilterType.BLUR, FilterType.BOX_BLUR, "blur", "box_blur")
            and data.get("amount") is not None
        ):
            data["amount"] = _scaled_length(data["amount"], length_factor)
        scaled.append(data)
    return scaled


def _scaled_mask(mask: Any, length_factor: float) -> Any:
    if not isinstance(mask, Mapping):
        return mask
    data = dict(mask)
    if "feather" in data:
        data["feather"] = _scaled_length(data["feather"], length_factor)
    return data


def _scaled_layer(
    layer: Any,
    x_factor: float,
    y_factor: float,
    length_factor: float,
) -> Any:
    if not isinstance(layer, Mapping):
        return layer
    data = dict(layer)
    if "x" in data:
        data["x"] = _scaled_axis(data["x"], x_factor)
    if "y" in data:
        data["y"] = _scaled_axis(data["y"], y_factor)
    if "width" in data:
        data["width"] = _scaled_axis(data["width"], x_factor, minimum=1)
    if "height" in data:
        data["height"] = _scaled_axis(data["height"], y_factor, minimum=1)
    if "position" in data:
        data["position"] = _scaled_geometry_pair(
            data["position"],
            x_factor,
            y_factor,
        )
    if "size" in data:
        data["size"] = _scaled_geometry_pair(
            data["size"],
            x_factor,
            y_factor,
            minimum=1,
        )
    if "position_keyframes" in data:
        data["position_keyframes"] = _scaled_position_track(
            data["position_keyframes"],
            x_factor,
            y_factor,
        )
    keyframes = data.get("keyframes")
    if isinstance(keyframes, Mapping):
        tracks = dict(keyframes)
        if "position" in tracks:
            tracks["position"] = _scaled_position_track(
                tracks["position"],
                x_factor,
                y_factor,
            )
        data["keyframes"] = tracks
    if "motion" in data:
        data["motion"] = _scaled_motion(data["motion"], x_factor, y_factor)
    for name in ("entrance", "exit"):
        if name in data:
            data[name] = _scaled_animation(
                data[name],
                x_factor,
                y_factor,
                length_factor,
            )
    if "filters" in data:
        data["filters"] = _scaled_filters(data["filters"], length_factor)
    if "mask" in data:
        data["mask"] = _scaled_mask(data["mask"], length_factor)

    layer_type = data.get("type")
    if layer_type is LayerType.TEXT:
        for name in (
            "font_size",
            "min_font_size",
            "letter_spacing",
            "stroke_width",
        ):
            if name in data:
                data[name] = _scaled_length(data[name], length_factor)
        if "placement" in data:
            data["placement"] = _scaled_placement(
                data["placement"],
                x_factor,
                y_factor,
                length_factor,
            )
    elif layer_type is LayerType.SHAPE:
        for name in ("stroke_width", "radius"):
            if name in data:
                data[name] = _scaled_length(data[name], length_factor)
    elif layer_type in (LayerType.STICKER, LayerType.LOGO):
        margin = data.get("margin")
        if isinstance(margin, (tuple, list)):
            data["margin"] = _scaled_length_pair(margin, x_factor, y_factor)
        elif margin is not None:
            data["margin"] = _scaled_length(margin, length_factor)
        if "shadow_offset" in data:
            data["shadow_offset"] = _scaled_length_pair(
                data["shadow_offset"],
                x_factor,
                y_factor,
            )
        if "shadow_blur" in data:
            data["shadow_blur"] = _scaled_length(
                data["shadow_blur"],
                length_factor,
            )
    return data


def _scaled_subtitles(
    subtitles: Any,
    length_factor: float,
) -> Any:
    if not isinstance(subtitles, Mapping):
        return subtitles
    data = dict(subtitles)
    style = data.get("style")
    if style is None or isinstance(style, str):
        from aivia.media.subtitles import resolve_style

        style = resolve_style(style).model_dump(by_alias=False)
    if isinstance(style, Mapping):
        scaled = dict(style)
        for name in (
            "font_size",
            "outline",
            "shadow",
            "spacing",
            "margin_l",
            "margin_r",
            "margin_v",
        ):
            if name in scaled:
                value = _number(scaled[name])
                if value is not None:
                    scaled[name] = max(1, round(value * length_factor))
        data["style"] = scaled
    return data


def scale_composition(
    composition: Composition | Mapping[str, Any],
    width: int | PreviewSettings,
    height: int | None = None,
    *,
    fps: int | None = None,
) -> Composition:
    """Return *composition* reduced to *width* by *height* pixels.

    The canvas and every absolute layer measurement are scaled: position, size,
    text styles, masks, filters, motion pans and position keyframes, overlay
    margins and shadows. Relative values such as opacity, scale, rotation and
    fractions are left untouched, so the reduced composition composites the same
    scene at a smaller resolution instead of scaling a finished frame.

    A :class:`PreviewSettings` may be passed as *width* instead of a separate
    *height* and *fps*.
    """
    value = (
        composition
        if isinstance(composition, Composition)
        else Composition.model_validate(dict(composition))
    )
    if isinstance(width, PreviewSettings):
        target_width = width.width
        target_height = width.height
        target_fps = width.fps if fps is None else fps
    else:
        if height is None:
            raise PreviewError("height is required when width is an integer")
        target_width = width
        target_height = height
        target_fps = value.fps if fps is None else fps
    if target_width <= 0 or target_height <= 0:
        raise PreviewError("preview dimensions must be positive")
    if target_fps < _MIN_PREVIEW_FRAME_RATE:
        raise PreviewError("preview frame rate must be positive")

    x_factor = target_width / value.canvas.width
    y_factor = target_height / value.canvas.height
    length_factor = math.sqrt(x_factor * y_factor)
    data = value.model_dump(by_alias=False)
    data["canvas"] = {
        **data["canvas"],
        "width": target_width,
        "height": target_height,
    }
    data["fps"] = target_fps
    data["layers"] = [
        _scaled_layer(layer, x_factor, y_factor, length_factor)
        for layer in data.get("layers", [])
    ]
    if "subtitles" in data:
        data["subtitles"] = _scaled_subtitles(
            data["subtitles"],
            length_factor,
        )
    return Composition.model_validate(data)


preview_composition = scale_composition
"""Alias for :func:`scale_composition`."""

scaled_composition = scale_composition
"""Alias for :func:`scale_composition`."""


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
    *video*, *platform* or *profile* is supplied. The canvas and layer geometry
    are then scaled to the preview size so compositing happens on the reduced
    canvas. The remaining ``command_options`` are forwarded to
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
    preview = scale_composition(value, settings)

    result = render_video(
        preview,
        output,
        ffmpeg_executable=ffmpeg_executable,
        ffprobe_executable=ffprobe_executable,
        timeout_seconds=timeout_seconds,
        ffprobe_timeout_seconds=ffprobe_timeout_seconds,
        ffmpeg_runner=ffmpeg_runner,
        ffprobe_runner=ffprobe_runner,
        **options,
    )
    if not isinstance(result, VideoRenderResult):
        raise PreviewError("a preview must render to a video")
    return PreviewRenderResult(output=result.output, settings=settings, video=result)
