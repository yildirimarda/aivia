"""Render a composition to a deterministic video file with FFmpeg.

Command construction lives in :mod:`aivia.compose.ffmpeg_cmd` and stays pure.
This module is the thin execution layer around it: it rasterizes text and
shape layers plus procedural overlay effects with Pillow, runs the built argv,
writes through a temporary file so a failed render cannot leave a partial
output, and
verifies the result with ffprobe before publishing it. Both runners are
injectable so the offline suite never needs the real binaries.
"""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

from aivia.core.config.media import (
    DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS,
    DEFAULT_MEDIA_FFPROBE_TIMEOUT_SECONDS,
)
from aivia.core.config.sections import RenderConfig
from aivia.core.errors import ValidationError

from .audio_subs import subtitle_for, write_subtitle_document
from .contrast import parse_color
from .errors import RenderError
from .ffmpeg_cmd import (
    FFmpegCommand,
    FFmpegOverlayFrames,
    _as_render_config,
    _text_background,
    build_ffmpeg_command,
)
from .keyframes import AnimatedProperty, Easing
from .motion import MotionTracks, tracks_for_layer
from .output_profiles import (
    OutputProfile,
    apply_output_profile,
    resolve_output_profile,
)
from .overlays import apply_overlay_effects, overlay_from_effect
from .render import AssetResolver, PillowStillRenderer
from .schema import Composition, Layer, LayerType

if TYPE_CHECKING:
    from aivia.core.media_probe import FFprobeInfo, FFprobeRunner

    from .anim_outputs import AnimatedRenderResult
    from .audio_outputs import AudioRenderResult

__all__ = [
    "VIDEO_RENDERER_VERSION",
    "FFmpegRunner",
    "VideoRenderError",
    "VideoRenderResult",
    "VideoRenderer",
    "render_video",
]

VIDEO_RENDERER_VERSION = "2"
"""The version of the FFmpeg video renderer, recorded with every output."""

FFmpegRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]
"""A callable that runs an ffmpeg argv and returns the completed process."""


class VideoRenderError(RenderError):
    """A composition could not be rendered to a verified video."""

    code = "video_render"


@dataclass(frozen=True, slots=True)
class VideoRenderResult:
    """A rendered video, the command that produced it and its ffprobe result."""

    output: Path
    command: FFmpegCommand
    probe: FFprobeInfo


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VideoRenderError(f"{name} must be a positive number")
    number = float(value)
    if not math.isfinite(number) or not number > 0.0:
        raise VideoRenderError(f"{name} must be a positive number")
    return number


def _frame_rate(value: str | None) -> float | None:
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


def _raster_layer_data(layer: Layer) -> dict[str, Any]:
    data = layer.model_dump(by_alias=False, exclude_none=True)
    for key in (
        "entrance",
        "exit",
        "entrance_animation",
        "exit_animation",
        "text_animation",
        "animation",
        "animations",
        "keyframes",
        "motion",
        "mask",
        "filters",
        "blend_mode",
        "transition",
        "transition_in",
        "enter_transition",
        "in_transition",
    ):
        data.pop(key, None)
    data["type"] = layer.type.value
    data["start_seconds"] = 0.0
    data["opacity"] = 1.0
    data["visible"] = True
    data["z_index"] = 0
    data.pop("end_seconds", None)
    data.pop("duration_seconds", None)
    return data


@dataclass(frozen=True, slots=True)
class _RasterExpansion:
    """How far a rasterized layer's base box must grow to cover its motion."""

    scale_x: float = 1.0
    scale_y: float = 1.0
    rotation: tuple[float, float] | None = None


def _raster_expansion(
    layer: Layer,
    start_seconds: float,
    end_seconds: float,
) -> _RasterExpansion:
    tracks = tracks_for_layer(layer, start_seconds, end_seconds)
    maximum_x = maximum_y = 1.0
    if tracks.scale is not None:
        for frame in tracks.scale.keyframes:
            value = frame.value
            if isinstance(value, tuple):
                scale_x, scale_y = value
            else:
                scale_x = scale_y = value
            maximum_x = max(maximum_x, float(scale_x))
            maximum_y = max(maximum_y, float(scale_y))
    rotation = None
    if tracks.rotation is not None:
        angles: list[float] = []
        for frame in tracks.rotation.keyframes:
            value = frame.value
            if not isinstance(value, tuple):
                angles.append(float(value))
        if angles:
            rotation = (min(angles), max(angles))
    return _RasterExpansion(maximum_x, maximum_y, rotation)


def _max_rotated_extent(
    horizontal: float,
    vertical: float,
    low_degrees: float,
    high_degrees: float,
) -> float:
    """Return the widest projection of a box over an interval of angles.

    A rectangle of *horizontal* by *vertical* size projects to
    ``horizontal * |cos(angle)| + vertical * |sin(angle)|``. The projection
    repeats every 180 degrees, so an interval that spans a half turn already
    covers every orientation. Otherwise only the interval ends and the angles
    where the projection peaks can be maxima.
    """
    span = high_degrees - low_degrees
    if span >= 180.0:
        return math.hypot(horizontal, vertical)
    period = math.pi
    low = math.radians(low_degrees) % period
    width = math.radians(span)
    if low + width <= period:
        intervals = [(low, low + width)]
    else:
        intervals = [(low, period), (0.0, low + width - period)]
    critical = math.atan2(vertical, horizontal)
    best = 0.0
    for start, end in intervals:
        candidates = [start, end]
        for angle in (critical, period - critical):
            shift = math.floor((start - angle) / period)
            for step in (shift, shift + 1):
                candidate = angle + step * period
                if start - 1e-12 <= candidate <= end + 1e-12:
                    candidates.append(candidate)
        for candidate in candidates:
            projection = horizontal * abs(math.cos(candidate)) + vertical * abs(
                math.sin(candidate)
            )
            best = max(best, projection)
    return best


def _expand_box(
    box: tuple[int, int, int, int],
    expansion: _RasterExpansion,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    width = right - left
    height = bottom - top
    scaled_width = width * expansion.scale_x
    scaled_height = height * expansion.scale_y
    target_width = scaled_width
    target_height = scaled_height
    if expansion.rotation is not None:
        low, high = expansion.rotation
        target_width = _max_rotated_extent(scaled_width, scaled_height, low, high)
        target_height = _max_rotated_extent(scaled_height, scaled_width, low, high)
    target_width = max(float(width), target_width)
    target_height = max(float(height), target_height)
    if target_width == width and target_height == height:
        return box
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    return (
        math.floor(center_x - target_width / 2.0),
        math.floor(center_y - target_height / 2.0),
        math.ceil(center_x + target_width / 2.0),
        math.ceil(center_y + target_height / 2.0),
    )


def _rasterize_layer(
    composition: Composition,
    layer: Layer,
    directory: Path,
    index: int,
    *,
    render: RenderConfig | None = None,
    contrast_background: Image.Image | None = None,
    expand: _RasterExpansion | None = None,
) -> tuple[Path, tuple[int, int, int, int]] | None:
    if expand is None:
        expand = _RasterExpansion()
    static = Composition.model_validate(
        {
            "canvas": {
                "width": composition.canvas.width,
                "height": composition.canvas.height,
                "background": "#00000000",
            },
            "duration_seconds": composition.duration_seconds,
            "fps": composition.fps,
            "platform": composition.platform,
            "safe_area": composition.safe_area,
            "layers": [_raster_layer_data(layer)],
        }
    )
    image = PillowStillRenderer(render=render).render_still(
        static,
        0.0,
        contrast_background=contrast_background,
    )
    box = image.getchannel("A").getbbox()
    if box is None:
        return None
    box = _expand_box(box, expand)
    cropped = image.crop(box)
    source = directory / f"{layer.type.value}-{index}.png"
    cropped.save(source)
    return source, box


def _overlay_windows(composition: Composition) -> list[tuple[int, int]]:
    fps = composition.fps
    total = max(1, composition.frame_count)
    windows: list[tuple[int, int]] = []
    for effect in composition.effects:
        if not effect.enabled or overlay_from_effect(effect) is None:
            continue
        start = effect.start_seconds
        end = (
            composition.duration_seconds
            if effect.end_seconds is None
            else min(effect.end_seconds, composition.duration_seconds)
        )
        if end <= start or start >= composition.duration_seconds:
            continue
        first = max(0, math.ceil(start * fps - 1e-9))
        last = min(total - 1, math.ceil(end * fps - 1e-9) - 1)
        if last < first:
            continue
        windows.append((first, last))
    windows.sort()
    merged: list[tuple[int, int]] = []
    for first, last in windows:
        if merged and first <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], last))
        else:
            merged.append((first, last))
    return merged


def _prepare_overlay_frames(
    composition: Composition,
    directory: Path,
) -> tuple[FFmpegOverlayFrames, ...]:
    windows = _overlay_windows(composition)
    if not windows:
        return ()
    root = directory / "overlay-frames"
    size = composition.canvas.size
    sequences: list[FFmpegOverlayFrames] = []
    for offset, (first, last) in enumerate(windows):
        frames = root / f"{offset:04d}"
        frames.mkdir(parents=True, exist_ok=True)
        for index in range(first, last + 1):
            frame = Image.new("RGBA", size, (0, 0, 0, 0))
            apply_overlay_effects(frame, composition, index / composition.fps)
            frame.save(frames / f"{index - first:04d}.png")
        sequences.append(
            FFmpegOverlayFrames(
                pattern=str(frames / "%04d.png"),
                fps=composition.fps,
                duration_seconds=(last - first + 1) / composition.fps,
                start_seconds=first / composition.fps,
            )
        )
    return tuple(sequences)


def _easing_at(tracks: MotionTracks, time_seconds: float) -> Easing:
    for track in (tracks.position, tracks.pan, tracks.scale):
        if track is None:
            continue
        for frame in track.keyframes:
            if frame.time_seconds == time_seconds:
                return frame.easing
    return Easing.LINEAR


def _pan_position_keyframes(tracks: MotionTracks) -> list[dict[str, Any]] | None:
    pan = tracks.pan
    if pan is None:
        return None
    times = sorted(
        {
            frame.time_seconds
            for track in (tracks.position, pan, tracks.scale)
            if track is not None
            for frame in track.keyframes
        }
    )
    keyframes: list[dict[str, Any]] = []
    for time_seconds in times:
        position = (
            tracks.position.sample(time_seconds)
            if tracks.position is not None
            else (0.0, 0.0)
        )
        position_x, position_y = position if isinstance(position, tuple) else (0.0, 0.0)
        scale = tracks.scale.sample(time_seconds) if tracks.scale is not None else 1.0
        if isinstance(scale, tuple):
            scale_x, scale_y = scale
        else:
            scale_x = scale_y = float(scale)
        offset = pan.sample(time_seconds)
        offset_x, offset_y = offset if isinstance(offset, tuple) else (0.0, 0.0)
        keyframes.append(
            {
                "time": time_seconds,
                "value": (
                    position_x - offset_x * scale_x,
                    position_y - offset_y * scale_y,
                ),
                "easing": _easing_at(tracks, time_seconds).value,
            }
        )
    return keyframes


def _keyframe_mapping(
    layer: Layer,
    start_seconds: float,
    end_seconds: float,
    *,
    pan_as_position: bool = False,
) -> dict[str, Any]:
    tracks = tracks_for_layer(layer, start_seconds, end_seconds)
    mapping: dict[str, Any] = {}
    for property_name in AnimatedProperty:
        track = getattr(tracks, property_name.value, None)
        if track is None:
            continue
        mapping[property_name.value] = [
            {
                "time": frame.time_seconds,
                "value": frame.value,
                "easing": frame.easing.value,
            }
            for frame in track.keyframes
        ]
    if pan_as_position:
        position = _pan_position_keyframes(tracks)
        if position is not None:
            mapping["position"] = position
    return mapping


_RASTER_TYPES = frozenset({LayerType.TEXT, LayerType.SHAPE})


def _prepare_raster_layers(
    composition: Composition,
    directory: Path,
    *,
    render: RenderConfig | Mapping[str, Any] | None = None,
    assets: AssetResolver | None = None,
) -> Composition:
    if not any(layer.type in _RASTER_TYPES for layer in composition.layers):
        return composition
    render_config = _as_render_config(render)
    layers: list[Any] = []
    for index, layer in enumerate(composition.layers):
        if layer.type not in _RASTER_TYPES:
            layers.append(layer)
            continue
        end = layer.end if layer.end is not None else composition.duration_seconds
        if not layer.visible or end <= layer.start_seconds:
            continue
        contrast_background = None
        if layer.type is LayerType.TEXT:
            color = _text_background(
                composition,
                layer,
                composition.layers[:index],
                render_config,
                assets,
            )
            contrast_background = Image.new(
                "RGBA",
                composition.canvas.size,
                parse_color(color),
            )
        expand = _raster_expansion(layer, layer.start_seconds, end)
        raster = _rasterize_layer(
            composition,
            layer,
            directory,
            index,
            render=render_config,
            contrast_background=contrast_background,
            expand=expand,
        )
        if raster is None:
            continue
        source, box = raster
        mapping: dict[str, Any] = {
            "id": layer.id,
            "type": LayerType.IMAGE.value,
            "source": str(source),
            "x": box[0],
            "y": box[1],
            "width": box[2] - box[0],
            "height": box[3] - box[1],
            "fit": "none",
            "z_index": layer.z_index,
            "opacity": layer.opacity,
            "visible": True,
            "start_seconds": layer.start_seconds,
            "end_seconds": end,
        }
        keyframes = _keyframe_mapping(
            layer,
            layer.start_seconds,
            end,
            pan_as_position=layer.type is LayerType.SHAPE,
        )
        if keyframes:
            mapping["keyframes"] = keyframes
        if layer.transition is not None:
            mapping["transition"] = layer.transition.model_dump(by_alias=False)
        layers.append(mapping)
    data = composition.model_dump(by_alias=False)
    data["layers"] = layers
    return Composition.model_validate(data)


class VideoRenderer:
    """Execute the pure ffmpeg command and verify the resulting video."""

    version = VIDEO_RENDERER_VERSION

    def __init__(
        self,
        *,
        ffmpeg_executable: str = "ffmpeg",
        ffprobe_executable: str = "ffprobe",
        timeout_seconds: float = DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS,
        ffprobe_timeout_seconds: float = DEFAULT_MEDIA_FFPROBE_TIMEOUT_SECONDS,
        ffmpeg_runner: FFmpegRunner | None = None,
        ffprobe_runner: FFprobeRunner | None = None,
    ) -> None:
        if not isinstance(ffmpeg_executable, str) or not ffmpeg_executable.strip():
            raise VideoRenderError("ffmpeg_executable must be a non-empty string")
        if not isinstance(ffprobe_executable, str) or not ffprobe_executable.strip():
            raise VideoRenderError("ffprobe_executable must be a non-empty string")
        self.ffmpeg_executable = ffmpeg_executable.strip()
        self.ffprobe_executable = ffprobe_executable.strip()
        self.timeout_seconds = _positive_number(timeout_seconds, "timeout_seconds")
        self.ffprobe_timeout_seconds = _positive_number(
            ffprobe_timeout_seconds, "ffprobe_timeout_seconds"
        )
        self.ffmpeg_runner = ffmpeg_runner
        self.ffprobe_runner = ffprobe_runner

    def _run(self, argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        if self.ffmpeg_runner is not None:
            return self.ffmpeg_runner(argv)
        try:
            return subprocess.run(
                list(argv),
                capture_output=True,
                text=True,
                check=False,
                timeout=self.timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise VideoRenderError(
                f"ffmpeg is not available: {argv[0]}",
                details={"executable": argv[0]},
            ) from exc
        except (OSError, subprocess.SubprocessError) as exc:
            raise VideoRenderError(
                "ffmpeg could not render the composition",
                details={"executable": argv[0], "error": type(exc).__name__},
            ) from exc

    def _verify(
        self,
        composition: Composition,
        probe: FFprobeInfo,
        *,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        stream = probe.primary_video
        expected_width = composition.canvas.width if width is None else width
        expected_height = composition.canvas.height if height is None else height
        if stream is None:
            raise VideoRenderError("the rendered file has no video stream")
        if stream.width != expected_width:
            raise VideoRenderError(
                "the rendered video width does not match the canvas",
                details={
                    "expected": expected_width,
                    "actual": stream.width,
                },
            )
        if stream.height != expected_height:
            raise VideoRenderError(
                "the rendered video height does not match the canvas",
                details={
                    "expected": expected_height,
                    "actual": stream.height,
                },
            )
        tolerance = max(0.25, 2.0 / composition.fps)
        duration = probe.duration_seconds
        if duration is None or abs(duration - composition.duration_seconds) > tolerance:
            raise VideoRenderError(
                "the rendered video duration does not match the composition",
                details={
                    "expected": composition.duration_seconds,
                    "actual": duration,
                },
            )
        rate = _frame_rate(stream.frame_rate)
        if rate is not None and abs(rate - composition.fps) > 0.01:
            raise VideoRenderError(
                "the rendered video frame rate does not match the composition",
                details={
                    "expected": composition.fps,
                    "actual": rate,
                },
            )
        if any(track.enabled for track in composition.audio_tracks) and not (
            probe.audio_streams
        ):
            raise VideoRenderError(
                "the rendered video has no audio stream for its audio tracks"
            )

    def render(
        self,
        composition: Composition | Mapping[str, Any],
        output: str | Path = "out.mp4",
        **command_options: Any,
    ) -> VideoRenderResult:
        """Render *composition* to *output* and verify it with ffprobe.

        ``command_options`` are forwarded to
        :func:`~aivia.compose.ffmpeg_cmd.build_ffmpeg_command`, except the
        executable which comes from the renderer.
        """
        value = (
            composition
            if isinstance(composition, Composition)
            else Composition.model_validate(dict(composition))
        )
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        suffix = destination.suffix or ".mp4"
        handle, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.stem}-",
            suffix=suffix,
            dir=destination.parent,
        )
        os.close(handle)
        temporary = Path(temporary_name)
        text_directory = tempfile.TemporaryDirectory(
            prefix=f".{destination.stem}-text-",
            dir=destination.parent,
        )
        try:
            options = dict(command_options)
            options.pop("executable", None)
            output_width = options.pop("output_width", None)
            output_height = options.pop("output_height", None)
            output_profile = options.pop("output_profile", None)
            config = options.pop("config", options.pop("video", None))
            profile_name = options.pop("profile", None)
            platform = options.pop("platform", None)
            if output_profile is not None or config is not None:
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
                value = apply_output_profile(value, resolved)
                options.setdefault("video_codec", resolved.codec)
                options.setdefault("audio_codec", resolved.audio_codec)
                options.setdefault("loudness_lufs", resolved.loudness_lufs)
                if resolved.bitrate is not None:
                    options.setdefault("bitrate", resolved.bitrate)
                if resolved.audio_bitrate is not None:
                    options.setdefault("audio_bitrate", resolved.audio_bitrate)
            render_config = options.get("render")
            if render_config is None and config is not None:
                render_config = getattr(config, "render", None)
                if render_config is None and isinstance(config, Mapping):
                    render_config = config.get("render")
            if render_config is not None:
                options.setdefault("render", render_config)
            value = _prepare_raster_layers(
                value,
                Path(text_directory.name),
                render=render_config,
                assets=options.get("assets"),
            )
            overlay_frames = _prepare_overlay_frames(value, Path(text_directory.name))
            subtitles = options.pop("subtitles", None)
            subtitle_file = options.pop(
                "subtitle_file",
                options.pop("subtitle_path", None),
            )
            safe_area = options.pop("safe_area", None)
            subtitle_preset = options.pop("subtitle_preset", None)
            highlight = options.pop("highlight", True)
            track = subtitle_for(value, subtitles)
            subtitle_options: dict[str, Any] = {
                "width": value.canvas.width,
                "height": value.canvas.height,
                "safe_area": (value.safe_area if safe_area is None else safe_area),
                "platform": platform or value.platform or None,
                "preset": subtitle_preset,
                "highlight": highlight,
                "background": value.canvas.background,
                "render": render_config,
            }
            referenced = value.subtitle_file.strip()
            if (
                subtitle_file is None
                and track is not None
                and (track.cues or not referenced)
            ):
                subtitle_file = Path(text_directory.name) / f"{destination.stem}.ass"
                write_subtitle_document(track, subtitle_file, **subtitle_options)
            elif subtitle_file is None and referenced:
                subtitle_file = Path(referenced)
            if subtitle_file is not None and (
                Path(subtitle_file).suffix.lower() in {".srt", ".vtt"}
            ):
                converted = Path(text_directory.name) / f"{destination.stem}.ass"
                conversion_options = dict(subtitle_options)
                if track is not None and not track.cues:
                    conversion_options["style"] = track.style
                write_subtitle_document(
                    subtitle_file,
                    converted,
                    **conversion_options,
                )
                subtitle_file = converted
            if subtitle_file is not None:
                options["subtitle_file"] = subtitle_file
            command = build_ffmpeg_command(
                value,
                temporary,
                executable=self.ffmpeg_executable,
                output_width=output_width,
                output_height=output_height,
                overlay_frames=overlay_frames,
                **options,
            )
            completed = self._run(command.argv)
            if completed.returncode != 0:
                raise VideoRenderError(
                    "ffmpeg could not render the composition",
                    details={
                        "returncode": completed.returncode,
                        "stderr": _text(completed.stderr)[:500],
                    },
                )
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise VideoRenderError("ffmpeg produced no output")
            from aivia.core.media_probe import run_ffprobe

            try:
                probe = run_ffprobe(
                    temporary,
                    executable=self.ffprobe_executable,
                    timeout_seconds=self.ffprobe_timeout_seconds,
                    runner=self.ffprobe_runner,
                )
            except ValidationError as exc:
                raise VideoRenderError(
                    "the rendered file could not be inspected",
                    details={"error": exc.message},
                ) from exc
            self._verify(
                value,
                probe,
                width=output_width,
                height=output_height,
            )
            os.replace(temporary, destination)
        finally:
            text_directory.cleanup()
            temporary.unlink(missing_ok=True)
        return VideoRenderResult(output=destination, command=command, probe=probe)


def render_video(
    composition: Composition | Mapping[str, Any],
    output: str | Path = "out.mp4",
    *,
    ffmpeg_executable: str = "ffmpeg",
    ffprobe_executable: str = "ffprobe",
    timeout_seconds: float = DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS,
    ffprobe_timeout_seconds: float = DEFAULT_MEDIA_FFPROBE_TIMEOUT_SECONDS,
    ffmpeg_runner: FFmpegRunner | None = None,
    ffprobe_runner: FFprobeRunner | None = None,
    **command_options: Any,
) -> VideoRenderResult | AnimatedRenderResult | AudioRenderResult:
    """Render *composition* to MP4, or to GIF/WebP/audio when the suffix asks."""
    suffix = Path(output).suffix.lower()
    if suffix in {".gif", ".webp"}:
        from .anim_outputs import render_animated

        animated: dict[str, Any] = {}
        for name in ("output_format", "loop", "max_bytes", "fps", "quality"):
            if name in command_options:
                animated[name] = command_options.pop(name)
        return render_animated(
            composition,
            output,
            ffmpeg_executable=ffmpeg_executable,
            ffprobe_executable=ffprobe_executable,
            ffmpeg_runner=ffmpeg_runner,
            ffprobe_runner=ffprobe_runner,
            timeout_seconds=timeout_seconds,
            **animated,
            **command_options,
        )
    if suffix in {".mp3", ".m4a", ".wav", ".flac", ".ogg", ".oga", ".opus", ".aac"}:
        from .audio_outputs import render_audio

        allowed = {
            "output_format",
            "metadata",
            "audio_codec",
            "audio_bitrate",
            "sample_rate",
            "channels",
            "normalize_loudness",
            "ducking",
            "audio_settings",
            "loudness_lufs",
            "output_profile",
            "config",
            "video",
            "platform",
            "profile",
            "overwrite",
        }
        audio = {
            name: command_options[name] for name in allowed if name in command_options
        }
        return render_audio(
            composition,
            output,
            ffmpeg_executable=ffmpeg_executable,
            ffprobe_executable=ffprobe_executable,
            ffmpeg_runner=ffmpeg_runner,
            ffprobe_runner=ffprobe_runner,
            timeout_seconds=timeout_seconds,
            **audio,
        )
    renderer = VideoRenderer(
        ffmpeg_executable=ffmpeg_executable,
        ffprobe_executable=ffprobe_executable,
        timeout_seconds=timeout_seconds,
        ffprobe_timeout_seconds=ffprobe_timeout_seconds,
        ffmpeg_runner=ffmpeg_runner,
        ffprobe_runner=ffprobe_runner,
    )
    return renderer.render(composition, output, **command_options)
