"""Render a composition to a deterministic video file with FFmpeg.

Command construction lives in :mod:`aivia.compose.ffmpeg_cmd` and stays pure.
This module is the thin execution layer around it: it rasterizes text layers
and procedural overlay effects with Pillow, runs the built argv, writes
through a temporary file so a failed render cannot leave a partial output, and
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
from aivia.core.errors import ValidationError

from .audio_subs import subtitle_for, write_subtitle_document
from .errors import RenderError
from .ffmpeg_cmd import FFmpegCommand, FFmpegOverlayFrames, build_ffmpeg_command
from .keyframes import AnimatedProperty
from .motion import tracks_for_layer
from .output_profiles import (
    OutputProfile,
    apply_output_profile,
    resolve_output_profile,
)
from .overlays import apply_overlay_effects, overlay_from_effect
from .render import PillowStillRenderer
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


def _text_layer_data(layer: Layer) -> dict[str, Any]:
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
    data["type"] = LayerType.TEXT.value
    data["start_seconds"] = 0.0
    data["opacity"] = 1.0
    data["visible"] = True
    data["z_index"] = 0
    data.pop("end_seconds", None)
    data.pop("duration_seconds", None)
    return data


def _rasterize_text_layer(
    composition: Composition,
    layer: Layer,
    directory: Path,
    index: int,
) -> tuple[Path, tuple[int, int, int, int]] | None:
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
            "layers": [_text_layer_data(layer)],
        }
    )
    image = PillowStillRenderer().render_still(static, 0.0)
    box = image.getchannel("A").getbbox()
    if box is None:
        return None
    cropped = image.crop(box)
    source = directory / f"text-{index}.png"
    cropped.save(source)
    return source, box


def _has_overlay_effects(composition: Composition) -> bool:
    for effect in composition.effects:
        if not effect.enabled:
            continue
        end = (
            composition.duration_seconds
            if effect.end_seconds is None
            else effect.end_seconds
        )
        if end <= effect.start_seconds:
            continue
        if effect.start_seconds >= composition.duration_seconds:
            continue
        if overlay_from_effect(effect) is not None:
            return True
    return False


def _prepare_overlay_frames(
    composition: Composition,
    directory: Path,
) -> FFmpegOverlayFrames | None:
    if not _has_overlay_effects(composition):
        return None
    frames = directory / "overlay-frames"
    frames.mkdir(parents=True, exist_ok=True)
    size = composition.canvas.size
    for index in range(max(1, composition.frame_count)):
        frame = Image.new("RGBA", size, (0, 0, 0, 0))
        apply_overlay_effects(frame, composition, index / composition.fps)
        frame.save(frames / f"{index:04d}.png")
    return FFmpegOverlayFrames(
        pattern=str(frames / "%04d.png"),
        fps=composition.fps,
        duration_seconds=composition.duration_seconds,
    )


def _keyframe_mapping(
    layer: Layer,
    start_seconds: float,
    end_seconds: float,
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
    return mapping


def _prepare_text_layers(
    composition: Composition,
    directory: Path,
) -> Composition:
    if not any(layer.type is LayerType.TEXT for layer in composition.layers):
        return composition
    layers: list[Any] = []
    for index, layer in enumerate(composition.layers):
        if layer.type is not LayerType.TEXT:
            layers.append(layer)
            continue
        end = layer.end if layer.end is not None else composition.duration_seconds
        if not layer.visible or end <= layer.start_seconds:
            continue
        raster = _rasterize_text_layer(composition, layer, directory, index)
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
        keyframes = _keyframe_mapping(layer, layer.start_seconds, end)
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
            value = _prepare_text_layers(value, Path(text_directory.name))
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
            render_config = None
            if config is not None:
                render_config = getattr(config, "render", None)
                if render_config is None and isinstance(config, Mapping):
                    render_config = config.get("render")
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
            if subtitle_file is None and track is not None:
                subtitle_file = Path(text_directory.name) / f"{destination.stem}.ass"
                write_subtitle_document(track, subtitle_file, **subtitle_options)
            elif subtitle_file is None:
                referenced = value.subtitle_file.strip()
                if referenced:
                    subtitle_file = Path(referenced)
            if subtitle_file is not None and (
                Path(subtitle_file).suffix.lower() in {".srt", ".vtt"}
            ):
                converted = Path(text_directory.name) / f"{destination.stem}.ass"
                write_subtitle_document(
                    subtitle_file,
                    converted,
                    **subtitle_options,
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
