"""Render a composition to a deterministic video file with FFmpeg.

Command construction lives in :mod:`aivia.compose.ffmpeg_cmd` and stays pure.
This module is the thin execution layer around it: it runs the built argv,
writes through a temporary file so a failed render cannot leave a partial
output, and verifies the result with ffprobe before publishing it. Both
runners are injectable so the offline suite never needs the real binaries.
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

from aivia.core.config.media import (
    DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS,
    DEFAULT_MEDIA_FFPROBE_TIMEOUT_SECONDS,
)
from aivia.core.errors import ValidationError

from .errors import RenderError
from .ffmpeg_cmd import FFmpegCommand, build_ffmpeg_command
from .schema import Composition

if TYPE_CHECKING:
    from aivia.core.media_probe import FFprobeInfo, FFprobeRunner

__all__ = [
    "VIDEO_RENDERER_VERSION",
    "FFmpegRunner",
    "VideoRenderError",
    "VideoRenderResult",
    "VideoRenderer",
    "render_video",
]

VIDEO_RENDERER_VERSION = "1"
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

    def _verify(self, composition: Composition, probe: FFprobeInfo) -> None:
        stream = probe.primary_video
        if stream is None:
            raise VideoRenderError("the rendered file has no video stream")
        if stream.width != composition.canvas.width:
            raise VideoRenderError(
                "the rendered video width does not match the canvas",
                details={
                    "expected": composition.canvas.width,
                    "actual": stream.width,
                },
            )
        if stream.height != composition.canvas.height:
            raise VideoRenderError(
                "the rendered video height does not match the canvas",
                details={
                    "expected": composition.canvas.height,
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
        try:
            options = dict(command_options)
            options.pop("executable", None)
            command = build_ffmpeg_command(
                value,
                temporary,
                executable=self.ffmpeg_executable,
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
            self._verify(value, probe)
            os.replace(temporary, destination)
        finally:
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
) -> VideoRenderResult:
    """Render *composition* with the default FFmpeg video renderer."""
    renderer = VideoRenderer(
        ffmpeg_executable=ffmpeg_executable,
        ffprobe_executable=ffprobe_executable,
        timeout_seconds=timeout_seconds,
        ffprobe_timeout_seconds=ffprobe_timeout_seconds,
        ffmpeg_runner=ffmpeg_runner,
        ffprobe_runner=ffprobe_runner,
    )
    return renderer.render(composition, output, **command_options)
