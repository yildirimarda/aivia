"""Lip sync provider adapter with a no-AI sprite sheet fallback.

The adapter animates a character from an audio track. When a sprite sheet is
supplied it uses the no-AI fallback: mouth frames are swapped in time with the
audio, blinking is inserted at regular intervals and a gentle idle drift moves
the character across the sheet. Otherwise the request is delegated to the
configured :class:`~aivia.providers.protocols.lipsync.LipSyncProvider`.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from aivia.core.errors import ValidationError
from aivia.core.media_probe import image_dimensions
from aivia.providers.protocols.lipsync import (
    CAPABILITY_LIPSYNC,
    LipSyncProvider,
    LipSyncRequest,
    LipSyncResponse,
    LipSyncResult,
    LipSyncSpriteSheet,
)

__all__ = ["LipSyncAdapter", "LipSyncError"]

FFmpegRunner = Callable[[Sequence[str]], bytes | subprocess.CompletedProcess[Any]]
"""A callable that runs an ffmpeg argv and returns the encoded bytes."""


class LipSyncError(ValidationError):
    """A lip sync request could not be animated."""

    code = "lipsync"


def _number(value: float) -> str:
    """Format a float for an ffmpeg expression without trailing noise."""
    if value == 0.0:
        return "0"
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text or "0"


def _index_expr(index_expr: str, values: Sequence[int]) -> str:
    """Map an integer index expression onto one of *values* with a nested if."""
    expr = str(values[-1])
    for position in range(len(values) - 2, -1, -1):
        expr = f"if(eq({index_expr},{position}),{values[position]},{expr})"
    return expr


def _clamp(expr: str, low: float, high: float) -> str:
    """Clamp *expr* to ``[low, high]`` in an ffmpeg expression."""
    return (
        f"if(lt({expr},{_number(low)}),{_number(low)},"
        f"if(gt({expr},{_number(high)}),{_number(high)},{expr}))"
    )


def _sprite_sheet_filter(
    sheet: LipSyncSpriteSheet,
    duration: float,
    fps: float,
    *,
    width: int,
    height: int,
) -> str:
    """Build the ffmpeg video filter that animates one sprite sheet.

    The sprite sheet is a grid of ``frame_count`` frames, each
    ``frame_width`` by ``frame_height``. Over the clip duration the visible
    frame cycles through the mouth frames (audio-reactive pose swaps), the
    blink frame is shown briefly at regular intervals (blinking) and a slow
    vertical drift moves the character across the sheet (idle motion).
    """
    frame_width = sheet.frame_width
    frame_height = sheet.frame_height
    cols = width // frame_width
    if cols < 1:
        raise LipSyncError(
            "sprite sheet is narrower than one frame",
            details={"frame_width": frame_width, "sheet_width": width},
        )

    mouths = sorted(sheet.mouth_frames.values())
    if not mouths:
        mouths = list(sheet.idle_frames) or [0]
    mouth_count = len(mouths)
    mouth_segment = duration / mouth_count
    phase_expr = f"floor(mod(t/{_number(mouth_segment)},{mouth_count}))"
    mouth_expr = _index_expr(phase_expr, mouths)

    key_frame_expr = mouth_expr
    if sheet.blink_frame is not None and sheet.blink_frame >= 0:
        blink_every = max(duration / max(1, round(duration / 2.0)), 0.1)
        blink_dur = 0.1
        key_frame_expr = (
            f"if(lt(mod(t/{_number(blink_every)}),{_number(blink_dur)}),"
            f"{sheet.blink_frame},{mouth_expr})"
        )

    # Gentle idle drift: the character floats vertically across the window.
    drift_y = 4.0
    x_raw = f"({key_frame_expr}%{cols})*{frame_width}"
    y_raw = f"floor({key_frame_expr}/{cols})*{frame_height}"
    max_x = width - frame_width
    max_y = height - frame_height
    x_expr = _clamp(x_raw, 0.0, float(max_x))
    y_expr = _clamp(f"{y_raw} + {drift_y}*t/{_number(duration)}", 0.0, float(max_y))
    return (
        f"fps={_number(fps)},"
        f"crop={frame_width}:{frame_height}:{x_expr}:{y_expr},"
        f"scale=trunc(iw/2)*2:trunc(ih/2)*2"
    )


class LipSyncAdapter:
    """Animate a character from audio, with a no-AI sprite sheet fallback.

    ``provider`` is the AI lip sync provider used when no sprite sheet is
    supplied. When a sprite sheet is present the adapter animates it locally
    with ffmpeg: mouth frames are swapped in time with the audio, blinking is
    inserted and a gentle idle drift is applied. No model call is made.
    """

    def __init__(
        self,
        provider: LipSyncProvider | None = None,
        *,
        ffmpeg_executable: str = "ffmpeg",
        ffmpeg_runner: FFmpegRunner | None = None,
    ) -> None:
        self._provider = provider
        self._ffmpeg_executable = ffmpeg_executable
        self._ffmpeg_runner = ffmpeg_runner

    @property
    def provider(self) -> LipSyncProvider | None:
        """The AI lip sync provider, or ``None`` when only the fallback exists."""
        return self._provider

    async def animate(self, request: LipSyncRequest) -> LipSyncResponse:
        """Animate *request* and return the resulting clip."""
        if request.sprite_sheet is not None:
            return await self._animate_sprite_sheet(request)
        if self._provider is not None:
            return await self._provider.animate(request)
        raise LipSyncError(
            "lip sync needs a sprite sheet or a provider",
            details={"capability": CAPABILITY_LIPSYNC},
        )

    async def _animate_sprite_sheet(self, request: LipSyncRequest) -> LipSyncResponse:
        sheet = request.sprite_sheet
        if sheet is None:
            raise LipSyncError(
                "the no-AI fallback needs a sprite sheet",
                details={"capability": CAPABILITY_LIPSYNC},
            )
        if sheet.image.data is None:
            raise LipSyncError(
                "the no-AI fallback needs inline sprite sheet bytes",
                details={"capability": CAPABILITY_LIPSYNC},
            )
        duration = request.duration_seconds
        if duration is None:
            raise LipSyncError(
                "the no-AI fallback needs a duration_seconds",
                details={"capability": CAPABILITY_LIPSYNC},
            )
        fps = float(request.fps)
        dimensions = image_dimensions(sheet.image.data)
        if dimensions is None:
            raise LipSyncError(
                "could not read the sprite sheet dimensions",
                details={"capability": CAPABILITY_LIPSYNC},
            )
        width, height = dimensions

        with tempfile.TemporaryDirectory(prefix="aivia-lipsync-fallback-") as directory:
            directory_path = Path(directory)
            sheet_path = directory_path / "sheet.png"
            sheet_path.write_bytes(sheet.image.data)
            audio_path = directory_path / "audio.wav"
            audio_path.write_bytes(request.audio.data or b"")
            output_path = directory_path / "output.mp4"
            args = self._ffmpeg_argv(
                sheet_path,
                audio_path,
                output_path,
                request,
                sheet,
                duration,
                fps,
                width,
                height,
            )
            data = await asyncio.to_thread(self._run_ffmpeg, args)

        return LipSyncResponse(
            video=LipSyncResult(
                data=data,
                format=request.format,
                width=sheet.frame_width,
                height=sheet.frame_height,
                duration_seconds=duration,
                fps=request.fps,
                used_fallback=True,
            ),
            model="aivia-lipsync-fallback",
            provider="aivia",
        )

    def _ffmpeg_argv(
        self,
        sheet_path: Path,
        audio_path: Path,
        output_path: Path,
        request: LipSyncRequest,
        sheet: LipSyncSpriteSheet,
        duration: float,
        fps: float,
        width: int,
        height: int,
    ) -> list[str]:
        filter_graph = _sprite_sheet_filter(
            sheet, duration, fps, width=width, height=height
        )
        return [
            self._ffmpeg_executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-loop",
            "1",
            "-i",
            str(sheet_path),
            "-i",
            str(audio_path),
            "-filter_complex",
            filter_graph,
            "-map",
            "[v]",
            "-map",
            "1:a:0?",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-t",
            f"{duration:.6f}",
            str(output_path),
        ]

    def _run_ffmpeg(self, args: Sequence[str]) -> bytes:
        output_path = Path(args[-1])
        if self._ffmpeg_runner is not None:
            result = self._ffmpeg_runner(list(args))
            if isinstance(result, (bytes, bytearray, memoryview)):
                return bytes(result)
            if isinstance(output_path, Path) and output_path.is_file():
                return output_path.read_bytes()
            return b""
        if shutil.which(self._ffmpeg_executable) is None:
            raise LipSyncError(
                "ffmpeg is required for the no-AI lip sync fallback",
                details={
                    "capability": CAPABILITY_LIPSYNC,
                    "executable": self._ffmpeg_executable,
                },
            )
        subprocess.run(list(args), check=True, capture_output=True)
        return output_path.read_bytes()
