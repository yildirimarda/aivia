"""Procedural overlay effects animate in rendered videos."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from aivia.compose import (
    Composition,
    FFmpegOverlayFrames,
    OverlayType,
    VideoRenderer,
    VideoRenderError,
    build_ffmpeg_command,
    render_video,
)

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
requires_ffmpeg = pytest.mark.skipif(
    FFMPEG is None or FFPROBE is None,
    reason="ffmpeg and ffprobe are required for a real video render",
)


def _composition(
    effects: list[dict[str, Any]],
    *,
    duration_seconds: float = 1.5,
    fps: int = 10,
) -> Composition:
    return Composition.model_validate(
        {
            "canvas": {"width": 160, "height": 120, "background": "#000000"},
            "duration_seconds": duration_seconds,
            "fps": fps,
            "effects": effects,
        }
    )


def _frame(source: Path, time_seconds: float, destination: Path) -> Image.Image:
    assert FFMPEG is not None
    subprocess.run(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-ss",
            f"{time_seconds:.6f}",
            "-frames:v",
            "1",
            str(destination),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    with Image.open(destination) as image:
        return image.convert("RGB")


def _brightest(image: Image.Image) -> int:
    _, high = image.convert("L").getextrema()
    assert isinstance(high, int)
    return high


def test_builder_adds_an_overlay_sequence_after_the_layers() -> None:
    composition = _composition([])
    frames = FFmpegOverlayFrames(
        pattern="/tmp/frames/%04d.png",
        fps=10,
        duration_seconds=1.5,
    )

    command = build_ffmpeg_command(
        composition,
        "out.mp4",
        overlay_frames=frames,
    )

    assert command.inputs[-1].source == frames.pattern
    assert command.inputs[-1].options == ("-framerate", "10", "-start_number", "0")
    assert "[1:v]format=rgba,setsar=1[overlay0]" in command.filtergraph
    assert (
        "[base0][overlay0]overlay=x=0:y=0:"
        "eof_action=pass:repeatlast=0:shortest=0[base1]"
    ) in command.filtergraph


def test_video_renderer_rasterizes_overlay_frames_before_running_ffmpeg(
    tmp_path: Path,
) -> None:
    captured: dict[str, list[str]] = {}

    def fail(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        captured["argv"] = list(argv)
        return subprocess.CompletedProcess(list(argv), 1, stdout="", stderr="boom")

    composition = _composition(
        [
            {
                "type": "snow",
                "start": 0.0,
                "end": 1.0,
                "params": {"seed": 4, "count": 8},
            }
        ],
        duration_seconds=1.0,
        fps=8,
    )
    renderer = VideoRenderer(ffmpeg_runner=fail)

    with pytest.raises(VideoRenderError, match="could not render"):
        renderer.render(composition, tmp_path / "failed.mp4")

    argv = captured["argv"]
    graph = argv[argv.index("-filter_complex") + 1]
    assert "overlay=x=0:y=0" in graph
    assert any(argument.endswith("%04d.png") for argument in argv)


@pytest.mark.slow
@requires_ffmpeg
@pytest.mark.parametrize("overlay_type", list(OverlayType))
def test_every_procedural_overlay_animates_inside_its_window(
    overlay_type: OverlayType,
    tmp_path: Path,
) -> None:
    composition = _composition(
        [
            {
                "type": overlay_type.value,
                "start": 0.1,
                "end": 1.4,
                "params": {
                    "seed": 17,
                    "count": 250,
                    "size": [0.02, 0.05],
                    "colors": ["#ffffff"],
                    "speed": 2.0,
                },
            }
        ]
    )
    output = tmp_path / f"{overlay_type.value}.mp4"

    render_video(composition, output, preset="ultrafast", crf=20)

    before = _frame(output, 0.0, tmp_path / "before.png")
    early = _frame(output, 0.2, tmp_path / "early.png")
    late = _frame(output, 1.2, tmp_path / "late.png")
    after = _frame(output, 1.4, tmp_path / "after.png")

    assert _brightest(before) < 20
    assert _brightest(early) > 20
    assert _brightest(late) > 20
    assert early.tobytes() != late.tobytes()
    assert _brightest(after) < 20
