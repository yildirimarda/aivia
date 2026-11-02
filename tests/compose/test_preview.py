"""A preview renders fewer, smaller frames and is measurably faster."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from aivia.compose import (
    Composition,
    FFmpegCommandError,
    PreviewError,
    PreviewRenderResult,
    PreviewSettings,
    VideoRenderResult,
    build_ffmpeg_command,
    preview_settings,
    render_preview,
    render_video,
)

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

pytestmark = pytest.mark.slow

requires_ffmpeg = pytest.mark.skipif(
    FFMPEG is None or FFPROBE is None,
    reason="ffmpeg and ffprobe are required for a real preview render",
)


def _composition(
    *,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    duration: float = 4.0,
) -> Composition:
    return Composition.model_validate(
        {
            "canvas": {"width": width, "height": height, "background": "#101010"},
            "duration_seconds": duration,
            "fps": fps,
            "layers": [],
        }
    )


def _probe_json(width: int, height: int, fps: int, duration: float) -> str:
    return json.dumps(
        {
            "streams": [
                {
                    "index": 0,
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": width,
                    "height": height,
                    "duration": f"{duration:.3f}",
                    "avg_frame_rate": f"{fps}/1",
                }
            ],
            "format": {
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                "duration": f"{duration:.3f}",
                "size": "12345",
            },
        }
    )


def _ffmpeg_writer(
    calls: list[list[str]],
) -> Callable[[Sequence[str]], subprocess.CompletedProcess[str]]:
    def run(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        Path(argv[-1]).write_bytes(b"preview")
        return subprocess.CompletedProcess(list(argv), 0, stdout="", stderr="")

    return run


def _probe_runner(
    payload: str,
) -> Callable[[Sequence[str]], subprocess.CompletedProcess[str]]:
    def run(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(list(argv), 0, stdout=payload, stderr="")

    return run


def _composition_with_asset(tmp_path: Path) -> Composition:
    asset = tmp_path / "gradient.png"
    Image.new("RGB", (160, 90), (40, 120, 200)).save(asset)
    return Composition.model_validate(
        {
            "canvas": {"width": 1280, "height": 720, "background": "#101010"},
            "duration_seconds": 4.0,
            "fps": 30,
            "layers": [
                {
                    "id": "background",
                    "type": "image",
                    "source": str(asset),
                    "fit": "stretch",
                    "z": 0,
                }
            ],
        }
    )


def test_preview_settings_reduce_resolution_and_frame_rate() -> None:
    settings = preview_settings(_composition())

    assert isinstance(settings, PreviewSettings)
    assert (settings.width, settings.height) == (640, 360)
    assert settings.fps == 15
    assert (settings.source_width, settings.source_height, settings.source_fps) == (
        1280,
        720,
        30,
    )
    assert settings.resolution == "640x360"
    assert settings.is_reduced
    assert settings.work_ratio < 0.25


def test_preview_settings_honour_caps_without_enlarging() -> None:
    settings = preview_settings(
        _composition(width=1920, height=1080, fps=60),
        scale=0.5,
        max_width=640,
        max_height=400,
        max_fps=12,
    )

    assert (settings.width, settings.height) == (640, 400)
    assert settings.fps == 12

    unchanged = preview_settings(_composition(width=320, height=240, fps=10), scale=1.0)

    assert (unchanged.width, unchanged.height) == (320, 240)
    assert unchanged.fps == 10
    assert not unchanged.is_reduced


@pytest.mark.parametrize(
    "options",
    [
        {"scale": 0.0},
        {"scale": 1.5},
        {"max_width": 1},
        {"max_height": 1},
        {"max_fps": 0},
    ],
)
def test_invalid_preview_settings_are_rejected(options: dict[str, Any]) -> None:
    with pytest.raises(PreviewError):
        preview_settings(_composition(), **options)


def test_the_preview_command_scales_the_finished_frame() -> None:
    command = build_ffmpeg_command(
        _composition(),
        "preview.mp4",
        output_width=320,
        output_height=180,
    )

    assert "scale=320:180:flags=lanczos[vout]" in command.filtergraph
    assert command.filtergraph.endswith("[vout]")

    with pytest.raises(FFmpegCommandError):
        build_ffmpeg_command(_composition(), "preview.mp4", output_width=320)


def test_render_preview_scales_the_output_and_verifies_the_probe(
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []
    output = tmp_path / "preview.mp4"

    result = render_preview(
        _composition(),
        output,
        scale=0.25,
        max_fps=6,
        ffmpeg_runner=_ffmpeg_writer(calls),
        ffprobe_runner=_probe_runner(_probe_json(320, 180, 6, 4.0)),
    )

    assert isinstance(result, PreviewRenderResult)
    assert result.output == output
    assert output.read_bytes() == b"preview"
    assert (result.width, result.height, result.fps) == (320, 180, 6)
    assert result.probe.duration_seconds == pytest.approx(4.0)

    argv = calls[0]
    graph = argv[argv.index("-filter_complex") + 1]
    assert "scale=320:180:flags=lanczos[vout]" in graph
    assert argv[argv.index("-r") + 1] == "6"


@requires_ffmpeg
def test_a_preview_render_keeps_layers_and_duration(tmp_path: Path) -> None:
    composition = _composition_with_asset(tmp_path)
    output = tmp_path / "preview-layers.mp4"

    result = render_preview(
        composition,
        output,
        scale=0.5,
        max_fps=10,
        preset="ultrafast",
        crf=28,
    )

    assert isinstance(result, PreviewRenderResult)
    assert output.is_file()
    assert (result.width, result.height, result.fps) == (640, 360, 10)
    stream = result.probe.primary_video
    assert stream is not None
    assert (stream.width, stream.height) == (640, 360)
    assert stream.frame_rate == "10/1"
    assert result.probe.duration_seconds == pytest.approx(4.0, abs=0.2)


@requires_ffmpeg
def test_a_preview_render_is_smaller_and_measurably_faster(
    tmp_path: Path,
) -> None:
    composition = _composition()
    warmup = _composition(width=64, height=64, fps=5, duration=0.2)
    render_video(warmup, tmp_path / "warmup.mp4", preset="ultrafast", crf=28)

    started = time.perf_counter()
    full = render_video(
        composition,
        tmp_path / "full.mp4",
        preset="ultrafast",
        crf=28,
    )
    full_elapsed = time.perf_counter() - started
    assert isinstance(full, VideoRenderResult)

    started = time.perf_counter()
    preview = render_preview(
        composition,
        tmp_path / "preview.mp4",
        scale=0.25,
        max_fps=6,
        preset="ultrafast",
        crf=28,
    )
    preview_elapsed = time.perf_counter() - started

    assert isinstance(preview, PreviewRenderResult)
    assert preview.output.is_file()
    assert preview.output.stat().st_size > 0
    assert preview.width == 320
    assert preview.height == 180
    assert preview.fps == 6
    assert preview.settings.work_ratio < 0.1

    stream = preview.probe.primary_video
    assert stream is not None
    assert (stream.width, stream.height) == (320, 180)
    assert stream.frame_rate == "6/1"
    assert preview.probe.duration_seconds == pytest.approx(4.0, abs=0.2)

    full_stream = full.probe.primary_video
    assert full_stream is not None
    assert (full_stream.width, full_stream.height) == (1280, 720)
    assert full_stream.frame_rate == "30/1"
    assert preview_elapsed < full_elapsed
