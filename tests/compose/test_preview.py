"""A preview renders fewer, smaller frames and is measurably faster."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import pytest
from PIL import Image

from aivia.compose import (
    Composition,
    FFmpegCommandError,
    ImageLayer,
    LogoLayer,
    PreviewError,
    PreviewRenderResult,
    PreviewSettings,
    ShapeLayer,
    TextLayer,
    VideoRenderResult,
    build_ffmpeg_command,
    preview_settings,
    render_preview,
    render_video,
    scale_composition,
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


def _layered_composition() -> Composition:
    return Composition.model_validate(
        {
            "canvas": {"width": 1280, "height": 720, "background": "#101010"},
            "duration_seconds": 4.0,
            "fps": 30,
            "layers": [
                {
                    "id": "title",
                    "type": "text",
                    "text": "Hello",
                    "x": 200,
                    "y": 100,
                    "width": 800,
                    "height": 200,
                    "font_size": 48,
                    "min_font_size": 16,
                    "stroke_width": 4,
                    "letter_spacing": 2,
                    "placement": {
                        "mode": "smart",
                        "padding": 10,
                        "margin": 20,
                        "face_padding": 5,
                        "step": 4,
                        "faces": [{"x": 100, "y": 200, "width": 80, "height": 40}],
                    },
                    "entrance": {
                        "type": "slide",
                        "distance": 40,
                        "offset": [10, 20],
                    },
                    "keyframes": {
                        "position": [
                            {"time": 0, "value": [100, 200]},
                            {"time": 2, "value": [300, 400]},
                        ]
                    },
                    "z": 0,
                },
                {
                    "id": "photo",
                    "type": "image",
                    "source": "photo.png",
                    "x": 10,
                    "y": 20,
                    "width": 640,
                    "height": 360,
                    "motion": {
                        "type": "parallax",
                        "start_pan": [40, 20],
                        "drift": [30, 40],
                    },
                    "mask": {"source": "mask.png", "feather": 6},
                    "filters": [{"type": "blur", "radius": 4}],
                    "z": 1,
                },
                {
                    "id": "bar",
                    "type": "shape",
                    "shape": "rounded_rectangle",
                    "x": 0,
                    "y": 0,
                    "width": 400,
                    "height": 100,
                    "stroke_width": 3,
                    "radius": 8,
                    "z": 2,
                },
                {
                    "id": "logo",
                    "type": "logo",
                    "source": "logo.png",
                    "width": 100,
                    "height": 50,
                    "margin": 20,
                    "shadow": True,
                    "shadow_offset": [4, 8],
                    "shadow_blur": 6,
                    "z": 3,
                },
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


def test_scale_composition_scales_canvas_and_absolute_geometry() -> None:
    settings = preview_settings(_layered_composition(), scale=0.5, max_fps=15)

    scaled = scale_composition(_layered_composition(), settings)

    assert (scaled.width, scaled.height, scaled.fps) == (640, 360, 15)
    title = scaled.layer("title")
    assert isinstance(title, TextLayer)
    assert (title.x, title.y, title.width, title.height) == (100, 50, 400, 100)
    assert title.font_size == 24
    assert title.min_font_size == 8
    assert title.stroke_width == 2
    assert title.letter_spacing == 1
    assert title.placement is not None
    assert title.placement.padding == 5
    assert title.placement.face_padding == 2.5
    assert title.placement.margin == 10
    assert title.placement.step == 2
    assert title.placement.faces[0].box == (50, 100, 40, 20)
    assert title.entrance is not None
    assert title.entrance.distance == 20
    assert title.entrance.offset == (5, 10)
    assert title.keyframes is not None
    position = cast(dict[str, Any], title.keyframes)["position"]
    assert [frame["value"] for frame in position] == [
        (50.0, 100.0),
        (150.0, 200.0),
    ]

    photo = scaled.layer("photo")
    assert isinstance(photo, ImageLayer)
    assert (photo.x, photo.y, photo.width, photo.height) == (5, 10, 320, 180)
    assert photo.motion is not None
    assert photo.motion.start_pan == (20, 10)
    assert photo.motion.drift == (15, 20)
    assert photo.mask is not None
    assert photo.mask.feather == 3
    assert photo.filters[0].radius == 2

    bar = scaled.layer("bar")
    assert isinstance(bar, ShapeLayer)
    assert (bar.width, bar.height) == (200, 50)
    assert bar.stroke_width == 1.5
    assert bar.radius == 4

    logo = scaled.layer("logo")
    assert isinstance(logo, LogoLayer)
    assert (logo.width, logo.height) == (50, 25)
    assert logo.margin == 10
    assert logo.shadow_offset == (2, 4)
    assert logo.shadow_blur == 3


def test_scale_composition_scales_subtitle_style_to_the_canvas() -> None:
    composition = Composition.model_validate(
        {
            "canvas": {"width": 1280, "height": 720},
            "duration_seconds": 4.0,
            "fps": 30,
            "layers": [],
            "subtitles": {
                "cues": [{"start": 0, "end": 1, "text": "Hello"}],
                "style": {"font_size": 42, "outline": 3, "margin_l": 10},
            },
        }
    )
    default_style = Composition.model_validate(
        {
            "canvas": {"width": 1280, "height": 720},
            "duration_seconds": 4.0,
            "fps": 30,
            "layers": [],
            "subtitles": {"cues": [{"start": 0, "end": 1, "text": "Hello"}]},
        }
    )

    scaled = scale_composition(composition, 640, 360, fps=15)
    default_scaled = scale_composition(default_style, 640, 360, fps=15)

    assert scaled.subtitles is not None
    assert scaled.subtitles.style["font_size"] == 21
    assert scaled.subtitles.style["outline"] == 2
    assert scaled.subtitles.style["margin_l"] == 5
    assert default_scaled.subtitles is not None
    assert default_scaled.subtitles.style["font_size"] == 16


def test_render_preview_scales_layer_geometry_before_compositing(
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []
    output = tmp_path / "preview-geometry.mp4"

    result = render_preview(
        _layered_composition(),
        output,
        scale=0.5,
        max_fps=15,
        ffmpeg_runner=_ffmpeg_writer(calls),
        ffprobe_runner=_probe_runner(_probe_json(640, 360, 15, 4.0)),
    )

    assert (result.width, result.height, result.fps) == (640, 360, 15)
    argv = calls[0]
    graph = argv[argv.index("-filter_complex") + 1]
    assert "s=640x360" in graph
    assert "s=1280x720" not in graph
    assert "scale=320:180" in graph
    assert "scale=50:25" in graph
    assert "x='(115)+(if(lt(t,0),50" in graph
    assert "scale=640:360:flags=lanczos[vout]" not in graph
    assert graph.endswith("[vout]")


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
    assert "s=320x180" in graph
    assert "scale=320:180:flags=lanczos[vout]" not in graph
    assert graph.endswith("[vout]")
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
