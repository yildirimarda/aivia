"""Shape layers rasterized to transparent overlays in video renders."""

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
    VideoRenderer,
    VideoRenderError,
    render_still,
    render_video,
)

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

requires_ffmpeg = pytest.mark.skipif(
    FFMPEG is None or FFPROBE is None,
    reason="ffmpeg and ffprobe are required for a real video render",
)

BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)


def _run(argv: list[str]) -> None:
    subprocess.run(argv, check=True, capture_output=True, text=True)


def _composition(
    layers: list[dict[str, Any]],
    *,
    width: int = 160,
    height: int = 120,
    duration_seconds: float = 2.0,
    fps: int = 12,
) -> Composition:
    return Composition.model_validate(
        {
            "canvas": {"width": width, "height": height, "background": "#000000"},
            "duration_seconds": duration_seconds,
            "fps": fps,
            "layers": layers,
        }
    )


def _frame(source: Path, time_seconds: float, destination: Path) -> Image.Image:
    assert FFMPEG is not None
    _run(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{time_seconds:.6f}",
            "-i",
            str(source),
            "-frames:v",
            "1",
            str(destination),
        ]
    )
    with Image.open(destination) as image:
        return image.convert("RGB")


def _assert_color(
    image: Image.Image,
    point: tuple[int, int],
    expected: tuple[int, int, int],
    *,
    tolerance: int = 20,
) -> None:
    actual = image.getpixel(point)
    assert isinstance(actual, tuple)
    for channel, target in zip(actual, expected, strict=True):
        assert abs(channel - target) <= tolerance, (point, actual, expected)


def test_video_renderer_rasterizes_shape_layers_before_building_the_command(
    tmp_path: Path,
) -> None:
    captured: dict[str, list[str]] = {}

    def fail(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        captured["argv"] = list(argv)
        return subprocess.CompletedProcess(list(argv), 1, stdout="", stderr="boom")

    composition = _composition(
        [
            {
                "id": "box",
                "type": "shape",
                "shape": "rectangle",
                "x": 10,
                "y": 20,
                "width": 40,
                "height": 30,
                "fill": "#ff0000",
            }
        ],
        duration_seconds=1.0,
        fps=8,
    )
    renderer = VideoRenderer(ffmpeg_runner=fail)

    with pytest.raises(VideoRenderError, match="could not render"):
        renderer.render(composition, tmp_path / "failed.mp4")

    argv = captured["argv"]
    assert any(argument.endswith("shape-0.png") for argument in argv)
    graph = argv[argv.index("-filter_complex") + 1]
    assert "overlay=" in graph
    assert "drawtext=" not in graph


def test_rotation_motion_expands_the_rasterized_shape_source(
    tmp_path: Path,
) -> None:
    captured: dict[str, Image.Image] = {}

    def fail(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        source = next(Path(arg) for arg in argv if arg.endswith("shape-0.png"))
        with Image.open(source) as image:
            captured["raster"] = image.convert("RGBA").copy()
        return subprocess.CompletedProcess(list(argv), 1, stdout="", stderr="boom")

    composition = _composition(
        [
            {
                "id": "box",
                "type": "shape",
                "shape": "rectangle",
                "x": 50,
                "y": 35,
                "width": 60,
                "height": 50,
                "fill": "#ff0000",
                "keyframes": {
                    "rotation": [
                        {"time": 0, "value": 0, "easing": "linear"},
                        {"time": 2, "value": 90},
                    ]
                },
            }
        ]
    )
    with pytest.raises(VideoRenderError, match="could not render"):
        VideoRenderer(ffmpeg_runner=fail).render(composition, tmp_path / "failed.mp4")

    raster = captured["raster"]
    assert raster.width > 60
    assert raster.height > 50
    bounds = raster.getchannel("A").getbbox()
    assert bounds is not None
    left, top, right, bottom = bounds
    assert abs((left + right) / 2.0 - raster.width / 2.0) < 2.0
    assert abs((top + bottom) / 2.0 - raster.height / 2.0) < 2.0


@pytest.mark.slow
@requires_ffmpeg
def test_a_shape_is_visible_only_inside_its_window(tmp_path: Path) -> None:
    output = tmp_path / "shape-window.mp4"
    render_video(
        _composition(
            [
                {
                    "id": "box",
                    "type": "shape",
                    "shape": "rectangle",
                    "x": 20,
                    "y": 20,
                    "width": 60,
                    "height": 40,
                    "fill": "#ff0000",
                    "start": 0.5,
                    "end": 1.5,
                }
            ]
        ),
        output,
        preset="ultrafast",
        crf=20,
    )

    before = _frame(output, 0.2, tmp_path / "before.png")
    during = _frame(output, 1.0, tmp_path / "during.png")
    after = _frame(output, 1.8, tmp_path / "after.png")

    _assert_color(before, (50, 40), BLACK)
    _assert_color(during, (50, 40), RED)
    _assert_color(during, (10, 10), BLACK)
    _assert_color(after, (50, 40), BLACK)


@pytest.mark.slow
@requires_ffmpeg
def test_position_keyframes_move_a_shape_across_sampled_frames(
    tmp_path: Path,
) -> None:
    output = tmp_path / "shape-motion.mp4"
    render_video(
        _composition(
            [
                {
                    "id": "box",
                    "type": "shape",
                    "shape": "rectangle",
                    "x": 10,
                    "y": 40,
                    "width": 30,
                    "height": 30,
                    "fill": "#ff0000",
                    "keyframes": {
                        "position": [
                            {"time": 0, "value": [0, 0], "easing": "linear"},
                            {"time": 2, "value": [80, 0]},
                        ]
                    },
                }
            ]
        ),
        output,
        preset="ultrafast",
        crf=20,
    )

    start = _frame(output, 0.0, tmp_path / "start.png")
    end = _frame(output, 1.9, tmp_path / "end.png")

    _assert_color(start, (20, 55), RED)
    _assert_color(start, (100, 55), BLACK)
    _assert_color(end, (20, 55), BLACK)
    _assert_color(end, (100, 55), RED)


@pytest.mark.slow
@requires_ffmpeg
def test_motion_preset_pan_moves_a_shape_like_the_preview(tmp_path: Path) -> None:
    composition = _composition(
        [
            {
                "id": "box",
                "type": "shape",
                "shape": "rectangle",
                "x": 80,
                "y": 40,
                "width": 30,
                "height": 30,
                "fill": "#ff0000",
                "motion": {
                    "type": "ken_burns",
                    "start_scale": 1.0,
                    "end_scale": 1.0,
                    "start_pan": [0, 0],
                    "end_pan": [60, 0],
                    "easing": "linear",
                },
            }
        ]
    )
    output = tmp_path / "shape-pan.mp4"
    render_video(composition, output, preset="ultrafast", crf=20)

    start = _frame(output, 0.0, tmp_path / "start.png")
    end = _frame(output, 1.9, tmp_path / "end.png")
    preview = render_still(composition, 1.9).convert("RGB")

    _assert_color(start, (95, 55), RED)
    _assert_color(start, (35, 55), BLACK)
    _assert_color(end, (95, 55), BLACK)
    _assert_color(end, (35, 55), RED)
    for point in ((95, 55), (35, 55)):
        expected = preview.getpixel(point)
        assert isinstance(expected, tuple)
        _assert_color(end, point, (expected[0], expected[1], expected[2]))


@pytest.mark.slow
@requires_ffmpeg
def test_shape_layers_keep_their_z_index_over_image_layers(tmp_path: Path) -> None:
    background = tmp_path / "background.png"
    Image.new("RGB", (160, 120), RED).save(background)
    output = tmp_path / "shape-z.mp4"
    render_video(
        _composition(
            [
                {
                    "id": "background",
                    "type": "image",
                    "source": str(background),
                    "fit": "stretch",
                    "z_index": 0,
                },
                {
                    "id": "box",
                    "type": "shape",
                    "shape": "rectangle",
                    "x": 40,
                    "y": 30,
                    "width": 50,
                    "height": 40,
                    "fill": "#0000ff",
                    "z_index": 1,
                },
            ]
        ),
        output,
        preset="ultrafast",
        crf=20,
    )

    frame = _frame(output, 1.0, tmp_path / "frame.png")

    _assert_color(frame, (60, 45), BLUE)
    _assert_color(frame, (10, 10), RED)


@pytest.mark.slow
@requires_ffmpeg
@pytest.mark.parametrize(
    ("start_scale", "end_scale"),
    [
        (1.0, 2.0),
        ([1.0, 1.0], [2.0, 1.5]),
    ],
    ids=["uniform", "pair"],
)
def test_scale_motion_expands_a_shape_beyond_its_base_box(
    tmp_path: Path,
    start_scale: Any,
    end_scale: Any,
) -> None:
    composition = _composition(
        [
            {
                "id": "box",
                "type": "shape",
                "shape": "rectangle",
                "x": 60,
                "y": 45,
                "width": 40,
                "height": 30,
                "fill": "#ff0000",
                "keyframes": {
                    "scale": [
                        {"time": 0, "value": start_scale, "easing": "linear"},
                        {"time": 2, "value": end_scale},
                    ]
                },
            }
        ]
    )
    output = tmp_path / "shape-scale.mp4"
    render_video(composition, output, preset="ultrafast", crf=20)

    start = _frame(output, 0.0, tmp_path / "start.png")
    end = _frame(output, 1.9, tmp_path / "end.png")
    preview = render_still(composition, 1.9).convert("RGB")

    _assert_color(start, (80, 60), RED)
    for point in ((45, 60), (80, 40)):
        _assert_color(end, point, RED)
        expected = preview.getpixel(point)
        assert isinstance(expected, tuple)
        _assert_color(end, point, (expected[0], expected[1], expected[2]))
    _assert_color(end, (20, 20), BLACK)


@pytest.mark.slow
@requires_ffmpeg
@pytest.mark.parametrize("end_angle", [90.0, -90.0])
def test_rotation_motion_expands_a_shape_beyond_its_base_box(
    tmp_path: Path,
    end_angle: float,
) -> None:
    composition = _composition(
        [
            {
                "id": "box",
                "type": "shape",
                "shape": "rectangle",
                "x": 50,
                "y": 35,
                "width": 60,
                "height": 50,
                "fill": "#ff0000",
                "keyframes": {
                    "rotation": [
                        {"time": 0, "value": 0, "easing": "linear"},
                        {"time": 2, "value": end_angle},
                    ]
                },
            }
        ]
    )
    output = tmp_path / "shape-rotation.mp4"
    render_video(composition, output, preset="ultrafast", crf=20)

    start = _frame(output, 0.0, tmp_path / "start.png")
    middle = _frame(output, 1.0, tmp_path / "middle.png")
    preview = render_still(composition, 1.0).convert("RGB")

    _assert_color(start, (80, 60), RED)
    _assert_color(start, (80, 30), BLACK)
    for point in ((80, 30), (80, 90)):
        _assert_color(middle, point, RED)
        expected = preview.getpixel(point)
        assert isinstance(expected, tuple)
        _assert_color(middle, point, (expected[0], expected[1], expected[2]))
    _assert_color(middle, (50, 35), BLACK)
