"""Render a multi-layer composition to a real video and verify it with ffprobe."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest
from PIL import Image

from aivia.compose import (
    Composition,
    VideoRenderer,
    VideoRenderError,
    VideoRenderResult,
    render_video,
)
from aivia.core.media_probe import run_ffprobe

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        FFMPEG is None or FFPROBE is None,
        reason="ffmpeg and ffprobe are required for a real video render",
    ),
]


def _run(argv: list[str]) -> None:
    subprocess.run(
        argv,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture()
def assets(tmp_path: Path) -> dict[str, Path]:
    assert FFMPEG is not None
    red = tmp_path / "red.png"
    Image.new("RGB", (40, 20), (255, 0, 0)).save(red)
    green = tmp_path / "green.png"
    Image.new("RGBA", (16, 16), (0, 255, 0, 255)).save(green)
    blue = tmp_path / "blue.mp4"
    _run(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=0x0000ff:s=64x48:r=24:d=2.000000",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(blue),
        ]
    )
    return {"red": red, "green": green, "blue": blue}


def _composition(assets: dict[str, Path]) -> Composition:
    return Composition.model_validate(
        {
            "canvas": {"width": 320, "height": 240, "background": "#000000"},
            "duration_seconds": 2.0,
            "fps": 24,
            "layers": [
                {
                    "id": "back",
                    "type": "image",
                    "source": str(assets["red"]),
                    "fit": "stretch",
                    "z": 0,
                },
                {
                    "id": "clip",
                    "type": "video",
                    "source": str(assets["blue"]),
                    "x": 40,
                    "y": 30,
                    "width": 160,
                    "height": 120,
                    "fit": "stretch",
                    "trim_start": 0.5,
                    "trim_end": 1.5,
                    "start": 0.5,
                    "end": 1.5,
                    "z": 1,
                },
                {
                    "id": "badge",
                    "type": "sticker",
                    "source": str(assets["green"]),
                    "anchor": "top_right",
                    "margin": 8,
                    "width": 32,
                    "height": 32,
                    "scale": 2.0,
                    "start": 1.0,
                    "end": 2.0,
                    "z": 2,
                },
            ],
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
            "-i",
            str(source),
            "-ss",
            f"{time_seconds:.6f}",
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
    tolerance: int = 16,
) -> None:
    actual = image.getpixel(point)
    assert isinstance(actual, tuple)
    for channel, target in zip(actual, expected, strict=True):
        assert abs(channel - target) <= tolerance, (point, actual, expected)


def test_a_multi_layer_composition_renders_to_a_verified_mp4(
    assets: dict[str, Path],
    tmp_path: Path,
) -> None:
    output = tmp_path / "render" / "multi-layer.mp4"

    result = render_video(
        _composition(assets),
        output,
        preset="ultrafast",
        crf=28,
    )

    assert isinstance(result, VideoRenderResult)
    assert result.output == output
    assert output.is_file()
    assert output.stat().st_size > 0

    assert FFPROBE is not None
    probe = run_ffprobe(output, executable=FFPROBE)
    stream = probe.primary_video
    assert stream is not None
    assert (stream.width, stream.height) == (320, 240)
    assert stream.frame_rate == "24/1"
    assert probe.duration_seconds == pytest.approx(2.0, abs=0.1)
    assert probe.audio_streams == ()
    assert result.probe.duration_seconds == probe.duration_seconds


def test_timing_scaling_and_overlay_are_visible_in_sampled_frames(
    assets: dict[str, Path],
    tmp_path: Path,
) -> None:
    output = tmp_path / "multi-layer.mp4"
    render_video(_composition(assets), output, preset="ultrafast", crf=28)

    red = (252, 0, 0)
    green = (0, 252, 0)
    blue = (0, 0, 252)

    early = _frame(output, 0.25, tmp_path / "early.png")
    _assert_color(early, (100, 80), red)
    _assert_color(early, (280, 40), red)

    video_only = _frame(output, 0.75, tmp_path / "video-only.png")
    _assert_color(video_only, (100, 80), blue)
    _assert_color(video_only, (280, 40), red)
    _assert_color(video_only, (48, 38), blue)
    _assert_color(video_only, (32, 22), red)
    _assert_color(video_only, (192, 142), blue)
    _assert_color(video_only, (208, 158), red)

    middle = _frame(output, 1.0, tmp_path / "middle.png")
    _assert_color(middle, (100, 80), blue)
    _assert_color(middle, (254, 14), green)
    _assert_color(middle, (242, 14), red)
    _assert_color(middle, (254, 66), green)
    _assert_color(middle, (254, 78), red)

    late = _frame(output, 1.75, tmp_path / "late.png")
    _assert_color(late, (100, 80), red)
    _assert_color(late, (254, 14), green)
    _assert_color(late, (242, 14), red)


def test_a_failed_render_leaves_no_partial_output(tmp_path: Path) -> None:
    output = tmp_path / "failed.mp4"

    def fail(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(list(argv), 1, stdout="", stderr="boom")

    renderer = VideoRenderer(ffmpeg_runner=fail)

    with pytest.raises(VideoRenderError, match="could not render"):
        renderer.render(
            Composition.model_validate(
                {
                    "canvas": {"width": 32, "height": 24},
                    "duration_seconds": 1.0,
                    "fps": 24,
                }
            ),
            output,
        )

    assert not output.exists()
    assert list(tmp_path.glob(".failed-*")) == []
