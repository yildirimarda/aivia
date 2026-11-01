"""Seamless loop outputs have similar first and last frames."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from PIL import Image

from aivia.compose import (
    Composition,
    LoopOutputError,
    LoopOutputFormat,
    LoopRenderResult,
    LoopSeamError,
    LoopSizeError,
    export_loop,
    export_seamless,
    export_seamless_loop,
    frame_similarity,
    measure_loop_seam,
    render_loop,
    render_seamless,
    render_seamless_loop,
    seam_similarity,
)
from aivia.core.media_probe import run_ffprobe
from aivia.media.video_utils import VideoOutputFormat

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

pytestmark = pytest.mark.slow

requires_ffmpeg = pytest.mark.skipif(
    FFMPEG is None or FFPROBE is None,
    reason="ffmpeg and ffprobe are required for a real loop render",
)


def _composition(
    tmp_path: Path,
    *,
    width: int = 128,
    height: int = 128,
    duration_seconds: float = 1.0,
    fps: int = 12,
) -> Composition:
    asset = tmp_path / "box.png"
    Image.new("RGB", (16, 16), (255, 0, 0)).save(asset)
    return Composition.model_validate(
        {
            "canvas": {"width": width, "height": height, "background": "#000000"},
            "duration_seconds": duration_seconds,
            "fps": fps,
            "layers": [
                {
                    "id": "box",
                    "type": "image",
                    "source": str(asset),
                    "x": 0,
                    "y": 0,
                    "width": 32,
                    "height": 32,
                    "keyframes": {
                        "position": [
                            {"time": 0, "value": [0, 0], "easing": "linear"},
                            {
                                "time": duration_seconds,
                                "value": [width - 32, height - 32],
                            },
                        ]
                    },
                }
            ],
        }
    )


def _still(tmp_path: Path, name: str, color: tuple[int, int, int]) -> Path:
    path = tmp_path / name
    Image.new("RGB", (40, 30), color).save(path)
    return path


def _animated(tmp_path: Path, name: str, *colors: tuple[int, int, int]) -> Path:
    path = tmp_path / name
    frames = [Image.new("RGB", (40, 30), color) for color in colors]
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
    )
    return path


def _loop_info(path: Path) -> int | None:
    with Image.open(path) as image:
        value = image.info.get("loop")
        return None if value is None else int(value)


def test_frame_similarity_is_one_for_identical_frames(tmp_path: Path) -> None:
    red = _still(tmp_path, "red.png", (255, 0, 0))

    assert frame_similarity(red, red) == pytest.approx(1.0)
    assert seam_similarity(red, red) == pytest.approx(1.0)


def test_frame_similarity_is_zero_for_opposite_frames(tmp_path: Path) -> None:
    black = _still(tmp_path, "black.png", (0, 0, 0))
    white = _still(tmp_path, "white.png", (255, 255, 255))

    assert frame_similarity(black, white) == pytest.approx(0.0)


def test_frame_similarity_rejects_different_sizes(tmp_path: Path) -> None:
    small = _still(tmp_path, "small.png", (255, 0, 0))
    large = tmp_path / "large.png"
    Image.new("RGB", (80, 60), (255, 0, 0)).save(large)

    with pytest.raises(LoopOutputError, match="same size"):
        frame_similarity(small, large)


def test_measure_loop_seam_reads_animated_containers(tmp_path: Path) -> None:
    palindrome = _animated(
        tmp_path,
        "palindrome.gif",
        (255, 0, 0),
        (0, 0, 255),
        (255, 0, 0),
    )
    jumping = _animated(
        tmp_path,
        "jumping.gif",
        (255, 0, 0),
        (0, 0, 255),
        (0, 255, 0),
    )

    assert measure_loop_seam(palindrome) == pytest.approx(1.0)
    assert measure_loop_seam(jumping) < 0.5


def test_invalid_loop_outputs_are_rejected_before_rendering(
    tmp_path: Path,
) -> None:
    composition = _composition(tmp_path)

    with pytest.raises(LoopOutputError, match="mp4, gif or webp"):
        render_loop(composition, tmp_path / "out.avi")
    with pytest.raises(LoopOutputError, match="unsupported loop format"):
        render_loop(composition, tmp_path / "out.mp4", output_format="avi")
    with pytest.raises(LoopOutputError, match="cannot infer a loop format"):
        render_loop(composition, tmp_path / "out")
    with pytest.raises(LoopOutputError, match="loop"):
        render_loop(composition, tmp_path / "out.mp4", loop=-2)
    with pytest.raises(LoopOutputError, match="loop"):
        render_loop(composition, tmp_path / "out.mp4", loop=65536)
    with pytest.raises(LoopOutputError, match="max_bytes"):
        render_loop(composition, tmp_path / "out.mp4", max_bytes=0)
    with pytest.raises(LoopOutputError, match="fps"):
        render_loop(composition, tmp_path / "out.mp4", fps=0)
    with pytest.raises(LoopOutputError, match="quality"):
        render_loop(composition, tmp_path / "out.mp4", quality=101)
    with pytest.raises(LoopOutputError, match="seam threshold"):
        render_loop(composition, tmp_path / "out.mp4", seam_threshold=1.1)


def test_measure_loop_seam_rejects_a_missing_output(tmp_path: Path) -> None:
    with pytest.raises(LoopOutputError, match="does not exist"):
        measure_loop_seam(tmp_path / "missing.mp4")


@requires_ffmpeg
def test_mp4_loop_doubles_the_duration_and_closes_the_seam(
    tmp_path: Path,
) -> None:
    composition = _composition(tmp_path)
    output = tmp_path / "loop.mp4"

    result = render_loop(composition, output)

    assert isinstance(result, LoopRenderResult)
    assert output.is_file()
    assert result.format is VideoOutputFormat.MP4
    assert result.seam_similarity >= result.seam_threshold
    assert result.seamless
    probe = run_ffprobe(output)
    assert probe.duration_seconds == pytest.approx(
        composition.duration_seconds * 2.0,
        abs=0.3,
    )
    assert measure_loop_seam(output) >= result.seam_threshold


@requires_ffmpeg
@pytest.mark.parametrize("suffix", [".gif", ".webp"])
def test_animated_loops_honour_the_loop_flag_and_close_the_seam(
    tmp_path: Path,
    suffix: str,
) -> None:
    output = tmp_path / f"loop{suffix}"

    result = render_loop(_composition(tmp_path), output, loop=3, fps=10)

    assert result.format.value == suffix.lstrip(".")
    assert result.loop == 3
    assert result.seam_similarity >= result.seam_threshold
    assert _loop_info(output) == 3
    assert measure_loop_seam(output) >= result.seam_threshold


@requires_ffmpeg
def test_a_loop_size_cap_retries_until_the_file_fits(tmp_path: Path) -> None:
    composition = _composition(
        tmp_path,
        width=128,
        height=128,
        duration_seconds=2.0,
        fps=24,
    )
    uncapped = render_loop(composition, tmp_path / "uncapped.gif", fps=15)
    cap = max(100, int(uncapped.size_bytes * 0.9))
    output = tmp_path / "capped.gif"

    result = render_loop(composition, output, fps=15, max_bytes=cap)

    assert result.max_bytes == cap
    assert result.size_bytes <= cap
    assert output.stat().st_size <= cap
    assert result.seam_similarity >= result.seam_threshold


@requires_ffmpeg
def test_an_impossible_loop_size_cap_leaves_no_output(tmp_path: Path) -> None:
    output = tmp_path / "too-big.gif"

    with pytest.raises(LoopSizeError) as raised:
        render_loop(_composition(tmp_path), output, fps=10, max_bytes=50)

    assert raised.value.details["max_bytes"] == 50
    assert not output.exists()


@requires_ffmpeg
def test_a_failed_seam_check_removes_the_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aivia.compose import loop_outputs

    monkeypatch.setattr(loop_outputs, "measure_loop_seam", lambda *a, **k: 0.1)
    output = tmp_path / "rejected.mp4"

    with pytest.raises(LoopSeamError) as raised:
        render_loop(_composition(tmp_path), output, seam_threshold=0.9)

    assert raised.value.details["similarity"] == pytest.approx(0.1)
    assert raised.value.details["threshold"] == pytest.approx(0.9)
    assert not output.exists()


def test_seamless_loop_aliases_point_at_the_same_renderer() -> None:
    assert render_seamless_loop is render_loop
    assert render_seamless is render_loop
    assert export_loop is render_loop
    assert export_seamless is render_loop
    assert export_seamless_loop is render_loop
    assert LoopOutputFormat is VideoOutputFormat
