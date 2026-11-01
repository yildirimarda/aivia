"""Animated GIF and WebP outputs honour loop flags and byte caps."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from PIL import Image

from aivia.compose import (
    AnimatedOutputError,
    AnimatedRenderResult,
    AnimatedSizeError,
    Composition,
    render_animated,
    render_gif,
    render_video,
    render_webp,
)
from aivia.media.video_utils import VideoSizeError

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

pytestmark = pytest.mark.slow

requires_ffmpeg = pytest.mark.skipif(
    FFMPEG is None or FFPROBE is None,
    reason="ffmpeg and ffprobe are required for a real animated render",
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


def _frames(path: Path) -> int:
    with Image.open(path) as image:
        return int(getattr(image, "n_frames", 1))


def _loop_info(path: Path) -> int | None:
    with Image.open(path) as image:
        value = image.info.get("loop")
        return None if value is None else int(value)


@requires_ffmpeg
def test_gif_is_animated_and_loops_forever_by_default(tmp_path: Path) -> None:
    output = tmp_path / "out.gif"

    result = render_animated(_composition(tmp_path), output, fps=10)

    assert result.output == output
    assert output.read_bytes()[:6] in {b"GIF87a", b"GIF89a"}
    assert _frames(output) > 1
    assert _loop_info(output) == 0
    assert result.format.value == "gif"
    assert result.loop == 0
    assert result.size_bytes == output.stat().st_size


@requires_ffmpeg
def test_webp_is_animated_and_keeps_a_positive_loop_count(tmp_path: Path) -> None:
    output = tmp_path / "out.webp"

    result = render_animated(_composition(tmp_path), output, loop=3, fps=10)

    payload = output.read_bytes()
    assert payload[:4] == b"RIFF"
    assert payload[8:12] == b"WEBP"
    assert _frames(output) > 1
    assert _loop_info(output) == 3
    assert result.format.value == "webp"
    assert result.loop == 3


@requires_ffmpeg
@pytest.mark.parametrize("suffix", [".gif", ".webp"])
def test_no_loop_is_respected_for_both_containers(tmp_path: Path, suffix: str) -> None:
    output = tmp_path / f"once{suffix}"

    result = render_animated(_composition(tmp_path), output, loop=False, fps=10)

    assert result.loop == -1
    if suffix == ".gif":
        assert _loop_info(output) is None
    else:
        assert result.container_loop == 1
        assert _loop_info(output) == 1


@requires_ffmpeg
@pytest.mark.parametrize("suffix", [".gif", ".webp"])
def test_a_size_cap_retries_until_the_file_fits(tmp_path: Path, suffix: str) -> None:
    composition = _composition(
        tmp_path,
        width=128,
        height=128,
        duration_seconds=2.0,
        fps=24,
    )
    uncapped = render_animated(composition, tmp_path / f"uncapped{suffix}", fps=15)
    cap = max(100, int(uncapped.size_bytes * 0.9))
    output = tmp_path / f"capped{suffix}"

    result = render_animated(composition, output, fps=15, max_bytes=cap)

    assert result.max_bytes == cap
    assert result.size_bytes <= cap
    assert output.stat().st_size <= cap


@requires_ffmpeg
@pytest.mark.parametrize("suffix", [".gif", ".webp"])
def test_an_impossible_size_cap_leaves_no_output(tmp_path: Path, suffix: str) -> None:
    output = tmp_path / f"too-big{suffix}"

    with pytest.raises(AnimatedSizeError) as raised:
        render_animated(_composition(tmp_path), output, fps=10, max_bytes=50)

    assert isinstance(raised.value, VideoSizeError)
    assert raised.value.details["max_bytes"] == 50
    assert not output.exists()


@requires_ffmpeg
def test_convenience_helpers_and_render_video_dispatch(tmp_path: Path) -> None:
    composition = _composition(tmp_path)

    gif = render_gif(composition, tmp_path / "helper.gif", fps=10)
    webp = render_webp(composition, tmp_path / "helper.webp", fps=10)
    dispatched = render_video(composition, tmp_path / "dispatch.gif", fps=10)

    assert gif.format.value == "gif"
    assert webp.format.value == "webp"
    assert isinstance(dispatched, AnimatedRenderResult)
    assert dispatched.format.value == "gif"


def test_invalid_animated_outputs_are_rejected_before_rendering(
    tmp_path: Path,
) -> None:
    composition = _composition(tmp_path)

    with pytest.raises(AnimatedOutputError, match="gif or webp"):
        render_animated(composition, tmp_path / "out.mp4")
    with pytest.raises(AnimatedOutputError, match="unsupported animated format"):
        render_animated(composition, tmp_path / "out.gif", output_format="mp4")
    with pytest.raises(AnimatedOutputError, match="loop"):
        render_animated(composition, tmp_path / "out.gif", loop=-2)
    with pytest.raises(AnimatedOutputError, match="loop"):
        render_animated(composition, tmp_path / "out.gif", loop=65536)
    with pytest.raises(AnimatedOutputError, match="max_bytes"):
        render_animated(composition, tmp_path / "out.gif", max_bytes=0)
    with pytest.raises(AnimatedOutputError, match="fps"):
        render_animated(composition, tmp_path / "out.gif", fps=0)
    with pytest.raises(AnimatedOutputError, match="quality"):
        render_animated(composition, tmp_path / "out.gif", quality=101)
