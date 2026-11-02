"""Keyframed motion in video, including Ken Burns and parallax."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from aivia.compose import (
    Composition,
    Layer,
    Motion,
    MotionError,
    build_ffmpeg_filtergraph,
    ken_burns,
    motion_for_layer,
    motion_tracks,
    pair_expression,
    parallax,
    render_still,
    render_video,
    track_expression,
    tracks_for_layer,
)

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

pytestmark = pytest.mark.slow

requires_ffmpeg = pytest.mark.skipif(
    FFMPEG is None or FFPROBE is None,
    reason="ffmpeg and ffprobe are required for a real video render",
)

BLUE = (0, 0, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)


def _run(argv: list[str]) -> None:
    subprocess.run(argv, check=True, capture_output=True, text=True)


def _solid(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> Path:
    Image.new("RGB", size, color).save(path)
    return path


def _center_box(path: Path) -> Path:
    image = Image.new("RGB", (320, 240), BLUE)
    draw = ImageDraw.Draw(image)
    draw.rectangle((80, 60, 239, 179), fill=RED)
    image.save(path)
    return path


def _sticker(path: Path) -> Path:
    image = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 29, 29), fill=(255, 0, 0, 255))
    image.save(path)
    return path


def _video(path: Path, color: str = "0x0000ff") -> Path:
    assert FFMPEG is not None
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
            f"color=c={color}:s=64x48:r=24:d=2.000000",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(path),
        ]
    )
    return path


def _composition(*layers: dict[str, object], **overrides: object) -> Composition:
    values: dict[str, object] = {
        "canvas": {"width": 320, "height": 240, "background": "#000000"},
        "duration_seconds": 2.0,
        "fps": 24,
        "layers": list(layers),
    }
    values.update(overrides)
    return Composition.model_validate(values)


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
    expected: tuple[int, ...],
    *,
    tolerance: int = 20,
) -> None:
    actual = image.getpixel(point)
    assert isinstance(actual, tuple)
    for channel, target in zip(actual, expected, strict=True):
        assert abs(channel - target) <= tolerance, (point, actual, expected)


def test_ken_burns_generates_a_zoom_and_a_pan_track() -> None:
    motion = ken_burns(
        start_scale=1.0,
        end_scale=2.0,
        start_pan=(0, 0),
        end_pan=(40, -20),
        easing="linear",
    )

    tracks = motion_tracks(motion, 0.0, 2.0)

    assert tracks.scale is not None
    assert tracks.pan is not None
    assert tracks.position is None
    assert tracks.scale.sample(1.0) == pytest.approx(1.5)
    assert tracks.pan.sample(1.0) == pytest.approx((20.0, -10.0))


def test_parallax_scales_its_drift_by_depth() -> None:
    motion = parallax(depth=0.5, drift=(100, 40), start_pan=(10, 5))

    tracks = motion_tracks(motion, 0.0, 2.0)

    assert tracks.position is not None
    assert tracks.position.sample(0.0) == pytest.approx((10.0, 5.0))
    assert tracks.position.sample(2.0) == pytest.approx((60.0, 25.0))


def test_explicit_keyframes_win_over_generated_motion() -> None:
    layer = Layer.model_validate(
        {
            "type": "image",
            "source": "still.png",
            "start": 1.0,
            "end": 3.0,
            "motion": ken_burns(start_scale=1.0, end_scale=2.0, end_pan=(10, 0)),
            "keyframes": {"scale": [(1, 1.0), (3, 4.0)]},
        }
    )

    tracks = tracks_for_layer(layer, 1.0, 3.0)

    assert tracks.scale is not None
    assert tracks.scale.sample(2.0) == pytest.approx(2.5)
    assert tracks.pan is not None


def test_motion_shorthand_infers_parallax_and_accepts_zoom_and_pan_pairs() -> None:
    inferred = Motion.model_validate({"depth": 0.5, "drift": [10, 0]})
    shorthand = Motion.model_validate({"zoom": [1, 2], "pan": [[0, 0], [10, 5]]})

    assert inferred.is_parallax
    assert shorthand.is_ken_burns
    assert shorthand.start_scale == pytest.approx(1.0)
    assert shorthand.end_scale == pytest.approx(2.0)
    assert shorthand.start_pan == pytest.approx((0.0, 0.0))
    assert shorthand.end_pan == pytest.approx((10.0, 5.0))


def test_motion_accepts_a_string_and_rejects_unknown_presets() -> None:
    assert Motion.model_validate("ken_burns").type.value == "ken_burns"
    assert Motion.model_validate("parallax").type.value == "parallax"
    with pytest.raises(ValueError):
        Motion.model_validate("dolly_zoom")


def test_expressions_are_piecewise_and_follow_the_keyframes() -> None:
    motion = ken_burns(
        start_scale=1.0,
        end_scale=2.0,
        start_pan=(0, 0),
        end_pan=(100, -50),
        easing="linear",
    )
    tracks = motion_tracks(motion, 0.0, 2.0)
    assert tracks.scale is not None
    assert tracks.pan is not None

    scale = track_expression(tracks.scale)
    pan_x, pan_y = pair_expression(tracks.pan)

    assert "if(lt(t,0)" in scale
    assert "if(lt(t,2)" in scale
    assert "100" in pan_x
    assert "-50" in pan_y


def test_a_layer_motion_can_be_read_from_the_schema() -> None:
    layer = Layer.model_validate(
        {
            "type": "image",
            "source": "still.png",
            "motion": {"type": "parallax", "depth": 0.25, "drift": [40, 0]},
        }
    )

    assert isinstance(layer.motion, Motion)
    assert layer.motion is not None
    assert layer.motion.depth == pytest.approx(0.25)

    with pytest.raises(ValueError):
        Layer.model_validate(
            {
                "type": "image",
                "source": "still.png",
                "motion": {"type": "parallax", "depth": -1},
            }
        )

    class Broken:
        motion = 42

    with pytest.raises(MotionError):
        motion_for_layer(Broken())


@requires_ffmpeg
def test_ken_burns_zoom_is_visible_in_sampled_frames(tmp_path: Path) -> None:
    source = _center_box(tmp_path / "center.png")
    output = tmp_path / "ken-burns.mp4"
    render_video(
        _composition(
            {
                "id": "still",
                "type": "image",
                "source": str(source),
                "fit": "stretch",
                "motion": ken_burns(start_scale=1.0, end_scale=2.0),
            }
        ),
        output,
        preset="ultrafast",
        crf=28,
    )

    start = _frame(output, 0.0, tmp_path / "start.png")
    middle = _frame(output, 1.0, tmp_path / "middle.png")
    end = _frame(output, 1.9, tmp_path / "end.png")

    _assert_color(start, (10, 10), BLUE)
    _assert_color(middle, (10, 10), BLUE)
    _assert_color(end, (10, 10), RED)
    _assert_color(start, (160, 120), RED)
    _assert_color(end, (160, 120), RED)


def test_image_unit_zoom_pan_translates_the_layer_without_zoompan() -> None:
    composition = _composition(
        {
            "id": "still",
            "type": "image",
            "source": "center.png",
            "fit": "stretch",
            "motion": ken_burns(
                start_scale=1.0,
                end_scale=1.0,
                start_pan=(0, 0),
                end_pan=(80, 0),
                easing="linear",
            ),
        }
    )

    graph = build_ffmpeg_filtergraph(composition)

    assert "zoompan" not in graph
    assert "overlay=x='(0)-(" in graph
    assert "80*" in graph


@requires_ffmpeg
def test_image_motion_pan_at_unit_zoom_matches_the_still_renderer(
    tmp_path: Path,
) -> None:
    source = _center_box(tmp_path / "center.png")
    composition = _composition(
        {
            "id": "still",
            "type": "image",
            "source": str(source),
            "fit": "stretch",
            "motion": ken_burns(
                start_scale=1.0,
                end_scale=1.0,
                start_pan=(0, 0),
                end_pan=(80, 0),
                easing="linear",
            ),
        }
    )
    output = tmp_path / "unit-zoom-pan.mp4"
    render_video(composition, output, preset="ultrafast", crf=28)

    start = _frame(output, 0.0, tmp_path / "start.png")
    end = _frame(output, 1.9, tmp_path / "end.png")
    still_start = _still(composition, 0.0, tmp_path)
    still_end = _still(composition, 1.9, tmp_path)

    for point in ((10, 10), (100, 120), (200, 120), (300, 120)):
        expected_start = still_start.getpixel(point)
        expected_end = still_end.getpixel(point)
        assert isinstance(expected_start, tuple)
        assert isinstance(expected_end, tuple)
        _assert_color(start, point, expected_start)
        _assert_color(end, point, expected_end)

    _assert_color(start, (200, 120), RED)
    _assert_color(end, (200, 120), BLUE)
    _assert_color(end, (100, 120), RED)


@requires_ffmpeg
def test_parallax_moves_a_layer_across_sampled_frames(tmp_path: Path) -> None:
    source = _solid(tmp_path / "bar.png", (40, 40), RED)
    output = tmp_path / "parallax.mp4"
    render_video(
        _composition(
            {
                "id": "foreground",
                "type": "image",
                "source": str(source),
                "fit": "stretch",
                "x": 0,
                "y": 0,
                "width": 40,
                "height": 40,
                "motion": parallax(depth=1.0, drift=(100, 0)),
            }
        ),
        output,
        preset="ultrafast",
        crf=28,
    )

    start = _frame(output, 0.0, tmp_path / "start.png")
    end = _frame(output, 1.9, tmp_path / "end.png")

    _assert_color(start, (20, 20), RED)
    _assert_color(start, (120, 20), BLACK)
    _assert_color(end, (20, 20), BLACK)
    _assert_color(end, (120, 20), RED)


@requires_ffmpeg
def test_position_keyframes_move_a_video_layer_in_sampled_frames(
    tmp_path: Path,
) -> None:
    source = _video(tmp_path / "clip.mp4")
    output = tmp_path / "keyframed-video.mp4"
    render_video(
        _composition(
            {
                "id": "clip",
                "type": "video",
                "source": str(source),
                "fit": "stretch",
                "width": 40,
                "height": 40,
                "keyframes": {
                    "position": [
                        {"time": 0, "value": [0, 0], "easing": "linear"},
                        {"time": 2, "value": [100, 0]},
                    ]
                },
            }
        ),
        output,
        preset="ultrafast",
        crf=28,
    )

    start = _frame(output, 0.0, tmp_path / "start.png")
    end = _frame(output, 1.9, tmp_path / "end.png")

    _assert_color(start, (20, 20), BLUE)
    _assert_color(start, (120, 20), BLACK)
    _assert_color(end, (20, 20), BLACK)
    _assert_color(end, (120, 20), BLUE)


def test_overlay_motion_builds_a_time_dependent_scale_and_pad() -> None:
    composition = _composition(
        {
            "id": "badge",
            "type": "sticker",
            "source": "badge.png",
            "width": 40,
            "height": 40,
            "motion": ken_burns(start_scale=1.0, end_scale=2.0),
        }
    )

    graph = build_ffmpeg_filtergraph(composition)

    assert "scale=w='40*(" in graph
    assert "eval=frame" in graph
    assert "pad=w=320:h=240" in graph
    assert "zoompan" not in graph


def _scale_keyframe_graph(
    *,
    layer_type: str = "video",
    start: tuple[float, float],
    end: tuple[float, float],
) -> str:
    layer: dict[str, object] = {
        "id": "moving",
        "type": layer_type,
        "source": "clip.mp4" if layer_type == "video" else "badge.png",
        "width": 40,
        "height": 40,
        "keyframes": {
            "scale": [
                {"time": 0, "value": list(start), "easing": "linear"},
                {"time": 2, "value": list(end)},
            ]
        },
    }
    if layer_type == "video":
        layer["fit"] = "stretch"
    return build_ffmpeg_filtergraph(_composition(layer))


def test_pair_scale_keyframes_keep_independent_x_and_y_axes() -> None:
    x_only = _scale_keyframe_graph(start=(1, 1), end=(2, 1))
    y_only = _scale_keyframe_graph(start=(1, 1), end=(1, 2))

    assert "sqrt(" not in x_only
    assert "sqrt(" not in y_only
    assert "zoompan" not in x_only
    assert "zoompan" not in y_only
    assert x_only.count("(1+1*") == 2
    assert y_only.count("(1+1*") == 2
    assert x_only != y_only


def test_overlay_pair_scale_keyframes_keep_independent_x_and_y_axes() -> None:
    x_only = _scale_keyframe_graph(
        layer_type="sticker",
        start=(1, 1),
        end=(2, 1),
    )
    y_only = _scale_keyframe_graph(
        layer_type="sticker",
        start=(1, 1),
        end=(1, 2),
    )

    assert "sqrt(" not in x_only
    assert "sqrt(" not in y_only
    assert x_only.count("(1+1*") == 1
    assert y_only.count("(1+1*") == 1
    assert x_only != y_only


@requires_ffmpeg
def test_sticker_scale_keyframes_grow_across_sampled_frames(
    tmp_path: Path,
) -> None:
    source = _sticker(tmp_path / "badge.png")
    output = tmp_path / "sticker-scale.mp4"
    render_video(
        _composition(
            {
                "id": "badge",
                "type": "sticker",
                "source": str(source),
                "width": 40,
                "height": 40,
                "keyframes": {
                    "scale": [
                        {"time": 0, "value": 1.0, "easing": "linear"},
                        {"time": 2, "value": 2.0},
                    ]
                },
            }
        ),
        output,
        preset="ultrafast",
        crf=28,
    )

    start = _frame(output, 0.0, tmp_path / "start.png")
    end = _frame(output, 1.9, tmp_path / "end.png")

    _assert_color(start, (160, 120), RED)
    _assert_color(start, (145, 120), BLACK)
    _assert_color(end, (160, 120), RED)
    _assert_color(end, (145, 120), RED)
    _assert_color(end, (185, 120), BLACK)


@requires_ffmpeg
def test_logo_motion_preset_scales_and_pans_across_sampled_frames(
    tmp_path: Path,
) -> None:
    source = _sticker(tmp_path / "logo.png")
    output = tmp_path / "logo-motion.mp4"
    render_video(
        _composition(
            {
                "id": "logo",
                "type": "logo",
                "source": str(source),
                "motion": ken_burns(
                    start_scale=1.0,
                    end_scale=2.0,
                    start_pan=(0, 0),
                    end_pan=(60, 0),
                ),
            }
        ),
        output,
        preset="ultrafast",
        crf=28,
    )

    start = _frame(output, 0.0, tmp_path / "start.png")
    end = _frame(output, 1.9, tmp_path / "end.png")

    _assert_color(start, (160, 120), RED)
    _assert_color(start, (220, 120), BLACK)
    _assert_color(end, (160, 120), BLACK)
    _assert_color(end, (220, 120), RED)


@requires_ffmpeg
def test_independent_scale_keyframes_stretch_only_the_x_axis(
    tmp_path: Path,
) -> None:
    source = _center_box(tmp_path / "center.png")
    output = tmp_path / "independent-scale.mp4"
    render_video(
        _composition(
            {
                "id": "still",
                "type": "image",
                "source": str(source),
                "fit": "stretch",
                "x": 140,
                "y": 100,
                "width": 40,
                "height": 40,
                "keyframes": {
                    "scale": [
                        {"time": 0, "value": [1, 1], "easing": "linear"},
                        {"time": 2, "value": [2, 1]},
                    ]
                },
            }
        ),
        output,
        preset="ultrafast",
        crf=28,
    )

    start = _frame(output, 0.0, tmp_path / "start.png")
    end = _frame(output, 1.9, tmp_path / "end.png")

    _assert_color(start, (145, 120), BLUE)
    _assert_color(start, (160, 120), RED)
    _assert_color(start, (175, 120), BLUE)
    _assert_color(end, (145, 120), RED)
    _assert_color(end, (160, 120), RED)
    _assert_color(end, (175, 120), RED)
    _assert_color(start, (160, 105), BLUE)
    _assert_color(end, (160, 105), BLUE)
    _assert_color(start, (160, 135), BLUE)
    _assert_color(end, (160, 135), BLUE)


def _still(
    composition: Composition,
    time_seconds: float,
    source_root: Path,
) -> Image.Image:
    return render_still(
        composition,
        time_seconds,
        source_root=source_root,
    ).convert("RGB")


def test_still_renderer_honors_ken_burns_zoom(tmp_path: Path) -> None:
    source = _center_box(tmp_path / "center.png")
    composition = _composition(
        {
            "id": "still",
            "type": "image",
            "source": str(source),
            "fit": "stretch",
            "motion": ken_burns(start_scale=1.0, end_scale=2.0),
        }
    )

    start = _still(composition, 0.0, tmp_path)
    middle = _still(composition, 1.0, tmp_path)
    end = _still(composition, 1.9, tmp_path)

    _assert_color(start, (10, 10), BLUE)
    _assert_color(middle, (10, 10), BLUE)
    _assert_color(end, (10, 10), RED)
    _assert_color(start, (160, 120), RED)
    _assert_color(end, (160, 120), RED)


def test_still_renderer_honors_ken_burns_pan(tmp_path: Path) -> None:
    source = _center_box(tmp_path / "center.png")
    composition = _composition(
        {
            "id": "still",
            "type": "image",
            "source": str(source),
            "fit": "stretch",
            "motion": ken_burns(
                start_scale=1.0,
                end_scale=1.0,
                start_pan=(0, 0),
                end_pan=(80, 0),
                easing="linear",
            ),
        }
    )

    start = _still(composition, 0.0, tmp_path)
    end = _still(composition, 1.9, tmp_path)

    _assert_color(start, (200, 120), RED)
    _assert_color(end, (200, 120), BLUE)
    _assert_color(end, (100, 120), RED)


def test_still_renderer_honors_parallax_drift(tmp_path: Path) -> None:
    source = _solid(tmp_path / "bar.png", (40, 40), RED)
    composition = _composition(
        {
            "id": "foreground",
            "type": "image",
            "source": str(source),
            "fit": "stretch",
            "x": 0,
            "y": 0,
            "width": 40,
            "height": 40,
            "motion": parallax(depth=1.0, drift=(100, 0)),
        }
    )

    start = _still(composition, 0.0, tmp_path)
    end = _still(composition, 1.9, tmp_path)

    _assert_color(start, (20, 20), RED)
    _assert_color(start, (120, 20), BLACK)
    _assert_color(end, (20, 20), BLACK)
    _assert_color(end, (120, 20), RED)


def test_still_renderer_honors_logo_motion_preset(tmp_path: Path) -> None:
    source = _sticker(tmp_path / "logo.png")
    composition = _composition(
        {
            "id": "logo",
            "type": "logo",
            "source": str(source),
            "motion": ken_burns(
                start_scale=1.0,
                end_scale=2.0,
                start_pan=(0, 0),
                end_pan=(60, 0),
            ),
        }
    )

    start = _still(composition, 0.0, tmp_path)
    end = _still(composition, 1.9, tmp_path)

    _assert_color(start, (160, 120), RED)
    _assert_color(start, (220, 120), BLACK)
    _assert_color(end, (160, 120), BLACK)
    _assert_color(end, (220, 120), RED)


def test_still_renderer_explicit_keyframes_win_over_motion(tmp_path: Path) -> None:
    source = _center_box(tmp_path / "center.png")
    composition = _composition(
        {
            "id": "still",
            "type": "image",
            "source": str(source),
            "fit": "stretch",
            "motion": ken_burns(start_scale=1.0, end_scale=4.0),
            "keyframes": {
                "scale": [
                    {"time": 0, "value": 1.0},
                    {"time": 2, "value": 1.0},
                ]
            },
        }
    )

    end = _still(composition, 1.9, tmp_path)

    _assert_color(end, (10, 10), BLUE)
    _assert_color(end, (160, 120), RED)
