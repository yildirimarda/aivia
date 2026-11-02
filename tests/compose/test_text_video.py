"""Text layers in video with entrance and exit animations."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from aivia.compose import (
    Composition,
    PillowStillRenderer,
    TextAnimation,
    TextAnimationType,
    TextLayer,
    animation_tracks,
    fade,
    render_video,
    slide,
    text_animation_for_layer,
    tracks_for_layer,
    zoom,
)

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

requires_ffmpeg = pytest.mark.skipif(
    FFMPEG is None or FFPROBE is None,
    reason="ffmpeg and ffprobe are required for a real video render",
)

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


def _run(argv: list[str]) -> None:
    subprocess.run(argv, check=True, capture_output=True, text=True)


def _composition(**overrides: object) -> Composition:
    values: dict[str, object] = {
        "canvas": {"width": 320, "height": 240, "background": "#000000"},
        "duration_seconds": 2.0,
        "fps": 24,
        "layers": [
            {
                "id": "title",
                "type": "text",
                "text": "HELLO",
                "font_size": 48,
                "color": "#ffffff",
                "x": 60,
                "y": 80,
                "width": 200,
                "height": 80,
                "entrance": {"type": "fade", "duration": 0.4},
                "exit": {"type": "fade", "duration": 0.4},
            }
        ],
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


def _brightest(image: Image.Image) -> int:
    _, high = image.convert("L").getextrema()
    assert isinstance(high, int)
    return high


def _has_white(image: Image.Image, *, threshold: int = 200) -> bool:
    return _brightest(image) >= threshold


def _bbox(image: Image.Image) -> tuple[int, int, int, int] | None:
    return image.convert("RGB").getbbox()


def test_entrance_and_exit_accept_shorthand_and_mapping() -> None:
    layer = TextLayer.model_validate(
        {
            "text": "Hello",
            "entrance": "fade",
            "exit": {"type": "slide_up", "duration": 0.25, "distance": 30},
        }
    )

    assert isinstance(layer.entrance, TextAnimation)
    assert isinstance(layer.exit, TextAnimation)
    assert layer.entrance.type is TextAnimationType.FADE
    assert layer.exit.type is TextAnimationType.SLIDE_UP
    assert layer.exit.duration == pytest.approx(0.25)
    assert layer.exit.distance == pytest.approx(30)


def test_animation_bundle_is_folded_into_entrance_and_exit() -> None:
    composition = Composition.model_validate(
        {
            "canvas": {"width": 320, "height": 240},
            "duration_seconds": 2.0,
            "layers": [
                {
                    "type": "text",
                    "text": "Hi",
                    "animation": {"entrance": "zoom", "exit": "fade"},
                }
            ],
        }
    )

    animations = text_animation_for_layer(composition.layers[0])

    assert animations is not None
    assert animations.entrance is not None
    assert animations.entrance.type is TextAnimationType.ZOOM
    assert animations.exit is not None
    assert animations.exit.type is TextAnimationType.FADE


def test_fade_is_clear_at_the_start_visible_in_the_middle_and_clear_at_the_end() -> (
    None
):
    tracks = animation_tracks(fade(duration=0.4), fade(duration=0.4), 0.0, 2.0)

    assert tracks.opacity is not None
    assert tracks.opacity.sample(0.0) == pytest.approx(0.0)
    assert tracks.opacity.sample(1.0) == pytest.approx(1.0)
    assert tracks.opacity.sample(2.0) == pytest.approx(0.0)


def test_slide_moves_the_text_in_and_out() -> None:
    tracks = animation_tracks(
        slide("up", duration=0.5, distance=40),
        slide("up", duration=0.5, distance=40),
        0.0,
        2.0,
    )

    assert tracks.position is not None
    assert tracks.position.sample(0.0) == pytest.approx((0.0, 40.0))
    assert tracks.position.sample(1.0) == pytest.approx((0.0, 0.0))
    assert tracks.position.sample(2.0) == pytest.approx((0.0, -40.0))


def test_zoom_scales_the_text_in_and_out() -> None:
    tracks = animation_tracks(
        zoom(duration=0.5, start_scale=0.5),
        zoom(duration=0.5, start_scale=0.5),
        0.0,
        2.0,
    )

    assert tracks.scale is not None
    assert tracks.scale.sample(0.0) == pytest.approx(0.5)
    assert tracks.scale.sample(1.0) == pytest.approx(1.0)
    assert tracks.scale.sample(2.0) == pytest.approx(0.5)


def test_explicit_keyframes_win_over_generated_text_animation() -> None:
    layer = TextLayer.model_validate(
        {
            "text": "Hi",
            "entrance": "fade",
            "keyframes": {"opacity": [(0.0, 1.0), (2.0, 1.0)]},
        }
    )

    tracks = tracks_for_layer(layer, 0.0, 2.0)

    assert tracks.opacity is not None
    assert tracks.opacity.sample(0.0) == pytest.approx(1.0)
    assert tracks.opacity.sample(1.0) == pytest.approx(1.0)


def test_text_without_animation_generates_no_tracks() -> None:
    layer = TextLayer(text="Hi")

    assert text_animation_for_layer(layer) is None
    assert tracks_for_layer(layer, 0.0, 2.0).is_empty


def test_still_preview_fades_text_in_and_out() -> None:
    renderer = PillowStillRenderer()
    composition = _composition()

    start = renderer.render_still(composition, 0.0)
    middle = renderer.render_still(composition, 1.0)
    end = renderer.render_still(composition, 1.95)

    assert _brightest(start) < 40
    assert _has_white(middle)
    assert _brightest(end) < 40


def test_still_preview_slides_text_between_entrance_and_exit() -> None:
    composition = _composition(
        layers=[
            {
                "id": "title",
                "type": "text",
                "text": "HELLO",
                "font_size": 48,
                "color": "#ffffff",
                "x": 60,
                "y": 80,
                "width": 200,
                "height": 80,
                "entrance": {
                    "type": "slide_up",
                    "duration": 0.5,
                    "distance": 30,
                    "opacity_from": 1.0,
                },
                "exit": {
                    "type": "slide_up",
                    "duration": 0.5,
                    "distance": 30,
                    "opacity_from": 1.0,
                },
            }
        ]
    )
    renderer = PillowStillRenderer()

    start = _bbox(renderer.render_still(composition, 0.0))
    middle = _bbox(renderer.render_still(composition, 1.0))
    end = _bbox(renderer.render_still(composition, 1.95))

    assert start is not None and middle is not None and end is not None
    assert start[1] > middle[1] + 10
    assert end[1] < middle[1] - 10


def test_still_preview_scales_text_for_zoom_presets() -> None:
    composition = _composition(
        layers=[
            {
                "id": "title",
                "type": "text",
                "text": "HELLO",
                "font_size": 48,
                "color": "#ffffff",
                "x": 60,
                "y": 80,
                "width": 200,
                "height": 80,
                "entrance": {
                    "type": "zoom",
                    "duration": 0.5,
                    "start_scale": 0.5,
                    "opacity_from": 1.0,
                },
                "exit": {
                    "type": "zoom",
                    "duration": 0.5,
                    "start_scale": 0.5,
                    "opacity_from": 1.0,
                },
            }
        ]
    )
    renderer = PillowStillRenderer()

    start = _bbox(renderer.render_still(composition, 0.0))
    middle = _bbox(renderer.render_still(composition, 1.0))
    end = _bbox(renderer.render_still(composition, 1.95))

    assert start is not None and middle is not None and end is not None
    middle_width = middle[2] - middle[0]
    assert start[2] - start[0] < middle_width - 20
    assert end[2] - end[0] < middle_width - 20


def test_still_preview_keeps_explicit_keyframes_over_entrance() -> None:
    composition = _composition(
        layers=[
            {
                "id": "title",
                "type": "text",
                "text": "HELLO",
                "font_size": 48,
                "color": "#ffffff",
                "x": 60,
                "y": 80,
                "width": 200,
                "height": 80,
                "entrance": {"type": "fade", "duration": 0.4},
                "keyframes": {"opacity": [(0.0, 1.0), (2.0, 1.0)]},
            }
        ]
    )

    start = PillowStillRenderer().render_still(composition, 0.0)

    assert _has_white(start)


def test_an_unknown_text_animation_is_rejected() -> None:
    with pytest.raises(ValueError):
        TextLayer.model_validate({"text": "Hi", "entrance": "wobble"})


@pytest.mark.slow
@requires_ffmpeg
def test_text_fades_in_and_out_in_sampled_frames(tmp_path: Path) -> None:
    output = tmp_path / "text-fade.mp4"

    render_video(_composition(), output, preset="ultrafast", crf=28)

    start = _frame(output, 0.0, tmp_path / "start.png")
    middle = _frame(output, 1.0, tmp_path / "middle.png")
    end = _frame(output, 1.95, tmp_path / "end.png")

    assert _brightest(start) < 40
    assert _has_white(middle)
    assert _brightest(end) < 40
    assert start.getpixel((10, 10)) == BLACK


@pytest.mark.slow
@requires_ffmpeg
def test_text_slide_is_visible_in_the_middle_of_the_layer(tmp_path: Path) -> None:
    composition = _composition(
        layers=[
            {
                "id": "title",
                "type": "text",
                "text": "HELLO",
                "font_size": 48,
                "color": "#ffffff",
                "x": 60,
                "y": 80,
                "width": 200,
                "height": 80,
                "entrance": {"type": "slide_up", "duration": 0.5, "distance": 60},
                "exit": {"type": "slide_up", "duration": 0.5, "distance": 60},
            }
        ]
    )
    output = tmp_path / "text-slide.mp4"

    render_video(composition, output, preset="ultrafast", crf=28)

    start = _frame(output, 0.0, tmp_path / "start.png")
    middle = _frame(output, 1.0, tmp_path / "middle.png")

    assert _brightest(start) < 40
    assert _has_white(middle)
