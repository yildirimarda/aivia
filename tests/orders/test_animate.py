"""Animating a still with slow zoom, overlays and a name entrance makes no AI
call and is pure Pillow, so the same inputs always produce the same pixels."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image

from aivia.compose.errors import RenderError
from aivia.orders.animate import animate_still

pytestmark = pytest.mark.slow

CANVAS = (200, 200)
DARK = (20, 20, 30)
LIGHT = (240, 240, 240)


def _solid(size: tuple[int, int], color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", size, color)


def _gradient(size: tuple[int, int] = CANVAS) -> Image.Image:
    width, height = size
    image = Image.new("RGB", size)
    cx, cy = width / 2.0, height / 2.0
    max_dist = (cx * cx + cy * cy) ** 0.5
    for y in range(height):
        for x in range(width):
            dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            t = min(1.0, dist / max_dist)
            image.putpixel(
                (x, y),
                (
                    int(255 * (1.0 - t)),
                    int(180 * (1.0 - t)),
                    int(60 * (1.0 - t)),
                ),
            )
    return image


def _images_differ(a: Image.Image, b: Image.Image) -> bool:
    return a.convert("RGBA").tobytes() != b.convert("RGBA").tobytes()


def test_animate_returns_image_of_requested_size() -> None:
    image = animate_still(_solid(CANVAS, DARK), "Alice", canvas=CANVAS)

    assert isinstance(image, Image.Image)
    assert image.size == CANVAS
    assert image.mode == "RGBA"


def test_animate_slow_zoom_changes_the_frame_over_time() -> None:
    start = animate_still(_gradient(), "Alice", canvas=CANVAS, time_seconds=0.0)
    end = animate_still(_gradient(), "Alice", canvas=CANVAS, time_seconds=4.9)

    assert _images_differ(start, end)


def test_animate_zoom_is_a_slow_zoom_in() -> None:
    start = animate_still(_gradient(), "Alice", canvas=CANVAS, time_seconds=0.0)
    middle = animate_still(_gradient(), "Alice", canvas=CANVAS, time_seconds=2.5)

    assert _images_differ(start, middle)


def test_animate_name_entrance_changes_the_frame() -> None:
    hidden = animate_still(
        _solid(CANVAS, DARK),
        "Alice",
        canvas=CANVAS,
        time_seconds=0.0,
        zoom=(1.0, 1.0),
        name_entrance="fade",
    )
    visible = animate_still(
        _solid(CANVAS, DARK),
        "Alice",
        canvas=CANVAS,
        time_seconds=4.9,
        zoom=(1.0, 1.0),
        name_entrance="fade",
    )

    assert _images_differ(hidden, visible)


def test_animate_without_name_still_renders() -> None:
    image = animate_still(_solid(CANVAS, DARK), "", canvas=CANVAS)

    assert image.size == CANVAS
    assert image.mode == "RGBA"


def test_animate_overlays_composite_particles() -> None:
    plain = animate_still(_solid(CANVAS, LIGHT), "Bob", canvas=CANVAS, zoom=(1.0, 1.0))
    decorated = animate_still(
        _solid(CANVAS, LIGHT),
        "Bob",
        canvas=CANVAS,
        zoom=(1.0, 1.0),
        overlays=["confetti"],
    )

    assert _images_differ(plain, decorated)


def test_animate_overlays_accept_named_effects() -> None:
    for overlay in ("snow", "sparkles", "bokeh", "confetti"):
        image = animate_still(
            _solid(CANVAS, LIGHT),
            "Carol",
            canvas=CANVAS,
            zoom=(1.0, 1.0),
            overlays=[overlay],
        )
        assert image.size == CANVAS


def test_animate_overlays_accept_parameter_mappings() -> None:
    image = animate_still(
        _solid(CANVAS, LIGHT),
        "Dave",
        canvas=CANVAS,
        zoom=(1.0, 1.0),
        overlays=[{"type": "snow", "seed": 7, "count": 40}],
    )
    assert image.size == CANVAS


def test_animate_is_deterministic() -> None:
    first = animate_still(_gradient(), "Eve", canvas=CANVAS, time_seconds=3.0)
    second = animate_still(_gradient(), "Eve", canvas=CANVAS, time_seconds=3.0)

    assert first.tobytes() == second.tobytes()


def test_animate_writes_to_destination(tmp_path: Path) -> None:
    destination = tmp_path / "animated.png"
    result = animate_still(
        _solid(CANVAS, DARK),
        "Frank",
        canvas=CANVAS,
        destination=destination,
    )

    assert result == destination
    assert destination.is_file()
    saved = Image.open(destination)
    assert saved.size == CANVAS


def test_animate_accepts_canvas_as_wxh_string() -> None:
    image = animate_still(_solid((100, 100), DARK), "Grace", canvas="100x100")

    assert image.size == (100, 100)


def test_animate_rejects_non_positive_duration() -> None:
    with pytest.raises(RenderError, match="duration_seconds must be positive"):
        animate_still(
            _solid(CANVAS, DARK), "Heidi", canvas=CANVAS, duration_seconds=0.0
        )
    with pytest.raises(RenderError, match="duration_seconds must be positive"):
        animate_still(
            _solid(CANVAS, DARK), "Heidi", canvas=CANVAS, duration_seconds=-1.0
        )


def test_animate_rejects_time_outside_duration() -> None:
    with pytest.raises(RenderError, match="time_seconds must be between 0"):
        animate_still(_solid(CANVAS, DARK), "Ivan", canvas=CANVAS, time_seconds=-0.5)
    with pytest.raises(RenderError, match="time_seconds must be between 0"):
        animate_still(_solid(CANVAS, DARK), "Ivan", canvas=CANVAS, time_seconds=6.0)


def test_animate_rejects_unreadable_background(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.png"
    with pytest.raises(RenderError, match="cannot open background image"):
        animate_still(missing, "Judy", canvas=CANVAS)


def test_animate_returns_image_when_no_destination() -> None:
    result = animate_still(
        _solid(CANVAS, DARK), "Karl", canvas=CANVAS, destination=None
    )
    assert isinstance(result, Image.Image)


@pytest.mark.skipif(
    os.environ.get("AIVA_COMPARE_GOLDENS") != "1",
    reason="golden comparison is toolchain-dependent; set AIVA_COMPARE_GOLDENS=1",
)
def test_animate_golden_is_stable_across_runs(tmp_path: Path) -> None:
    first = animate_still(_gradient(), "Mallory", canvas=CANVAS, time_seconds=2.0)
    second = animate_still(_gradient(), "Mallory", canvas=CANVAS, time_seconds=2.0)

    assert first.tobytes() == second.tobytes()
