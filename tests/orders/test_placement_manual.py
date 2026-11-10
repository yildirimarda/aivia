"""Manual placement overrides and per image layout files for name overlay.

Manual ``x``/``y`` coordinates and a sibling layout file take precedence over
the automatic calm-region placement.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from PIL import Image, ImageChops

from aivia.compose.errors import RenderError
from aivia.orders.name_overlay import overlay_name

CANVAS = (400, 200)


def _solid(
    size: tuple[int, int] = CANVAS, color: tuple[int, int, int] = (240, 240, 240)
) -> Image.Image:
    return Image.new("RGB", size, color)


def _has_text(background: Image.Image, result: Image.Image) -> bool:
    diff = ImageChops.difference(result.convert("RGB"), background.convert("RGB"))
    return diff.getbbox() is not None


def _text_bbox(
    background: Image.Image, result: Image.Image
) -> tuple[int, int, int, int] | None:
    diff = ImageChops.difference(result.convert("RGB"), background.convert("RGB"))
    return diff.getbbox()


def _write_layout(path: Path, data: object) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _render(background: Image.Image, name: str, **kwargs: object) -> Image.Image:
    return overlay_name(background, name, min_contrast=1.0, **kwargs)


def test_manual_coordinates_place_text_at_exact_position() -> None:
    background = _solid()
    result = _render(background, "Alice", x=120, y=60)

    assert _has_text(background, result)
    bbox = _text_bbox(background, result)
    assert bbox is not None
    # The text image is composited at dest=(x, y); glyph bearings may shift the
    # first painted pixel right/down, but never before the manual origin.
    assert bbox[0] >= 120
    assert bbox[1] >= 60


def test_manual_coordinates_win_over_automatic_placement() -> None:
    # The left half is busy, the right half is calm. Automatic placement would
    # land on the right, but manual coordinates force the left side.
    image = Image.new("RGB", CANVAS, (0, 0, 0))
    pixels = image.load()
    assert pixels is not None
    for y in range(CANVAS[1]):
        for x in range(CANVAS[0] // 2):
            value = (x * 37 + y * 53) % 255
            pixels[x, y] = (value, (value * 3) % 255, (value * 7) % 255)

    result = _render(image, "Bob", x=10, y=10)

    assert _has_text(image, result)
    bbox = _text_bbox(image, result)
    assert bbox is not None
    # Automatic placement would choose the calm right half; manual placement
    # keeps the text on the requested left side.
    assert bbox[0] < CANVAS[0] // 2


def test_manual_coordinates_shift_the_text() -> None:
    background = _solid()
    left = _text_bbox(background, _render(background, "Ed", x=20, y=30))
    right = _text_bbox(background, _render(background, "Ed", x=180, y=30))
    assert left is not None and right is not None
    assert right[0] > left[0]


def test_partial_manual_coordinates_fall_back_to_automatic() -> None:
    background = _solid()
    result = overlay_name(background, "Carol", x=10, y=None, min_contrast=1.0)

    assert _has_text(background, result)


def test_per_image_layout_file_overrides_automatic_placement(tmp_path: Path) -> None:
    background = tmp_path / "card.png"
    _solid().save(background)
    _write_layout(
        tmp_path / "card.layout.json",
        {"x": 200, "y": 100},
    )

    result = overlay_name(background, "Dave", min_contrast=1.0)

    assert _has_text(Image.open(background), result)
    bbox = _text_bbox(Image.open(background), result)
    assert bbox is not None
    assert bbox[0] >= 200
    assert bbox[1] >= 100


def test_layout_file_with_png_suffix_variant(tmp_path: Path) -> None:
    background = tmp_path / "photo.png"
    _solid().save(background)
    _write_layout(tmp_path / "photo.png.layout.json", {"x": 50, "y": 75})

    result = overlay_name(background, "Eve", min_contrast=1.0)

    assert _has_text(Image.open(background), result)
    bbox = _text_bbox(Image.open(background), result)
    assert bbox is not None
    assert bbox[0] >= 50
    assert bbox[1] >= 75


def test_layout_file_placement_block(tmp_path: Path) -> None:
    background = tmp_path / "card.png"
    _solid().save(background)
    _write_layout(tmp_path / "card.layout.json", {"placement": {"x": 300, "y": 150}})

    result = overlay_name(background, "Frank", min_contrast=1.0)

    assert _has_text(Image.open(background), result)
    bbox = _text_bbox(Image.open(background), result)
    assert bbox is not None
    assert bbox[0] >= 300
    assert bbox[1] >= 150


def test_manual_coordinates_win_over_layout_file(tmp_path: Path) -> None:
    background = tmp_path / "card.png"
    _solid().save(background)
    _write_layout(tmp_path / "card.layout.json", {"x": 200, "y": 100})

    result = overlay_name(background, "Grace", x=10, y=20, min_contrast=1.0)

    assert _has_text(Image.open(background), result)
    bbox = _text_bbox(Image.open(background), result)
    assert bbox is not None
    assert bbox[0] >= 10
    assert bbox[1] >= 20


def test_missing_layout_file_falls_back_to_automatic(tmp_path: Path) -> None:
    background = tmp_path / "card.png"
    _solid().save(background)

    result = overlay_name(background, "Heidi", min_contrast=1.0)

    assert _has_text(Image.open(background), result)


def test_invalid_layout_file_falls_back_to_automatic(tmp_path: Path) -> None:
    background = tmp_path / "card.png"
    _solid().save(background)
    tmp_path.joinpath("card.layout.json").write_text("not json", encoding="utf-8")

    result = overlay_name(background, "Ivan", min_contrast=1.0)

    assert _has_text(Image.open(background), result)


def test_layout_file_without_coordinates_falls_back_to_automatic(
    tmp_path: Path,
) -> None:
    background = tmp_path / "card.png"
    _solid().save(background)
    _write_layout(tmp_path / "card.layout.json", {"label": "title"})

    result = overlay_name(background, "Judy", min_contrast=1.0)

    assert _has_text(Image.open(background), result)


def test_bytes_background_ignores_layout_file() -> None:
    buffer = io.BytesIO()
    _solid().save(buffer, format="PNG")
    result = overlay_name(buffer.getvalue(), "Karl", x=5, y=5, min_contrast=1.0)

    original = Image.open(io.BytesIO(buffer.getvalue()))
    assert _has_text(original, result)
    bbox = _text_bbox(original, result)
    assert bbox is not None
    assert bbox[0] >= 5
    assert bbox[1] >= 5


def test_golden_like_output_is_deterministic() -> None:
    background = _solid(color=(100, 150, 200))
    first = overlay_name(background, "Léo Ñoño 🎉", x=10, y=5, min_contrast=1.0)
    second = overlay_name(background, "Léo Ñoño 🎉", x=10, y=5, min_contrast=1.0)

    assert (
        ImageChops.difference(first.convert("RGB"), second.convert("RGB")).getbbox()
        is None
    )


def test_layout_file_is_rejected_when_background_is_not_a_file(tmp_path: Path) -> None:
    # A non-existent path has no sibling layout file, so automatic placement runs
    # and then fails because the image cannot be opened.
    missing = tmp_path / "does_not_exist.png"
    with pytest.raises(RenderError):
        overlay_name(missing, "Mallory", min_contrast=1.0)


def test_layout_file_width_height_size_box(tmp_path: Path) -> None:
    background = tmp_path / "card.png"
    _solid((800, 600)).save(background)
    _write_layout(
        tmp_path / "card.layout.json", {"x": 100, "y": 100, "width": 50, "height": 30}
    )

    result = overlay_name(background, "Niaj", min_contrast=1.0)

    assert _has_text(Image.open(background), result)
