"""Placing a recipient name on a provided background image avoids faces and
low contrast regions, with golden tests covering special characters."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageChops

from aivia.compose.errors import RenderError
from aivia.orders.name_overlay import (
    DEFAULT_FONT_FAMILY,
    DEFAULT_FONT_SIZE,
    overlay_name,
)

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"


def _solid_background(
    size: tuple[int, int],
    color: tuple[int, int, int],
) -> Image.Image:
    return Image.new("RGB", size, color)


def _has_opaque_pixels(image: Image.Image) -> bool:
    return image.getchannel("A").getbbox() is not None


def test_overlay_name_returns_composited_image() -> None:
    background = _solid_background((400, 200), (240, 240, 240))
    result = overlay_name(background, "Alice")

    assert isinstance(result, Image.Image)
    assert result.size == (400, 200)
    assert result.mode == "RGBA"
    assert _has_opaque_pixels(result)


def test_overlay_name_writes_to_destination(tmp_path: Path) -> None:
    background = _solid_background((400, 200), (240, 240, 240))
    destination = tmp_path / "output.png"
    result = overlay_name(background, "Bob", destination=destination)

    assert result == destination
    assert destination.is_file()
    saved = Image.open(destination)
    assert saved.size == (400, 200)


def test_overlay_name_rejects_empty_name() -> None:
    background = _solid_background((400, 200), (240, 240, 240))

    with pytest.raises(RenderError):
        overlay_name(background, "")
    with pytest.raises(RenderError):
        overlay_name(background, "   ")


def test_overlay_name_avoids_face_region() -> None:
    background = _solid_background((400, 200), (200, 200, 200))
    face = (150, 60, 100, 100)
    result = overlay_name(background, "Carol", faces=[face], avoid_faces=True)

    assert result.size == (400, 200)
    assert _has_opaque_pixels(result)


def test_overlay_name_ignores_faces_when_disabled() -> None:
    background = _solid_background((400, 200), (200, 200, 200))
    face = (0, 0, 400, 200)
    result = overlay_name(background, "Dave", faces=[face], avoid_faces=False)

    assert result.size == (400, 200)
    assert _has_opaque_pixels(result)


def test_overlay_name_low_contrast_background_gets_treatment() -> None:
    background = _solid_background((400, 200), (128, 128, 128))
    result = overlay_name(background, "Eve")

    assert result.size == (400, 200)
    assert _has_opaque_pixels(result)


def test_overlay_name_explicit_colour_is_used() -> None:
    background = _solid_background((400, 200), (0, 0, 0))
    result = overlay_name(background, "Frank", color="#ff0000")

    assert result.size == (400, 200)
    assert _has_opaque_pixels(result)


def test_overlay_name_fits_to_requested_size() -> None:
    background = _solid_background((800, 600), (200, 100, 50))
    result = overlay_name(background, "Grace", size=(400, 200))

    assert result.size == (400, 200)


def test_overlay_name_supports_special_characters() -> None:
    background = _solid_background((600, 200), (100, 150, 200))
    result = overlay_name(background, "José García Ñoño")

    assert result.size == (600, 200)
    assert _has_opaque_pixels(result)


def test_overlay_name_supports_cjk_and_emoji() -> None:
    background = _solid_background((600, 200), (100, 150, 200))
    result = overlay_name(background, "中文 테スト 🎉")

    assert result.size == (600, 200)
    assert _has_opaque_pixels(result)


def test_default_font_family_is_lato() -> None:
    assert DEFAULT_FONT_FAMILY == "Lato"


def test_default_font_size_is_positive() -> None:
    assert DEFAULT_FONT_SIZE > 0


def _assert_matches_golden(
    rendered: Image.Image,
    golden_path: Path,
    tolerance: int = 0,
) -> None:
    expected = Image.open(golden_path).convert("RGBA")
    assert rendered.size == expected.size, f"size {rendered.size} != {expected.size}"
    difference = ImageChops.difference(rendered, expected)
    maximum = max(
        int(band[1]) for band in difference.getextrema() if isinstance(band, tuple)
    )
    assert maximum <= tolerance, (
        f"golden differs by {maximum} channel levels (tolerance {tolerance})"
    )


@pytest.mark.parametrize(
    "name,file",
    [
        ("Alice", "alice_light.png"),
        ("Bob", "bob_dark.png"),
        ("José García", "jose_special.png"),
        ("Ñoño 中文 🎉", "multilingual.png"),
        ("Midtone", "midtone.png"),
    ],
)
def test_golden_name_overlay_matches_reference(name: str, file: str) -> None:
    background = _solid_background((400, 200), (240, 240, 240))
    if file == "bob_dark.png":
        background = _solid_background((400, 200), (40, 40, 40))
    elif file == "jose_special.png" or file == "multilingual.png":
        background = _solid_background((600, 200), (100, 150, 200))
    elif file == "midtone.png":
        background = _solid_background((400, 200), (128, 128, 128))

    rendered = overlay_name(background, name)
    _assert_matches_golden(rendered, GOLDEN_DIR / file, tolerance=0)
