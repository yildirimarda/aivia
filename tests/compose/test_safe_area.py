"""Per platform safe area margins for text and sticker layers."""

from __future__ import annotations

import warnings
from typing import Any

import pytest
from PIL import Image, ImageChops

from aivia.compose import (
    Composition,
    PillowStillRenderer,
    SafeAreaWarning,
    build_ffmpeg_filtergraph,
    effective_safe_area,
    safe_area_box,
    safe_area_violations,
    validate_composition,
)
from aivia.core.platforms import SafeArea

RED = (255, 0, 0, 255)
BLACK = (0, 0, 0, 255)


def _composition(
    layers: list[dict[str, Any]],
    *,
    width: int = 200,
    height: int = 200,
    platform: str = "",
    safe_area: SafeArea | None = None,
    background: str = "#000000",
) -> Composition:
    values: dict[str, Any] = {
        "canvas": {"width": width, "height": height, "background": background},
        "duration_seconds": 1.0,
        "platform": platform,
        "layers": layers,
    }
    if safe_area is not None:
        values["safe_area"] = safe_area
    return Composition.model_validate(values)


def _alpha_box(image: Image.Image) -> tuple[int, int, int, int]:
    box = image.getchannel("A").getbbox()
    assert box is not None
    return box


def _sticker_box(composition: Composition) -> tuple[int, int, int, int]:
    renderer = PillowStillRenderer({"sticker.png": Image.new("RGBA", (20, 10), RED)})
    with_sticker = renderer.render_still(composition)
    without = renderer.render_still(
        _composition(
            [],
            width=composition.width,
            height=composition.height,
            background=composition.canvas.background,
        )
    )
    box = ImageChops.difference(
        with_sticker.convert("RGB"), without.convert("RGB")
    ).getbbox()
    assert box is not None
    return box


def test_an_explicit_safe_area_wins_over_the_platform() -> None:
    composition = _composition(
        [],
        platform="tiktok",
        safe_area=SafeArea(top=0.3, bottom=0.1, left=0.2, right=0.05),
    )

    area = effective_safe_area(composition)

    assert area == SafeArea(top=0.3, bottom=0.1, left=0.2, right=0.05)
    assert safe_area_box(composition) == (40, 60, 150, 120)


def test_the_platform_supplies_the_safe_area_when_none_is_given() -> None:
    composition = _composition([], platform="tiktok")

    assert effective_safe_area(composition) == SafeArea(
        top=0.1, bottom=0.25, left=0.06, right=0.06
    )


def test_text_defaults_stay_inside_the_platform_safe_area() -> None:
    composition = _composition(
        [{"id": "title", "type": "text", "text": "Safe", "font_size": 24}],
        width=300,
        height=400,
        platform="tiktok",
        background="#00000000",
    )

    box = _alpha_box(PillowStillRenderer().render_still(composition))

    assert box[0] >= round(0.06 * 300)
    assert box[1] >= round(0.10 * 400)
    assert box[2] <= 300 - round(0.06 * 300)
    assert box[3] <= 400 - round(0.25 * 400)


def test_smart_text_placement_stays_inside_the_safe_area() -> None:
    background = Image.new("RGB", (200, 200), (255, 255, 255))
    pixels = background.load()
    assert pixels is not None
    for y in range(100):
        for x in range(100):
            value = (x * 37 + y * 53) % 255
            pixels[x, y] = (value, (value * 3) % 255, (value * 7) % 255)

    composition = _composition(
        [
            {"type": "image", "source": "bg", "fit": "stretch"},
            {
                "id": "caption",
                "type": "text",
                "text": "Calm",
                "font_size": 18,
                "color": "#000000",
                "placement": "smart",
            },
        ],
        platform="tiktok",
    )

    image = PillowStillRenderer({"bg": background}).render_still(composition)
    without_text = PillowStillRenderer({"bg": background}).render_still(
        _composition(
            [{"type": "image", "source": "bg", "fit": "stretch"}],
            platform="tiktok",
        )
    )
    box = ImageChops.difference(
        image.convert("RGB"), without_text.convert("RGB")
    ).getbbox()
    assert box is not None

    assert box[0] >= round(0.06 * 200)
    assert box[1] >= round(0.10 * 200)
    assert box[2] <= 200 - round(0.06 * 200)
    assert box[3] <= 200 - round(0.25 * 200)


def test_a_sticker_anchor_is_inset_by_the_platform_safe_area() -> None:
    composition = _composition(
        [
            {
                "id": "badge",
                "type": "sticker",
                "source": "sticker.png",
                "anchor": "bottom_right",
                "margin": (4, 6),
                "width": 20,
                "height": 10,
            }
        ],
        platform="tiktok",
    )

    box = _sticker_box(composition)

    assert box[0] == 200 - round(0.06 * 200) - 20 - 4
    assert box[1] == 200 - round(0.25 * 200) - 10 - 6
    assert box[2] <= 200 - round(0.06 * 200)
    assert box[3] <= 200 - round(0.25 * 200)


def test_an_anchored_sticker_has_no_safe_area_violation() -> None:
    composition = _composition(
        [
            {
                "id": "badge",
                "type": "sticker",
                "source": "sticker.png",
                "anchor": "bottom_right",
                "width": 20,
                "height": 10,
            }
        ],
        platform="tiktok",
    )

    assert safe_area_violations(composition) == []


def test_manual_text_outside_the_safe_area_warns() -> None:
    composition = _composition(
        [
            {
                "id": "title",
                "type": "text",
                "text": "Edge",
                "x": 0,
                "y": 0,
                "width": 80,
                "height": 40,
            }
        ],
        platform="tiktok",
    )

    violations = safe_area_violations(composition)

    assert [violation.code for violation in violations] == ["safe_area"]
    assert violations[0].path == "layers[0]"
    assert violations[0].layer_id == "title"
    assert "outside the safe area" in violations[0].message
    with pytest.warns(SafeAreaWarning):
        validate_composition(composition)


def test_manual_sticker_coordinates_outside_the_safe_area_warn() -> None:
    composition = _composition(
        [
            {
                "id": "badge",
                "type": "sticker",
                "source": "sticker.png",
                "x": 0,
                "y": 0,
                "width": 20,
                "height": 10,
            }
        ],
        platform="tiktok",
    )

    violations = safe_area_violations(composition)

    assert [violation.code for violation in violations] == ["safe_area"]
    assert violations[0].layer_id == "badge"


def test_manual_coordinates_inside_the_safe_area_do_not_warn() -> None:
    composition = _composition(
        [
            {
                "id": "title",
                "type": "text",
                "text": "Inside",
                "x": 20,
                "y": 45,
                "width": 80,
                "height": 40,
            }
        ],
        platform="tiktok",
    )

    assert safe_area_violations(composition) == []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_composition(composition)

    assert not any(isinstance(item.message, SafeAreaWarning) for item in caught)


def test_a_composition_without_a_safe_area_never_warns() -> None:
    composition = _composition(
        [
            {
                "id": "title",
                "type": "text",
                "text": "Free",
                "x": 0,
                "y": 0,
                "width": 20,
                "height": 10,
            }
        ]
    )

    assert safe_area_violations(composition) == []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_composition(composition)

    assert not caught


def test_ffmpeg_sticker_anchors_use_the_platform_safe_area() -> None:
    composition = _composition(
        [
            {
                "id": "badge",
                "type": "sticker",
                "source": "sticker.png",
                "anchor": "bottom_right",
                "width": 10,
                "height": 10,
            }
        ],
        width=100,
        height=100,
        platform="tiktok",
    )

    graph = build_ffmpeg_filtergraph(composition)

    assert "overlay=x=W-w-6:y=H-h-25" in graph


def test_ffmpeg_text_defaults_use_the_platform_safe_area() -> None:
    composition = _composition(
        [{"id": "title", "type": "text", "text": "Hi", "font_size": 20}],
        width=100,
        height=100,
        platform="tiktok",
    )

    graph = build_ffmpeg_filtergraph(composition)

    assert ":x=6:y=10" in graph
