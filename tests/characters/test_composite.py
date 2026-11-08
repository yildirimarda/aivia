"""Place pose cutouts into compositions with anchor, scale, flip, shadow,
safe area awareness and idle animation.

A cutout is composited as an overlay (sticker) layer pinned to a canvas anchor,
scaled, optionally mirrored, optionally given a drop shadow, kept inside the
platform safe area and given a subtle idle animation. Rendering is
deterministic and offline: no AI call and no network request happen here.
"""

from __future__ import annotations

from typing import Any

import pytest
from PIL import Image
from pydantic import ValidationError as PydanticValidationError

from aivia.characters.composite import PlaceCutoutError, place_cutout
from aivia.characters.pose import Pose
from aivia.compose import Composition, PillowStillRenderer
from aivia.compose.motion import MotionType
from aivia.compose.schema import (
    Anchor,
    FitMode,
    Flip,
    LayerType,
    StickerLayer,
)
from aivia.core.platforms import SafeArea


def _cutout(
    size: tuple[int, int] = (8, 8),
    color: tuple[int, int, int, int] = (255, 0, 0, 255),
) -> Image.Image:
    return Image.new("RGBA", size, color)


def _quadrants() -> Image.Image:
    image = Image.new("RGBA", (4, 2), (0, 0, 0, 0))
    image.paste((255, 0, 0, 255), (0, 0, 2, 1))
    image.paste((0, 255, 0, 255), (2, 0, 4, 1))
    return image


def _composition(
    *,
    size: tuple[int, int] = (100, 100),
    background: str = "#ffffff",
    duration_seconds: float = 3.0,
    fps: int = 30,
    safe_area: SafeArea | None = None,
    layers: list[Any] | None = None,
) -> Composition:
    values: dict[str, Any] = {
        "canvas": {"width": size[0], "height": size[1], "background": background},
        "duration_seconds": duration_seconds,
        "fps": fps,
    }
    if safe_area is not None:
        values["safe_area"] = safe_area
    if layers is not None:
        values["layers"] = layers
    return Composition.model_validate(values)


def _render(
    composition: Composition,
    time_seconds: float = 0.0,
    assets: dict[str, Image.Image] | None = None,
) -> Image.Image:
    return PillowStillRenderer(assets or {}).render_still(composition, time_seconds)


def _pixel(image: Image.Image, x: int, y: int) -> tuple[int, int, int, int]:
    value = image.getpixel((x, y))
    assert isinstance(value, tuple)
    return value[0], value[1], value[2], value[3]


def test_place_cutout_builds_a_sticker_layer() -> None:
    layer = place_cutout(source="idle.png", duration_seconds=3.0)

    assert layer.type is LayerType.STICKER
    assert layer.source == "idle.png"
    assert layer.anchor is Anchor.CENTER
    assert layer.scale == 1.0
    assert layer.flip is Flip.NONE
    assert layer.shadow is False
    assert layer.duration == 3.0
    assert layer.end_seconds == 3.0
    assert layer.motion is not None


def test_place_cutout_uses_a_pose_cutout_image() -> None:
    pose = Pose(name="idle", image="idle.png", anchors={"hand": (0.5, 0.8)})
    layer = place_cutout(source=pose.image, duration_seconds=1.0)

    assert layer.source == "idle.png"
    assert layer.type is LayerType.STICKER


def test_scale_must_be_positive() -> None:
    with pytest.raises(PlaceCutoutError):
        place_cutout(source="idle.png", duration_seconds=1.0, scale=0.0)
    with pytest.raises(PlaceCutoutError):
        place_cutout(source="idle.png", duration_seconds=1.0, scale=-1.0)


def test_duration_must_be_positive() -> None:
    with pytest.raises(ValueError):
        place_cutout(source="idle.png", duration_seconds=0.0)


def test_flip_enum_parses_aliases_and_rejects_unknown() -> None:
    assert Flip("horizontal") is Flip.HORIZONTAL
    assert Flip("vertical") is Flip.VERTICAL
    assert Flip("both") is Flip.BOTH
    assert Flip("x") is Flip.HORIZONTAL
    assert Flip("y") is Flip.VERTICAL
    with pytest.raises(ValueError):
        Flip("sideways")


def test_flip_is_stored_on_the_layer() -> None:
    layer = place_cutout(
        source="cut.png",
        duration_seconds=1.0,
        anchor=Anchor.TOP_LEFT,
        flip=Flip.HORIZONTAL,
    )
    assert layer.flip is Flip.HORIZONTAL


def test_shadow_fields_are_stored() -> None:
    layer = place_cutout(
        source="cut.png",
        duration_seconds=1.0,
        shadow=True,
        shadow_color="#ffffff",
        shadow_offset=(2.0, 3.0),
        shadow_blur=1.0,
        shadow_opacity=0.8,
    )
    assert layer.shadow is True
    assert layer.shadow_color == "#ffffff"
    assert layer.shadow_offset == (2.0, 3.0)
    assert layer.shadow_blur == 1.0
    assert layer.shadow_opacity == 0.8


def test_idle_animation_is_a_parallax_drift() -> None:
    layer = place_cutout(
        source="cut.png", duration_seconds=3.0, idle=True, idle_drift=(4.0, 0.0)
    )
    assert layer.motion is not None
    assert layer.motion.type is MotionType.PARALLAX
    assert layer.motion.drift == (4.0, 0.0)


def test_idle_can_be_disabled() -> None:
    layer = place_cutout(source="cut.png", duration_seconds=1.0, idle=False)
    assert layer.motion is None


def test_scale_changes_the_cutout_size() -> None:
    layer = place_cutout(
        source="cut.png", duration_seconds=1.0, anchor=Anchor.TOP_LEFT, scale=2.0
    )
    composition = _composition(layers=[layer])
    assets = {"cut.png": _cutout((8, 8), (255, 0, 0, 255))}
    image = _render(composition, 0.0, assets)

    assert _pixel(image, 0, 0) == (255, 0, 0, 255)
    assert _pixel(image, 15, 15) == (255, 0, 0, 255)
    assert _pixel(image, 20, 20) == (255, 255, 255, 255)


def test_flip_horizontal_mirrors_the_cutout() -> None:
    layer = place_cutout(
        source="cut.png",
        duration_seconds=1.0,
        anchor=Anchor.CENTER,
        flip=Flip.HORIZONTAL,
    )
    composition = _composition(layers=[layer])
    assets = {"cut.png": _quadrants()}

    image = _render(composition, 0.0, assets)
    # Canvas is 100x100, the 4x2 cutout is centred at x=48, y=49. Source
    # column 3 (green) is mirrored onto canvas column 48.
    assert _pixel(image, 48, 49) == (0, 255, 0, 255)
    # Source column 0 (red) is mirrored onto canvas column 51.
    assert _pixel(image, 51, 49) == (255, 0, 0, 255)


def test_flip_vertical_mirrors_the_cutout() -> None:
    image_source = Image.new("RGBA", (2, 4), (0, 0, 0, 0))
    image_source.paste((255, 0, 0, 255), (0, 0, 2, 2))  # top half red
    image_source.paste((0, 255, 0, 255), (0, 2, 2, 4))  # bottom half green

    layer = place_cutout(
        source="cut.png",
        duration_seconds=1.0,
        anchor=Anchor.CENTER,
        flip=Flip.VERTICAL,
    )
    composition = _composition(layers=[layer])
    assets = {"cut.png": image_source}
    rendered = _render(composition, 0.0, assets)

    # Canvas is 100x100, cutout 2x4 centred at x=49, y=48.
    # Top source row (red) is mirrored onto canvas row 51.
    assert _pixel(rendered, 49, 51) == (255, 0, 0, 255)
    # Bottom source row (green) is mirrored onto canvas row 48.
    assert _pixel(rendered, 49, 48) == (0, 255, 0, 255)


def test_shadow_changes_the_rendered_frame() -> None:
    assets = {"cut.png": _cutout((4, 4), (255, 0, 0, 255))}
    base = place_cutout(source="cut.png", duration_seconds=1.0, anchor=Anchor.TOP_LEFT)
    shadowed = place_cutout(
        source="cut.png",
        duration_seconds=1.0,
        anchor=Anchor.TOP_LEFT,
        shadow=True,
        shadow_offset=(2.0, 2.0),
        shadow_blur=0.0,
        shadow_opacity=0.8,
    )
    base_image = _render(_composition(layers=[base]), 0.0, assets)
    shadow_image = _render(_composition(layers=[shadowed]), 0.0, assets)

    assert shadow_image.tobytes() != base_image.tobytes()


def test_safe_area_keeps_cutout_inside_the_safe_box() -> None:
    safe_area = SafeArea(top=0.2, right=0.2, bottom=0.2, left=0.2)
    layer = place_cutout(
        source="cut.png", duration_seconds=1.0, anchor=Anchor.BOTTOM_RIGHT
    )
    composition = _composition(safe_area=safe_area, layers=[layer])
    assets = {"cut.png": _cutout((8, 8), (255, 0, 0, 255))}
    image = _render(composition, 0.0, assets)

    # The cutout is inset by the 20 px safe area margins, so its bottom right
    # corner sits at canvas (80, 80) and the margin stays the background.
    assert _pixel(image, 95, 95) == (255, 255, 255, 255)
    assert _pixel(image, 75, 75) == (255, 0, 0, 255)


def test_without_safe_area_the_cutout_reaches_the_canvas_edge() -> None:
    layer = place_cutout(
        source="cut.png", duration_seconds=1.0, anchor=Anchor.BOTTOM_RIGHT
    )
    composition = _composition(layers=[layer])
    assets = {"cut.png": _cutout((8, 8), (255, 0, 0, 255))}
    image = _render(composition, 0.0, assets)

    # With no safe area the cutout is pinned to the canvas bottom right corner.
    assert _pixel(image, 95, 95) == (255, 0, 0, 255)


def test_idle_animation_drifts_the_cutout_over_time() -> None:
    layer = place_cutout(
        source="cut.png",
        duration_seconds=3.0,
        anchor=Anchor.CENTER,
        idle=True,
        idle_drift=(6.0, 0.0),
    )
    composition = _composition(layers=[layer])
    assets = {"cut.png": _cutout((4, 4), (255, 0, 0, 255))}
    renderer = PillowStillRenderer(assets)

    start = renderer.render_still(composition, 0.0)
    end = renderer.render_still(composition, 1.5)

    def red_columns(image: Image.Image) -> list[int]:
        return [x for x in range(100) if _pixel(image, x, 50)[:3] == (255, 0, 0)]

    start_cols = red_columns(start)
    end_cols = red_columns(end)
    assert start_cols and end_cols
    assert max(end_cols) > max(start_cols)


def test_placement_defaults_use_fit_contain() -> None:
    layer = place_cutout(source="cut.png", duration_seconds=1.0)
    assert layer.fit is FitMode.CONTAIN
    assert layer.margin == 0.0


def test_invalid_flip_on_the_layer_is_rejected() -> None:
    with pytest.raises(PydanticValidationError):
        StickerLayer.model_validate(
            {
                "type": LayerType.STICKER,
                "source": "cut.png",
                "flip": "sideways",
            }
        )
