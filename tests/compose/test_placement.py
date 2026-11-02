"""Smart text placement: calm regions, face avoidance and renderer wiring."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from PIL import Image, ImageChops

from aivia.compose import (
    Composition,
    FaceBox,
    PillowStillRenderer,
    RenderError,
    TextLayer,
    TextPlacement,
    TextPlacementMode,
    faces_from_analysis,
    faces_with_vision,
    find_low_detail_regions,
    find_text_placement,
    scale_faces_to_canvas,
)

CANVAS = (200, 100)
WHITE = (255, 255, 255, 255)
BLACK = (0, 0, 0, 255)


def _busy_left() -> Image.Image:
    image = Image.new("RGB", CANVAS, (0, 0, 0))
    pixels = image.load()
    assert pixels is not None
    for y in range(CANVAS[1]):
        for x in range(CANVAS[0] // 2):
            value = (x * 37 + y * 53) % 255
            pixels[x, y] = (value, (value * 3) % 255, (value * 7) % 255)
    return image


def _blank() -> Image.Image:
    return Image.new("RGB", CANVAS, (255, 255, 255))


def _composition(
    layers: list[dict[str, Any]],
    *,
    background: str = "#000000",
) -> Composition:
    return Composition.model_validate(
        {
            "canvas": {
                "width": CANVAS[0],
                "height": CANVAS[1],
                "background": background,
            },
            "duration_seconds": 1.0,
            "layers": layers,
        }
    )


def _text_bbox(
    layers: list[dict[str, Any]],
    *,
    assets: dict[str, Image.Image] | None = None,
    faces: Any = None,
) -> tuple[int, int, int, int] | None:
    renderer = PillowStillRenderer(assets, faces=faces)
    with_text = renderer.render_still(_composition(layers))
    without_text = renderer.render_still(
        _composition([layer for layer in layers if layer.get("type") != "text"])
    )
    return ImageChops.difference(
        with_text.convert("RGB"), without_text.convert("RGB")
    ).getbbox()


def test_the_calmest_half_wins_over_a_busy_half() -> None:
    placement = find_text_placement(_busy_left(), (40, 20))

    assert placement.width == 40
    assert placement.height == 20
    assert placement.x >= CANVAS[0] // 2


def test_regions_are_ordered_from_calmest_to_busiest() -> None:
    regions = find_low_detail_regions(_busy_left(), (40, 20), limit=None)

    assert regions
    assert regions[0].detail < regions[-1].detail
    assert regions[0].x >= CANVAS[0] // 2


def test_placement_is_deterministic() -> None:
    first = find_text_placement(_busy_left(), (40, 20))
    second = find_text_placement(_busy_left(), (40, 20))

    assert first == second


def test_a_candidate_always_stays_inside_the_canvas_margin() -> None:
    placement = find_text_placement(_busy_left(), (40, 20), margin=10)

    assert placement.x >= 10
    assert placement.y >= 10
    assert placement.x + placement.width <= CANVAS[0] - 10
    assert placement.y + placement.height <= CANVAS[1] - 10


def test_faces_are_avoided_when_the_option_is_on() -> None:
    face = FaceBox(x=0, y=0, width=100, height=100)

    avoided = find_text_placement(_blank(), (20, 20), faces=[face])
    allowed = find_text_placement(_blank(), (20, 20), faces=[face], avoid_faces=False)

    assert avoided.x >= 100
    assert allowed.x < 100


def test_face_padding_keeps_the_text_box_clear_of_a_face() -> None:
    face = FaceBox(x=0, y=0, width=80, height=80)

    placement = find_text_placement(
        _blank(),
        (20, 20),
        faces=[face],
        face_padding=30,
    )

    assert placement.x >= 110


def test_a_box_larger_than_the_canvas_is_clamped() -> None:
    placement = find_text_placement(_blank(), (500, 500))

    assert placement == type(placement)(0, 0, CANVAS[0], CANVAS[1])


def test_invalid_boxes_and_faces_are_rejected() -> None:
    with pytest.raises(RenderError):
        find_text_placement(_blank(), (0, 20))
    with pytest.raises(RenderError):
        find_text_placement(_blank(), (20, 20), face_padding=-1)


def test_faces_from_analysis_accepts_mappings_and_objects() -> None:
    assert faces_from_analysis({"faces": [{"x": 1, "y": 2, "w": 3, "h": 4}]})
    assert faces_from_analysis({"face_regions": [(1, 2, 3, 4)]})
    assert faces_from_analysis([FaceBox(x=1, y=2, width=3, height=4)])
    assert faces_from_analysis(None) == []


def test_the_vision_check_can_use_an_offline_analyzer() -> None:
    async def analyzer(source: Any) -> dict[str, Any]:
        return {"faces": [{"x": 10, "y": 20, "w": 30, "h": 40}]}

    faces = asyncio.run(faces_with_vision(_blank(), analyzer=analyzer))

    assert len(faces) == 1
    assert faces[0].box == (10.0, 20.0, 30.0, 40.0)


def test_text_layer_accepts_smart_placement_shorthand() -> None:
    layer = TextLayer.model_validate(
        {
            "text": "Hello",
            "smart_placement": True,
            "avoid_faces": False,
            "face_regions": [{"x": 1, "y": 2, "w": 3, "h": 4}],
        }
    )

    assert layer.placement is not None
    assert layer.placement.mode is TextPlacementMode.SMART
    assert layer.smart_placement is True
    assert layer.avoid_faces is False
    assert layer.placement.faces[0].box == (1.0, 2.0, 3.0, 4.0)


def test_text_layer_accepts_a_placement_string_or_block() -> None:
    shorthand = TextLayer.model_validate({"text": "Hi", "placement": "auto"})
    block = TextLayer.model_validate(
        {"text": "Hi", "placement": {"mode": "smart", "padding": 4}}
    )

    assert shorthand.placement is not None
    assert shorthand.placement.mode is TextPlacementMode.AUTO
    assert block.placement == TextPlacement(mode=TextPlacementMode.SMART, padding=4)


def test_an_unknown_placement_mode_is_rejected() -> None:
    with pytest.raises(ValueError):
        TextLayer.model_validate({"text": "Hi", "placement": "diagonal"})


def test_the_renderer_places_text_in_the_low_detail_region() -> None:
    bbox = _text_bbox(
        [
            {"type": "image", "source": "bg", "fit": "stretch"},
            {
                "type": "text",
                "text": "Hi",
                "font_size": 20,
                "placement": "smart",
            },
        ],
        assets={"bg": _busy_left()},
    )

    assert bbox is not None
    assert bbox[0] >= CANVAS[0] // 2


def test_the_renderer_avoids_faces_stored_on_the_layer() -> None:
    bbox = _text_bbox(
        [
            {
                "type": "text",
                "text": "X",
                "font_size": 20,
                "placement": {
                    "mode": "smart",
                    "faces": [{"x": 0, "y": 0, "w": 100, "h": 100}],
                },
            }
        ]
    )

    assert bbox is not None
    assert bbox[0] >= 100


def test_the_renderer_accepts_faces_from_a_resolver() -> None:
    bbox = _text_bbox(
        [
            {
                "type": "text",
                "id": "caption",
                "text": "X",
                "font_size": 20,
                "placement": "smart",
            }
        ],
        faces={"caption": [FaceBox(x=0, y=0, width=100, height=100)]},
    )

    assert bbox is not None
    assert bbox[0] >= 100


def test_manual_placement_is_unchanged() -> None:
    bbox = _text_bbox(
        [
            {
                "type": "text",
                "text": "X",
                "font_size": 20,
                "x": 10,
                "y": 10,
                "width": 40,
                "height": 30,
            }
        ]
    )

    assert bbox is not None
    assert 10 <= bbox[0] <= 20
    assert 10 <= bbox[1] <= 20


def test_face_rectangles_are_scaled_to_canvas_when_the_background_is_fitted() -> None:
    bbox = _text_bbox(
        [
            {"type": "image", "id": "bg", "source": "bg", "fit": "cover"},
            {
                "type": "text",
                "id": "caption",
                "text": "X",
                "font_size": 20,
                "color": "#000000",
                "placement": "smart",
            },
        ],
        assets={"bg": Image.new("RGB", (400, 200), (255, 255, 255))},
        faces={"caption": [FaceBox(x=0, y=0, width=200, height=200)]},
    )

    assert bbox is not None
    assert bbox[0] >= 100


def test_face_rectangles_follow_background_position_keyframes() -> None:
    bbox = _text_bbox(
        [
            {
                "type": "image",
                "id": "bg",
                "source": "bg",
                "fit": "cover",
                "keyframes": {"position": [(0, (100, 0))]},
            },
            {
                "type": "text",
                "id": "caption",
                "text": "X",
                "font_size": 20,
                "color": "#000000",
                "placement": "smart",
            },
        ],
        assets={"bg": Image.new("RGB", (400, 200), (255, 255, 255))},
        faces={"caption": [FaceBox(x=0, y=0, width=200, height=200)]},
    )

    assert bbox is not None
    assert bbox[0] < 100


def test_face_rectangles_follow_background_rotation_keyframes() -> None:
    layers: list[dict[str, Any]] = [
        {
            "type": "image",
            "id": "bg",
            "source": "bg",
            "fit": "cover",
        },
        {
            "type": "text",
            "id": "caption",
            "text": "X",
            "font_size": 20,
            "color": "#000000",
            "placement": "smart",
        },
    ]
    assets = {"bg": Image.new("RGB", (200, 100), (255, 255, 255))}
    faces = {"caption": [FaceBox(x=0, y=0, width=100, height=100)]}

    without_keyframe = _text_bbox(layers, assets=assets, faces=faces)
    with_keyframe = _text_bbox(
        [{**layers[0], "keyframes": {"rotation": [(0, 180)]}}, layers[1]],
        assets=assets,
        faces=faces,
    )

    assert without_keyframe is not None
    assert with_keyframe is not None
    assert without_keyframe[0] >= 100
    assert with_keyframe[0] < 100


def test_face_rectangles_follow_background_motion_pan() -> None:
    layers: list[dict[str, Any]] = [
        {
            "type": "image",
            "id": "bg",
            "source": "bg",
            "fit": "cover",
        },
        {
            "type": "text",
            "id": "caption",
            "text": "X",
            "font_size": 20,
            "color": "#000000",
            "placement": "smart",
        },
    ]
    assets = {"bg": Image.new("RGB", (400, 200), (255, 255, 255))}
    faces = {"caption": [FaceBox(x=0, y=0, width=200, height=200)]}

    without_pan = _text_bbox(layers, assets=assets, faces=faces)
    with_pan = _text_bbox(
        [
            {
                **layers[0],
                "motion": {
                    "type": "ken_burns",
                    "start_pan": [100, 0],
                    "end_pan": [200, 0],
                },
            },
            layers[1],
        ],
        assets=assets,
        faces=faces,
    )

    assert without_pan is not None
    assert with_pan is not None
    assert without_pan[0] >= 100
    assert with_pan[0] < 100


def test_face_rectangles_follow_background_scale_keyframes() -> None:
    layers: list[dict[str, Any]] = [
        {
            "type": "image",
            "id": "bg",
            "source": "bg",
            "fit": "cover",
        },
        {
            "type": "text",
            "id": "caption",
            "text": "XXXX",
            "font_size": 20,
            "color": "#000000",
            "placement": "smart",
        },
    ]
    assets = {"bg": Image.new("RGB", (400, 200), (255, 255, 255))}
    faces = {"caption": [FaceBox(x=260, y=0, width=140, height=200)]}

    without_keyframe = _text_bbox(layers, assets=assets, faces=faces)
    with_keyframe = _text_bbox(
        [{**layers[0], "keyframes": {"scale": [(0, 0.5)]}}, layers[1]],
        assets=assets,
        faces=faces,
    )

    assert without_keyframe is not None
    assert with_keyframe is not None
    assert with_keyframe[0] < without_keyframe[0]


def test_face_scaling_follows_stretch_cover_and_contain_geometry() -> None:
    face = FaceBox(x=10, y=20, width=30, height=40, confidence=0.9)

    stretched = scale_faces_to_canvas([face], (100, 100), (200, 50), "stretch")
    covered = scale_faces_to_canvas([face], (100, 100), (200, 50), "cover")
    contained = scale_faces_to_canvas([face], (100, 100), (200, 50), "contain")

    assert stretched[0].box == (20.0, 10.0, 60.0, 20.0)
    assert covered[0].box == (20.0, 0.0, 60.0, 45.0)
    assert contained[0].box == (80.0, 10.0, 15.0, 20.0)
    assert covered[0].confidence == 0.9


def test_face_scaling_applies_position_and_scale_keyframe_geometry() -> None:
    face = FaceBox(x=10, y=20, width=30, height=40)

    positioned = scale_faces_to_canvas(
        [face],
        (100, 100),
        (200, 50),
        "cover",
        offset=(5, -10),
    )
    scaled = scale_faces_to_canvas(
        [face],
        (100, 100),
        (200, 50),
        "cover",
        scale=2.0,
    )

    assert positioned[0].box == pytest.approx((25.0, 0.0, 60.0, 35.0))
    assert scaled[0].box == pytest.approx((0.0, 0.0, 60.0, 50.0))


def test_face_scaling_applies_motion_pan_geometry() -> None:
    face = FaceBox(x=10, y=20, width=30, height=40)
    centered = FaceBox(x=40, y=40, width=10, height=10)

    panned = scale_faces_to_canvas(
        [face],
        (100, 100),
        (200, 50),
        "cover",
        pan=(10, 0),
    )
    scaled_pan = scale_faces_to_canvas(
        [centered],
        (100, 100),
        (100, 100),
        "stretch",
        scale=2.0,
        pan=(10, 0),
    )

    assert panned[0].box == pytest.approx((10.0, 0.0, 60.0, 45.0))
    assert scaled_pan[0].box == pytest.approx((10.0, 30.0, 20.0, 20.0))


def test_face_scaling_applies_rotation_keyframe_geometry() -> None:
    face = FaceBox(x=40, y=0, width=20, height=10)

    quarter_turn = scale_faces_to_canvas(
        [face],
        (100, 100),
        (100, 100),
        fit="stretch",
        rotation=90,
    )
    half_turn = scale_faces_to_canvas(
        [face],
        (100, 100),
        (100, 100),
        fit="stretch",
        rotation=180,
    )
    shifted = scale_faces_to_canvas(
        [face],
        (100, 100),
        (100, 100),
        fit="stretch",
        offset=(10, 0),
        rotation=90,
    )

    assert quarter_turn[0].box == pytest.approx((0.0, 40.0, 10.0, 20.0))
    assert half_turn[0].box == pytest.approx((40.0, 90.0, 20.0, 10.0))
    assert shifted[0].box == pytest.approx((10.0, 40.0, 10.0, 20.0))


def test_face_scaling_accepts_an_explicit_placement() -> None:
    face = FaceBox(x=10, y=20, width=30, height=40)

    scaled = scale_faces_to_canvas(
        [face],
        (100, 100),
        fit="stretch",
        placement=(10, 20, 200, 50),
    )

    assert scaled[0].box == (30.0, 30.0, 60.0, 20.0)


def test_face_scaling_rejects_bad_geometry() -> None:
    face = FaceBox(x=0, y=0, width=1, height=1)
    with pytest.raises(RenderError):
        scale_faces_to_canvas([face], (0, 100), (200, 50), "cover")
    with pytest.raises(RenderError):
        scale_faces_to_canvas([face], (100, 100), (200, 50), "diagonal")
