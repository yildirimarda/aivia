"""The renderer interface and the deterministic Pillow still renderer."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageChops

from aivia.compose import (
    PILLOW_STILL_RENDERER_VERSION,
    RENDERER_VERSION,
    Composition,
    FitMode,
    LayerPlacement,
    PillowRenderer,
    PillowStillRenderer,
    Renderer,
    RenderError,
    StillRenderer,
    placement_for_layer,
    render_still,
)
from aivia.compose import render as render_module

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "golden" / "compose"
GOLDEN_ENV = "AIVA_COMPARE_GOLDENS"


def _composition(**overrides: Any) -> Composition:
    values: dict[str, Any] = {
        "canvas": "4x4",
        "duration_seconds": 3.0,
        "fps": 30,
    }
    values.update(overrides)
    return Composition.model_validate(values)


def _solid(
    color: tuple[int, int, int, int],
    size: tuple[int, int] = (2, 2),
) -> Image.Image:
    return Image.new("RGBA", size, color)


def _quadrants() -> Image.Image:
    image = Image.new("RGBA", (4, 2), (0, 0, 0, 0))
    image.paste((255, 0, 0, 255), (0, 0, 2, 1))
    image.paste((0, 255, 0, 255), (2, 0, 4, 1))
    image.paste((0, 0, 255, 255), (0, 1, 2, 2))
    image.paste((255, 255, 0, 255), (2, 1, 4, 2))
    return image


def _pixel(image: Image.Image, x: int, y: int) -> tuple[int, int, int, int]:
    value = image.getpixel((x, y))
    assert isinstance(value, tuple)
    return value[0], value[1], value[2], value[3]


def _assert_color(
    image: Image.Image,
    x: int,
    y: int,
    expected: tuple[int, int, int, int],
    *,
    tolerance: int = 0,
) -> None:
    actual = _pixel(image, x, y)
    for channel, target in zip(actual, expected, strict=True):
        assert abs(channel - target) <= tolerance, (actual, expected)


def test_the_pillow_renderer_satisfies_the_renderer_interface() -> None:
    renderer = PillowStillRenderer()

    assert isinstance(renderer, Renderer)
    assert isinstance(renderer, StillRenderer)
    assert renderer.version == PILLOW_STILL_RENDERER_VERSION == RENDERER_VERSION
    assert PillowRenderer is PillowStillRenderer


def test_the_canvas_uses_the_requested_size_and_background_colour() -> None:
    composition = _composition(
        canvas={"width": 5, "height": 3, "background": "#336699"}
    )

    image = PillowStillRenderer().render_still(composition)

    assert image.size == (5, 3)
    assert image.mode == "RGBA"
    for y in range(3):
        for x in range(5):
            _assert_color(image, x, y, (51, 102, 153, 255))


@pytest.mark.parametrize(
    ("fit", "expected"),
    [
        (FitMode.STRETCH, ((255, 0, 0, 255), (0, 255, 0, 255))),
        (FitMode.COVER, ((255, 0, 0, 255), (0, 255, 0, 255))),
    ],
)
def test_image_fit_modes_fill_the_layer_box(
    fit: FitMode,
    expected: tuple[tuple[int, int, int, int], tuple[int, int, int, int]],
) -> None:
    composition = _composition(
        layers=[{"id": "cover", "type": "image", "source": "src.png", "fit": fit}]
    )

    image = PillowStillRenderer({"src.png": _quadrants()}).render_still(composition)

    _assert_color(image, 0, 0, expected[0])
    _assert_color(image, 3, 0, expected[1])
    _assert_color(image, 0, 3, (0, 0, 255, 255))
    _assert_color(image, 3, 3, (255, 255, 0, 255))


def test_contain_and_none_keep_the_source_aspect_ratio_and_centre_it() -> None:
    composition = _composition(
        layers=[
            {"id": "contain", "type": "image", "source": "src.png", "fit": "contain"}
        ]
    )
    renderer = PillowStillRenderer({"src.png": _solid((255, 0, 0, 255), (4, 2))})

    image = renderer.render_still(composition)

    _assert_color(image, 0, 0, (0, 0, 0, 255))
    _assert_color(image, 0, 1, (255, 0, 0, 255))
    _assert_color(image, 3, 2, (255, 0, 0, 255))
    _assert_color(image, 0, 3, (0, 0, 0, 255))

    composition = _composition(
        layers=[{"id": "none", "type": "image", "source": "src.png", "fit": "none"}]
    )
    image = renderer.render_still(composition)
    _assert_color(image, 0, 0, (0, 0, 0, 255))
    _assert_color(image, 0, 1, (255, 0, 0, 255))
    _assert_color(image, 0, 3, (0, 0, 0, 255))


def test_geometry_opacity_and_z_order_are_applied_to_image_layers() -> None:
    composition = _composition(
        layers=[
            {
                "id": "back",
                "type": "image",
                "source": "red.png",
                "fit": "stretch",
                "z": 0,
            },
            {
                "id": "front",
                "type": "image",
                "source": "white.png",
                "x": 1,
                "y": 1,
                "width": 2,
                "height": 2,
                "fit": "stretch",
                "opacity": 0.5,
                "z": 1,
            },
        ]
    )
    renderer = PillowStillRenderer(
        {
            "red.png": _solid((255, 0, 0, 255)),
            "white.png": _solid((255, 255, 255, 255)),
        }
    )

    image = renderer.render_still(composition)

    _assert_color(image, 0, 0, (255, 0, 0, 255))
    _assert_color(image, 1, 1, (255, 128, 128, 255), tolerance=1)
    _assert_color(image, 2, 2, (255, 128, 128, 255), tolerance=1)
    _assert_color(image, 3, 3, (255, 0, 0, 255))


def test_position_size_and_left_top_aliases_define_the_layer_box() -> None:
    composition = _composition(
        layers=[
            {
                "id": "aliased",
                "type": "image",
                "source": "red.png",
                "position": (1, 1),
                "size": (2, 2),
                "fit": "stretch",
            }
        ]
    )

    image = PillowStillRenderer({"red.png": _solid((255, 0, 0, 255))}).render_still(
        composition
    )

    assert placement_for_layer(composition.layers[0], image) == LayerPlacement(
        1, 1, 2, 2
    )
    _assert_color(image, 0, 0, (0, 0, 0, 255))
    _assert_color(image, 1, 1, (255, 0, 0, 255))
    _assert_color(image, 2, 2, (255, 0, 0, 255))
    _assert_color(image, 3, 3, (0, 0, 0, 255))


def test_a_layer_is_drawn_only_inside_its_time_window() -> None:
    composition = _composition(
        layers=[
            {
                "id": "timed",
                "type": "image",
                "source": "red.png",
                "start": 1,
                "end": 3,
            }
        ]
    )
    renderer = PillowStillRenderer({"red.png": _solid((255, 0, 0, 255))})

    _assert_color(renderer.render_still(composition, 0.5), 0, 0, (0, 0, 0, 255))
    _assert_color(renderer.render_still(composition, 1.0), 0, 0, (255, 0, 0, 255))
    _assert_color(renderer.render_still(composition, 2.5), 0, 0, (255, 0, 0, 255))
    _assert_color(renderer.render_still(composition, 3.0), 0, 0, (0, 0, 0, 255))


def test_position_and_opacity_keyframes_are_sampled_at_the_still_time() -> None:
    composition = _composition(
        duration_seconds=3.0,
        layers=[
            {
                "id": "moving",
                "type": "image",
                "source": "red.png",
                "width": 2,
                "height": 2,
                "fit": "stretch",
                "keyframes": {
                    "position": [
                        {"time": 0, "value": (0, 0)},
                        {"time": 2, "value": (2, 2)},
                    ],
                    "opacity": [
                        {"time": 0, "value": 0},
                        {"time": 2, "value": 1},
                    ],
                },
            }
        ],
    )
    renderer = PillowStillRenderer({"red.png": _solid((255, 0, 0, 255))})

    start = renderer.render_still(composition, 0.0)
    middle = renderer.render_still(composition, 1.0)
    end = renderer.render_still(composition, 2.0)

    _assert_color(start, 0, 0, (0, 0, 0, 255))
    _assert_color(middle, 0, 0, (0, 0, 0, 255))
    _assert_color(middle, 1, 1, (128, 0, 0, 255), tolerance=1)
    _assert_color(end, 2, 2, (255, 0, 0, 255))
    _assert_color(end, 0, 0, (0, 0, 0, 255))


def test_hidden_and_unsupported_layers_are_skipped() -> None:
    composition = _composition(
        layers=[
            {"id": "hidden", "type": "image", "source": "red.png", "visible": False},
            {"id": "badge", "type": "sticker", "source": "missing.png"},
        ]
    )

    image = PillowStillRenderer({"red.png": _solid((255, 0, 0, 255))}).render_still(
        composition
    )

    _assert_color(image, 0, 0, (0, 0, 0, 255))


def test_sources_can_be_bytes_a_callable_or_relative_to_a_root(tmp_path: Path) -> None:
    composition = _composition(
        layers=[{"id": "cover", "type": "image", "source": "cover.png"}]
    )
    source = tmp_path / "cover.png"
    _solid((0, 128, 0, 255)).save(source)

    from_root = PillowStillRenderer(source_root=tmp_path).render_still(composition)
    _assert_color(from_root, 0, 0, (0, 128, 0, 255))

    raw = source.read_bytes()
    from_mapping = PillowStillRenderer({"cover.png": raw}).render_still(composition)
    _assert_color(from_mapping, 0, 0, (0, 128, 0, 255))

    from_callable = PillowStillRenderer(
        lambda name: raw if name == "cover.png" else _solid((1, 2, 3, 255))
    ).render_still(composition)
    _assert_color(from_callable, 0, 0, (0, 128, 0, 255))


@pytest.mark.parametrize(
    ("payload", "time_seconds", "assets"),
    [
        ({"canvas": "4x4", "duration_seconds": 1}, -0.1, {}),
        ({"canvas": "4x4", "duration_seconds": 1}, 1.1, {}),
        (
            {
                "canvas": {"width": 4, "height": 4, "background": "not-a-colour"},
                "duration_seconds": 1,
            },
            0.0,
            {},
        ),
        (
            {
                "canvas": "4x4",
                "duration_seconds": 1,
                "layers": [{"id": "missing", "type": "image", "source": "gone.png"}],
            },
            0.0,
            {},
        ),
        (
            {
                "canvas": "4x4",
                "duration_seconds": 1,
                "layers": [
                    {
                        "id": "bad",
                        "type": "image",
                        "source": "red.png",
                        "width": 0,
                    }
                ],
            },
            0.0,
            {"red.png": _solid((255, 0, 0, 255))},
        ),
    ],
)
def test_invalid_renders_raise_a_render_error(
    payload: dict[str, Any],
    time_seconds: float,
    assets: dict[str, Image.Image],
) -> None:
    composition = Composition.model_validate(payload)

    with pytest.raises(RenderError):
        PillowStillRenderer(assets).render_still(composition, time_seconds)


def test_the_module_level_helper_renders_a_still() -> None:
    composition = _composition(
        layers=[{"id": "cover", "type": "image", "source": "red.png"}]
    )

    image = render_still(composition, assets={"red.png": _solid((255, 0, 0, 255))})

    _assert_color(image, 0, 0, (255, 0, 0, 255))


def test_the_render_module_imports_no_ai_or_network_code() -> None:
    source = Path(render_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)

    forbidden = (
        "aivia.llm",
        "aivia.providers",
        "aivia.media",
        "aivia.core.http",
        "aivia.core.media_probe",
        "httpx",
        "requests",
        "urllib",
        "socket",
        "http",
    )
    assert not [
        name
        for name in imported
        for prefix in forbidden
        if name == prefix or name.startswith(f"{prefix}.")
    ]

    script = "\n".join(
        [
            "import sys",
            "import aivia.compose.render",
            "bad = sorted(",
            "    name for name in sys.modules",
            "    if name == 'httpx'",
            "    or name == 'requests'",
            "    or name.startswith(('aivia.providers', 'aivia.llm', 'aivia.media'))",
            ")",
            "print(','.join(bad))",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


def _golden_cases() -> dict[str, tuple[Composition, dict[str, Image.Image]]]:
    canvas = _composition(
        id="golden_canvas",
        canvas={"width": 8, "height": 6, "background": "#336699"},
    )
    image_layers = _composition(
        id="golden_image_layers",
        canvas={"width": 8, "height": 6, "background": "#101820"},
        layers=[
            {
                "id": "background",
                "type": "image",
                "source": "quadrants.png",
                "fit": "cover",
                "z": 0,
            },
            {
                "id": "badge",
                "type": "image",
                "source": "badge.png",
                "x": 2,
                "y": 2,
                "width": 4,
                "height": 2,
                "fit": "stretch",
                "opacity": 0.5,
                "z": 1,
            },
        ],
    )
    assets = {
        "quadrants.png": _quadrants(),
        "badge.png": _solid((255, 255, 255, 255)),
    }
    return {"canvas": (canvas, {}), "image_layers": (image_layers, assets)}


@pytest.mark.golden
@pytest.mark.parametrize("case_id", ["canvas", "image_layers"])
def test_the_still_render_matches_the_golden_image(case_id: str) -> None:
    if os.environ.get(GOLDEN_ENV) != "1":
        pytest.skip(f"set {GOLDEN_ENV}=1 to compare golden renders")

    manifest = json.loads(
        (GOLDEN_DIR / "still_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["renderer"] == PILLOW_STILL_RENDERER_VERSION

    composition, assets = _golden_cases()[case_id]
    entry = manifest["cases"][case_id]
    rendered = PillowStillRenderer(assets).render_still(composition)
    expected = Image.open(GOLDEN_DIR / entry["file"]).convert("RGBA")

    assert rendered.size == expected.size
    assert ImageChops.difference(rendered, expected).getbbox() is None
