"""The built-in image post and carousel formats use aspect presets per slide."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from PIL import Image

from aivia.compose import ImageLayer, TextLayer
from aivia.core.config.formats import OutputHandler, OutputType
from aivia.core.errors import ConfigError
from aivia.core.models.script import Beat, ImagePrompt, Script
from aivia.core.stages import StageContext
from aivia.formats import (
    ASPECT_PRESETS,
    DEFAULT_ASPECT_PRESET,
    CarouselRenderer,
    CarouselStage,
    ImagePost,
    ImagePostRenderer,
    ImagePostSlide,
    ImagePostStage,
    carousel_compositions,
    image_post_composition,
    image_post_slides,
    load_packaged_formats,
    render_image_post,
    resolve_aspect_preset,
    save_carousel,
    save_image_post,
)


def _script(count: int = 3) -> Script:
    return Script(
        title="Three ideas worth sharing",
        beats=[
            Beat(
                index=index,
                text=f"Slide {index + 1} shares one clear idea.",
                media_prompts=[
                    ImagePrompt(
                        prompt=f"Calm illustration {index + 1}.",
                        aspect_ratio="4:5",
                    )
                ],
            )
            for index in range(count)
        ],
    )


def _text_layers(composition: Any) -> list[TextLayer]:
    return [layer for layer in composition.layers if isinstance(layer, TextLayer)]


def _image_layers(composition: Any) -> list[ImageLayer]:
    return [layer for layer in composition.layers if isinstance(layer, ImageLayer)]


def test_the_packaged_image_post_recipe_uses_a_square_still() -> None:
    recipe = load_packaged_formats().get("image_post")

    assert [stage.stage for stage in recipe.stages] == [
        "input_topic",
        "write",
        "visuals",
        "render",
    ]
    assert recipe.stages[2].options == {"strategy": "image"}
    assert recipe.stages[3].options == {"aspect": "square"}
    assert recipe.capabilities == ["llm", "image_gen"]
    assert recipe.output_type is OutputType.IMAGE
    assert recipe.handler is OutputHandler.SAVE


def test_the_packaged_carousel_recipe_uses_a_portrait_carousel() -> None:
    recipe = load_packaged_formats().get("carousel")

    assert [stage.stage for stage in recipe.stages] == [
        "input_topic",
        "write",
        "visuals",
        "render",
    ]
    assert recipe.stages[3].options == {
        "aspect": "portrait",
        "carousel": True,
    }
    assert recipe.capabilities == ["llm", "image_gen"]
    assert recipe.output_type is OutputType.CAROUSEL
    assert recipe.handler is OutputHandler.SAVE


def test_aspect_presets_cover_the_social_shapes() -> None:
    assert {"square", "portrait", "story", "landscape"} <= set(ASPECT_PRESETS)
    assert DEFAULT_ASPECT_PRESET == "square"
    assert ASPECT_PRESETS["square"].size == (1080, 1080)
    assert ASPECT_PRESETS["portrait"].size == (1080, 1350)
    assert ASPECT_PRESETS["story"].size == (1080, 1920)
    assert ASPECT_PRESETS["landscape"].size == (1920, 1080)
    assert ASPECT_PRESETS["portrait"].ratio == "4:5"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("square", (1080, 1080)),
        ("portrait", (1080, 1350)),
        ("4:5", (1080, 1350)),
        ("9:16", (1080, 1920)),
        ("16:9", (1920, 1080)),
        ("1080x1350", (1080, 1350)),
    ],
)
def test_aspect_presets_resolve_names_ratios_and_sizes(
    value: str,
    expected: tuple[int, int],
) -> None:
    assert resolve_aspect_preset(value).size == expected


def test_an_unknown_aspect_preset_names_the_available_ones() -> None:
    with pytest.raises(ConfigError, match="unknown aspect preset"):
        resolve_aspect_preset("billboard")


def test_an_image_post_uses_the_preset_canvas_and_its_text() -> None:
    composition = image_post_composition(
        {"text": "A single clear idea.", "title": "One thought"},
        aspect="story",
    )

    assert composition.canvas.size == (1080, 1920)
    assert [layer.text for layer in _text_layers(composition)] == [
        "One thought",
        "A single clear idea.",
    ]


def test_a_carousel_puts_each_beats_text_on_its_own_slide() -> None:
    compositions = carousel_compositions(_script(3), aspect="portrait")

    assert len(compositions) == 3
    assert [composition.id for composition in compositions] == [
        "carousel-slide-1",
        "carousel-slide-2",
        "carousel-slide-3",
    ]
    assert [
        [layer.text for layer in _text_layers(composition)]
        for composition in compositions
    ] == [
        ["Slide 1 shares one clear idea."],
        ["Slide 2 shares one clear idea."],
        ["Slide 3 shares one clear idea."],
    ]
    assert all(composition.canvas.size == (1080, 1350) for composition in compositions)


def test_an_explicit_carousel_title_lands_on_the_first_slide() -> None:
    compositions = carousel_compositions(
        [{"text": "First."}, {"text": "Second."}],
        title="Cover title",
    )

    assert [layer.text for layer in _text_layers(compositions[0])] == [
        "Cover title",
        "First.",
    ]
    assert [layer.text for layer in _text_layers(compositions[1])] == ["Second."]


def test_slide_images_become_cover_layers_and_keep_their_text() -> None:
    compositions = carousel_compositions(
        [
            {"text": "First slide.", "image": "one.png"},
            {"caption": "Second slide.", "path": "two.png"},
        ],
        aspect="square",
    )

    assert [layer.source for layer in _image_layers(compositions[0])] == ["one.png"]
    assert [layer.source for layer in _image_layers(compositions[1])] == ["two.png"]
    assert [layer.text for layer in _text_layers(compositions[1])] == ["Second slide."]


def test_images_can_be_supplied_separately_from_the_script() -> None:
    slides = image_post_slides(_script(2), images=["first.png", "second.png"])

    assert [slide.text for slide in slides] == [
        "Slide 1 shares one clear idea.",
        "Slide 2 shares one clear idea.",
    ]
    assert [slide.source for slide in slides] == ["first.png", "second.png"]


def test_a_slide_without_text_or_an_image_is_rejected() -> None:
    with pytest.raises(ConfigError, match="needs text or an image"):
        image_post_composition({"text": "   "})

    with pytest.raises(ConfigError, match="needs at least one slide"):
        carousel_compositions([])


def test_an_image_post_renders_and_saves_a_still(tmp_path: Path) -> None:
    output = tmp_path / "post.png"

    saved = save_image_post(
        {"text": "A clear idea."},
        output,
        aspect="square",
    )

    assert saved == output
    assert output.is_file()
    assert output.stat().st_size > 0
    image = render_image_post({"text": "A clear idea."}, aspect="square")
    assert image.size == (1080, 1080)


def test_a_carousel_renders_one_png_per_slide(tmp_path: Path) -> None:
    background = tmp_path / "background.png"
    Image.new("RGB", (64, 64), "#336699").save(background)
    source = [
        {"text": "First slide.", "image": str(background)},
        {"text": "Second slide.", "image": str(background)},
    ]

    paths = save_carousel(source, tmp_path / "slides", aspect="square")

    assert [path.name for path in paths] == ["slide-1.png", "slide-2.png"]
    assert all(path.is_file() for path in paths)
    assert [Image.open(path).size for path in paths] == [
        (1080, 1080),
        (1080, 1080),
    ]


def test_the_renderers_pin_an_aspect_preset_for_many_posts() -> None:
    post = ImagePostRenderer("portrait")
    carousel = CarouselRenderer("story")

    assert post.composition({"text": "One idea."}).canvas.size == (1080, 1350)
    compositions = carousel.compositions([{"text": "One."}, {"text": "Two."}])
    assert [composition.canvas.size for composition in compositions] == [
        (1080, 1920),
        (1080, 1920),
    ]


def test_the_image_post_stage_writes_a_composition_and_a_png(
    tmp_path: Path,
) -> None:
    script = _script(1)
    (tmp_path / "script.en.json").write_text(script.to_json(), encoding="utf-8")
    stage = ImagePostStage(aspect="square")

    result = asyncio.run(stage.run(StageContext(run_dir=tmp_path, config={})))

    assert result.artifacts == ["image-post.yaml", "image-post.png"]
    assert (tmp_path / "image-post.yaml").is_file()
    assert (tmp_path / "image-post.png").is_file()
    payload = yaml.safe_load((tmp_path / "image-post.yaml").read_text(encoding="utf-8"))
    texts = [
        layer["text"] for layer in payload["layers"] if layer.get("type") == "text"
    ]
    assert "Slide 1 shares one clear idea." in texts


def test_the_carousel_stage_writes_one_slide_per_beat(tmp_path: Path) -> None:
    script = _script(3)
    (tmp_path / "script.en.json").write_text(script.to_json(), encoding="utf-8")
    stage = CarouselStage(aspect="portrait")

    result = asyncio.run(stage.run(StageContext(run_dir=tmp_path, config={})))

    assert result.artifacts[0] == "carousel.json"
    manifest = json.loads((tmp_path / "carousel.json").read_text(encoding="utf-8"))
    assert manifest["aspect"] == "portrait"
    assert manifest["width"] == 1080
    assert manifest["height"] == 1350
    assert [slide["text"] for slide in manifest["slides"]] == [
        "Slide 1 shares one clear idea.",
        "Slide 2 shares one clear idea.",
        "Slide 3 shares one clear idea.",
    ]
    for slide in manifest["slides"]:
        assert (tmp_path / slide["composition"]).is_file()
        assert (tmp_path / slide["image"]).is_file()


def test_the_carousel_stage_uses_generated_visuals_when_present(
    tmp_path: Path,
) -> None:
    background = tmp_path / "generated.png"
    Image.new("RGB", (64, 64), "#336699").save(background)
    (tmp_path / "script.en.json").write_text(
        _script(2).to_json(),
        encoding="utf-8",
    )
    (tmp_path / "visual-reuse.json").write_text(
        json.dumps(
            {
                "version": 1,
                "beats": [
                    {"index": 0, "asset": {"path": str(background)}},
                    {"index": 1, "asset": {"path": str(background)}},
                ],
            }
        ),
        encoding="utf-8",
    )
    stage = CarouselStage(aspect="square")

    asyncio.run(stage.run(StageContext(run_dir=tmp_path, config={})))

    manifest = json.loads((tmp_path / "carousel.json").read_text(encoding="utf-8"))
    assert [slide["source"] for slide in manifest["slides"]] == [
        str(background),
        str(background),
    ]
    composition = yaml.safe_load(
        (tmp_path / manifest["slides"][0]["composition"]).read_text(encoding="utf-8")
    )
    image_layers = [
        layer for layer in composition["layers"] if layer.get("type") == "image"
    ]
    assert image_layers[0]["source"] == str(background)


def test_a_slide_mapping_accepts_caption_and_image_aliases() -> None:
    slide = ImagePostSlide.model_validate(
        {"caption": "Alias text.", "image": "cover.png", "alt": "A cover"}
    )

    assert slide.text == "Alias text."
    assert slide.source == "cover.png"
    assert slide.alt_text == "A cover"
    assert ImagePost(slides=[slide]).slides[0] == slide
