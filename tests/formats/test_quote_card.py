"""The built-in quote card format renders typography templates stored as data."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
import yaml

from aivia.compose import (
    TextLayer,
    build_quote_card_composition,
    load_packaged_quote_card_templates,
    load_quote_card_template,
    load_quote_card_templates,
    quote_card_composition,
    render_quote_card,
    save_quote_card,
)
from aivia.core.config.formats import FormatRecipeConfig, OutputHandler, OutputType
from aivia.core.errors import ConfigError
from aivia.core.stages import StageContext
from aivia.formats import (
    QuoteCardRenderer,
    QuoteCardStage,
    load_packaged_formats,
)
from aivia.writing.quote import QuoteMode


def _quote_card() -> FormatRecipeConfig:
    return load_packaged_formats().get("quote_card")


def _script() -> Any:
    return QuoteMode().build_script(
        {
            "quote": "The unexamined life is not worth living.",
            "attribution": "Socrates",
            "source_urls": ["https://source.test/socrates"],
        }
    )


def test_the_packaged_quote_card_recipe_writes_a_typographic_still() -> None:
    recipe = _quote_card()

    assert [stage.stage for stage in recipe.stages] == [
        "input_topic",
        "write",
        "render",
    ]
    assert recipe.stages[1].options == {"mode": "quote"}
    assert recipe.stages[2].options == {"template": "classic"}
    assert recipe.capabilities == ["llm", "image_gen"]
    assert recipe.output_type is OutputType.IMAGE
    assert recipe.handler is OutputHandler.SAVE


def test_quote_card_templates_are_loaded_from_packaged_data() -> None:
    templates = load_packaged_quote_card_templates()

    assert templates.default.name == "classic"
    assert {"classic", "minimal", "portrait", "square"} <= set(templates.names())
    classic = templates.require("classic")
    assert classic.canvas.width == 1080
    assert classic.canvas.height == 1080
    assert classic.quote_layer is not None
    assert classic.quote_layer["font_size"] == 96
    assert classic.attribution_layer is not None
    assert classic.attribution_layer["font_size"] == 34


def test_the_quote_and_attribution_are_placed_in_the_template_layers() -> None:
    composition = build_quote_card_composition(_script(), "classic")

    text_layers = [
        layer for layer in composition.layers if isinstance(layer, TextLayer)
    ]
    assert [layer.text for layer in text_layers] == [
        "The unexamined life is not worth living.",
        "— Socrates",
    ]
    quote = text_layers[0]
    assert quote.font_family == "Lato"
    assert quote.font_size == pytest.approx(96)
    assert quote.font_weight.value == "bold"
    assert quote.align.value == "center"
    assert quote.vertical_align.value == "middle"
    assert quote.shrink_to_fit is True
    assert composition.canvas.width == 1080
    assert composition.duration_seconds == pytest.approx(6)


def test_a_style_only_template_is_turned_into_text_layers() -> None:
    template = load_quote_card_template("minimal")

    composition = quote_card_composition(
        {
            "quote": "Simplicity is the ultimate sophistication.",
            "attribution": "Anonymous",
        },
        template,
    )

    text_layers = [
        layer for layer in composition.layers if isinstance(layer, TextLayer)
    ]
    assert [layer.text for layer in text_layers] == [
        "Simplicity is the ultimate sophistication.",
        "— Anonymous",
    ]
    assert text_layers[0].font_size == pytest.approx(82)
    assert text_layers[0].color == "#111827"


def test_a_user_template_file_overrides_the_packaged_one(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    templates_dir = config_dir / "templates"
    templates_dir.mkdir(parents=True)
    (templates_dir / "quote_card.yaml").write_text(
        yaml.safe_dump(
            {
                "default": "classic",
                "templates": {
                    "classic": {
                        "canvas": {"width": 640, "height": 640},
                        "quote": {"font_size": 48, "color": "#ff0000"},
                        "attribution": {"font_size": 20},
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    template = load_quote_card_template(
        "classic",
        config_dir=config_dir,
    )
    composition = quote_card_composition(_script(), template)

    assert template.canvas.width == 640
    assert composition.canvas.width == 640
    text_layers = [
        layer for layer in composition.layers if isinstance(layer, TextLayer)
    ]
    assert text_layers[0].font_size == pytest.approx(48)
    assert text_layers[0].color == "#ff0000"


def test_a_template_mapping_can_be_loaded_without_a_file() -> None:
    templates = load_quote_card_templates(
        {
            "templates": {
                "custom": {
                    "canvas": {"width": 800, "height": 800, "background": "#123456"},
                    "quote": {"font_size": 60, "align": "left"},
                    "attribution": {"font_size": 24, "align": "right"},
                }
            }
        }
    )

    assert templates.names() == ("custom",)
    composition = quote_card_composition(_script(), templates.default)
    assert composition.canvas.background == "#123456"


def test_placeholders_fill_title_and_context_when_a_template_has_them() -> None:
    template = load_quote_card_templates(
        {
            "templates": {
                "editorial": {
                    "canvas": {"width": 1080, "height": 1080},
                    "layers": [
                        {
                            "id": "title",
                            "type": "text",
                            "text": "{{ title }}",
                            "x": 10,
                            "y": 10,
                            "width": 1060,
                            "height": 100,
                        },
                        {
                            "id": "quote",
                            "type": "text",
                            "text": "{{ quote }}",
                            "x": 10,
                            "y": 120,
                            "width": 1060,
                            "height": 700,
                        },
                        {
                            "id": "context",
                            "type": "text",
                            "text": "{{ context }}",
                            "x": 10,
                            "y": 900,
                            "width": 1060,
                            "height": 80,
                        },
                    ],
                }
            }
        }
    ).default

    composition = quote_card_composition(
        _script(),
        template,
        title="A short history",
        context="At his trial.",
    )

    text_layers = [
        layer for layer in composition.layers if isinstance(layer, TextLayer)
    ]
    assert [layer.text for layer in text_layers] == [
        "A short history",
        "The unexamined life is not worth living.",
        "At his trial.",
    ]


def test_an_empty_quote_is_rejected() -> None:
    with pytest.raises(ConfigError, match="needs a quote"):
        quote_card_composition({"quote": "  ", "attribution": "Anonymous"})


def test_a_quote_card_renders_and_saves_a_still(tmp_path: Path) -> None:
    output = tmp_path / "quote-card.png"

    saved = save_quote_card(_script(), output, "classic")

    assert saved == output
    assert output.is_file()
    assert output.stat().st_size > 0
    image = render_quote_card(_script(), "classic")
    assert image.size == (1080, 1080)


def test_the_renderer_pins_one_template_for_many_quotes() -> None:
    renderer = QuoteCardRenderer("portrait")

    first = renderer.composition(_script())
    second = renderer.composition(
        {"quote": "Another quotation.", "attribution": "Anonymous"}
    )

    assert first.canvas.width == 1080
    assert first.canvas.height == 1920
    assert second.canvas.height == 1920
    assert [layer.text for layer in second.layers if isinstance(layer, TextLayer)] == [
        "Another quotation.",
        "— Anonymous",
    ]


def test_the_quote_card_stage_writes_a_composition_and_a_still(tmp_path: Path) -> None:
    script = _script()
    (tmp_path / "script.en.json").write_text(script.to_json(), encoding="utf-8")
    stage = QuoteCardStage(template="classic")

    result = asyncio.run(stage.run(StageContext(run_dir=tmp_path, config={})))

    assert result.artifacts == ["quote-card.yaml", "quote-card.png"]
    assert (tmp_path / "quote-card.yaml").is_file()
    assert (tmp_path / "quote-card.png").is_file()
    assert "The unexamined life is not worth living." in (
        tmp_path / "quote-card.yaml"
    ).read_text(encoding="utf-8")
