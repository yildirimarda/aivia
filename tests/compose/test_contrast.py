"""Contrast selection and outline, shadow and backing box treatments."""

from __future__ import annotations

from typing import Any, cast

import pytest
from PIL import Image

from aivia.compose import (
    ContrastStrategy,
    ContrastUtility,
    RenderError,
    add_backing_box,
    add_outline,
    add_shadow,
    apply_contrast,
    apply_contrast_treatment,
    choose_text_color,
    contrast_ratio,
    ensure_contrast,
    parse_color,
    relative_luminance,
)
from aivia.core.config.sections import RenderConfig

WHITE = (255, 255, 255, 255)
BLACK = (0, 0, 0, 255)


def _text_image() -> Image.Image:
    image = Image.new("RGBA", (24, 12), (0, 0, 0, 0))
    image.paste(WHITE, (6, 4, 18, 8))
    return image


def test_relative_luminance_matches_the_wcag_extremes_and_primaries() -> None:
    assert relative_luminance("#000000") == pytest.approx(0.0)
    assert relative_luminance("#ffffff") == pytest.approx(1.0)
    assert relative_luminance("#ff0000") == pytest.approx(0.2126, abs=1e-4)
    assert relative_luminance("#00ff00") == pytest.approx(0.7152, abs=1e-4)
    assert relative_luminance("#0000ff") == pytest.approx(0.0722, abs=1e-4)


def test_contrast_ratio_covers_the_full_range_and_is_symmetric() -> None:
    assert contrast_ratio("#000000", "#ffffff") == pytest.approx(21.0)
    assert contrast_ratio("#ffffff", "#000000") == pytest.approx(21.0)
    assert contrast_ratio("#ffffff", "#ffffff") == pytest.approx(1.0)


def test_parse_color_accepts_names_hex_triples_and_quadruples() -> None:
    assert parse_color("white") == WHITE
    assert parse_color("#000") == BLACK
    assert parse_color((255, 0, 0)) == (255, 0, 0, 255)
    assert parse_color([1, 2, 3, 4]) == (1, 2, 3, 4)


def test_choose_text_color_picks_the_opposite_polarity() -> None:
    assert choose_text_color("#000000") == "#ffffff"
    assert choose_text_color("#ffffff") == "#000000"
    assert choose_text_color("#101010") == "#ffffff"
    assert choose_text_color("#f0f0f0") == "#000000"


def test_choose_text_color_honours_custom_light_and_dark_colours() -> None:
    assert choose_text_color("#000000", light="#fefefe", dark="#010101") == "#fefefe"


def test_already_high_contrast_text_needs_no_treatment() -> None:
    result = ensure_contrast("#000000", min_contrast=4.5)

    assert result.text_color == "#ffffff"
    assert result.base_ratio == pytest.approx(21.0)
    assert result.ratio == pytest.approx(21.0)
    assert result.applied is False
    assert result.meets_contrast is True
    assert result.treatment is None


def test_low_contrast_text_gets_an_outline_that_meets_the_minimum() -> None:
    result = ensure_contrast(
        "#888888",
        "#777777",
        min_contrast=4.5,
        strategy="outline",
    )

    assert result.applied is True
    assert result.strategy is ContrastStrategy.OUTLINE
    assert result.outline_color == "#000000"
    assert result.outline_width == 2.0
    assert result.ratio >= 4.5
    assert result.meets_contrast is True


def test_shadow_strategy_reports_its_colour_offset_and_blur() -> None:
    result = ensure_contrast(
        "#888888",
        "#777777",
        min_contrast=4.5,
        strategy="shadow",
        shadow_offset=(3, -2),
        shadow_blur=1.5,
    )

    assert result.applied is True
    assert result.strategy is ContrastStrategy.SHADOW
    assert result.shadow_color == "#000000"
    assert result.shadow_offset == (3.0, -2.0)
    assert result.shadow_blur == pytest.approx(1.5)
    assert result.ratio >= 4.5


def test_box_strategy_chooses_text_against_the_backing_box() -> None:
    result = ensure_contrast(
        "#888888",
        min_contrast=7.0,
        strategy="box",
        box_padding=12,
    )

    assert result.applied is True
    assert result.strategy is ContrastStrategy.BOX
    assert result.box_color == "#ffffff"
    assert result.text_color == "#000000"
    assert result.box_padding == 12.0
    assert result.ratio == pytest.approx(21.0)
    assert result.meets_contrast is True


def test_box_strategy_keeps_explicit_text_colour_when_it_contrasts_the_box() -> None:
    result = ensure_contrast(
        "#888888",
        "#777777",
        min_contrast=4.5,
        strategy="box",
        box_color="#000000",
    )

    assert result.text_color == "#777777"
    assert result.box_color == "#000000"
    assert result.ratio >= 4.5


def test_render_config_supplies_the_minimum_and_strategy() -> None:
    config = RenderConfig.from_mapping(
        {"min_contrast": 7.0, "contrast_strategy": "shadow"}
    )

    result = ensure_contrast("#888888", "#777777", config=config)

    assert result.min_contrast == 7.0
    assert result.strategy is ContrastStrategy.SHADOW
    assert result.applied is True


def test_mapping_config_supplies_the_minimum_and_strategy() -> None:
    result = ensure_contrast(
        "#888888",
        "#777777",
        config={"min_contrast": 4.5, "contrast_strategy": "box"},
    )

    assert result.strategy is ContrastStrategy.BOX
    assert result.applied is True


def test_explicit_arguments_override_the_config() -> None:
    config = RenderConfig.from_mapping(
        {"min_contrast": 7.0, "contrast_strategy": "shadow"}
    )

    result = ensure_contrast(
        "#000000",
        min_contrast=4.5,
        strategy="outline",
        config=config,
    )

    assert result.min_contrast == 4.5
    assert result.strategy is ContrastStrategy.OUTLINE
    assert result.applied is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_contrast": 0.5},
        {"min_contrast": 21.5},
        {"strategy": "sparkle"},
        {"outline_width": -1},
        {"strategy": "shadow", "shadow_blur": -1},
        {"strategy": "box", "box_padding": -1},
    ],
)
def test_invalid_options_are_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(RenderError):
        ensure_contrast("#888888", "#777777", **kwargs)


def test_invalid_colours_are_rejected() -> None:
    with pytest.raises(RenderError):
        ensure_contrast("not-a-colour")


def test_the_utility_facade_uses_its_configured_defaults() -> None:
    utility = ContrastUtility(min_contrast=7.0, strategy="box")

    result = utility.ensure_contrast("#888888")

    assert result.min_contrast == 7.0
    assert result.strategy is ContrastStrategy.BOX
    assert result.applied is True


def test_outline_grows_the_alpha_around_the_text() -> None:
    source = _text_image()

    outlined = add_outline(source, "#000000", 1)

    assert outlined.size == source.size
    assert outlined.getpixel((12, 6)) == WHITE
    assert outlined.getpixel((5, 6)) == BLACK
    assert cast(tuple[int, ...], outlined.getpixel((0, 0)))[3] == 0


def test_zero_width_outline_returns_an_unchanged_copy() -> None:
    source = _text_image()

    outlined = add_outline(source, "#000000", 0)

    assert outlined is not source
    assert outlined.tobytes() == source.tobytes()


def test_shadow_is_offset_and_padded_behind_the_text() -> None:
    source = _text_image()

    shadowed = add_shadow(source, "#000000", (2, 2), blur=0, opacity=1.0)

    assert shadowed.width > source.width
    assert shadowed.height > source.height
    assert shadowed.getpixel((12, 8)) == WHITE
    assert cast(tuple[int, ...], shadowed.getpixel((12, 8)))[3] == 255
    alpha = shadowed.getchannel("A")
    opaque = sum(count for value, count in enumerate(alpha.histogram()) if value > 0)
    assert opaque > 12 * 4


def test_backing_box_is_drawn_behind_the_text() -> None:
    source = _text_image()

    boxed = add_backing_box(source, "#000000", 4)

    assert boxed.width >= source.width
    assert boxed.getpixel((12, 6)) == WHITE
    assert boxed.getpixel((2, 1)) == BLACK


def test_treatment_dispatch_matches_the_individual_helpers() -> None:
    source = _text_image()

    assert (
        apply_contrast_treatment(source, "outline", width=1).tobytes()
        == add_outline(source, "#000000", 1).tobytes()
    )
    assert (
        apply_contrast_treatment(
            source, "shadow", offset=(1, 1), blur=0, opacity=1.0
        ).tobytes()
        == add_shadow(source, "#000000", (1, 1), 0, opacity=1.0).tobytes()
    )


def test_apply_contrast_dispatches_images_to_treatments() -> None:
    source = _text_image()

    result = apply_contrast(source, "outline", width=1)

    assert isinstance(result, Image.Image)
    assert result.tobytes() == add_outline(source, "#000000", 1).tobytes()
