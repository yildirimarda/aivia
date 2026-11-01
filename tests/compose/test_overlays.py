"""Procedural confetti, sparkles, snow and bokeh overlays."""

from __future__ import annotations

from typing import Any

import pytest
from PIL import Image
from pydantic import ValidationError

from aivia.compose import (
    Composition,
    Effect,
    Overlay,
    OverlayError,
    OverlayType,
    PillowStillRenderer,
    apply_overlay_effects,
    overlay_from_effect,
    particles,
    render_overlay,
)

pytestmark = pytest.mark.slow

SIZE = (160, 120)


def _composition(effects: list[dict[str, Any]], **overrides: Any) -> Composition:
    values: dict[str, Any] = {
        "canvas": {"width": SIZE[0], "height": SIZE[1], "background": "#000000"},
        "duration_seconds": 3.0,
        "fps": 24,
        "effects": effects,
    }
    values.update(overrides)
    return Composition.model_validate(values)


def _frame(overlay: Overlay, time_seconds: float = 0.0) -> Image.Image:
    return render_overlay(overlay, SIZE, time_seconds)


def _highest_alpha(image: Image.Image) -> int:
    _, high = image.getchannel("A").getextrema()
    assert isinstance(high, int)
    return high


def _brightest(image: Image.Image) -> int:
    _, high = image.convert("L").getextrema()
    assert isinstance(high, int)
    return high


def test_overlay_types_accept_shorthand_aliases_and_sizes() -> None:
    assert Overlay.model_validate("confetti").type is OverlayType.CONFETTI
    assert Overlay.model_validate("snowfall").type is OverlayType.SNOW
    assert Overlay.model_validate("glitter").type is OverlayType.SPARKLES
    assert Overlay.model_validate("lights").type is OverlayType.BOKEH

    overlay = Overlay.model_validate(
        {
            "type": "confetti",
            "seed": "launch",
            "particles": 12,
            "colour": "#ff0000",
            "size": [0.01, 0.02],
        }
    )

    assert overlay.count == 12
    assert overlay.colors == ("#ff0000",)
    assert overlay.size_range[0] == pytest.approx(0.01)
    assert overlay.size_range[1] == pytest.approx(0.02)


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "fireworks"},
        {"type": "snow", "count": -1},
        {"type": "snow", "opacity": 2},
        {"type": "snow", "colors": []},
        {"type": "snow", "colors": ["not-a-colour"]},
        {"type": "snow", "min_size": 0.02, "max_size": 0.01},
        {"type": "snow", "size": [0.01, 0.02, 0.03]},
    ],
)
def test_overlay_rejects_invalid_values(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Overlay.model_validate(payload)


def test_seed_accepts_text_and_stays_stable() -> None:
    first = Overlay.model_validate({"type": "snow", "seed": "winter"})
    second = Overlay.model_validate({"type": "snow", "seed": "winter"})

    assert first.seed == second.seed
    assert first.seed != Overlay.model_validate({"type": "snow", "seed": "summer"}).seed


@pytest.mark.parametrize("overlay_type", list(OverlayType))
def test_same_seed_renders_identical_pixels(overlay_type: OverlayType) -> None:
    first = _frame(Overlay(type=overlay_type, seed=11), 0.7)
    second = _frame(Overlay(type=overlay_type, seed=11), 0.7)
    different = _frame(Overlay(type=overlay_type, seed=12), 0.7)

    assert first.mode == "RGBA"
    assert first.size == SIZE
    assert first.tobytes() == second.tobytes()
    assert first.tobytes() != different.tobytes()


@pytest.mark.parametrize("overlay_type", list(OverlayType))
def test_every_overlay_type_draws_visible_pixels(overlay_type: OverlayType) -> None:
    image = _frame(Overlay(type=overlay_type, seed=3))

    assert _highest_alpha(image) > 0
    assert image.getchannel("A").getbbox() is not None


@pytest.mark.parametrize("overlay_type", list(OverlayType))
def test_overlays_move_between_frames(overlay_type: OverlayType) -> None:
    first = _frame(Overlay(type=overlay_type, seed=5), 0.0)
    later = _frame(Overlay(type=overlay_type, seed=5), 1.0)

    assert first.tobytes() != later.tobytes()


def test_particles_are_normalised_and_repeatable() -> None:
    overlay = Overlay(type=OverlayType.SNOW, seed=9, count=25)

    first = particles(overlay, 1.5)
    second = particles(overlay, 1.5)

    assert first == second
    assert len(first) == 25
    assert all(0.0 <= particle.x < 1.0 for particle in first)
    assert all(0.0 <= particle.y < 1.0 for particle in first)


def test_opacity_scales_the_whole_overlay() -> None:
    opaque = _frame(Overlay(type=OverlayType.SNOW, seed=4, opacity=1.0))
    faint = _frame(Overlay(type=OverlayType.SNOW, seed=4, opacity=0.25))

    assert _highest_alpha(faint) < _highest_alpha(opaque)


def test_overlay_effects_are_applied_after_the_layers() -> None:
    composition = _composition(
        [{"type": "snow", "params": {"seed": 7, "count": 200}}]
    )

    image = PillowStillRenderer().render_still(composition, 0.5)

    assert image.mode == "RGBA"
    assert _brightest(image) > 0


def test_overlay_effect_uses_time_relative_to_its_start() -> None:
    composition = _composition(
        [
            {
                "type": "sparkles",
                "start": 1.0,
                "end": 2.5,
                "params": {"seed": 21},
            }
        ]
    )

    rendered = PillowStillRenderer().render_still(composition, 1.5)
    direct = render_overlay(Overlay(type=OverlayType.SPARKLES, seed=21), SIZE, 0.5)
    expected = Image.alpha_composite(
        Image.new("RGBA", SIZE, (0, 0, 0, 255)), direct
    )

    assert rendered.tobytes() == expected.tobytes()


def test_overlay_effect_respects_its_window_and_enabled_flag() -> None:
    before = _composition([{"type": "confetti", "start": 1.0, "end": 2.0}])
    during = _composition([{"type": "confetti", "start": 1.0, "end": 2.0}])
    disabled = _composition([{"type": "confetti", "enabled": False}])
    renderer = PillowStillRenderer()

    assert _brightest(renderer.render_still(before, 0.5)) == 0
    assert _brightest(renderer.render_still(during, 1.5)) > 0
    assert _brightest(renderer.render_still(disabled, 1.5)) == 0


def test_unknown_effects_are_ignored() -> None:
    composition = _composition([{"type": "fade", "params": {"duration": 1}}])

    image = PillowStillRenderer().render_still(composition, 0.5)

    assert _brightest(image) == 0


def test_invalid_overlay_effect_parameters_raise_a_render_error() -> None:
    composition = _composition(
        [{"id": "bad-snow", "type": "snow", "params": {"count": -1}}]
    )

    with pytest.raises(OverlayError, match="bad-snow"):
        PillowStillRenderer().render_still(composition, 0.5)


def test_effect_parser_accepts_direct_and_generic_overlay_types() -> None:
    direct = overlay_from_effect(Effect.model_validate({"type": "snowfall"}))
    generic = overlay_from_effect(
        Effect.model_validate(
            {
                "type": "overlay",
                "params": {
                    "type": "bokeh",
                    "seed": 5,
                    "count": 3,
                    "size": [0.05, 0.1],
                },
            }
        )
    )

    assert direct is not None
    assert direct.type is OverlayType.SNOW
    assert generic is not None
    assert generic.type is OverlayType.BOKEH
    assert generic.count == 3
    assert generic.size_range[0] == pytest.approx(0.05)
    assert generic.size_range[1] == pytest.approx(0.1)
    assert overlay_from_effect(Effect.model_validate("fade")) is None
    assert overlay_from_effect(Effect.model_validate("overlay")) is None


def test_apply_overlay_effects_returns_the_same_image() -> None:
    image = Image.new("RGBA", SIZE, (0, 0, 0, 255))
    composition = _composition([{"type": "snow", "params": {"seed": 1}}])

    result = apply_overlay_effects(image, composition, 0.5)

    assert result is image
    assert _brightest(result) > 0
