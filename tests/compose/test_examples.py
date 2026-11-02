"""The bundled example compositions validate and render deterministically."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageChops, ImageColor

from aivia.compose import (
    load_composition,
    render_still,
    validate_composition,
    validate_composition_or_raise,
)

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = ROOT / "examples" / "compositions"

EXAMPLES = (
    "square-quote.yaml",
    "story-teaser.yaml",
    "wide-banner.yaml",
)


def _example_paths() -> list[Path]:
    return sorted(EXAMPLES_DIR.glob("*.yaml"))


def test_examples_directory_contains_exactly_the_three_examples() -> None:
    assert {path.name for path in _example_paths()} == set(EXAMPLES)


@pytest.mark.parametrize("name", EXAMPLES)
def test_each_example_validates_and_renders(name: str, tmp_path: Path) -> None:
    path = EXAMPLES_DIR / name
    composition = load_composition(path)

    issues = validate_composition(composition)
    assert issues.ok, [str(issue) for issue in issues]
    assert validate_composition_or_raise(composition) is composition

    start = render_still(composition, 0.0)
    middle = render_still(composition, composition.duration_seconds / 2.0)

    for image in (start, middle):
        assert image.size == composition.canvas.size
        assert image.mode == "RGBA"

    background = Image.new(
        "RGBA",
        composition.canvas.size,
        ImageColor.getcolor(composition.canvas.background, "RGBA"),
    )
    difference = ImageChops.difference(middle.convert("RGB"), background.convert("RGB"))
    assert difference.getbbox() is not None

    repeated = render_still(composition, composition.duration_seconds / 2.0)
    assert ImageChops.difference(middle, repeated).getbbox() is None

    output = tmp_path / f"{path.stem}.png"
    middle.save(output)
    assert output.stat().st_size > 0
    with Image.open(output) as reopened:
        assert reopened.size == composition.canvas.size
