"""Character short and character card formats: fail clearly when no character is
selected, and proceed when one is.

These formats are strictly opt-in: a run must opt into a character (``--character``,
the order or instruction field, or the profile's character). When nothing opts in
the stage raises a clear error before any media work starts.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

import pytest
import yaml

from aivia.characters import CharacterLibrary
from aivia.core.config.formats import FormatRecipeConfig, OutputHandler, OutputType
from aivia.core.errors import ConfigError
from aivia.core.models.run import StageResult
from aivia.core.stages import StageContext
from aivia.formats import (
    CharacterCardStage,
    CharacterShortStage,
    build_character_card_stage,
    build_character_short_stage,
    load_packaged_formats,
)


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine)


def _context(run_dir: Path) -> StageContext:
    return StageContext(run_dir=run_dir, config={})


def _write_character(characters_dir: Path, name: str, **extra: Any) -> None:
    characters_dir.mkdir(parents=True, exist_ok=True)
    payload = {"name": name, **extra}
    (characters_dir / f"{name}.yaml").write_text(
        yaml.safe_dump(payload), encoding="utf-8"
    )


def _character_short() -> FormatRecipeConfig:
    return load_packaged_formats().get("character_short")


def _character_card() -> FormatRecipeConfig:
    return load_packaged_formats().get("character_card")


def test_the_packaged_character_short_recipe_requires_a_character() -> None:
    recipe = _character_short()

    assert [stage.stage for stage in recipe.stages] == [
        "input_topic",
        "write",
        "voice",
        "visuals",
        "render",
    ]
    assert recipe.capabilities == ["llm", "tts", "video_gen"]
    assert recipe.output_type is OutputType.VIDEO
    assert recipe.handler is OutputHandler.PUBLISH


def test_the_packaged_character_card_recipe_requires_a_character() -> None:
    recipe = _character_card()

    assert [stage.stage for stage in recipe.stages] == [
        "input_topic",
        "write",
        "render",
    ]
    assert recipe.capabilities == ["llm", "image_gen"]
    assert recipe.output_type is OutputType.IMAGE
    assert recipe.handler is OutputHandler.SAVE


def test_character_short_fails_clearly_when_no_character_is_selected(
    tmp_path: Path,
) -> None:
    stage = CharacterShortStage()

    with pytest.raises(ConfigError) as caught:
        _run(stage.run(_context(tmp_path)))

    error = caught.value
    assert "character short requires a selected character" in str(error)
    assert "none was selected" in str(error)
    assert error.details["stage"] == "render"
    assert error.details["format"] == "character_short"
    assert error.details["error"] == "character_not_selected"


def test_character_card_fails_clearly_when_no_character_is_selected(
    tmp_path: Path,
) -> None:
    stage = CharacterCardStage()

    with pytest.raises(ConfigError) as caught:
        _run(stage.run(_context(tmp_path)))

    error = caught.value
    assert "character card requires a selected character" in str(error)
    assert "none was selected" in str(error)
    assert error.details["stage"] == "render"
    assert error.details["format"] == "character_card"
    assert error.details["error"] == "character_not_selected"


def test_character_short_fails_clearly_when_character_is_blank(tmp_path: Path) -> None:
    stage = CharacterShortStage(character="   ")

    with pytest.raises(ConfigError) as caught:
        _run(stage.run(_context(tmp_path)))

    assert "none was selected" in str(caught.value)


def test_character_short_proceeds_when_a_character_is_selected(tmp_path: Path) -> None:
    characters_dir = tmp_path / "data" / "characters"
    _write_character(characters_dir, "mascot", art_style="friendly robot")
    library = CharacterLibrary(data_dir=tmp_path / "data")
    run_dir = tmp_path / "run"
    stage = CharacterShortStage(character="mascot", library=library)

    result: StageResult = _run(stage.run(_context(run_dir)))

    assert result.status.value == "succeeded"
    assert result.cost_usd == 0.0
    assert result.metadata["character"] == "mascot"
    composition_path = run_dir / "character-short.yaml"
    assert composition_path.is_file()
    text = composition_path.read_text(encoding="utf-8")
    assert "mascot" in text
    assert "friendly robot" in text


def test_character_card_proceeds_when_a_character_is_selected(tmp_path: Path) -> None:
    characters_dir = tmp_path / "data" / "characters"
    _write_character(characters_dir, "hero", art_style="bold explorer")
    library = CharacterLibrary(data_dir=tmp_path / "data")
    run_dir = tmp_path / "run"
    stage = CharacterCardStage(character="hero", library=library)

    result: StageResult = _run(stage.run(_context(run_dir)))

    assert result.status.value == "succeeded"
    assert result.cost_usd == 0.0
    assert result.metadata["character"] == "hero"
    composition_path = run_dir / "character-card.yaml"
    image_path = run_dir / "character-card.png"
    assert composition_path.is_file()
    assert image_path.is_file()
    assert image_path.stat().st_size > 0
    text = composition_path.read_text(encoding="utf-8")
    assert "hero" in text
    assert "bold explorer" in text


def test_character_short_fails_clearly_when_character_is_missing(
    tmp_path: Path,
) -> None:
    library = CharacterLibrary(data_dir=tmp_path / "data")
    stage = CharacterShortStage(character="no-such-character", library=library)

    with pytest.raises(ConfigError) as caught:
        _run(stage.run(_context(tmp_path / "run")))

    error = caught.value
    assert "unknown character" in str(error)
    assert "no-such-character" in str(error)
    assert error.details["name"] == "no-such-character"


def test_build_helpers_wire_the_character_option() -> None:
    short = build_character_short_stage(character="mascot")
    card = build_character_card_stage(character="hero")

    assert short.character == "mascot"
    assert card.character == "hero"
    assert isinstance(short, CharacterShortStage)
    assert isinstance(card, CharacterCardStage)


@pytest.mark.slow
def test_character_card_renders_a_deterministic_still(tmp_path: Path) -> None:
    characters_dir = tmp_path / "data" / "characters"
    _write_character(characters_dir, "icon")
    library = CharacterLibrary(data_dir=tmp_path / "data")
    run_dir = tmp_path / "run"
    stage = CharacterCardStage(character="icon", library=library)

    first: StageResult = _run(stage.run(_context(run_dir)))
    second_dir = tmp_path / "run2"
    second_result: StageResult = _run(
        CharacterCardStage(character="icon", library=library).run(_context(second_dir))
    )

    first_image = (run_dir / "character-card.png").read_bytes()
    second_image = (second_dir / "character-card.png").read_bytes()
    assert first_image == second_image
    assert first.status.value == "succeeded"
    assert second_result.status.value == "succeeded"
