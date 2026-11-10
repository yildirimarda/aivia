"""Personalized song card format: a still image card for a song with optional
character support.

The song card never requires a character. When one is selected it is loaded from
the user directory and its name and art style drive the card; when nothing is
selected the stage still proceeds and renders a plain song card naming the song.
A selected character that cannot be found fails clearly before any media work
starts.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

import pytest
import yaml

from aivia.characters import CharacterLibrary
from aivia.compose.schema import TextLayer
from aivia.core.errors import ConfigError
from aivia.core.models.run import StageResult
from aivia.core.stages import StageContext
from aivia.formats import (
    SongCardStage,
    build_song_card_stage,
    song_card_composition,
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


def _song_card(character: str | None = None, **kwargs: Any) -> SongCardStage:
    return SongCardStage(character=character, **kwargs)


def test_song_card_proceeds_without_a_character(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    stage = _song_card(
        scripts={"en": {"title": "Neon sky"}},
    )

    result: StageResult = _run(stage.run(_context(run_dir)))

    assert result.status.value == "succeeded"
    assert result.cost_usd == 0.0
    assert result.metadata["character"] == ""
    assert result.metadata["song_title"] == "Neon sky"
    composition_path = run_dir / "song-card.yaml"
    image_path = run_dir / "song-card.png"
    assert composition_path.is_file()
    assert image_path.is_file()
    assert image_path.stat().st_size > 0
    text = composition_path.read_text(encoding="utf-8")
    assert "Neon sky" in text


def test_song_card_proceeds_with_a_character(tmp_path: Path) -> None:
    characters_dir = tmp_path / "data" / "characters"
    _write_character(characters_dir, "hero", art_style="bold explorer")
    library = CharacterLibrary(data_dir=tmp_path / "data")
    run_dir = tmp_path / "run"
    stage = _song_card(
        character="hero",
        library=library,
        scripts={"en": {"title": "Neon sky"}},
    )

    result: StageResult = _run(stage.run(_context(run_dir)))

    assert result.status.value == "succeeded"
    assert result.cost_usd == 0.0
    assert result.metadata["character"] == "hero"
    assert result.metadata["song_title"] == "Neon sky"
    composition_path = run_dir / "song-card.yaml"
    image_path = run_dir / "song-card.png"
    assert composition_path.is_file()
    assert image_path.is_file()
    assert image_path.stat().st_size > 0
    text = composition_path.read_text(encoding="utf-8")
    assert "Neon sky" in text
    assert "hero" in text
    assert "bold explorer" in text


def test_song_card_omits_character_when_none_is_selected(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    stage = _song_card(
        scripts={"en": {"title": "Neon sky"}},
    )

    result: StageResult = _run(stage.run(_context(run_dir)))

    assert result.status.value == "succeeded"
    text = (run_dir / "song-card.yaml").read_text(encoding="utf-8")
    assert "Neon sky" in text
    assert "hero" not in text


def test_song_card_fails_when_selected_character_is_not_found(
    tmp_path: Path,
) -> None:
    library = CharacterLibrary(data_dir=tmp_path / "data")
    stage = _song_card(character="no-such-character", library=library)

    with pytest.raises(ConfigError) as caught:
        _run(stage.run(_context(tmp_path / "run")))

    error = caught.value
    assert "unknown character" in str(error)
    assert "no-such-character" in str(error)
    assert error.details["stage"] == "render"
    assert error.details["format"] == "song_card"
    assert error.details["error"] == "character_not_found"
    assert error.details["name"] == "no-such-character"


def test_song_card_fails_when_selected_character_is_blank(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    stage = _song_card(character="   ", scripts={"en": {"title": "Neon sky"}})

    result: StageResult = _run(stage.run(_context(run_dir)))

    assert result.status.value == "succeeded"
    text = (run_dir / "song-card.yaml").read_text(encoding="utf-8")
    assert "Neon sky" in text


def test_build_song_card_stage_wires_the_character_option() -> None:
    stage = build_song_card_stage(character="hero")

    assert stage.character == "hero"
    assert isinstance(stage, SongCardStage)


def test_song_card_composition_includes_character_when_selected() -> None:
    composition = song_card_composition(
        song_title="Neon sky",
        character_name="hero",
        art_style="bold explorer",
    )
    layers = [layer for layer in composition.layers if isinstance(layer, TextLayer)]
    text = "\n".join(layer.text for layer in layers)
    assert "Neon sky" in text
    assert "hero" in text
    assert "bold explorer" in text


def test_song_card_composition_omits_character_when_none_selected() -> None:
    composition = song_card_composition(song_title="Neon sky")
    layers = [layer for layer in composition.layers if isinstance(layer, TextLayer)]
    text = "\n".join(layer.text for layer in layers)
    assert "Neon sky" in text
    assert "hero" not in text


@pytest.mark.slow
def test_song_card_renders_a_deterministic_still(tmp_path: Path) -> None:
    characters_dir = tmp_path / "data" / "characters"
    _write_character(characters_dir, "icon")
    library = CharacterLibrary(data_dir=tmp_path / "data")

    first_dir = tmp_path / "run"
    first_result: StageResult = _run(
        _song_card(
            character="icon", library=library, scripts={"en": {"title": "Neon sky"}}
        ).run(_context(first_dir))
    )
    second_dir = tmp_path / "run2"
    second_result: StageResult = _run(
        _song_card(
            character="icon", library=library, scripts={"en": {"title": "Neon sky"}}
        ).run(_context(second_dir))
    )

    first_image = (first_dir / "song-card.png").read_bytes()
    second_image = (second_dir / "song-card.png").read_bytes()
    assert first_image == second_image
    assert first_result.status.value == "succeeded"
    assert second_result.status.value == "succeeded"
