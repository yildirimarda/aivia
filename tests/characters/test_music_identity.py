"""Character music identity: style prompts, personality lyrics and a jingle.

Every character carries a reusable musical identity: style prompts used to
generate music in the character's voice, lyrics expressing the character's
personality and a reusable jingle or intro. These are plain validated fields:
no AI call, no network request and no asset lookup happens here.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError as PydanticValidationError

from aivia.characters import (
    Character,
    CharacterLibrary,
    CharacterNotFoundError,
    load_character_file,
)

DEFAULT = {
    "name": "Mascot",
    "voice": "narrator-en",
    "art_style": "flat vector, bold outlines",
    "music_style_prompts": [
        "upbeat, major key, 120 bpm",
        "bright synth, catchy hook",
    ],
    "personality_lyrics": "I am Mascot, bright and bold, here to tell a story.",
    "jingle": "Mascot, Mascot, here to play!",
}


def write_character(root: Path, name: str, payload: Mapping[str, object]) -> Path:
    path = root / "characters" / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def write_asset(root: Path, name: str, content: bytes = b"fake-asset") -> Path:
    path = root / "characters" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_character_music_identity_defaults_are_empty() -> None:
    character = Character.model_validate({"name": "Mascot"})

    assert character.music_style_prompts == []
    assert character.personality_lyrics == ""
    assert character.jingle == ""


def test_character_music_identity_parses_every_field() -> None:
    character = Character.model_validate(DEFAULT)

    assert character.music_style_prompts == [
        "upbeat, major key, 120 bpm",
        "bright synth, catchy hook",
    ]
    assert character.personality_lyrics == (
        "I am Mascot, bright and bold, here to tell a story."
    )
    assert character.jingle == "Mascot, Mascot, here to play!"


def test_character_music_identity_round_trips() -> None:
    character = Character.model_validate(DEFAULT)
    restored = Character.model_validate(character.model_dump(mode="json"))

    assert restored == character


def test_character_music_identity_rejects_unknown_fields() -> None:
    with pytest.raises(PydanticValidationError) as caught:
        Character.model_validate({**DEFAULT, "bogus_music_field": True})

    assert any(
        part == "bogus_music_field" for part in caught.value.errors()[0].get("loc", ())
    )


def test_character_music_style_prompts_strip_and_reject_blanks() -> None:
    character = Character.model_validate(
        {**DEFAULT, "music_style_prompts": ["  upbeat  ", "bright synth"]}
    )
    assert character.music_style_prompts == ["upbeat", "bright synth"]

    with pytest.raises(PydanticValidationError) as caught:
        Character.model_validate({**DEFAULT, "music_style_prompts": ["ok", ""]})
    assert "music style prompts" in str(caught.value)

    with pytest.raises(PydanticValidationError) as caught:
        Character.model_validate({**DEFAULT, "music_style_prompts": ["ok", "   "]})
    assert "music style prompts" in str(caught.value)


def test_character_personality_lyrics_strips() -> None:
    character = Character.model_validate(
        {**DEFAULT, "personality_lyrics": "  bright and bold  "}
    )
    assert character.personality_lyrics == "bright and bold"


def test_character_jingle_strips() -> None:
    character = Character.model_validate({**DEFAULT, "jingle": "  Mascot jingle  "})
    assert character.jingle == "Mascot jingle"


def test_character_music_identity_defaults_to_empty_strings() -> None:
    character = Character.model_validate(
        {"name": "Mascot", "music_style_prompts": ["one prompt"]}
    )
    assert character.personality_lyrics == ""
    assert character.jingle == ""


def test_loader_loads_character_with_music_identity(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    write_character(data_dir, "mascot", DEFAULT)

    library = CharacterLibrary(data_dir=data_dir)
    character = library.load("mascot")

    assert character.name == "Mascot"
    assert character.music_style_prompts == DEFAULT["music_style_prompts"]
    assert character.personality_lyrics == DEFAULT["personality_lyrics"]
    assert character.jingle == DEFAULT["jingle"]
    assert "mascot" in library
    assert library.names() == ("mascot",)


def test_loader_load_character_file_directly_with_music_identity(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    path = write_character(data_dir, "mascot", DEFAULT)

    character = load_character_file(path)

    assert character.personality_lyrics == DEFAULT["personality_lyrics"]
    assert character.jingle == DEFAULT["jingle"]


def test_loader_unknown_character_with_music_identity(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    write_character(data_dir, "mascot", DEFAULT)

    library = CharacterLibrary(data_dir=data_dir)

    with pytest.raises(CharacterNotFoundError) as caught:
        library.load("missing")

    assert caught.value.code == "character_not_found"
    assert "unknown character: missing" in str(caught.value)
