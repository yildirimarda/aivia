"""Character versioning: sheet, pose and voice changes bump the version.

A run pins the character version it used so it keeps reproducing the same
output even after the character file on disk is revised to a newer version.
"""

from __future__ import annotations

from aivia.characters import CURRENT_CHARACTER_SCHEMA_VERSION, Character
from aivia.characters.sheet import (
    CURRENT_CHARACTER_SHEET_SCHEMA_VERSION,
    CharacterSheet,
)
from aivia.core.models import Run


def test_bump_version_returns_a_copy_with_version_incremented() -> None:
    character = Character.model_validate(
        {"name": "Mascot", "version": 1, "voice": "narrator-en"}
    )

    bumped = character.bump_version()

    assert bumped.version == 2
    assert bumped.name == "Mascot"
    assert bumped.voice == "narrator-en"
    # The original character is left untouched.
    assert character.version == 1
    assert character.voice == "narrator-en"
    assert bumped is not character


def test_voice_change_bumps_version() -> None:
    character = Character.model_validate(
        {"name": "Mascot", "version": 1, "voice": "narrator-en"}
    )

    revised = character.model_copy(update={"voice": "narrator-fr"}).bump_version()

    assert revised.version == 2
    assert revised.voice == "narrator-fr"
    assert character.voice == "narrator-en"


def test_pose_change_bumps_version() -> None:
    character = Character.model_validate(
        {"name": "Mascot", "version": 1, "pose_library": {"idle": "idle.png"}}
    )

    revised = character.model_copy(
        update={"pose_library": {"idle": "idle.png", "wave": "wave.png"}}
    ).bump_version()

    assert revised.version == 2
    assert revised.pose_library == {"idle": "idle.png", "wave": "wave.png"}
    assert character.pose_library == {"idle": "idle.png"}


def test_sheet_change_bumps_character_version() -> None:
    character = Character.model_validate({"name": "Mascot", "version": 1})
    sheet = CharacterSheet.model_validate(
        {
            "name": "Mascot",
            "version": CURRENT_CHARACTER_SHEET_SCHEMA_VERSION,
            "pose": "idle",
            "expression": "happy",
        }
    )

    # A revised sheet publishes a new character version.
    revised_sheet = sheet.model_copy(update={"pose": "waving"})
    revised_character = character.bump_version()

    assert revised_character.version == 2
    assert revised_sheet.version == CURRENT_CHARACTER_SHEET_SCHEMA_VERSION
    assert revised_sheet.pose == "waving"
    assert character.version == 1


def test_default_character_version_is_current_schema_version() -> None:
    character = Character.model_validate({"name": "Mascot"})

    assert character.version == CURRENT_CHARACTER_SCHEMA_VERSION
    assert character.schema_version == CURRENT_CHARACTER_SCHEMA_VERSION


def test_run_pins_the_character_version() -> None:
    run = Run(character_version="1")

    assert run.character_version == "1"

    # A later run can pin the newer character version while the earlier run
    # keeps reproducing output from the older one.
    newer = Run(character_version="2")
    assert newer.character_version == "2"
    assert run.character_version == "1"


def test_run_character_version_defaults_to_empty() -> None:
    run = Run()

    assert run.character_version == ""
