"""Character schema and loader: file and field errors for every failure."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError as PydanticValidationError

from aivia.characters import (
    CURRENT_CHARACTER_SCHEMA_VERSION,
    Character,
    CharacterError,
    CharacterLibrary,
    CharacterNotFoundError,
    CharacterSchemaError,
    load_character_file,
)
from aivia.core.errors import ValidationError

DEFAULT = {
    "name": "Mascot",
    "voice": "narrator-en",
    "art_style": "flat vector, bold outlines",
    "negatives": ["photorealistic", "blurry"],
    "reference_images": ["face.png", "turnaround.png"],
    "pose_library": {"idle": "idle.png", "wave": "wave.png"},
    "brand_kit": "default",
    "ownership_required": True,
    "allow_real_person_likeness": False,
    "resemblance_warning": True,
}


def write_character(root: Path, name: str, payload: dict) -> Path:
    path = root / "characters" / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def write_asset(root: Path, name: str, content: bytes = b"fake-asset") -> Path:
    path = root / "characters" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_character_schema_parses_every_field() -> None:
    character = Character.model_validate(DEFAULT)

    assert character.name == "Mascot"
    assert character.version == CURRENT_CHARACTER_SCHEMA_VERSION
    assert character.schema_version == 1
    assert character.voice == "narrator-en"
    assert character.art_style == "flat vector, bold outlines"
    assert character.negatives == ["photorealistic", "blurry"]
    assert character.reference_images == ["face.png", "turnaround.png"]
    assert character.pose_library == {"idle": "idle.png", "wave": "wave.png"}
    assert character.brand_kit == "default"
    assert character.ownership_required is True
    assert character.allow_real_person_likeness is False
    assert character.resemblance_warning is True


def test_character_schema_defaults_are_empty_and_versioned() -> None:
    character = Character.model_validate({"name": "Mascot"})

    assert character.name == "Mascot"
    assert character.version == CURRENT_CHARACTER_SCHEMA_VERSION
    assert character.voice == ""
    assert character.art_style == ""
    assert character.negatives == []
    assert character.reference_images == []
    assert character.pose_library == {}
    assert character.brand_kit == ""
    assert character.ownership_required is True
    assert character.allow_real_person_likeness is False
    assert character.resemblance_warning is True


def test_character_schema_round_trips() -> None:
    character = Character.model_validate(DEFAULT)
    restored = Character.model_validate(character.model_dump(mode="json"))

    assert restored == character


def test_character_schema_rejects_unknown_fields() -> None:
    with pytest.raises(PydanticValidationError) as caught:
        Character.model_validate({**DEFAULT, "bogus": True})

    assert any(part == "bogus" for part in caught.value.errors()[0].get("loc", ()))


def test_character_schema_requires_a_name() -> None:
    with pytest.raises(PydanticValidationError):
        Character.model_validate({})

    with pytest.raises(PydanticValidationError):
        Character.model_validate({"name": "   "})


def test_character_schema_rejects_version_below_one() -> None:
    with pytest.raises(PydanticValidationError):
        Character.model_validate({"name": "Mascot", "version": 0})

    with pytest.raises(PydanticValidationError):
        Character.model_validate({"name": "Mascot", "version": -1})


def test_character_schema_strips_text_fields() -> None:
    character = Character.model_validate(
        {
            "name": "  Mascot  ",
            "voice": "  narrator-en  ",
            "art_style": "  flat vector  ",
            "brand_kit": "  default  ",
        }
    )

    assert character.name == "Mascot"
    assert character.voice == "narrator-en"
    assert character.art_style == "flat vector"
    assert character.brand_kit == "default"


def test_character_schema_rejects_blank_pose_names() -> None:
    with pytest.raises(PydanticValidationError) as caught:
        Character.model_validate(
            {**DEFAULT, "pose_library": {"idle": "idle.png", "  ": "wave.png"}}
        )
    assert "pose names must be non-empty" in str(caught.value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("negatives", ["photorealistic", "  ", "blurry"]),
        ("reference_images", ["face.png", "", "turnaround.png"]),
    ],
)
def test_character_schema_rejects_blank_list_entries(
    field: str, value: list[str]
) -> None:
    with pytest.raises(PydanticValidationError) as caught:
        Character.model_validate({**DEFAULT, field: value})

    assert field in str(caught.value)


def test_loader_loads_a_character_from_the_user_directory(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    write_character(data_dir, "mascot", DEFAULT)
    write_asset(data_dir, "face.png")
    write_asset(data_dir, "turnaround.png")
    write_asset(data_dir, "idle.png")
    write_asset(data_dir, "wave.png")

    library = CharacterLibrary(data_dir=data_dir)
    character = library.load("mascot")

    assert character.name == "Mascot"
    assert character.reference_images == ["face.png", "turnaround.png"]
    assert character.pose_library == {"idle": "idle.png", "wave": "wave.png"}
    assert "mascot" in library
    assert library.names() == ("mascot",)


def test_loader_load_character_file_directly(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    path = write_character(data_dir, "mascot", DEFAULT)
    for asset in ("face.png", "turnaround.png", "idle.png", "wave.png"):
        write_asset(data_dir, asset)

    character = load_character_file(path)

    assert character.name == "Mascot"
    assert character.reference_images == ["face.png", "turnaround.png"]


def test_loader_unknown_character_names_the_file(tmp_path: Path) -> None:
    library = CharacterLibrary(data_dir=tmp_path / "data")

    with pytest.raises(CharacterNotFoundError) as caught:
        library.load("missing")

    assert caught.value.code == "character_not_found"
    assert caught.value.details["name"] == "missing"
    assert "characters" in caught.value.details["file"]
    assert caught.value.details["file"].endswith("characters/missing.yaml")
    assert "unknown character: missing" in str(caught.value)


def test_loader_invalid_yaml_names_the_file(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    path = write_character(data_dir, "mascot", DEFAULT)
    path.write_text("name: Mascot\nvoice: [unclosed\n", encoding="utf-8")

    library = CharacterLibrary(data_dir=data_dir)

    with pytest.raises(CharacterError) as caught:
        library.load("mascot")

    assert caught.value.code == "character"
    assert caught.value.details["file"] == str(path)
    assert "invalid YAML" in str(caught.value)


def test_loader_missing_required_field_names_file_and_field(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    path = write_character(data_dir, "mascot", {"voice": "narrator-en"})

    library = CharacterLibrary(data_dir=data_dir)

    with pytest.raises(CharacterSchemaError) as caught:
        library.load("mascot")

    assert caught.value.code == "character_schema"
    assert isinstance(caught.value, ValidationError)
    assert caught.value.details["file"] == str(path)
    assert caught.value.details["field"] == "name"
    assert "name" in str(caught.value)
    assert str(path) in str(caught.value)


def test_loader_unknown_key_names_file_and_field(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    payload = dict(DEFAULT)
    payload["bogus_key"] = True
    path = write_character(data_dir, "mascot", payload)

    library = CharacterLibrary(data_dir=data_dir)

    with pytest.raises(CharacterSchemaError) as caught:
        library.load("mascot")

    assert caught.value.details["file"] == str(path)
    assert caught.value.details["field"] == "bogus_key"
    assert "bogus_key" in str(caught.value)


def test_loader_missing_reference_image_names_file_and_field(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    payload = dict(DEFAULT)
    payload["reference_images"] = ["face.png", "missing.png"]
    path = write_character(data_dir, "mascot", payload)
    write_asset(data_dir, "face.png")

    library = CharacterLibrary(data_dir=data_dir)

    with pytest.raises(CharacterSchemaError) as caught:
        library.load("mascot")

    assert caught.value.details["file"] == str(path)
    assert caught.value.details["field"] == "reference_images.1"
    assert caught.value.details["asset"] == "missing.png"
    assert "missing.png" in str(caught.value)


def test_loader_missing_pose_asset_names_file_and_field(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    payload = dict(DEFAULT)
    payload["reference_images"] = []
    payload["pose_library"] = {"idle": "idle.png", "wave": "missing.png"}
    path = write_character(data_dir, "mascot", payload)
    write_asset(data_dir, "idle.png")

    library = CharacterLibrary(data_dir=data_dir)

    with pytest.raises(CharacterSchemaError) as caught:
        library.load("mascot")

    assert caught.value.details["file"] == str(path)
    assert caught.value.details["field"] == "pose_library.wave"
    assert caught.value.details["asset"] == "missing.png"
    assert "wave" in str(caught.value)


def test_loader_assets_are_resolved_relative_to_the_character_file(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    sub = data_dir / "characters" / "series"
    sub.mkdir(parents=True)
    character_file = sub / "mascot.yaml"
    character_file.write_text(
        yaml.safe_dump(
            {
                **DEFAULT,
                "reference_images": ["face.png"],
                "pose_library": {},
            }
        ),
        encoding="utf-8",
    )
    (sub / "face.png").write_bytes(b"asset")

    library = CharacterLibrary(data_dir=data_dir)
    character = library.load("series/mascot")

    assert character.name == "Mascot"
    assert character.reference_images == ["face.png"]


def test_loader_character_name_cannot_escape_the_directory(tmp_path: Path) -> None:
    library = CharacterLibrary(data_dir=tmp_path / "data")

    with pytest.raises(CharacterError):
        library.load("../secret")
    with pytest.raises(CharacterError):
        library.load("/etc/passwd")


def test_loader_default_uses_the_user_data_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    library = CharacterLibrary()

    assert library.data_dir == tmp_path / "aivia" / "data"
    assert library.characters_dir == tmp_path / "aivia" / "data" / "characters"
