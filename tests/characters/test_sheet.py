"""Character sheets: generated once with a vision model, editable by hand.

A sheet is produced from a character's reference images and stored as YAML.
Loading a sheet is a pure file read -- no vision call happens -- and
re-analysis happens only when regeneration is explicitly requested.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from PIL import Image
from pydantic import ValidationError as PydanticValidationError

from aivia.characters import Character
from aivia.characters.sheet import (
    CURRENT_CHARACTER_SHEET_SCHEMA_VERSION,
    CharacterSheet,
    CharacterSheetExistsError,
    CharacterSheetNotFoundError,
    CharacterSheetSchemaError,
    CharacterSheetVisionOutput,
    generate_character_sheet,
    load_character_sheet,
)
from aivia.core.errors import ValidationError
from aivia.llm.structured import StructuredOutputHelper
from aivia.providers.protocols.llm import LLMResponse, LLMUsage


def _run(coroutine) -> Any:
    return asyncio.run(coroutine)


def write_png(path: Path, size: tuple[int, int] = (32, 32), color: str = "red") -> Path:
    Image.new("RGB", size, color).save(path, format="PNG")
    return path


class FakeVisionProvider:
    """An async LLM provider that returns one canned sheet payload."""

    def __init__(
        self,
        payload: dict[str, Any] | str,
        *,
        name: str = "acme",
        model: str = "vision-model",
    ) -> None:
        self.payload = payload
        self.name = name
        self.model = model
        self.requests: list[Any] = []

    async def complete(self, request: Any) -> LLMResponse:
        self.requests.append(request)
        content = (
            self.payload if isinstance(self.payload, str) else json.dumps(self.payload)
        )
        return LLMResponse(
            content=content,
            provider=self.name,
            model=self.model,
            usage=LLMUsage(input_tokens=8, output_tokens=4),
        )


def character_with_images(
    data_dir: Path, name: str, images: list[tuple[str, str]]
) -> Character:
    """Write reference images and return a character referencing them."""
    characters_dir = data_dir / "characters"
    characters_dir.mkdir(parents=True, exist_ok=True)
    for file_name, color in images:
        write_png(characters_dir / file_name, color=color)
    return Character.model_validate(
        {
            "name": name,
            "reference_images": [file_name for file_name, _ in images],
        }
    )


def sheet_path(data_dir: Path, name: str) -> Path:
    return data_dir / "characters" / "sheets" / f"{name}.yaml"


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


def test_sheet_schema_parses_every_field() -> None:
    sheet = CharacterSheet.model_validate(
        {
            "name": "Mascot",
            "version": CURRENT_CHARACTER_SHEET_SCHEMA_VERSION,
            "description": "A red mascot with round ears.",
            "art_style": "flat vector",
            "pose": "idle",
            "expression": "happy",
            "reference_images": ["face.png", "turnaround.png"],
            "colors": ["#ff0000", "#0000ff"],
            "source_hash": "abc123",
            "provider": "acme",
            "model": "vision-model",
            "analyzed_at": "2026-01-01T00:00:00+00:00",
        }
    )

    assert sheet.name == "Mascot"
    assert sheet.version == CURRENT_CHARACTER_SHEET_SCHEMA_VERSION
    assert sheet.description == "A red mascot with round ears."
    assert sheet.art_style == "flat vector"
    assert sheet.pose == "idle"
    assert sheet.expression == "happy"
    assert sheet.reference_images == ["face.png", "turnaround.png"]
    assert sheet.colors == ["#ff0000", "#0000ff"]
    assert sheet.source_hash == "abc123"
    assert sheet.provider == "acme"
    assert sheet.model == "vision-model"
    assert sheet.analyzed_at == "2026-01-01T00:00:00+00:00"


def test_sheet_schema_defaults_are_empty_and_versioned() -> None:
    sheet = CharacterSheet.model_validate({"name": "Mascot"})

    assert sheet.name == "Mascot"
    assert sheet.version == CURRENT_CHARACTER_SHEET_SCHEMA_VERSION
    assert sheet.description == ""
    assert sheet.art_style == ""
    assert sheet.pose == ""
    assert sheet.expression == ""
    assert sheet.reference_images == []
    assert sheet.colors == []
    assert sheet.source_hash == ""
    assert sheet.provider == ""
    assert sheet.model == ""
    assert sheet.analyzed_at == ""


def test_sheet_schema_rejects_unknown_fields() -> None:
    with pytest.raises(PydanticValidationError) as caught:
        CharacterSheet.model_validate({"name": "Mascot", "bogus_key": True})
    assert any(part == "bogus_key" for part in caught.value.errors()[0].get("loc", ()))


def test_sheet_schema_requires_a_name() -> None:
    with pytest.raises(PydanticValidationError):
        CharacterSheet.model_validate({})
    with pytest.raises(PydanticValidationError):
        CharacterSheet.model_validate({"name": "   "})


def test_sheet_schema_strips_text_fields() -> None:
    sheet = CharacterSheet.model_validate(
        {
            "name": "  Mascot  ",
            "description": "  A red mascot  ",
            "art_style": "  flat vector  ",
            "provider": "  acme  ",
        }
    )
    assert sheet.name == "Mascot"
    assert sheet.description == "A red mascot"
    assert sheet.art_style == "flat vector"
    assert sheet.provider == "acme"


# ---------------------------------------------------------------------------
# Generate and load tests
# ---------------------------------------------------------------------------


def test_generate_character_sheet_writes_a_sheet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    data_dir = tmp_path / "aivia" / "data"
    character = character_with_images(data_dir, "mascot", [("face.png", "red")])
    provider = FakeVisionProvider(
        {
            "description": "A red mascot with round ears.",
            "art_style": "flat vector",
            "pose": "idle",
            "expression": "happy",
        }
    )
    helper = StructuredOutputHelper(provider, CharacterSheetVisionOutput)

    sheet = _run(generate_character_sheet(character, helper=helper, data_dir=data_dir))

    assert sheet.name == "mascot"
    assert sheet.version == CURRENT_CHARACTER_SHEET_SCHEMA_VERSION
    assert sheet.description == "A red mascot with round ears."
    assert sheet.art_style == "flat vector"
    assert sheet.pose == "idle"
    assert sheet.expression == "happy"
    assert sheet.reference_images == ["face.png"]
    assert sheet.colors == ["#ff0000"]
    assert sheet.source_hash
    assert sheet.provider == "acme"
    assert sheet.model == "vision-model"
    assert sheet.analyzed_at

    path = sheet_path(data_dir, "mascot")
    assert path.is_file()
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert document["name"] == "mascot"
    assert document["description"] == "A red mascot with round ears."
    # One vision call total.
    assert len(provider.requests) == 1


def test_generate_character_sheet_records_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    data_dir = tmp_path / "aivia" / "data"
    character = character_with_images(data_dir, "mascot", [("face.png", "green")])
    provider = FakeVisionProvider(
        {"description": "A green mascot.", "art_style": "pixel art"}, name="beta"
    )
    helper = StructuredOutputHelper(provider, CharacterSheetVisionOutput)

    sheet = _run(generate_character_sheet(character, helper=helper, data_dir=data_dir))

    assert sheet.provider == "beta"
    assert sheet.model == "vision-model"
    assert sheet.description == "A green mascot."
    assert sheet.art_style == "pixel art"


def test_load_character_sheet_reads_the_file_without_vision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    data_dir = tmp_path / "aivia" / "data"
    character = character_with_images(data_dir, "mascot", [("face.png", "red")])
    provider = FakeVisionProvider(
        {"description": "A red mascot.", "art_style": "flat vector"}
    )
    helper = StructuredOutputHelper(provider, CharacterSheetVisionOutput)

    sheet = _run(generate_character_sheet(character, helper=helper, data_dir=data_dir))
    request_count = len(provider.requests)

    loaded = load_character_sheet("mascot", data_dir=data_dir)

    assert loaded == sheet
    assert loaded.description == "A red mascot."
    assert loaded.colors == ["#ff0000"]
    # Loading made no additional vision call.
    assert len(provider.requests) == request_count


def test_regenerating_without_force_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    data_dir = tmp_path / "aivia" / "data"
    character = character_with_images(data_dir, "mascot", [("face.png", "red")])
    provider = FakeVisionProvider(
        {"description": "A red mascot.", "art_style": "flat vector"}
    )
    helper = StructuredOutputHelper(provider, CharacterSheetVisionOutput)

    _run(generate_character_sheet(character, helper=helper, data_dir=data_dir))

    with pytest.raises(CharacterSheetExistsError) as caught:
        _run(generate_character_sheet(character, helper=helper, data_dir=data_dir))
    assert caught.value.code == "character_sheet_exists"
    assert "mascot" in str(caught.value)
    # The original sheet is untouched.
    assert load_character_sheet("mascot", data_dir=data_dir).description == (
        "A red mascot."
    )


def test_regenerating_with_force_reanalyzes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    data_dir = tmp_path / "aivia" / "data"
    character = character_with_images(data_dir, "mascot", [("face.png", "red")])
    provider = FakeVisionProvider(
        {"description": "A red mascot.", "art_style": "flat vector"}
    )
    helper = StructuredOutputHelper(provider, CharacterSheetVisionOutput)

    _run(generate_character_sheet(character, helper=helper, data_dir=data_dir))
    first_requests = len(provider.requests)

    provider.payload = {
        "description": "A revised red mascot.",
        "art_style": "flat vector",
    }
    sheet = _run(
        generate_character_sheet(
            character, helper=helper, force=True, data_dir=data_dir
        )
    )

    assert len(provider.requests) > first_requests
    assert sheet.description == "A revised red mascot."
    assert load_character_sheet("mascot", data_dir=data_dir).description == (
        "A revised red mascot."
    )


def test_sheet_is_editable_by_hand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    data_dir = tmp_path / "aivia" / "data"
    character = character_with_images(data_dir, "mascot", [("face.png", "red")])
    provider = FakeVisionProvider(
        {"description": "A red mascot.", "art_style": "flat vector"}
    )
    helper = StructuredOutputHelper(provider, CharacterSheetVisionOutput)

    _run(generate_character_sheet(character, helper=helper, data_dir=data_dir))

    path = sheet_path(data_dir, "mascot")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["description"] = "A hand-edited red mascot."
    document["pose"] = "waving"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    loaded = load_character_sheet("mascot", data_dir=data_dir)
    assert loaded.description == "A hand-edited red mascot."
    assert loaded.pose == "waving"
    # The hand edit did not trigger any new vision call.
    assert len(provider.requests) == 1


def test_generate_without_vision_is_offline_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    data_dir = tmp_path / "aivia" / "data"
    character = character_with_images(data_dir, "mascot", [("face.png", "red")])

    sheet = _run(generate_character_sheet(character, data_dir=data_dir))

    assert sheet.description == ""
    assert sheet.art_style == ""
    assert sheet.colors == ["#ff0000"]
    assert sheet.reference_images == ["face.png"]
    assert sheet.provider == ""
    assert sheet.model == ""
    assert sheet_path(data_dir, "mascot").is_file()


def test_load_character_sheet_not_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    data_dir = tmp_path / "aivia" / "data"

    with pytest.raises(CharacterSheetNotFoundError) as caught:
        load_character_sheet("missing", data_dir=data_dir)
    assert caught.value.code == "character_sheet_not_found"
    assert caught.value.details["name"] == "missing"
    assert "characters/sheets/missing.yaml" in caught.value.details["file"]


def test_sheet_schema_error_names_file_and_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    data_dir = tmp_path / "aivia" / "data"
    (data_dir / "characters" / "sheets").mkdir(parents=True)
    bad = data_dir / "characters" / "sheets" / "mascot.yaml"
    bad.write_text("name: Mascot\nbogus_key: true\n", encoding="utf-8")

    with pytest.raises(CharacterSheetSchemaError) as caught:
        load_character_sheet("mascot", data_dir=data_dir)
    assert caught.value.code == "character_sheet_schema"
    assert caught.value.details["file"] == str(bad)
    assert "bogus_key" in str(caught.value)


def test_library_names_and_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    data_dir = tmp_path / "aivia" / "data"
    character = character_with_images(data_dir, "mascot", [("face.png", "red")])
    provider = FakeVisionProvider(
        {"description": "A red mascot.", "art_style": "flat vector"}
    )
    helper = StructuredOutputHelper(provider, CharacterSheetVisionOutput)

    from aivia.characters.sheet import CharacterSheetLibrary

    library = CharacterSheetLibrary(data_dir=data_dir)
    assert library.names() == ()
    assert library.exists("mascot") is False

    _run(generate_character_sheet(character, helper=helper, data_dir=data_dir))

    assert library.names() == ("mascot",)
    assert library.exists("mascot") is True


def test_generate_rejects_invalid_provider_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    data_dir = tmp_path / "aivia" / "data"
    character = character_with_images(data_dir, "mascot", [("face.png", "red")])
    provider = FakeVisionProvider("not json at all")
    helper = StructuredOutputHelper(provider, CharacterSheetVisionOutput)

    with pytest.raises((ValidationError, PydanticValidationError)):
        _run(generate_character_sheet(character, helper=helper, data_dir=data_dir))


def test_sheet_vision_output_defaults() -> None:
    output = CharacterSheetVisionOutput()
    assert output.description == ""
    assert output.art_style == ""
    assert output.pose == ""
    assert output.expression == ""
