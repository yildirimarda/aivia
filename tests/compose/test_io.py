"""Composition I/O: YAML, JSON and the exported JSON Schema."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from aivia.compose import (
    COMPOSITION_JSON_SCHEMA_DIALECT,
    COMPOSITION_JSON_SCHEMA_TITLE,
    Composition,
    CompositionIOError,
    build_composition_schema,
    composition_from_json,
    composition_from_yaml,
    composition_json_schema,
    composition_schema,
    composition_to_dict,
    composition_to_json,
    composition_to_mapping,
    composition_to_yaml,
    dump_composition,
    dump_composition_schema,
    export_composition_schema,
    load_composition,
    load_composition_file,
    loads_composition,
    read_composition,
    read_composition_file,
    save_composition,
    save_composition_file,
    save_composition_schema,
    write_composition,
)


def _composition() -> Composition:
    return Composition.model_validate(
        {
            "id": "demo",
            "schema_version": 1,
            "canvas": "1080x1920",
            "duration": 12.0,
            "frame_rate": 25,
            "layers": [
                {
                    "id": "background",
                    "type": "image",
                    "source": "bg.png",
                    "fit": "cover",
                    "start": 0,
                    "end": 12,
                },
                {
                    "id": "title",
                    "type": "text",
                    "text": "Merhaba dünya",
                    "font_family": "Lato",
                    "font_size": 64,
                    "align": "center",
                    "start": 1,
                    "duration": 3,
                    "keyframes": {
                        "opacity": [
                            {"time": 0, "value": 0},
                            {"time": 0.5, "value": 1},
                        ]
                    },
                },
            ],
            "tracks": [
                {
                    "id": "music",
                    "source": "music.mp3",
                    "role": "music",
                    "volume_db": -3,
                    "fade_in": 0.5,
                }
            ],
            "effects": [
                "fade",
                {
                    "id": "vignette",
                    "type": "vignette",
                    "start": 2,
                    "end": 8,
                    "params": {"strength": 0.4},
                },
            ],
        }
    )


@pytest.mark.parametrize(("suffix", "format"), [(".yaml", "yaml"), (".json", "json")])
def test_a_saved_composition_loads_back_unchanged(
    tmp_path: Path, suffix: str, format: str
) -> None:
    composition = _composition()
    target = tmp_path / f"composition{suffix}"

    written = save_composition(composition, target)

    assert written == target
    assert target.is_file()
    assert load_composition(target) == composition
    assert load_composition(str(target)) == composition
    assert load_composition_file(target) == composition
    assert load_composition(target, format=format) == composition


def test_dumping_is_deterministic_and_keeps_unicode() -> None:
    composition = _composition()

    first_yaml = dump_composition(composition, format="yaml")
    second_yaml = dump_composition(composition, format="yml")
    first_json = dump_composition(composition, format="json")
    second_json = dump_composition(composition, format="json")

    assert first_yaml == second_yaml
    assert first_json == second_json
    assert first_yaml.endswith("\n")
    assert first_json.endswith("\n")
    assert "Merhaba dünya" in first_yaml
    assert "Merhaba dünya" in first_json


def test_a_document_can_be_loaded_from_text_or_a_mapping() -> None:
    composition = _composition()
    yaml_text = dump_composition(composition, format="yaml")
    json_text = dump_composition(composition, format="json")

    assert loads_composition(yaml_text) == composition
    assert loads_composition(json_text) == composition
    assert loads_composition(yaml_text, format="yaml") == composition
    assert loads_composition(json_text, format="json") == composition
    assert load_composition(yaml_text) == composition
    assert load_composition(json_text) == composition
    assert load_composition(composition_to_dict(composition)) == composition
    assert load_composition(composition) is composition


def test_format_specific_helpers_and_read_write_aliases(tmp_path: Path) -> None:
    composition = _composition()

    assert composition_from_yaml(composition_to_yaml(composition)) == composition
    assert composition_from_json(composition_to_json(composition)) == composition
    assert read_composition(composition_to_yaml(composition)) == composition
    assert export_composition_schema() == build_composition_schema()

    target = tmp_path / "alias.yaml"
    write_composition(composition, target)

    assert read_composition_file(target) == composition


def test_saving_creates_parents_and_supports_an_explicit_format(
    tmp_path: Path,
) -> None:
    composition = _composition()
    target = tmp_path / "nested" / "composition"

    save_composition(composition, target, format="json")

    assert json.loads(target.read_text(encoding="utf-8"))["id"] == "demo"
    assert load_composition(target) == composition


def test_a_mapping_can_be_saved_directly(tmp_path: Path) -> None:
    target = tmp_path / "composition.yaml"

    save_composition_file(composition_to_mapping(_composition()), target)

    assert load_composition(target) == _composition()


def test_composition_to_dict_is_json_compatible_and_uses_field_names() -> None:
    payload = composition_to_dict(_composition())

    assert payload["duration_seconds"] == 12.0
    assert payload["fps"] == 25
    assert payload["canvas"] == {
        "width": 1080,
        "height": 1920,
        "background": "#000000",
    }
    assert payload["layers"][0]["start_seconds"] == 0.0
    assert "end_seconds" in payload["layers"][0]
    assert "duration_seconds" not in payload["layers"][0]
    assert json.loads(json.dumps(payload)) == payload


def test_a_missing_file_is_reported_as_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_composition(tmp_path / "missing.yaml")
    with pytest.raises(FileNotFoundError):
        load_composition(str(tmp_path / "missing.yaml"))


def test_unsupported_formats_and_extensions_are_rejected(tmp_path: Path) -> None:
    composition = _composition()

    with pytest.raises(CompositionIOError, match="unsupported composition format"):
        dump_composition(composition, format="toml")
    with pytest.raises(CompositionIOError, match="unsupported composition file"):
        save_composition(composition, tmp_path / "composition.toml")
    with pytest.raises(CompositionIOError, match="unsupported composition file"):
        load_composition(tmp_path / "composition.toml")


@pytest.mark.parametrize(
    ("text", "format", "message"),
    [
        ("{not json", "json", "invalid JSON composition"),
        ("canvas: [", "yaml", "invalid YAML composition"),
        ("", None, "composition document is empty"),
        ("[]", "json", "composition document must be a mapping"),
    ],
)
def test_invalid_documents_are_reported(
    text: str, format: str | None, message: str
) -> None:
    with pytest.raises(CompositionIOError, match=message):
        loads_composition(text, format=format)


def test_an_invalid_composition_still_fails_schema_validation() -> None:
    with pytest.raises(ValidationError):
        loads_composition('{"canvas": "1080x1920"}', format="json")


def test_the_json_schema_describes_every_composition_field() -> None:
    schema = build_composition_schema()
    properties = schema["properties"]

    assert schema["$schema"] == COMPOSITION_JSON_SCHEMA_DIALECT
    assert schema["title"] == COMPOSITION_JSON_SCHEMA_TITLE
    assert schema["type"] == "object"
    assert set(properties) == {
        "id",
        "schema_version",
        "canvas",
        "duration_seconds",
        "fps",
        "layers",
        "audio_tracks",
        "effects",
        "subtitles",
        "subtitle_file",
        "platform",
        "safe_area",
        "loudness_lufs",
    }
    assert properties["schema_version"]["default"] == 1
    assert properties["layers"]["type"] == "array"
    assert {"Canvas", "Layer", "AudioTrack", "Effect"} <= set(schema["$defs"])
    assert {"SubtitleTrack", "SubtitleCue"} <= set(schema["$defs"])
    assert schema == composition_json_schema() == composition_schema()


def test_the_schema_dump_is_stable_and_can_be_saved(tmp_path: Path) -> None:
    schema = build_composition_schema()

    assert dump_composition_schema() == dump_composition_schema()
    assert dump_composition_schema() == dump_composition_schema(schema)
    assert json.loads(dump_composition_schema()) == schema

    target = tmp_path / "nested" / "composition.schema.json"
    written = save_composition_schema(target)

    assert written == target
    assert target.read_text(encoding="utf-8") == dump_composition_schema()
    assert json.loads(target.read_text(encoding="utf-8")) == schema
