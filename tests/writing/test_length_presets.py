"""Reading time presets and the default rate load from packaged data."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aivia.core import paths
from aivia.core.errors import ConfigError
from aivia.writing import (
    DEFAULT_LENGTH_PRESETS_FILENAME,
    DEFAULT_WORDS_PER_SECOND,
    LENGTH_PRESETS,
    LENGTH_PRESETS_FILENAME,
    LengthPresetError,
    LengthPresets,
    LengthRange,
    coerce_length_range,
    default_length_presets_path,
    load_length_presets,
    resolve_words_per_second,
)


def _write_manifest(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "length_presets.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_the_package_ships_the_reading_time_presets_as_data() -> None:
    path = default_length_presets_path()

    assert path == paths.package_defaults_dir() / LENGTH_PRESETS_FILENAME
    assert DEFAULT_LENGTH_PRESETS_FILENAME == LENGTH_PRESETS_FILENAME
    assert path.is_file()

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["default_words_per_second"] == 2.5
    assert set(payload["presets"]) == {
        "micro",
        "short",
        "medium",
        "long",
        "epic",
    }


def test_module_constants_come_from_the_packaged_manifest() -> None:
    loaded = load_length_presets()

    assert loaded.source == default_length_presets_path()
    assert loaded.default_words_per_second == 2.5
    assert loaded.default_words_per_second == DEFAULT_WORDS_PER_SECOND
    assert dict(LENGTH_PRESETS) == dict(loaded.presets)
    assert LENGTH_PRESETS["short"] == (0.0, 30.0)
    assert LENGTH_PRESETS["epic"] == (90.0, 600.0)
    assert resolve_words_per_second("en") == 2.5


def test_presets_load_from_a_mapping_and_resolve_names_case_insensitively() -> None:
    loaded = load_length_presets(
        {
            "schema_version": 1,
            "default_words_per_second": 3.5,
            "presets": {
                "Tiny": {"min_seconds": 0, "max_seconds": 5},
                "Talk": {"target_seconds": 40},
            },
        }
    )

    assert isinstance(loaded, LengthPresets)
    assert loaded.default_words_per_second == 3.5
    assert loaded.default_rate == 3.5
    assert loaded.names == ("talk", "tiny")
    assert loaded["tiny"] == (0.0, 5.0)
    assert loaded.range("TINY") == LengthRange(0.0, 5.0)
    assert loaded.ranges["talk"] == LengthRange(32.0, 48.0)
    assert loaded.to_dict()["presets"]["tiny"] == {
        "min_seconds": 0.0,
        "max_seconds": 5.0,
    }


def test_presets_load_from_a_file_and_keep_the_source_path(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "default_words_per_second": 4.0,
            "presets": {
                "blink": [0, 2],
                "spot": {"min": 1, "max": 4},
            },
        },
    )

    loaded = load_length_presets(path)

    assert loaded.source == path
    assert loaded.default_words_per_second == 4.0
    assert loaded.presets == {"blink": (0.0, 2.0), "spot": (1.0, 4.0)}


def test_a_direct_preset_mapping_is_accepted_with_the_default_rate() -> None:
    loaded = load_length_presets({"short": [0, 30], "long": [45, 180]})

    assert loaded.presets == {"short": (0.0, 30.0), "long": (45.0, 180.0)}
    assert loaded.default_words_per_second == DEFAULT_WORDS_PER_SECOND


def test_preset_entries_can_be_a_list_of_named_entries() -> None:
    loaded = load_length_presets(
        [
            {"name": "short", "min_seconds": 0, "max_seconds": 30},
            ["long", 45, 180],
        ]
    )

    assert loaded.presets == {"short": (0.0, 30.0), "long": (45.0, 180.0)}


def test_loaded_presets_drive_length_coercion() -> None:
    assert coerce_length_range("short") == LengthRange(*LENGTH_PRESETS["short"])
    assert coerce_length_range("epic") == LengthRange(*LENGTH_PRESETS["epic"])


def test_invalid_manifests_are_rejected() -> None:
    with pytest.raises(LengthPresetError, match="at least one preset"):
        load_length_presets({"schema_version": 1})
    with pytest.raises(LengthPresetError, match="positive"):
        load_length_presets(
            {"default_words_per_second": 0, "presets": {"short": [0, 30]}}
        )
    with pytest.raises(LengthPresetError, match="maximum"):
        load_length_presets({"presets": {"short": [30, 0]}})
    with pytest.raises(LengthPresetError, match="unsupported"):
        load_length_presets({"schema_version": 2, "presets": {"short": [0, 30]}})
    with pytest.raises(LengthPresetError, match="duplicate"):
        load_length_presets({"presets": {"short": [0, 30], "Short": [0, 20]}})
    with pytest.raises(ConfigError):
        load_length_presets({"presets": {"short": [0]}})


def test_a_missing_or_unreadable_manifest_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(LengthPresetError, match="not found"):
        load_length_presets(tmp_path / "missing.yaml")

    path = tmp_path / "broken.yaml"
    path.write_text("presets: [", encoding="utf-8")
    with pytest.raises(LengthPresetError, match="invalid"):
        load_length_presets(path)

    path = tmp_path / "manifest.txt"
    path.write_text("presets: {}", encoding="utf-8")
    with pytest.raises(LengthPresetError, match="unsupported"):
        load_length_presets(path)
