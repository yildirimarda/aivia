"""Composition schema migrations: upgrade older documents, refuse newer ones."""

from __future__ import annotations

import json
from typing import Any

import pytest

from aivia.compose import (
    CURRENT_COMPOSITION_SCHEMA_VERSION,
    Composition,
    CompositionIOError,
    CompositionMigration,
    CompositionMigrationError,
    CompositionMigrationResult,
    MigrationResult,
    load_composition,
    loads_composition,
    migrate_composition,
    schema_version_of,
)
from aivia.compose.migrations import MIGRATIONS


def _document(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "id": "demo",
        "canvas": "1080x1920",
        "duration": 4.0,
        "fps": 30,
        "layers": [
            {
                "id": "title",
                "type": "text",
                "text": "Merhaba",
                "start": 0,
                "duration": 2,
            }
        ],
    }
    values.update(overrides)
    return values


def _load(document: dict[str, Any]) -> Composition:
    return loads_composition(json.dumps(document), format="json")


def test_an_unversioned_document_is_upgraded_before_validation() -> None:
    composition = _load(_document())

    assert composition.schema_version == CURRENT_COMPOSITION_SCHEMA_VERSION
    assert composition.schema_version == 1
    assert composition.id == "demo"


def test_an_explicit_version_zero_is_upgraded_before_validation() -> None:
    composition = _load(_document(schema_version=0))

    assert composition.schema_version == CURRENT_COMPOSITION_SCHEMA_VERSION


def test_the_version_alias_is_also_upgraded() -> None:
    composition = _load(_document(version=0))

    assert composition.schema_version == CURRENT_COMPOSITION_SCHEMA_VERSION


def test_a_current_document_loads_unchanged() -> None:
    composition = _load(_document(schema_version=CURRENT_COMPOSITION_SCHEMA_VERSION))

    assert composition.schema_version == CURRENT_COMPOSITION_SCHEMA_VERSION


def test_a_newer_document_is_rejected_with_a_clear_error() -> None:
    newer = CURRENT_COMPOSITION_SCHEMA_VERSION + 1

    with pytest.raises(
        CompositionIOError,
        match=f"newer than supported version {CURRENT_COMPOSITION_SCHEMA_VERSION}",
    ):
        _load(_document(schema_version=newer))
    with pytest.raises(CompositionIOError, match="upgrade aivia"):
        load_composition(_document(schema_version=newer))
    built = Composition.model_validate(_document(schema_version=newer))
    with pytest.raises(CompositionIOError, match="newer than supported version"):
        load_composition(built)


@pytest.mark.parametrize("version", [-1, "one", 1.5, True, None])
def test_an_invalid_schema_version_is_rejected(version: object) -> None:
    with pytest.raises(CompositionIOError, match="invalid schema_version"):
        _load(_document(schema_version=version))


def test_migrate_composition_preserves_unknown_fields_and_reports_the_steps() -> None:
    document = _document()
    document["future_field"] = {"enabled": True}

    result = migrate_composition(document)

    assert isinstance(result, CompositionMigrationResult)
    assert MigrationResult is CompositionMigrationResult
    assert result.from_version == 0
    assert result.to_version == CURRENT_COMPOSITION_SCHEMA_VERSION
    assert result.changed
    assert [step.description for step in result.steps] == [
        "add the mandatory schema_version key"
    ]
    assert result.values["schema_version"] == CURRENT_COMPOSITION_SCHEMA_VERSION
    assert result.values["future_field"] == {"enabled": True}
    assert next(iter(result.values)) == "schema_version"
    assert document.get("schema_version") is None


def test_migrate_composition_applies_each_step_in_order() -> None:
    def add_version(values: dict[str, Any]) -> None:
        values["schema_version"] = 1

    def rename_duration(values: dict[str, Any]) -> None:
        values["duration_seconds"] = values.pop("duration")
        values["schema_version"] = 2

    migrations = (
        CompositionMigration(0, 1, "add version", add_version),
        CompositionMigration(1, 2, "rename duration", rename_duration),
    )

    result = migrate_composition(
        {"duration": 5, "unknown": {"keep": True}},
        migrations=migrations,
        target_version=2,
    )

    assert result.from_version == 0
    assert result.to_version == 2
    assert [step.description for step in result.steps] == [
        "add version",
        "rename duration",
    ]
    assert result.values == {
        "duration_seconds": 5,
        "unknown": {"keep": True},
        "schema_version": 2,
    }


def test_migrate_composition_reports_a_missing_upgrade_path() -> None:
    with pytest.raises(CompositionMigrationError, match="no composition migration"):
        migrate_composition(
            {"schema_version": 1},
            migrations=MIGRATIONS,
            target_version=2,
        )


def test_schema_version_of_treats_an_absent_key_as_unversioned() -> None:
    assert schema_version_of({}) == 0
    assert schema_version_of({"version": 0}) == 0
    assert schema_version_of({"schema_version": 1}) == 1


def test_a_newer_version_is_rejected_by_migrate_composition_directly() -> None:
    newer = CURRENT_COMPOSITION_SCHEMA_VERSION + 1

    with pytest.raises(CompositionMigrationError) as caught:
        migrate_composition({"schema_version": newer})

    assert "newer than supported version" in str(caught.value)
    assert caught.value.details == {
        "schema_version": newer,
        "supported": CURRENT_COMPOSITION_SCHEMA_VERSION,
    }
