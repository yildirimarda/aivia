"""Model base contract: versioning, timestamps, canonical JSON and warnings."""

from __future__ import annotations

import json
import warnings
from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from aivia.core.models import (
    CURRENT_MODEL_SCHEMA_VERSION,
    ModelBase,
    RecordModel,
    UnknownFieldWarning,
    canonical_json,
)


class Example(ModelBase):
    title: str = "untitled"
    count: int = 0


class Child(ModelBase):
    label: str = "child"
    zeta: int = 0
    alpha: int = 0


class Parent(ModelBase):
    child: Child = Child()


class Sorted(ModelBase):
    zeta: str = ""
    alpha: str = ""
    middle: int = 0


def test_schema_version_defaults_to_the_current_version() -> None:
    record = Example()

    assert record.schema_version == CURRENT_MODEL_SCHEMA_VERSION == 1


def test_schema_version_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Example(schema_version=0)


def test_schema_version_can_be_raised_for_a_newer_record() -> None:
    assert Example(schema_version=2).schema_version == 2


def test_timestamps_default_to_aware_utc_times() -> None:
    before = datetime.now(UTC)
    record = Example()
    after = datetime.now(UTC)

    assert record.created_at.tzinfo is UTC
    assert record.updated_at.tzinfo is UTC
    assert before <= record.created_at <= after
    assert before <= record.updated_at <= after


def test_supplied_timestamps_are_normalized_to_utc() -> None:
    local = timezone(timedelta(hours=2))
    supplied = datetime(2026, 1, 2, 12, 30, tzinfo=local)

    record = Example(created_at=supplied, updated_at=supplied)

    assert record.created_at == datetime(2026, 1, 2, 10, 30, tzinfo=UTC)
    assert record.created_at.tzinfo is UTC


def test_naive_timestamps_are_treated_as_utc() -> None:
    record = Example(created_at=datetime(2026, 1, 2, 10, 30))

    assert record.created_at == datetime(2026, 1, 2, 10, 30, tzinfo=UTC)
    assert record.created_at.tzinfo is UTC


def test_json_is_canonical_with_sorted_compact_keys() -> None:
    record = Sorted(zeta="last", alpha="first", middle=2)

    text = record.to_json()

    assert text == record.model_dump_json()
    assert ", " not in text
    assert ': "' not in text
    keys = list(json.loads(text))
    assert keys == sorted(keys)


def test_json_does_not_depend_on_input_field_order() -> None:
    stamp = datetime(2026, 1, 2, 10, 30, tzinfo=UTC)
    first = Sorted.model_validate(
        {
            "zeta": "z",
            "alpha": "a",
            "middle": 1,
            "created_at": stamp,
            "updated_at": stamp,
        }
    )
    second = Sorted.model_validate(
        {
            "middle": 1,
            "alpha": "a",
            "updated_at": stamp,
            "created_at": stamp,
            "zeta": "z",
        }
    )

    assert first.to_json() == second.to_json()


def test_json_is_stable_across_a_round_trip() -> None:
    record = Example(title="round trip", count=3)
    restored = Example.model_validate_json(record.to_json())

    assert restored.to_json() == record.to_json()


def test_nested_models_are_serialized_canonically() -> None:
    parent = Parent(child=Child(label="nested", zeta=1, alpha=2))

    payload = json.loads(parent.to_json())

    assert list(payload["child"]) == sorted(payload["child"])
    assert payload["child"]["label"] == "nested"


def test_canonical_json_rejects_non_finite_numbers() -> None:
    with pytest.raises(ValueError):
        canonical_json({"value": float("nan")})


def test_unknown_field_warns_and_is_kept() -> None:
    with pytest.warns(UnknownFieldWarning, match="unknown field 'titel'") as caught:
        record = Example.model_validate({"title": "a", "titel": "typo"})

    assert len(caught) == 1
    assert record.model_extra == {"titel": "typo"}
    assert json.loads(record.to_json())["titel"] == "typo"


def test_known_fields_do_not_warn() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Example(title="a", count=1)

    assert caught == []


def test_unknown_nested_field_warns_for_the_nested_model() -> None:
    with pytest.warns(UnknownFieldWarning, match="on Child"):
        Parent.model_validate({"child": {"label": "x", "unknown": True}})


def test_unknown_fields_warn_when_validated_from_a_mapping() -> None:
    with pytest.warns(UnknownFieldWarning, match="unknown field 'extra'"):
        record = Example.from_dict({"title": "a", "extra": 1})

    assert record.model_extra == {"extra": 1}


def test_record_model_alias_points_at_the_same_base() -> None:
    assert RecordModel is ModelBase
