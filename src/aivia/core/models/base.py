"""Base model for persisted records.

Every record carries a mandatory schema version and creation/update timestamps
so stored data can be read back and migrated. Serialization is canonical:
object keys are sorted, separators are compact and timestamps are rendered in
UTC, so the same values always produce the same JSON text.

Unknown fields are kept rather than rejected. A record written by a newer
release must remain readable by an older one; the reader is warned about each
field it does not understand so typos and schema drift stay visible.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "CURRENT_MODEL_SCHEMA_VERSION",
    "CURRENT_SCHEMA_VERSION",
    "ModelBase",
    "RecordModel",
    "UnknownFieldWarning",
    "canonical_json",
    "utcnow",
]

CURRENT_MODEL_SCHEMA_VERSION = 1
"""The record schema version written by this release."""

CURRENT_SCHEMA_VERSION = CURRENT_MODEL_SCHEMA_VERSION
"""Alias for :data:`CURRENT_MODEL_SCHEMA_VERSION`."""


class UnknownFieldWarning(UserWarning):
    """A record carried a field this model version does not know about."""


def utcnow() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(UTC)


def canonical_json(
    value: Any,
    *,
    indent: int | None = None,
    ensure_ascii: bool = False,
) -> str:
    """Serialize a JSON-compatible *value* deterministically.

    Keys are sorted at every level and the compact separators avoid incidental
    whitespace. ``allow_nan`` is disabled because NaN and infinity are not
    valid JSON and would make the output depend on the encoder.
    """
    separators = (",", ":") if indent is None else None
    return json.dumps(
        value,
        sort_keys=True,
        separators=separators,
        ensure_ascii=ensure_ascii,
        allow_nan=False,
    )


class ModelBase(BaseModel):
    """Base class for every persisted record.

    ``schema_version`` defaults to the version written by this release and may
    only be positive. ``created_at`` and ``updated_at`` default to the current
    UTC time; supplied values are normalized to UTC.

    Unknown input fields are preserved and reported with
    :class:`UnknownFieldWarning` instead of failing validation. This keeps
    records written by a newer release readable while making accidental typos
    visible.
    """

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        validate_assignment=True,
    )

    schema_version: int = Field(default=CURRENT_MODEL_SCHEMA_VERSION, ge=1)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="before")
    @classmethod
    def _warn_unknown_fields(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        known = set(cls.model_fields)
        for field in cls.model_fields.values():
            if field.alias:
                known.add(field.alias)
        for key in data:
            if isinstance(key, str) and key not in known:
                warnings.warn(
                    f"unknown field {key!r} on {cls.__name__}; keeping it",
                    UnknownFieldWarning,
                    stacklevel=2,
                )
        return data

    @field_validator("created_at", "updated_at")
    @classmethod
    def _normalize_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> Self:
        """Validate a mapping into a record, warning about unknown fields."""
        return cls.model_validate(dict(values))

    def to_json(
        self,
        *,
        indent: int | None = None,
        ensure_ascii: bool = False,
        **kwargs: Any,
    ) -> str:
        """Return this record as canonical JSON."""
        data = self.model_dump(mode="json", **kwargs)
        return canonical_json(data, indent=indent, ensure_ascii=ensure_ascii)

    def model_dump_json(self, **kwargs: Any) -> str:
        """Return canonical JSON, overriding Pydantic's field-order output."""
        indent = kwargs.pop("indent", None)
        ensure_ascii = kwargs.pop("ensure_ascii", False)
        data = self.model_dump(mode="json", **kwargs)
        return canonical_json(data, indent=indent, ensure_ascii=ensure_ascii)


RecordModel = ModelBase
"""Alias for :class:`ModelBase` for callers that prefer the record wording."""
