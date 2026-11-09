"""Series continuity log: facts that keep a series consistent across outputs.

A series is a YAML file in ``<data dir>/series/<name>.yaml`` next to the
characters it references. Every fact carries a category, a key and a value, so
a writer can be reminded of what is already established and warned when new
output contradicts it. This module is plain validation and logic only: no AI
call and no network request happens here.

The series carries a ``version`` so later items can bump it when the canon
changes while runs pin an older one.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)
from pydantic import (
    ValidationError as PydanticValidationError,
)

from aivia.core import paths
from aivia.core.errors import ConfigError, ValidationError

__all__ = [
    "CURRENT_SERIES_SCHEMA_VERSION",
    "Series",
    "SeriesContradiction",
    "SeriesError",
    "SeriesFact",
    "SeriesLibrary",
    "SeriesNotFoundError",
    "SeriesSchemaError",
    "check_contradictions",
    "find_contradictions",
    "load_series_file",
]

CURRENT_SERIES_SCHEMA_VERSION = 1
"""The series schema version written by this release."""

_SERIES_DIR = "series"
"""The user data subdirectory holding series continuity logs."""

_NEGATION = re.compile(
    r"\b(?:not|never|no longer|doesn.t|does not|is not|aren.t|are not|"
    r"wasn.t|was not|weren.t|were not|hasn.t|has not|haven.t|have not|"
    r"won.t|will not|can.t|cannot|ain.t)\b",
    re.IGNORECASE,
)


class SeriesError(ConfigError):
    """A series file or its facts could not be loaded."""

    code = "series"


class SeriesNotFoundError(SeriesError):
    """No series file exists for the requested name."""

    code = "series_not_found"


class SeriesSchemaError(ValidationError):
    """A series file failed validation.

    Details always carry ``file`` and ``field``.
    """

    code = "series_schema"

    def __init__(
        self,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        merged: dict[str, Any] = dict(details) if details else {}
        merged.setdefault("file", "")
        merged.setdefault("field", "")
        super().__init__(message, details=merged)


class SeriesFact(BaseModel):
    """One established fact about a series.

    ``category`` groups related facts (for example ``character``, ``setting`` or
    ``plot``), ``key`` names the thing established and ``value`` is what the
    series has established about it.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    category: str = Field(min_length=1)
    key: str = Field(min_length=1)
    value: str = Field(min_length=1)
    source: str = ""
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("category", "key", "value", "source", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value

    @property
    def label(self) -> str:
        """A short, human readable label for this fact."""
        return f"{self.category}.{self.key}"


class Series(BaseModel):
    """One series continuity log: established facts plus metadata."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    name: str = Field(min_length=1)
    version: int = Field(default=CURRENT_SERIES_SCHEMA_VERSION, ge=1)
    facts: list[SeriesFact] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("name", mode="before")
    @classmethod
    def _strip_name(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value

    @property
    def schema_version(self) -> int:
        """The schema version this series was written against."""
        return self.version

    def bump_version(self) -> Series:
        """Return a copy of this series with its version incremented by one."""
        return self.model_copy(
            update={
                "version": self.version + 1,
                "updated_at": datetime.now(UTC),
            }
        )

    def fact_for(self, category: str, key: str) -> SeriesFact | None:
        """Return the fact for *category*/*key*, or ``None`` when absent."""
        for fact in self.facts:
            if fact.category == category and fact.key == key:
                return fact
        return None

    def add_or_replace_fact(self, fact: SeriesFact) -> Series:
        """Return a copy with *fact* added or replacing an existing same key."""
        facts = [
            existing
            for existing in self.facts
            if not (existing.category == fact.category and existing.key == fact.key)
        ]
        facts.append(fact)
        return self.model_copy(
            update={"facts": facts, "updated_at": datetime.now(UTC)}
        )

    def remove_fact(self, category: str, key: str) -> Series:
        """Return a copy without the fact for *category*/*key*."""
        facts = [
            existing
            for existing in self.facts
            if not (existing.category == category and existing.key == key)
        ]
        return self.model_copy(
            update={"facts": facts, "updated_at": datetime.now(UTC)}
        )


class SeriesContradiction(BaseModel):
    """One contradiction between established canon and new text."""

    model_config = ConfigDict(extra="forbid")

    fact: SeriesFact
    reason: str
    severity: str = Field(default="warning")

    @field_validator("severity", mode="before")
    @classmethod
    def _normalise_severity(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    def to_dict(self) -> dict[str, Any]:
        """Return the contradiction as a JSON-compatible mapping."""
        return {
            "category": self.fact.category,
            "key": self.fact.key,
            "value": self.fact.value,
            "reason": self.reason,
            "severity": self.severity,
        }


def _normalise(text: str) -> str:
    return " ".join(text.split()).casefold()


def _negated(value: str, text: str) -> bool:
    """Return whether *text* negates *value* with a negation word nearby."""
    normalised = _normalise(text)
    target = _normalise(value)
    if not target:
        return False
    for match in _NEGATION.finditer(normalised):
        start = match.end()
        window = normalised[start : start + len(target) + 2]
        if window.startswith(target) or window.startswith(" " + target):
            return True
    return False


def check_contradictions(series: Series, text: str) -> list[SeriesContradiction]:
    """Return every established fact that *text* contradicts.

    A fact is contradicted when the text negates its established value (for
    example a fact ``mascot.color = blue`` is contradicted by "the mascot is not
    blue"). Negation matching is case-insensitive and whitespace-normalised.
    """
    contradictions: list[SeriesContradiction] = []
    for fact in series.facts:
        if _negated(fact.value, text):
            contradictions.append(
                SeriesContradiction(
                    fact=fact,
                    reason=(
                        f"established {fact.label} is {fact.value!r} but the text "
                        f"negates it"
                    ),
                )
            )
    return contradictions


find_contradictions = check_contradictions


class SeriesLibrary:
    """Series continuity logs in the user data directory."""

    def __init__(
        self,
        *,
        data_dir: str | Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        if data_dir is None:
            self.data_dir = paths.data_dir(environ)
        else:
            self.data_dir = Path(data_dir).expanduser().resolve()
        self.series_dir = self.data_dir / _SERIES_DIR

    def series_path(self, name: str) -> Path:
        """Return the file for *name*: ``<data dir>/series/<name>.yaml``."""
        return self.series_dir / f"{self._normalise_name(name)}.yaml"

    @staticmethod
    def _normalise_name(name: str) -> str:
        if not isinstance(name, str) or not name.strip():
            raise SeriesError("series name must be a non-empty string")
        text = name.strip().replace("\\", "/")
        if any(part in {"", ".", ".."} for part in text.split("/")):
            raise SeriesError(
                f"series name may not escape the series directory: {name}",
                details={"name": text},
            )
        return text

    def names(self) -> tuple[str, ...]:
        """Return every series name with a file in the series directory."""
        if not self.series_dir.is_dir():
            return ()
        names: list[str] = []
        for path in sorted(self.series_dir.glob("*.yaml")):
            names.append(path.stem)
        return tuple(names)

    def resolve(self, name: str) -> Path:
        """Return the series file for *name*.

        Raises :class:`SeriesNotFoundError`, naming the file, when it does
        not exist.
        """
        path = self.series_path(name)
        if not path.is_file():
            raise SeriesNotFoundError(
                f"unknown series: {name}",
                details={"name": name, "file": str(path)},
            )
        return path

    def load(self, name: str) -> Series:
        """Load and validate *name*."""
        path = self.resolve(name)
        return load_series_file(path)

    def save(self, series: Series) -> Path:
        """Write *series* to its file, creating parent directories as needed.

        The original ``created_at`` is preserved and ``updated_at`` is refreshed
        to now, so the stored file reflects when it was last written.
        """
        self.series_dir.mkdir(parents=True, exist_ok=True)
        path = self.series_path(series.name)
        now = datetime.now(UTC)
        updated = series.model_copy(update={"updated_at": now})
        payload = {
            "name": updated.name,
            "version": updated.version,
            "created_at": updated.created_at,
            "updated_at": updated.updated_at,
            "facts": [fact.model_dump(mode="python") for fact in updated.facts],
        }
        text = yaml.safe_dump(payload, sort_keys=False)
        try:
            path.write_text(text, encoding="utf-8")
        except OSError as exc:
            raise SeriesError(
                f"cannot write series file {path}: {exc}",
                details={"file": str(path)},
            ) from exc
        return path

    def delete(self, name: str) -> Path:
        """Remove the series file for *name* and return its path.

        Raises :class:`SeriesNotFoundError` when no file exists.
        """
        path = self.resolve(name)
        try:
            path.unlink()
        except OSError as exc:
            raise SeriesError(
                f"cannot delete series file {path}: {exc}",
                details={"file": str(path)},
            ) from exc
        return path

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        try:
            self.resolve(name)
        except SeriesError:
            return False
        return True


def load_series_file(path: str | Path) -> Series:
    """Read *path*, validate it and return a :class:`Series`.

    Raises :class:`~aivia.characters.series.SeriesSchemaError`, naming the file
    and field, when the document is invalid.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SeriesError(
            f"cannot read series file {path}: {exc}",
            details={"file": str(path)},
        ) from exc
    try:
        payload: Any = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SeriesError(
            f"invalid YAML in series file {path}: {exc}",
            details={"file": str(path)},
        ) from exc
    if payload is None:
        payload = {}
    if not isinstance(payload, Mapping):
        raise SeriesError(
            f"series root must be a mapping: {path}",
            details={"file": str(path)},
        )
    try:
        series = Series.model_validate(dict(payload))
    except PydanticValidationError as exc:
        raise _schema_error(exc, path) from exc
    return series


def _schema_error(exc: PydanticValidationError, path: Path) -> SeriesSchemaError:
    errors = [dict(error) for error in exc.errors()]
    first = errors[0] if errors else {}
    loc = cast(tuple[object, ...], first.get("loc", ()) or ())
    field = ".".join(str(part) for part in loc)
    message = str(first.get("msg", "invalid value"))
    raise SeriesSchemaError(
        f"series file {path} has invalid {field or 'root'}: {message}",
        details={
            "file": str(path),
            "field": field,
            "errors": errors,
        },
    )
