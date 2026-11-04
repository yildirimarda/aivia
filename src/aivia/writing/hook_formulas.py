"""The hook formula library, loaded from packaged data.

A hook formula is a repeatable opening shape: a question, a statistic, a
contrarian claim and so on. Each formula records how the line is shaped, the
way that shape usually fails and the patterns that identify it in candidate
text. The library is data, not code, so a release can tune it without touching
the generator, and an offline test can prove every formula is usable.

Candidates are required to name distinct formulas. Repeating a formula hides
the alternatives a model was asked to explore, so the schema rejects a
duplicate instead of silently keeping two near-identical openings.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from aivia.core import paths
from aivia.core.errors import ConfigError, ValidationError

__all__ = [
    "DEFAULT_FORMULAS",
    "DEFAULT_FORMULAS_FILENAME",
    "DEFAULT_HOOK_FORMULAS",
    "DEFAULT_HOOK_FORMULAS_FILENAME",
    "DEFAULT_HOOK_FORMULA_LIBRARY",
    "FORMULA_LIBRARY_FILENAME",
    "HOOK_FORMULAS_FILENAME",
    "HOOK_FORMULAS_SCHEMA_VERSION",
    "HOOK_FORMULA_FILENAME",
    "HOOK_FORMULA_LIBRARY",
    "HOOK_FORMULA_LIBRARY_FILENAME",
    "LIBRARY_SCHEMA_VERSION",
    "FormulaLibrary",
    "HookFormula",
    "HookFormulaError",
    "HookFormulaLibrary",
    "HookFormulaLibraryError",
    "HookFormulaNotFoundError",
    "HookFormulaSet",
    "HookFormulas",
    "build_hook_formulas",
    "default_formulas",
    "default_hook_formulas",
    "default_hook_formulas_path",
    "distinct_formulas",
    "hook_formula_library",
    "hook_formula_library_path",
    "hook_formulas",
    "load_default_hook_formulas",
    "load_hook_formula_library",
    "load_hook_formulas",
    "normalise_formula",
    "package_hook_formulas_path",
    "require_distinct_formulas",
    "validate_hook_formulas",
]

HOOK_FORMULAS_FILENAME = "hook_formulas.yaml"
DEFAULT_HOOK_FORMULAS_FILENAME = HOOK_FORMULAS_FILENAME
HOOK_FORMULA_FILENAME = HOOK_FORMULAS_FILENAME
HOOK_FORMULA_LIBRARY_FILENAME = HOOK_FORMULAS_FILENAME
FORMULA_LIBRARY_FILENAME = HOOK_FORMULAS_FILENAME
DEFAULT_FORMULAS_FILENAME = HOOK_FORMULAS_FILENAME
HOOK_FORMULAS_SCHEMA_VERSION = 1
LIBRARY_SCHEMA_VERSION = HOOK_FORMULAS_SCHEMA_VERSION

_RESERVED_KEYS = frozenset({"schema_version", "version"})
_FORMULA_KEYS = (
    "shape",
    "template",
    "structure",
    "form",
    "failure_mode",
    "failure",
    "failure_modes",
    "match_patterns",
    "patterns",
    "matches",
    "match",
)
_LIBRARY_KEYS = (
    "formulas",
    "hook_formulas",
    "hooks",
    "entries",
    "items",
    "library",
    "catalog",
)


class HookFormulaError(ValidationError):
    """A hook candidate or formula library failed a formula rule."""

    code = "hook_formula"


HookFormulaLibraryError = HookFormulaError
HookFormulaNotFoundError = HookFormulaError


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _normalise_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def _formula_sequence(value: Any, *, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        if label == "aliases" and "," in text:
            return tuple(part.strip() for part in text.split(",") if part.strip())
        return (text,)
    if isinstance(value, (bytes, bytearray, Mapping)):
        raise ValueError(f"hook formula {label} must be a string or a sequence")
    if isinstance(value, Iterable):
        items = tuple(str(item).strip() for item in value if str(item).strip())
        return items
    raise ValueError(f"hook formula {label} must be a string or a sequence")


class HookFormula(BaseModel):
    """One repeatable opening shape and the way it usually fails.

    ``match_patterns`` are case-insensitive regular expressions. They let the
    writer classify a candidate by its text and let a test prove that the
    shipped examples really belong to the formula they claim.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str = Field(
        validation_alias=AliasChoices("id", "key", "slug", "formula", "name"),
    )
    name: str = Field(
        default="",
        validation_alias=AliasChoices("name", "title", "label"),
    )
    shape: str = Field(
        validation_alias=AliasChoices("shape", "template", "structure", "form"),
    )
    failure_mode: str = Field(
        validation_alias=AliasChoices(
            "failure_mode",
            "failure",
            "failure_modes",
            "risk",
            "pitfall",
        ),
    )
    match_patterns: tuple[str, ...] = Field(
        validation_alias=AliasChoices(
            "match_patterns",
            "patterns",
            "matches",
            "match",
        ),
    )
    aliases: tuple[str, ...] = ()
    description: str = Field(
        default="",
        validation_alias=AliasChoices("description", "notes", "summary"),
    )
    examples: tuple[str, ...] = Field(
        default=(),
        validation_alias=AliasChoices("examples", "samples"),
    )

    @field_validator("id", mode="before")
    @classmethod
    def _validate_id(cls, value: Any) -> str:
        text = _clean(value)
        if not text:
            raise ValueError("hook formula id must not be empty")
        return text

    @field_validator("name", "shape", "failure_mode", "description", mode="before")
    @classmethod
    def _clean_text(cls, value: Any, info: ValidationInfo) -> str:
        if isinstance(value, (list, tuple)):
            value = "; ".join(str(item) for item in value)
        text = _clean(value)
        if info.field_name in {"shape", "failure_mode"} and not text:
            raise ValueError(f"hook formula {info.field_name} must not be empty")
        return text

    @field_validator("match_patterns", "aliases", "examples", mode="before")
    @classmethod
    def _normalise_sequence(cls, value: Any, info: ValidationInfo) -> tuple[str, ...]:
        label = info.field_name or "values"
        return _formula_sequence(value, label=label)

    @field_validator("match_patterns")
    @classmethod
    def _validate_patterns(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("a hook formula must declare at least one match pattern")
        for pattern in value:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(
                    f"invalid hook formula match pattern {pattern!r}: {exc}"
                ) from exc
        return value

    @model_validator(mode="after")
    def _fill_name(self) -> HookFormula:
        if not self.name:
            self.name = self.id
        return self

    @property
    def key(self) -> str:
        """A case and separator insensitive key for this formula."""
        return _normalise_key(self.id)

    @property
    def patterns(self) -> tuple[str, ...]:
        """Alias for :attr:`match_patterns`."""
        return self.match_patterns

    @property
    def example(self) -> str:
        """The first example, or an empty string when none was supplied."""
        return self.examples[0] if self.examples else ""

    def match(self, text: str) -> str | None:
        """Return the first pattern matching *text*, or ``None``."""
        if not isinstance(text, str) or not text.strip():
            return None
        for pattern in self.match_patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                return pattern
        return None

    def matches(self, text: str) -> bool:
        """Whether any match pattern identifies *text* as this formula."""
        return self.match(text) is not None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "id": self.id,
            "name": self.name,
            "shape": self.shape,
            "failure_mode": self.failure_mode,
            "match_patterns": list(self.match_patterns),
            "aliases": list(self.aliases),
            "description": self.description,
            "examples": list(self.examples),
        }

    def to_prompt_dict(self) -> dict[str, Any]:
        """Return the fields a hook prompt needs, without examples."""
        return {
            "id": self.id,
            "shape": self.shape,
            "failure_mode": self.failure_mode,
            "match_patterns": list(self.match_patterns),
        }


def _candidate_formula(value: Any) -> Any:
    if isinstance(value, HookFormula):
        return value.id
    if isinstance(value, Mapping):
        for key in ("formula", "style", "pattern", "shape", "type", "id"):
            if key in value:
                return value[key]
        return ""
    formula = getattr(value, "formula", None)
    if formula is not None:
        return formula
    return value


def normalise_formula(value: Any) -> str:
    """Return a comparison key for a formula id, alias or candidate."""
    raw = _candidate_formula(value)
    if isinstance(raw, str):
        return _normalise_key(raw)
    return ""


def distinct_formulas(
    candidates: Iterable[Any],
    *,
    require_all: bool = False,
) -> tuple[str, ...]:
    """Return the formula keys on *candidates*, rejecting a repeat.

    Empty formulas are ignored unless *require_all* is set. The order of first
    appearance is kept so a caller can report which candidate used which shape.
    """
    names: list[str] = []
    seen: set[str] = set()
    for index, candidate in enumerate(candidates):
        key = normalise_formula(candidate)
        if not key:
            if require_all:
                raise HookFormulaError(
                    f"hook candidate {index + 1} must name a formula"
                )
            continue
        if key in seen:
            raise HookFormulaError(
                f"hook candidates must use distinct formulas: {key!r} is repeated"
            )
        seen.add(key)
        names.append(key)
    return tuple(names)


require_distinct_formulas = distinct_formulas
validate_hook_formulas = distinct_formulas


def _mapping_formulas(value: Mapping[str, Any]) -> list[Any]:
    formulas: list[Any] = []
    for key, spec in value.items():
        if isinstance(spec, Mapping):
            item = dict(spec)
            item.setdefault("id", str(key))
            formulas.append(item)
        else:
            formulas.append({"id": str(key), "shape": spec})
    return formulas


def _raw_formulas(value: object) -> list[Any]:
    if isinstance(value, HookFormulaLibrary):
        return list(value.formulas)
    if value is None:
        return []
    if isinstance(value, Mapping):
        data = dict(value)
        for key in _LIBRARY_KEYS:
            if key in data:
                inner = data[key]
                if isinstance(inner, Mapping):
                    return _mapping_formulas(inner)
                if isinstance(inner, (str, bytes, bytearray)):
                    return [inner]
                if isinstance(inner, Iterable):
                    return list(inner)
                return [inner]
        if any(key in data for key in _FORMULA_KEYS):
            return [data]
        data = {key: item for key, item in data.items() if key not in _RESERVED_KEYS}
        return _mapping_formulas(data)
    if isinstance(value, (str, bytes, bytearray)):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


class HookFormulaLibrary:
    """The set of known hook formulas, indexed by id and alias."""

    def __init__(
        self,
        formulas: Iterable[HookFormula | Mapping[str, Any]] = (),
        *,
        source: str | Path | None = None,
    ) -> None:
        by_key: dict[str, HookFormula] = {}
        aliases: dict[str, str] = {}
        for value in formulas:
            if isinstance(value, HookFormula):
                formula = value
            elif isinstance(value, Mapping):
                formula = HookFormula.model_validate(dict(value))
            else:
                raise ConfigError(
                    "a hook formula must be a mapping or a HookFormula",
                    details={"value": repr(value)},
                )
            key = formula.key
            if key in by_key:
                raise ConfigError(
                    f"duplicate hook formula id: {formula.id}",
                    details={"id": formula.id},
                )
            by_key[key] = formula
            for alias in (formula.id, formula.name, *formula.aliases):
                alias_key = _normalise_key(alias)
                if not alias_key:
                    continue
                existing = aliases.get(alias_key)
                if existing is not None and existing != key:
                    raise ConfigError(
                        f"duplicate hook formula alias: {alias}",
                        details={"alias": alias, "formula": formula.id},
                    )
                aliases[alias_key] = key
        if not by_key:
            raise ConfigError("a hook formula library must define at least one formula")
        self._formulas = tuple(sorted(by_key.values(), key=lambda item: item.key))
        self._by_key = by_key
        self._aliases = aliases
        self.source = None if source is None else Path(source).expanduser()

    @classmethod
    def from_mapping(cls, value: object) -> HookFormulaLibrary:
        """Build a library from a manifest mapping, list or existing library."""
        if isinstance(value, HookFormulaLibrary):
            return value
        return cls(_raw_formulas(value))

    @classmethod
    def from_file(cls, path: str | Path) -> HookFormulaLibrary:
        """Load a YAML or JSON formula manifest from *path*."""
        manifest = Path(path).expanduser()
        if not manifest.is_file():
            raise ConfigError(
                f"hook formula library not found: {manifest}",
                details={"path": str(manifest)},
            )
        text = manifest.read_text(encoding="utf-8")
        suffix = manifest.suffix.lower()
        try:
            if suffix in {".yaml", ".yml"}:
                payload: object = yaml.safe_load(text)
            elif suffix == ".json":
                payload = json.loads(text)
            else:
                raise ConfigError(
                    f"unsupported hook formula library extension: {suffix}",
                    details={"path": str(manifest)},
                )
        except (yaml.YAMLError, json.JSONDecodeError) as exc:
            raise ConfigError(
                f"invalid hook formula library: {manifest}",
                details={"path": str(manifest), "error": str(exc)},
            ) from exc
        library = cls(_raw_formulas(payload))
        library.source = manifest
        return library

    @classmethod
    def from_default(cls) -> HookFormulaLibrary:
        """Load the library shipped as package data."""
        return cls.from_file(default_hook_formulas_path())

    @property
    def formulas(self) -> tuple[HookFormula, ...]:
        """Every formula, sorted by id."""
        return self._formulas

    @property
    def entries(self) -> tuple[HookFormula, ...]:
        """Alias for :attr:`formulas`."""
        return self.formulas

    @property
    def ids(self) -> tuple[str, ...]:
        """Every formula id, sorted."""
        return tuple(formula.id for formula in self._formulas)

    @property
    def keys(self) -> tuple[str, ...]:
        """Every normalised formula key, sorted."""
        return tuple(formula.key for formula in self._formulas)

    @property
    def names(self) -> tuple[str, ...]:
        """Every display name, in id order."""
        return tuple(formula.name for formula in self._formulas)

    @property
    def by_id(self) -> dict[str, HookFormula]:
        """A copy of the id index."""
        return dict(self._by_key)

    def get(self, value: Any) -> HookFormula | None:
        """Return a formula by id, key or alias, or ``None``."""
        key = normalise_formula(value)
        if not key:
            return None
        return self._by_key.get(self._aliases.get(key, key))

    def require(self, value: Any) -> HookFormula:
        """Return a formula or raise :class:`HookFormulaError`."""
        formula = self.get(value)
        if formula is None:
            raise HookFormulaError(
                f"unknown hook formula: {value!r}",
                details={"formula": repr(value), "known": list(self.ids)},
            )
        return formula

    def matching(self, text: str) -> tuple[HookFormula, ...]:
        """Return every formula whose patterns match *text*."""
        return tuple(formula for formula in self._formulas if formula.matches(text))

    def match(self, text: str) -> HookFormula | None:
        """Return the first formula whose patterns match *text*."""
        for formula in self._formulas:
            if formula.matches(text):
                return formula
        return None

    classify = match

    def validate_candidates(
        self,
        candidates: Iterable[Any],
        *,
        require_all: bool = False,
        require_known: bool = False,
    ) -> tuple[HookFormula, ...]:
        """Validate candidate formulas and return the known formulas used.

        When *require_known* is set, every non-empty formula must resolve to a
        formula in this library. All unknown ids are collected before raising so
        the caller can see exactly which model-supplied ids were rejected.
        """
        resolved: list[str] = []
        for index, candidate in enumerate(candidates):
            key = normalise_formula(candidate)
            if not key:
                if require_all:
                    raise HookFormulaError(
                        f"hook candidate {index + 1} must name a formula"
                    )
                continue
            formula = self.get(key)
            resolved.append(formula.key if formula is not None else key)
        keys = distinct_formulas(resolved)
        used: list[HookFormula] = []
        unknown: list[str] = []
        for key in keys:
            formula = self.get(key)
            if formula is None:
                unknown.append(key)
                continue
            used.append(formula)
        if unknown and require_known:
            label = "formula" if len(unknown) == 1 else "formulas"
            raise HookFormulaError(
                f"unknown hook {label}: " + ", ".join(repr(key) for key in unknown),
                details={"unknown": unknown, "known": list(self.ids)},
            )
        return tuple(used)

    def prompt_lines(self) -> tuple[str, ...]:
        """Return one readable line per formula for a model prompt."""
        return tuple(
            f"- {formula.id}: {formula.shape} Avoid: {formula.failure_mode}"
            for formula in self._formulas
        )

    def __contains__(self, value: object) -> bool:
        return self.get(value) is not None

    def __iter__(self) -> Iterator[HookFormula]:
        return iter(self._formulas)

    def __len__(self) -> int:
        return len(self._formulas)


def default_hook_formulas_path() -> Path:
    """Return the packaged hook formula library path."""
    return paths.package_defaults_dir() / HOOK_FORMULAS_FILENAME


def load_hook_formulas(
    source: str
    | Path
    | Mapping[str, Any]
    | Iterable[Any]
    | HookFormulaLibrary
    | None = None,
) -> HookFormulaLibrary:
    """Load a formula library from package data, a file, a mapping or a library."""
    if isinstance(source, HookFormulaLibrary):
        return source
    if source is None:
        return HookFormulaLibrary.from_default()
    if isinstance(source, (str, Path)):
        return HookFormulaLibrary.from_file(source)
    return HookFormulaLibrary.from_mapping(source)


hook_formulas = load_hook_formulas
build_hook_formulas = load_hook_formulas


def load_default_hook_formulas() -> HookFormulaLibrary:
    """Load the library shipped with this release."""
    return HookFormulaLibrary.from_default()


default_hook_formulas = load_default_hook_formulas
default_formulas = load_default_hook_formulas
hook_formula_library = load_default_hook_formulas
load_hook_formula_library = load_hook_formulas
hook_formula_library_path = default_hook_formulas_path
package_hook_formulas_path = default_hook_formulas_path

HookFormulas = HookFormulaLibrary
HookFormulaSet = HookFormulaLibrary
FormulaLibrary = HookFormulaLibrary

DEFAULT_HOOK_FORMULAS = load_default_hook_formulas()
DEFAULT_FORMULAS = DEFAULT_HOOK_FORMULAS
DEFAULT_HOOK_FORMULA_LIBRARY = DEFAULT_HOOK_FORMULAS
HOOK_FORMULA_LIBRARY = DEFAULT_HOOK_FORMULAS
