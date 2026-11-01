"""Load and save compositions as YAML or JSON and export their JSON Schema.

Compositions are plain data. This module is the only place that knows which
text format a document is stored in: callers hand it a path or a document and
get a validated :class:`~aivia.compose.schema.Composition` back. Serialisation
is deterministic so a saved composition can be diffed, hashed or compared byte
for byte.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml

from aivia.compose.schema import Composition

__all__ = [
    "COMPOSITION_JSON_SCHEMA_DIALECT",
    "COMPOSITION_JSON_SCHEMA_TITLE",
    "CompositionIOError",
    "build_composition_schema",
    "composition_from_json",
    "composition_from_yaml",
    "composition_json_schema",
    "composition_schema",
    "composition_to_dict",
    "composition_to_json",
    "composition_to_mapping",
    "composition_to_yaml",
    "dump_composition",
    "dump_composition_schema",
    "export_composition_schema",
    "load_composition",
    "load_composition_file",
    "loads_composition",
    "read_composition",
    "read_composition_file",
    "save_composition",
    "save_composition_file",
    "save_composition_schema",
    "write_composition",
    "write_composition_file",
]

COMPOSITION_JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"
"""The JSON Schema dialect used by the exported composition schema."""

COMPOSITION_JSON_SCHEMA_TITLE = "AIVIA composition"
"""The title written into the exported composition schema."""

Format = Literal["json", "yaml"]

_YAML_SUFFIXES = {".yaml", ".yml"}
_JSON_SUFFIXES = {".json"}
_FORMAT_ALIASES: dict[str, Format] = {
    "json": "json",
    "yaml": "yaml",
    "yml": "yaml",
}


class CompositionIOError(ValueError):
    """A composition document could not be read, parsed or written."""


def _normalise_format(value: str) -> Format:
    key = value.strip().lower()
    try:
        return _FORMAT_ALIASES[key]
    except KeyError:
        raise CompositionIOError(
            f"unsupported composition format {value!r}; use 'yaml' or 'json'"
        ) from None


def _format_for_path(path: Path) -> Format | None:
    suffix = path.suffix.lower()
    if suffix in _YAML_SUFFIXES:
        return "yaml"
    if suffix in _JSON_SUFFIXES:
        return "json"
    if not suffix:
        return None
    raise CompositionIOError(
        f"unsupported composition file extension {suffix!r}; use .yaml, .yml or .json"
    )


def _infer_format(text: str) -> Format:
    stripped = text.lstrip()
    if stripped.startswith(("{", "[")):
        return "json"
    return "yaml"


def _looks_like_document(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    if any(character in value for character in "\r\n"):
        return True
    if stripped.startswith(("{", "[", "---")):
        return True
    return ":" in stripped


def _as_composition(value: Composition | Mapping[str, Any]) -> Composition:
    if isinstance(value, Composition):
        return value
    if isinstance(value, Mapping):
        return Composition.model_validate(dict(value))
    raise TypeError("expected a Composition or a mapping")


def composition_to_dict(composition: Composition | Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-compatible mapping for *composition*.

    Field names are used instead of aliases and unset optional fields are
    omitted, so the result round-trips through either supported text format.
    """
    return _as_composition(composition).model_dump(
        mode="json",
        by_alias=False,
        exclude_none=True,
    )


composition_to_mapping = composition_to_dict
"""Alias for :func:`composition_to_dict`."""


def composition_to_yaml(composition: Composition | Mapping[str, Any]) -> str:
    """Render *composition* as deterministic YAML text."""
    return dump_composition(composition, format="yaml")


def composition_to_json(composition: Composition | Mapping[str, Any]) -> str:
    """Render *composition* as deterministic JSON text."""
    return dump_composition(composition, format="json")


def dump_composition(
    composition: Composition | Mapping[str, Any],
    *,
    format: str = "yaml",
) -> str:
    """Render *composition* as deterministic YAML or JSON text."""
    fmt = _normalise_format(format)
    payload = composition_to_dict(composition)
    if fmt == "json":
        return (
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n"
        )
    text = yaml.safe_dump(
        payload,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )
    return text if text.endswith("\n") else text + "\n"


def loads_composition(
    text: str,
    *,
    format: str | None = None,
) -> Composition:
    """Parse and validate a composition from YAML or JSON text."""
    if not isinstance(text, str):
        raise TypeError("composition text must be a string")
    fmt = _normalise_format(format) if format is not None else _infer_format(text)
    try:
        payload = json.loads(text) if fmt == "json" else yaml.safe_load(text)
    except json.JSONDecodeError as exc:
        raise CompositionIOError(f"invalid JSON composition: {exc}") from exc
    except yaml.YAMLError as exc:
        raise CompositionIOError(f"invalid YAML composition: {exc}") from exc
    if payload is None:
        raise CompositionIOError("composition document is empty")
    if not isinstance(payload, Mapping):
        raise CompositionIOError("composition document must be a mapping")
    return Composition.model_validate(dict(payload))


def composition_from_yaml(text: str) -> Composition:
    """Parse and validate a composition from YAML text."""
    return loads_composition(text, format="yaml")


def composition_from_json(text: str) -> Composition:
    """Parse and validate a composition from JSON text."""
    return loads_composition(text, format="json")


def _load_file(path: Path, format: str | None) -> Composition:
    fmt = _normalise_format(format) if format is not None else _format_for_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"composition file not found: {path}")
    return loads_composition(path.read_text(encoding="utf-8"), format=fmt)


def load_composition(
    source: Composition | Mapping[str, Any] | str | Path,
    *,
    format: str | None = None,
) -> Composition:
    """Load a composition from a file path, a document or a mapping.

    A string is treated as a path when that file exists. Otherwise it is
    parsed as YAML or JSON text, so ``load_composition("{...}")`` works too.
    An explicit ``format`` overrides the file extension or the detected text
    format.
    """
    if isinstance(source, Composition):
        return source
    if isinstance(source, Mapping):
        return Composition.model_validate(dict(source))
    if isinstance(source, Path):
        return _load_file(source.expanduser(), format)
    if isinstance(source, str):
        if _looks_like_document(source):
            return loads_composition(source, format=format)
        candidate = Path(source).expanduser()
        if candidate.is_file():
            return _load_file(candidate, format)
        raise FileNotFoundError(f"composition file not found: {candidate}")
    raise TypeError("expected a Composition, mapping, path or document text")


load_composition_file = load_composition
"""Alias for :func:`load_composition`."""

read_composition = load_composition
"""Alias for :func:`load_composition`."""

read_composition_file = load_composition
"""Alias for :func:`load_composition`."""


def save_composition(
    composition: Composition | Mapping[str, Any],
    path: str | Path,
    *,
    format: str | None = None,
) -> Path:
    """Write *composition* to *path*, inferring the format from its suffix.

    A path without a suffix defaults to YAML unless ``format`` is given.
    Parent directories are created when they do not exist.
    """
    target = Path(path).expanduser()
    fmt = _normalise_format(format) if format is not None else _format_for_path(target)
    text = dump_composition(composition, format=fmt or "yaml")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


save_composition_file = save_composition
"""Alias for :func:`save_composition`."""

write_composition = save_composition
"""Alias for :func:`save_composition`."""

write_composition_file = save_composition
"""Alias for :func:`save_composition`."""


def build_composition_schema() -> dict[str, Any]:
    """Return the JSON Schema describing a composition document."""
    schema = Composition.model_json_schema()
    schema["$schema"] = COMPOSITION_JSON_SCHEMA_DIALECT
    schema["title"] = COMPOSITION_JSON_SCHEMA_TITLE
    return schema


composition_json_schema = build_composition_schema
"""Alias for :func:`build_composition_schema`."""

composition_schema = build_composition_schema
"""Alias for :func:`build_composition_schema`."""

export_composition_schema = build_composition_schema
"""Alias for :func:`build_composition_schema`."""


def dump_composition_schema(
    schema: Mapping[str, Any] | None = None,
) -> str:
    """Render *schema* (or the current composition schema) as stable JSON."""
    payload = build_composition_schema() if schema is None else dict(schema)
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    )


def save_composition_schema(
    path: str | Path,
    schema: Mapping[str, Any] | None = None,
) -> Path:
    """Write the composition JSON Schema to *path*."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(dump_composition_schema(schema), encoding="utf-8")
    return target
