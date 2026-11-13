"""Every file under examples validates against its schema, so examples cannot drift.

The ``examples/`` directory ships reference documents for two schemas:

* compositions -- validated against the composition JSON schema plus the
  cross-object checks in :mod:`aivia.compose.validation`
* provider specs -- validated against
  :class:`aivia.providers.http_spec.HttpProviderSpec`

Any file added to ``examples/`` without a matching validator fails this test,
so an example can never silently fall out of sync with its schema.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aivia.compose import load_composition, validate_composition_or_raise
from aivia.providers.http_spec import HttpProviderSpec

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = ROOT / "examples"

_COMPOSITION_SUFFIXES = {".yaml", ".yml", ".json"}
_PROVIDER_SPEC_SUFFIXES = {".yaml", ".yml"}


def _example_files() -> list[Path]:
    return sorted(path for path in EXAMPLES_DIR.rglob("*") if path.is_file())


def _schema_kind(path: Path) -> str | None:
    """Return the schema that *path* must validate against, or None if unknown."""
    relative = path.relative_to(EXAMPLES_DIR)
    parts = relative.parts
    suffix = path.suffix.lower()
    if parts[0] == "compositions" and suffix in _COMPOSITION_SUFFIXES:
        return "composition"
    if len(parts) == 1 and suffix in _PROVIDER_SPEC_SUFFIXES:
        return "provider_spec"
    return None


def _validate(path: Path, kind: str) -> None:
    if kind == "composition":
        composition = load_composition(path)
        validate_composition_or_raise(composition)
    elif kind == "provider_spec":
        HttpProviderSpec.from_file(path)
    else:  # pragma: no cover - defensive; _schema_kind always names a real kind
        raise AssertionError(f"no validator registered for schema kind {kind!r}")


@pytest.mark.parametrize(
    "path",
    _example_files(),
    ids=lambda path: str(path.relative_to(EXAMPLES_DIR)),
)
def test_each_example_validates_against_its_schema(path: Path) -> None:
    kind = _schema_kind(path)
    assert kind is not None, (
        f"no schema validator for {path.relative_to(EXAMPLES_DIR)}; "
        "add it to _schema_kind or remove it from examples/"
    )
    _validate(path, kind)


def test_examples_directory_is_not_empty() -> None:
    assert _example_files(), f"expected at least one example under {EXAMPLES_DIR}"
