"""Store style specs as editable YAML files in the user config directory.

A style spec is learned from references, then reviewed and edited by a person.
Keeping it as a plain YAML file under ``<config>/style_specs`` makes that edit
natural: any text editor can change a colour, a duration or a prompt suffix and
the next run reads the same validated :class:`~aivia.reference.schema.StyleSpec`
again by name.

The store never caches a spec. :meth:`StyleSpecStore.load` always reads the file
from disk, so an edit made between two calls is picked up without restarting the
process. Writes are atomic and deterministic, and every load validates the file
through the schema, raising an error that names both the file and the failing
field instead of letting a malformed spec reach generation.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from aivia.core import paths
from aivia.core.errors import ConfigError
from aivia.reference.schema import StyleSpec

__all__ = [
    "DEFAULT_STYLE_SPEC_EXTENSION",
    "STYLE_SPEC_DIR",
    "STYLE_SPEC_DIRNAME",
    "STYLE_SPEC_EXTENSIONS",
    "StyleSpecError",
    "StyleSpecInvalidError",
    "StyleSpecNotFoundError",
    "StyleSpecStore",
    "load_style_spec",
    "resolve_style_spec_path",
    "save_style_spec",
    "style_spec_dir",
    "style_spec_names",
    "style_spec_path",
]

STYLE_SPEC_DIR = "style_specs"
"""The user directory, beside the config file, that holds editable specs."""

STYLE_SPEC_DIRNAME = STYLE_SPEC_DIR
"""Alias for :data:`STYLE_SPEC_DIR`."""

STYLE_SPEC_EXTENSIONS = (".yaml", ".yml")
"""Every file extension a style spec may use."""

DEFAULT_STYLE_SPEC_EXTENSION = ".yaml"
"""The extension appended when a name has none."""


class StyleSpecError(ConfigError):
    """A style spec is missing, unreadable or invalid."""

    code = "style_spec"


class StyleSpecNotFoundError(StyleSpecError):
    """No style spec file matches the requested name."""

    code = "style_spec_not_found"


class StyleSpecInvalidError(StyleSpecError):
    """A style spec file exists but its contents are not a valid spec."""

    code = "style_spec_invalid"


def style_spec_dir(
    config_dir: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Return the directory where editable style specs are stored."""
    base = paths.config_dir(environ) if config_dir is None else Path(config_dir)
    return Path(base).expanduser() / STYLE_SPEC_DIR


def _clean_name(name: str | Path) -> str:
    text = str(name).strip()
    if not text:
        raise StyleSpecError("style spec name must not be empty")
    if "\x00" in text:
        raise StyleSpecError("style spec name must not contain a null byte")
    return text


def _filename(name: str | Path) -> str:
    text = _clean_name(name)
    candidate = Path(text)
    if candidate.name in {"", ".", ".."} or candidate.parent != Path("."):
        raise StyleSpecError(
            f"style spec name must not contain a path: {text!r}",
            details={"name": text},
        )
    suffix = candidate.suffix.lower()
    if suffix and suffix not in STYLE_SPEC_EXTENSIONS:
        raise StyleSpecError(
            f"unsupported style spec extension: {suffix or text!r}",
            details={"name": text, "extensions": list(STYLE_SPEC_EXTENSIONS)},
        )
    if suffix:
        return candidate.with_suffix(suffix).name
    return candidate.name + DEFAULT_STYLE_SPEC_EXTENSION


def _variants(name: str | Path) -> tuple[str, ...]:
    text = _clean_name(name)
    candidate = Path(text)
    if candidate.suffix.lower() in STYLE_SPEC_EXTENSIONS:
        return (_filename(candidate),)
    stem = _filename(candidate)
    stem = stem[: -len(DEFAULT_STYLE_SPEC_EXTENSION)]
    return tuple(stem + suffix for suffix in STYLE_SPEC_EXTENSIONS)


def _direct_path(name: str | Path, config_dir: Path) -> Path | None:
    candidate = Path(str(name)).expanduser()
    if candidate.is_absolute():
        return candidate
    if candidate.parent != Path("."):
        return config_dir / candidate
    return None


def _validation_message(path: Path, exc: ValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return f"invalid style spec: {path}"
    first = errors[0]
    location = ".".join(str(part) for part in first.get("loc", ()))
    message = str(first.get("msg", exc))
    field = location or "<root>"
    return f"invalid style spec: {path}: {field}: {message}"


class StyleSpecStore:
    """Read and write named style specs in the user config directory.

    ``config_dir`` overrides the resolved user config directory, which makes the
    store easy to point at a temporary directory in tests. ``environ`` is
    honoured when ``config_dir`` is omitted. Every method resolves a name
    against ``<config>/style_specs``; an explicit path may also be passed to
    :meth:`resolve` and :meth:`load`.
    """

    def __init__(
        self,
        *,
        config_dir: str | Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.config_dir = (
            paths.config_dir(environ)
            if config_dir is None
            else Path(config_dir).expanduser().resolve()
        )
        self.user_dir = self.config_dir / STYLE_SPEC_DIR

    @property
    def directory(self) -> Path:
        """The directory holding the editable YAML files."""
        return self.user_dir

    def _target(self, name: str | Path) -> Path:
        return self.user_dir / _filename(name)

    def resolve(self, name: str | Path) -> Path:
        """Return the file selected by *name*.

        A name without an extension is tried with every supported extension. A
        name containing a directory component, or an absolute path, is treated
        as an explicit file path. A missing file raises
        :class:`StyleSpecNotFoundError` naming every path searched.
        """
        direct = _direct_path(name, self.config_dir)
        if direct is not None:
            if direct.is_file():
                return direct
            raise StyleSpecNotFoundError(
                f"style spec not found: {direct}",
                details={"name": str(name), "searched": [str(direct)]},
            )

        searched = tuple(self.user_dir / variant for variant in _variants(name))
        for candidate in searched:
            if candidate.is_file():
                return candidate

        available = self.names()
        hint = f"; available: {', '.join(available)}" if available else ""
        raise StyleSpecNotFoundError(
            f"style spec not found: {name!s} "
            f"(searched {', '.join(map(str, searched))}{hint})",
            details={
                "name": str(name),
                "searched": [str(path) for path in searched],
                "available": list(available),
            },
        )

    path = resolve
    resolve_path = resolve

    def load(self, name: str | Path) -> StyleSpec:
        """Read, validate and return the style spec selected by *name*."""
        return self._load_file(self.resolve(name))

    def reload(self, name: str | Path) -> StyleSpec:
        """Read the file again, picking up edits made since the last load."""
        return self.load(name)

    def _load_file(self, path: Path) -> StyleSpec:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise StyleSpecError(
                f"cannot read style spec: {path}",
                details={"path": str(path)},
            ) from exc
        if not text.strip():
            raise StyleSpecInvalidError(
                f"empty style spec: {path}",
                details={"path": str(path)},
            )
        try:
            payload: Any = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise StyleSpecInvalidError(
                f"invalid style spec YAML: {path}",
                details={"path": str(path), "error": str(exc)},
            ) from exc
        if not isinstance(payload, Mapping):
            raise StyleSpecInvalidError(
                f"style spec root must be a mapping: {path}",
                details={"path": str(path), "type": type(payload).__name__},
            )
        try:
            return StyleSpec.model_validate(dict(payload))
        except ValidationError as exc:
            raise StyleSpecInvalidError(
                _validation_message(path, exc),
                details={"path": str(path), "errors": exc.errors()},
            ) from exc

    def save(self, spec: StyleSpec, name: str | Path | None = None) -> Path:
        """Write *spec* atomically and return the YAML file it was saved to.

        *name* selects the file. When omitted, the spec's own ``name`` is used.
        A name without an extension receives ``.yaml``. The spec itself is not
        modified, so saving under an alias keeps the human-readable name.
        """
        if not isinstance(spec, StyleSpec):
            raise TypeError("spec must be a StyleSpec")
        selected = spec.name if name is None else name
        if selected is None or not str(selected).strip():
            raise StyleSpecError(
                "style spec name is required to save a spec",
                details={"name": str(selected)},
            )
        target = self._target(selected)
        payload = spec.to_dict(exclude_none=True)
        text = yaml.safe_dump(
            payload,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
            width=1000,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(text)
            os.replace(temporary, target)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
        return target

    write = save

    def delete(self, name: str | Path) -> Path:
        """Remove the style spec selected by *name* and return its path."""
        target = self.resolve(name)
        target.unlink()
        return target

    def names(self) -> tuple[str, ...]:
        """Return every style spec name visible in the user directory."""
        if not self.user_dir.is_dir():
            return ()
        names = {
            path.stem
            for path in self.user_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in STYLE_SPEC_EXTENSIONS
            and not path.name.startswith(".")
        }
        return tuple(sorted(names))

    def __contains__(self, name: object) -> bool:
        try:
            self.resolve(str(name))
        except StyleSpecError:
            return False
        return True

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self.names())


def style_spec_path(
    name: str | Path,
    *,
    config_dir: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Return the user file selected by *name*."""
    return StyleSpecStore(config_dir=config_dir, environ=environ).resolve(name)


resolve_style_spec_path = style_spec_path


def load_style_spec(
    name: str | Path,
    *,
    config_dir: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> StyleSpec:
    """Load and validate the style spec selected by *name*."""
    return StyleSpecStore(config_dir=config_dir, environ=environ).load(name)


def save_style_spec(
    spec: StyleSpec,
    name: str | Path | None = None,
    *,
    config_dir: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Save *spec* as editable YAML and return the file path."""
    return StyleSpecStore(config_dir=config_dir, environ=environ).save(spec, name)


def style_spec_names(
    *,
    config_dir: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Return every style spec name in the user directory."""
    return StyleSpecStore(config_dir=config_dir, environ=environ).names()
