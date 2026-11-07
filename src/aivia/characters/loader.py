"""Load character files and their assets from the user directory.

A character file lives at ``<data dir>/characters/<name>.yaml`` next to the
reference images and pose cutouts it references. The loader reads the file,
validates it against the :class:`~aivia.characters.schema.Character` schema and
confirms every referenced asset exists on disk. Every error names the file and
the field that failed.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import ValidationError as PydanticValidationError

from aivia.characters.errors import (
    CharacterError,
    CharacterNotFoundError,
    CharacterSchemaError,
)
from aivia.characters.schema import Character
from aivia.core import paths

__all__ = ["CharacterLibrary", "load_character_file"]

_CHARACTERS_DIR = "characters"


class CharacterLibrary:
    """Character files in the user data directory with on-disk assets."""

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
        self.characters_dir = self.data_dir / _CHARACTERS_DIR

    def character_path(self, name: str) -> Path:
        """Return the file for *name*: ``<data dir>/characters/<name>.yaml``."""
        return self.characters_dir / f"{self._normalise_name(name)}.yaml"

    @staticmethod
    def _normalise_name(name: str) -> str:
        if not isinstance(name, str) or not name.strip():
            raise CharacterError("character name must be a non-empty string")
        text = name.strip().replace("\\", "/")
        if any(part in {"", ".", ".."} for part in text.split("/")):
            raise CharacterError(
                f"character name may not escape the characters directory: {name}",
                details={"name": text},
            )
        return text

    def names(self) -> tuple[str, ...]:
        """Return every character name with a file in the characters directory."""
        if not self.characters_dir.is_dir():
            return ()
        names: list[str] = []
        for path in sorted(self.characters_dir.glob("*.yaml")):
            names.append(path.stem)
        return tuple(names)

    def resolve(self, name: str) -> Path:
        """Return the character file for *name*.

        Raises :class:`CharacterNotFoundError`, naming the file, when it does
        not exist.
        """
        path = self.character_path(name)
        if not path.is_file():
            raise CharacterNotFoundError(
                f"unknown character: {name}",
                details={"name": name, "file": str(path)},
            )
        return path

    def load(self, name: str) -> Character:
        """Load and validate *name*, confirming its assets exist on disk."""
        path = self.resolve(name)
        return load_character_file(path)

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        try:
            self.resolve(name)
        except CharacterError:
            return False
        return True


def load_character_file(
    path: str | Path,
) -> Character:
    """Read *path*, validate it and confirm its assets exist on disk.

    Raises :class:`~aivia.characters.errors.CharacterSchemaError`, naming the
    file and field, when the document is invalid or references a missing asset.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CharacterError(
            f"cannot read character file {path}: {exc}",
            details={"file": str(path)},
        ) from exc
    try:
        payload: Any = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise CharacterError(
            f"invalid YAML in character file {path}: {exc}",
            details={"file": str(path)},
        ) from exc
    if payload is None:
        payload = {}
    if not isinstance(payload, Mapping):
        raise CharacterError(
            f"character root must be a mapping: {path}",
            details={"file": str(path)},
        )
    try:
        character = Character.model_validate(dict(payload))
    except PydanticValidationError as exc:
        raise _schema_error(exc, path) from exc
    _check_assets(character, path)
    return character


def _schema_error(exc: PydanticValidationError, path: Path) -> CharacterSchemaError:
    errors = [dict(error) for error in exc.errors()]
    first = errors[0] if errors else {}
    loc = cast(tuple[object, ...], first.get("loc", ()) or ())
    field = ".".join(str(part) for part in loc)
    message = str(first.get("msg", "invalid value"))
    raise CharacterSchemaError(
        f"character file {path} has invalid {field or 'root'}: {message}",
        details={
            "file": str(path),
            "field": field,
            "errors": errors,
        },
    )


def _check_assets(character: Character, path: Path) -> None:
    base = path.parent
    for index, image in enumerate(character.reference_images):
        if not (base / image).is_file():
            raise CharacterSchemaError(
                f"character file {path} references missing asset {image!r}",
                details={
                    "file": str(path),
                    "field": f"reference_images.{index}",
                    "asset": image,
                },
            )
    for pose, image in character.pose_library.items():
        if not (base / image).is_file():
            message = (
                f"character file {path} pose {pose!r} "
                f"references missing asset {image!r}"
            )
            raise CharacterSchemaError(
                message,
                details={
                    "file": str(path),
                    "field": f"pose_library.{pose}",
                    "asset": image,
                },
            )
