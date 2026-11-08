"""Character errors: every failure names the file and the field.

A character file lives in the user data directory next to its reference images
and pose cutouts. Every error raised here carries ``file`` (the character file)
and ``field`` (the offending key) so a caller can point at the exact problem.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aivia.core.errors import ConfigError, ValidationError

__all__ = [
    "BrandKitError",
    "BrandKitNotFoundError",
    "BrandKitSchemaError",
    "CharacterError",
    "CharacterNotFoundError",
    "CharacterSchemaError",
]


class CharacterError(ConfigError):
    """A character file or its assets could not be loaded."""

    code = "character"


class CharacterNotFoundError(CharacterError):
    """No character file exists for the requested name."""

    code = "character_not_found"


class CharacterSchemaError(ValidationError):
    """A character file failed validation or references a missing asset.

    Details always carry ``file`` and ``field``.
    """

    code = "character_schema"

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


class BrandKitError(ConfigError):
    """A brand kit file or its assets could not be loaded."""

    code = "brand_kit"


class BrandKitNotFoundError(BrandKitError):
    """No brand kit file exists for the requested name."""

    code = "brand_kit_not_found"


class BrandKitSchemaError(ValidationError):
    """A brand kit file failed validation or references a missing asset.

    Details always carry ``file`` and ``field``.
    """

    code = "brand_kit_schema"

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
