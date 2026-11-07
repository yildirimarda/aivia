"""Character schema: a reusable recurring character stored in the user directory.

A character bundles the look, voice and assets a run needs to keep one figure
consistent across outputs. The schema is stored as a YAML file next to its
reference images and pose cutouts in the user data directory. This module is
plain validation only: no AI call and no network request happens here.

The character carries a ``version`` so later items can bump it when the sheet,
pose or voice changes while runs pin an older one.
"""

from __future__ import annotations

from typing import Any, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

__all__ = [
    "CURRENT_CHARACTER_SCHEMA_VERSION",
    "Character",
]

CURRENT_CHARACTER_SCHEMA_VERSION = 1
"""The character schema version written by this release."""


class Character(BaseModel):
    """One recurring character: identity, voice, look and referenced assets.

    ``reference_images`` and ``pose_library`` hold paths relative to the
    character file's directory; the loader confirms they exist on disk.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    name: str = Field(min_length=1)
    version: int = Field(default=CURRENT_CHARACTER_SCHEMA_VERSION, ge=1)
    voice: str = ""
    art_style: str = ""
    negatives: list[str] = Field(default_factory=list)
    reference_images: list[str] = Field(default_factory=list)
    pose_library: dict[str, str] = Field(default_factory=dict)
    brand_kit: str = ""
    ownership_required: bool = True
    allow_real_person_likeness: bool = False
    resemblance_warning: bool = True

    @field_validator("name")
    @classmethod
    def _strip_name(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("name must be non-empty")
        return text

    @field_validator("voice", "art_style", "brand_kit", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("negatives")
    @classmethod
    def _normalise_negatives(cls, value: list[str]) -> list[str]:
        normalised: list[str] = []
        for entry in value:
            text = entry.strip()
            if not text:
                raise ValueError("negatives must not contain blank entries")
            normalised.append(text)
        return normalised

    @field_validator("reference_images")
    @classmethod
    def _normalise_reference_images(cls, value: list[str]) -> list[str]:
        normalised: list[str] = []
        for entry in value:
            text = entry.strip()
            if not text:
                raise ValueError("reference images must not contain blank entries")
            normalised.append(text)
        return normalised

    @field_validator("pose_library")
    @classmethod
    def _normalise_pose_library(cls, value: dict[str, str]) -> dict[str, str]:
        normalised: dict[str, str] = {}
        for pose, image in value.items():
            key = pose.strip()
            asset = image.strip()
            if not key:
                raise ValueError("pose names must be non-empty")
            if not asset:
                raise ValueError("pose assets must be non-empty")
            if key in normalised:
                raise ValueError(f"duplicate pose: {key}")
            normalised[key] = asset
        return normalised

    @property
    def schema_version(self) -> int:
        """The schema version this character was written against."""
        return self.version

    def bump_version(self) -> Self:
        """Return a copy of this character with its version incremented by one.

        Sheet, pose and voice changes are published as a new character version.
        Callers bump the version when any of those change so a run can pin the
        older one and keep reproducing the same output. The original character
        is left untouched.
        """
        return self.model_copy(update={"version": self.version + 1})
