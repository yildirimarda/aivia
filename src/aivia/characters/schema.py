"""Character schema: a reusable recurring character stored in the user directory.

A character bundles the look, voice and assets a run needs to keep one figure
consistent across outputs. The schema is stored as a YAML file next to its
reference images and pose cutouts in the user data directory. This module is
plain validation only: no AI call and no network request happens here.

The character carries a ``version`` so later items can bump it when the sheet,
pose or voice changes while runs pin an older one.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from aivia.core.errors import ContentPolicyError, InvalidRequestError

if TYPE_CHECKING:
    from aivia.core.models.voice import VoiceClone

__all__ = [
    "CURRENT_CHARACTER_SCHEMA_VERSION",
    "Character",
    "resolve_character_voice",
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
    voices: dict[str, str] = Field(default_factory=dict)
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

    @field_validator("voices", mode="before")
    @classmethod
    def _normalise_voices(cls, value: Any) -> dict[str, str]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("voices must be a mapping of language to voice id")
        normalised: dict[str, str] = {}
        for language, voice_id in value.items():
            if not isinstance(language, str) or not language.strip():
                raise ValueError("voice language keys must be non-empty")
            key = language.strip().replace("_", "-").lower()
            if not isinstance(voice_id, str) or not voice_id.strip():
                raise ValueError("voice ids must be non-empty")
            normalised[key] = voice_id.strip()
        return normalised

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

    def voice_for_language(self, language: str) -> str:
        """Return the voice configured for *language*, falling back to ``voice``.

        A per-language entry in ``voices`` wins; otherwise the default ``voice``
        is returned. An empty or unknown language yields the default voice.
        """
        if not isinstance(language, str) or not language.strip():
            return self.voice
        key = language.strip().replace("_", "-").lower()
        voice_id = self.voices.get(key)
        if voice_id:
            return voice_id
        return self.voice


class _VoiceCloneStore(Protocol):
    """A store able to list voice clones by language and owner."""

    def list(
        self, *, language: str | None = None, owner: str | None = None
    ) -> list[VoiceClone]: ...


def resolve_character_voice(
    character: Character,
    language: str,
    *,
    voice_clones: _VoiceCloneStore | None = None,
    owner: str | None = None,
) -> str:
    """Return the pinned provider voice id for *character* in *language*.

    Resolves the character's per-language voice (falling back to the default
    ``voice``). When *voice_clones* is supplied, the resolved voice id must be
    a registered clone for *owner* in *language* that carries explicit consent,
    and the exact pinned ``provider_voice_id`` is returned. Without a matching
    consented clone the call raises; without explicit consent it raises a
    content policy error.
    """
    voice_id = character.voice_for_language(language)
    if not voice_id:
        return ""
    if voice_clones is None:
        return voice_id
    if owner is not None:
        clones = voice_clones.list(language=language, owner=owner)
    else:
        clones = voice_clones.list(language=language)
    for clone in clones:
        if clone.provider_voice_id == voice_id:
            if not clone.consent_granted:
                raise ContentPolicyError(
                    "voice cloning requires explicit consent",
                    details={"language": language, "voice": voice_id},
                )
            return clone.provider_voice_id
    raise InvalidRequestError(
        f"no consented voice clone registered for {voice_id!r} in {language!r}",
        details={"language": language, "voice": voice_id},
    )
