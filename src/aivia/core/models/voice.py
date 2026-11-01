"""Records for locally registered voice clones.

A voice clone is a provider voice id that belongs to a real owner and may only
be used when that owner consented to cloning. The record keeps the consent
metadata beside the id so a later speech request can prove why the voice was
allowed, not just that it exists.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import AliasChoices, Field, field_validator

from aivia.core.ids import new_voice_clone_id
from aivia.core.models.base import ModelBase

__all__ = ["VoiceClone"]


def _strip(value: Any) -> Any:
    return value.strip() if isinstance(value, str) else value


def _text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("must be a string")
    return value.strip()


def _mapping(value: Any) -> Any:
    if value is None:
        return {}
    return value


class VoiceClone(ModelBase):
    """One provider voice id registered for an owner.

    ``provider_voice_id`` is the exact id a speech request must pin.
    ``owner`` names the person or account whose voice it is. ``consent`` stores
    the explicit grant and any supporting metadata, such as who granted it, when
    and under which reference. ``language`` is optional because a cloned voice
    may be used for more than one language.
    """

    id: str = Field(
        default_factory=new_voice_clone_id,
        min_length=1,
        validation_alias=AliasChoices("id", "clone_id", "voice_clone_id"),
    )
    name: str = Field(
        default="",
        validation_alias=AliasChoices("name", "label", "title"),
    )
    provider: str = Field(
        min_length=1,
        validation_alias=AliasChoices("provider", "provider_name"),
    )
    provider_voice_id: str = Field(
        min_length=1,
        validation_alias=AliasChoices(
            "provider_voice_id",
            "voice_id",
            "provider_voice",
        ),
    )
    owner: str = Field(
        min_length=1,
        validation_alias=AliasChoices("owner", "owner_id", "account"),
    )
    language: str = Field(
        default="",
        validation_alias=AliasChoices("language", "lang", "locale"),
    )
    consent: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("consent", "consent_metadata"),
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "provider", "provider_voice_id", "owner", mode="before")
    @classmethod
    def _strip_required(cls, value: Any) -> Any:
        return _strip(value)

    @field_validator("name", "language", mode="before")
    @classmethod
    def _strip_optional(cls, value: Any) -> Any:
        return _text(value)

    @field_validator("language")
    @classmethod
    def _normalize_language(cls, value: str) -> str:
        return value.casefold()

    @field_validator("consent", "metadata", mode="before")
    @classmethod
    def _coerce_mapping(cls, value: Any) -> Any:
        return _mapping(value)

    @field_validator("consent", "metadata")
    @classmethod
    def _validate_mapping(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise ValueError("must be a mapping")
        return dict(value)

    @property
    def clone_id(self) -> str:
        """Alias for :attr:`id`."""
        return self.id

    @property
    def voice_id(self) -> str:
        """Alias for :attr:`provider_voice_id`."""
        return self.provider_voice_id

    @property
    def consent_granted(self) -> bool:
        """Whether the stored consent metadata records an explicit grant."""
        return bool(self.consent.get("granted"))

    @property
    def consent_metadata(self) -> dict[str, Any]:
        """A copy of the stored consent metadata."""
        return dict(self.consent)
