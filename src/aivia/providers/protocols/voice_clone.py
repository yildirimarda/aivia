"""Typed voice clone protocol, request and response models.

A voice clone provider turns reference recordings into a reusable provider
voice id. The request carries the display name, the audio samples and any
metadata the caller already has, while the response records the exact provider
voice id a later speech request must pin.

The capability is optional. An adapter declares ``voice_clone`` only when the
operator enables it, so a router never sends a rights-sensitive request to a
provider that cannot serve it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from pydantic import AliasChoices, ConfigDict, Field, field_validator, model_validator

from aivia.providers.protocols.base import CapabilityProvider, ProtocolModel
from aivia.providers.protocols.media import MediaInput

__all__ = [
    "CAPABILITY_VOICE_CLONE",
    "CAPABILITY_VOICE_CLONING",
    "VoiceCloneProvider",
    "VoiceCloneRequest",
    "VoiceCloneResponse",
]

CAPABILITY_VOICE_CLONE = "voice_clone"
"""The capability name a voice cloning provider declares."""

CAPABILITY_VOICE_CLONING = CAPABILITY_VOICE_CLONE
"""Alias for callers that name the operation ``voice_cloning``."""


def _strip(value: Any) -> Any:
    return value.strip() if isinstance(value, str) else value


def _optional_text(value: Any) -> Any:
    if value is None:
        return ""
    return _strip(value) if isinstance(value, str) else value


def _sample(value: Any) -> MediaInput:
    if isinstance(value, MediaInput):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return MediaInput(data=bytes(value))
    if isinstance(value, str):
        return MediaInput(url=value)
    if isinstance(value, tuple):
        filename = value[0] if value and isinstance(value[0], str) else None
        data = value[1] if len(value) > 1 else None
        media_type = value[2] if len(value) > 2 else None
        return MediaInput(
            data=data if isinstance(data, bytes) else None,
            url=data if isinstance(data, str) else None,
            media_type=media_type,
            filename=filename,
        )
    if isinstance(value, Mapping):
        sample = dict(value)
        if "data" not in sample and "url" not in sample:
            for key in ("content", "bytes", "audio", "sample"):
                if key in sample:
                    sample["data"] = sample.pop(key)
                    break
        if "media_type" not in sample:
            for key in ("content_type", "mime_type", "type"):
                if key in sample:
                    sample["media_type"] = sample.pop(key)
                    break
        if "url" not in sample:
            for key in ("uri", "href"):
                if key in sample:
                    sample["url"] = sample.pop(key)
                    break
        if "filename" not in sample:
            for key in ("name", "file_name"):
                if key in sample:
                    sample["filename"] = sample.pop(key)
                    break
        return MediaInput.model_validate(sample)
    raise TypeError("voice clone samples must be bytes, URLs, mappings or MediaInput")


def _samples(value: Any) -> list[MediaInput]:
    if value is None:
        return []
    if isinstance(value, (MediaInput, bytes, bytearray, memoryview, str, Mapping)):
        return [_sample(value)]
    if isinstance(value, Sequence):
        return [_sample(item) for item in value]
    raise TypeError("voice clone samples must be a sequence")


def _labels(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except ValueError:
            parsed = None
        if isinstance(parsed, Mapping):
            value = parsed
        else:
            return {
                key.strip(): item.strip()
                for key, _, item in (
                    entry.partition(":") for entry in text.split(",")
                )
                if key.strip() and item.strip()
            }
    if isinstance(value, Mapping):
        labels: dict[str, str] = {}
        for raw_key, raw_value in value.items():
            key = _strip(raw_key)
            item = _strip(raw_value)
            if not isinstance(key, str) or not key:
                raise ValueError("label keys must be non-empty strings")
            if not isinstance(item, str) or not item:
                raise ValueError("label values must be non-empty strings")
            labels[key] = item
        return labels
    if isinstance(value, Sequence):
        return _labels(",".join(str(entry) for entry in value))
    raise TypeError("labels must be a mapping, a sequence or a JSON object string")


class VoiceCloneRequest(ProtocolModel):
    """A validated request to create one provider voice clone.

    ``samples`` accepts inline audio, remote URLs, :class:`MediaInput` values or
    a mixture of them. ``consent`` is carried for provenance only; the caller
    that owns the rights decision enforces it before reaching the adapter.
    """

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        validate_assignment=True,
        frozen=True,
    )

    name: str = Field(
        min_length=1,
        validation_alias=AliasChoices("name", "voice_name", "label", "title"),
    )
    description: str = Field(
        default="",
        validation_alias=AliasChoices("description", "about", "notes"),
    )
    samples: list[MediaInput] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "samples",
            "sample",
            "files",
            "file",
            "audio",
            "audio_files",
            "audio_samples",
            "sample_audio",
            "reference_audio",
            "reference_samples",
            "voice_samples",
            "voice_sample",
        ),
    )
    owner: str = ""
    consent: bool | dict[str, Any] | None = None
    language: str = ""
    labels: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: float | None = Field(default=None, gt=0.0)

    @field_validator("name", "description", "owner", "language", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _optional_text(value)

    @field_validator("samples", mode="before")
    @classmethod
    def _coerce_samples(cls, value: Any) -> Any:
        return _samples(value)

    @field_validator("samples")
    @classmethod
    def _validate_samples(cls, value: list[MediaInput]) -> list[MediaInput]:
        for sample in value:
            if sample.media_type is not None and not sample.media_type.startswith(
                "audio/"
            ):
                raise ValueError("voice clone samples must be audio")
        return value

    @field_validator("labels", mode="before")
    @classmethod
    def _coerce_labels(cls, value: Any) -> Any:
        return _labels(value)

    @field_validator("consent", mode="before")
    @classmethod
    def _coerce_consent(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return dict(value)
        return value

    @field_validator("metadata", mode="before")
    @classmethod
    def _coerce_metadata(cls, value: Any) -> Any:
        return {} if value is None else value

    @model_validator(mode="after")
    def _validate_language(self) -> VoiceCloneRequest:
        if self.language:
            object.__setattr__(self, "language", self.language.casefold())
        return self


class VoiceCloneResponse(ProtocolModel):
    """The provider voice id created by a clone request."""

    voice_id: str = Field(
        min_length=1,
        validation_alias=AliasChoices(
            "voice_id",
            "id",
            "provider_voice_id",
            "voice",
            "clone_id",
        ),
    )
    provider: str = ""
    name: str = ""
    owner: str = ""
    language: str = ""
    consent: bool | dict[str, Any] | None = None
    request_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("provider", "name", "owner", "language", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _optional_text(value)

    @field_validator("request_id", mode="before")
    @classmethod
    def _strip_request_id(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip()
            return text or None
        return value

    @field_validator("consent", mode="before")
    @classmethod
    def _coerce_consent(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return dict(value)
        return value

    @property
    def id(self) -> str:
        """Alias for :attr:`voice_id`."""
        return self.voice_id

    @property
    def provider_voice_id(self) -> str:
        """Alias for :attr:`voice_id`."""
        return self.voice_id

    @property
    def clone_id(self) -> str:
        """Alias for :attr:`voice_id`."""
        return self.voice_id

    def __str__(self) -> str:
        return self.voice_id

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.voice_id == other
        return super().__eq__(other)

    def __hash__(self) -> int:
        return hash(self.voice_id)


@runtime_checkable
class VoiceCloneProvider(CapabilityProvider, Protocol):
    """A provider that serves the ``voice_clone`` capability."""

    async def create_voice(self, request: VoiceCloneRequest) -> VoiceCloneResponse:
        """Create a provider voice from *request* and return its id."""

    async def delete_voice(self, voice_id: str) -> bool:
        """Delete a provider voice and report whether it was removed."""
