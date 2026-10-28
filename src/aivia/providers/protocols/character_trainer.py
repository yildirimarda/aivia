"""Typed character trainer protocol, request and job models.

A character trainer teaches a reusable adapter from reference images. The
request records the character, trigger words and training options, while the
job records the adapter model id, state and training cost. A provider that
cannot train a character must raise a permanent error with a reason instead of
silently returning an unusable job.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol, Self, runtime_checkable

from pydantic import Field, field_validator, model_validator

from aivia.providers.protocols.base import (
    CapabilityProvider,
    ProtocolModel,
    ensure_utc,
)
from aivia.providers.protocols.media import MediaInput

__all__ = [
    "CAPABILITY_CHARACTER_TRAINER",
    "CharacterTrainer",
    "CharacterTrainerRequest",
    "CharacterTrainingJob",
    "TrainingState",
]

CAPABILITY_CHARACTER_TRAINER = "character_trainer"
"""The capability name a character trainer declares."""


class TrainingState(StrEnum):
    """The lifecycle state of a character training job."""

    PENDING = "pending"
    QUEUED = "queued"
    TRAINING = "training"
    READY = "ready"
    FAILED = "failed"
    CANCELLED = "cancelled"


_TERMINAL_TRAINING_STATES = frozenset(
    {TrainingState.READY, TrainingState.FAILED, TrainingState.CANCELLED}
)


def _strip(value: Any) -> Any:
    return value.strip() if isinstance(value, str) else value


def _strip_optional(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return value


class CharacterTrainerRequest(ProtocolModel):
    """A validated request to train a character adapter.

    ``reference_images`` must contain at least one image and ``trigger_words``
    at least one non-empty token. ``base_model`` is optional because an adapter
    may own the default model choice.
    """

    character_id: str = Field(min_length=1)
    name: str = ""
    trigger_words: list[str] = Field(min_length=1)
    reference_images: list[MediaInput] = Field(min_length=1)
    base_model: str | None = None
    steps: int | None = Field(default=None, gt=0)
    learning_rate: float | None = Field(default=None, gt=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("character_id", mode="before")
    @classmethod
    def _strip_character_id(cls, value: Any) -> Any:
        return _strip(value)

    @field_validator("name", "base_model", mode="before")
    @classmethod
    def _strip_optional_text(cls, value: Any) -> Any:
        return _strip_optional(value)

    @field_validator("trigger_words")
    @classmethod
    def _normalize_trigger_words(cls, value: list[str]) -> list[str]:
        words: list[str] = []
        for raw in value:
            word = _strip(raw)
            if not word:
                raise ValueError("trigger words must be non-empty")
            if word not in words:
                words.append(word)
        return words

    @field_validator("reference_images")
    @classmethod
    def _validate_reference_images(cls, value: list[MediaInput]) -> list[MediaInput]:
        for image in value:
            if image.media_type is not None and not image.media_type.startswith(
                "image/"
            ):
                raise ValueError("reference images must be images")
        return value


class CharacterTrainingJob(ProtocolModel):
    """One training job and the adapter it produced.

    A ready job must name its adapter ``model_id``; a failed job should carry a
    safe ``error`` message. ``cost_usd`` is the training cost recorded by the
    provider.
    """

    job_id: str = Field(min_length=1)
    character_id: str = Field(min_length=1)
    state: TrainingState = TrainingState.PENDING
    model_id: str | None = None
    trigger_words: list[str] = Field(default_factory=list)
    cost_usd: float = Field(default=0.0, ge=0.0)
    error: str = ""
    created_at: datetime | None = None
    completed_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("job_id", "character_id", mode="before")
    @classmethod
    def _strip_ids(cls, value: Any) -> Any:
        return _strip(value)

    @field_validator("model_id", mode="before")
    @classmethod
    def _strip_model_id(cls, value: Any) -> Any:
        return _strip_optional(value)

    @field_validator("error", mode="before")
    @classmethod
    def _strip_error(cls, value: Any) -> Any:
        return _strip(value)

    @field_validator("trigger_words")
    @classmethod
    def _normalize_trigger_words(cls, value: list[str]) -> list[str]:
        words: list[str] = []
        for raw in value:
            word = _strip(raw)
            if not word:
                raise ValueError("trigger words must be non-empty")
            if word not in words:
                words.append(word)
        return words

    @field_validator("created_at", "completed_at")
    @classmethod
    def _normalize_timestamp(cls, value: datetime | None) -> datetime | None:
        return None if value is None else ensure_utc(value)

    @model_validator(mode="after")
    def _validate_ready_job(self) -> Self:
        if self.state is TrainingState.READY and not self.model_id:
            raise ValueError("a ready training job must name its model_id")
        return self

    @property
    def is_terminal(self) -> bool:
        """Whether the job will not change state again."""
        return self.state in _TERMINAL_TRAINING_STATES

    @property
    def is_ready(self) -> bool:
        """Whether the adapter is trained and usable."""
        return self.state is TrainingState.READY


@runtime_checkable
class CharacterTrainer(CapabilityProvider, Protocol):
    """A provider that serves the ``character_trainer`` capability."""

    async def train(self, request: CharacterTrainerRequest) -> CharacterTrainingJob:
        """Start training *request* and return the initial job."""

    async def status(self, job_id: str) -> CharacterTrainingJob:
        """Return the current state of a training job."""
