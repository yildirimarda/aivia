"""Trained character adapters and the interface for using them.

A trainer adapter is the reusable output of character training: a model id, the
trigger words that activate it and the cost of training it. An adapter that
cannot be used (no model id) is skipped with a reason so the consistency chain
can fall back to the next strategy.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from aivia.providers.protocols.character_trainer import CharacterTrainingJob

__all__ = [
    "AdapterUnusableError",
    "TrainerAdapter",
]


class AdapterUnusableError(ValueError):
    """An adapter cannot be used and the caller must skip it.

    ``reason`` explains why the adapter is unusable and ``details`` carries
    structured, log-safe context.
    """

    def __init__(self, reason: str, *, details: dict[str, Any] | None = None) -> None:
        merged: dict[str, Any] = {"reason": reason}
        if details:
            merged.update(details)
        super().__init__(f"adapter is unusable: {reason}")
        self.reason = reason
        self.details = merged


class TrainerAdapter(BaseModel):
    """A trained character adapter ready for the consistency chain.

    ``model_id`` names the trained adapter, ``trigger_words`` activate it and
    ``cost_usd`` records what training cost. An adapter is usable only when it
    names a model; otherwise :attr:`reason` explains why and the caller should
    skip it.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True, frozen=True)

    model_id: str = Field(default="")
    trigger_words: list[str] = Field(default_factory=list)
    cost_usd: float = Field(default=0.0, ge=0.0)

    @field_validator("model_id", mode="before")
    @classmethod
    def _strip_model_id(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator("trigger_words")
    @classmethod
    def _normalise_trigger_words(cls, value: list[str]) -> list[str]:
        words: list[str] = []
        for raw in value:
            word = raw.strip() if isinstance(raw, str) else str(raw).strip()
            if not word:
                raise ValueError("trigger words must be non-empty")
            if word not in words:
                words.append(word)
        return words

    @classmethod
    def from_job(cls, job: CharacterTrainingJob) -> TrainerAdapter:
        """Build an adapter from a finished training job."""
        return cls(
            model_id=job.model_id or "",
            trigger_words=list(job.trigger_words),
            cost_usd=job.cost_usd,
        )

    @property
    def model(self) -> str:
        """The trained model id."""
        return self.model_id

    @property
    def triggers(self) -> list[str]:
        """The trigger words that activate this adapter."""
        return list(self.trigger_words)

    @property
    def cost(self) -> float:
        """The training cost in USD."""
        return self.cost_usd

    @property
    def reason(self) -> str:
        """Why this adapter cannot be used, or an empty string when usable."""
        if not self.model_id:
            return "adapter has no model id"
        return ""

    @property
    def is_usable(self) -> bool:
        """Whether this adapter can be used to render a character."""
        return self.reason == ""

    def require_usable(self) -> TrainerAdapter:
        """Return this adapter when usable, otherwise raise AdapterUnusableError."""
        if not self.is_usable:
            raise AdapterUnusableError(self.reason)
        return self
