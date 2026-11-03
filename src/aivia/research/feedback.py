"""Topic feedback: store likes, dislikes and comments and summarize guidance.

Feedback is deliberately small and deterministic. A like or dislike names one
topic; an optional comment explains it. Summarizing the entries produces the
guidance that later relevance, scoring, tone and style prompts can carry, so a
reviewer can steer future topic selection without editing the profile brief.

The SQLite ``feedback`` table is the source of truth for the entries. This
module owns the pure summary model and a tiny JSON sidecar that prompt builders
can load without touching the database.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from aivia.core import paths
from aivia.core.errors import ConfigError
from aivia.core.models.base import canonical_json, utcnow

__all__ = [
    "DEFAULT_FEEDBACK_FILENAME",
    "SENTIMENTS",
    "FeedbackGuidance",
    "FeedbackGuidanceStore",
    "TopicFeedback",
    "feedback_guidance_path",
    "load_feedback_guidance",
    "normalize_sentiment",
    "save_feedback_guidance",
    "summarize_feedback",
]

DEFAULT_FEEDBACK_FILENAME = "topic_feedback.json"
"""The guidance sidecar under ``data/research``."""

SENTIMENTS: tuple[str, ...] = ("like", "dislike", "neutral")
"""The sentiment values stored for a topic."""

_STORE_VERSION = 1

_SENTIMENT_ALIASES: dict[str, str] = {
    "": "neutral",
    "like": "like",
    "liked": "like",
    "likes": "like",
    "positive": "like",
    "up": "like",
    "dislike": "dislike",
    "disliked": "dislike",
    "dislikes": "dislike",
    "negative": "dislike",
    "down": "dislike",
    "neutral": "neutral",
    "comment": "neutral",
    "note": "neutral",
}


def normalize_sentiment(value: Any) -> str:
    """Return the canonical sentiment for *value*.

    Aliases such as ``liked`` or ``negative`` are accepted so callers can pass
    the words a person would use. An unknown value fails with the allowed set.
    """
    if value is None:
        return "neutral"
    text = str(value).strip().casefold().replace("-", "_").replace(" ", "_")
    try:
        return _SENTIMENT_ALIASES[text]
    except KeyError:
        allowed = ", ".join(SENTIMENTS)
        raise ValueError(
            f"unknown feedback sentiment: {value!r} (allowed: {allowed})"
        ) from None


def _clean(value: Any) -> str:
    return " ".join(value.split()) if isinstance(value, str) else ""


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _unique(values: Iterable[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean(value)
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            unique.append(text)
    return unique


class TopicFeedback(BaseModel):
    """One like, dislike or comment about a topic."""

    model_config = ConfigDict(extra="ignore")

    topic: str
    sentiment: str = "neutral"
    comment: str = ""
    profile: str = ""
    source: str = "cli"
    run_id: str = ""
    created_at: datetime = Field(default_factory=utcnow)

    @field_validator("topic", "comment", "profile", "source", "run_id", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _clean(value) if value is not None else ""

    @field_validator("topic")
    @classmethod
    def _validate_topic(cls, value: str) -> str:
        if not value:
            raise ValueError("feedback topic must not be empty")
        return value

    @field_validator("sentiment", mode="before")
    @classmethod
    def _validate_sentiment(cls, value: Any) -> Any:
        return normalize_sentiment(value)

    @field_validator("created_at", mode="before")
    @classmethod
    def _parse_created_at(cls, value: Any) -> Any:
        if isinstance(value, str) and value.strip():
            try:
                return datetime.fromisoformat(value.strip())
            except ValueError:
                return utcnow()
        return value

    @field_validator("created_at")
    @classmethod
    def _utc_created_at(cls, value: datetime) -> datetime:
        return _ensure_utc(value)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable feedback record."""
        payload = self.model_dump(mode="json")
        payload["created_at"] = self.created_at.isoformat()
        return payload


class FeedbackGuidance(BaseModel):
    """The deterministic guidance summarized from topic feedback."""

    model_config = ConfigDict(extra="ignore")

    likes: list[str] = Field(default_factory=list)
    dislikes: list[str] = Field(default_factory=list)
    comments: list[str] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
    summary: str = ""

    @property
    def empty(self) -> bool:
        """Whether there is no feedback to guide a prompt."""
        return not self.likes and not self.dislikes and not self.comments

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable guidance record."""
        return {
            "likes": list(self.likes),
            "dislikes": list(self.dislikes),
            "comments": list(self.comments),
            "counts": dict(self.counts),
            "summary": self.summary,
        }

    def to_prompt_dict(self) -> dict[str, Any]:
        """Return the guidance exactly as a prompt context carries it."""
        return self.to_dict()


def _as_feedback(value: TopicFeedback | Mapping[str, Any]) -> TopicFeedback:
    if isinstance(value, TopicFeedback):
        return value
    return TopicFeedback.model_validate(dict(value))


def summarize_feedback(
    entries: Iterable[TopicFeedback | Mapping[str, Any]],
) -> FeedbackGuidance:
    """Summarize feedback into likes, dislikes, comments and one summary line.

    The input order is preserved so repeated runs over the same rows produce
    the same guidance. Titles and comments are deduplicated case-insensitively.
    """
    values = [_as_feedback(entry) for entry in entries]
    likes = _unique(entry.topic for entry in values if entry.sentiment == "like")
    dislikes = _unique(entry.topic for entry in values if entry.sentiment == "dislike")
    comments = _unique(
        f"{entry.topic}: {entry.comment}" for entry in values if entry.comment
    )
    counts = {
        "like": sum(1 for entry in values if entry.sentiment == "like"),
        "dislike": sum(1 for entry in values if entry.sentiment == "dislike"),
        "neutral": sum(1 for entry in values if entry.sentiment == "neutral"),
        "total": len(values),
    }
    parts: list[str] = []
    if likes:
        parts.append("Prefer topics like: " + ", ".join(likes) + ".")
    if dislikes:
        parts.append("Avoid topics like: " + ", ".join(dislikes) + ".")
    if comments:
        parts.append("Feedback notes: " + "; ".join(comments) + ".")
    summary = " ".join(parts) or "No topic feedback yet."
    return FeedbackGuidance(
        likes=likes,
        dislikes=dislikes,
        comments=comments,
        counts=counts,
        summary=summary,
    )


def feedback_guidance_path(
    data_dir: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Return the default guidance sidecar path under ``data/research``."""
    root = paths.data_dir(environ) if data_dir is None else Path(data_dir)
    return Path(root).expanduser() / "research" / DEFAULT_FEEDBACK_FILENAME


class FeedbackGuidanceStore:
    """Read and write the deterministic guidance sidecar atomically."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        data_dir: str | Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self._path = (
            feedback_guidance_path(data_dir, environ=environ)
            if path is None
            else Path(path).expanduser()
        )

    @property
    def path(self) -> Path:
        """The JSON file backing this store."""
        return self._path

    def load(self) -> FeedbackGuidance | None:
        """Return the stored guidance, or ``None`` when the file is absent."""
        if not self._path.exists():
            return None
        try:
            text = self._path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"cannot read feedback guidance: {self._path}") from exc
        if not text.strip():
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(
                f"invalid feedback guidance JSON: {self._path}: {exc}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise ConfigError(f"feedback guidance root must be a mapping: {self._path}")
        raw = payload.get("guidance", payload)
        if not isinstance(raw, Mapping):
            raise ConfigError(f"feedback guidance must be a mapping: {self._path}")
        return FeedbackGuidance.model_validate(dict(raw))

    def save(self, guidance: FeedbackGuidance) -> FeedbackGuidance:
        """Write *guidance* atomically and deterministically."""
        if not isinstance(guidance, FeedbackGuidance):
            raise TypeError("guidance must be a FeedbackGuidance")
        payload = {
            "version": _STORE_VERSION,
            "guidance": guidance.to_dict(),
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{self._path.name}.",
            suffix=".tmp",
            dir=self._path.parent,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(canonical_json(payload, indent=2))
                handle.write("\n")
            os.replace(temporary, self._path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
        return guidance


def load_feedback_guidance(
    path: str | Path | None = None,
    *,
    data_dir: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> FeedbackGuidance | None:
    """Load the guidance sidecar, returning ``None`` when it does not exist."""
    return FeedbackGuidanceStore(path, data_dir=data_dir, environ=environ).load()


def save_feedback_guidance(
    guidance: FeedbackGuidance,
    path: str | Path | None = None,
    *,
    data_dir: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> FeedbackGuidance:
    """Save *guidance* to the sidecar and return it."""
    return FeedbackGuidanceStore(path, data_dir=data_dir, environ=environ).save(
        guidance
    )
