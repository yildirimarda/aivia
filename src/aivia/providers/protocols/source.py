"""Typed topic source protocol, request and response models.

A source provider discovers topic candidates from one external feed such as a
wiki, a community site or an RSS feed. It answers a :class:`SourceRequest` with
a :class:`SourceResponse` whose ``candidates`` are validated
:class:`~aivia.core.models.topic.TopicCandidate` records, so the research
module never handles a vendor payload directly.

A source does not score, deduplicate, verify or select topics; those policies
belong to later research stages. It only reports what it found, in the order it
found it, together with any source-specific signal such as pageviews.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, Self, runtime_checkable

from pydantic import AliasChoices, Field, field_validator, model_validator

from aivia.core.models.topic import TopicCandidate
from aivia.providers.protocols.base import (
    CapabilityProvider,
    ProtocolModel,
    ensure_utc,
)

__all__ = [
    "CAPABILITY_SOURCE",
    "SourceCandidate",
    "SourceProvider",
    "SourceRequest",
    "SourceResponse",
    "SourceResult",
]

CAPABILITY_SOURCE = "source"
"""The capability name a topic source provider declares."""

SourceCandidate = TopicCandidate
"""Alias for :class:`~aivia.core.models.topic.TopicCandidate`."""

SourceResult = TopicCandidate
"""Alias for :class:`~aivia.core.models.topic.TopicCandidate`."""


def _strip(value: Any) -> Any:
    return value.strip() if isinstance(value, str) else value


def _strip_optional(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return value


class SourceRequest(ProtocolModel):
    """A validated request for topic candidates from one source.

    ``query`` is the brief or keyword phrase the candidates should relate to
    and may be empty for feeds that are inherently date or popularity driven.
    ``language`` is a lower-case language tag, ``limit`` bounds the number of
    candidates, and the window narrows a source that accepts a date range.
    """

    query: str = Field(
        default="",
        validation_alias=AliasChoices("query", "brief", "topic", "subject"),
    )
    language: str | None = Field(
        default=None,
        validation_alias=AliasChoices("language", "lang", "locale"),
    )
    limit: int = Field(
        default=20,
        ge=1,
        le=200,
        validation_alias=AliasChoices(
            "limit",
            "max_results",
            "max_items",
            "count",
            "per_page",
        ),
    )
    mode: str = Field(
        default="",
        validation_alias=AliasChoices("mode", "topic_mode", "source_mode"),
    )
    window_start: datetime | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "window_start",
            "since",
            "start",
            "start_date",
            "from_date",
        ),
    )
    window_end: datetime | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "window_end",
            "until",
            "end",
            "end_date",
            "to_date",
        ),
    )
    timeout_seconds: float | None = Field(
        default=None,
        gt=0.0,
        validation_alias=AliasChoices("timeout_seconds", "timeout"),
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("query", "mode", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _strip(value)

    @field_validator("language", mode="before")
    @classmethod
    def _normalize_language(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip().lower().replace("_", "-")
            return text or None
        return value

    @field_validator("window_start", "window_end")
    @classmethod
    def _normalize_window(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return ensure_utc(value)

    @model_validator(mode="after")
    def _validate_window(self) -> Self:
        if (
            self.window_start is not None
            and self.window_end is not None
            and self.window_start > self.window_end
        ):
            raise ValueError("window_start must not be after window_end")
        return self


class SourceResponse(ProtocolModel):
    """A validated set of topic candidates from one source provider.

    ``provider`` names the adapter that served the request; ``source`` is a
    read-only alias for callers that name the feed instead. ``candidates``
    accepts the ``topics``, ``results`` and ``items`` aliases because those are
    the names neighbouring protocols use.
    """

    provider: str = Field(
        default="",
        validation_alias=AliasChoices("provider", "source", "source_name"),
    )
    query: str = ""
    candidates: list[TopicCandidate] = Field(
        default_factory=list,
        validation_alias=AliasChoices("candidates", "topics", "results", "items"),
    )
    request_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("provider", "query", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _strip(value)

    @field_validator("request_id", mode="before")
    @classmethod
    def _strip_request_id(cls, value: Any) -> Any:
        return _strip_optional(value)

    @property
    def source(self) -> str:
        """Alias for :attr:`provider`."""
        return self.provider

    @property
    def topics(self) -> list[TopicCandidate]:
        """Alias for :attr:`candidates`."""
        return self.candidates

    @property
    def results(self) -> list[TopicCandidate]:
        """Alias for :attr:`candidates`."""
        return self.candidates

    @property
    def count(self) -> int:
        """The number of returned candidates."""
        return len(self.candidates)

    @property
    def first(self) -> TopicCandidate | None:
        """The first candidate, or ``None`` when the source found nothing."""
        return self.candidates[0] if self.candidates else None

    @property
    def titles(self) -> list[str]:
        """Every candidate title in response order."""
        return [candidate.title for candidate in self.candidates]

    @property
    def urls(self) -> list[str]:
        """Every distinct source URL in response order."""
        urls: list[str] = []
        for candidate in self.candidates:
            for url in candidate.source_urls:
                if url not in urls:
                    urls.append(url)
        return urls


@runtime_checkable
class SourceProvider(CapabilityProvider, Protocol):
    """A provider that serves the ``source`` capability."""

    async def discover(self, request: SourceRequest) -> SourceResponse:
        """Discover topic candidates for *request*."""
