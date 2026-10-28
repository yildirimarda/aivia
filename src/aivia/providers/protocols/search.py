"""Typed search protocol, request and response models.

A search provider answers a :class:`SearchRequest` with a
:class:`SearchResponse`. Results are normalized into :class:`SearchResult`
records so the research module never handles a vendor payload directly.

The protocol does not deduplicate results: providers may return duplicates and
the normalization stage owns that policy. :meth:`SearchResponse.unique_results`
is available to callers that need a de-duplicated view without mutating the
original response.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Any, Protocol, Self, runtime_checkable
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator

from aivia.providers.protocols.base import (
    CapabilityProvider,
    ProtocolModel,
    ensure_utc,
    normalize_domain,
    validate_http_url,
)

__all__ = [
    "CAPABILITY_SEARCH",
    "SafeSearch",
    "SearchDepth",
    "SearchProvider",
    "SearchRecency",
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
]

CAPABILITY_SEARCH = "search"
"""The capability name a search provider declares."""


class SearchRecency(StrEnum):
    """A relative time window accepted by most search engines."""

    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"


class SafeSearch(StrEnum):
    """The safe-search level for a query."""

    OFF = "off"
    MODERATE = "moderate"
    STRICT = "strict"


class SearchDepth(StrEnum):
    """How much content a provider should retrieve per result."""

    BASIC = "basic"
    ADVANCED = "advanced"


class SearchRequest(ProtocolModel):
    """A validated search query.

    Domain filters are bare host names; a URL or ``www.`` prefix is normalized
    so two spellings of the same host cannot slip past the filter.
    """

    query: str = Field(min_length=1)
    max_results: int = Field(default=10, ge=1, le=100)
    include_domains: list[str] = Field(default_factory=list)
    exclude_domains: list[str] = Field(default_factory=list)
    recency: SearchRecency | None = None
    start_date: date | None = None
    end_date: date | None = None
    language: str | None = None
    country: str | None = None
    safe_search: SafeSearch = SafeSearch.MODERATE
    search_depth: SearchDepth = SearchDepth.BASIC
    timeout_seconds: float | None = Field(default=None, gt=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("query", mode="before")
    @classmethod
    def _strip_query(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("include_domains", "exclude_domains", mode="before")
    @classmethod
    def _coerce_domains(cls, value: Any) -> Any:
        if value is None:
            return []
        return value

    @field_validator("include_domains", "exclude_domains")
    @classmethod
    def _normalize_domains(cls, value: list[str]) -> list[str]:
        domains: list[str] = []
        for entry in value:
            domain = normalize_domain(entry)
            if domain not in domains:
                domains.append(domain)
        return domains

    @field_validator("language", mode="before")
    @classmethod
    def _normalize_language(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip().lower()
            return text or None
        return value

    @field_validator("country", mode="before")
    @classmethod
    def _normalize_country(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip().upper()
            return text or None
        return value

    @model_validator(mode="after")
    def _validate_dates(self) -> Self:
        if (
            self.start_date is not None
            and self.end_date is not None
            and self.start_date > self.end_date
        ):
            raise ValueError("start_date must not be after end_date")
        return self


class SearchResult(ProtocolModel):
    """One normalized search result."""

    title: str = ""
    url: str
    snippet: str = ""
    content: str = ""
    published_at: datetime | None = None
    score: float | None = Field(default=None, ge=0.0)
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("url")
    @classmethod
    def _validate_url(cls, value: str) -> str:
        return validate_http_url(value, field="result url")

    @field_validator("title", "snippet", "content", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("source", mode="before")
    @classmethod
    def _strip_source(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip()
            return text or None
        return value

    @field_validator("published_at")
    @classmethod
    def _normalize_published_at(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return ensure_utc(value)

    @property
    def domain(self) -> str:
        """The lower-case host of :attr:`url` without a ``www.`` prefix."""
        host = (urlsplit(self.url).hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        return host

    @property
    def text(self) -> str:
        """The extracted content when present, otherwise the snippet."""
        return self.content or self.snippet


class SearchResponse(ProtocolModel):
    """A validated response from one search provider."""

    query: str = Field(min_length=1)
    results: list[SearchResult] = Field(default_factory=list)
    provider: str = ""
    answer: str | None = None
    total_results: int | None = Field(default=None, ge=0)
    took_ms: float | None = Field(default=None, ge=0.0)
    request_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("query", mode="before")
    @classmethod
    def _strip_query(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("provider", mode="before")
    @classmethod
    def _strip_provider(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("answer", "request_id", mode="before")
    @classmethod
    def _strip_optional_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip()
            return text or None
        return value

    @property
    def urls(self) -> list[str]:
        """Every result URL in response order."""
        return [result.url for result in self.results]

    def unique_results(self) -> list[SearchResult]:
        """Return results de-duplicated by URL, keeping the first occurrence."""
        seen: set[str] = set()
        unique: list[SearchResult] = []
        for result in self.results:
            if result.url in seen:
                continue
            seen.add(result.url)
            unique.append(result)
        return unique


@runtime_checkable
class SearchProvider(CapabilityProvider, Protocol):
    """A provider that serves the ``search`` capability."""

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Execute *request* and return a validated response."""
