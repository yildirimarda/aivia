"""Typed page fetch protocol, request and response models.

A page fetch provider retrieves one URL and returns the extracted main text,
not just raw HTML. Bounding the size and time belongs to the request so the
caller can enforce a policy without knowing the provider; validating resolved
addresses against SSRF is the provider's responsibility.

Fetched text is untrusted data. The response never interprets it as
instructions and the caller must keep it inside a data block when it reaches a
model.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import Field, field_validator

from aivia.core.models.base import utcnow
from aivia.providers.protocols.base import (
    CapabilityProvider,
    ProtocolModel,
    ensure_utc,
    validate_http_url,
)

__all__ = [
    "CAPABILITY_PAGE_FETCH",
    "PageFetchProvider",
    "PageFetchRequest",
    "PageFetchResponse",
    "PageKind",
]

CAPABILITY_PAGE_FETCH = "page_fetch"
"""The capability name a page fetch provider declares."""


class PageKind(StrEnum):
    """The coarse media type of a fetched page."""

    HTML = "html"
    TEXT = "text"
    JSON = "json"
    XML = "xml"
    BINARY = "binary"
    OTHER = "other"


def _normalize_headers(value: dict[str, str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for raw_name, raw_value in value.items():
        name = raw_name.strip()
        header_value = raw_value.strip()
        if not name:
            raise ValueError("header names must be non-empty")
        if any(character in name for character in "\r\n:"):
            raise ValueError(f"invalid header name: {raw_name!r}")
        if any(character in header_value for character in "\r\n"):
            raise ValueError(f"invalid header value for {name!r}")
        headers[name] = header_value
    return headers


class PageFetchRequest(ProtocolModel):
    """A validated request to fetch and extract one page.

    Every limit is optional so the provider or configuration supplies its own
    default. Redirects and JavaScript rendering are explicit because both can
    widen the set of contacted hosts.
    """

    url: str
    max_bytes: int | None = Field(default=None, ge=1)
    max_characters: int | None = Field(default=None, ge=1)
    timeout_seconds: float | None = Field(default=None, gt=0.0)
    follow_redirects: bool = True
    render_js: bool = False
    user_agent: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("url")
    @classmethod
    def _validate_url(cls, value: str) -> str:
        return validate_http_url(value)

    @field_validator("user_agent", mode="before")
    @classmethod
    def _strip_user_agent(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip()
            return text or None
        return value

    @field_validator("headers")
    @classmethod
    def _validate_headers(cls, value: dict[str, str]) -> dict[str, str]:
        return _normalize_headers(value)


class PageFetchResponse(ProtocolModel):
    """A validated page fetch and extraction result."""

    url: str
    requested_url: str | None = None
    status_code: int = Field(ge=100, le=599)
    content_type: str | None = None
    kind: PageKind = PageKind.HTML
    title: str | None = None
    text: str = ""
    html: str | None = None
    author: str | None = None
    published_at: datetime | None = None
    language: str | None = None
    links: list[str] = Field(default_factory=list)
    fetched_at: datetime = Field(default_factory=utcnow)
    elapsed_ms: float | None = Field(default=None, ge=0.0)
    size_bytes: int | None = Field(default=None, ge=0)
    truncated: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("url", "requested_url")
    @classmethod
    def _validate_urls(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_http_url(value)

    @field_validator(
        "content_type",
        "title",
        "author",
        "language",
        mode="before",
    )
    @classmethod
    def _strip_optional_text(cls, value: Any) -> Any:
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

    @field_validator("fetched_at")
    @classmethod
    def _normalize_fetched_at(cls, value: datetime) -> datetime:
        return ensure_utc(value)

    @field_validator("links")
    @classmethod
    def _validate_links(cls, value: list[str]) -> list[str]:
        links: list[str] = []
        for entry in value:
            link = validate_http_url(entry, field="link")
            if link not in links:
                links.append(link)
        return links

    @property
    def ok(self) -> bool:
        """Whether the response status is in the 2xx range."""
        return 200 <= self.status_code < 300

    @property
    def word_count(self) -> int:
        """The number of whitespace-separated words in the extracted text."""
        return len(self.text.split())

    @property
    def is_html(self) -> bool:
        """Whether the fetched content is HTML."""
        return self.kind is PageKind.HTML


@runtime_checkable
class PageFetchProvider(CapabilityProvider, Protocol):
    """A provider that serves the ``page_fetch`` capability."""

    async def fetch(self, request: PageFetchRequest) -> PageFetchResponse:
        """Fetch *request* and return a validated extraction result."""
