"""Topic, candidate and claim models plus the core fact list contract.

A :class:`Topic` is the subject a run is about. A :class:`TopicCandidate` is
one option discovered before selection and a :class:`Claim` is a factual
statement that must point at the sources supporting it. The models are plain
records: they carry data and validate it, but they do not call providers or
know anything about the optional research module.

The :class:`FactList` contract is structural. Research and writing can exchange
fact lists without importing each other: anything that exposes ``claims`` is a
fact list, and :class:`Topic` already satisfies it. This keeps the core usable
with no research module installed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Protocol, Self, runtime_checkable
from urllib.parse import urlsplit

from pydantic import (
    AfterValidator,
    AliasChoices,
    BeforeValidator,
    Field,
    field_validator,
    model_validator,
)

from aivia.core.models.base import ModelBase

__all__ = [
    "Candidate",
    "Claim",
    "Fact",
    "FactBase",
    "FactList",
    "FactListContract",
    "FactListModel",
    "SourceUrls",
    "Topic",
    "TopicCandidate",
]

_SOURCE_URL_ALIASES = AliasChoices(
    "source_urls",
    "sources",
    "urls",
    "source_url",
    "source",
)


def _coerce_source_urls(value: Any) -> Any:
    """Accept a single URL or a sequence of URLs, but reject other shapes."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        url = value.get("url") or value.get("source_url")
        return [url] if url is not None else []
    return value


def _validate_source_urls(values: Sequence[Any]) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for raw in values:
        if not isinstance(raw, str):
            raise ValueError("source URL must be a string")
        url = raw.strip()
        if not url:
            raise ValueError("source URL must be non-empty")
        parsed = urlsplit(url)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"source URL must be an absolute http(s) URL: {url!r}")
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


SourceUrls = Annotated[
    list[str],
    BeforeValidator(_coerce_source_urls),
    AfterValidator(_validate_source_urls),
]
"""A list of absolute http(s) source URLs, de-duplicated and stripped."""


def _source_domains(urls: Sequence[str]) -> list[str]:
    domains: list[str] = []
    for url in urls:
        host = (urlsplit(url).hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        if host and host not in domains:
            domains.append(host)
    return domains


def _non_empty(value: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError("must be non-empty")
    return text


class Claim(ModelBase):
    """One factual statement with the sources that support it.

    The accuracy invariant requires every claim to map to at least one source
    URL, so ``source_urls`` is mandatory and cannot be empty. Duplicate URLs
    are removed. ``confidence`` is a normalized estimate in ``[0, 1]`` and
    ``verified`` records whether cross-source verification has happened.
    """

    text: str = Field(
        min_length=1,
        validation_alias=AliasChoices("text", "statement", "claim", "fact", "content"),
    )
    source_urls: SourceUrls = Field(
        min_length=1,
        validation_alias=_SOURCE_URL_ALIASES,
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    verified: bool = False

    @field_validator("text", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _non_empty(value) if isinstance(value, str) else value

    @property
    def statement(self) -> str:
        """Alias for :attr:`text`."""
        return self.text

    @property
    def sources(self) -> list[str]:
        """Alias for :attr:`source_urls`."""
        return self.source_urls

    @property
    def urls(self) -> list[str]:
        """Alias for :attr:`source_urls`."""
        return self.source_urls

    @property
    def source_domains(self) -> list[str]:
        """The distinct hostnames behind the claim's sources."""
        return _source_domains(self.source_urls)


class TopicCandidate(ModelBase):
    """A possible topic discovered before scoring and selection.

    Candidates may be incomplete: ``score`` and ``reason`` are filled in by the
    scoring and selection stages. The ``mode`` records which topic mode found
    it and ``signal`` carries an optional source-specific strength such as
    pageviews. Source URLs are optional here because a candidate can still be
    rejected before it is verified.
    """

    title: str = Field(
        min_length=1,
        validation_alias=AliasChoices("title", "name", "topic", "subject"),
    )
    summary: str = Field(
        default="",
        validation_alias=AliasChoices("summary", "description", "excerpt", "abstract"),
    )
    source_urls: SourceUrls = Field(
        default_factory=list,
        validation_alias=_SOURCE_URL_ALIASES,
    )
    mode: str = Field(
        default="",
        validation_alias=AliasChoices("mode", "topic_mode", "source_mode"),
    )
    score: float = Field(default=0.0, ge=0.0)
    reason: str = Field(
        default="",
        validation_alias=AliasChoices("reason", "rationale", "explanation"),
    )
    category: str = ""
    keywords: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("keywords", "tags"),
    )
    signal: float | None = None
    rank: int | None = Field(default=None, ge=0)

    @field_validator("title", mode="before")
    @classmethod
    def _strip_title(cls, value: Any) -> Any:
        return _non_empty(value) if isinstance(value, str) else value

    @property
    def name(self) -> str:
        """Alias for :attr:`title`."""
        return self.title

    @property
    def sources(self) -> list[str]:
        """Alias for :attr:`source_urls`."""
        return self.source_urls

    @property
    def urls(self) -> list[str]:
        """Alias for :attr:`source_urls`."""
        return self.source_urls

    @property
    def topic_mode(self) -> str:
        """Alias for :attr:`mode`."""
        return self.mode

    @property
    def source_domains(self) -> list[str]:
        """The distinct hostnames behind the candidate's sources."""
        return _source_domains(self.source_urls)

    def to_topic(
        self,
        *,
        claims: Sequence[Claim] | None = None,
        summary: str | None = None,
    ) -> Topic:
        """Build a :class:`Topic` from this candidate and its claims."""
        return Topic(
            title=self.title,
            summary=self.summary if summary is None else summary,
            claims=list(claims or []),
            source_urls=list(self.source_urls),
            category=self.category,
            mode=self.mode,
            keywords=list(self.keywords),
            candidate=self,
        )


Candidate = TopicCandidate
"""Alias for :class:`TopicCandidate`."""


class Topic(ModelBase):
    """The selected subject of a run and the claims supporting it.

    When ``source_urls`` is omitted it is derived from the nested claims, so a
    topic built from verified claims cannot lose its provenance. A topic that
    came from a candidate keeps that candidate in ``candidate`` for auditing
    the selection reason and score.
    """

    title: str = Field(
        min_length=1,
        validation_alias=AliasChoices("title", "name", "topic", "subject"),
    )
    summary: str = Field(
        default="",
        validation_alias=AliasChoices("summary", "description", "excerpt", "abstract"),
    )
    claims: list[Claim] = Field(
        default_factory=list,
        validation_alias=AliasChoices("claims", "facts"),
    )
    source_urls: SourceUrls = Field(
        default_factory=list,
        validation_alias=_SOURCE_URL_ALIASES,
    )
    category: str = ""
    mode: str = Field(
        default="",
        validation_alias=AliasChoices("mode", "topic_mode", "source_mode"),
    )
    keywords: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("keywords", "tags"),
    )
    candidate: TopicCandidate | None = None

    @field_validator("title", mode="before")
    @classmethod
    def _strip_title(cls, value: Any) -> Any:
        return _non_empty(value) if isinstance(value, str) else value

    @model_validator(mode="after")
    def _derive_source_urls(self) -> Self:
        if not self.source_urls and self.claims:
            merged: list[str] = []
            for claim in self.claims:
                for url in claim.source_urls:
                    if url not in merged:
                        merged.append(url)
            self.source_urls = merged
        return self

    @property
    def name(self) -> str:
        """Alias for :attr:`title`."""
        return self.title

    @property
    def facts(self) -> list[Claim]:
        """Alias for :attr:`claims`."""
        return self.claims

    @property
    def sources(self) -> list[str]:
        """Alias for :attr:`source_urls`."""
        return self.source_urls

    @property
    def urls(self) -> list[str]:
        """Alias for :attr:`source_urls`."""
        return self.source_urls

    @property
    def topic_mode(self) -> str:
        """Alias for :attr:`mode`."""
        return self.mode

    @property
    def source_domains(self) -> list[str]:
        """The distinct hostnames behind the topic's sources."""
        return _source_domains(self.source_urls)

    @classmethod
    def from_candidate(
        cls,
        candidate: TopicCandidate,
        *,
        claims: Sequence[Claim] | None = None,
        summary: str | None = None,
    ) -> Self:
        """Build a topic from a selected candidate."""
        return cls(
            title=candidate.title,
            summary=candidate.summary if summary is None else summary,
            claims=list(claims or []),
            source_urls=list(candidate.source_urls),
            category=candidate.category,
            mode=candidate.mode,
            keywords=list(candidate.keywords),
            candidate=candidate,
        )


Fact = Claim
"""Alias for :class:`Claim` for callers that call a claim a fact."""


@runtime_checkable
class FactList(Protocol):
    """Structural contract for a fact list.

    The optional research module implements this by exposing a ``claims``
    attribute; it never imports this module. Writing and fact-checking stages
    accept any object with the contract, so user supplied text and research
    output take the same path.
    """

    @property
    def claims(self) -> Sequence[Claim]:
        """The claims making up the fact list."""


FactListContract = FactList
"""Alias for :class:`FactList`."""

FactBase = FactList
"""Alias for :class:`FactList` for callers that prefer the fact base wording."""


class FactListModel(ModelBase):
    """A concrete fact list for callers that want a serializable container."""

    topic: Topic | None = None
    claims: list[Claim] = Field(default_factory=list)

    @property
    def source_urls(self) -> list[str]:
        """All source URLs in the list, in first-seen order."""
        urls: list[str] = []
        if self.topic is not None:
            urls.extend(self.topic.source_urls)
        for claim in self.claims:
            for url in claim.source_urls:
                if url not in urls:
                    urls.append(url)
        return urls

    @property
    def facts(self) -> list[Claim]:
        """Alias for :attr:`claims`."""
        return self.claims
