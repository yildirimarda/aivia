"""Generic RSS and Atom topic source.

The adapter reads one or more configured feeds and maps each item into a
vendor-neutral :class:`~aivia.providers.protocols.source.SourceResponse`. RSS
2.0, RSS 1.0 (RDF) and Atom are accepted because feeds in the wild use all
three. A request may override the configured feed list through its metadata.

Feeds are untrusted and often unreliable: a feed may be offline, rate limited,
return an HTML error page or contain broken XML. Unlike the curated sources,
this adapter never lets one bad feed fail the research stage. A failed feed is
skipped and recorded in the response metadata, and a request with no usable
feeds returns an empty response. Invalid provider configuration still raises
:class:`~aivia.core.errors.ConfigError`, because a typo in a configured feed
should be visible before any run starts.

The provider owns no routing, retry or budget policy. Transport and retry
behaviour come from the shared HTTP client, but every transport failure is
degraded to an empty result for that feed.
"""

from __future__ import annotations

import html
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urljoin, urlsplit
from xml.etree import ElementTree

import httpx
from pydantic import ValidationError as PydanticValidationError

from aivia.core.config.http import HttpClientConfig
from aivia.core.errors import ConfigError
from aivia.core.models.provider import ProviderMeta
from aivia.core.models.topic import TopicCandidate
from aivia.providers.protocols.source import (
    CAPABILITY_SOURCE,
    SourceRequest,
    SourceResponse,
)
from aivia.providers.registry import register_provider
from aivia.providers.sources.base import SourceProviderBase, register_source
from aivia.providers.sources.wikipedia import DEFAULT_USER_AGENT

__all__ = [
    "DEFAULT_RSS_FEEDS",
    "RSS",
    "RSSAPI",
    "RSSProvider",
    "RSSSource",
    "RSSSourceProvider",
    "normalize_feeds",
    "parse_feed",
]

DEFAULT_RSS_FEEDS: tuple[str, ...] = ()
"""The default feed list; empty until a caller configures one."""

_DEFAULT_METADATA = ProviderMeta(
    name="rss",
    capabilities=[CAPABILITY_SOURCE],
)
_FEED_KEYS = ("feeds", "feed", "feed_urls", "urls")
_TAG_PATTERN = re.compile(r"<[^>]+>")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_TRAFFIC_PATTERN = re.compile(r"(\d[\d.,]*)\s*([kmb])?", re.IGNORECASE)


def normalize_feeds(value: str | Sequence[str] | None) -> tuple[str, ...]:
    """Return validated, de-duplicated feed URLs or raise :class:`ConfigError`.

    ``None`` selects :data:`DEFAULT_RSS_FEEDS`. A comma or whitespace separated
    string is accepted because config files often store a list in one scalar.
    The configured order is kept.
    """
    if value is None:
        return DEFAULT_RSS_FEEDS
    if isinstance(value, str):
        raw = [part for part in re.split(r"[,\s]+", value) if part]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw = list(value)
    else:
        raise ConfigError(
            "feeds must be a string or a sequence of strings",
            details={"field": "feeds"},
        )

    feeds: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            raise ConfigError(
                "feeds must contain only strings",
                details={"field": "feeds"},
            )
        text = item.strip()
        if not text:
            continue
        try:
            parsed = urlsplit(text)
        except ValueError as exc:
            raise ConfigError(
                f"invalid feed URL: {item!r}",
                details={"field": "feeds"},
            ) from exc
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            raise ConfigError(
                f"feed URL must be an absolute http(s) URL: {item!r}",
                details={"field": "feeds"},
            )
        if parsed.username is not None or parsed.password is not None:
            raise ConfigError(
                "feed URL must not contain credentials",
                details={"field": "feeds"},
            )
        if text not in feeds:
            feeds.append(text)

    return tuple(feeds)


def _local(tag: Any) -> str:
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1].lower()


def _element_text(element: ElementTree.Element) -> str:
    return "".join(element.itertext()).strip()


def _clean_text(value: str) -> str:
    text = _TAG_PATTERN.sub(" ", value)
    text = html.unescape(text)
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def _child(
    element: ElementTree.Element,
    names: Sequence[str],
) -> ElementTree.Element | None:
    for child in element:
        if _local(child.tag) in names:
            return child
    return None


def _child_text(element: ElementTree.Element, names: Sequence[str]) -> str:
    found = _child(element, names)
    if found is None:
        return ""
    return _clean_text(_element_text(found))


def _absolute(url: str, base: str) -> str:
    text = url.strip()
    if not text:
        return ""
    try:
        resolved = urljoin(base, text)
        parsed = urlsplit(resolved)
    except ValueError:
        return ""
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return ""
    return resolved


def _parse_date(value: str) -> datetime | None:
    text = value.strip()
    if not text:
        return None
    try:
        parsed = parsedate_to_datetime(text)
    except (TypeError, ValueError, OverflowError):
        parsed = None
    if parsed is None:
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _entry_date(entry: ElementTree.Element) -> datetime | None:
    for name in ("pubdate", "published", "updated", "date", "created"):
        found = _child(entry, (name,))
        if found is None:
            continue
        parsed = _parse_date(_element_text(found))
        if parsed is not None:
            return parsed
    return None


def _entry_title(entry: ElementTree.Element) -> str:
    return _child_text(entry, ("title",))


def _entry_link(entry: ElementTree.Element, feed_url: str) -> str:
    fallback = ""
    for child in entry:
        name = _local(child.tag)
        if name == "link":
            href = child.get("href")
            rel = (child.get("rel") or "alternate").strip().lower()
            candidate = href if href else _element_text(child)
            if not candidate:
                continue
            resolved = _absolute(candidate, feed_url)
            if not resolved:
                continue
            if rel in {"", "alternate"}:
                return resolved
            if not fallback:
                fallback = resolved
        elif name == "guid":
            if (child.get("isPermaLink") or "true").strip().lower() == "false":
                continue
            resolved = _absolute(_element_text(child), feed_url)
            if resolved and not fallback:
                fallback = resolved
        elif name == "id" and not fallback:
            resolved = _absolute(_element_text(child), feed_url)
            if resolved:
                fallback = resolved
    return fallback


def _entry_summary(entry: ElementTree.Element) -> str:
    for names in (
        ("summary", "description"),
        ("content", "encoded"),
        ("subtitle",),
    ):
        text = _child_text(entry, names)
        if text:
            return text
    return ""


def _entry_categories(entry: ElementTree.Element) -> list[str]:
    categories: list[str] = []
    for child in entry:
        if _local(child.tag) not in {"category", "subject"}:
            continue
        value = child.get("term") or _element_text(child)
        text = _clean_text(value)
        if text and text not in categories:
            categories.append(text)
    return categories


def _entry_signal(entry: ElementTree.Element) -> float | None:
    for child in entry:
        if _local(child.tag) not in {"approx_traffic", "traffic"}:
            continue
        text = _element_text(child)
        match = _TRAFFIC_PATTERN.search(text)
        if match is None:
            continue
        try:
            number = float(match.group(1).replace(",", ""))
        except ValueError:
            continue
        multiplier = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}.get(
            (match.group(2) or "").lower(),
            1,
        )
        return number * multiplier
    return None


def _feed_title(root: ElementTree.Element) -> str:
    containers = [root, *list(root)]
    for container in containers:
        if _local(container.tag) not in {"channel", "feed"}:
            continue
        for child in container:
            if _local(child.tag) == "title":
                text = _clean_text(_element_text(child))
                if text:
                    return text
    return ""


def _feed_host(feed_url: str) -> str:
    try:
        return (urlsplit(feed_url).hostname or "").lower()
    except ValueError:
        return ""


def _build_candidate(
    entry: ElementTree.Element,
    feed_url: str,
    request: SourceRequest,
    rank: int,
    feed_title: str,
) -> TopicCandidate | None:
    title = _entry_title(entry)
    if not title:
        return None

    published = _entry_date(entry)
    if request.window_start is not None or request.window_end is not None:
        if published is None:
            return None
        if request.window_start is not None and published < request.window_start:
            return None
        if request.window_end is not None and published > request.window_end:
            return None

    url = _entry_link(entry, feed_url)
    categories = _entry_categories(entry)
    category = (
        categories[0] if categories else (feed_title or _feed_host(feed_url) or "rss")
    )
    try:
        return TopicCandidate(
            title=title,
            summary=_entry_summary(entry),
            source_urls=[url] if url else [],
            mode=request.mode,
            category=category,
            keywords=categories,
            signal=_entry_signal(entry),
            rank=rank,
        )
    except PydanticValidationError:
        return None


def parse_feed(
    text: str,
    *,
    feed_url: str,
    request: SourceRequest,
    start_rank: int = 0,
    limit: int = 20,
) -> tuple[list[TopicCandidate], str] | None:
    """Parse *text* into candidates and the feed title, or ``None`` if malformed.

    A document type declaration is rejected before parsing because an RSS feed
    has no legitimate need for one and it is the usual vector for entity
    expansion attacks.
    """
    if not isinstance(text, str):
        return None
    text = text.lstrip("\ufeff \t\r\n")
    if not text:
        return None
    lowered = text.casefold()
    if "<!doctype" in lowered or "<!entity" in lowered:
        return None
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError:
        return None

    feed_title = _feed_title(root)
    candidates: list[TopicCandidate] = []
    for entry in root.iter():
        if limit <= 0 or len(candidates) >= limit:
            break
        if _local(entry.tag) not in {"item", "entry"}:
            continue
        candidate = _build_candidate(
            entry,
            feed_url,
            request,
            start_rank + len(candidates),
            feed_title,
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates, feed_title


def _metadata_value(
    metadata: Mapping[str, Any],
    keys: Sequence[str],
) -> Any:
    for key in keys:
        if key in metadata and metadata[key] is not None:
            return metadata[key]
    return None


class RSSSourceProvider(SourceProviderBase):
    """A topic source backed by one or more generic RSS or Atom feeds.

    ``feeds`` is the default list read for every request; a request may
    override it through its metadata. ``transport`` injects an ``httpx`` mock
    transport so the offline suite never touches the network.
    """

    default_metadata = _DEFAULT_METADATA

    def __init__(
        self,
        *,
        feeds: str | Sequence[str] | None = None,
        feed: str | Sequence[str] | None = None,
        feed_urls: str | Sequence[str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        name: str | None = None,
        health_timeout_seconds: float | None = None,
        close_timeout_seconds: float | None = None,
        http_config: HttpClientConfig | None = None,
    ) -> None:
        super().__init__(
            name=name,
            health_timeout_seconds=health_timeout_seconds,
            close_timeout_seconds=close_timeout_seconds,
            http_config=http_config,
        )
        if transport is not None and not isinstance(
            transport, httpx.AsyncBaseTransport
        ):
            raise TypeError("transport must be an httpx.AsyncBaseTransport")
        supplied = [value for value in (feeds, feed, feed_urls) if value is not None]
        if len(supplied) > 1:
            raise ConfigError(
                "use only one of feeds, feed or feed_urls",
                details={"field": "feeds"},
            )
        self._feeds = normalize_feeds(supplied[0] if supplied else None)
        if transport is not None:
            self._http = self.create_http_client(transport=transport)

    @property
    def feeds(self) -> tuple[str, ...]:
        """The configured feed URLs, in order."""
        return self._feeds

    @property
    def feed_urls(self) -> tuple[str, ...]:
        """Alias for :attr:`feeds`."""
        return self._feeds

    @property
    def endpoint(self) -> str:
        """The first configured feed URL, or an empty string when none is set."""
        return self._feeds[0] if self._feeds else ""

    @property
    def api_url(self) -> str:
        """Alias for :attr:`endpoint`."""
        return self.endpoint

    def _headers(self) -> dict[str, str]:
        return {
            "accept": (
                "application/rss+xml, application/atom+xml, "
                "application/xml, text/xml;q=0.9, */*;q=0.8"
            ),
            "user-agent": DEFAULT_USER_AGENT,
        }

    def _resolve_feeds(self, request: SourceRequest) -> tuple[str, ...]:
        value = _metadata_value(request.metadata, _FEED_KEYS)
        if value is None:
            return self._feeds
        return normalize_feeds(value)

    async def discover(self, request: SourceRequest) -> SourceResponse:
        """Discover candidates from every configured feed, degrading on failure."""
        if not isinstance(request, SourceRequest):
            raise TypeError("request must be a SourceRequest")

        errors: list[dict[str, Any]] = []
        try:
            feeds = self._resolve_feeds(request)
        except ConfigError as exc:
            return self._response(
                request,
                (),
                [],
                errors=[{"feed": None, "error": str(exc)}],
            )

        candidates: list[TopicCandidate] = []
        for feed in feeds:
            if len(candidates) >= request.limit:
                break
            try:
                response = await self.http.get(
                    feed,
                    headers=self._headers(),
                    timeout=request.timeout_seconds,
                )
            except Exception as exc:
                errors.append({"feed": feed, "error": type(exc).__name__})
                continue
            if response.status_code >= 400:
                errors.append({"feed": feed, "error": f"HTTP {response.status_code}"})
                continue
            try:
                parsed = parse_feed(
                    response.text,
                    feed_url=feed,
                    request=request,
                    start_rank=len(candidates),
                    limit=request.limit - len(candidates),
                )
            except Exception as exc:
                errors.append({"feed": feed, "error": type(exc).__name__})
                continue
            if parsed is None:
                errors.append({"feed": feed, "error": "malformed feed"})
                continue
            feed_candidates, _title = parsed
            candidates.extend(feed_candidates)

        return self._response(request, feeds, candidates, errors=errors)

    def _response(
        self,
        request: SourceRequest,
        feeds: tuple[str, ...],
        candidates: list[TopicCandidate],
        *,
        errors: list[dict[str, Any]],
    ) -> SourceResponse:
        metadata: dict[str, Any] = {
            "feeds": list(feeds),
            "total": len(candidates),
        }
        if errors:
            metadata["errors"] = errors
        if request.query:
            metadata["query"] = request.query
        return SourceResponse(
            provider=self.name,
            query=request.query,
            candidates=candidates,
            metadata=metadata,
        )


RSSSource = RSSSourceProvider
"""Alias for :class:`RSSSourceProvider`."""

RSSProvider = RSSSourceProvider
"""Alias for :class:`RSSSourceProvider`."""

RSSAPI = RSSSourceProvider
"""Alias for :class:`RSSSourceProvider`."""

RSS = RSSSourceProvider
"""Alias for :class:`RSSSourceProvider`."""

register_provider("rss", RSSSourceProvider, module=__name__)
register_provider("rss-source", RSSSourceProvider, module=__name__)
register_provider("feed", RSSSourceProvider, module=__name__)
register_provider("atom", RSSSourceProvider, module=__name__)
register_source("rss", RSSSourceProvider, module=__name__)
register_source("rss-source", RSSSourceProvider, module=__name__)
register_source("feed", RSSSourceProvider, module=__name__)
register_source("atom", RSSSourceProvider, module=__name__)
