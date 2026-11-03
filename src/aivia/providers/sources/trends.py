"""Optional Google Trends topic source backed by its public RSS feed.

Google Trends has no supported public API for trending searches. The adapter
therefore reads the same public RSS feed a browser uses and maps each trend
into a vendor-neutral :class:`~aivia.providers.protocols.source.SourceResponse`.
The feed is undocumented and occasionally unavailable, so it is an optional
source: it can be disabled in configuration, and every transport, HTTP or parse
failure degrades to an empty response instead of failing the research stage.

The request's ``geo`` metadata, or the configured ``geo``, selects the country
whose trends are read. A request may override it through its metadata. The
provider owns no routing, retry or budget policy; transport and retry behaviour
come from the shared HTTP client.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx

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
from aivia.providers.sources.rss import parse_feed
from aivia.providers.sources.wikipedia import (
    DEFAULT_USER_AGENT,
    _validate_base_url,
)

__all__ = [
    "DEFAULT_TRENDS_BASE_URL",
    "DEFAULT_TRENDS_GEO",
    "GoogleTrends",
    "GoogleTrendsAPI",
    "GoogleTrendsProvider",
    "GoogleTrendsSource",
    "GoogleTrendsSourceProvider",
    "Trends",
    "TrendsSource",
    "normalize_geo",
    "trends_url",
]

DEFAULT_TRENDS_BASE_URL = "https://trends.google.com/trending/rss"
"""The public Google Trends RSS endpoint."""

DEFAULT_TRENDS_GEO = "US"
"""The country read when neither the provider nor request sets one."""

_GEO_PATTERN = re.compile(r"^[A-Z]{2}$")
_GEO_KEYS = ("geo", "region", "country")
_WORLDWIDE = {"", "all", "global", "world", "worldwide"}
_DEFAULT_METADATA = ProviderMeta(
    name="google-trends",
    capabilities=[CAPABILITY_SOURCE],
)


def normalize_geo(value: str | None) -> str:
    """Return an upper-case two-letter country code, or ``""`` for worldwide."""
    if value is None:
        return DEFAULT_TRENDS_GEO
    if not isinstance(value, str):
        raise ConfigError(
            "geo must be a string",
            details={"field": "geo"},
        )
    text = value.strip()
    if text.casefold() in _WORLDWIDE:
        return ""
    upper = text.upper()
    if not _GEO_PATTERN.fullmatch(upper):
        raise ConfigError(
            f"invalid Google Trends geo: {value!r}",
            details={"field": "geo"},
        )
    return upper


def trends_url(geo: str | None = None, base_url: str | None = None) -> str:
    """Return the trends RSS URL for *geo*."""
    code = normalize_geo(geo)
    base = DEFAULT_TRENDS_BASE_URL if base_url is None else _validate_base_url(base_url)
    parsed = urlsplit(base)
    path = parsed.path.rstrip("/")
    if not path:
        path = "/trending/rss"
    elif not path.endswith("/rss"):
        path = f"{path}/rss"
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    if code:
        query["geo"] = code
    else:
        query.pop("geo", None)
    return urlunsplit(
        (parsed.scheme, parsed.netloc, path, urlencode(query), parsed.fragment)
    )


def _metadata_value(
    metadata: Mapping[str, Any],
    keys: Sequence[str],
) -> Any:
    for key in keys:
        if key in metadata and metadata[key] is not None:
            return metadata[key]
    return None


class GoogleTrendsSourceProvider(SourceProviderBase):
    """A topic source backed by the public Google Trends RSS feed.

    ``geo`` selects the country and ``enabled`` lets a profile opt out of an
    undocumented feed. ``transport`` injects an ``httpx`` mock transport so the
    offline suite never touches the network.
    """

    default_metadata = _DEFAULT_METADATA

    def __init__(
        self,
        *,
        geo: str | None = None,
        region: str | None = None,
        country: str | None = None,
        base_url: str | None = None,
        enabled: bool = True,
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
        supplied = [value for value in (geo, region, country) if value is not None]
        if len(supplied) > 1:
            raise ConfigError(
                "use only one of geo, region or country",
                details={"field": "geo"},
            )
        if not isinstance(enabled, bool):
            raise ConfigError(
                "enabled must be a boolean",
                details={"field": "enabled"},
            )
        self._geo = normalize_geo(supplied[0] if supplied else None)
        self._base_url = _validate_base_url(
            DEFAULT_TRENDS_BASE_URL if base_url is None else base_url
        )
        self._enabled = enabled
        if transport is not None:
            self._http = self.create_http_client(transport=transport)

    @property
    def geo(self) -> str:
        """The configured country code, or ``""`` for worldwide."""
        return self._geo

    @property
    def region(self) -> str:
        """Alias for :attr:`geo`."""
        return self._geo

    @property
    def country(self) -> str:
        """Alias for :attr:`geo`."""
        return self._geo

    @property
    def base_url(self) -> str:
        """The configured trends RSS origin or URL."""
        return self._base_url

    @property
    def enabled(self) -> bool:
        """Whether the source reads the feed or returns an empty response."""
        return self._enabled

    @property
    def endpoint(self) -> str:
        """The trends RSS URL for the configured country."""
        return trends_url(self._geo, self._base_url)

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

    def _resolve_geo(self, request: SourceRequest) -> str:
        value = _metadata_value(request.metadata, _GEO_KEYS)
        if value is None:
            return self._geo
        return normalize_geo(value)

    async def discover(self, request: SourceRequest) -> SourceResponse:
        """Discover trending searches, degrading to an empty response."""
        if not isinstance(request, SourceRequest):
            raise TypeError("request must be a SourceRequest")
        if not self._enabled:
            return self._response(
                request,
                geo=self._geo,
                candidates=[],
                metadata={"enabled": False},
            )

        try:
            geo = self._resolve_geo(request)
        except ConfigError as exc:
            return self._response(
                request,
                geo=self._geo,
                candidates=[],
                errors=[{"feed": None, "error": str(exc)}],
            )

        url = trends_url(geo, self._base_url)
        try:
            response = await self.http.get(
                url,
                headers=self._headers(),
                timeout=request.timeout_seconds,
            )
        except Exception as exc:
            return self._response(
                request,
                geo=geo,
                candidates=[],
                errors=[{"feed": url, "error": type(exc).__name__}],
            )
        if response.status_code >= 400:
            return self._response(
                request,
                geo=geo,
                candidates=[],
                errors=[{"feed": url, "error": f"HTTP {response.status_code}"}],
            )
        try:
            parsed = parse_feed(
                response.text,
                feed_url=url,
                request=request,
                limit=request.limit,
            )
        except Exception as exc:
            return self._response(
                request,
                geo=geo,
                candidates=[],
                errors=[{"feed": url, "error": type(exc).__name__}],
            )
        if parsed is None:
            return self._response(
                request,
                geo=geo,
                candidates=[],
                errors=[{"feed": url, "error": "malformed feed"}],
            )
        candidates, _title = parsed
        return self._response(request, geo=geo, candidates=candidates)

    def _response(
        self,
        request: SourceRequest,
        *,
        geo: str,
        candidates: list[TopicCandidate],
        errors: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SourceResponse:
        details: dict[str, Any] = {"geo": geo, "total": len(candidates)}
        if metadata:
            details.update(metadata)
        if errors:
            details["errors"] = errors
        if request.query:
            details["query"] = request.query
        return SourceResponse(
            provider=self.name,
            query=request.query,
            candidates=candidates,
            metadata=details,
        )


GoogleTrendsSource = GoogleTrendsSourceProvider
"""Alias for :class:`GoogleTrendsSourceProvider`."""

GoogleTrendsProvider = GoogleTrendsSourceProvider
"""Alias for :class:`GoogleTrendsSourceProvider`."""

GoogleTrendsAPI = GoogleTrendsSourceProvider
"""Alias for :class:`GoogleTrendsSourceProvider`."""

GoogleTrends = GoogleTrendsSourceProvider
"""Alias for :class:`GoogleTrendsSourceProvider`."""

TrendsSource = GoogleTrendsSourceProvider
"""Alias for :class:`GoogleTrendsSourceProvider`."""

Trends = GoogleTrendsSourceProvider
"""Alias for :class:`GoogleTrendsSourceProvider`."""

register_provider("google-trends", GoogleTrendsSourceProvider, module=__name__)
register_provider("google_trends", GoogleTrendsSourceProvider, module=__name__)
register_provider("googletrends", GoogleTrendsSourceProvider, module=__name__)
register_provider("trends", GoogleTrendsSourceProvider, module=__name__)
register_source("google-trends", GoogleTrendsSourceProvider, module=__name__)
register_source("google_trends", GoogleTrendsSourceProvider, module=__name__)
register_source("googletrends", GoogleTrendsSourceProvider, module=__name__)
register_source("trends", GoogleTrendsSourceProvider, module=__name__)
