"""Wikipedia pageviews topic source.

The adapter reads the Wikimedia Pageviews top-pages endpoint for one language
edition and maps the most viewed articles of a day into vendor-neutral
:class:`~aivia.providers.protocols.source.SourceResponse` candidates. A request
does not need a query because the feed is driven by traffic: the date comes
from request metadata, the request window, the provider default or today in
UTC.

Each candidate carries the article's view count as its ``signal`` so later
research stages can rank or gate on traffic without parsing prose. Ranks follow
the feed order, translated from the API's one-based rank to the zero-based rank
the candidate model uses. Obvious non-article pages such as the main page and
``Special:`` or ``Portal:`` namespaces are skipped because they are not usable
topics.

The provider owns no routing, retry or budget policy. Transport and retry
behaviour come from the shared HTTP client and failures map to the shared error
taxonomy.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from datetime import date as date_cls
from typing import Any
from urllib.parse import urlsplit

import httpx
from pydantic import ValidationError as PydanticValidationError

from aivia.core.config.http import HttpClientConfig
from aivia.core.errors import (
    ConfigError,
    InvalidRequestError,
    ValidationError,
)
from aivia.core.models.provider import ProviderMeta
from aivia.core.models.topic import TopicCandidate
from aivia.providers.protocols.source import (
    CAPABILITY_SOURCE,
    SourceRequest,
    SourceResponse,
)
from aivia.providers.registry import register_provider
from aivia.providers.sources.base import SourceProviderBase, register_source
from aivia.providers.sources.wikipedia import (
    DEFAULT_USER_AGENT,
    _article_url,
    _raise_api_error,
    _raise_for_response,
    _validate_base_url,
    normalize_language,
)

__all__ = [
    "ACCESS_LEVELS",
    "DEFAULT_ACCESS",
    "DEFAULT_PAGEVIEWS_BASE_URL",
    "Pageviews",
    "WikipediaPageviews",
    "WikipediaPageviewsAPI",
    "WikipediaPageviewsProvider",
    "WikipediaPageviewsSource",
    "WikipediaPageviewsSourceProvider",
    "normalize_access",
    "pageviews_url",
    "pv_url",
]

DEFAULT_PAGEVIEWS_BASE_URL = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/top/"
    "{language}.wikipedia/{access}"
)
"""The top-pages API template; ``{language}`` and ``{access}`` are filled in."""

ACCESS_LEVELS = ("all-access", "desktop", "mobile-app", "mobile-web")
"""The access levels the Pageviews API accepts."""

DEFAULT_ACCESS = "all-access"
"""Every client is counted unless the provider or request narrows the access."""

_ACCESS_ALIASES = {
    "all": "all-access",
    "all_access": "all-access",
    "app": "mobile-app",
    "mobile": "mobile-web",
    "mobile_app": "mobile-app",
    "mobile_web": "mobile-web",
}
_DATE_PATTERN = re.compile(
    r"^(?P<year>\d{4})[-/](?P<month>\d{1,2})[-/](?P<day>\d{1,2})$"
)
_DATE_KEYS = ("date", "pageviews_date", "snapshot_date")
_NON_ARTICLE_PREFIXES = (
    "special:",
    "wikipedia:",
    "portal:",
    "file:",
    "help:",
    "category:",
    "talk:",
    "template:",
    "user:",
    "draft:",
    "module:",
    "mediawiki:",
    "book:",
    "timedtext:",
)
_NON_ARTICLE_TITLES = {"main page"}
_DEFAULT_METADATA = ProviderMeta(
    name="wikipedia-pageviews",
    capabilities=[CAPABILITY_SOURCE],
)


def normalize_access(value: str | None) -> str:
    """Return a validated access level or raise :class:`ConfigError`.

    ``None`` or an empty string selects the all-access default. Common aliases
    such as ``all``, ``mobile`` and ``app`` are accepted because config files
    tend to use the shorter names.
    """
    if value is None:
        return DEFAULT_ACCESS
    if not isinstance(value, str):
        raise ConfigError(
            "access must be a string",
            details={"field": "access"},
        )
    text = value.strip().lower().replace("_", "-")
    if not text:
        return DEFAULT_ACCESS
    text = _ACCESS_ALIASES.get(text, text)
    if text not in ACCESS_LEVELS:
        allowed = ", ".join(ACCESS_LEVELS)
        raise ConfigError(
            f"unknown pageviews access level: {value!r} (allowed: {allowed})",
            details={"field": "access"},
        )
    return text


def _date_from_parts(
    year: Any,
    month: Any,
    day: Any,
    *,
    field: str = "date",
) -> date_cls:
    if (
        isinstance(year, bool)
        or isinstance(month, bool)
        or isinstance(day, bool)
    ):
        raise ConfigError(
            f"{field} year, month and day must be integers",
            details={"field": field},
        )
    try:
        year_number = int(year)
        month_number = int(month)
        day_number = int(day)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"{field} year, month and day must be integers",
            details={"field": field},
        ) from exc
    try:
        return date_cls(year_number, month_number, day_number)
    except ValueError as exc:
        raise ConfigError(
            f"{field} must be a valid calendar date: "
            f"{year_number}-{month_number}-{day_number}",
            details={"field": field},
        ) from exc


def _parse_date(value: Any, *, field: str = "date") -> date_cls:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date_cls):
        return value
    if isinstance(value, str):
        text = value.strip()
        match = _DATE_PATTERN.fullmatch(text)
        if match is not None:
            return _date_from_parts(
                match.group("year"),
                match.group("month"),
                match.group("day"),
                field=field,
            )
        try:
            return date_cls.fromisoformat(text)
        except ValueError:
            pass
    raise ConfigError(
        f"{field} must be a date or an ISO date: {value!r}",
        details={"field": field},
    )


def _metadata_date(metadata: Mapping[str, Any]) -> date_cls | None:
    for key in _DATE_KEYS:
        if key in metadata and metadata[key] is not None:
            return _parse_date(metadata[key], field=key)
    parts = ("year", "month", "day")
    present = [key for key in parts if key in metadata and metadata[key] is not None]
    if not present:
        return None
    if len(present) != len(parts):
        raise ConfigError(
            "year, month and day must be set together",
            details={"field": "date"},
        )
    return _date_from_parts(
        metadata["year"],
        metadata["month"],
        metadata["day"],
        field="date",
    )


def _resolve_date(
    request: SourceRequest,
    default_date: date_cls | None,
    *,
    now: datetime | None = None,
) -> date_cls:
    value = _metadata_date(request.metadata)
    if value is not None:
        return value
    if request.window_start is not None:
        return request.window_start.date()
    if request.window_end is not None:
        return request.window_end.date()
    if default_date is not None:
        return default_date
    return (now or datetime.now(UTC)).date()


def _resolve_access(request: SourceRequest) -> str | None:
    metadata = request.metadata
    for key in ("access", "access_level", "device"):
        if key in metadata and metadata[key] is not None:
            return normalize_access(metadata[key])
    return None


def _pageviews_root(language: str, access: str, base_url: str) -> str:
    url = base_url.replace("{language}", language)
    url = url.replace("{project}", f"{language}.wikipedia")
    url = url.replace("{access}", access)
    path = urlsplit(url).path.rstrip("/")
    if "pageviews/top" not in path:
        url = f"{url.rstrip('/')}/pageviews/top"
        path = urlsplit(url).path.rstrip("/")
    if path.endswith("/top"):
        url = f"{url.rstrip('/')}/{language}.wikipedia/{access}"
    return url.rstrip("/")


def pageviews_url(
    language: str | None = None,
    *,
    access: str | None = None,
    date: date_cls | str | None = None,
    year: int | None = None,
    month: int | None = None,
    day: int | None = None,
    base_url: str | None = None,
) -> str:
    """Return the top-pages URL, or the API root when no date is given."""
    code = normalize_language(language)
    access_code = normalize_access(access)
    template = (
        DEFAULT_PAGEVIEWS_BASE_URL if base_url is None else _validate_base_url(base_url)
    )
    root = _pageviews_root(code, access_code, template)
    if date is None and year is None and month is None and day is None:
        return root
    if date is not None:
        if year is not None or month is not None or day is not None:
            raise ConfigError(
                "use either date or year, month and day, not both",
                details={"field": "date"},
            )
        target = _parse_date(date)
    else:
        if year is None or month is None or day is None:
            raise ConfigError(
                "year, month and day must be set together",
                details={"field": "date"},
            )
        target = _date_from_parts(year, month, day)
    return f"{root}/{target.year}/{target.month:02d}/{target.day:02d}"


pv_url = pageviews_url
"""Alias for :func:`pageviews_url`."""


def _title_key(title: str) -> str:
    return " ".join(title.replace("_", " ").split()).casefold()


def _article_title(article: Mapping[str, Any]) -> str:
    for key in ("article", "title", "page"):
        value = article.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().replace("_", " ")
    return ""


def _is_article_title(title: str) -> bool:
    key = _title_key(title)
    if key in _NON_ARTICLE_TITLES:
        return False
    lowered = title.strip().casefold()
    return not any(lowered.startswith(prefix) for prefix in _NON_ARTICLE_PREFIXES)


def _views(article: Mapping[str, Any]) -> int | None:
    for key in ("views", "pageviews", "count"):
        if key not in article:
            continue
        value = article[key]
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value if value >= 0 else None
        if isinstance(value, float):
            if value.is_integer() and value >= 0:
                return int(value)
            return None
        if isinstance(value, str):
            text = value.strip().replace(",", "").replace("_", "")
            if text.isdigit():
                return int(text)
        return None
    return None


def _rank(article: Mapping[str, Any], fallback: int) -> int:
    value = article.get("rank")
    if isinstance(value, bool) or not isinstance(value, int):
        return fallback
    if value >= 1:
        return value - 1
    return 0


def _articles(body: Mapping[str, Any]) -> list[Mapping[str, Any]] | None:
    if "items" in body:
        items = body["items"]
        if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
            return None
        articles: list[Mapping[str, Any]] = []
        for item in items:
            if not isinstance(item, Mapping):
                continue
            raw = item.get("articles")
            if raw is None:
                continue
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
                return None
            articles.extend(entry for entry in raw if isinstance(entry, Mapping))
        return articles
    if "articles" in body:
        raw = body["articles"]
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            return None
        return [entry for entry in raw if isinstance(entry, Mapping)]
    return None


class WikipediaPageviewsSourceProvider(SourceProviderBase):
    """A topic source backed by one Wikipedia Pageviews top-pages feed.

    ``language`` is the default edition and ``access`` selects the client
    segment. ``date`` sets a default snapshot date; a request can override it
    through its window or metadata. ``base_url`` may be a concrete API URL or a
    template containing ``{language}``, ``{project}`` or ``{access}``;
    ``transport`` injects an ``httpx`` mock transport so the offline suite never
    touches the network.
    """

    default_metadata = _DEFAULT_METADATA

    def __init__(
        self,
        *,
        language: str | None = None,
        base_url: str | None = None,
        access: str | None = None,
        date: date_cls | datetime | str | None = None,
        year: int | None = None,
        month: int | None = None,
        day: int | None = None,
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
        if date is not None and any(
            value is not None for value in (year, month, day)
        ):
            raise ConfigError(
                "use either date or year, month and day, not both",
                details={"field": "date"},
            )

        self._language = normalize_language(language)
        self._base_url = _validate_base_url(
            DEFAULT_PAGEVIEWS_BASE_URL if base_url is None else base_url
        )
        self._access = normalize_access(access)
        if date is not None:
            self._date: date_cls | None = _parse_date(date)
        elif any(value is not None for value in (year, month, day)):
            if year is None or month is None or day is None:
                raise ConfigError(
                    "year, month and day must be set together",
                    details={"field": "date"},
                )
            self._date = _date_from_parts(year, month, day)
        else:
            self._date = None
        if transport is not None:
            self._http = self.create_http_client(transport=transport)

    @property
    def language(self) -> str:
        """The default Wikipedia edition code."""
        return self._language

    @property
    def base_url(self) -> str:
        """The configured Pageviews API template or URL."""
        return self._base_url

    @property
    def access(self) -> str:
        """The default access level."""
        return self._access

    @property
    def date(self) -> date_cls | None:
        """The configured default snapshot date, if any."""
        return self._date

    @property
    def endpoint(self) -> str:
        """The top-pages API root for the default edition and access level."""
        return self.endpoint_for()

    @property
    def api_url(self) -> str:
        """Alias for :attr:`endpoint`."""
        return self.endpoint

    def endpoint_for(
        self,
        language: str | None = None,
        *,
        access: str | None = None,
        date: date_cls | datetime | str | None = None,
        year: int | None = None,
        month: int | None = None,
        day: int | None = None,
    ) -> str:
        """Return a top-pages URL, or the API root when no date is requested."""
        code = self._language if language is None else normalize_language(language)
        access_code = self._access if access is None else normalize_access(access)
        root = _pageviews_root(code, access_code, self._base_url)
        if date is None and year is None and month is None and day is None:
            return root
        if date is not None:
            if year is not None or month is not None or day is not None:
                raise ConfigError(
                    "use either date or year, month and day, not both",
                    details={"field": "date"},
                )
            target = _parse_date(date)
        else:
            if year is None or month is None or day is None:
                raise ConfigError(
                    "year, month and day must be set together",
                    details={"field": "date"},
                )
            target = _date_from_parts(year, month, day)
        return f"{root}/{target.year}/{target.month:02d}/{target.day:02d}"

    def _headers(self) -> dict[str, str]:
        return {
            "accept": "application/json",
            "user-agent": DEFAULT_USER_AGENT,
        }

    def _to_response(
        self,
        body: Mapping[str, Any],
        request: SourceRequest,
        language: str,
        access: str,
        target: date_cls,
        response: httpx.Response,
    ) -> SourceResponse:
        error = body.get("error")
        if isinstance(error, Mapping):
            _raise_api_error(error, language, self.name)
        elif isinstance(error, str) and error.strip():
            _raise_api_error({"info": error}, language, self.name)

        articles = _articles(body)
        if articles is None:
            raise ValidationError(
                "wikipedia pageviews returned a malformed response",
                details={"provider": self.name, "language": language},
            )

        candidates: list[TopicCandidate] = []
        seen: set[str] = set()
        for article in articles:
            if len(candidates) >= request.limit:
                break
            title = _article_title(article)
            if not title or not _is_article_title(title):
                continue
            views = _views(article)
            if views is None:
                continue
            key = _title_key(title)
            if key in seen:
                continue
            seen.add(key)
            try:
                candidate = TopicCandidate(
                    title=title,
                    summary=f"{views:,} pageviews on {target.isoformat()}",
                    source_urls=[_article_url(language, title)],
                    mode=request.mode,
                    signal=float(views),
                    rank=_rank(article, len(candidates)),
                )
            except PydanticValidationError as exc:
                raise ValidationError(
                    "wikipedia pageviews returned a malformed article",
                    details={"provider": self.name, "language": language},
                ) from exc
            candidates.append(candidate)

        metadata: dict[str, Any] = {
            "language": language,
            "project": f"{language}.wikipedia",
            "access": access,
            "date": target.isoformat(),
            "total": len(candidates),
        }
        request_id = response.headers.get("x-request-id")
        if request_id:
            metadata["request_id"] = request_id
        return SourceResponse(
            provider=self.name,
            query=request.query,
            candidates=candidates,
            request_id=request_id,
            metadata=metadata,
        )

    async def discover(self, request: SourceRequest) -> SourceResponse:
        """Discover the most viewed articles for *request*."""
        if not isinstance(request, SourceRequest):
            raise TypeError("request must be a SourceRequest")

        language = normalize_language(request.language or self._language)
        try:
            access = _resolve_access(request) or self._access
            target = _resolve_date(request, self._date)
        except ConfigError as exc:
            raise InvalidRequestError(
                str(exc),
                details={"provider": self.name, "language": language},
            ) from exc

        endpoint = self.endpoint_for(language, access=access, date=target)
        response = await self.http.get(
            endpoint,
            headers=self._headers(),
            timeout=request.timeout_seconds,
        )
        _raise_for_response(response, language, self.name)

        try:
            body = response.json()
        except ValueError as exc:
            raise ValidationError(
                "wikipedia pageviews returned a non-JSON response",
                details={"provider": self.name, "language": language},
            ) from exc
        if not isinstance(body, Mapping):
            raise ValidationError(
                "wikipedia pageviews returned a malformed response",
                details={"provider": self.name, "language": language},
            )
        return self._to_response(body, request, language, access, target, response)


WikipediaPageviewsSource = WikipediaPageviewsSourceProvider
"""Alias for :class:`WikipediaPageviewsSourceProvider`."""

WikipediaPageviewsProvider = WikipediaPageviewsSourceProvider
"""Alias for :class:`WikipediaPageviewsSourceProvider`."""

WikipediaPageviews = WikipediaPageviewsSourceProvider
"""Alias for :class:`WikipediaPageviewsSourceProvider`."""

WikipediaPageviewsAPI = WikipediaPageviewsSourceProvider
"""Alias for :class:`WikipediaPageviewsSourceProvider`."""

Pageviews = WikipediaPageviewsSourceProvider
"""Alias for :class:`WikipediaPageviewsSourceProvider`."""

register_provider(
    "wikipedia-pageviews",
    WikipediaPageviewsSourceProvider,
    module=__name__,
)
register_provider(
    "wikipedia_pageviews",
    WikipediaPageviewsSourceProvider,
    module=__name__,
)
register_source(
    "wikipedia-pageviews",
    WikipediaPageviewsSourceProvider,
    module=__name__,
)
register_source(
    "wikipedia_pageviews",
    WikipediaPageviewsSourceProvider,
    module=__name__,
)
register_source("pageviews", WikipediaPageviewsSourceProvider, module=__name__)
