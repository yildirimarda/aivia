"""Wikipedia On this day topic source.

The adapter reads the Wikimedia On this day feed for one language edition and
maps each dated entry into a vendor-neutral
:class:`~aivia.providers.protocols.source.SourceResponse`. Unlike the search
source, a request does not need a query: the feed is driven by a month and a
day, taken from the request window, request metadata or the provider default.

Every candidate keeps the feed's structured ``year`` and ``category`` instead
of burying them in prose. Categories are the feed sections ``selected``,
``births``, ``deaths``, ``events`` and ``holidays``. The feed can return the
same article for several years, so candidates are deduplicated by normalized
title across years and categories, keeping the first entry in feed order.

The provider owns no routing, retry or budget policy. Transport and retry
behaviour come from the shared HTTP client and failures map to the shared
error taxonomy.
"""

from __future__ import annotations

import calendar
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
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
    _strip_html,
    _validate_base_url,
    normalize_language,
)

__all__ = [
    "DEFAULT_OTD_BASE_URL",
    "DEFAULT_OTD_CATEGORIES",
    "OTD_CATEGORIES",
    "OnThisDay",
    "WikipediaOTD",
    "WikipediaOTDSource",
    "WikipediaOnThisDay",
    "WikipediaOnThisDayAPI",
    "WikipediaOnThisDayProvider",
    "WikipediaOnThisDaySource",
    "WikipediaOnThisDaySourceProvider",
    "normalize_categories",
    "on_this_day_url",
    "otd_url",
]

DEFAULT_OTD_BASE_URL = (
    "https://api.wikimedia.org/feed/v1/wikipedia/{language}/onthisday"
)
"""The On this day API template; ``{language}`` becomes the edition code."""

OTD_CATEGORIES = ("selected", "births", "deaths", "events", "holidays")
"""The feed sections the On this day API can return."""

DEFAULT_OTD_CATEGORIES = OTD_CATEGORIES
"""Every feed section is requested unless the provider or request narrows it."""

_CATEGORY_ALIASES = {
    "birth": "births",
    "death": "deaths",
    "event": "events",
    "holiday": "holidays",
    "select": "selected",
}
_DATE_PATTERN = re.compile(r"^(?P<month>\d{1,2})[-/](?P<day>\d{1,2})$")
_YEAR_PATTERN = re.compile(r"^-?\d{1,6}$")
_DEFAULT_METADATA = ProviderMeta(
    name="wikipedia-otd",
    capabilities=[CAPABILITY_SOURCE],
)


def normalize_categories(
    value: str | Sequence[str] | None,
) -> tuple[str, ...]:
    """Return validated feed section names or raise :class:`ConfigError`.

    ``None`` keeps every section, ``all`` expands to every section and a
    comma-separated string is accepted because config files often store the
    sections in one scalar.
    """
    if value is None:
        return DEFAULT_OTD_CATEGORIES
    if isinstance(value, str):
        raw = [part for part in value.replace(",", " ").split() if part]
    elif isinstance(value, Sequence):
        raw = list(value)
    else:
        raise ConfigError(
            "categories must be a string or a sequence of strings",
            details={"field": "categories"},
        )

    categories: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            raise ConfigError(
                "categories must contain only strings",
                details={"field": "categories"},
            )
        name = item.strip().lower().replace("_", "-")
        name = _CATEGORY_ALIASES.get(name, name)
        if name == "all":
            for category in OTD_CATEGORIES:
                if category not in categories:
                    categories.append(category)
            continue
        if name not in OTD_CATEGORIES:
            allowed = ", ".join(OTD_CATEGORIES)
            raise ConfigError(
                f"unknown on this day category: {item!r} (allowed: {allowed})",
                details={"field": "categories"},
            )
        if name not in categories:
            categories.append(name)

    if not categories:
        return DEFAULT_OTD_CATEGORIES
    return tuple(categories)


def _month_day(month: Any, day: Any, *, field: str) -> tuple[int, int]:
    if isinstance(month, bool) or isinstance(day, bool):
        raise ConfigError(
            f"{field} month and day must be integers",
            details={"field": field},
        )
    try:
        month_number = int(month)
        day_number = int(day)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"{field} month and day must be integers",
            details={"field": field},
        ) from exc
    if not 1 <= month_number <= 12:
        raise ConfigError(
            f"{field} month must be between 1 and 12: {month!r}",
            details={"field": field},
        )
    last_day = calendar.monthrange(2000, month_number)[1]
    if not 1 <= day_number <= last_day:
        raise ConfigError(
            f"{field} day must be between 1 and {last_day}: {day!r}",
            details={"field": field},
        )
    return month_number, day_number


def _parse_date(value: Any, *, field: str) -> tuple[int, int]:
    if isinstance(value, datetime):
        return value.month, value.day
    if isinstance(value, date):
        return value.month, value.day
    if isinstance(value, str):
        text = value.strip()
        match = _DATE_PATTERN.fullmatch(text)
        if match is not None:
            return _month_day(match.group("month"), match.group("day"), field=field)
        try:
            parsed = date.fromisoformat(text)
        except ValueError:
            parsed = None
        if parsed is not None:
            return parsed.month, parsed.day
    raise ConfigError(
        f"{field} must be a date, an ISO date or MM-DD: {value!r}",
        details={"field": field},
    )


def _resolve_categories(request: SourceRequest) -> tuple[str, ...] | None:
    metadata = request.metadata
    for key in ("categories", "category", "types"):
        if key in metadata and metadata[key] is not None:
            return normalize_categories(metadata[key])
    return None


def _resolve_date(
    request: SourceRequest,
    default_month: int | None,
    default_day: int | None,
    *,
    now: datetime | None = None,
) -> tuple[int, int]:
    metadata = request.metadata
    for key in ("date", "on_this_day_date", "day_of_year"):
        if key in metadata and metadata[key] is not None:
            return _parse_date(metadata[key], field="date")
    if "month" in metadata or "day" in metadata:
        return _month_day(metadata.get("month"), metadata.get("day"), field="date")
    if request.window_start is not None:
        return request.window_start.month, request.window_start.day
    if default_month is not None and default_day is not None:
        return default_month, default_day
    today = (now or datetime.now(UTC)).date()
    return today.month, today.day


def _otd_root(language: str, base_url: str) -> str:
    url = base_url.replace("{language}", language)
    path = urlsplit(url).path.rstrip("/")
    if "onthisday" not in path:
        url = f"{url.rstrip('/')}/onthisday"
    return url.rstrip("/")


def on_this_day_url(
    language: str | None = None,
    *,
    category: str | Sequence[str] | None = None,
    month: int | None = None,
    day: int | None = None,
    base_url: str | None = None,
) -> str:
    """Return a feed URL, or the API root when no category or date is given."""
    code = normalize_language(language)
    template = (
        DEFAULT_OTD_BASE_URL if base_url is None else _validate_base_url(base_url)
    )
    root = _otd_root(code, template)
    if category is None and month is None and day is None:
        return root

    if category is None:
        category_name = "all"
    else:
        categories = normalize_categories(category)
        category_name = categories[0] if len(categories) == 1 else "all"

    if month is None and day is None:
        today = datetime.now(UTC).date()
        month_number, day_number = today.month, today.day
    elif month is None or day is None:
        raise ConfigError(
            "month and day must be set together",
            details={"field": "date"},
        )
    else:
        month_number, day_number = _month_day(month, day, field="date")
    return f"{root}/{category_name}/{month_number:02d}/{day_number:02d}"


otd_url = on_this_day_url
"""Alias for :func:`on_this_day_url`."""


def _clean_title(value: str) -> str:
    return _strip_html(value).replace("_", " ").strip()


def _page_title(page: Mapping[str, Any]) -> str:
    titles = page.get("titles")
    if isinstance(titles, Mapping):
        for key in ("normalized", "canonical", "display"):
            value = titles.get(key)
            if isinstance(value, str) and value.strip():
                return _clean_title(value)
    for key in ("title", "displaytitle"):
        value = page.get(key)
        if isinstance(value, str) and value.strip():
            return _clean_title(value)
    return ""


def _item_pages(item: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    pages = item.get("pages")
    if not isinstance(pages, Sequence) or isinstance(pages, (str, bytes)):
        return []
    return [page for page in pages if isinstance(page, Mapping)]


def _primary_page(
    pages: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    for page in pages:
        if _page_title(page):
            return page
    return None


def _page_url(page: Mapping[str, Any], language: str, title: str) -> str:
    content_urls = page.get("content_urls")
    if isinstance(content_urls, Mapping):
        for platform in ("desktop", "mobile"):
            view = content_urls.get(platform)
            if not isinstance(view, Mapping):
                continue
            value = view.get("page")
            if isinstance(value, str) and value.strip():
                return value.strip()
    for key in ("fullurl", "canonicalurl", "url"):
        value = page.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return _article_url(language, title)


def _item_urls(
    pages: Sequence[Mapping[str, Any]],
    language: str,
    title: str,
) -> list[str]:
    urls: list[str] = []
    for page in pages:
        page_title = _page_title(page)
        if not page_title:
            continue
        url = _page_url(page, language, page_title)
        if url not in urls:
            urls.append(url)
    if not urls:
        urls.append(_article_url(language, title))
    return urls


def _item_summary(item: Mapping[str, Any], page: Mapping[str, Any]) -> str:
    text = item.get("text")
    if isinstance(text, str) and text.strip():
        return _strip_html(text)
    for key in ("extract", "description"):
        value = page.get(key)
        if isinstance(value, str) and value.strip():
            return _strip_html(value)
    return ""


def _item_year(item: Mapping[str, Any]) -> int | None:
    value = item.get("year")
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and _YEAR_PATTERN.fullmatch(value.strip()):
        return int(value.strip())
    return None


def _title_key(title: str) -> str:
    return " ".join(title.replace("_", " ").split()).casefold()


class WikipediaOnThisDaySourceProvider(SourceProviderBase):
    """A topic source backed by one Wikipedia On this day feed.

    ``language`` is the default edition and ``categories`` narrows the feed
    sections. ``month`` and ``day`` set a default date; a request can override
    both through its window or metadata. ``base_url`` may be a concrete API
    root or a template containing ``{language}``; ``transport`` injects an
    ``httpx`` mock transport so the offline suite never touches the network.
    """

    default_metadata = _DEFAULT_METADATA

    def __init__(
        self,
        *,
        language: str | None = None,
        base_url: str | None = None,
        categories: str | Sequence[str] | None = None,
        category: str | None = None,
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
        if categories is not None and category is not None:
            raise ConfigError(
                "use either categories or category, not both",
                details={"field": "categories"},
            )

        self._language = normalize_language(language)
        self._base_url = _validate_base_url(
            DEFAULT_OTD_BASE_URL if base_url is None else base_url
        )
        self._categories = normalize_categories(
            categories if categories is not None else category
        )
        if (month is None) != (day is None):
            raise ConfigError(
                "month and day must be set together",
                details={"field": "date"},
            )
        self._month: int | None
        self._day: int | None
        if month is not None and day is not None:
            self._month, self._day = _month_day(month, day, field="date")
        else:
            self._month, self._day = None, None
        if transport is not None:
            self._http = self.create_http_client(transport=transport)

    @property
    def language(self) -> str:
        """The default Wikipedia edition code."""
        return self._language

    @property
    def base_url(self) -> str:
        """The configured On this day API template or root."""
        return self._base_url

    @property
    def categories(self) -> tuple[str, ...]:
        """The default feed sections in request order."""
        return self._categories

    @property
    def month(self) -> int | None:
        """The configured default month, if any."""
        return self._month

    @property
    def day(self) -> int | None:
        """The configured default day, if any."""
        return self._day

    @property
    def endpoint(self) -> str:
        """The On this day API root for the default language edition."""
        return self.endpoint_for()

    @property
    def api_url(self) -> str:
        """Alias for :attr:`endpoint`."""
        return self.endpoint

    def endpoint_for(
        self,
        language: str | None = None,
        *,
        category: str | None = None,
        month: int | None = None,
        day: int | None = None,
    ) -> str:
        """Return a feed URL, or the API root when no date is requested."""
        code = self._language if language is None else normalize_language(language)
        root = _otd_root(code, self._base_url)
        if category is None and month is None and day is None:
            return root
        if category is None:
            category = self._categories[0] if len(self._categories) == 1 else "all"
        if month is None or day is None:
            month, day = _resolve_date(SourceRequest(), self._month, self._day)
        return f"{root}/{category}/{month:02d}/{day:02d}"

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
        response: httpx.Response,
        categories: tuple[str, ...],
    ) -> SourceResponse:
        error = body.get("error")
        if isinstance(error, Mapping):
            _raise_api_error(error, language, self.name)
        elif isinstance(error, str) and error.strip():
            _raise_api_error({"info": error}, language, self.name)

        present = [key for key in body if key in OTD_CATEGORIES]
        if not present:
            raise ValidationError(
                "wikipedia on this day returned a malformed response",
                details={"provider": self.name, "language": language},
            )

        candidates: list[TopicCandidate] = []
        seen: set[str] = set()
        for category, items in body.items():
            if category not in categories or category not in OTD_CATEGORIES:
                continue
            if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
                continue
            for item in items:
                if len(candidates) >= request.limit:
                    break
                if not isinstance(item, Mapping):
                    continue
                pages = _item_pages(item)
                if not pages:
                    continue
                page = _primary_page(pages)
                if page is None:
                    raise ValidationError(
                        "wikipedia on this day returned a page without a title",
                        details={"provider": self.name, "language": language},
                    )
                title = _page_title(page)
                key = _title_key(title)
                if key in seen:
                    continue
                seen.add(key)
                try:
                    candidate = TopicCandidate(
                        title=title,
                        summary=_item_summary(item, page),
                        source_urls=_item_urls(pages, language, title),
                        mode=request.mode,
                        category=category,
                        year=_item_year(item),
                        rank=len(candidates),
                    )
                except PydanticValidationError as exc:
                    raise ValidationError(
                        "wikipedia on this day returned a malformed entry",
                        details={"provider": self.name, "language": language},
                    ) from exc
                candidates.append(candidate)
            if len(candidates) >= request.limit:
                break

        metadata: dict[str, Any] = {
            "language": language,
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
        """Discover On this day entries for *request*."""
        if not isinstance(request, SourceRequest):
            raise TypeError("request must be a SourceRequest")

        language = normalize_language(request.language or self._language)
        try:
            categories = _resolve_categories(request) or self._categories
            month, day = _resolve_date(request, self._month, self._day)
        except ConfigError as exc:
            raise InvalidRequestError(
                str(exc),
                details={"provider": self.name, "language": language},
            ) from exc

        api_category = categories[0] if len(categories) == 1 else "all"
        endpoint = self.endpoint_for(
            language,
            category=api_category,
            month=month,
            day=day,
        )
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
                "wikipedia on this day returned a non-JSON response",
                details={"provider": self.name, "language": language},
            ) from exc
        if not isinstance(body, Mapping):
            raise ValidationError(
                "wikipedia on this day returned a malformed response",
                details={"provider": self.name, "language": language},
            )
        return self._to_response(body, request, language, response, categories)


WikipediaOnThisDaySource = WikipediaOnThisDaySourceProvider
"""Alias for :class:`WikipediaOnThisDaySourceProvider`."""

WikipediaOnThisDayProvider = WikipediaOnThisDaySourceProvider
"""Alias for :class:`WikipediaOnThisDaySourceProvider`."""

WikipediaOnThisDay = WikipediaOnThisDaySourceProvider
"""Alias for :class:`WikipediaOnThisDaySourceProvider`."""

WikipediaOnThisDayAPI = WikipediaOnThisDaySourceProvider
"""Alias for :class:`WikipediaOnThisDaySourceProvider`."""

WikipediaOTD = WikipediaOnThisDaySourceProvider
"""Alias for :class:`WikipediaOnThisDaySourceProvider`."""

WikipediaOTDSource = WikipediaOnThisDaySourceProvider
"""Alias for :class:`WikipediaOnThisDaySourceProvider`."""

OnThisDay = WikipediaOnThisDaySourceProvider
"""Alias for :class:`WikipediaOnThisDaySourceProvider`."""

register_provider("wikipedia-otd", WikipediaOnThisDaySourceProvider, module=__name__)
register_provider("wikipedia_otd", WikipediaOnThisDaySourceProvider, module=__name__)
register_source("wikipedia-otd", WikipediaOnThisDaySourceProvider, module=__name__)
register_source("wikipedia_otd", WikipediaOnThisDaySourceProvider, module=__name__)
register_source(
    "wikipedia-on-this-day",
    WikipediaOnThisDaySourceProvider,
    module=__name__,
)
