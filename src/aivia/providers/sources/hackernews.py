"""Hacker News topic source backed by the Algolia search API.

The adapter reads Hacker News stories through the public Algolia index and maps
each hit into a vendor-neutral
:class:`~aivia.providers.protocols.source.SourceResponse`. A story's points
become the candidate ``signal`` so later research stages can rank or gate on
community interest without parsing prose.

The window, minimum points and minimum comment count are provider settings
that a request may override through its metadata. The window and thresholds
also become Algolia ``numericFilters`` so the server does the heavy filtering,
while a local check keeps a malformed or mocked response honest.

The provider owns no routing, retry or budget policy. Transport and retry
behaviour come from the shared HTTP client, a rejected rate limit maps to
:class:`~aivia.core.errors.RateLimitError` with the server's ``Retry-After``
delay, and every other failure maps to the shared error taxonomy.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic import ValidationError as PydanticValidationError

from aivia.core.config.http import HttpClientConfig
from aivia.core.errors import (
    ConfigError,
    InvalidRequestError,
    RateLimitError,
    RetryableError,
    TimeoutError,
    ValidationError,
)
from aivia.core.http import _retry_after_seconds, error_for_response
from aivia.core.models.provider import ProviderMeta
from aivia.core.models.topic import TopicCandidate
from aivia.providers.protocols.source import (
    CAPABILITY_SOURCE,
    SourceRequest,
    SourceResponse,
)
from aivia.providers.registry import register_provider
from aivia.providers.sources._timewindow import (
    DEFAULT_WINDOW,
    WINDOW_NAMES,
    WINDOW_SECONDS,
    derive_window,
    normalize_window,
    window_from_metadata,
)
from aivia.providers.sources.base import SourceProviderBase, register_source
from aivia.providers.sources.wikipedia import (
    DEFAULT_USER_AGENT,
    _validate_base_url,
)

__all__ = [
    "DEFAULT_HN_BASE_URL",
    "DEFAULT_HN_MIN_COMMENTS",
    "DEFAULT_HN_MIN_POINTS",
    "DEFAULT_HN_WINDOW",
    "DEFAULT_MIN_POINTS",
    "HN",
    "HN_WINDOWS",
    "MAX_RESULTS",
    "HNSource",
    "HackerNews",
    "HackerNewsAPI",
    "HackerNewsProvider",
    "HackerNewsSource",
    "HackerNewsSourceProvider",
]

DEFAULT_HN_BASE_URL = "https://hn.algolia.com/api/v1/search"
"""The Algolia story search endpoint."""

DEFAULT_HN_WINDOW = DEFAULT_WINDOW
"""The default story window."""

HN_WINDOWS = WINDOW_NAMES
"""Every window the adapter accepts."""

MAX_RESULTS = 100
"""The largest page Algolia returns in one call."""

DEFAULT_HN_MIN_POINTS = 0
"""The default minimum story points."""

DEFAULT_HN_MIN_COMMENTS = 0
"""The default minimum comment count."""

DEFAULT_MIN_POINTS = DEFAULT_HN_MIN_POINTS
DEFAULT_MIN_COMMENTS = DEFAULT_HN_MIN_COMMENTS

_DEFAULT_METADATA = ProviderMeta(
    name="hacker-news",
    capabilities=[CAPABILITY_SOURCE],
)
_POINT_KEYS = ("min_points", "min_score", "points_threshold", "threshold", "points")
_COMMENT_KEYS = ("min_comments", "comments_threshold", "comments")


def _integer(
    value: Any,
    *,
    field: str,
    default: int,
) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ConfigError(
            f"{field} must be an integer",
            details={"field": field},
        )
    if isinstance(value, str):
        text = value.strip().replace(",", "").replace("_", "")
        try:
            value = int(text)
        except ValueError as exc:
            raise ConfigError(
                f"{field} must be an integer: {value!r}",
                details={"field": field},
            ) from exc
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ConfigError(
                f"{field} must be an integer",
                details={"field": field},
            )
        value = int(value)
    if not isinstance(value, int):
        raise ConfigError(
            f"{field} must be an integer",
            details={"field": field},
        )
    if value < 0:
        raise ConfigError(
            f"{field} must not be negative",
            details={"field": field},
        )
    return value


def _metadata_value(
    metadata: Mapping[str, Any],
    keys: Sequence[str],
) -> Any:
    for key in keys:
        if key in metadata and metadata[key] is not None:
            return metadata[key]
    return None


def _resolve_window(request: SourceRequest, configured: str) -> str:
    selected = window_from_metadata(request.metadata, default=configured)
    if selected is not None:
        return selected
    if request.window_start is not None or request.window_end is not None:
        return derive_window(
            request.window_start,
            request.window_end,
            default=configured,
        )
    return configured


def _resolve_integer(
    request: SourceRequest,
    keys: Sequence[str],
    configured: int,
    *,
    field: str,
) -> int:
    value = _metadata_value(request.metadata, keys)
    if value is None:
        return configured
    return _integer(value, field=field, default=configured)


def _search_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    path = parsed.path.rstrip("/")
    if not path:
        path = "/api/v1/search"
    elif path.endswith("/api/v1") or not path.endswith(("/search", "/search_by_date")):
        path = f"{path}/search"
    return urlunsplit(
        (parsed.scheme, parsed.netloc, path, parsed.query, parsed.fragment)
    )


def _timestamp(hit: Mapping[str, Any]) -> datetime | None:
    for key in ("created_at_i", "created_at"):
        if key not in hit:
            continue
        value = hit[key]
        if key == "created_at" and isinstance(value, str):
            text = value.strip().replace("Z", "+00:00")
            try:
                parsed = datetime.fromisoformat(text)
            except ValueError:
                continue
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return parsed.astimezone(UTC)
        if isinstance(value, bool):
            return None
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(seconds) or seconds < 0:
            return None
        return datetime.fromtimestamp(seconds, tz=UTC)
    return None


def _int_field(hit: Mapping[str, Any], keys: Sequence[str]) -> int | None:
    for key in keys:
        if key not in hit:
            continue
        value = hit[key]
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            if math.isfinite(value) and value.is_integer():
                return int(value)
            return None
        if isinstance(value, str):
            text = value.strip().replace(",", "").replace("_", "")
            try:
                return int(text)
            except ValueError:
                return None
        return None
    return None


def _hit_title(hit: Mapping[str, Any]) -> str:
    for key in ("title", "story_title"):
        value = hit.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _hn_item_url(hit: Mapping[str, Any]) -> str:
    for key in ("objectID", "story_id", "id"):
        value = hit.get(key)
        if isinstance(value, str) and value.strip():
            return f"https://news.ycombinator.com/item?id={value.strip()}"
        if isinstance(value, int) and not isinstance(value, bool):
            return f"https://news.ycombinator.com/item?id={value}"
    return ""


def _hit_url(hit: Mapping[str, Any]) -> str:
    for key in ("url", "story_url"):
        value = hit.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        text = value.strip()
        if text.startswith(("http://", "https://")):
            return text
    return _hn_item_url(hit)


def _hit_summary(points: int, comments: int) -> str:
    parts = [f"{points:,} points"]
    if comments:
        parts.append(f"{comments:,} comments")
    return f"{' and '.join(parts)} on Hacker News"


def _hits(body: Mapping[str, Any]) -> list[Mapping[str, Any]] | None:
    for key in ("hits", "results"):
        if key not in body:
            continue
        raw = body[key]
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            return None
        return [entry for entry in raw if isinstance(entry, Mapping)]
    return None


def _candidate(
    hit: Mapping[str, Any],
    request: SourceRequest,
    rank: int,
    *,
    min_points: int,
    min_comments: int,
    start: datetime | None,
    end: datetime | None,
) -> TopicCandidate | None:
    title = _hit_title(hit)
    if not title or title.casefold() in {"[dead]", "[flagged]"}:
        return None
    points = _int_field(hit, ("points", "score"))
    if points is None or points < min_points:
        return None
    comments = _int_field(hit, ("num_comments", "comments"))
    if comments is None:
        comments = 0
    if comments < min_comments:
        return None
    created = _timestamp(hit)
    if start is not None or end is not None:
        if created is None:
            return None
        if start is not None and created < start:
            return None
        if end is not None and created > end:
            return None

    url = _hit_url(hit)
    source_urls = [url] if url else []
    try:
        return TopicCandidate(
            title=title,
            summary=_hit_summary(points, comments),
            source_urls=source_urls,
            mode=request.mode,
            category="hacker-news",
            signal=float(points),
            rank=rank,
        )
    except PydanticValidationError as exc:
        raise ValidationError(
            "hacker news returned a malformed story",
            details={"provider": "hacker-news"},
        ) from exc


def _raise_for_response(response: httpx.Response, provider: str) -> None:
    error = error_for_response(response)
    if error is None:
        return
    details = dict(error.details)
    details["provider"] = provider
    if isinstance(error, RateLimitError):
        raise RateLimitError(
            error.message,
            retry_after=error.retry_after,
            details=details,
        ) from None
    if isinstance(error, TimeoutError):
        raise TimeoutError(error.message, details=details) from None
    if isinstance(error, RetryableError):
        raise RetryableError(error.message, details=details) from None
    raise type(error)(error.message, details=details) from None


def _raise_body_error(
    body: Mapping[str, Any],
    response: httpx.Response,
    provider: str,
) -> None:
    error = body.get("error")
    message = body.get("message")
    if error is None and not isinstance(message, str):
        return
    text = message.strip() if isinstance(message, str) else str(error).strip()
    lowered = text.casefold()
    details: dict[str, Any] = {
        "provider": provider,
        "status": response.status_code,
        "error": error,
    }
    if (
        error in {429, "429"}
        or "too many requests" in lowered
        or "rate limit" in lowered
    ):
        raise RateLimitError(
            "hacker news rate limit reached",
            retry_after=_retry_after_seconds(response),
            details=details,
        )
    raise InvalidRequestError(
        f"hacker news rejected the request: {text}",
        details=details,
    )


class HackerNewsSourceProvider(SourceProviderBase):
    """A topic source backed by the public Hacker News Algolia index.

    ``window`` is the default story window and ``min_points`` and
    ``min_comments`` filter weak stories. A request may override each of these
    through its metadata. ``transport`` injects an ``httpx`` mock transport so
    the offline suite never touches the network.
    """

    default_metadata = _DEFAULT_METADATA

    def __init__(
        self,
        *,
        base_url: str | None = None,
        window: str | None = None,
        timeframe: str | None = None,
        min_points: Any = None,
        min_score: Any = None,
        min_comments: Any = None,
        thresholds: Mapping[str, Any] | None = None,
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
        if min_points is not None and min_score is not None:
            raise ConfigError(
                "use either min_points or min_score, not both",
                details={"field": "min_points"},
            )
        if window is not None and timeframe is not None:
            raise ConfigError(
                "use either window or timeframe, not both",
                details={"field": "window"},
            )
        if thresholds is not None:
            if not isinstance(thresholds, Mapping):
                raise ConfigError(
                    "thresholds must be a mapping",
                    details={"field": "thresholds"},
                )
            if min_points is None and min_score is None:
                min_points = _metadata_value(thresholds, _POINT_KEYS)
            if min_comments is None:
                min_comments = _metadata_value(thresholds, _COMMENT_KEYS)

        self._base_url = _validate_base_url(
            DEFAULT_HN_BASE_URL if base_url is None else base_url
        )
        self._window = normalize_window(
            window if window is not None else timeframe,
            default=DEFAULT_HN_WINDOW,
        )
        self._min_points = _integer(
            min_points if min_points is not None else min_score,
            field="min_points",
            default=DEFAULT_HN_MIN_POINTS,
        )
        self._min_comments = _integer(
            min_comments,
            field="min_comments",
            default=DEFAULT_HN_MIN_COMMENTS,
        )
        if transport is not None:
            self._http = self.create_http_client(transport=transport)

    @property
    def base_url(self) -> str:
        """The configured Algolia endpoint."""
        return self._base_url

    @property
    def window(self) -> str:
        """The default story window."""
        return self._window

    @property
    def min_points(self) -> int:
        """The default minimum story points."""
        return self._min_points

    @property
    def min_score(self) -> int:
        """Alias for :attr:`min_points`."""
        return self._min_points

    @property
    def min_comments(self) -> int:
        """The default minimum comment count."""
        return self._min_comments

    @property
    def endpoint(self) -> str:
        """The Algolia search URL."""
        return _search_url(self._base_url)

    @property
    def api_url(self) -> str:
        """Alias for :attr:`endpoint`."""
        return self.endpoint

    def _headers(self) -> dict[str, str]:
        return {
            "accept": "application/json",
            "user-agent": DEFAULT_USER_AGENT,
        }

    def _params(
        self,
        request: SourceRequest,
        window: str,
        min_points: int,
        min_comments: int,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "tags": "story",
            "hitsPerPage": min(request.limit, MAX_RESULTS),
        }
        if request.query:
            params["query"] = request.query

        filters: list[str] = []
        if request.window_start is not None:
            filters.append(f"created_at_i>={int(request.window_start.timestamp())}")
        elif window != "all":
            since = datetime.now(UTC).timestamp() - WINDOW_SECONDS[window]
            filters.append(f"created_at_i>={int(since)}")
        if request.window_end is not None:
            filters.append(f"created_at_i<={int(request.window_end.timestamp())}")
        if min_points > 0:
            filters.append(f"points>={min_points}")
        if min_comments > 0:
            filters.append(f"num_comments>={min_comments}")
        if filters:
            params["numericFilters"] = ",".join(filters)
        return params

    async def discover(self, request: SourceRequest) -> SourceResponse:
        """Discover popular or matching Hacker News stories for *request*."""
        if not isinstance(request, SourceRequest):
            raise TypeError("request must be a SourceRequest")

        try:
            window = _resolve_window(request, self._window)
            min_points = _resolve_integer(
                request,
                _POINT_KEYS,
                self._min_points,
                field="min_points",
            )
            min_comments = _resolve_integer(
                request,
                _COMMENT_KEYS,
                self._min_comments,
                field="min_comments",
            )
        except ConfigError as exc:
            raise InvalidRequestError(
                str(exc),
                details={"provider": self.name},
            ) from exc

        response = await self.http.get(
            self.endpoint,
            headers=self._headers(),
            params=self._params(request, window, min_points, min_comments),
            timeout=request.timeout_seconds,
        )
        _raise_for_response(response, self.name)

        try:
            body = response.json()
        except ValueError as exc:
            raise ValidationError(
                "hacker news returned a non-JSON response",
                details={"provider": self.name},
            ) from exc
        if not isinstance(body, Mapping):
            raise ValidationError(
                "hacker news returned a malformed response",
                details={"provider": self.name},
            )
        _raise_body_error(body, response, self.name)

        hits = _hits(body)
        if hits is None:
            raise ValidationError(
                "hacker news returned a malformed response",
                details={"provider": self.name},
            )

        start = request.window_start
        end = request.window_end
        candidates: list[TopicCandidate] = []
        for hit in hits:
            if len(candidates) >= request.limit:
                break
            candidate = _candidate(
                hit,
                request,
                len(candidates),
                min_points=min_points,
                min_comments=min_comments,
                start=start,
                end=end,
            )
            if candidate is not None:
                candidates.append(candidate)

        metadata: dict[str, Any] = {
            "window": window,
            "min_points": min_points,
            "min_comments": min_comments,
            "total": len(candidates),
        }
        if request.query:
            metadata["query"] = request.query
        return SourceResponse(
            provider=self.name,
            query=request.query,
            candidates=candidates,
            metadata=metadata,
        )


HackerNewsSource = HackerNewsSourceProvider
"""Alias for :class:`HackerNewsSourceProvider`."""

HackerNewsProvider = HackerNewsSourceProvider
"""Alias for :class:`HackerNewsSourceProvider`."""

HackerNewsAPI = HackerNewsSourceProvider
"""Alias for :class:`HackerNewsSourceProvider`."""

HackerNews = HackerNewsSourceProvider
"""Alias for :class:`HackerNewsSourceProvider`."""

HNSource = HackerNewsSourceProvider
"""Alias for :class:`HackerNewsSourceProvider`."""

HN = HackerNewsSourceProvider
"""Alias for :class:`HackerNewsSourceProvider`."""

register_provider("hacker-news", HackerNewsSourceProvider, module=__name__)
register_provider("hacker_news", HackerNewsSourceProvider, module=__name__)
register_provider("hackernews", HackerNewsSourceProvider, module=__name__)
register_provider("hn", HackerNewsSourceProvider, module=__name__)
register_source("hacker-news", HackerNewsSourceProvider, module=__name__)
register_source("hacker_news", HackerNewsSourceProvider, module=__name__)
register_source("hackernews", HackerNewsSourceProvider, module=__name__)
register_source("hn", HackerNewsSourceProvider, module=__name__)
