"""Reddit topic source for popular posts in configurable subreddits.

The adapter reads Reddit's public JSON listings and maps each post into a
vendor-neutral :class:`~aivia.providers.protocols.source.SourceResponse`. It
can read the ``top`` listing of one or more subreddits, or search inside them
when the request carries a query. A post's score becomes the candidate
``signal`` so later research stages can rank or gate on community interest
without parsing prose.

Subreddits, the listing window, the minimum score, comment count and upvote
ratio are provider settings that a request may override through its metadata.
The request window also narrows the listing to the smallest coarse Reddit
window that covers it, and any explicit request bounds filter posts by their
creation time.

The provider owns no routing, retry or budget policy. Transport and retry
behaviour come from the shared HTTP client, a rejected rate limit maps to
:class:`~aivia.core.errors.RateLimitError` with the server's ``Retry-After``
delay, and every other failure maps to the shared error taxonomy.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic import ValidationError as PydanticValidationError

from aivia.core.config.http import HttpClientConfig
from aivia.core.errors import (
    AuthError,
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
    "DEFAULT_MIN_COMMENTS",
    "DEFAULT_MIN_RATIO",
    "DEFAULT_MIN_SCORE",
    "DEFAULT_REDDIT_BASE_URL",
    "DEFAULT_REDDIT_WINDOW",
    "DEFAULT_SUBREDDIT",
    "DEFAULT_SUBREDDITS",
    "MAX_RESULTS",
    "REDDIT_WINDOWS",
    "Reddit",
    "RedditAPI",
    "RedditProvider",
    "RedditSource",
    "RedditSourceProvider",
    "normalize_subreddits",
]

DEFAULT_REDDIT_BASE_URL = "https://www.reddit.com"
"""The public Reddit origin; ``{subreddit}`` may appear in a custom URL."""

DEFAULT_SUBREDDIT = "popular"
"""The subreddit read when the caller configures none."""

DEFAULT_SUBREDDITS = (DEFAULT_SUBREDDIT,)
"""The default subreddit list."""

DEFAULT_REDDIT_WINDOW = DEFAULT_WINDOW
"""The default top-listing window."""

REDDIT_WINDOWS = WINDOW_NAMES
"""Every window Reddit's ``top`` listing accepts."""

MAX_RESULTS = 100
"""The largest listing Reddit returns in one call."""

DEFAULT_MIN_SCORE = 0
"""The default minimum post score."""

DEFAULT_MIN_COMMENTS = 0
"""The default minimum comment count."""

DEFAULT_MIN_RATIO = 0.0
"""The default minimum upvote ratio."""

_SUBREDDIT_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_+-]{0,49}$")
_REMOVED_TITLES = {"[deleted]", "[removed]"}
_DEFAULT_METADATA = ProviderMeta(
    name="reddit",
    capabilities=[CAPABILITY_SOURCE],
)
_SUBREDDIT_KEYS = ("subreddits", "subreddit")
_SCORE_KEYS = ("min_score", "min_upvotes", "score_threshold", "threshold", "score")
_COMMENT_KEYS = ("min_comments", "comments_threshold", "comments")
_RATIO_KEYS = ("min_ratio", "ratio_threshold", "upvote_ratio")


def normalize_subreddits(
    value: str | Sequence[str] | None,
) -> tuple[str, ...]:
    """Return validated, lower-case subreddit names or raise :class:`ConfigError`.

    ``None`` selects :data:`DEFAULT_SUBREDDITS`. A comma or whitespace
    separated string is accepted because config files often store the list in
    one scalar. A leading ``r/`` is removed, duplicates are dropped and the
    configured order is kept.
    """
    if value is None:
        return DEFAULT_SUBREDDITS
    if isinstance(value, str):
        raw = [part for part in re.split(r"[,\s]+", value) if part]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw = list(value)
    else:
        raise ConfigError(
            "subreddits must be a string or a sequence of strings",
            details={"field": "subreddits"},
        )

    subreddits: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            raise ConfigError(
                "subreddits must contain only strings",
                details={"field": "subreddits"},
            )
        name = item.strip().lower()
        if name.startswith("/r/"):
            name = name[3:]
        elif name.startswith("r/"):
            name = name[2:]
        if not name or not _SUBREDDIT_PATTERN.fullmatch(name):
            raise ConfigError(
                f"invalid subreddit name: {item!r}",
                details={"field": "subreddits"},
            )
        if name not in subreddits:
            subreddits.append(name)

    return tuple(subreddits) or DEFAULT_SUBREDDITS


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


def _ratio(value: Any, *, field: str, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ConfigError(
            f"{field} must be a number between 0 and 1",
            details={"field": field},
        )
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"{field} must be a number between 0 and 1",
            details={"field": field},
        ) from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ConfigError(
            f"{field} must be a number between 0 and 1",
            details={"field": field},
        )
    return number


def _metadata_value(
    metadata: Mapping[str, Any],
    keys: Sequence[str],
) -> Any:
    for key in keys:
        if key in metadata and metadata[key] is not None:
            return metadata[key]
    return None


def _resolve_subreddits(
    request: SourceRequest,
    configured: tuple[str, ...],
) -> tuple[str, ...]:
    value = _metadata_value(request.metadata, _SUBREDDIT_KEYS)
    if value is None:
        return configured
    return normalize_subreddits(value)


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


def _resolve_ratio(request: SourceRequest, configured: float) -> float:
    value = _metadata_value(request.metadata, _RATIO_KEYS)
    if value is None:
        return configured
    return _ratio(value, field="min_ratio", default=configured)


def _endpoint(
    base_url: str,
    subreddit: str,
    *,
    query: str | None = None,
) -> str:
    template = base_url.replace("{subreddit}", subreddit)
    parsed = urlsplit(template)
    path = parsed.path.rstrip("/")
    if path.endswith(".json"):
        return urlunsplit(
            (parsed.scheme, parsed.netloc, path, parsed.query, parsed.fragment)
        )
    if "/r/" not in path:
        path = f"{path}/r/{subreddit}" if path else f"/r/{subreddit}"
    if query:
        if not path.endswith("/search"):
            path = f"{path}/search"
    elif not path.endswith("/top"):
        path = f"{path}/top"
    path = f"{path}.json"
    return urlunsplit(
        (parsed.scheme, parsed.netloc, path, parsed.query, parsed.fragment)
    )


def _timestamp(post: Mapping[str, Any]) -> datetime | None:
    for key in ("created_utc", "created"):
        if key not in post:
            continue
        value = post[key]
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


def _int_field(post: Mapping[str, Any], keys: Sequence[str]) -> int | None:
    for key in keys:
        if key not in post:
            continue
        value = post[key]
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


def _float_field(post: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        if key not in post:
            continue
        value = post[key]
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if math.isfinite(number):
            return number
        return None
    return None


def _post_title(post: Mapping[str, Any]) -> str:
    for key in ("title", "link_title"):
        value = post.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _post_subreddit(post: Mapping[str, Any], fallback: str) -> str:
    value = post.get("subreddit")
    if isinstance(value, str) and value.strip():
        name = value.strip()
        if name.startswith("/r/"):
            name = name[3:]
        elif name.startswith("r/"):
            name = name[2:]
        return name or fallback
    return fallback


def _post_url(post: Mapping[str, Any], subreddit: str) -> str:
    for key in ("url", "url_overridden_by_dest", "permalink"):
        value = post.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        text = value.strip()
        if text.startswith("/"):
            return f"https://www.reddit.com{text}"
        if text.startswith(("http://", "https://")):
            return text
    post_id = post.get("id")
    if isinstance(post_id, str) and post_id.strip():
        return f"https://www.reddit.com/r/{subreddit}/comments/{post_id.strip()}"
    return ""


def _post_summary(score: int, comments: int, ratio: float, subreddit: str) -> str:
    parts = [f"{score:,} upvotes"]
    if comments:
        parts.append(f"{comments:,} comments")
    if ratio > 0.0:
        parts.append(f"{ratio:.0%} upvoted")
    return f"{' and '.join(parts)} in r/{subreddit}"


def _children(body: Mapping[str, Any]) -> list[Mapping[str, Any]] | None:
    data = body.get("data")
    container: Any = data if isinstance(data, Mapping) else body
    raw = container.get("children")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return None
    children: list[Mapping[str, Any]] = []
    for child in raw:
        if not isinstance(child, Mapping):
            continue
        post = child.get("data")
        if isinstance(post, Mapping):
            children.append(post)
    return children


def _candidate(
    post: Mapping[str, Any],
    request: SourceRequest,
    subreddit: str,
    rank: int,
    *,
    min_score: int,
    min_comments: int,
    min_ratio: float,
    start: datetime | None,
    end: datetime | None,
) -> TopicCandidate | None:
    title = _post_title(post)
    if not title or title.casefold() in _REMOVED_TITLES:
        return None
    if post.get("removed_by_category") is not None:
        return None
    score = _int_field(post, ("score", "ups"))
    if score is None or score < min_score:
        return None
    comments = _int_field(post, ("num_comments", "comments"))
    if comments is None:
        comments = 0
    if comments < min_comments:
        return None
    ratio = _float_field(post, ("upvote_ratio",))
    if ratio is None:
        ratio = 1.0
    if ratio < min_ratio:
        return None
    created = _timestamp(post)
    if start is not None or end is not None:
        if created is None:
            return None
        if start is not None and created < start:
            return None
        if end is not None and created > end:
            return None

    category = _post_subreddit(post, subreddit)
    summary = _post_summary(score, comments, ratio, category)
    url = _post_url(post, category)
    source_urls = [url] if url else []
    try:
        return TopicCandidate(
            title=title,
            summary=summary,
            source_urls=source_urls,
            mode=request.mode,
            category=category,
            signal=float(score),
            rank=rank,
        )
    except PydanticValidationError as exc:
        raise ValidationError(
            "reddit returned a malformed post",
            details={"provider": "reddit", "subreddit": subreddit},
        ) from exc


def _raise_for_response(
    response: httpx.Response,
    subreddit: str,
    provider: str,
) -> None:
    error = error_for_response(response)
    if error is None:
        return
    details = dict(error.details)
    details.update({"provider": provider, "subreddit": subreddit})
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
    subreddit: str,
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
        "subreddit": subreddit,
        "status": response.status_code,
        "error": error,
    }
    if (
        error in {429, "429"}
        or "too many requests" in lowered
        or "rate limit" in lowered
    ):
        raise RateLimitError(
            f"reddit rate limit reached for r/{subreddit}",
            retry_after=_retry_after_seconds(response),
            details=details,
        )
    if error in {401, 403, "401", "403"} or "forbidden" in lowered:
        raise AuthError(
            f"reddit rejected access to r/{subreddit}",
            details=details,
        )
    raise InvalidRequestError(
        f"reddit rejected the request for r/{subreddit}: {text}",
        details=details,
    )


class RedditSourceProvider(SourceProviderBase):
    """A topic source backed by Reddit's public JSON listings.

    ``subreddits`` is the default list read for every request. ``window`` is
    the top-listing range and ``min_score``, ``min_comments`` and
    ``min_ratio`` filter weak posts. A request may override each of these
    through its metadata. ``transport`` injects an ``httpx`` mock transport so
    the offline suite never touches the network.
    """

    default_metadata = _DEFAULT_METADATA

    def __init__(
        self,
        *,
        subreddits: str | Sequence[str] | None = None,
        subreddit: str | None = None,
        base_url: str | None = None,
        window: str | None = None,
        timeframe: str | None = None,
        min_score: Any = None,
        min_upvotes: Any = None,
        min_comments: Any = None,
        min_ratio: Any = None,
        min_upvote_ratio: Any = None,
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
        if subreddits is not None and subreddit is not None:
            raise ConfigError(
                "use either subreddits or subreddit, not both",
                details={"field": "subreddits"},
            )
        if window is not None and timeframe is not None:
            raise ConfigError(
                "use either window or timeframe, not both",
                details={"field": "window"},
            )
        if min_score is not None and min_upvotes is not None:
            raise ConfigError(
                "use either min_score or min_upvotes, not both",
                details={"field": "min_score"},
            )
        if min_ratio is not None and min_upvote_ratio is not None:
            raise ConfigError(
                "use either min_ratio or min_upvote_ratio, not both",
                details={"field": "min_ratio"},
            )
        if thresholds is not None:
            if not isinstance(thresholds, Mapping):
                raise ConfigError(
                    "thresholds must be a mapping",
                    details={"field": "thresholds"},
                )
            if min_score is None and min_upvotes is None:
                min_score = _metadata_value(thresholds, _SCORE_KEYS)
            if min_comments is None:
                min_comments = _metadata_value(thresholds, _COMMENT_KEYS)
            if min_ratio is None and min_upvote_ratio is None:
                min_ratio = _metadata_value(thresholds, _RATIO_KEYS)

        self._subreddits = normalize_subreddits(
            subreddits if subreddits is not None else subreddit
        )
        self._base_url = _validate_base_url(
            DEFAULT_REDDIT_BASE_URL if base_url is None else base_url
        )
        self._window = normalize_window(
            window if window is not None else timeframe,
            default=DEFAULT_REDDIT_WINDOW,
        )
        self._min_score = _integer(
            min_score if min_score is not None else min_upvotes,
            field="min_score",
            default=DEFAULT_MIN_SCORE,
        )
        self._min_comments = _integer(
            min_comments,
            field="min_comments",
            default=DEFAULT_MIN_COMMENTS,
        )
        self._min_ratio = _ratio(
            min_ratio if min_ratio is not None else min_upvote_ratio,
            field="min_ratio",
            default=DEFAULT_MIN_RATIO,
        )
        if transport is not None:
            self._http = self.create_http_client(transport=transport)

    @property
    def subreddits(self) -> tuple[str, ...]:
        """The default subreddit list, in configured order."""
        return self._subreddits

    @property
    def base_url(self) -> str:
        """The configured Reddit origin or URL template."""
        return self._base_url

    @property
    def window(self) -> str:
        """The default top-listing window."""
        return self._window

    @property
    def min_score(self) -> int:
        """The default minimum post score."""
        return self._min_score

    @property
    def min_upvotes(self) -> int:
        """Alias for :attr:`min_score`."""
        return self._min_score

    @property
    def min_comments(self) -> int:
        """The default minimum comment count."""
        return self._min_comments

    @property
    def min_ratio(self) -> float:
        """The default minimum upvote ratio."""
        return self._min_ratio

    @property
    def min_upvote_ratio(self) -> float:
        """Alias for :attr:`min_ratio`."""
        return self._min_ratio

    @property
    def endpoint(self) -> str:
        """The top-listing URL for the first configured subreddit."""
        return self.endpoint_for()

    @property
    def api_url(self) -> str:
        """Alias for :attr:`endpoint`."""
        return self.endpoint

    def endpoint_for(
        self,
        subreddit: str | None = None,
        *,
        query: str | None = None,
    ) -> str:
        """Return the listing URL for *subreddit*, or the first configured one."""
        name = (
            self._subreddits[0]
            if subreddit is None
            else (normalize_subreddits(subreddit)[0])
        )
        return _endpoint(self._base_url, name, query=query)

    def _headers(self) -> dict[str, str]:
        return {
            "accept": "application/json",
            "user-agent": DEFAULT_USER_AGENT,
        }

    def _params(
        self,
        request: SourceRequest,
        window: str,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "limit": min(request.limit, MAX_RESULTS),
            "raw_json": "1",
            "t": window,
        }
        if request.query:
            params.update(
                {
                    "q": request.query,
                    "restrict_sr": "1",
                    "sort": "top",
                }
            )
        return params

    async def discover(self, request: SourceRequest) -> SourceResponse:
        """Discover popular or matching Reddit posts for *request*."""
        if not isinstance(request, SourceRequest):
            raise TypeError("request must be a SourceRequest")

        try:
            subreddits = _resolve_subreddits(request, self._subreddits)
            window = _resolve_window(request, self._window)
            min_score = _resolve_integer(
                request,
                _SCORE_KEYS,
                self._min_score,
                field="min_score",
            )
            min_comments = _resolve_integer(
                request,
                _COMMENT_KEYS,
                self._min_comments,
                field="min_comments",
            )
            min_ratio = _resolve_ratio(request, self._min_ratio)
        except ConfigError as exc:
            raise InvalidRequestError(
                str(exc),
                details={"provider": self.name},
            ) from exc

        start = request.window_start
        end = request.window_end
        candidates: list[TopicCandidate] = []

        for subreddit in subreddits:
            endpoint = self.endpoint_for(subreddit, query=request.query)
            response = await self.http.get(
                endpoint,
                headers=self._headers(),
                params=self._params(request, window),
                timeout=request.timeout_seconds,
            )
            _raise_for_response(response, subreddit, self.name)

            try:
                body = response.json()
            except ValueError as exc:
                raise ValidationError(
                    "reddit returned a non-JSON response",
                    details={"provider": self.name, "subreddit": subreddit},
                ) from exc
            if not isinstance(body, Mapping):
                raise ValidationError(
                    "reddit returned a malformed response",
                    details={"provider": self.name, "subreddit": subreddit},
                )
            _raise_body_error(body, response, subreddit, self.name)

            posts = _children(body)
            if posts is None:
                raise ValidationError(
                    "reddit returned a malformed response",
                    details={"provider": self.name, "subreddit": subreddit},
                )

            for post in posts:
                if len(candidates) >= request.limit:
                    break
                candidate = _candidate(
                    post,
                    request,
                    subreddit,
                    len(candidates),
                    min_score=min_score,
                    min_comments=min_comments,
                    min_ratio=min_ratio,
                    start=start,
                    end=end,
                )
                if candidate is not None:
                    candidates.append(candidate)

        metadata: dict[str, Any] = {
            "subreddits": list(subreddits),
            "window": window,
            "min_score": min_score,
            "min_comments": min_comments,
            "min_ratio": min_ratio,
            "total": len(candidates),
        }
        if len(subreddits) == 1:
            metadata["subreddit"] = subreddits[0]
        if request.query:
            metadata["query"] = request.query
        return SourceResponse(
            provider=self.name,
            query=request.query,
            candidates=candidates,
            metadata=metadata,
        )


RedditSource = RedditSourceProvider
"""Alias for :class:`RedditSourceProvider`."""

RedditProvider = RedditSourceProvider
"""Alias for :class:`RedditSourceProvider`."""

RedditAPI = RedditSourceProvider
"""Alias for :class:`RedditSourceProvider`."""

Reddit = RedditSourceProvider
"""Alias for :class:`RedditSourceProvider`."""

register_provider("reddit", RedditSourceProvider, module=__name__)
register_provider("reddit-source", RedditSourceProvider, module=__name__)
register_source("reddit", RedditSourceProvider, module=__name__)
register_source("reddit-source", RedditSourceProvider, module=__name__)
