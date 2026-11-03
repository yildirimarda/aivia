"""Reddit and Hacker News sources: windows, thresholds and rate limits."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from aivia.core.config.http import HttpClientConfig
from aivia.core.errors import (
    AuthError,
    ConfigError,
    InvalidRequestError,
    RateLimitError,
    RetryableError,
    ValidationError,
)
from aivia.providers import provider_registry
from aivia.providers.protocols import CAPABILITY_SOURCE, SourceRequest
from aivia.providers.sources import (
    DEFAULT_HN_BASE_URL,
    DEFAULT_HN_MIN_COMMENTS,
    DEFAULT_HN_MIN_POINTS,
    DEFAULT_HN_WINDOW,
    DEFAULT_MIN_COMMENTS,
    DEFAULT_MIN_RATIO,
    DEFAULT_MIN_SCORE,
    DEFAULT_REDDIT_BASE_URL,
    DEFAULT_REDDIT_WINDOW,
    DEFAULT_SUBREDDITS,
    HN,
    HackerNews,
    HackerNewsAPI,
    HackerNewsProvider,
    HackerNewsSource,
    HackerNewsSourceProvider,
    HNSource,
    Reddit,
    RedditAPI,
    RedditProvider,
    RedditSource,
    RedditSourceProvider,
    normalize_subreddits,
    resolve_source,
)

NOW = datetime(2026, 5, 5, 12, tzinfo=UTC)
NOW_TS = int(NOW.timestamp())


def _post(
    title: str,
    *,
    score: Any = 100,
    comments: Any = 10,
    ratio: Any = 0.95,
    subreddit: str = "python",
    created: Any = NOW_TS,
    url: str | None = None,
    permalink: str | None = None,
    post_id: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    post: dict[str, Any] = {
        "title": title,
        "score": score,
        "num_comments": comments,
        "upvote_ratio": ratio,
        "subreddit": subreddit,
        "created_utc": created,
    }
    if url is not None:
        post["url"] = url
    if permalink is not None:
        post["permalink"] = permalink
    if post_id is not None:
        post["id"] = post_id
    post.update(extra)
    return post


def _reddit_body(*posts: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "Listing",
        "data": {
            "children": [{"kind": "t3", "data": post} for post in posts],
        },
    }


def _hit(
    title: str,
    *,
    points: Any = 100,
    comments: Any = 10,
    url: str | None = None,
    object_id: str | None = "1",
    created: Any = NOW_TS,
    **extra: Any,
) -> dict[str, Any]:
    hit: dict[str, Any] = {
        "title": title,
        "points": points,
        "num_comments": comments,
        "created_at_i": created,
    }
    if url is not None:
        hit["url"] = url
    if object_id is not None:
        hit["objectID"] = object_id
    hit.update(extra)
    return hit


def _hn_body(*hits: dict[str, Any]) -> dict[str, Any]:
    return {"hits": list(hits), "nbHits": len(hits), "page": 0}


def _reddit(
    handler: Callable[[httpx.Request], httpx.Response],
    **kwargs: Any,
) -> RedditSourceProvider:
    return RedditSourceProvider(
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def _hn(
    handler: Callable[[httpx.Request], httpx.Response],
    **kwargs: Any,
) -> HackerNewsSourceProvider:
    return HackerNewsSourceProvider(
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def _capture(
    body: Any = None,
    status: int = 200,
    headers: dict[str, str] | None = None,
) -> tuple[list[httpx.Request], Callable[[httpx.Request], httpx.Response]]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status, json=body, headers=headers)

    return requests, handler


def _fast_http() -> HttpClientConfig:
    return HttpClientConfig(max_attempts=1)


def test_reddit_metadata_registration_and_aliases() -> None:
    provider = _reddit(lambda request: httpx.Response(200, json=_reddit_body()))

    assert provider.name == "reddit"
    assert provider.capabilities == (CAPABILITY_SOURCE,)
    assert provider.subreddits == DEFAULT_SUBREDDITS == ("popular",)
    assert provider.base_url == DEFAULT_REDDIT_BASE_URL
    assert provider.window == DEFAULT_REDDIT_WINDOW == "day"
    assert provider.min_score == DEFAULT_MIN_SCORE == 0
    assert provider.min_comments == DEFAULT_MIN_COMMENTS == 0
    assert provider.min_ratio == DEFAULT_MIN_RATIO == 0.0
    assert provider.endpoint == "https://www.reddit.com/r/popular/top.json"
    assert provider.api_url == provider.endpoint

    assert provider_registry.resolve("reddit") is RedditSourceProvider
    assert provider_registry.resolve("reddit-source") is RedditSourceProvider
    assert resolve_source("reddit") is RedditSourceProvider
    assert resolve_source("reddit-source") is RedditSourceProvider
    assert RedditSource is RedditSourceProvider
    assert RedditProvider is RedditSourceProvider
    assert RedditAPI is RedditSourceProvider
    assert Reddit is RedditSourceProvider


def test_reddit_maps_top_posts_with_a_score_signal() -> None:
    body = _reddit_body(
        _post(
            "Tide",
            score=1234,
            comments=56,
            ratio=0.95,
            url="https://example.com/tide",
        ),
        _post(
            "Ocean",
            score=500,
            comments=10,
            ratio=0.80,
            permalink="/r/python/comments/abc/ocean/",
        ),
    )
    requests, handler = _capture(body)
    provider = _reddit(handler, subreddits="python")

    response = asyncio.run(
        provider.discover(
            SourceRequest(
                limit=2,
                mode="trending",
                metadata={"window": "day"},
            )
        )
    )

    sent = requests[0]
    assert sent.method == "GET"
    assert sent.url.path == "/r/python/top.json"
    assert sent.url.params["t"] == "day"
    assert sent.url.params["limit"] == "2"
    assert sent.url.params["raw_json"] == "1"
    assert sent.headers["user-agent"]

    assert response.provider == "reddit"
    assert response.titles == ["Tide", "Ocean"]
    assert [candidate.signal for candidate in response.candidates] == [
        1234.0,
        500.0,
    ]
    assert [candidate.rank for candidate in response.candidates] == [0, 1]
    assert [candidate.category for candidate in response.candidates] == [
        "python",
        "python",
    ]
    assert [candidate.mode for candidate in response.candidates] == [
        "trending",
        "trending",
    ]
    assert response.first is not None
    assert (
        response.first.summary
        == "1,234 upvotes and 56 comments and 95% upvoted in r/python"
    )
    assert response.first.source_urls == ["https://example.com/tide"]
    assert response.candidates[1].source_urls == [
        "https://www.reddit.com/r/python/comments/abc/ocean/"
    ]
    assert response.metadata == {
        "subreddits": ["python"],
        "subreddit": "python",
        "window": "day",
        "min_score": 0,
        "min_comments": 0,
        "min_ratio": 0.0,
        "total": 2,
    }


def test_reddit_reads_each_configured_subreddit_and_applies_the_limit() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if "/r/python/" in request.url.path:
            return httpx.Response(
                200,
                json=_reddit_body(
                    _post("Python one", subreddit="python"),
                    _post("Python two", subreddit="python"),
                ),
            )
        return httpx.Response(
            200,
            json=_reddit_body(
                _post("Rust one", subreddit="rust"),
                _post("Rust two", subreddit="rust"),
            ),
        )

    provider = _reddit(handler, subreddits=["python", "rust"])

    response = asyncio.run(provider.discover(SourceRequest(limit=3)))

    assert [request.url.path for request in requests] == [
        "/r/python/top.json",
        "/r/rust/top.json",
    ]
    assert response.titles == ["Python one", "Python two", "Rust one"]
    assert [candidate.rank for candidate in response.candidates] == [0, 1, 2]
    assert response.metadata["subreddits"] == ["python", "rust"]
    assert "subreddit" not in response.metadata


def test_reddit_thresholds_filter_weak_posts_and_request_metadata_wins() -> None:
    body = _reddit_body(
        _post("Strong", score=500, comments=50, ratio=0.90),
        _post("Low score", score=50, comments=50, ratio=0.90),
        _post("Few comments", score=500, comments=5, ratio=0.90),
        _post("Controversial", score=500, comments=50, ratio=0.50),
    )
    _requests, handler = _capture(body)
    provider = _reddit(handler, min_score=10, min_comments=10, min_ratio=0.8)

    response = asyncio.run(
        provider.discover(
            SourceRequest(
                metadata={
                    "min_score": 100,
                    "min_comments": 20,
                    "min_ratio": 0.85,
                }
            )
        )
    )

    assert response.titles == ["Strong"]
    assert response.metadata["min_score"] == 100
    assert response.metadata["min_comments"] == 20
    assert response.metadata["min_ratio"] == 0.85


def test_reddit_window_metadata_and_request_bounds() -> None:
    body = _reddit_body(
        _post("Recent", created=NOW_TS),
        _post("Old", created=NOW_TS - 9000),
    )
    requests, handler = _capture(body)
    provider = _reddit(handler, window="week")

    response = asyncio.run(
        provider.discover(
            SourceRequest(
                window_start=datetime(2026, 5, 5, 10, tzinfo=UTC),
                window_end=NOW,
                metadata={"timeframe": "month"},
            )
        )
    )

    assert requests[0].url.params["t"] == "month"
    assert response.titles == ["Recent"]
    assert response.metadata["window"] == "month"


def test_reddit_derives_the_window_from_request_bounds() -> None:
    body = _reddit_body(
        _post("Recent", created=NOW_TS),
        _post("Old", created=NOW_TS - 9000),
    )
    requests, handler = _capture(body)
    provider = _reddit(handler)

    response = asyncio.run(
        provider.discover(
            SourceRequest(
                window_start=datetime(2026, 5, 5, 11, 30, tzinfo=UTC),
                window_end=NOW,
            )
        )
    )

    assert requests[0].url.params["t"] == "hour"
    assert response.titles == ["Recent"]


def test_reddit_query_uses_the_search_listing() -> None:
    requests, handler = _capture(_reddit_body(_post("Tide")))
    provider = _reddit(handler, subreddits="python")

    response = asyncio.run(provider.discover(SourceRequest(query="tide")))

    sent = requests[0]
    assert sent.url.path == "/r/python/search.json"
    assert sent.url.params["q"] == "tide"
    assert sent.url.params["restrict_sr"] == "1"
    assert sent.url.params["sort"] == "top"
    assert response.metadata["query"] == "tide"


def test_reddit_skips_removed_and_incomplete_posts() -> None:
    body = _reddit_body(
        _post("[removed]"),
        _post("Gone", removed_by_category="moderator"),
        {"score": 100, "num_comments": 10, "created_utc": NOW_TS},
        {"title": "No score", "num_comments": 10},
        _post("Valid"),
    )
    _requests, handler = _capture(body)
    provider = _reddit(handler)

    response = asyncio.run(provider.discover(SourceRequest()))

    assert response.titles == ["Valid"]
    assert response.first is not None
    assert response.first.rank == 0


@pytest.mark.parametrize(
    "body",
    [
        {"unexpected": True},
        {"data": {"children": "not a list"}},
    ],
)
def test_reddit_a_malformed_response_is_a_validation_error(body: Any) -> None:
    _requests, handler = _capture(body)
    provider = _reddit(handler)

    with pytest.raises(ValidationError, match="malformed"):
        asyncio.run(provider.discover(SourceRequest()))


def test_reddit_a_non_json_response_is_a_validation_error() -> None:
    provider = _reddit(lambda request: httpx.Response(200, text="<html>"))

    with pytest.raises(ValidationError, match="non-JSON"):
        asyncio.run(provider.discover(SourceRequest()))


@pytest.mark.parametrize(
    ("status", "error_type"),
    [
        (400, InvalidRequestError),
        (401, AuthError),
        (403, AuthError),
        (404, InvalidRequestError),
        (429, RateLimitError),
        (500, RetryableError),
        (503, RetryableError),
    ],
)
def test_reddit_http_errors_map_to_the_shared_taxonomy(
    status: int,
    error_type: type[Exception],
) -> None:
    _requests, handler = _capture({"message": "rejected"}, status=status)
    provider = _reddit(handler, http_config=_fast_http())

    with pytest.raises(error_type):
        asyncio.run(provider.discover(SourceRequest()))


def test_reddit_rate_limit_carries_retry_after_and_context() -> None:
    _requests, handler = _capture(
        {"message": "Too Many Requests"},
        status=429,
        headers={"Retry-After": "7"},
    )
    provider = _reddit(handler, subreddits="python", http_config=_fast_http())

    with pytest.raises(RateLimitError) as error:
        asyncio.run(provider.discover(SourceRequest()))

    assert error.value.retry_after == 7.0
    assert error.value.details["subreddit"] == "python"
    assert error.value.details["provider"] == "reddit"


def test_reddit_a_body_rate_limit_is_a_rate_limit_error() -> None:
    body = {"error": 429, "message": "Too Many Requests"}
    _requests, handler = _capture(body)
    provider = _reddit(handler)

    with pytest.raises(RateLimitError, match="rate limit"):
        asyncio.run(provider.discover(SourceRequest()))


def test_reddit_normalize_subreddits_accepts_config_shapes() -> None:
    assert normalize_subreddits(None) == DEFAULT_SUBREDDITS
    assert normalize_subreddits("r/Python, programming") == (
        "python",
        "programming",
    )
    assert normalize_subreddits(["Python", "/r/rust", "python"]) == (
        "python",
        "rust",
    )
    assert normalize_subreddits("all") == ("all",)
    with pytest.raises(ConfigError, match="subreddit"):
        normalize_subreddits(["bad name!"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"subreddits": "python", "subreddit": "rust"},
        {"subreddits": ["bad name!"]},
        {"window": "fortnight"},
        {"min_score": -1},
        {"min_score": 1.5},
        {"min_comments": "many"},
        {"min_ratio": 1.5},
    ],
)
def test_reddit_invalid_config_is_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        RedditSourceProvider(**kwargs)


def test_reddit_constructor_accepts_threshold_aliases_and_a_mapping() -> None:
    aliases = RedditSourceProvider(
        min_upvotes=100,
        min_upvote_ratio=0.75,
        min_comments=20,
    )
    mapping = RedditSourceProvider(
        thresholds={"min_score": 50, "comments": 5, "min_ratio": 0.6}
    )

    assert aliases.min_score == aliases.min_upvotes == 100
    assert aliases.min_ratio == aliases.min_upvote_ratio == 0.75
    assert aliases.min_comments == 20
    assert mapping.min_score == 50
    assert mapping.min_comments == 5
    assert mapping.min_ratio == 0.6


def test_reddit_invalid_request_metadata_is_an_invalid_request() -> None:
    _requests, handler = _capture(_reddit_body(_post("Tide")))
    provider = _reddit(handler)

    with pytest.raises(InvalidRequestError, match="window"):
        asyncio.run(provider.discover(SourceRequest(metadata={"window": "fortnight"})))


def test_reddit_fetch_and_collect_delegate_to_discover() -> None:
    requests, handler = _capture(_reddit_body(_post("Tide")))
    provider = _reddit(handler)

    fetched = asyncio.run(provider.fetch(SourceRequest()))
    collected = asyncio.run(provider.collect(SourceRequest()))

    assert fetched.titles == ["Tide"]
    assert [candidate.title for candidate in collected] == fetched.titles
    assert len(requests) == 2


def test_hacker_news_metadata_registration_and_aliases() -> None:
    provider = _hn(lambda request: httpx.Response(200, json=_hn_body()))

    assert provider.name == "hacker-news"
    assert provider.capabilities == (CAPABILITY_SOURCE,)
    assert provider.base_url == DEFAULT_HN_BASE_URL
    assert provider.window == DEFAULT_HN_WINDOW == "day"
    assert provider.min_points == DEFAULT_HN_MIN_POINTS == 0
    assert provider.min_score == provider.min_points
    assert provider.min_comments == DEFAULT_HN_MIN_COMMENTS == 0
    assert provider.endpoint == DEFAULT_HN_BASE_URL
    assert provider.api_url == provider.endpoint

    assert provider_registry.resolve("hacker-news") is HackerNewsSourceProvider
    assert provider_registry.resolve("hacker_news") is HackerNewsSourceProvider
    assert provider_registry.resolve("hackernews") is HackerNewsSourceProvider
    assert provider_registry.resolve("hn") is HackerNewsSourceProvider
    assert resolve_source("hn") is HackerNewsSourceProvider
    assert HackerNewsSource is HackerNewsSourceProvider
    assert HackerNewsProvider is HackerNewsSourceProvider
    assert HackerNewsAPI is HackerNewsSourceProvider
    assert HackerNews is HackerNewsSourceProvider
    assert HNSource is HackerNewsSourceProvider
    assert HN is HackerNewsSourceProvider


def test_hacker_news_maps_hits_with_a_point_signal() -> None:
    body = _hn_body(
        _hit(
            "A new database",
            points=321,
            comments=45,
            url="https://example.com/db",
            object_id="101",
        ),
        _hit(
            "Ask HN: Anything",
            points=123,
            comments=9,
            object_id="102",
        ),
    )
    requests, handler = _capture(body)
    provider = _hn(handler)

    response = asyncio.run(
        provider.discover(
            SourceRequest(
                limit=2,
                mode="trending",
                metadata={"window": "week"},
            )
        )
    )

    sent = requests[0]
    assert sent.method == "GET"
    assert sent.url.path == "/api/v1/search"
    assert sent.url.params["tags"] == "story"
    assert sent.url.params["hitsPerPage"] == "2"
    assert "created_at_i>=" in sent.url.params["numericFilters"]
    assert sent.headers["user-agent"]

    assert response.provider == "hacker-news"
    assert response.titles == ["A new database", "Ask HN: Anything"]
    assert [candidate.signal for candidate in response.candidates] == [321.0, 123.0]
    assert [candidate.rank for candidate in response.candidates] == [0, 1]
    assert [candidate.category for candidate in response.candidates] == [
        "hacker-news",
        "hacker-news",
    ]
    assert response.first is not None
    assert response.first.summary == "321 points and 45 comments on Hacker News"
    assert response.first.source_urls == ["https://example.com/db"]
    assert response.candidates[1].source_urls == [
        "https://news.ycombinator.com/item?id=102"
    ]
    assert response.metadata == {
        "window": "week",
        "min_points": 0,
        "min_comments": 0,
        "total": 2,
    }


def test_hacker_news_thresholds_and_request_window_become_numeric_filters() -> None:
    body = _hn_body(
        _hit("Strong", points=500, comments=50),
        _hit("Low points", points=50, comments=50),
        _hit("Few comments", points=500, comments=5),
    )
    requests, handler = _capture(body)
    provider = _hn(handler, min_points=10, min_comments=10)

    start = datetime(2026, 5, 5, 10, tzinfo=UTC)
    end = datetime(2026, 5, 5, 12, tzinfo=UTC)
    response = asyncio.run(
        provider.discover(
            SourceRequest(
                window_start=start,
                window_end=end,
                metadata={"min_points": 100, "min_comments": 20},
            )
        )
    )

    filters = requests[0].url.params["numericFilters"]
    assert f"created_at_i>={int(start.timestamp())}" in filters
    assert f"created_at_i<={int(end.timestamp())}" in filters
    assert "points>=100" in filters
    assert "num_comments>=20" in filters
    assert response.titles == ["Strong"]
    assert response.metadata["min_points"] == 100
    assert response.metadata["min_comments"] == 20


def test_hacker_news_window_metadata_and_local_time_filtering() -> None:
    body = _hn_body(
        _hit("Recent", created=NOW_TS),
        _hit("Old", created=NOW_TS - 9000),
    )
    requests, handler = _capture(body)
    provider = _hn(handler)

    response = asyncio.run(
        provider.discover(
            SourceRequest(
                window_start=datetime(2026, 5, 5, 10, tzinfo=UTC),
                window_end=NOW,
                metadata={"window": "month"},
            )
        )
    )

    assert "created_at_i>=" in requests[0].url.params["numericFilters"]
    assert response.titles == ["Recent"]
    assert response.metadata["window"] == "month"


def test_hacker_news_query_and_endpoint_shapes() -> None:
    requests, handler = _capture(_hn_body(_hit("Python")))
    provider = _hn(handler, base_url="https://hn.example/api/v1")

    asyncio.run(provider.discover(SourceRequest(query="python")))

    assert requests[0].url.path == "/api/v1/search"
    assert requests[0].url.params["query"] == "python"
    assert provider.endpoint == "https://hn.example/api/v1/search"

    root = HackerNewsSourceProvider(base_url="https://hn.example")
    assert root.endpoint == "https://hn.example/api/v1/search"


@pytest.mark.parametrize(
    "body",
    [
        {"unexpected": True},
        {"hits": "not a list"},
    ],
)
def test_hacker_news_a_malformed_response_is_a_validation_error(body: Any) -> None:
    _requests, handler = _capture(body)
    provider = _hn(handler)

    with pytest.raises(ValidationError, match="malformed"):
        asyncio.run(provider.discover(SourceRequest()))


def test_hacker_news_a_non_json_response_is_a_validation_error() -> None:
    provider = _hn(lambda request: httpx.Response(200, text="<html>"))

    with pytest.raises(ValidationError, match="non-JSON"):
        asyncio.run(provider.discover(SourceRequest()))


def test_hacker_news_skips_dead_and_incomplete_stories() -> None:
    body = _hn_body(
        _hit("[dead]"),
        {"points": 100, "num_comments": 10, "objectID": "1"},
        {"title": "No points", "num_comments": 10},
        _hit("Valid"),
    )
    _requests, handler = _capture(body)
    provider = _hn(handler)

    response = asyncio.run(provider.discover(SourceRequest()))

    assert response.titles == ["Valid"]
    assert response.first is not None
    assert response.first.rank == 0


@pytest.mark.parametrize(
    ("status", "error_type"),
    [
        (400, InvalidRequestError),
        (401, AuthError),
        (403, AuthError),
        (404, InvalidRequestError),
        (429, RateLimitError),
        (500, RetryableError),
        (503, RetryableError),
    ],
)
def test_hacker_news_http_errors_map_to_the_shared_taxonomy(
    status: int,
    error_type: type[Exception],
) -> None:
    _requests, handler = _capture({"message": "rejected"}, status=status)
    provider = _hn(handler, http_config=_fast_http())

    with pytest.raises(error_type):
        asyncio.run(provider.discover(SourceRequest()))


def test_hacker_news_rate_limit_carries_retry_after_and_context() -> None:
    _requests, handler = _capture(
        {"message": "Too Many Requests"},
        status=429,
        headers={"Retry-After": "11"},
    )
    provider = _hn(handler, http_config=_fast_http())

    with pytest.raises(RateLimitError) as error:
        asyncio.run(provider.discover(SourceRequest()))

    assert error.value.retry_after == 11.0
    assert error.value.details["provider"] == "hacker-news"


def test_hacker_news_a_body_rate_limit_is_a_rate_limit_error() -> None:
    body = {"error": 429, "message": "Too Many Requests"}
    _requests, handler = _capture(body)
    provider = _hn(handler)

    with pytest.raises(RateLimitError, match="rate limit"):
        asyncio.run(provider.discover(SourceRequest()))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_points": 10, "min_score": 20},
        {"window": "fortnight"},
        {"min_points": -1},
        {"min_points": 1.5},
        {"min_comments": "many"},
    ],
)
def test_hacker_news_invalid_config_is_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        HackerNewsSourceProvider(**kwargs)


def test_hacker_news_constructor_accepts_threshold_aliases_and_a_mapping() -> None:
    alias = HackerNewsSourceProvider(min_score=100, min_comments=20)
    mapping = HackerNewsSourceProvider(thresholds={"points": 50, "comments": 5})

    assert alias.min_points == alias.min_score == 100
    assert alias.min_comments == 20
    assert mapping.min_points == 50
    assert mapping.min_comments == 5


def test_hacker_news_invalid_request_metadata_is_an_invalid_request() -> None:
    _requests, handler = _capture(_hn_body(_hit("Tide")))
    provider = _hn(handler)

    with pytest.raises(InvalidRequestError, match="min_points"):
        asyncio.run(provider.discover(SourceRequest(metadata={"min_points": "many"})))


def test_hacker_news_fetch_and_collect_delegate_to_discover() -> None:
    requests, handler = _capture(_hn_body(_hit("Tide")))
    provider = _hn(handler)

    fetched = asyncio.run(provider.fetch(SourceRequest()))
    collected = asyncio.run(provider.collect(SourceRequest()))

    assert fetched.titles == ["Tide"]
    assert [candidate.title for candidate in collected] == fetched.titles
    assert len(requests) == 2
