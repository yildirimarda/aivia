"""Generic RSS and Google Trends sources: parsing and graceful degradation."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from aivia.core.config.http import HttpClientConfig
from aivia.core.errors import ConfigError
from aivia.providers import provider_registry
from aivia.providers.protocols import CAPABILITY_SOURCE, SourceRequest
from aivia.providers.sources import (
    DEFAULT_RSS_FEEDS,
    DEFAULT_TRENDS_BASE_URL,
    DEFAULT_TRENDS_GEO,
    RSS,
    RSSAPI,
    GoogleTrends,
    GoogleTrendsAPI,
    GoogleTrendsProvider,
    GoogleTrendsSource,
    GoogleTrendsSourceProvider,
    RSSProvider,
    RSSSource,
    RSSSourceProvider,
    Trends,
    TrendsSource,
    normalize_feeds,
    normalize_geo,
    parse_feed,
    resolve_source,
    trends_url,
)

NOW = datetime(2026, 5, 5, 12, tzinfo=UTC)


def _rss_item(
    title: str,
    *,
    link: str | None = "https://example.com/item",
    description: str | None = None,
    pub_date: str | None = "Mon, 05 May 2026 10:00:00 GMT",
    categories: Sequence[str] = (),
) -> str:
    parts = [f"<title>{title}</title>"]
    if link is not None:
        parts.append(f"<link>{link}</link>")
    if description is not None:
        parts.append(f"<description>{description}</description>")
    if pub_date is not None:
        parts.append(f"<pubDate>{pub_date}</pubDate>")
    for category in categories:
        parts.append(f"<category>{category}</category>")
    return f"<item>{''.join(parts)}</item>"


def _rss_document(*items: str, title: str = "Example Feed") -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<rss version="2.0"><channel>'
        f"<title>{title}</title>{''.join(items)}"
        "</channel></rss>"
    )


def _atom_document(*entries: str, title: str = "Atom Feed") -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom">'
        f"<title>{title}</title>{''.join(entries)}"
        "</feed>"
    )


def _trends_document(*items: str) -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<rss version="2.0" '
        'xmlns:ht="https://trends.google.com/trending/rss">'
        "<channel><title>Daily Search Trends</title>"
        f"{''.join(items)}"
        "</channel></rss>"
    )


def _trends_item(
    title: str,
    *,
    traffic: str = "200+",
    link: str | None = "https://trends.google.com/trends/explore?q=trend",
) -> str:
    parts = [f"<title>{title}</title>"]
    if link is not None:
        parts.append(f"<link>{link}</link>")
    parts.append(f"<ht:approx_traffic>{traffic}</ht:approx_traffic>")
    parts.append("<pubDate>Mon, 05 May 2026 10:00:00 GMT</pubDate>")
    return f"<item>{''.join(parts)}</item>"


def _rss(
    handler: Callable[[httpx.Request], httpx.Response],
    **kwargs: Any,
) -> RSSSourceProvider:
    return RSSSourceProvider(transport=httpx.MockTransport(handler), **kwargs)


def _trends(
    handler: Callable[[httpx.Request], httpx.Response],
    **kwargs: Any,
) -> GoogleTrendsSourceProvider:
    return GoogleTrendsSourceProvider(
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def _capture(
    text: str = "",
    *,
    status: int = 200,
) -> tuple[list[httpx.Request], Callable[[httpx.Request], httpx.Response]]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status, text=text)

    return requests, handler


def _fast_http() -> HttpClientConfig:
    return HttpClientConfig(max_attempts=1)


def test_rss_metadata_registration_and_aliases() -> None:
    provider = _rss(
        lambda request: httpx.Response(200, text=_rss_document()),
        feeds=["https://feeds.example.com/rss.xml"],
    )

    assert provider.name == "rss"
    assert provider.capabilities == (CAPABILITY_SOURCE,)
    assert DEFAULT_RSS_FEEDS == ()
    assert provider.feeds == ("https://feeds.example.com/rss.xml",)
    assert provider.feed_urls == provider.feeds
    assert provider.endpoint == "https://feeds.example.com/rss.xml"
    assert provider.api_url == provider.endpoint

    assert provider_registry.resolve("rss") is RSSSourceProvider
    assert provider_registry.resolve("rss-source") is RSSSourceProvider
    assert provider_registry.resolve("feed") is RSSSourceProvider
    assert provider_registry.resolve("atom") is RSSSourceProvider
    assert resolve_source("rss") is RSSSourceProvider
    assert resolve_source("feed") is RSSSourceProvider
    assert RSSSource is RSSSourceProvider
    assert RSSProvider is RSSSourceProvider
    assert RSSAPI is RSSSourceProvider
    assert RSS is RSSSourceProvider


def test_rss_parses_items_and_applies_the_request_window_and_limit() -> None:
    body = _rss_document(
        _rss_item(
            "First &amp; Best",
            link="https://example.com/first",
            description="<![CDATA[<p>Hello <b>world</b></p>]]>",
            categories=("tech", "news"),
        ),
        _rss_item(
            "Outside",
            pub_date="Mon, 05 May 2026 08:00:00 GMT",
        ),
        _rss_item("Second", link="https://example.com/second"),
        _rss_item("Third", link="https://example.com/third"),
    )
    requests, handler = _capture(body)
    provider = _rss(handler, feeds=["https://feeds.example.com/rss.xml"])

    response = asyncio.run(
        provider.discover(
            SourceRequest(
                limit=2,
                mode="feeds",
                window_start=datetime(2026, 5, 5, 9, tzinfo=UTC),
                window_end=datetime(2026, 5, 5, 11, tzinfo=UTC),
            )
        )
    )

    sent = requests[0]
    assert sent.method == "GET"
    assert sent.headers["user-agent"]
    assert "application/rss+xml" in sent.headers["accept"]

    assert response.provider == "rss"
    assert response.titles == ["First & Best", "Second"]
    assert [candidate.rank for candidate in response.candidates] == [0, 1]
    assert [candidate.mode for candidate in response.candidates] == [
        "feeds",
        "feeds",
    ]
    assert response.first is not None
    assert response.first.summary == "Hello world"
    assert response.first.source_urls == ["https://example.com/first"]
    assert response.first.category == "tech"
    assert response.first.keywords == ["tech", "news"]
    assert response.first.signal is None
    assert response.metadata == {
        "feeds": ["https://feeds.example.com/rss.xml"],
        "total": 2,
    }


def test_rss_parses_atom_entries_and_resolves_relative_links() -> None:
    body = _atom_document(
        "<entry>"
        "<title>Atom One</title>"
        '<link rel="self" href="https://example.com/self"/>'
        '<link rel="alternate" href="/atom-one"/>'
        "<summary>Atom summary</summary>"
        "<updated>2026-05-05T10:00:00Z</updated>"
        '<category term="science"/>'
        "<id>tag:example.com,2026:1</id>"
        "</entry>"
        "<entry>"
        "<title>Atom Two</title>"
        "<updated>2026-05-05T11:00:00Z</updated>"
        "<id>tag:example.com,2026:2</id>"
        "</entry>"
    )
    _requests, handler = _capture(body)
    provider = _rss(handler, feeds=["https://feeds.example.com/atom.xml"])

    response = asyncio.run(provider.discover(SourceRequest()))

    assert response.titles == ["Atom One", "Atom Two"]
    assert response.first is not None
    assert response.first.summary == "Atom summary"
    assert response.first.source_urls == ["https://feeds.example.com/atom-one"]
    assert response.first.category == "science"
    assert response.first.keywords == ["science"]
    assert response.candidates[1].source_urls == []
    assert response.candidates[1].category == "Atom Feed"


def test_rss_skips_failed_feeds_and_keeps_the_good_ones() -> None:
    good = _rss_document(_rss_item("Good", link="https://example.com/good"))

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "broken" in url:
            return httpx.Response(500, text="server error")
        if "malformed" in url:
            return httpx.Response(200, text="<not-xml")
        return httpx.Response(200, text=good)

    provider = _rss(
        handler,
        feeds=[
            "https://feeds.example.com/broken.xml",
            "https://feeds.example.com/malformed.xml",
            "https://feeds.example.com/good.xml",
        ],
        http_config=_fast_http(),
    )

    response = asyncio.run(provider.discover(SourceRequest()))

    assert response.titles == ["Good"]
    assert response.metadata["total"] == 1
    assert [error["feed"] for error in response.metadata["errors"]] == [
        "https://feeds.example.com/broken.xml",
        "https://feeds.example.com/malformed.xml",
    ]
    assert response.metadata["errors"][0]["error"] == "HTTP 500"
    assert response.metadata["errors"][1]["error"] == "malformed feed"


def test_rss_transport_failure_degrades_to_an_empty_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("unreachable", request=request)

    provider = _rss(
        handler,
        feeds=["https://feeds.example.com/rss.xml"],
        http_config=_fast_http(),
    )

    response = asyncio.run(provider.discover(SourceRequest()))

    assert response.candidates == []
    assert response.metadata["total"] == 0
    assert response.metadata["errors"][0]["feed"] == (
        "https://feeds.example.com/rss.xml"
    )


def test_rss_without_feeds_returns_empty_without_http() -> None:
    provider = RSSSourceProvider()

    response = asyncio.run(provider.discover(SourceRequest()))

    assert response.candidates == []
    assert response.metadata == {"feeds": [], "total": 0}


@pytest.mark.parametrize(
    "body",
    [
        "",
        "<not-xml",
        "<!DOCTYPE rss [<!ENTITY x 'boom'>]><rss><channel /></rss>",
    ],
)
def test_rss_malformed_documents_degrade_to_empty(body: str) -> None:
    _requests, handler = _capture(body)
    provider = _rss(handler, feeds=["https://feeds.example.com/rss.xml"])

    response = asyncio.run(provider.discover(SourceRequest()))

    assert response.candidates == []
    assert response.metadata["errors"][0]["error"] == "malformed feed"


def test_rss_request_metadata_overrides_the_configured_feeds() -> None:
    requests, handler = _capture(
        _rss_document(_rss_item("Override", link="https://example.com/override"))
    )
    provider = _rss(handler, feeds=["https://feeds.example.com/default.xml"])

    response = asyncio.run(
        provider.discover(
            SourceRequest(metadata={"feed_urls": "https://feeds.example.com/other"})
        )
    )

    assert str(requests[0].url) == "https://feeds.example.com/other"
    assert response.titles == ["Override"]
    assert response.metadata["feeds"] == ["https://feeds.example.com/other"]


def test_rss_invalid_request_metadata_degrades_to_empty() -> None:
    requests, handler = _capture(_rss_document(_rss_item("Never")))
    provider = _rss(handler, feeds=["https://feeds.example.com/default.xml"])

    response = asyncio.run(
        provider.discover(SourceRequest(metadata={"feeds": ["not-a-url"]}))
    )

    assert requests == []
    assert response.candidates == []
    assert response.metadata["errors"][0]["feed"] is None


def test_rss_normalize_feeds_accepts_config_shapes() -> None:
    assert normalize_feeds(None) == DEFAULT_RSS_FEEDS
    assert normalize_feeds("https://a.example/rss, https://b.example/atom") == (
        "https://a.example/rss",
        "https://b.example/atom",
    )
    assert normalize_feeds(["https://a.example/rss", "https://a.example/rss"]) == (
        "https://a.example/rss",
    )
    with pytest.raises(ConfigError, match="feed URL"):
        normalize_feeds(["ftp://a.example/rss"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "feeds": ["https://a.example/rss"],
            "feed": ["https://b.example/rss"],
        },
        {"feeds": ["not-a-url"]},
    ],
)
def test_rss_invalid_config_is_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        RSSSourceProvider(**kwargs)


def test_rss_fetch_and_collect_delegate_to_discover() -> None:
    requests, handler = _capture(
        _rss_document(_rss_item("Tide", link="https://example.com/tide"))
    )
    provider = _rss(handler, feeds=["https://feeds.example.com/rss.xml"])

    fetched = asyncio.run(provider.fetch(SourceRequest()))
    collected = asyncio.run(provider.collect(SourceRequest()))

    assert fetched.titles == ["Tide"]
    assert [candidate.title for candidate in collected] == fetched.titles
    assert len(requests) == 2


def test_parse_feed_returns_none_for_malformed_input() -> None:
    assert parse_feed("", feed_url="https://x", request=SourceRequest()) is None
    assert parse_feed("<not-xml", feed_url="https://x", request=SourceRequest()) is None


def test_parse_feed_accepts_a_leading_bom_and_whitespace() -> None:
    parsed = parse_feed(
        "\ufeff \n" + _rss_document(_rss_item("Tide")),
        feed_url="https://feeds.example.com/rss.xml",
        request=SourceRequest(),
    )

    assert parsed is not None
    assert [candidate.title for candidate in parsed[0]] == ["Tide"]


def test_trends_metadata_registration_and_aliases() -> None:
    provider = _trends(
        lambda request: httpx.Response(200, text=_trends_document()),
        geo="gb",
    )

    assert provider.name == "google-trends"
    assert provider.capabilities == (CAPABILITY_SOURCE,)
    assert DEFAULT_TRENDS_GEO == "US"
    assert provider.geo == provider.region == provider.country == "GB"
    assert provider.base_url == DEFAULT_TRENDS_BASE_URL
    assert provider.enabled is True
    assert provider.endpoint == "https://trends.google.com/trending/rss?geo=GB"
    assert provider.api_url == provider.endpoint

    assert provider_registry.resolve("google-trends") is GoogleTrendsSourceProvider
    assert provider_registry.resolve("google_trends") is GoogleTrendsSourceProvider
    assert provider_registry.resolve("googletrends") is GoogleTrendsSourceProvider
    assert provider_registry.resolve("trends") is GoogleTrendsSourceProvider
    assert resolve_source("google-trends") is GoogleTrendsSourceProvider
    assert GoogleTrendsSource is GoogleTrendsSourceProvider
    assert GoogleTrendsProvider is GoogleTrendsSourceProvider
    assert GoogleTrendsAPI is GoogleTrendsSourceProvider
    assert GoogleTrends is GoogleTrendsSourceProvider
    assert TrendsSource is GoogleTrendsSourceProvider
    assert Trends is GoogleTrendsSourceProvider


def test_trends_parses_queries_with_a_traffic_signal() -> None:
    body = _trends_document(
        _trends_item(
            "Solar storm",
            traffic="200+",
            link="https://trends.google.com/trends/explore?q=Solar+storm",
        ),
        _trends_item("New phone", traffic="1,000+"),
    )
    requests, handler = _capture(body)
    provider = _trends(handler, geo="us")

    response = asyncio.run(provider.discover(SourceRequest(limit=2, mode="trending")))

    assert requests[0].url.params["geo"] == "US"
    assert requests[0].headers["user-agent"]
    assert response.provider == "google-trends"
    assert response.titles == ["Solar storm", "New phone"]
    assert [candidate.signal for candidate in response.candidates] == [200.0, 1000.0]
    assert [candidate.rank for candidate in response.candidates] == [0, 1]
    assert [candidate.mode for candidate in response.candidates] == [
        "trending",
        "trending",
    ]
    assert response.first is not None
    assert response.first.source_urls == [
        "https://trends.google.com/trends/explore?q=Solar+storm"
    ]
    assert response.first.category == "Daily Search Trends"
    assert response.metadata == {"geo": "US", "total": 2}


def test_trends_request_metadata_overrides_the_configured_geo() -> None:
    requests, handler = _capture(_trends_document(_trends_item("One")))
    provider = _trends(handler, geo="US")

    response = asyncio.run(provider.discover(SourceRequest(metadata={"region": "de"})))

    assert requests[0].url.params["geo"] == "DE"
    assert response.metadata["geo"] == "DE"


def test_trends_disabled_returns_empty_without_http() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, text=_trends_document())

    provider = _trends(handler, geo="US", enabled=False)

    response = asyncio.run(provider.discover(SourceRequest()))

    assert calls == []
    assert response.candidates == []
    assert response.metadata == {"geo": "US", "total": 0, "enabled": False}


@pytest.mark.parametrize(
    "handler",
    [
        lambda request: httpx.Response(500, text="server error"),
        lambda request: httpx.Response(200, text="<not-xml"),
        lambda request: (_ for _ in ()).throw(
            httpx.ConnectError("unreachable", request=request)
        ),
    ],
)
def test_trends_failures_degrade_to_empty(
    handler: Callable[[httpx.Request], httpx.Response],
) -> None:
    provider = _trends(
        handler,
        geo="US",
        http_config=_fast_http(),
    )

    response = asyncio.run(provider.discover(SourceRequest()))

    assert response.candidates == []
    assert response.metadata["total"] == 0
    assert response.metadata["errors"][0]["feed"]


def test_trends_invalid_request_geo_degrades_to_empty() -> None:
    requests, handler = _capture(_trends_document(_trends_item("Never")))
    provider = _trends(handler, geo="US")

    response = asyncio.run(provider.discover(SourceRequest(metadata={"geo": "USA"})))

    assert requests == []
    assert response.candidates == []
    assert response.metadata["errors"][0]["feed"] is None


def test_trends_normalize_geo_accepts_country_codes_and_worldwide() -> None:
    assert normalize_geo(None) == DEFAULT_TRENDS_GEO
    assert normalize_geo("us") == "US"
    assert normalize_geo(" tr ") == "TR"
    assert normalize_geo("worldwide") == ""
    assert trends_url("de") == "https://trends.google.com/trending/rss?geo=DE"
    assert trends_url("worldwide") == "https://trends.google.com/trending/rss"
    with pytest.raises(ConfigError, match="geo"):
        normalize_geo("USA")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"geo": "USA"},
        {"geo": "US", "region": "GB"},
        {"enabled": "yes"},
        {"base_url": "not-a-url"},
    ],
)
def test_trends_invalid_config_is_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        GoogleTrendsSourceProvider(**kwargs)


def test_trends_fetch_and_collect_delegate_to_discover() -> None:
    requests, handler = _capture(_trends_document(_trends_item("One")))
    provider = _trends(handler, geo="US")

    fetched = asyncio.run(provider.fetch(SourceRequest()))
    collected = asyncio.run(provider.collect(SourceRequest()))

    assert fetched.titles == ["One"]
    assert [candidate.title for candidate in collected] == fetched.titles
    assert len(requests) == 2
