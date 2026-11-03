"""Keyword and feeds modes: explicit keyword queries and feed-only discovery."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from aivia.core.errors import ConfigError
from aivia.providers.protocols.source import SourceRequest, SourceResponse
from aivia.research import (
    Feed,
    FeedMode,
    Feeds,
    FeedsMode,
    FeedSourceMode,
    FeedsTopicMode,
    FeedTopicMode,
    Keyword,
    KeywordGate,
    KeywordMode,
    KeywordModeProvider,
    Keywords,
    KeywordsMode,
    KeywordSourceMode,
    KeywordsTopicMode,
    KeywordTopicMode,
    RSSFeedsMode,
    RSSMode,
    RssMode,
    WikipediaKeywordsMode,
    resolve_mode,
)
from aivia.research.brief import BriefDerivation


def _candidate(
    title: str, *, summary: str = "", keywords: list[str] | None = None
) -> Any:
    from aivia.core.models.topic import TopicCandidate

    return TopicCandidate(
        title=title,
        summary=summary,
        keywords=list(keywords or []),
        source_urls=[f"https://example.test/{title.replace(' ', '_')}"],
    )


def _run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


class FakeSource:
    """An offline source returning fixed candidates per query."""

    def __init__(
        self,
        results: dict[str, list[Any]] | None = None,
        *,
        name: str = "fake",
        provider: str | None = None,
        candidates: list[Any] | None = None,
    ) -> None:
        self.name = name
        self.provider = provider or name
        self.results = dict(results or {})
        self.candidates = list(candidates or [])
        self.requests: list[SourceRequest] = []

    async def discover(self, request: SourceRequest) -> SourceResponse:
        self.requests.append(request)
        items = self.candidates or self.results.get(request.query, [])
        return SourceResponse(
            provider=self.provider,
            query=request.query,
            candidates=[candidate.model_copy() for candidate in items],
            metadata={"feed": "fake"},
        )


def test_the_modes_are_registered_and_aliases_share_one_class() -> None:
    assert resolve_mode("keywords") is KeywordsMode
    assert Keyword is KeywordsMode
    assert KeywordMode is KeywordsMode
    assert KeywordTopicMode is KeywordsMode
    assert KeywordSourceMode is KeywordsMode
    assert KeywordModeProvider is KeywordsMode
    assert Keywords is KeywordsMode
    assert KeywordsTopicMode is KeywordsMode
    assert WikipediaKeywordsMode is KeywordsMode
    assert KeywordsMode().name == "keywords"

    assert resolve_mode("feeds") is FeedsMode
    assert Feed is FeedsMode
    assert FeedMode is FeedsMode
    assert FeedTopicMode is FeedsMode
    assert FeedSourceMode is FeedsMode
    assert Feeds is FeedsMode
    assert FeedsTopicMode is FeedsMode
    assert RSSFeedsMode is FeedsMode
    assert RSSMode is FeedsMode
    assert RssMode is FeedsMode
    assert FeedsMode().name == "feeds"


def test_keywords_are_queries_and_candidates_matching_any_keyword_are_kept() -> None:
    source = FakeSource(
        {
            "history": [_candidate("History of Rome"), _candidate("Modern cooking")],
            "science": [_candidate("Science of stars")],
        }
    )
    mode = KeywordsMode(source, keywords=["history", "science"])
    request = SourceRequest(query="ignored", language="en", limit=7)

    response = _run(mode.discover(request))

    assert [item.query for item in source.requests] == ["history", "science"]
    assert all(item.mode == "keywords" for item in source.requests)
    assert all(item.language == "en" for item in source.requests)
    assert all(item.limit == 7 for item in source.requests)
    assert response.titles == ["History of Rome", "Science of stars"]
    assert response.provider == "keywords"
    assert response.query == "ignored"
    assert [candidate.mode for candidate in response.candidates] == [
        "keywords",
        "keywords",
    ]
    assert [candidate.rank for candidate in response.candidates] == [0, 1]
    assert response.metadata["mode"] == "keywords"
    assert response.metadata["keywords"] == ["history", "science"]
    assert response.metadata["include_terms"] == ["history", "science"]
    assert response.metadata["source_provider"] == "fake"
    assert response.metadata["feed"] == "fake"
    assert response.metadata["skipped_relevance"] == 1
    assert response.metadata["skipped_keywords"] == 1


def test_the_any_keyword_gate_accepts_one_match_and_rejects_none() -> None:
    gate = KeywordGate(include_terms=["history", "science", "space"])
    assert gate.term_score(_candidate("History of Rome")) == 1.0
    assert gate.term_score(_candidate("Science of stars")) == 1.0
    assert gate.term_score(_candidate("Modern cooking")) == 0.0
    assert gate.is_relevant(_candidate("History of Rome"))
    assert not gate.is_relevant(_candidate("Modern cooking"))


def test_exclude_terms_still_reject_a_keyword_match() -> None:
    source = FakeSource(
        {
            "history": [
                _candidate("History of Rome", summary="Football rivalry"),
                _candidate("History of science"),
            ]
        }
    )
    mode = KeywordsMode(
        source,
        keywords=["history"],
        exclude_terms=["football"],
    )

    response = _run(mode.discover(SourceRequest()))

    assert response.titles == ["History of science"]
    assert response.metadata["skipped_relevance"] == 1


def test_profile_and_derivation_keywords_are_used_when_none_are_given() -> None:
    source = FakeSource(
        {
            "history": [_candidate("History of Rome")],
            "science": [_candidate("Science of stars")],
            "space": [_candidate("Space exploration")],
        }
    )

    from_profile = KeywordsMode(source, profile={"keywords": ["history", "science"]})
    assert from_profile.keywords == ["history", "science"]
    assert from_profile.queries == ["history", "science"]

    derivation = BriefDerivation(keywords=["space"])
    from_derivation = KeywordsMode(source, derivation=derivation)
    assert from_derivation.keywords == ["space"]
    assert from_derivation.queries == ["space"]

    assert _run(from_profile.discover(SourceRequest())).titles == [
        "History of Rome",
        "Science of stars",
    ]
    assert _run(from_derivation.discover(SourceRequest())).titles == [
        "Space exploration"
    ]


def test_explicit_queries_override_the_keyword_search_but_not_the_gate() -> None:
    source = FakeSource(
        {
            "custom query": [_candidate("History of Rome"), _candidate("Cooking")],
        }
    )
    mode = KeywordsMode(
        source,
        keywords=["history"],
        queries=["custom query"],
    )

    response = _run(mode.discover(SourceRequest()))

    assert [item.query for item in source.requests] == ["custom query"]
    assert mode.keywords == ["history"]
    assert response.titles == ["History of Rome"]
    assert response.metadata["skipped_relevance"] == 1


def test_keywords_are_normalised_deduplicated_and_can_be_a_string() -> None:
    mode = KeywordsMode(FakeSource(), keywords=" History,  history ; SCIENCE ")

    assert mode.keywords == ["History", "SCIENCE"]
    assert mode.queries == ["History", "SCIENCE"]
    assert mode.include_terms == ["History", "SCIENCE"]


def test_a_custom_scorer_replaces_the_any_keyword_gate() -> None:
    source = FakeSource({"q": [_candidate("Keep"), _candidate("Drop")]})

    def scorer(candidate: Any) -> float:
        return 0.9 if candidate.title == "Keep" else 0.1

    mode = KeywordsMode(
        source,
        keywords=["q"],
        queries=["q"],
        relevance=scorer,
        min_relevance=0.5,
    )

    response = _run(mode.discover(SourceRequest()))

    assert response.titles == ["Keep"]


def test_duplicate_titles_across_keywords_are_reported_once() -> None:
    source = FakeSource(
        {
            "history": [_candidate("History of Rome")],
            "science": [_candidate("History of Rome")],
        }
    )
    mode = KeywordsMode(source, keywords=["history", "science"])

    response = _run(mode.discover(SourceRequest()))

    assert response.titles == ["History of Rome"]
    assert response.metadata["skipped_duplicates"] == 1


def test_feeds_mode_uses_only_feed_sources_and_ignores_the_rest() -> None:
    wikipedia = FakeSource(name="wikipedia", provider="wikipedia")
    rss = FakeSource(
        name="rss",
        provider="rss",
        candidates=[_candidate("Feed item"), _candidate("Another feed item")],
    )
    mode = FeedsMode(sources=[wikipedia, rss])

    response = _run(mode.discover(SourceRequest(query="ignored")))

    assert wikipedia.requests == []
    assert len(rss.requests) == 1
    assert response.titles == ["Feed item", "Another feed item"]
    assert [candidate.mode for candidate in response.candidates] == ["feeds", "feeds"]
    assert [candidate.rank for candidate in response.candidates] == [0, 1]
    assert response.provider == "feeds"
    assert response.metadata["mode"] == "feeds"
    assert response.metadata["source_provider"] == "rss"
    assert response.metadata["ignored_sources"] == ["wikipedia"]
    assert response.metadata["min_relevance"] is None
    assert response.metadata["skipped_relevance"] == 0


def test_feeds_mode_accepts_a_single_feed_source() -> None:
    rss = FakeSource(
        name="fake-rss",
        provider="fake-rss",
        candidates=[_candidate("Only item")],
    )
    mode = FeedsMode(rss)

    response = _run(mode.discover(SourceRequest()))

    assert mode.source is rss
    assert response.titles == ["Only item"]


def test_feeds_mode_reads_feed_urls_from_the_profile_options() -> None:
    profile = {
        "sources": ["wikipedia", "rss"],
        "source_options": {
            "rss": {"feeds": ["https://feeds.example.com/one.xml"]},
        },
    }
    rss = FakeSource(
        name="rss",
        provider="rss",
        candidates=[_candidate("Feed item")],
    )
    mode = FeedsMode(profile=profile, source=rss)

    response = _run(mode.discover(SourceRequest()))

    assert mode.feeds == ("https://feeds.example.com/one.xml",)
    assert mode.feed_urls == mode.feeds
    assert response.titles == ["Feed item"]
    assert rss.requests[0].metadata["feeds"] == ["https://feeds.example.com/one.xml"]
    assert mode.ignored_sources == ["wikipedia"]


def test_explicit_feeds_override_profile_feeds() -> None:
    profile = {
        "sources": ["rss"],
        "source_options": {
            "rss": {"feeds": ["https://feeds.example.com/old.xml"]},
        },
    }
    rss = FakeSource(name="rss", provider="rss", candidates=[_candidate("Item")])
    mode = FeedsMode(
        rss,
        profile=profile,
        feeds=["https://feeds.example.com/new.xml"],
    )

    _run(mode.discover(SourceRequest()))

    assert mode.feeds == ("https://feeds.example.com/new.xml",)
    assert rss.requests[0].metadata["feeds"] == ["https://feeds.example.com/new.xml"]


def test_feeds_mode_defaults_to_an_empty_rss_run_without_network() -> None:
    mode = FeedsMode()

    response = _run(mode.discover(SourceRequest()))

    assert response.titles == []
    assert response.metadata["mode"] == "feeds"
    assert response.metadata["feeds"] == []
    assert response.metadata["source_count"] == 1


def test_feeds_mode_ignores_a_profile_with_no_feed_sources() -> None:
    wikipedia = FakeSource(name="wikipedia", provider="wikipedia")
    mode = FeedsMode(sources=[wikipedia])

    response = _run(mode.discover(SourceRequest()))

    assert wikipedia.requests == []
    assert response.titles == []
    assert response.metadata["source_count"] == 0
    assert mode.ignored_sources == ["wikipedia"]


def test_feeds_mode_deduplicates_titles_across_feed_sources() -> None:
    first = FakeSource(name="rss", provider="rss", candidates=[_candidate("Same")])
    second = FakeSource(name="atom", provider="atom", candidates=[_candidate("Same")])
    mode = FeedsMode(sources=[first, second])

    response = _run(mode.discover(SourceRequest()))

    assert response.titles == ["Same"]
    assert response.metadata["skipped_duplicates"] == 1
    assert response.metadata["source_count"] == 2


def test_feeds_mode_rejects_conflicting_source_arguments() -> None:
    source = FakeSource(name="rss", provider="rss")

    with pytest.raises(ConfigError, match="either source or sources"):
        FeedsMode(source, sources=[source])

    with pytest.raises(ConfigError, match="only one of feeds"):
        FeedsMode(
            source,
            feeds=["https://feeds.example.com/a.xml"],
            feed_urls=["https://feeds.example.com/b.xml"],
        )
