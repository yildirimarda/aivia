"""A research run keeps going when one of its sources fails.

The run resolves real source adapters through the registry, serves one feed
from an offline HTTP transport and lets a second adapter fail at request time.
Collection must record the failure, keep the surviving candidates and still
carry them through dedupe, gating, scoring, verification and selection.
"""

from __future__ import annotations

import httpx

from aivia.core.config.http import HttpClientConfig
from aivia.core.config.profile import ContentProfileConfig
from aivia.providers.protocols.source import SourceRequest
from aivia.providers.sources import (
    RSSSourceProvider,
    SourceRegistry,
    WikipediaSourceProvider,
)
from aivia.research import (
    BriefDerivation,
    ScoringRubric,
    research_topics,
    resolve_sources,
)

FEED_URL = "https://feeds.example.com/energy.xml"

FEED_BODY = """<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0"><channel>
<title>Energy Weekly</title>
<item>
  <title>Solar panel costs</title>
  <link>https://news.example/solar-costs</link>
  <description>Falling prices for home solar.</description>
  <category>energy</category>
</item>
<item>
  <title>Solar battery storage</title>
  <link>https://analysis.example/solar-storage</link>
  <description>Storing solar power at home.</description>
  <category>storage</category>
</item>
</channel></rss>
"""


def _feed_response(request: httpx.Request) -> httpx.Response:
    assert request.url == FEED_URL
    return httpx.Response(200, text=FEED_BODY)


def _unavailable(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("wikipedia is unavailable", request=request)


class OfflineFeedSource(RSSSourceProvider):
    """A real RSS adapter reading one fixed feed through a mock transport."""

    def __init__(self) -> None:
        super().__init__(
            feeds=[FEED_URL],
            transport=httpx.MockTransport(_feed_response),
        )


class UnavailableWikipediaSource(WikipediaSourceProvider):
    """A real Wikipedia adapter whose transport refuses every connection."""

    def __init__(self) -> None:
        super().__init__(
            language="en",
            transport=httpx.MockTransport(_unavailable),
            http_config=HttpClientConfig(max_attempts=1),
        )


def _derivation() -> BriefDerivation:
    return BriefDerivation(
        brief="Practical guides to home solar energy.",
        brief_hash="solar-brief",
        include_terms=["solar"],
    )


def _rubric() -> ScoringRubric:
    return ScoringRubric(
        weights={"relevance": 0.5, "novelty": 0.5},
        min_score=0.6,
    )


def test_a_research_run_tolerates_one_failing_source() -> None:
    registry = SourceRegistry()
    registry.register_source("feed", OfflineFeedSource)
    registry.register_source("wikipedia", UnavailableWikipediaSource)
    profile = ContentProfileConfig.from_mapping(
        {
            "brief": "Practical guides to home solar energy.",
            "sources": ["feed", "wikipedia"],
        }
    )
    selection = resolve_sources(profile, registry=registry)

    result = research_topics(
        sources=selection,
        request=SourceRequest(query="home solar energy"),
        derivation=_derivation(),
        rubric=_rubric(),
        scores={
            "Solar panel costs": {"relevance": 0.9, "novelty": 0.9},
            "Solar battery storage": {"relevance": 0.8, "novelty": 0.8},
        },
        require_verified=False,
        limit=None,
        max_per_category=None,
        record=False,
    )

    assert result.complete is False
    assert result.ok is False
    assert result.failed_sources == ["wikipedia"]

    assert result.collection is not None
    assert result.collection.succeeded_sources == ["feed"]
    assert result.collection.failed_sources == ["wikipedia"]
    failure = result.collection.failures[0]
    assert failure.source == "wikipedia"
    assert failure.error_type
    assert "ConnectError" in failure.error
    assert failure.timeout is False
    assert result.collection.failure_report.to_dict()["attempted"] == 2
    assert result.collection.failure_report.to_dict()["succeeded"] == 1

    assert result.metadata["sources"] == {
        "attempted": 2,
        "succeeded": 1,
        "failed": 1,
    }
    assert result.metadata["failed_sources"] == ["wikipedia"]
    assert result.metadata["complete"] is False

    assert result.selected_titles == [
        "Solar panel costs",
        "Solar battery storage",
    ]
    assert result.count == 2
    assert result.verification is not None
    assert result.verification.unverified_titles == [
        "Solar panel costs",
        "Solar battery storage",
    ]
