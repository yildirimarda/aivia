"""Trending and manual modes: signal ordering and an ordered topic queue."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from aivia.core.errors import ConfigError
from aivia.providers.protocols.source import SourceRequest, SourceResponse
from aivia.providers.sources import WikipediaPageviewsSourceProvider
from aivia.research import (
    DEFAULT_DEDUPE,
    DEFAULT_DESCENDING,
    DEFAULT_MIN_SIGNAL,
    DEFAULT_REQUEUE,
    DEFAULT_REQUIRE_SIGNAL,
    DEFAULT_SORT_BY,
    DEFAULT_TRENDING_SOURCE,
    Manual,
    ManualMode,
    ManualModeProvider,
    ManualQueue,
    ManualSourceMode,
    ManualTopicMode,
    ManualTopicQueue,
    Queue,
    QueueMode,
    TopicQueue,
    Trending,
    TrendingMode,
    TrendingModeProvider,
    TrendingSourceMode,
    TrendingTopicMode,
    Trends,
    TrendsMode,
    WikipediaTrendingMode,
    order_by_signal,
    rank_by_signal,
    resolve_mode,
    signal_order,
    sort_by_signal,
)


def _candidate(title: str, signal: float | None = None, **values: Any) -> Any:
    from aivia.core.models.topic import TopicCandidate

    return TopicCandidate(
        title=title,
        signal=signal,
        source_urls=[f"https://example.test/{title.replace(' ', '_')}"],
        **values,
    )


def _run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


class FakeSource:
    """An offline source returning fixed candidates and recording requests."""

    def __init__(
        self,
        candidates: list[Any] | None = None,
        *,
        provider: str = "fake-trending",
    ) -> None:
        self.candidates = list(candidates or [])
        self.provider = provider
        self.requests: list[SourceRequest] = []

    async def discover(self, request: SourceRequest) -> SourceResponse:
        self.requests.append(request)
        return SourceResponse(
            provider=self.provider,
            query=request.query,
            candidates=[candidate.model_copy() for candidate in self.candidates],
            metadata={"feed": "fake"},
        )


def test_the_modes_are_registered_and_aliases_share_one_class() -> None:
    assert resolve_mode("trending") is TrendingMode
    assert Trending is TrendingMode
    assert TrendingTopicMode is TrendingMode
    assert TrendingSourceMode is TrendingMode
    assert TrendingModeProvider is TrendingMode
    assert Trends is TrendingMode
    assert TrendsMode is TrendingMode
    assert WikipediaTrendingMode is TrendingMode
    assert TrendingMode().name == "trending"
    assert DEFAULT_SORT_BY == "signal"
    assert DEFAULT_DESCENDING is True
    assert DEFAULT_MIN_SIGNAL is None
    assert DEFAULT_REQUIRE_SIGNAL is False
    assert DEFAULT_TRENDING_SOURCE is WikipediaPageviewsSourceProvider

    assert resolve_mode("manual") is ManualMode
    assert Manual is ManualMode
    assert ManualTopicMode is ManualMode
    assert ManualSourceMode is ManualMode
    assert ManualModeProvider is ManualMode
    assert ManualQueue is ManualMode
    assert ManualTopicQueue is ManualMode
    assert TopicQueue is ManualMode
    assert QueueMode is ManualMode
    assert Queue is ManualMode
    assert ManualMode().name == "manual"
    assert DEFAULT_DEDUPE is True
    assert DEFAULT_REQUEUE is False


def test_trending_orders_by_signal_with_missing_signals_last() -> None:
    source = FakeSource(
        [
            _candidate("Low", 10),
            _candidate("Highest", 100),
            _candidate("No signal"),
            _candidate("Middle", 50),
        ]
    )
    mode = TrendingMode(source)

    response = _run(mode.discover(SourceRequest(query="news", limit=10)))

    assert response.titles == ["Highest", "Middle", "Low", "No signal"]
    assert [candidate.signal for candidate in response.candidates] == [
        100.0,
        50.0,
        10.0,
        None,
    ]
    assert [candidate.mode for candidate in response.candidates] == [
        "trending",
        "trending",
        "trending",
        "trending",
    ]
    assert [candidate.rank for candidate in response.candidates] == [0, 1, 2, 3]
    assert response.provider == "trending"
    assert response.query == "news"
    assert response.metadata["mode"] == "trending"
    assert response.metadata["source_provider"] == "fake-trending"
    assert response.metadata["feed"] == "fake"
    assert response.metadata["signal_ordered"] is True
    assert response.metadata["sort_by"] == "signal"
    assert response.metadata["descending"] is True
    assert response.metadata["found_count"] == 4
    assert response.metadata["served_count"] == 4
    assert response.metadata["skipped_duplicates"] == 0
    assert response.metadata["skipped_signal"] == 0
    assert source.requests[0].mode == "trending"


def test_trending_can_order_ascending() -> None:
    source = FakeSource(
        [_candidate("High", 100), _candidate("Low", 1), _candidate("Missing")]
    )
    mode = TrendingMode(source, ascending=True)

    response = _run(mode.discover(SourceRequest()))

    assert response.titles == ["Low", "High", "Missing"]
    assert mode.descending is False
    assert response.metadata["descending"] is False


def test_trending_can_preserve_source_order() -> None:
    source = FakeSource(
        [_candidate("First", 1), _candidate("Second", 100), _candidate("Third", 50)]
    )
    mode = TrendingMode(source, sort_by="source")

    response = _run(mode.discover(SourceRequest()))

    assert response.titles == ["First", "Second", "Third"]
    assert [candidate.rank for candidate in response.candidates] == [0, 1, 2]
    assert mode.signal_ordered is False
    assert response.metadata["signal_ordered"] is False
    assert response.metadata["sort_by"] == "source"


def test_trending_min_signal_drops_weak_and_missing_candidates() -> None:
    source = FakeSource(
        [_candidate("Weak", 5), _candidate("Strong", 50), _candidate("Missing")]
    )
    mode = TrendingMode(source, min_signal=20)

    response = _run(mode.discover(SourceRequest()))

    assert response.titles == ["Strong"]
    assert mode.min_signal == 20.0
    assert response.metadata["min_signal"] == 20.0
    assert response.metadata["skipped_signal"] == 2
    assert response.metadata["skipped_low_signal"] == 2


def test_trending_require_signal_drops_candidates_without_one() -> None:
    source = FakeSource([_candidate("With signal", 5), _candidate("Without signal")])
    mode = TrendingMode(source, require_signal=True)

    response = _run(mode.discover(SourceRequest()))

    assert response.titles == ["With signal"]
    assert mode.require_signal is True
    assert response.metadata["require_signal"] is True
    assert response.metadata["skipped_missing_signal"] == 1
    assert response.metadata["skipped_signal"] == 0


def test_trending_duplicate_titles_keep_the_strongest_signal() -> None:
    weak = FakeSource([_candidate("Same topic", 10)], provider="weak")
    strong = FakeSource([_candidate("Same topic", 99)], provider="strong")
    mode = TrendingMode(sources=[weak, strong])

    response = _run(mode.discover(SourceRequest()))

    assert response.titles == ["Same topic"]
    assert response.first is not None
    assert response.first.signal == 99.0
    assert response.metadata["skipped_duplicates"] == 1
    assert response.metadata["source_count"] == 2
    assert response.metadata["source_provider"] == "weak, strong"


def test_trending_applies_the_limit_after_sorting() -> None:
    source = FakeSource(
        [
            _candidate("A", 10),
            _candidate("B", 40),
            _candidate("C", 30),
            _candidate("D", 20),
        ]
    )
    mode = TrendingMode(source)

    response = _run(mode.discover(SourceRequest(limit=2)))

    assert response.titles == ["B", "C"]
    assert [candidate.rank for candidate in response.candidates] == [0, 1]
    assert response.metadata["found_count"] == 4
    assert response.metadata["served_count"] == 2


def test_trending_reads_sources_from_the_profile() -> None:
    source = FakeSource([_candidate("From profile", 1)])
    mode = TrendingMode(profile={"sources": [source]})

    response = _run(mode.discover(SourceRequest()))

    assert mode.sources == [source]
    assert response.titles == ["From profile"]


def test_trending_defaults_to_the_pageviews_source_without_network() -> None:
    mode = TrendingMode()

    assert isinstance(mode.sources[0], WikipediaPageviewsSourceProvider)
    assert mode.source is mode.sources[0]


def test_trending_rejects_conflicting_order_and_signal_arguments() -> None:
    source = FakeSource()

    with pytest.raises(ConfigError, match="only one of sort_by"):
        TrendingMode(source, sort_by="signal", order_by="source")
    with pytest.raises(ConfigError, match="only one of descending"):
        TrendingMode(source, descending=True, ascending=True)
    with pytest.raises(ConfigError, match="unknown trending order"):
        TrendingMode(source, sort_by="random")
    with pytest.raises(ConfigError, match="min signal must be finite"):
        TrendingMode(source, min_signal=float("inf"))
    with pytest.raises(ConfigError, match="min signal must not exceed max signal"):
        TrendingMode(source, min_signal=10, max_signal=5)
    with pytest.raises(ConfigError, match="require_signal must be a boolean"):
        TrendingMode(source, require_signal=1)  # type: ignore[arg-type]


def test_order_by_signal_is_stable_and_exported_under_aliases() -> None:
    candidates = [
        _candidate("A", 5),
        _candidate("B", 9),
        _candidate("C", 5),
        _candidate("D"),
    ]

    assert [item.title for item in order_by_signal(candidates)] == [
        "B",
        "A",
        "C",
        "D",
    ]
    assert [item.title for item in order_by_signal(candidates, descending=False)] == [
        "A",
        "C",
        "B",
        "D",
    ]
    assert sort_by_signal is order_by_signal
    assert rank_by_signal is order_by_signal
    assert signal_order is order_by_signal


def test_manual_serves_topics_in_insertion_order_and_drains_the_queue() -> None:
    mode = ManualMode(["Second first", "Then this", "Last"])
    request = SourceRequest(query="manual run", limit=2)

    first = _run(mode.discover(request))

    assert first.titles == ["Second first", "Then this"]
    assert [candidate.rank for candidate in first.candidates] == [0, 1]
    assert [candidate.mode for candidate in first.candidates] == ["manual", "manual"]
    assert first.provider == "manual"
    assert first.query == "manual run"
    assert first.metadata["mode"] == "manual"
    assert first.metadata["queue_before"] == 3
    assert first.metadata["queue_after"] == 1
    assert first.metadata["served_count"] == 2
    assert list(mode.titles) == ["Last"]

    second = _run(mode.discover(request))
    assert second.titles == ["Last"]
    assert second.metadata["queue_after"] == 0

    third = _run(mode.discover(request))
    assert third.titles == []
    assert third.metadata["served_count"] == 0
    assert mode.empty is True


def test_manual_add_enqueue_and_extend_deduplicate_titles() -> None:
    mode = ManualMode(["Alpha"])

    added = mode.add(["alpha", "Beta", "ALPHA", "Gamma"])
    assert [candidate.title for candidate in added] == ["Beta", "Gamma"]
    assert list(mode.titles) == ["Alpha", "Beta", "Gamma"]
    assert mode.pending == 3
    assert mode.count == 3
    assert mode.size == 3
    assert mode.enqueue("Delta")[0].title == "Delta"
    assert mode.extend(["Epsilon"])[0].title == "Epsilon"
    assert list(mode.titles) == ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]


def test_manual_deduplication_can_be_disabled() -> None:
    mode = ManualMode(["Alpha"], dedupe=False)

    added = mode.add(["alpha", "Beta"])

    assert [candidate.title for candidate in added] == ["alpha", "Beta"]
    assert list(mode.titles) == ["Alpha", "alpha", "Beta"]
    assert mode.dedupe is False


def test_manual_queue_operations_peek_remove_discard_and_clear() -> None:
    mode = ManualMode(["Alpha", "Beta", "Gamma"])

    next_topic = mode.peek()
    assert next_topic is not None
    assert next_topic.title == "Alpha"
    assert "beta" in mode
    assert "missing" not in mode

    removed = mode.remove("beta")
    assert removed.title == "Beta"
    assert list(mode.titles) == ["Alpha", "Gamma"]
    assert mode.discard("Gamma") is True
    assert mode.discard("Gamma") is False
    assert list(mode.titles) == ["Alpha"]
    assert mode.dequeue() is not None
    assert mode.dequeue() is None
    assert mode.peek() is None

    mode.add(["Delta", "Epsilon"])
    mode.clear()
    assert mode.empty is True
    assert len(mode) == 0
    assert list(mode) == []


def test_manual_accepts_candidates_and_mappings() -> None:
    candidate = _candidate("Candidate", 3, summary="From a candidate")
    mode = ManualMode([candidate, {"title": "Mapping", "summary": "From a mapping"}])

    response = _run(mode.discover(SourceRequest()))

    assert response.titles == ["Candidate", "Mapping"]
    assert response.first is not None
    assert response.first.summary == "From a candidate"
    assert response.candidates[1].summary == "From a mapping"
    assert all(item.mode == "manual" for item in response.candidates)


def test_manual_reads_topics_from_the_profile_and_explicit_topics_win() -> None:
    from_profile = ManualMode(profile={"manual_topics": ["Profile one", "Profile two"]})
    assert from_profile.titles == ("Profile one", "Profile two")

    explicit = ManualMode(
        ["Explicit"],
        profile={"manual_topics": ["Profile one"]},
    )
    assert explicit.titles == ("Explicit",)


def test_manual_can_requeue_served_topics() -> None:
    mode = ManualMode(["A", "B"], requeue=True)

    first = _run(mode.discover(SourceRequest(limit=2)))
    second = _run(mode.discover(SourceRequest(limit=2)))

    assert first.titles == ["A", "B"]
    assert second.titles == ["A", "B"]
    assert mode.requeue is True
    assert second.metadata["requeued"] is True


def test_manual_rejects_conflicting_arguments_and_bad_topics() -> None:
    with pytest.raises(ConfigError, match="use only one of topics"):
        ManualMode(["A"], queue=["B"])
    with pytest.raises(ConfigError, match="manual topics must be"):
        ManualMode([123])
    with pytest.raises(ConfigError, match="topic is already queued"):
        ManualMode(["A"]).append("a")
    with pytest.raises(ConfigError, match="topic is not queued"):
        ManualMode(["A"]).remove("missing")
    with pytest.raises(ConfigError, match="dedupe must be a boolean"):
        ManualMode(["A"], dedupe=1)  # type: ignore[arg-type]
