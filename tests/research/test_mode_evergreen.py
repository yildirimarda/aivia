"""Evergreen ideas mode: backlog, replenishment and past-topic dedupe."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from aivia.core.errors import ConfigError
from aivia.providers.protocols.source import SourceRequest, SourceResponse
from aivia.research import (
    DEFAULT_BACKLOG_SIZE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MIN_BACKLOG,
    DEFAULT_REPLENISH_AT,
    DEFAULT_REPLENISH_BATCH,
    DEFAULT_REPLENISH_THRESHOLD,
    Evergreen,
    EvergreenIdeas,
    EvergreenIdeasMode,
    EvergreenIdeasTopicMode,
    EvergreenMode,
    EvergreenModeProvider,
    IdeasMode,
    SeenTopic,
    TopicHistory,
    WikipediaEvergreenMode,
    normalize_title,
    resolve_mode,
)

NOW = datetime(2026, 5, 5, 12, tzinfo=UTC)


class Clock:
    """A deterministic clock that tests can move forward."""

    def __init__(self, value: datetime = NOW) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value

    def advance(self, delta: timedelta) -> None:
        self.value += delta


def _candidate(title: str, **values: Any) -> Any:
    from aivia.core.models.topic import TopicCandidate

    return TopicCandidate(
        title=title,
        source_urls=[f"https://example.test/{title.replace(' ', '_')}"],
        **values,
    )


def _run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


class FakeSource:
    """An offline source returning fixed batches in call order."""

    def __init__(
        self,
        batches: list[list[Any]] | None = None,
        *,
        candidates: list[Any] | None = None,
        provider: str = "fake-evergreen",
    ) -> None:
        self.batches = [list(batch) for batch in (batches or [])]
        self.candidates = list(candidates or [])
        self.provider = provider
        self.requests: list[SourceRequest] = []
        self.calls = 0

    async def discover(self, request: SourceRequest) -> SourceResponse:
        self.calls += 1
        self.requests.append(request)
        items = self.batches.pop(0) if self.batches else self.candidates
        return SourceResponse(
            provider=self.provider,
            query=request.query,
            candidates=[candidate.model_copy() for candidate in items],
            metadata={"feed": "fake"},
        )


def test_the_mode_is_registered_and_aliases_share_one_class() -> None:
    assert resolve_mode("evergreen_ideas") is EvergreenMode
    assert EvergreenIdeasMode is EvergreenMode
    assert EvergreenIdeasTopicMode is EvergreenMode
    assert EvergreenIdeas is EvergreenMode
    assert Evergreen is EvergreenMode
    assert IdeasMode is EvergreenMode
    assert EvergreenModeProvider is EvergreenMode
    assert WikipediaEvergreenMode is EvergreenMode
    assert EvergreenMode().name == "evergreen_ideas"
    assert DEFAULT_BACKLOG_SIZE == 20
    assert DEFAULT_BATCH_SIZE == DEFAULT_REPLENISH_BATCH
    assert DEFAULT_MIN_BACKLOG == DEFAULT_REPLENISH_THRESHOLD
    assert DEFAULT_REPLENISH_AT == DEFAULT_REPLENISH_THRESHOLD


def test_the_first_run_replenishes_and_serves_from_the_backlog() -> None:
    source = FakeSource(
        candidates=[_candidate(name) for name in ("A", "B", "C", "D", "E", "F")]
    )
    mode = EvergreenMode(
        source,
        backlog_size=5,
        replenish_threshold=1,
        replenish_batch=5,
        now=NOW,
    )

    response = _run(mode.discover(SourceRequest(query="ideas", limit=3)))

    assert response.provider == "evergreen_ideas"
    assert response.query == "ideas"
    assert response.titles == ["A", "B", "C"]
    assert [candidate.mode for candidate in response.candidates] == [
        "evergreen_ideas",
        "evergreen_ideas",
        "evergreen_ideas",
    ]
    assert [candidate.rank for candidate in response.candidates] == [0, 1, 2]
    assert source.calls == 1
    assert source.requests[0].query == "ideas"
    assert source.requests[0].mode == "evergreen_ideas"
    assert source.requests[0].limit == 5
    assert mode.backlog_titles == ("D", "E")
    assert mode.backlog_count == 2
    assert mode.pending == 2
    assert mode.queued == 2
    assert response.metadata["backlog_before"] == 0
    assert response.metadata["backlog_after"] == 2
    assert response.metadata["backlog_size"] == 2
    assert response.metadata["backlog_target"] == 5
    assert response.metadata["replenish_threshold"] == 1
    assert response.metadata["replenished"] is True
    assert response.metadata["replenished_count"] == 5
    assert response.metadata["feed"] == "fake"
    assert response.metadata["source_provider"] == "fake-evergreen"


def test_an_ample_backlog_is_served_without_touching_the_source() -> None:
    source = FakeSource(
        candidates=[_candidate(name) for name in ("A", "B", "C", "D", "E", "F")]
    )
    mode = EvergreenMode(
        source,
        backlog_size=5,
        replenish_threshold=1,
        replenish_batch=5,
        now=NOW,
    )

    first = _run(mode.discover(SourceRequest(limit=3)))
    second = _run(mode.discover(SourceRequest(limit=2)))

    assert first.titles == ["A", "B", "C"]
    assert second.titles == ["D", "E"]
    assert source.calls == 1
    assert second.metadata["replenished"] is False
    assert second.metadata["replenished_count"] == 0
    assert second.metadata["backlog_after"] == 0


def test_the_backlog_is_replenished_at_the_low_water_mark() -> None:
    source = FakeSource(
        [
            [_candidate("A"), _candidate("B"), _candidate("C"), _candidate("D")],
            [_candidate("E"), _candidate("F"), _candidate("G"), _candidate("H")],
        ]
    )
    mode = EvergreenMode(
        source,
        backlog_size=4,
        replenish_threshold=2,
        replenish_batch=4,
        now=NOW,
    )

    first = _run(mode.discover(SourceRequest(limit=2)))
    assert first.titles == ["A", "B"]
    assert mode.backlog_titles == ("C", "D")
    assert source.calls == 1
    assert mode.needs_replenishment is True

    second = _run(mode.discover(SourceRequest(limit=2)))

    assert second.titles == ["C", "D"]
    assert source.calls == 2
    assert second.metadata["replenished"] is True
    assert second.metadata["replenished_count"] == 2
    assert mode.backlog_titles == ("E", "F")


def test_past_topics_are_not_served_again_inside_the_window() -> None:
    source = FakeSource(candidates=[_candidate("A"), _candidate("B")])
    mode = EvergreenMode(
        source,
        backlog_size=2,
        replenish_threshold=0,
        replenish_batch=2,
        now=NOW,
    )

    first = _run(mode.discover(SourceRequest(limit=1)))
    second = _run(mode.discover(SourceRequest(limit=1)))
    third = _run(mode.discover(SourceRequest(limit=1)))

    assert first.titles == ["A"]
    assert second.titles == ["B"]
    assert third.titles == []
    assert source.calls == 2
    assert third.metadata["skipped_past"] == 2
    assert mode.history.titles == ("A", "B")
    assert mode.backlog_titles == ()


def test_a_preloaded_history_suppresses_known_titles() -> None:
    history = TopicHistory(["Old Idea"], now=NOW)
    source = FakeSource(candidates=[_candidate("Old Idea"), _candidate("Fresh Idea")])
    mode = EvergreenMode(
        source,
        history=history,
        backlog_size=2,
        replenish_threshold=0,
        now=NOW,
    )

    response = _run(mode.discover(SourceRequest(limit=2)))

    assert response.titles == ["Fresh Idea"]
    assert response.metadata["skipped_past"] == 1
    assert mode.history.titles == ("Old Idea", "Fresh Idea")


def test_dedupe_normalizes_case_underscores_and_whitespace() -> None:
    history = TopicHistory(["moon_landing"], now=NOW)
    source = FakeSource(
        candidates=[
            _candidate("  moon   LANDING "),
            _candidate("Another topic"),
        ]
    )
    mode = EvergreenMode(
        source,
        history=history,
        backlog_size=2,
        replenish_threshold=0,
        now=NOW,
    )

    response = _run(mode.discover(SourceRequest(limit=2)))

    assert response.titles == ["Another topic"]
    assert normalize_title("  Moon_LANDING ") == "moon landing"


def test_a_duplicate_in_the_source_batch_and_the_backlog_is_reported_once() -> None:
    source = FakeSource(
        candidates=[
            _candidate("A"),
            _candidate(" a "),
            _candidate("B"),
        ]
    )
    mode = EvergreenMode(
        source,
        backlog=[_candidate("A")],
        backlog_size=3,
        replenish_threshold=0,
        now=NOW,
    )

    response = _run(mode.discover(SourceRequest(limit=2)))

    assert response.titles == ["A", "B"]
    assert response.metadata["skipped_duplicates"] == 2
    assert mode.backlog_titles == ()


def test_only_served_candidates_enter_the_history() -> None:
    source = FakeSource(candidates=[_candidate("A"), _candidate("B"), _candidate("C")])
    mode = EvergreenMode(
        source,
        backlog_size=3,
        replenish_threshold=0,
        now=NOW,
    )

    response = _run(mode.discover(SourceRequest(limit=1)))

    assert response.titles == ["A"]
    assert mode.history.titles == ("A",)
    assert mode.backlog_titles == ("B", "C")


def test_a_zero_dedupe_window_disables_history_filtering() -> None:
    history = TopicHistory(["A"], now=NOW)
    source = FakeSource(candidates=[_candidate("A")])
    mode = EvergreenMode(
        source,
        history=history,
        dedupe_window=0,
        backlog_size=1,
        replenish_threshold=0,
        now=NOW,
    )

    response = _run(mode.discover(SourceRequest(limit=1)))

    assert response.titles == ["A"]
    assert response.metadata["skipped_past"] == 0
    assert response.metadata["dedupe_window_seconds"] == 0


def test_a_title_may_return_once_the_dedupe_window_expires() -> None:
    clock = Clock()
    source = FakeSource([[_candidate("A")], [_candidate("A")]])
    mode = EvergreenMode(
        source,
        dedupe_window=1,
        backlog_size=1,
        replenish_threshold=0,
        now=clock,
    )

    assert _run(mode.discover(SourceRequest(limit=1))).titles == ["A"]
    clock.advance(timedelta(days=1))
    assert _run(mode.discover(SourceRequest(limit=1))).titles == ["A"]
    assert source.calls == 2


def test_the_target_caps_how_many_ideas_are_kept() -> None:
    source = FakeSource(candidates=[_candidate(name) for name in ("A", "B", "C", "D")])
    mode = EvergreenMode(
        source,
        backlog_size=2,
        replenish_threshold=0,
        now=NOW,
    )

    response = _run(mode.discover(SourceRequest(limit=10)))

    assert response.titles == ["A", "B"]
    assert mode.backlog_titles == ()
    assert response.metadata["backlog_target"] == 2


def test_candidates_can_be_added_and_the_backlog_cleared_manually() -> None:
    mode = EvergreenMode(
        FakeSource(),
        history=TopicHistory(["Past"], now=NOW),
        backlog_size=4,
        replenish_threshold=0,
        now=NOW,
    )

    added = mode.add(["New", _candidate("Past"), _candidate("Another")])
    mode.add([_candidate("New")])

    assert [candidate.title for candidate in added] == ["New", "Another"]
    assert mode.backlog_titles == ("New", "Another")
    assert [candidate.mode for candidate in mode.backlog] == [
        "evergreen_ideas",
        "evergreen_ideas",
    ]

    mode.clear_backlog()

    assert mode.backlog == ()
    assert mode.history.titles == ("Past",)


def test_replenish_forces_a_refill() -> None:
    source = FakeSource([[_candidate("A")], [_candidate("B")]])
    mode = EvergreenMode(
        source,
        backlog_size=1,
        replenish_threshold=0,
        now=NOW,
    )

    assert [candidate.title for candidate in _run(mode.replenish())] == ["A"]
    assert mode.backlog_titles == ("A",)
    mode.clear_backlog()
    assert [candidate.title for candidate in _run(mode.refill())] == ["B"]
    assert source.calls == 2


def test_metadata_is_preserved_when_no_replenishment_happens() -> None:
    source = FakeSource([[_candidate("A"), _candidate("B")]])
    mode = EvergreenMode(
        source,
        backlog_size=2,
        replenish_threshold=0,
        now=NOW,
    )

    _run(mode.discover(SourceRequest(limit=1)))
    response = _run(mode.discover(SourceRequest(limit=1)))

    assert response.titles == ["B"]
    assert response.metadata["mode"] == "evergreen_ideas"
    assert response.metadata["source_provider"] == "fake-evergreen"
    assert response.metadata["replenished"] is False
    assert response.metadata["history_size"] == 2


def test_a_source_class_is_created_once() -> None:
    created: list[int] = []

    class CountingSource(FakeSource):
        def __init__(self) -> None:
            super().__init__([[_candidate("A")], [_candidate("B")]])
            created.append(1)

    mode = EvergreenMode(
        CountingSource,
        backlog_size=1,
        replenish_threshold=0,
        now=NOW,
    )

    _run(mode.discover(SourceRequest(limit=1)))
    _run(mode.discover(SourceRequest(limit=1)))

    assert created == [1]


@pytest.mark.parametrize(
    "options",
    [
        {"backlog_size": 0},
        {"backlog_size": -1},
        {"replenish_threshold": -1},
        {"replenish_batch": 0},
        {"backlog_size": 2, "replenish_threshold": 3},
        {"dedupe_window": -1},
        {"backlog_size": "many"},
        {"replenish_batch": 1.5},
    ],
)
def test_invalid_options_are_rejected(options: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        EvergreenMode(FakeSource(), **options)


def test_conflicting_aliases_are_rejected() -> None:
    with pytest.raises(ConfigError, match="either backlog_size or target_backlog"):
        EvergreenMode(FakeSource(), backlog_size=1, target_backlog=1)

    with pytest.raises(ConfigError, match="either replenish_threshold or min_backlog"):
        EvergreenMode(FakeSource(), replenish_threshold=1, min_backlog=1)

    with pytest.raises(ConfigError, match="either replenish_threshold or replenish_at"):
        EvergreenMode(FakeSource(), replenish_threshold=1, replenish_at=1)

    with pytest.raises(ConfigError, match="either replenish_batch or batch_size"):
        EvergreenMode(FakeSource(), replenish_batch=1, batch_size=1)


def test_discover_and_replenish_reject_a_non_request() -> None:
    mode = EvergreenMode(FakeSource(), now=NOW)

    with pytest.raises(TypeError, match="SourceRequest"):
        _run(mode.discover("not a request"))  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="SourceRequest"):
        _run(mode.replenish("not a request"))  # type: ignore[arg-type]


def test_a_custom_history_store_is_accepted() -> None:
    class Store:
        def __init__(self) -> None:
            self.entries: list[SeenTopic] = []

        def seen_within(
            self,
            title: str,
            window: timedelta,
            *,
            now: datetime | None = None,
        ) -> bool:
            return any(entry.title == title for entry in self.entries)

        def record(
            self,
            candidates: Any,
            *,
            at: datetime | None = None,
            mode: str = "",
        ) -> list[SeenTopic]:
            records = [
                SeenTopic(candidate.title, at or NOW, mode) for candidate in candidates
            ]
            self.entries.extend(records)
            return records

        def __len__(self) -> int:
            return len(self.entries)

    store = Store()
    source = FakeSource(candidates=[_candidate("A")])
    mode = EvergreenMode(
        source,
        history=store,
        backlog_size=1,
        replenish_threshold=0,
        now=NOW,
    )

    assert _run(mode.discover(SourceRequest(limit=1))).titles == ["A"]
    assert [entry.title for entry in store.entries] == ["A"]
    assert mode.history is store
