"""On this day mode: importance filtering and dedupe inside the window."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from aivia.core.errors import ConfigError
from aivia.providers.protocols.source import SourceRequest, SourceResponse
from aivia.research import (
    DEFAULT_DEDUPE_WINDOW,
    DEFAULT_MIN_IMPORTANCE,
    OnThisDay,
    OnThisDayMode,
    OnThisDayTopicMode,
    OTDMode,
    SeenTopic,
    TopicHistory,
    WikipediaOnThisDayMode,
    candidate_importance,
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


class FakeSource:
    """An offline On this day source returning fixed candidates."""

    def __init__(
        self,
        candidates: list[Any] | None = None,
        *,
        provider: str = "fake-otd",
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


def _candidate(
    title: str,
    category: str = "selected",
    *,
    year: int | None = None,
) -> Any:
    from aivia.core.models.topic import TopicCandidate

    return TopicCandidate(
        title=title,
        category=category,
        year=year,
        source_urls=[f"https://example.test/{title.replace(' ', '_')}"],
    )


def _run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


def test_the_mode_is_registered_and_aliases_share_one_class() -> None:
    assert resolve_mode("on_this_day") is OnThisDayMode
    assert OTDMode is OnThisDayMode
    assert OnThisDayTopicMode is OnThisDayMode
    assert WikipediaOnThisDayMode is OnThisDayMode
    assert OnThisDay is OnThisDayMode
    assert OnThisDayMode().name == "on_this_day"
    assert timedelta(days=365) == DEFAULT_DEDUPE_WINDOW
    assert DEFAULT_MIN_IMPORTANCE == 0.5


def test_discover_tags_candidates_and_preserves_the_source_context() -> None:
    source = FakeSource(
        [_candidate("Apollo 11", "selected"), _candidate("Moon landing", "events")]
    )
    mode = OnThisDayMode(source, now=NOW)
    request = SourceRequest(query="history", mode="something_else")

    response = _run(mode.discover(request))

    assert source.requests == [request]
    assert response.provider == "on_this_day"
    assert response.query == "history"
    assert response.titles == ["Apollo 11", "Moon landing"]
    assert [candidate.mode for candidate in response.candidates] == [
        "on_this_day",
        "on_this_day",
    ]
    assert [candidate.rank for candidate in response.candidates] == [0, 1]
    assert response.metadata["mode"] == "on_this_day"
    assert response.metadata["source_provider"] == "fake-otd"
    assert response.metadata["feed"] == "fake"
    assert response.metadata["skipped_importance"] == 0
    assert response.metadata["skipped_duplicates"] == 0
    assert response.metadata["dedupe_window_seconds"] == 365 * 86400


def test_importance_filter_drops_categories_below_the_threshold() -> None:
    source = FakeSource(
        [
            _candidate("Apollo 11", "selected"),
            _candidate("A treaty", "events"),
            _candidate("A birth", "births"),
            _candidate("A death", "deaths"),
            _candidate("A holiday", "holidays"),
        ]
    )
    mode = OnThisDayMode(source, min_importance=0.5, now=NOW)

    response = _run(mode.discover(SourceRequest()))

    assert response.titles == ["Apollo 11", "A treaty", "A birth", "A death"]
    assert response.metadata["skipped_importance"] == 1


def test_importance_weights_are_configurable() -> None:
    source = FakeSource(
        [_candidate("Apollo 11", "selected"), _candidate("A holiday", "holidays")]
    )
    mode = OnThisDayMode(
        source,
        min_importance=0.95,
        importance_weights={"selected": 1.0, "holidays": 0.9},
        now=NOW,
    )

    response = _run(mode.discover(SourceRequest()))

    assert response.titles == ["Apollo 11"]
    assert mode.importance_weights == {"selected": 1.0, "holidays": 0.9}


def test_a_custom_importance_callable_replaces_category_weights() -> None:
    source = FakeSource(
        [
            _candidate("Ancient event", "events", year=1000),
            _candidate("Modern event", "selected", year=2000),
        ]
    )
    mode = OnThisDayMode(
        source,
        min_importance=0.5,
        importance=lambda candidate: 1.0 if (candidate.year or 0) < 1950 else 0.0,
        now=NOW,
    )

    response = _run(mode.discover(SourceRequest()))

    assert response.titles == ["Ancient event"]


def test_candidate_importance_uses_the_category_and_unknown_default() -> None:
    assert candidate_importance(_candidate("x", "selected")) == 1.0
    assert candidate_importance(_candidate("x", "holidays")) == 0.4
    assert candidate_importance(_candidate("x", "unknown")) == 0.5


def test_repeated_calls_do_not_repeat_inside_the_dedupe_window() -> None:
    source = FakeSource(
        [_candidate("Apollo 11", "selected"), _candidate("Moon landing", "events")]
    )
    clock = Clock()
    mode = OnThisDayMode(source, now=clock)

    first = _run(mode.discover(SourceRequest()))
    second = _run(mode.discover(SourceRequest()))

    assert first.titles == ["Apollo 11", "Moon landing"]
    assert second.titles == []
    assert second.metadata["skipped_duplicates"] == 2
    assert mode.history.titles == ("Apollo 11", "Moon landing")

    clock.advance(timedelta(days=364))
    assert _run(mode.discover(SourceRequest())).titles == []

    clock.advance(timedelta(days=1))
    third = _run(mode.discover(SourceRequest()))
    assert third.titles == ["Apollo 11", "Moon landing"]


def test_a_preloaded_history_suppresses_known_titles() -> None:
    history = TopicHistory(["apollo_11"], now=NOW)
    source = FakeSource(
        [_candidate("Apollo 11", "selected"), _candidate("Moon landing", "events")]
    )
    mode = OnThisDayMode(source, history=history, now=NOW)

    response = _run(mode.discover(SourceRequest()))

    assert response.titles == ["Moon landing"]
    assert response.metadata["skipped_duplicates"] == 1


def test_dedupe_normalizes_case_underscores_and_whitespace() -> None:
    source = FakeSource(
        [
            _candidate("Moon_landing", "events"),
            _candidate("  moon   LANDING ", "events"),
            _candidate("Another topic", "events"),
        ]
    )
    mode = OnThisDayMode(source, now=NOW)

    response = _run(mode.discover(SourceRequest()))

    assert response.titles == ["Moon_landing", "Another topic"]
    assert response.metadata["skipped_duplicates"] == 1
    assert normalize_title("  Moon_LANDING ") == "moon landing"


def test_only_returned_candidates_enter_the_history() -> None:
    source = FakeSource(
        [
            _candidate("Apollo 11", "selected"),
            _candidate("A holiday", "holidays"),
        ]
    )
    mode = OnThisDayMode(source, min_importance=0.5, now=NOW)

    response = _run(mode.discover(SourceRequest()))

    assert response.titles == ["Apollo 11"]
    assert mode.history.titles == ("Apollo 11",)
    assert "A holiday" not in mode.history


def test_a_zero_window_disables_repetition_filtering() -> None:
    history = TopicHistory(["Apollo 11"], now=NOW)
    source = FakeSource([_candidate("Apollo 11", "selected")])
    mode = OnThisDayMode(source, history=history, dedupe_window=0, now=NOW)

    response = _run(mode.discover(SourceRequest()))

    assert response.titles == ["Apollo 11"]
    assert response.metadata["skipped_duplicates"] == 0


def test_a_numeric_dedupe_window_is_read_as_days() -> None:
    clock = Clock()
    source = FakeSource([_candidate("Apollo 11", "selected")])
    mode = OnThisDayMode(source, dedupe_window=2, now=clock)

    assert _run(mode.discover(SourceRequest())).titles == ["Apollo 11"]
    clock.advance(timedelta(days=1))
    assert _run(mode.discover(SourceRequest())).titles == []
    clock.advance(timedelta(days=1))
    assert _run(mode.discover(SourceRequest())).titles == ["Apollo 11"]


def test_disabling_the_importance_filter_keeps_every_category() -> None:
    source = FakeSource(
        [_candidate("A holiday", "holidays"), _candidate("Unknown", "mystery")]
    )
    mode = OnThisDayMode(source, min_importance=None, now=NOW)

    response = _run(mode.discover(SourceRequest()))

    assert response.titles == ["A holiday", "Unknown"]
    assert mode.min_importance is None


def test_history_prune_forgets_entries_outside_the_window() -> None:
    history = TopicHistory([SeenTopic("Apollo 11", NOW)])
    assert len(history) == 1
    assert history.seen_within("apollo 11", timedelta(days=30), now=NOW)

    removed = history.prune(timedelta(days=30), now=NOW + timedelta(days=30))

    assert removed == 1
    assert len(history) == 0
    assert "Apollo 11" not in history


def test_source_may_be_a_class_or_a_factory_and_is_created_once() -> None:
    created: list[int] = []

    class CountingSource(FakeSource):
        def __init__(self) -> None:
            super().__init__([_candidate("Apollo 11", "selected")])
            created.append(1)

    mode = OnThisDayMode(CountingSource, now=NOW)
    _run(mode.discover(SourceRequest()))
    _run(mode.discover(SourceRequest()))

    assert created == [1]

    factory_calls: list[int] = []

    def factory() -> FakeSource:
        factory_calls.append(1)
        return FakeSource([_candidate("Moon landing", "events")])

    factory_mode = OnThisDayMode(source_factory=factory, now=NOW)
    response = _run(factory_mode.discover(SourceRequest()))

    assert response.titles == ["Moon landing"]
    assert factory_calls == [1]


def test_a_source_without_discover_is_rejected() -> None:
    mode = OnThisDayMode(object(), now=NOW)

    with pytest.raises(ConfigError, match="must define discover"):
        _run(mode.discover(SourceRequest()))


@pytest.mark.parametrize(
    "options",
    [
        {"min_importance": -0.1},
        {"min_importance": 1.1},
        {"min_importance": float("inf")},
        {"dedupe_window": -1},
        {"importance_weights": {"selected": -1}},
        {"importance_weights": {"selected": float("nan")}},
    ],
)
def test_invalid_options_are_rejected(options: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        OnThisDayMode(FakeSource(), **options)


def test_both_source_and_provider_cannot_be_given() -> None:
    with pytest.raises(ConfigError, match="either source or provider"):
        OnThisDayMode(FakeSource(), provider=FakeSource())

    with pytest.raises(ConfigError, match="either source or source_factory"):
        OnThisDayMode(FakeSource(), source_factory=FakeSource)


def test_discover_rejects_a_non_request() -> None:
    mode = OnThisDayMode(FakeSource(), now=NOW)

    with pytest.raises(TypeError, match="SourceRequest"):
        _run(mode.discover("not a request"))  # type: ignore[arg-type]
