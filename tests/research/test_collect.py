"""The collection stage: concurrency, per source timeouts and failure reports."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

import pytest

from aivia.core.errors import ConfigError
from aivia.core.models.topic import TopicCandidate
from aivia.providers.protocols.source import SourceRequest, SourceResponse
from aivia.providers.sources import SourceProviderBase, SourceRegistry
from aivia.research import (
    CollectionStage,
    collect_sources,
    resolve_sources,
)


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine)


def _candidate(title: str) -> TopicCandidate:
    return TopicCandidate(
        title=title,
        source_urls=[f"https://example.test/{title.replace(' ', '_')}"],
    )


class Tracker:
    """Record how many source calls overlap."""

    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0

    def enter(self) -> None:
        self.active += 1
        self.max_active = max(self.max_active, self.active)

    def leave(self) -> None:
        self.active -= 1


class FakeSource:
    """An offline source that can be slow, fail or return fixed candidates."""

    def __init__(
        self,
        name: str,
        candidates: list[TopicCandidate] | None = None,
        *,
        delay: float = 0.0,
        error: BaseException | None = None,
        tracker: Tracker | None = None,
    ) -> None:
        self.name = name
        self.candidates = list(candidates or [])
        self.delay = delay
        self.error = error
        self.tracker = tracker
        self.requests: list[SourceRequest] = []

    async def discover(self, request: SourceRequest) -> SourceResponse:
        self.requests.append(request)
        if self.tracker is not None:
            self.tracker.enter()
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.error is not None:
                raise self.error
            return SourceResponse(
                provider=self.name,
                query=request.query,
                candidates=[candidate.model_copy() for candidate in self.candidates],
                metadata={"feed": self.name},
            )
        finally:
            if self.tracker is not None:
                self.tracker.leave()


class AlphaProvider(SourceProviderBase):
    """A registered source used to exercise name resolution."""

    def __init__(self) -> None:
        super().__init__(name="alpha")

    async def discover(self, request: SourceRequest) -> SourceResponse:
        return SourceResponse(
            provider="alpha",
            query=request.query,
            candidates=[_candidate("Alpha topic")],
        )


class BetaProvider(SourceProviderBase):
    """A second registered source used to exercise name resolution."""

    def __init__(self) -> None:
        super().__init__(name="beta")

    async def discover(self, request: SourceRequest) -> SourceResponse:
        return SourceResponse(
            provider="beta",
            query=request.query,
            candidates=[_candidate("Beta topic")],
        )


def test_sources_run_concurrently_and_results_keep_source_order() -> None:
    tracker = Tracker()
    slow = FakeSource("slow", [_candidate("Slow topic")], delay=0.05, tracker=tracker)
    fast = FakeSource("fast", [_candidate("Fast topic")], delay=0.01, tracker=tracker)
    other = FakeSource(
        "other", [_candidate("Other topic")], delay=0.02, tracker=tracker
    )

    result = _run(
        collect_sources(
            [slow, fast, other],
            SourceRequest(query="history"),
            concurrency=3,
        )
    )

    assert tracker.max_active == 3
    assert [candidate.title for candidate in result.candidates] == [
        "Slow topic",
        "Fast topic",
        "Other topic",
    ]
    assert [item.source for item in result.results] == ["slow", "fast", "other"]
    assert result.succeeded_sources == ["slow", "fast", "other"]
    assert result.attempted == 3
    assert result.complete
    assert result.failure_report.count == 0
    assert all(source.requests[0].query == "history" for source in (slow, fast, other))


def test_a_slow_source_times_out_and_fast_candidates_are_kept() -> None:
    fast = FakeSource("fast", [_candidate("Fast topic")])
    slow = FakeSource("slow", [_candidate("Slow topic")], delay=5.0)

    result = _run(
        collect_sources(
            [fast, slow],
            timeout=1.0,
            timeouts={"slow": 0.02},
            concurrency=2,
        )
    )

    assert [candidate.title for candidate in result.candidates] == ["Fast topic"]
    report = result.failure_report
    assert report.count == 1
    failure = report.failures[0]
    assert failure.source == "slow"
    assert failure.timeout is True
    assert failure.is_timeout is True
    assert "timed out" in failure.error
    assert not result.complete
    assert result.failed_sources == ["slow"]
    assert result.metadata["failed_sources"] == ["slow"]
    assert result.metadata["succeeded"] == 1


def test_a_failing_source_is_reported_without_losing_other_candidates() -> None:
    broken = FakeSource("broken", error=RuntimeError("boom"))
    good = FakeSource("good", [_candidate("Good topic")])

    result = _run(collect_sources([broken, good], timeout=None))

    assert [candidate.title for candidate in result.candidates] == ["Good topic"]
    failure = result.failure_report.failures[0]
    assert failure.source == "broken"
    assert failure.error == "boom"
    assert failure.error_type == "RuntimeError"
    assert failure.type == "RuntimeError"
    assert failure.timeout is False
    assert failure.to_dict()["source"] == "broken"
    assert result.metadata["errors"][0]["source"] == "broken"
    assert result.to_response().titles == ["Good topic"]


def test_a_failure_propagates_when_continue_on_error_is_false() -> None:
    broken = FakeSource("broken", error=RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        _run(collect_sources([broken], continue_on_error=False))


def test_a_timeout_propagates_when_continue_on_error_is_false() -> None:
    slow = FakeSource("slow", delay=5.0)

    with pytest.raises(TimeoutError):
        _run(
            collect_sources(
                [slow],
                timeout=0.02,
                continue_on_error=False,
            )
        )


def test_a_none_timeout_disables_the_outer_deadline() -> None:
    slow = FakeSource("slow", [_candidate("Slow topic")], delay=0.02)

    result = _run(collect_sources([slow], timeout=None))

    assert result.complete
    assert [candidate.title for candidate in result.candidates] == ["Slow topic"]


def test_registered_source_names_and_resolved_selection_are_accepted() -> None:
    registry = SourceRegistry()
    registry.register_source("alpha", AlphaProvider)
    registry.register_source("beta", BetaProvider)

    by_name = _run(
        collect_sources(
            ["alpha", "beta"],
            SourceRequest(query="science"),
            registry=registry,
        )
    )

    assert [candidate.title for candidate in by_name.candidates] == [
        "Alpha topic",
        "Beta topic",
    ]
    assert by_name.succeeded_sources == ["alpha", "beta"]

    selection = resolve_sources(
        {"sources": ["beta", "alpha"]},
        registry=registry,
    )
    resolved = _run(collect_sources(selection, SourceRequest(query="science")))

    assert [candidate.title for candidate in resolved.candidates] == [
        "Beta topic",
        "Alpha topic",
    ]

    by_mapping = _run(collect_sources({"alpha": AlphaProvider}, registry=registry))

    assert [candidate.title for candidate in by_mapping.candidates] == ["Alpha topic"]


def test_a_source_must_return_a_source_response() -> None:
    class InvalidSource:
        name = "invalid"

        async def discover(self, request: SourceRequest) -> Any:
            return {"candidates": []}

    result = _run(collect_sources([InvalidSource()]))

    failure = result.failure_report.failures[0]
    assert failure.source == "invalid"
    assert failure.error_type == "ConfigError"
    assert "SourceResponse" in failure.error


def test_invalid_timeouts_concurrency_and_unknown_names_are_rejected() -> None:
    for bad_timeout in (0, -1, float("nan"), True):
        with pytest.raises(ConfigError):
            _run(collect_sources([], timeout=bad_timeout))

    bad_concurrency_values: tuple[Any, ...] = (0, -1, 1.5, True, 1000)
    for bad_concurrency in bad_concurrency_values:
        with pytest.raises(ConfigError):
            _run(collect_sources([], concurrency=bad_concurrency))

    source = FakeSource("known")
    with pytest.raises(ConfigError, match="unknown source"):
        _run(collect_sources([source], timeouts={"missing": 1.0}))

    with pytest.raises(ConfigError, match="conflicting"):
        _run(
            collect_sources(
                [source],
                timeouts={"known": 1.0},
                source_timeouts={"known": 2.0},
            )
        )


def test_collection_stage_uses_its_configured_defaults() -> None:
    stage = CollectionStage(timeout=0.02, concurrency=1)
    fast = FakeSource("fast", [_candidate("Fast topic")])
    slow = FakeSource("slow", [_candidate("Slow topic")], delay=5.0)

    result = _run(stage.collect([fast, slow]))
    again = _run(stage.run([fast]))

    assert stage.timeout == 0.02
    assert stage.concurrency == 1
    assert stage.timeouts == {}
    assert stage.continue_on_error is True
    assert result.failure_report.failed_sources == ["slow"]
    assert [candidate.title for candidate in again.candidates] == ["Fast topic"]


def test_no_sources_returns_an_empty_complete_result() -> None:
    result = _run(collect_sources([]))

    assert result.attempted == 0
    assert result.candidates == []
    assert result.complete
    assert result.failure_report.ok
    assert result.to_response().candidates == []
