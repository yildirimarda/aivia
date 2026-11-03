"""The research stage chains collection, dedupe, gating, scoring, verification
and selection into one topic input."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

import pytest

from aivia.core.errors import ConfigError
from aivia.core.models.topic import TopicCandidate
from aivia.providers.protocols.source import SourceRequest, SourceResponse
from aivia.research import (
    BriefDerivation,
    CollectionResult,
    DedupeHistory,
    GatingResult,
    RelevanceGate,
    ResearchResult,
    ResearchStage,
    ScoringRubric,
    TopicInput,
    build_research_stage,
    collect_sources,
    research_topics,
    run_research_stage,
)


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine)


def _candidate(
    title: str,
    *urls: str,
    category: str = "",
) -> TopicCandidate:
    return TopicCandidate(
        title=title,
        source_urls=list(urls),
        category=category,
    )


def _derivation() -> BriefDerivation:
    return BriefDerivation(
        brief="Practical guides to home renewable energy.",
        queries=["home renewable energy", "solar panels"],
        subtopics=["battery storage", "electrical grid"],
        keywords=["renewable energy"],
        include_terms=["solar", "wind", "battery"],
        exclude_terms=["celebrity"],
    )


def _rubric() -> ScoringRubric:
    return ScoringRubric(
        weights={"relevance": 0.5, "novelty": 0.5},
        min_score=0.6,
    )


SCORES: dict[str, dict[str, float]] = {
    "Solar panel costs": {"relevance": 0.9, "novelty": 0.9},
    "Duplicate solar guide": {"relevance": 0.9, "novelty": 0.9},
    "Wind turbine guide": {"relevance": 0.9, "novelty": 0.9},
    "Battery storage guide": {"relevance": 0.8, "novelty": 0.8},
    "Solar gadget roundup": {"relevance": 0.2, "novelty": 0.2},
    "Celebrity chef opens restaurant": {"relevance": 0.9, "novelty": 0.9},
}


class FakeSource:
    """An offline source returning fixed candidates or raising an error."""

    def __init__(
        self,
        name: str,
        candidates: list[TopicCandidate],
        *,
        error: BaseException | None = None,
    ) -> None:
        self.name = name
        self.candidates = list(candidates)
        self.error = error
        self.requests: list[SourceRequest] = []

    async def discover(self, request: SourceRequest) -> SourceResponse:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return SourceResponse(
            provider=self.name,
            query=request.query,
            candidates=[candidate.model_copy() for candidate in self.candidates],
        )


def _sources() -> list[FakeSource]:
    alpha = FakeSource(
        "alpha",
        [
            _candidate(
                "Solar panel costs",
                "https://alpha.example/solar",
                "https://grid.example/solar",
                category="energy",
            ),
            _candidate(
                "Celebrity chef opens restaurant",
                "https://alpha.example/chef",
                "https://food.example/chef",
                category="culture",
            ),
            _candidate(
                "Duplicate solar guide",
                "https://alpha.example/duplicate",
                "https://mirror.example/duplicate",
                category="energy",
            ),
        ],
    )
    beta = FakeSource(
        "beta",
        [
            _candidate(
                "Wind turbine guide",
                "https://beta.example/wind",
                category="energy",
            ),
            _candidate(
                "Battery storage guide",
                "https://beta.example/battery",
                "https://storage.example/battery",
                category="storage",
            ),
            _candidate(
                "Solar gadget roundup",
                "https://beta.example/gadgets",
                "https://shop.example/gadgets",
                category="products",
            ),
            _candidate(
                "Duplicate solar guide",
                "https://beta.example/duplicate",
                "https://mirror.example/duplicate",
                category="energy",
            ),
        ],
    )
    return [alpha, beta]


def _stage(**overrides: Any) -> ResearchStage:
    options: dict[str, Any] = {
        "sources": _sources(),
        "request": SourceRequest(query="renewable energy"),
        "derivation": _derivation(),
        "min_relevance": 0.3,
        "rubric": _rubric(),
        "scores": SCORES,
        "limit": 2,
        "max_per_category": 1,
        "record": False,
    }
    options.update(overrides)
    return ResearchStage(**options)


def test_the_stage_chains_every_step_into_a_verified_topic_input() -> None:
    stage = _stage()

    result = stage.run_sync()

    assert isinstance(result, TopicInput)
    assert isinstance(result, ResearchResult)
    assert result.selected_titles == ["Solar panel costs", "Battery storage guide"]
    assert result.count == 2
    assert result.input_count == 7
    assert result.complete is True
    assert result.failed_sources == []

    assert result.collection is not None
    assert result.collection.count == 7
    assert result.dedupe is not None
    assert result.dedupe.duplicate_count == 1
    assert result.dedupe.duplicate_titles == ["Duplicate solar guide"]
    assert result.gating is not None
    assert isinstance(result.gating, GatingResult)
    assert result.gating.input_count == 6
    assert "Celebrity chef opens restaurant" in result.gating.rejected_titles
    assert result.scoring is not None
    assert result.scoring.input_count == 5
    assert result.scoring.dropped_count == 1
    assert result.verification is not None
    assert result.verification.unverified_titles == ["Wind turbine guide"]
    assert result.selection is not None
    assert all(topic.verified for topic in result.selected)

    assert result.metadata["counts"] == {
        "candidates": 7,
        "deduped": 6,
        "gated": 5,
        "scored": 4,
        "verified": 3,
        "selected": 2,
    }
    assert [topic.title for topic in result] == result.selected_titles
    assert len(result) == 2


def test_the_gating_result_records_scores_and_rejections() -> None:
    result = _stage().run_sync()

    assert result.gating is not None
    decisions = {decision.title: decision for decision in result.gating.decisions}
    solar = decisions["Solar panel costs"]
    chef = decisions["Celebrity chef opens restaurant"]

    assert solar.score == pytest.approx(1 / 3)
    assert solar.threshold == 0.3
    assert solar.passed is True
    assert solar.rejected is False
    assert chef.score == 0.0
    assert chef.passed is False
    assert chef.rejected is True
    assert chef.reason == "below minimum relevance"
    assert result.gating.titles == [
        "Solar panel costs",
        "Duplicate solar guide",
        "Wind turbine guide",
        "Battery storage guide",
        "Solar gadget roundup",
    ]
    assert result.gating.to_dict()["rejected"] == ["Celebrity chef opens restaurant"]


def test_single_source_topics_are_excluded_until_verification_is_optional() -> None:
    required = _stage(limit=None, max_per_category=None).run_sync()
    optional = _stage(
        limit=None,
        max_per_category=None,
        require_verified=False,
    ).run_sync()

    assert "Wind turbine guide" not in required.selected_titles
    assert "Wind turbine guide" in optional.selected_titles
    assert optional.verification is not None
    assert optional.verification.unverified_titles == ["Wind turbine guide"]
    assert all(
        not topic.verified
        for topic in optional.selected
        if topic.title == "Wind turbine guide"
    )


def test_selection_applies_the_run_limit_after_scoring_and_verification() -> None:
    result = _stage(limit=1, max_per_category=None).run_sync()

    assert result.selected_titles == ["Solar panel costs"]
    assert result.selection is not None
    assert result.selection.rejected_titles
    assert result.metadata["counts"]["selected"] == 1


def test_dedupe_history_is_shared_between_runs() -> None:
    history = DedupeHistory()
    stage = _stage(history=history, record=True)

    first = stage.run_sync()
    second = stage.run_sync()

    assert first.selected_titles == ["Solar panel costs", "Battery storage guide"]
    assert second.selected_titles == []
    assert second.dedupe is not None
    assert second.dedupe.duplicate_count == 7
    assert second.metadata["counts"]["deduped"] == 0


def test_the_async_entry_point_and_sync_wrapper_agree() -> None:
    async_stage = _stage()
    sync_stage = _stage()

    async_result = _run(async_stage.run())
    sync_result = sync_stage.run_sync()

    assert async_result.selected_titles == sync_result.selected_titles
    assert async_result.metadata["counts"] == sync_result.metadata["counts"]
    assert async_stage.sources is not None
    assert async_stage.sources[0].requests[0].query == "renewable energy"


def test_run_collected_chains_the_remaining_stages_from_a_collection() -> None:
    collection = _run(
        collect_sources(
            _sources(),
            SourceRequest(query="renewable energy"),
            timeout=None,
        )
    )
    assert isinstance(collection, CollectionResult)

    result = _stage().run_collected(collection, profile="demo", brief="Home energy")

    assert result.profile == "demo"
    assert result.brief == "Home energy"
    assert result.collection is collection
    assert result.selected_titles == ["Solar panel costs", "Battery storage guide"]


def test_one_failing_source_is_reported_without_losing_the_other_topics() -> None:
    broken = FakeSource(
        "broken",
        [],
        error=RuntimeError("feed unavailable"),
    )
    stage = _stage(sources=[broken, *_sources()])

    result = stage.run_sync()

    assert result.complete is False
    assert result.ok is False
    assert result.failed_sources == ["broken"]
    assert result.selected_titles == ["Solar panel costs", "Battery storage guide"]
    assert result.metadata["sources"] == {
        "attempted": 3,
        "succeeded": 2,
        "failed": 1,
    }


def test_an_async_gate_scorer_is_awaited() -> None:
    async def relevance(candidate: TopicCandidate) -> float:
        return 0.9 if "solar" in candidate.title.lower() else 0.0

    stage = _stage(
        gate=RelevanceGate(relevance=relevance, min_relevance=0.5),
        limit=None,
        max_per_category=None,
    )

    result = _run(stage.run())

    assert result.selected_titles == ["Solar panel costs", "Duplicate solar guide"]


def test_an_empty_source_set_produces_an_empty_topic_input() -> None:
    result = _stage(sources=[]).run_sync()

    assert result.selected == []
    assert result.count == 0
    assert result.metadata["counts"] == {
        "candidates": 0,
        "deduped": 0,
        "gated": 0,
        "scored": 0,
        "verified": 0,
        "selected": 0,
    }
    assert result.collection is not None
    assert result.selection is not None


def test_the_result_serializes_every_stage_summary() -> None:
    payload = _stage().run_sync().to_dict()

    assert payload["titles"] == ["Solar panel costs", "Battery storage guide"]
    assert payload["counts"]["selected"] == 2
    assert payload["request"]["query"] == "renewable energy"
    for stage in (
        "collection",
        "dedupe",
        "gating",
        "scoring",
        "verification",
        "selection",
    ):
        assert payload[stage] is not None


def test_the_builders_and_module_helpers_run_the_same_stage() -> None:
    stage = build_research_stage(
        _sources(),
        request=SourceRequest(query="renewable energy"),
        derivation=_derivation(),
        min_relevance=0.3,
        rubric=_rubric(),
        scores=SCORES,
        limit=1,
        max_per_category=None,
        record=False,
    )

    built = stage.run_sync()
    helper = research_topics(
        _sources(),
        SourceRequest(query="renewable energy"),
        derivation=_derivation(),
        min_relevance=0.3,
        rubric=_rubric(),
        scores=SCORES,
        limit=1,
        max_per_category=None,
        record=False,
    )
    awaited = _run(
        run_research_stage(
            _sources(),
            SourceRequest(query="renewable energy"),
            derivation=_derivation(),
            min_relevance=0.3,
            rubric=_rubric(),
            scores=SCORES,
            limit=1,
            max_per_category=None,
            record=False,
        )
    )

    assert built.selected_titles == ["Solar panel costs"]
    assert helper.selected_titles == built.selected_titles
    assert awaited.selected_titles == built.selected_titles


def test_run_sync_refuses_to_start_inside_a_running_loop() -> None:
    async def call() -> TopicInput:
        return _stage().run_sync()

    with pytest.raises(ConfigError, match="running event loop"):
        _run(call())
