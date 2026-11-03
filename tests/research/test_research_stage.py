"""The research stage chains collection, dedupe, gating, scoring, verification
and selection into one topic input."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from datetime import UTC, datetime, timedelta
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
    SeenTopic,
    SensitivityStage,
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


def _stage_with_config(**overrides: Any) -> ResearchStage:
    options: dict[str, Any] = {
        "sources": _sources(),
        "request": SourceRequest(query="renewable energy"),
        "derivation": _derivation(),
        "min_relevance": 0.3,
        "rubric": _rubric(),
        "scores": SCORES,
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
        "screened": 4,
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


def test_blocked_and_review_topics_never_reach_verification_or_selection() -> None:
    source = FakeSource(
        "alpha",
        [
            _candidate(
                "Solar panel costs",
                "https://alpha.example/solar",
                "https://grid.example/solar",
                category="energy",
            ),
            _candidate(
                "School shooting survivor speaks",
                "https://alpha.example/violence",
                "https://news.example/violence",
                category="violence",
            ),
            _candidate(
                "Suicide prevention helpline",
                "https://alpha.example/health",
                "https://health.example/helpline",
                category="health",
            ),
        ],
    )
    stage = _stage(
        sources=[source],
        gate=RelevanceGate(relevance=lambda candidate: 1.0, min_relevance=0.0),
        scores={
            **SCORES,
            "School shooting survivor speaks": {"relevance": 0.9, "novelty": 0.9},
            "Suicide prevention helpline": {"relevance": 0.9, "novelty": 0.9},
        },
        limit=None,
        max_per_category=None,
    )

    result = stage.run_sync()

    assert result.sensitivity is not None
    assert result.sensitivity.blocked_titles == ["School shooting survivor speaks"]
    assert result.sensitivity.review_titles == ["Suicide prevention helpline"]
    assert result.screened == [result.scored[0]]
    assert result.blocked_titles == ["School shooting survivor speaks"]
    assert result.review_titles == ["Suicide prevention helpline"]

    assert result.verification is not None
    assert result.verification.input_count == 1
    assert result.verification.verified_titles == ["Solar panel costs"]
    assert result.selected_titles == ["Solar panel costs"]
    assert result.metadata["counts"]["scored"] == 3
    assert result.metadata["counts"]["screened"] == 1
    assert result.metadata["counts"]["verified"] == 1
    assert result.metadata["sensitivity"] == {
        "input_count": 3,
        "allowed": 1,
        "blocked": 1,
        "manual_review": 1,
        "blocked_titles": ["School shooting survivor speaks"],
        "review_titles": ["Suicide prevention helpline"],
    }


def test_a_custom_sensitivity_stage_can_replace_the_default_rules() -> None:
    source = FakeSource(
        "alpha",
        [
            _candidate(
                "Solar panel costs",
                "https://alpha.example/solar",
                "https://grid.example/solar",
                category="energy",
            )
        ],
    )
    stage = _stage(
        sources=[source],
        sensitivity=SensitivityStage(
            [{"name": "solar", "terms": ["solar"], "action": "blocked"}]
        ),
        limit=None,
        max_per_category=None,
    )

    result = stage.run_sync()

    assert result.sensitivity is not None
    assert result.sensitivity.blocked_titles == ["Solar panel costs"]
    assert result.selected_titles == []
    assert result.verification is not None
    assert result.verification.input_count == 0


def test_the_sensitivity_filter_can_be_disabled() -> None:
    source = FakeSource(
        "alpha",
        [
            _candidate(
                "School shooting survivor speaks",
                "https://alpha.example/violence",
                "https://news.example/violence",
                category="violence",
            )
        ],
    )
    stage = _stage(
        sources=[source],
        sensitivity_enabled=False,
        gate=RelevanceGate(relevance=lambda candidate: 1.0, min_relevance=0.0),
        scores={
            "School shooting survivor speaks": {"relevance": 0.9, "novelty": 0.9},
        },
        limit=None,
        max_per_category=None,
    )

    result = stage.run_sync()

    assert result.sensitivity is not None
    assert result.sensitivity.metadata["enabled"] is False
    assert result.blocked_titles == []
    assert result.selected_titles == ["School shooting survivor speaks"]


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
    assert history.titles == tuple(first.selected_titles)

    second = stage.run_sync()

    assert first.selected_titles == ["Solar panel costs", "Battery storage guide"]
    assert second.dedupe is not None
    assert {"Solar panel costs", "Battery storage guide"} <= set(
        second.dedupe.duplicate_titles
    )
    assert not set(second.selected_titles) & set(first.selected_titles)


def test_only_selected_topics_are_recorded_in_dedupe_history() -> None:
    history = DedupeHistory()
    stage = _stage(history=history, record=True, limit=1, max_per_category=None)

    result = stage.run_sync()

    assert result.selected_titles == ["Solar panel costs"]
    assert history.titles == ("Solar panel costs",)
    assert result.dedupe is not None
    assert result.dedupe.metadata["recorded"] is True
    assert result.dedupe.metadata["recorded_count"] == 1


def test_gating_rejections_are_not_recorded_and_can_be_reconsidered() -> None:
    history = DedupeHistory()
    gate = RelevanceGate(
        relevance=lambda candidate: (
            0.1 if candidate.title == "Battery storage guide" else 1.0
        ),
        min_relevance=0.5,
    )

    first = _stage(
        history=history,
        record=True,
        gate=gate,
        limit=None,
        max_per_category=None,
    ).run_sync()

    assert first.gating is not None
    assert "Battery storage guide" in first.gating.rejected_titles
    assert "Battery storage guide" not in history.titles
    assert history.titles == tuple(first.selected_titles)

    second = _stage(
        history=history,
        record=True,
        limit=None,
        max_per_category=None,
    ).run_sync()

    assert "Battery storage guide" in second.selected_titles


def test_scoring_rejections_are_not_recorded_and_can_be_reconsidered() -> None:
    history = DedupeHistory()
    low_scores = {
        **SCORES,
        "Battery storage guide": {"relevance": 0.1, "novelty": 0.1},
    }

    first = _stage(
        history=history,
        record=True,
        scores=low_scores,
        limit=None,
        max_per_category=None,
    ).run_sync()

    assert first.scoring is not None
    assert "Battery storage guide" in [
        candidate.title for candidate in first.scoring.dropped
    ]
    assert "Battery storage guide" not in history.titles

    second = _stage(
        history=history,
        record=True,
        limit=None,
        max_per_category=None,
    ).run_sync()

    assert "Battery storage guide" in second.selected_titles


def test_the_stage_reads_the_dedupe_window_from_research_config() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)

    def history() -> DedupeHistory:
        return DedupeHistory(
            [
                SeenTopic(
                    title="Solar panel costs",
                    seen_at=now - timedelta(days=10),
                    profile="earlier",
                )
            ],
            now=now,
        )

    short = _stage(
        history=history(),
        now=now,
        config={"research": {"dedupe_window": 7}},
        record=False,
    ).run_sync()
    long = _stage(
        history=history(),
        now=now,
        config={"research": {"dedupe_window": 30}},
        record=False,
    ).run_sync()

    assert short.dedupe is not None
    assert short.dedupe.window == timedelta(days=7)
    assert "Solar panel costs" not in short.dedupe.duplicate_titles
    assert long.dedupe is not None
    assert long.dedupe.window == timedelta(days=30)
    assert "Solar panel costs" in long.dedupe.duplicate_titles


def test_the_stage_reads_the_embedding_threshold_from_research_config() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)

    def history() -> DedupeHistory:
        return DedupeHistory(
            [
                SeenTopic(
                    title="Earlier angle",
                    seen_at=now - timedelta(hours=1),
                    profile="earlier",
                    embedding=(1.0, 0.0),
                )
            ],
            now=now,
        )

    def stage(threshold: float) -> ResearchStage:
        source = FakeSource(
            "alpha",
            [
                _candidate(
                    "A new angle",
                    "https://alpha.example/angle",
                    "https://other.example/angle",
                )
            ],
        )
        return _stage(
            sources=[source],
            history=history(),
            embeddings={"A new angle": (0.98, 0.1)},
            now=now,
            config={"research": {"embedding_threshold": threshold}},
            gate=RelevanceGate(relevance=lambda candidate: 1.0, min_relevance=0.0),
            scores={"A new angle": {"relevance": 0.9, "novelty": 0.9}},
            limit=None,
            max_per_category=None,
            record=False,
        )

    low = stage(0.9).run_sync()
    high = stage(0.999).run_sync()

    assert low.dedupe is not None
    assert low.dedupe.embedding_threshold == 0.9
    assert low.dedupe.embedding_duplicates == 1
    assert low.selected_titles == []
    assert high.dedupe is not None
    assert high.dedupe.embedding_threshold == 0.999
    assert high.dedupe.embedding_duplicates == 0
    assert high.selected_titles == ["A new angle"]


def test_explicit_dedupe_options_override_the_research_config() -> None:
    stage = _stage(
        config={"research": {"dedupe_window": 7, "embedding_threshold": 0.95}},
        window=30,
        embedding_threshold=0.8,
    )

    result = stage.run_sync()

    assert stage.window == timedelta(days=30)
    assert stage.embedding_threshold == 0.8
    assert result.dedupe is not None
    assert result.dedupe.window == timedelta(days=30)
    assert result.dedupe.embedding_threshold == 0.8


def test_the_stage_reads_the_selection_limits_from_research_config() -> None:
    stage = _stage_with_config(
        config={"research": {"max_topics": 1, "max_per_category": 2}}
    )

    result = stage.run_sync()

    assert stage.limit == 1
    assert stage.max_per_category == 2
    assert result.selection is not None
    assert result.selection.limit == 1
    assert result.selection.max_per_category == 2
    assert result.selected_titles == ["Solar panel costs"]
    assert result.metadata["counts"]["selected"] == 1
    assert stage.to_dict()["limit"] == 1
    assert stage.to_dict()["max_per_category"] == 2


def test_the_stage_reads_the_per_category_limit_from_research_config() -> None:
    def stage(max_per_category: int) -> ResearchStage:
        return _stage_with_config(
            config={"research": {"max_per_category": max_per_category}},
            require_verified=False,
        )

    one = stage(1).run_sync()
    two = stage(2).run_sync()

    assert one.selection is not None
    assert one.selection.max_per_category == 1
    assert one.selected_titles == ["Solar panel costs", "Battery storage guide"]
    assert two.selection is not None
    assert two.selection.max_per_category == 2
    assert two.selected_titles == [
        "Solar panel costs",
        "Duplicate solar guide",
        "Battery storage guide",
    ]


def test_explicit_selection_options_override_the_research_config() -> None:
    stage = _stage(
        config={"research": {"max_topics": 5, "max_per_category": 2}},
        limit=1,
        max_per_category=None,
    )

    result = stage.run_sync()

    assert stage.limit == 1
    assert stage.max_per_category is None
    assert result.selection is not None
    assert result.selection.limit == 1
    assert result.selection.max_per_category is None
    assert result.selected_titles == ["Solar panel costs"]


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
        "screened": 0,
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
        "sensitivity",
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
