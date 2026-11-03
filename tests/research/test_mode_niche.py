"""Niche mode: query expansion, concurrent discovery and the relevance gate."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from aivia.core.errors import ConfigError
from aivia.providers.protocols.source import SourceRequest, SourceResponse
from aivia.research import (
    DEFAULT_CONCURRENCY,
    DEFAULT_MAX_QUERIES,
    DEFAULT_MIN_RELEVANCE,
    Niche,
    NicheMode,
    NicheTopicMode,
    QueryExpansion,
    RelevanceGate,
    WikipediaNicheMode,
    candidate_relevance,
    expand_queries,
    filter_candidates,
    is_relevant,
    normalize_terms,
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
        provider: str = "fake-niche",
        fail: set[str] | None = None,
    ) -> None:
        self.results = dict(results or {})
        self.provider = provider
        self.fail = set(fail or ())
        self.requests: list[SourceRequest] = []
        self.active = 0
        self.max_active = 0

    async def discover(self, request: SourceRequest) -> SourceResponse:
        self.requests.append(request)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.01)
            if request.query in self.fail:
                raise RuntimeError(f"boom: {request.query}")
            return SourceResponse(
                provider=self.provider,
                query=request.query,
                candidates=[
                    candidate.model_copy()
                    for candidate in self.results.get(request.query, [])
                ],
                metadata={"feed": "fake"},
            )
        finally:
            self.active -= 1


def _derivation() -> BriefDerivation:
    return BriefDerivation(
        queries=["ancient rome", "Roman engineering"],
        subtopics=["aqueducts", "ancient rome"],
        include_terms=["rome", "aqueduct"],
        exclude_terms=["football"],
        keywords=["history"],
    )


def test_the_mode_is_registered_and_aliases_share_one_class() -> None:
    assert resolve_mode("niche") is NicheMode
    assert Niche is NicheMode
    assert NicheTopicMode is NicheMode
    assert WikipediaNicheMode is NicheMode
    assert NicheMode().name == "niche"
    assert DEFAULT_MIN_RELEVANCE == 0.5
    assert DEFAULT_CONCURRENCY == 8
    assert DEFAULT_MAX_QUERIES == 20


def test_query_expansion_adds_subtopics_keywords_and_niche_without_repeats() -> None:
    derivation = _derivation()

    assert expand_queries(derivation) == [
        "ancient rome",
        "Roman engineering",
        "aqueducts",
        "history",
    ]
    assert expand_queries(derivation, limit=2) == ["ancient rome", "Roman engineering"]
    expansion = QueryExpansion.from_sources(derivation)
    assert expansion.expanded_queries == (
        "ancient rome",
        "Roman engineering",
        "aqueducts",
        "history",
    )
    assert expansion.required_terms == ("rome", "aqueduct")


def test_explicit_values_override_the_derivation() -> None:
    mode = NicheMode(
        FakeSource(),
        derivation=_derivation(),
        queries=["custom query"],
        include_terms=["custom"],
        min_relevance=0.25,
    )

    assert mode.queries == ["custom query", "aqueducts", "ancient rome", "history"]
    assert mode.include_terms == ["custom"]
    assert mode.min_relevance == 0.25
    assert mode.niche == ""


def test_terms_are_stripped_and_deduplicated_case_insensitively() -> None:
    assert normalize_terms(" Rome,  rome ;AQUEDUCT ") == ("Rome", "AQUEDUCT")
    assert normalize_terms(["  Space ", "space", "SPACE "]) == ("Space",)


def test_discovery_runs_every_query_concurrently_and_merges_in_query_order() -> None:
    source = FakeSource(
        {
            "ancient rome": [_candidate("Roman aqueducts")],
            "Roman engineering": [_candidate("Roman aqueduct design")],
            "aqueducts": [_candidate("Roman aqueducts")],
            "history": [_candidate("History of Rome")],
        }
    )
    mode = NicheMode(source, derivation=_derivation())
    request = SourceRequest(query="ignored", language="en", limit=7)

    response = _run(mode.discover(request))

    assert [item.query for item in source.requests] == mode.queries
    assert all(item.mode == "niche" for item in source.requests)
    assert all(item.language == "en" for item in source.requests)
    assert all(item.limit == 7 for item in source.requests)
    assert source.max_active > 1
    assert response.titles == [
        "Roman aqueducts",
        "Roman aqueduct design",
        "History of Rome",
    ]
    assert [candidate.mode for candidate in response.candidates] == ["niche"] * 3
    assert [candidate.rank for candidate in response.candidates] == [0, 1, 2]
    assert response.provider == "niche"
    assert response.query == "ignored"
    assert response.metadata["mode"] == "niche"
    assert response.metadata["source_provider"] == "fake-niche"
    assert response.metadata["feed"] == "fake"
    assert response.metadata["queries"] == mode.queries
    assert response.metadata["expanded_queries"] == mode.queries
    assert response.metadata["query_count"] == 4
    assert response.metadata["source_count"] == 1
    assert response.metadata["min_relevance"] == DEFAULT_MIN_RELEVANCE
    assert response.metadata["skipped_duplicates"] == 1
    assert response.metadata["skipped_relevance"] == 0


def test_the_relevance_gate_drops_excluded_and_off_theme_candidates() -> None:
    source = FakeSource(
        {
            "ancient rome": [
                _candidate("Roman aqueducts", summary="How Rome moved water"),
                _candidate("Football transfer news"),
                _candidate("Modern cooking"),
            ]
        }
    )
    mode = NicheMode(
        source,
        queries=["ancient rome"],
        include_terms=["rome", "aqueduct"],
        exclude_terms=["football"],
        min_relevance=0.5,
    )

    response = _run(mode.discover(SourceRequest()))

    assert response.titles == ["Roman aqueducts"]
    assert response.metadata["skipped_relevance"] == 2


def test_include_terms_set_the_score_and_the_threshold() -> None:
    candidate = _candidate("Roman aqueducts", summary="Water in Rome")
    assert candidate_relevance(candidate, ["rome", "aqueduct"]) == 1.0
    assert candidate_relevance(candidate, ["rome", "engineering"]) == 0.5
    assert candidate_relevance(candidate, ["engineering"]) == 0.0
    assert is_relevant(candidate, ["rome", "engineering"], min_relevance=0.5)
    assert not is_relevant(candidate, ["rome", "engineering"], min_relevance=0.75)
    assert not is_relevant(candidate, ["rome"], exclude_terms=["aqueducts"])
    assert filter_candidates([candidate], ["engineering"]) == []


def test_an_injected_sync_scorer_replaces_the_term_gate() -> None:
    source = FakeSource(
        {
            "ancient rome": [
                _candidate("On theme"),
                _candidate("Off theme"),
            ]
        }
    )

    def scorer(candidate: Any) -> float:
        return 0.9 if candidate.title == "On theme" else 0.1

    mode = NicheMode(
        source,
        queries=["ancient rome"],
        relevance=scorer,
        min_relevance=0.5,
    )

    response = _run(mode.discover(SourceRequest()))

    assert response.titles == ["On theme"]
    assert response.metadata["skipped_relevance"] == 1


def test_an_injected_async_scorer_is_awaited() -> None:
    source = FakeSource({"q": [_candidate("Keep"), _candidate("Drop")]})

    async def scorer(candidate: Any) -> dict[str, Any]:
        return {"relevance": 1.0 if candidate.title == "Keep" else 0.0}

    mode = NicheMode(
        source,
        queries=["q"],
        relevance=scorer,
        min_relevance=0.5,
    )

    response = _run(mode.discover(SourceRequest()))

    assert response.titles == ["Keep"]


def test_a_missing_query_falls_back_to_the_request_query() -> None:
    source = FakeSource({"from request": [_candidate("A topic")]})
    mode = NicheMode(source)

    response = _run(mode.discover(SourceRequest(query="from request")))

    assert [item.query for item in source.requests] == ["from request"]
    assert response.titles == ["A topic"]
    assert response.query == "from request"


def test_a_failed_query_is_reported_when_errors_are_tolerated() -> None:
    source = FakeSource(
        {"good": [_candidate("Good topic")], "bad": []},
        fail={"bad"},
    )
    mode = NicheMode(
        source,
        queries=["good", "bad"],
        continue_on_error=True,
    )

    response = _run(mode.discover(SourceRequest()))

    assert response.titles == ["Good topic"]
    assert response.metadata["failed_queries"] == 1
    assert response.metadata["errors"][0]["query"] == "bad"


def test_a_failed_query_propagates_by_default() -> None:
    source = FakeSource({"bad": []}, fail={"bad"})
    mode = NicheMode(source, queries=["bad"])

    with pytest.raises(RuntimeError, match="boom: bad"):
        _run(mode.discover(SourceRequest()))


def test_the_gate_can_be_disabled_and_rejects_invalid_thresholds() -> None:
    source = FakeSource({"q": [_candidate("Anything")]})
    mode = NicheMode(source, queries=["q"], min_relevance=None)

    assert _run(mode.discover(SourceRequest())).titles == ["Anything"]

    with pytest.raises(ConfigError, match="between zero and one"):
        NicheMode(source, min_relevance=1.5)
    with pytest.raises(ConfigError, match="at least one"):
        NicheMode(source, concurrency=0)
    with pytest.raises(ConfigError, match="max queries"):
        NicheMode(source, max_queries=0)


def test_a_gate_instance_can_be_supplied_directly() -> None:
    gate = RelevanceGate(include_terms=["keep"], min_relevance=0.5)
    source = FakeSource({"q": [_candidate("Keep this"), _candidate("Drop that")]})
    mode = NicheMode(source, queries=["q"], gate=gate)

    response = _run(mode.discover(SourceRequest()))

    assert mode.gate is gate
    assert response.titles == ["Keep this"]
