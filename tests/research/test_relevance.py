"""Relevance gating against a brief, with labelled on and off theme topics."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from aivia.core.errors import ConfigError
from aivia.core.models.topic import TopicCandidate
from aivia.research import (
    BriefDerivation,
    RelevanceGate,
    build_relevance_gate,
    candidate_relevance,
    filter_candidates,
    is_relevant,
)

BRIEF = (
    "Practical guides to home renewable energy: solar panels, wind turbines, "
    "battery storage and the electrical grid."
)

INCLUDE_TERMS = ["solar", "wind", "battery", "grid"]
EXCLUDE_TERMS = ["football", "celebrity"]

ON_THEME = [
    ("Solar panel and battery costs", "What a home installation costs."),
    ("Wind turbine and grid connection costs", "Planning a small wind setup."),
    ("Home battery storage for solar owners", "Storing daytime solar."),
    ("How the grid handles wind power", "Balancing variable supply."),
    ("Solar and wind farm planning", "Choosing a site and capacity."),
    ("Grid-scale battery systems", "Storage for peak demand."),
    ("Choosing a solar inverter and battery", "Matching the parts."),
    ("Offshore wind grid links", "Connecting sea turbines."),
    ("Community solar and battery projects", "Shared local generation."),
    ("Smart grid and home battery integration", "Automating stored power."),
]

OFF_THEME = [
    ("Celebrity chef opens restaurant", "A new dining spot."),
    ("Football transfer rumours", "The latest signings."),
    ("Stock market outlook", "Analysts forecast the quarter."),
    ("New smartphone review", "Camera and screen tested."),
    ("Gardening tips for spring", "When to plant seedlings."),
    ("Space telescope discoveries", "A distant galaxy observed."),
    ("History of ancient Rome", "The republic and its emperors."),
    ("Jazz album recommendations", "Ten records worth hearing."),
    ("Cybersecurity best practices", "Passwords and updates."),
    ("Travel guide to Japan", "Cities, trains and food."),
]


def _derivation() -> BriefDerivation:
    return BriefDerivation(
        brief=BRIEF,
        queries=["home renewable energy", "solar panels"],
        subtopics=["battery storage", "electrical grid"],
        keywords=["renewable energy"],
        include_terms=INCLUDE_TERMS,
        exclude_terms=EXCLUDE_TERMS,
    )


def _candidate(title: str, summary: str = "") -> TopicCandidate:
    return TopicCandidate(
        title=title,
        summary=summary,
        source_urls=[f"https://example.test/{title.replace(' ', '_')}"],
    )


def _on_theme() -> list[TopicCandidate]:
    return [_candidate(title, summary) for title, summary in ON_THEME]


def _off_theme() -> list[TopicCandidate]:
    return [_candidate(title, summary) for title, summary in OFF_THEME]


def test_the_labelled_dataset_has_at_least_ten_examples_per_class() -> None:
    assert len(ON_THEME) >= 10
    assert len(OFF_THEME) >= 10
    assert len({title for title, _ in ON_THEME}) == len(ON_THEME)
    assert len({title for title, _ in OFF_THEME}) == len(OFF_THEME)


def test_a_brief_derivation_gates_every_on_theme_and_off_theme_candidate() -> None:
    gate = RelevanceGate.from_derivation(_derivation())
    on_theme = _on_theme()
    off_theme = _off_theme()

    on_scores = [gate.score(candidate) for candidate in on_theme]
    off_scores = [gate.score(candidate) for candidate in off_theme]

    assert gate.min_relevance is not None
    assert all(score >= gate.min_relevance for score in on_scores)
    assert all(score == 0.0 for score in off_scores)
    assert gate.filter([*on_theme, *off_theme]) == on_theme
    assert [candidate.title for candidate in gate.filter(off_theme)] == []
    assert [gate.is_relevant(candidate) for candidate in on_theme] == [True] * 10
    assert [gate.is_relevant(candidate) for candidate in off_theme] == [False] * 10


def test_the_derivation_supplies_queries_subtopics_keywords_and_terms() -> None:
    gate = RelevanceGate.from_derivation(_derivation())

    assert gate.queries == ("home renewable energy", "solar panels")
    assert gate.subtopics == ("battery storage", "electrical grid")
    assert gate.keywords == ("renewable energy",)
    assert gate.include_terms == tuple(INCLUDE_TERMS)
    assert gate.exclude_terms == tuple(EXCLUDE_TERMS)
    assert gate.required_terms == tuple(INCLUDE_TERMS)
    assert gate.min_relevance == 0.5


def test_explicit_arguments_override_the_derived_terms() -> None:
    gate = RelevanceGate.from_derivation(
        _derivation(),
        include_terms=["battery"],
        exclude_terms=["solar"],
        queries=["custom query"],
        niche="home storage",
    )

    assert gate.include_terms == ("battery",)
    assert gate.exclude_terms == ("solar",)
    assert gate.queries == ("custom query",)
    assert gate.subtopics == ("battery storage", "electrical grid")
    assert gate.keywords == ("renewable energy",)
    assert gate.niche == "home storage"
    assert gate.is_relevant(_candidate("Home battery guide"))
    assert not gate.is_relevant(_candidate("Solar battery guide"))


def test_an_empty_explicit_value_clears_the_derived_field() -> None:
    gate = RelevanceGate.from_derivation(
        _derivation(),
        include_terms=[],
        exclude_terms=[],
    )

    assert gate.include_terms == ()
    assert gate.exclude_terms == ()


def test_an_excluded_term_drops_an_otherwise_on_theme_candidate() -> None:
    gate = RelevanceGate.from_derivation(_derivation())
    candidate = _candidate("Solar football stadium", "A stadium with solar panels.")

    assert gate.score(candidate) == 0.0
    assert not gate.is_relevant(candidate)


def test_the_module_helpers_accept_the_brief_derivation() -> None:
    derivation = _derivation()
    on_theme = _on_theme()
    off_theme = _off_theme()

    assert candidate_relevance(on_theme[0], derivation=derivation) == 0.5
    assert candidate_relevance(off_theme[0], derivation=derivation) == 0.0
    assert is_relevant(on_theme[0], derivation=derivation)
    assert not is_relevant(off_theme[0], derivation=derivation)
    assert filter_candidates([*on_theme, *off_theme], derivation=derivation) == on_theme


def test_the_gate_builder_accepts_both_derivation_aliases() -> None:
    derivation = _derivation()

    assert build_relevance_gate(derivation).include_terms == tuple(INCLUDE_TERMS)
    assert build_relevance_gate(brief_derivation=derivation).include_terms == tuple(
        INCLUDE_TERMS
    )

    with pytest.raises(ConfigError, match="either derivation or brief_derivation"):
        build_relevance_gate(derivation, brief_derivation=derivation)


def test_the_threshold_can_be_configured_or_disabled() -> None:
    candidate = _candidate("Solar panel costs")
    derivation = _derivation()

    assert candidate_relevance(candidate, derivation=derivation) == 0.25
    assert not is_relevant(candidate, derivation=derivation)
    assert is_relevant(candidate, derivation=derivation, min_relevance=0.2)
    assert is_relevant(candidate, derivation=derivation, min_relevance=None)

    with pytest.raises(ConfigError, match="between zero and one"):
        RelevanceGate.from_derivation(derivation, min_relevance=1.5)


def test_an_injected_scorer_can_replace_the_term_gate() -> None:
    derivation = _derivation()

    def scorer(candidate: TopicCandidate) -> float:
        return 0.9 if "solar" in candidate.title.casefold() else 0.1

    gate = RelevanceGate.from_derivation(derivation, relevance=scorer)

    assert gate.score(_candidate("Solar panel costs")) == 0.9
    assert gate.is_relevant(_candidate("Solar panel costs"))
    assert not gate.is_relevant(_candidate("Wind turbine costs"))


def test_an_async_scorer_is_awaited_by_the_async_score() -> None:
    async def scorer(candidate: TopicCandidate) -> dict[str, Any]:
        return {"relevance": 1.0 if candidate.title == "Keep" else 0.0}

    gate = RelevanceGate.from_derivation(_derivation(), relevance=scorer)

    assert asyncio.run(gate.score_async(_candidate("Keep"))) == 1.0
    assert asyncio.run(gate.score_async(_candidate("Drop"))) == 0.0
