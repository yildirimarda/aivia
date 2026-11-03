"""The dedupe stage: normalized titles, embedding similarity and history."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from aivia.core.errors import ConfigError
from aivia.core.models.topic import TopicCandidate
from aivia.providers.protocols.embeddings import (
    Embedding,
    EmbeddingRequest,
    EmbeddingResponse,
)
from aivia.providers.protocols.source import SourceResponse
from aivia.research import (
    Dedupe,
    DedupeDecision,
    DedupeHistory,
    Deduper,
    DedupeReason,
    DedupeResult,
    DedupeStage,
    Deduplicator,
    Duplicate,
    DuplicateReason,
    dedupe,
    dedupe_candidates,
    dedupe_candidates_async,
    dedupe_sync,
)

NOW = datetime(2026, 5, 5, 12, tzinfo=UTC)


def _candidate(title: str, **extra: Any) -> TopicCandidate:
    return TopicCandidate(
        title=title,
        source_urls=[f"https://example.test/{title.replace(' ', '_')}"],
        **extra,
    )


def test_normalized_titles_are_dropped_inside_the_batch_with_a_reason() -> None:
    candidates = [
        _candidate("Moon_landing"),
        _candidate("  moon   LANDING "),
        _candidate("Another topic"),
    ]

    result = dedupe_candidates(candidates, now=NOW, record=False)

    assert isinstance(result, DedupeResult)
    assert result.titles == ["Moon_landing", "Another topic"]
    assert result.duplicate_count == 1
    assert result.title_duplicates == 1
    assert result.embedding_duplicates == 0
    duplicate = result.duplicates[0]
    assert isinstance(duplicate, DedupeDecision)
    assert duplicate.reason is DedupeReason.TITLE
    assert duplicate.reason.value == "title"
    assert duplicate.matched_title == "Moon_landing"
    assert duplicate.explanation
    assert result.reasons == [DedupeReason.TITLE]
    assert result.metadata["skipped_duplicates"] == 1
    assert result.to_dict()["decisions"][1]["reason"] == "title"
    assert result.to_response().titles == ["Moon_landing", "Another topic"]


def test_embedding_similarity_drops_a_rephrased_topic() -> None:
    candidates = [
        _candidate("A mission to the moon"),
        _candidate("Lunar landing mission"),
        _candidate("A cooking recipe"),
    ]
    embeddings = {
        "A mission to the moon": [1.0, 0.0],
        "Lunar landing mission": [0.99, 0.05],
        "A cooking recipe": [0.0, 1.0],
    }

    result = dedupe_candidates(
        candidates,
        embeddings=embeddings,
        embedding_threshold=0.9,
        now=NOW,
        record=False,
    )

    assert result.titles == ["A mission to the moon", "A cooking recipe"]
    assert result.embedding_duplicates == 1
    duplicate = result.duplicates[0]
    assert duplicate.reason is DedupeReason.EMBEDDING
    assert duplicate.reason.value == "embedding"
    assert duplicate.matched_title == "A mission to the moon"
    assert duplicate.similarity is not None
    assert duplicate.similarity >= 0.9
    assert "similarity" in duplicate.explanation


def test_embedding_matching_can_be_disabled() -> None:
    candidates = [_candidate("One"), _candidate("Two")]
    embeddings = {"One": [1.0, 0.0], "Two": [1.0, 0.0]}

    result = dedupe_candidates(
        candidates,
        embeddings=embeddings,
        embedding_threshold=None,
        now=NOW,
        record=False,
    )

    assert result.titles == ["One", "Two"]
    assert result.duplicates == []


def test_history_dedupes_titles_across_profiles_inside_the_window() -> None:
    history = DedupeHistory(
        [
            {
                "title": "Moon_landing",
                "profile": "other-profile",
                "seen_at": NOW - timedelta(days=2),
            }
        ],
        now=NOW,
    )

    result = dedupe_candidates(
        [_candidate("moon landing"), _candidate("Fresh topic")],
        history=history,
        window=timedelta(days=7),
        profile="this-profile",
        now=NOW,
        record=False,
    )

    assert result.titles == ["Fresh topic"]
    duplicate = result.duplicates[0]
    assert duplicate.reason is DedupeReason.TITLE
    assert duplicate.matched_profile == "other-profile"
    assert duplicate.matched_seen_at == NOW - timedelta(days=2)
    assert duplicate.age == timedelta(days=2)


def test_history_outside_the_window_is_ignored() -> None:
    history = DedupeHistory(
        [{"title": "Old topic", "seen_at": NOW - timedelta(days=10)}],
        now=NOW,
    )

    result = dedupe_candidates(
        [_candidate("Old topic")],
        history=history,
        window=7,
        now=NOW,
        record=False,
    )

    assert result.titles == ["Old topic"]
    assert result.duplicates == []


def test_a_zero_window_disables_history_but_not_batch_dedupe() -> None:
    history = DedupeHistory(
        [{"title": "Old topic", "seen_at": NOW}],
        now=NOW,
    )

    result = dedupe_candidates(
        [_candidate("Old topic"), _candidate("old TOPIC")],
        history=history,
        window=0,
        now=NOW,
        record=False,
    )

    assert result.titles == ["Old topic"]
    assert result.title_duplicates == 1
    assert result.duplicates[0].matched_seen_at is None


def test_history_embeddings_match_across_profiles_and_return_the_score() -> None:
    history = DedupeHistory(
        [
            {
                "title": "Earlier angle",
                "profile": "other-profile",
                "seen_at": NOW - timedelta(hours=3),
                "embedding": [1.0, 0.0],
            }
        ],
        now=NOW,
    )

    result = dedupe_candidates(
        [_candidate("A new angle")],
        history=history,
        embeddings={"A new angle": [0.98, 0.1]},
        embedding_threshold=0.9,
        profile="this-profile",
        now=NOW,
        record=False,
    )

    duplicate = result.duplicates[0]
    assert duplicate.reason is DedupeReason.EMBEDDING
    assert duplicate.matched_title == "Earlier angle"
    assert duplicate.matched_profile == "other-profile"
    assert duplicate.similarity is not None
    assert duplicate.similarity >= 0.9
    assert duplicate.age == timedelta(hours=3)


def test_across_profiles_can_be_restricted_to_the_current_profile() -> None:
    history = DedupeHistory(
        [
            {
                "title": "Other title",
                "profile": "other-profile",
                "seen_at": NOW,
                "embedding": [1.0, 0.0],
            }
        ],
        now=NOW,
    )

    result = dedupe_candidates(
        [_candidate("A new angle")],
        history=history,
        embeddings={"A new angle": [1.0, 0.0]},
        profile="this-profile",
        across_profiles=False,
        now=NOW,
        record=False,
    )

    assert result.titles == ["A new angle"]
    assert result.duplicates == []


def test_title_matching_takes_precedence_over_embedding_matching() -> None:
    history = DedupeHistory(
        [
            {
                "title": "Same topic",
                "profile": "other",
                "seen_at": NOW,
                "embedding": [1.0, 0.0],
            }
        ],
        now=NOW,
    )

    result = dedupe_candidates(
        [_candidate("Same topic")],
        history=history,
        embeddings={"Same topic": [1.0, 0.0]},
        now=NOW,
        record=False,
    )

    assert result.duplicates[0].reason is DedupeReason.TITLE


def test_accepted_candidates_are_recorded_for_later_profiles() -> None:
    history = DedupeHistory(now=NOW)

    first = dedupe_candidates(
        [_candidate("Shared topic")],
        history=history,
        profile="profile-a",
        embeddings={"Shared topic": [1.0, 0.0]},
        now=NOW,
    )

    assert first.titles == ["Shared topic"]
    entry = history.entries[0]
    assert entry.title == "Shared topic"
    assert entry.profile == "profile-a"
    assert entry.embedding == (1.0, 0.0)
    assert entry.seen_at == NOW

    second = dedupe_candidates(
        [_candidate("Shared topic")],
        history=history,
        profile="profile-b",
        now=NOW,
    )

    assert second.titles == []
    assert second.duplicates[0].matched_profile == "profile-a"


def test_record_false_leaves_the_history_untouched() -> None:
    history = DedupeHistory(now=NOW)

    result = dedupe_candidates(
        [_candidate("Not recorded")],
        history=history,
        now=NOW,
        record=False,
    )

    assert result.titles == ["Not recorded"]
    assert history.entries == ()


def test_embeddings_can_come_from_a_callable_or_a_normalized_key() -> None:
    by_normalized_key = dedupe_candidates(
        [_candidate("One"), _candidate("Two")],
        embeddings={"one": [1.0, 0.0], "two": [1.0, 0.0]},
        now=NOW,
        record=False,
    )
    assert by_normalized_key.embedding_duplicates == 1

    def embed(candidate: TopicCandidate) -> list[float]:
        return [1.0, 0.0] if candidate.title != "Solo" else [0.0, 1.0]

    by_callable = dedupe_candidates(
        [_candidate("One"), _candidate("Solo")],
        embedder=embed,
        now=NOW,
        record=False,
    )
    assert by_callable.titles == ["One", "Solo"]


def test_source_responses_and_strings_are_accepted_as_candidates() -> None:
    response = SourceResponse(
        provider="fake",
        candidates=[_candidate("One"), _candidate("Two")],
    )

    result = dedupe_candidates(response, now=NOW, record=False)

    assert result.titles == ["One", "Two"]
    assert dedupe_candidates(["One", "one"], now=NOW, record=False).titles == ["One"]


def test_the_stage_reuses_its_window_threshold_and_history() -> None:
    stage = DedupeStage(
        window=7,
        embedding_threshold=0.9,
        profile="stage-profile",
        now=NOW,
    )

    assert isinstance(stage, Dedupe)
    assert isinstance(stage, Deduplicator)
    assert isinstance(stage, Deduper)
    assert stage.window == timedelta(days=7)
    assert stage.embedding_threshold == 0.9
    assert stage.profile == "stage-profile"

    first = stage.dedupe([_candidate("One")])
    second = stage.run([_candidate("One")])

    assert first.titles == ["One"]
    assert second.titles == []
    assert stage.history.entries[0].profile == "stage-profile"


def test_threshold_aliases_are_accepted() -> None:
    candidates = [_candidate("One"), _candidate("Two")]
    embeddings = {"One": [1.0, 0.0], "Two": [0.99, 0.01]}

    by_threshold = dedupe_candidates(
        candidates,
        embeddings=embeddings,
        threshold=0.9,
        now=NOW,
        record=False,
    )
    by_similarity = dedupe_candidates(
        candidates,
        embeddings=embeddings,
        similarity_threshold=0.9,
        now=NOW,
        record=False,
    )

    assert by_threshold.embedding_duplicates == 1
    assert by_similarity.embedding_duplicates == 1
    assert by_threshold.embedding_threshold == 0.9
    assert by_similarity.embedding_threshold == 0.9


def test_fuzzy_title_matching_is_optional() -> None:
    history = DedupeHistory(
        [{"title": "Breaking news: cat rescued", "seen_at": NOW}],
        now=NOW,
    )

    exact = dedupe_candidates(
        [_candidate("Cat rescued: breaking news")],
        history=history,
        now=NOW,
        record=False,
    )
    fuzzy = dedupe_candidates(
        [_candidate("Cat rescued: breaking news")],
        history=history,
        title_threshold=0.85,
        now=NOW,
        record=False,
    )

    assert exact.titles == ["Cat rescued: breaking news"]
    assert fuzzy.titles == []
    assert fuzzy.duplicates[0].reason is DedupeReason.TITLE


def test_async_embedding_provider_is_supported() -> None:
    class AsyncEmbedder:
        async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
            vectors = [
                [1.0, 0.0] if "moon" in text.casefold() else [0.0, 1.0]
                for text in request.inputs
            ]
            return EmbeddingResponse(
                provider="fake",
                embeddings=[
                    Embedding(index=index, vector=vector)
                    for index, vector in enumerate(vectors)
                ],
            )

    result = asyncio.run(
        dedupe_candidates_async(
            [_candidate("Moon mission"), _candidate("Moon landing")],
            embedder=AsyncEmbedder(),
            now=NOW,
            record=False,
        )
    )

    assert result.titles == ["Moon mission"]
    assert result.duplicates[0].reason is DedupeReason.EMBEDDING


def test_aliases_and_reason_members_share_one_identity() -> None:
    assert dedupe is dedupe_candidates
    assert dedupe_sync is dedupe_candidates
    assert Duplicate is DedupeDecision
    assert DuplicateReason is DedupeReason
    assert DedupeReason.NORMALIZED_TITLE.value == "title"
    assert DedupeReason.EMBEDDING_SIMILARITY.value == "embedding"
    assert DedupeReason.TITLE.value == "title"
    assert DedupeReason.EMBEDDING.value == "embedding"


def test_invalid_options_are_rejected() -> None:
    with pytest.raises(ConfigError):
        dedupe_candidates([], embedding_threshold=1.5)
    with pytest.raises(ConfigError):
        dedupe_candidates([], embedding_threshold=0.8, threshold=0.9)
    with pytest.raises(ConfigError):
        dedupe_candidates([], window=-1)
    with pytest.raises(ConfigError):
        dedupe_candidates([], across_profiles=1)  # type: ignore[arg-type]
    with pytest.raises(ConfigError):
        dedupe_candidates([], record="yes")  # type: ignore[arg-type]
    with pytest.raises(ConfigError):
        DedupeStage(window="soon")  # type: ignore[arg-type]

    async def embed(candidate: TopicCandidate) -> list[float]:
        return [1.0, 0.0]

    with pytest.raises(ConfigError, match="asynchronous"):
        dedupe_candidates([_candidate("One")], embedder=embed, now=NOW)
