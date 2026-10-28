"""Topic, candidate, claim and fact list contract."""

from __future__ import annotations

import json
import warnings
from typing import Any

import pytest
from pydantic import ValidationError

from aivia.core.models import (
    Candidate,
    Claim,
    Fact,
    FactBase,
    FactList,
    FactListContract,
    FactListModel,
    Topic,
    TopicCandidate,
    UnknownFieldWarning,
)

_SOURCE = "https://example.com/story"
_OTHER = "https://other.example.org/background"


def _claim(text: str = "A verified statement") -> Claim:
    return Claim(text=text, source_urls=[_SOURCE])


def test_claim_carries_text_sources_confidence_and_verification() -> None:
    claim = Claim(
        text="  The bridge opened in 1937  ",
        source_urls=[_SOURCE, _OTHER],
        confidence=0.75,
        verified=True,
    )

    assert claim.text == "The bridge opened in 1937"
    assert claim.source_urls == [_SOURCE, _OTHER]
    assert claim.confidence == 0.75
    assert claim.verified is True
    assert claim.sources == claim.source_urls
    assert claim.urls == claim.source_urls
    assert claim.statement == claim.text


def test_claim_requires_at_least_one_source_url() -> None:
    with pytest.raises(ValidationError):
        Claim.model_validate({"text": "Unsupported claim"})

    with pytest.raises(ValidationError):
        Claim(text="Unsupported claim", source_urls=[])


def test_claim_requires_non_empty_text() -> None:
    with pytest.raises(ValidationError):
        Claim(text="   ", source_urls=[_SOURCE])


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_claim_confidence_is_bounded(confidence: float) -> None:
    with pytest.raises(ValidationError):
        Claim(text="x", source_urls=[_SOURCE], confidence=confidence)


def test_claim_defaults_to_fully_confident_and_unverified() -> None:
    claim = _claim()

    assert claim.confidence == 1.0
    assert claim.verified is False


def test_source_urls_are_stripped_deduplicated_and_domains_extracted() -> None:
    claim = Claim(
        text="x",
        source_urls=[f"  {_SOURCE}  ", _SOURCE, _OTHER],
    )

    assert claim.source_urls == [_SOURCE, _OTHER]
    assert claim.source_domains == ["example.com", "other.example.org"]


def test_source_urls_accept_a_single_string_and_a_mapping() -> None:
    from_string = Claim.model_validate({"text": "x", "source_urls": _SOURCE})
    from_mapping = Claim.model_validate({"text": "x", "source_urls": {"url": _OTHER}})

    assert from_string.source_urls == [_SOURCE]
    assert from_mapping.source_urls == [_OTHER]


@pytest.mark.parametrize(
    "url",
    [
        "",
        "   ",
        "example.com/story",
        "/relative/path",
        "ftp://example.com/file",
        "javascript:alert(1)",
        "https://",
    ],
)
def test_source_urls_must_be_absolute_http_or_https(url: str) -> None:
    with pytest.raises(ValidationError):
        Claim(text="x", source_urls=[url])


def test_source_urls_reject_non_string_entries() -> None:
    with pytest.raises(ValidationError):
        Claim(text="x", source_urls=[123])  # type: ignore[list-item]


def test_candidate_carries_sources_mode_and_scoring_fields() -> None:
    candidate = TopicCandidate(
        title="  A discovered topic  ",
        summary="Why it matters",
        source_urls=[_SOURCE],
        mode="trending",
        score=42.5,
        reason="matches the brief",
        category="science",
        keywords=["space"],
        signal=0.8,
        rank=2,
    )

    assert candidate.title == "A discovered topic"
    assert candidate.source_urls == [_SOURCE]
    assert candidate.mode == "trending"
    assert candidate.topic_mode == "trending"
    assert candidate.score == 42.5
    assert candidate.reason == "matches the brief"
    assert candidate.category == "science"
    assert candidate.keywords == ["space"]
    assert candidate.signal == 0.8
    assert candidate.rank == 2
    assert candidate.source_domains == ["example.com"]


def test_candidate_defaults_allow_unscored_and_unsourced_discoveries() -> None:
    candidate = TopicCandidate(title="Idea")

    assert candidate.source_urls == []
    assert candidate.mode == ""
    assert candidate.score == 0.0
    assert candidate.reason == ""
    assert candidate.category == ""
    assert candidate.keywords == []
    assert candidate.signal is None
    assert candidate.rank is None


def test_candidate_requires_a_non_empty_title() -> None:
    with pytest.raises(ValidationError):
        TopicCandidate(title="  ")


def test_candidate_rejects_negative_score_and_rank() -> None:
    with pytest.raises(ValidationError):
        TopicCandidate(title="x", score=-1)

    with pytest.raises(ValidationError):
        TopicCandidate(title="x", rank=-1)


def test_candidate_to_topic_keeps_provenance_and_claims() -> None:
    candidate = TopicCandidate(
        title="Topic",
        summary="Candidate summary",
        source_urls=[_SOURCE],
        mode="niche",
        category="history",
        keywords=["bridge"],
    )
    claim = _claim()

    topic = candidate.to_topic(claims=[claim])

    assert isinstance(topic, Topic)
    assert topic.title == "Topic"
    assert topic.summary == "Candidate summary"
    assert topic.claims == [claim]
    assert topic.source_urls == [_SOURCE]
    assert topic.mode == "niche"
    assert topic.category == "history"
    assert topic.keywords == ["bridge"]
    assert topic.candidate == candidate


def test_topic_from_candidate_matches_to_topic() -> None:
    candidate = Candidate(title="Topic", source_urls=[_SOURCE], mode="manual")
    claim = _claim()

    from_candidate = Topic.from_candidate(candidate, claims=[claim])
    to_topic = candidate.to_topic(claims=[claim])

    assert from_candidate.model_dump(exclude={"created_at", "updated_at"}) == (
        to_topic.model_dump(exclude={"created_at", "updated_at"})
    )


def test_topic_carries_claims_and_derives_sources_from_them() -> None:
    first = Claim(text="one", source_urls=[_SOURCE])
    second = Claim(text="two", source_urls=[_OTHER, _SOURCE])

    topic = Topic(title="Topic", claims=[first, second])

    assert topic.claims == [first, second]
    assert topic.facts == topic.claims
    assert topic.source_urls == [_SOURCE, _OTHER]
    assert topic.source_domains == ["example.com", "other.example.org"]


def test_topic_keeps_explicit_source_urls_instead_of_deriving_them() -> None:
    topic = Topic(
        title="Topic",
        claims=[_claim()],
        source_urls=[_OTHER],
    )

    assert topic.source_urls == [_OTHER]


def test_topic_allows_no_claims_for_user_supplied_input() -> None:
    topic = Topic(title="Manual topic")

    assert topic.claims == []
    assert topic.source_urls == []


def test_topic_requires_a_non_empty_title() -> None:
    with pytest.raises(ValidationError):
        Topic(title=" ")


def test_models_serialize_canonically_with_sources_and_claims() -> None:
    topic = Topic(
        title="Topic",
        claims=[_claim()],
        source_urls=[_SOURCE],
        mode="manual",
    )

    payload = json.loads(topic.to_json())

    assert payload["source_urls"] == [_SOURCE]
    assert payload["claims"][0]["source_urls"] == [_SOURCE]
    assert list(payload) == sorted(payload)


def test_models_round_trip_through_json() -> None:
    topic = Topic(title="Topic", claims=[_claim()], source_urls=[_SOURCE])

    restored = Topic.model_validate_json(topic.to_json())

    assert restored == topic


def test_claim_statement_and_sources_aliases_do_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnknownFieldWarning)
        claim = Claim.model_validate({"statement": "Aliased", "sources": [_SOURCE]})

    assert claim.text == "Aliased"
    assert claim.source_urls == [_SOURCE]


def test_topic_facts_alias_and_candidate_sources_alias_do_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnknownFieldWarning)
        claim = _claim()
        topic = Topic.model_validate({"name": "Aliased topic", "facts": [claim]})
        candidate = TopicCandidate.model_validate(
            {"name": "Aliased candidate", "sources": [_OTHER]}
        )

    assert topic.title == "Aliased topic"
    assert topic.claims == [claim]
    assert candidate.title == "Aliased candidate"
    assert candidate.source_urls == [_OTHER]


class _UserFactList:
    """A research-shaped object that never imports the core models module."""

    def __init__(self, claims: list[Claim]) -> None:
        self.claims = claims


def test_topic_satisfies_the_fact_list_contract() -> None:
    topic = Topic(title="Topic", claims=[_claim()])

    assert isinstance(topic, FactList)


def test_a_plain_object_with_claims_satisfies_the_fact_list_contract() -> None:
    claim = _claim()
    fact_list = _UserFactList([claim])

    assert isinstance(fact_list, FactList)
    assert list(fact_list.claims) == [claim]


def test_objects_without_claims_are_not_fact_lists() -> None:
    assert not isinstance(object(), FactList)
    assert not isinstance(TopicCandidate(title="candidate"), FactList)


def test_fact_list_aliases_point_at_the_contract() -> None:
    assert FactListContract is FactList
    assert FactBase is FactList
    assert Fact is Claim
    assert Candidate is TopicCandidate


def test_fact_list_model_collects_claims_and_source_urls() -> None:
    first = Claim(text="one", source_urls=[_SOURCE])
    second = Claim(text="two", source_urls=[_OTHER, _SOURCE])
    topic = Topic(title="Topic", claims=[first])

    fact_list = FactListModel(topic=topic, claims=[first, second])

    assert fact_list.facts == [first, second]
    assert fact_list.claims == [first, second]
    assert fact_list.source_urls == [_SOURCE, _OTHER]
    assert isinstance(fact_list, FactList)


def test_fact_list_model_serializes_canonically() -> None:
    fact_list = FactListModel(claims=[_claim()])

    payload: dict[str, Any] = json.loads(fact_list.to_json())

    assert payload["claims"][0]["source_urls"] == [_SOURCE]
    assert list(payload) == sorted(payload)
