"""Verify topics against at least two independent source domains."""

from __future__ import annotations

from typing import Any

import pytest

from aivia.core.errors import ConfigError
from aivia.core.models.topic import Claim, Topic, TopicCandidate
from aivia.providers.protocols.source import SourceResponse
from aivia.research import (
    DEFAULT_MIN_DOMAINS,
    TopicVerifier,
    VerificationDecision,
    VerificationResult,
    VerificationStage,
    VerificationStatus,
    Verifier,
    VerifyStage,
    build_verification_stage,
    domain_of,
    independent_domains,
    is_verified,
    mark_verified,
    registrable_domain,
    verification,
    verification_stage,
    verify,
    verify_candidate,
    verify_candidates,
    verify_topics,
)

_FIRST = "https://example.com/story"
_SECOND = "https://other.test/background"


def _candidate(title: str, *urls: str) -> TopicCandidate:
    return TopicCandidate(title=title, source_urls=list(urls))


def test_two_independent_domains_mark_a_topic_verified() -> None:
    candidate = _candidate("Bridge history", _FIRST, _SECOND)

    result = verify_candidates([candidate])

    assert isinstance(result, VerificationResult)
    assert result.verified_titles == ["Bridge history"]
    assert result.unverified_titles == []
    assert result.verified_count == 1
    assert result.unverified_count == 0
    assert result.candidates == [candidate]
    assert candidate.verified is True
    decision = result.verified_decisions[0]
    assert isinstance(decision, VerificationDecision)
    assert decision.verified is True
    assert decision.unverified is False
    assert decision.status is VerificationStatus.VERIFIED
    assert decision.domains == ("example.com", "other.test")
    assert decision.source_count == 2
    assert decision.reason == "verified across 2 independent domains"


def test_one_domain_marks_a_topic_unverified_and_records_the_reason() -> None:
    candidate = _candidate("Bridge history", _FIRST)

    decision = verify_candidate(candidate)

    assert decision.verified is False
    assert decision.unverified is True
    assert decision.single_source is True
    assert decision.status is VerificationStatus.UNVERIFIED
    assert decision.domains == ("example.com",)
    assert decision.reason == "single source domain: example.com"
    assert candidate.verified is False


def test_urls_on_the_same_domain_do_not_count_as_two_sources() -> None:
    candidate = _candidate(
        "Bridge history",
        "https://www.example.com/a",
        "https://example.com/b",
        "https://news.example.com/c",
    )

    decision = verify_candidate(candidate)

    assert decision.verified is False
    assert decision.single_source is True
    assert decision.domains == ("example.com",)
    assert decision.source_count == 3


def test_subdomains_and_language_editions_collapse_to_one_domain() -> None:
    candidate = _candidate(
        "Bridge history",
        "https://en.wikipedia.org/wiki/Bridge",
        "https://de.wikipedia.org/wiki/Bruecke",
    )

    decision = verify_candidate(candidate)

    assert decision.domains == ("wikipedia.org",)
    assert decision.verified is False


def test_a_topic_without_sources_is_unverified() -> None:
    candidate = _candidate("An unsourced idea")

    decision = verify_candidate(candidate)

    assert decision.domains == ()
    assert decision.source_count == 0
    assert decision.single_source is False
    assert decision.verified is False
    assert decision.reason == "no source urls to verify"


def test_a_topic_model_derives_its_domains_from_its_claims() -> None:
    topic = Topic(
        title="Bridge history",
        claims=[
            Claim(text="It opened in 1937", source_urls=[_FIRST]),
            Claim(text="It was restored later", source_urls=[_SECOND]),
        ],
    )

    decision = verify_candidate(topic)

    assert decision.verified is True
    assert topic.verified is True
    assert independent_domains(topic) == ["example.com", "other.test"]


def test_verification_is_propagated_to_a_topic_built_from_a_candidate() -> None:
    candidate = _candidate("Bridge history", _FIRST, _SECOND)
    verify_candidate(candidate)

    topic = candidate.to_topic()

    assert topic.verified is True
    assert Topic.from_candidate(candidate).verified is True


def test_mark_verified_updates_the_candidate_and_is_verified_answers() -> None:
    candidate = _candidate("Bridge history", _FIRST, _SECOND)

    assert is_verified(candidate) is True
    assert candidate.verified is False

    returned = mark_verified(candidate)

    assert returned is candidate
    assert candidate.verified is True


def test_a_mixed_batch_keeps_every_topic_and_separates_the_subsets() -> None:
    verified = _candidate("Verified story", _FIRST, _SECOND)
    single = _candidate("Single source story", _FIRST)
    unsourced = _candidate("Unsourced story")

    result = verify_candidates([verified, single, unsourced])

    assert result.titles == [
        "Verified story",
        "Single source story",
        "Unsourced story",
    ]
    assert result.verified_titles == ["Verified story"]
    assert result.unverified_titles == ["Single source story", "Unsourced story"]
    assert result.single_source == [single]
    assert result.count == 3
    assert result.input_count == 3
    assert result.single_source_count == 1
    assert result.filter() == [verified]
    assert result.filter(verified_only=False) == [verified, single, unsourced]
    assert result.metadata == {
        "min_domains": DEFAULT_MIN_DOMAINS,
        "input_count": 3,
        "verified": 1,
        "unverified": 2,
        "single_source": 1,
    }


def test_the_stage_accepts_a_source_response_and_returns_a_response() -> None:
    candidate = _candidate("Bridge history", _FIRST, _SECOND)
    response = SourceResponse(provider="mock", candidates=[candidate])

    stage = VerificationStage()
    result = stage.run(response)

    assert result.candidates == [candidate]
    returned = result.to_response()
    assert isinstance(returned, SourceResponse)
    assert returned.candidates == [candidate]
    assert returned.metadata["verified"] == 1
    filtered = result.to_response(verified_only=True)
    assert filtered.candidates == [candidate]


def test_the_stage_can_require_more_than_two_domains() -> None:
    candidate = _candidate("Bridge history", _FIRST, _SECOND)

    decision = verify_candidate(candidate, min_domains=3)

    assert decision.verified is False
    assert decision.reason == "only 2 independent domains; need 3"
    assert candidate.verified is False

    third = _candidate(
        "Bridge history",
        _FIRST,
        _SECOND,
        "https://third.example.net/story",
    )
    assert verify_candidate(third, min_domains=3).verified is True


@pytest.mark.parametrize("value", [0, 1, -1, True, 1.5, "2"])
def test_the_domain_floor_must_be_an_integer_of_at_least_two(value: Any) -> None:
    with pytest.raises(ConfigError):
        verify_candidate(_candidate("Bridge history"), min_domains=value)


def test_stage_aliases_and_builder_share_one_implementation() -> None:
    candidate = _candidate("Bridge history", _FIRST, _SECOND)

    for stage in (
        VerificationStage(),
        Verifier(),
        TopicVerifier(),
        VerifyStage(),
        verification_stage(),
        build_verification_stage(),
    ):
        assert isinstance(stage, VerificationStage)
        assert stage.min_domains == 2
        assert stage.name == "verify"
        assert stage.verify([candidate]).verified == [candidate]
        assert stage.filter([candidate]) == [candidate]
        assert stage.to_response([candidate]).candidates == [candidate]


def test_module_level_aliases_verify_the_same_candidates() -> None:
    candidate = _candidate("Bridge history", _FIRST, _SECOND)

    for function in (verify_candidates, verify_topics, verify, verification):
        assert function([candidate]).verified_titles == ["Bridge history"]


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("en.wikipedia.org", "wikipedia.org"),
        ("www.example.com", "example.com"),
        ("news.example.co.uk", "example.co.uk"),
        ("example.com", "example.com"),
        ("localhost", "localhost"),
        ("127.0.0.1", "127.0.0.1"),
        ("2001:db8::1", "2001:db8::1"),
        ("", ""),
    ],
)
def test_registrable_domain_handles_subdomains_suffixes_and_ips(
    host: str, expected: str
) -> None:
    assert registrable_domain(host) == expected
    assert domain_of(host) == expected


def test_a_mapping_and_a_plain_title_are_accepted() -> None:
    from_mapping = verify_candidate(
        {"title": "Bridge history", "source_urls": [_FIRST, _SECOND]}
    )
    from_title = verify_candidate("An unsourced idea")

    assert from_mapping.verified is True
    assert from_title.verified is False
    assert from_title.title == "An unsourced idea"


def test_to_dict_is_json_ready_and_names_the_domains_and_reason() -> None:
    result = verify_candidates([_candidate("Bridge history", _FIRST)])

    payload = result.to_dict()

    assert payload["verified"] == []
    assert payload["unverified"] == ["Bridge history"]
    decision = payload["decisions"][0]
    assert decision["title"] == "Bridge history"
    assert decision["verified"] is False
    assert decision["domains"] == ["example.com"]
    assert decision["reason"] == "single source domain: example.com"
    assert decision["candidate"]["verified"] is False
