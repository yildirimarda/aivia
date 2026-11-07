"""The text similarity gate regenerates or rewrites near copies of a reference."""

from __future__ import annotations

from typing import Any

import pytest

from aivia.core.errors import ValidationError
from aivia.core.similarity import NgramUnit
from aivia.reference import (
    TextGateError,
    TextGateResult,
    gate_text_similarity,
    ngram_similarity,
)


def _run(coro: Any) -> Any:
    import asyncio

    return asyncio.run(coro)


def test_ngram_similarity_identical_is_one() -> None:
    assert ngram_similarity("the quick brown fox", "the quick brown fox") == 1.0


def test_ngram_similarity_different_is_lower() -> None:
    score = ngram_similarity("the quick brown fox", "jumps over the lazy dog")
    assert 0.0 <= score < 1.0


def test_ngram_similarity_empty_texts() -> None:
    assert ngram_similarity("", "") == 1.0
    assert ngram_similarity("the quick brown fox", "") == 0.0
    assert ngram_similarity("", "the quick brown fox") == 0.0


def test_ngram_similarity_unit_character() -> None:
    word_score = ngram_similarity("abc def", "abc def", unit=NgramUnit.WORD)
    char_score = ngram_similarity("abc def", "abc def", unit=NgramUnit.CHARACTER)
    assert word_score == 1.0
    assert char_score == 1.0


def test_ngram_similarity_rejects_bad_n() -> None:
    with pytest.raises((TypeError, ValueError)):
        ngram_similarity("a b c", "a b c", n=0)
    with pytest.raises((TypeError, ValueError)):
        ngram_similarity("a b c", "a b c", n=True)


def test_ngram_similarity_rejects_bad_unit() -> None:
    with pytest.raises(ValueError):
        ngram_similarity("a b c", "a b c", unit="byte")


def test_gate_passes_below_threshold() -> None:
    result = _run(
        gate_text_similarity(
            reference="the quick brown fox jumps over the lazy dog",
            candidate="a completely different sentence about cats",
            threshold=0.5,
        )
    )
    assert isinstance(result, TextGateResult)
    assert result.passed is True
    assert result.refused is False
    assert result.regenerated == 0
    assert result.reason is None
    assert result.score < 0.5


def test_gate_refuses_at_threshold_without_regenerator() -> None:
    identical = "the quick brown fox jumps over the lazy dog"
    result = _run(
        gate_text_similarity(
            reference=identical,
            candidate=identical,
            threshold=0.5,
        )
    )
    assert result.passed is False
    assert result.refused is True
    assert result.regenerated == 0
    assert result.score >= 0.5
    assert result.reason is not None


def test_gate_regenerates_then_passes() -> None:
    reference = "the quick brown fox jumps over the lazy dog"
    copies = [
        reference,
        "a completely different sentence about sunny days and cats",
    ]
    index = 0

    async def regenerate() -> Any:
        nonlocal index
        text = copies[index]
        index += 1
        return text

    result = _run(
        gate_text_similarity(
            reference=reference,
            candidate=reference,
            threshold=0.5,
            max_regenerations=2,
            regenerate=regenerate,
        )
    )
    assert result.passed is True
    assert result.refused is False
    assert result.regenerated == 2
    assert result.score < 0.5


def test_gate_refuses_after_exhausting_regenerations() -> None:
    reference = "the quick brown fox jumps over the lazy dog"

    async def regenerate() -> Any:
        return reference

    result = _run(
        gate_text_similarity(
            reference=reference,
            candidate=reference,
            threshold=0.5,
            max_regenerations=2,
            regenerate=regenerate,
        )
    )
    assert result.passed is False
    assert result.refused is True
    assert result.regenerated == 2
    assert result.score >= 0.5
    assert "after 2 regeneration(s)" in result.reason


def test_gate_refuses_immediately_when_no_regenerator() -> None:
    reference = "the quick brown fox jumps over the lazy dog"
    result = _run(
        gate_text_similarity(
            reference=reference,
            candidate=reference,
            threshold=0.5,
            max_regenerations=3,
        )
    )
    assert result.passed is False
    assert result.refused is True
    assert result.regenerated == 0
    assert "no regenerator is configured" in result.reason


def test_gate_zero_regenerations_refuses_above_limit() -> None:
    reference = "the quick brown fox jumps over the lazy dog"

    async def _rewrite() -> str:
        return "a completely different sentence"

    result = _run(
        gate_text_similarity(
            reference=reference,
            candidate=reference,
            threshold=0.5,
            max_regenerations=0,
            regenerate=_rewrite,
        )
    )
    assert result.passed is False
    assert result.refused is True
    assert result.regenerated == 0


def test_gate_rejects_bad_threshold() -> None:
    with pytest.raises((TypeError, ValueError)):
        _run(
            gate_text_similarity(
                reference="the quick brown fox",
                candidate="jumps over the lazy dog",
                threshold=1.5,
            )
        )
    with pytest.raises((TypeError, ValueError)):
        _run(
            gate_text_similarity(
                reference="the quick brown fox",
                candidate="jumps over the lazy dog",
                threshold=-0.1,
            )
        )


def test_gate_rejects_bad_max_regenerations() -> None:
    with pytest.raises((TypeError, ValueError)):
        _run(
            gate_text_similarity(
                reference="the quick brown fox",
                candidate="jumps over the lazy dog",
                threshold=0.5,
                max_regenerations=-1,
            )
        )


def test_gate_rejects_bad_n() -> None:
    with pytest.raises((TypeError, ValueError)):
        _run(
            gate_text_similarity(
                reference="the quick brown fox",
                candidate="jumps over the lazy dog",
                threshold=0.5,
                n=0,
            )
        )


def test_text_gate_error_is_a_validation_error() -> None:
    error = TextGateError("too similar")
    assert isinstance(error, ValidationError)
    assert error.code == "text_similarity"
    assert error.retryable is False
    assert str(error) == "too similar"


def test_text_gate_result_is_immutable() -> None:
    result = TextGateResult(
        passed=True,
        score=0.1,
        method="word_ngram",
        regenerated=0,
        refused=False,
        reason=None,
    )
    with pytest.raises(AttributeError):
        result.score = 0.9  # type: ignore[misc]
