"""Cosine, title and n-gram similarity with their documented thresholds."""

from __future__ import annotations

import math

import pytest

from aivia.core.similarity import (
    DEFAULT_COSINE_THRESHOLD,
    DEFAULT_NGRAM_THRESHOLD,
    DEFAULT_TITLE_THRESHOLD,
    DEFAULT_VISUAL_COSINE_THRESHOLD,
    NgramUnit,
    character_ngrams,
    cosine_similarity,
    ngram_containment,
    ngram_overlap,
    normalize_title,
    title_similarity,
    titles_match,
    word_ngrams,
)


def test_documented_thresholds_are_probabilities_in_a_sane_order() -> None:
    assert DEFAULT_COSINE_THRESHOLD == 0.85
    assert DEFAULT_VISUAL_COSINE_THRESHOLD == 0.90
    assert DEFAULT_TITLE_THRESHOLD == 0.85
    assert DEFAULT_NGRAM_THRESHOLD == 0.60
    for threshold in (
        DEFAULT_COSINE_THRESHOLD,
        DEFAULT_VISUAL_COSINE_THRESHOLD,
        DEFAULT_TITLE_THRESHOLD,
        DEFAULT_NGRAM_THRESHOLD,
    ):
        assert 0.0 <= threshold <= 1.0
    assert DEFAULT_VISUAL_COSINE_THRESHOLD > DEFAULT_COSINE_THRESHOLD


def test_cosine_is_scale_invariant_and_handles_orthogonal_and_opposite() -> None:
    assert cosine_similarity([1.0, 2.0], [2.0, 4.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_treats_a_zero_vector_as_unrelated() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0


def test_cosine_rejects_mismatched_or_invalid_vectors() -> None:
    with pytest.raises(ValueError, match="same length"):
        cosine_similarity([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="must not be empty"):
        cosine_similarity([], [])
    with pytest.raises(ValueError, match="finite"):
        cosine_similarity([math.inf, 0.0], [1.0, 0.0])
    with pytest.raises(TypeError, match="sequences of numbers"):
        cosine_similarity("abc", [1.0, 2.0, 3.0])  # type: ignore[arg-type]


def test_normalize_title_folds_case_accents_punctuation_and_space() -> None:
    assert normalize_title("  Café: The  Story! ") == "cafe the story"
    assert normalize_title("DON'T stop") == "don t stop"
    assert normalize_title("Straße") == "strasse"
    assert normalize_title("...") == ""


def test_title_similarity_ignores_word_order_and_small_changes() -> None:
    assert (
        title_similarity("Cat rescued: breaking news", "Breaking news, cat rescued")
        == 1.0
    )
    assert title_similarity("Cats", "cat") > 0.8
    assert 0.0 < title_similarity("Cat rescued", "Dog rescued") < 1.0
    assert title_similarity("cat", "dog") == 0.0


def test_title_similarity_handles_empty_titles() -> None:
    assert title_similarity("", "") == 1.0
    assert title_similarity("cat", "") == 0.0
    assert title_similarity("", "cat") == 0.0


def test_titles_match_uses_the_threshold() -> None:
    assert titles_match("Breaking news: cat rescued", "Breaking news, cat rescued")
    assert not titles_match("cat", "dog")
    assert titles_match("cats", "cat", threshold=0.8)
    with pytest.raises(ValueError, match="between 0 and 1"):
        titles_match("cat", "cat", threshold=1.5)
    with pytest.raises(TypeError, match="threshold must be a number"):
        titles_match("cat", "cat", threshold=True)


def test_word_ngrams_fold_text_and_keep_short_input() -> None:
    assert word_ngrams("The quick, brown fox", n=2) == [
        ("the", "quick"),
        ("quick", "brown"),
        ("brown", "fox"),
    ]
    assert word_ngrams("Cat", n=3) == [("cat",)]
    assert word_ngrams("...", n=3) == []


def test_character_ngrams_fold_text_and_keep_short_input() -> None:
    assert character_ngrams("ABC def", n=3)[:3] == ["abc", "bc ", "c d"]
    assert character_ngrams("ab", n=3) == ["ab"]
    assert character_ngrams("...", n=3) == []


def test_ngram_overlap_is_jaccard_over_shared_shingles() -> None:
    left = "the quick brown fox jumps"
    right = "the quick brown fox leaps"
    assert ngram_overlap(left, right, n=2) == pytest.approx(3 / 5)
    assert ngram_overlap(left, left, n=2) == 1.0
    assert ngram_overlap(left, "unrelated words only", n=2) == 0.0
    assert ngram_overlap("", "") == 1.0
    assert ngram_overlap("cat", "") == 0.0


def test_ngram_overlap_supports_character_shingles() -> None:
    assert ngram_overlap("abcdef", "abcxyz", n=3, unit=NgramUnit.CHARACTER) == (
        pytest.approx(1 / 7)
    )
    assert ngram_overlap("abcdef", "abcdef", n=3, unit="character") == 1.0


def test_ngram_containment_ignores_extra_reference_text() -> None:
    left = "the quick brown fox jumps"
    right = "the quick brown fox leaps"
    assert ngram_containment(left, right, n=2) == pytest.approx(3 / 4)
    assert ngram_containment(left, f"{left} with much more unrelated text", n=2) == 1.0
    assert ngram_containment("", "") == 1.0
    assert ngram_containment("cat", "") == 0.0


def test_ngram_helpers_validate_n_and_unit() -> None:
    with pytest.raises(ValueError, match="n must be positive"):
        word_ngrams("cat", n=0)
    with pytest.raises(TypeError, match="n must be an integer"):
        character_ngrams("cat", n=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unit must be one of"):
        ngram_overlap("cat", "cat", unit="sentence")
