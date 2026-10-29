"""Deterministic similarity measures with documented thresholds.

Three cheap, offline measures cover the comparisons the pipeline needs before
it spends money on a model:

* :func:`cosine_similarity` compares embedding vectors. It accepts vectors of
  any length (normalized or not) and treats a zero vector as unrelated.
* :func:`normalize_title` plus :func:`title_similarity` compare human titles.
  Normalization folds case, accents, punctuation and whitespace; the score then
  ignores word order so ``"Cat rescued: breaking news"`` and ``"Breaking news,
  cat rescued"`` match.
* :func:`ngram_overlap` compares word or character shingles. It is the cheap
  plagiarism signal for text gates and topic de-duplication.

The module constants are starting points, not laws. A caller with a calibrated
reference gate should pass its configured limit instead.

Thresholds
----------
``DEFAULT_COSINE_THRESHOLD`` (0.85)
    Text embeddings at or above this are near duplicates. The default text
    reference gate uses the same value. Visual embeddings are stricter; use
    ``DEFAULT_VISUAL_COSINE_THRESHOLD`` (0.90) for images and frames.
``DEFAULT_TITLE_THRESHOLD`` (0.85)
    Token-sorted title similarity at or above this is the same story. Lower it
    to catch more paraphrased headlines at the cost of false positives.
``DEFAULT_NGRAM_THRESHOLD`` (0.60)
    Word-trigram Jaccard overlap at or above this indicates reused wording.
    Character trigrams score differently, so calibrate before applying it to a
    character shingle comparison.

All scores are in ``[0.0, 1.0]`` except cosine, which is in ``[-1.0, 1.0]``.
Empty or missing text is never treated as a match, except when both sides are
empty and therefore identical.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Sequence
from difflib import SequenceMatcher
from enum import StrEnum

from aivia.core.config.reference import (
    DEFAULT_NGRAM_SIMILARITY_THRESHOLD,
    DEFAULT_TEXT_SIMILARITY_THRESHOLD,
    DEFAULT_TITLE_SIMILARITY_THRESHOLD,
    DEFAULT_VISUAL_SIMILARITY_THRESHOLD,
)

__all__ = [
    "DEFAULT_COSINE_THRESHOLD",
    "DEFAULT_NGRAM_THRESHOLD",
    "DEFAULT_TITLE_THRESHOLD",
    "DEFAULT_VISUAL_COSINE_THRESHOLD",
    "NgramUnit",
    "character_ngrams",
    "cosine_similarity",
    "ngram_containment",
    "ngram_overlap",
    "normalize_title",
    "title_similarity",
    "titles_match",
    "word_ngrams",
]

# The numeric defaults live with the reference configuration, where threshold
# tuning belongs. These names are the similarity-facing aliases.
DEFAULT_COSINE_THRESHOLD = DEFAULT_TEXT_SIMILARITY_THRESHOLD
"""Cosine at or above this marks text embeddings as near duplicates."""

DEFAULT_VISUAL_COSINE_THRESHOLD = DEFAULT_VISUAL_SIMILARITY_THRESHOLD
"""Cosine at or above this marks image or frame embeddings as near duplicates."""

DEFAULT_TITLE_THRESHOLD = DEFAULT_TITLE_SIMILARITY_THRESHOLD
"""Token-sorted title similarity at or above this marks the same story."""

DEFAULT_NGRAM_THRESHOLD = DEFAULT_NGRAM_SIMILARITY_THRESHOLD
"""Word-trigram Jaccard overlap at or above this indicates reused wording."""

_WORD = re.compile(r"[^\W_]+", re.UNICODE)
"""Unicode word characters, excluding the underscore."""


class NgramUnit(StrEnum):
    """Whether shingles are built from words or characters."""

    WORD = "word"
    CHARACTER = "character"


def _text(value: str, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    return value


def _threshold(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1")
    return number


def _fold(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in folded if not unicodedata.combining(char))


def _tokens(text: str) -> list[str]:
    return _WORD.findall(_fold(text))


def cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    """Return the cosine similarity of two vectors in ``[-1.0, 1.0]``.

    The vectors may be normalized or not; the result is scale invariant. A zero
    vector carries no direction and returns ``0.0`` rather than a NaN. Vectors
    of different lengths, empty vectors and non-finite components are rejected
    so a dimension mistake never looks like a low score.
    """
    if isinstance(left, (str, bytes)) or isinstance(right, (str, bytes)):
        raise TypeError("vectors must be sequences of numbers")
    left_values = [float(component) for component in left]
    right_values = [float(component) for component in right]
    if not left_values or not right_values:
        raise ValueError("vectors must not be empty")
    if len(left_values) != len(right_values):
        raise ValueError(
            "vectors must have the same length: "
            f"{len(left_values)} != {len(right_values)}"
        )
    if not all(math.isfinite(value) for value in left_values + right_values):
        raise ValueError("vector components must be finite")

    dot = sum(a * b for a, b in zip(left_values, right_values, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    score = dot / (left_norm * right_norm)
    return max(-1.0, min(1.0, score))


def normalize_title(title: str) -> str:
    """Return a comparison key for a title.

    Case, accents, punctuation and repeated whitespace are folded away, so
    ``"  Café: The  Story! "`` becomes ``"cafe the story"``. The words keep their
    order here; :func:`title_similarity` also ignores order when scoring.
    """
    return " ".join(_tokens(_text(title, "title")))


def title_similarity(left: str, right: str) -> float:
    """Return how alike two titles are in ``[0.0, 1.0]``.

    Titles are normalized, their tokens are sorted and the ratio is computed
    with :class:`difflib.SequenceMatcher`. Sorting makes word order irrelevant
    while the ratio still tolerates small spelling changes such as a plural.
    Two empty titles are identical (``1.0``); one empty title is ``0.0``.
    """
    left_key = normalize_title(left)
    right_key = normalize_title(right)
    if not left_key or not right_key:
        return 1.0 if left_key == right_key else 0.0
    if left_key == right_key:
        return 1.0
    left_sorted = " ".join(sorted(left_key.split()))
    right_sorted = " ".join(sorted(right_key.split()))
    return SequenceMatcher(None, left_sorted, right_sorted).ratio()


def titles_match(
    left: str,
    right: str,
    threshold: float = DEFAULT_TITLE_THRESHOLD,
) -> bool:
    """Return whether two titles score at or above *threshold*."""
    return title_similarity(left, right) >= _threshold(threshold, "threshold")


def _validate_n(n: int) -> int:
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n < 1:
        raise ValueError("n must be positive")
    return n


def _unit(value: NgramUnit | str) -> NgramUnit:
    if isinstance(value, NgramUnit):
        return value
    try:
        return NgramUnit(value)
    except ValueError as exc:
        allowed = ", ".join(unit.value for unit in NgramUnit)
        raise ValueError(f"unit must be one of: {allowed}") from exc


def word_ngrams(text: str, n: int = 3) -> list[tuple[str, ...]]:
    """Return the word shingles of *text*, folded and with punctuation dropped.

    A text shorter than *n* words yields one shingle containing every word, so
    a one-word title can still be compared. An empty text yields no shingles.
    """
    size = _validate_n(n)
    tokens = _tokens(_text(text, "text"))
    if not tokens:
        return []
    if len(tokens) < size:
        return [tuple(tokens)]
    return [
        tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)
    ]


def character_ngrams(text: str, n: int = 3) -> list[str]:
    """Return the character shingles of normalized *text*.

    Punctuation is removed and whitespace is collapsed, so formatting cannot
    hide reused wording. A text shorter than *n* characters yields one shingle.
    """
    size = _validate_n(n)
    normalized = " ".join(_tokens(_text(text, "text")))
    if not normalized:
        return []
    if len(normalized) < size:
        return [normalized]
    return [
        normalized[index : index + size] for index in range(len(normalized) - size + 1)
    ]


def _shingles(text: str, n: int, unit: NgramUnit) -> set[str]:
    if unit is NgramUnit.WORD:
        return {"\0".join(shingle) for shingle in word_ngrams(text, n)}
    return set(character_ngrams(text, n))


def ngram_overlap(
    left: str,
    right: str,
    n: int = 3,
    *,
    unit: NgramUnit | str = NgramUnit.WORD,
) -> float:
    """Return the Jaccard overlap of two texts' *n*-grams in ``[0.0, 1.0]``.

    Jaccard is the shared shingles divided by all shingles on either side, so
    adding unrelated text lowers the score. Use :func:`ngram_containment` when
    a short candidate is compared against a long reference. Two empty texts are
    identical (``1.0``); one empty text is ``0.0``.
    """
    size = _validate_n(n)
    kind = _unit(unit)
    left_shingles = _shingles(_text(left, "left"), size, kind)
    right_shingles = _shingles(_text(right, "right"), size, kind)
    union = left_shingles | right_shingles
    if not union:
        return 1.0
    return len(left_shingles & right_shingles) / len(union)


def ngram_containment(
    left: str,
    right: str,
    n: int = 3,
    *,
    unit: NgramUnit | str = NgramUnit.WORD,
) -> float:
    """Return the share of the smaller text's *n*-grams found in the other.

    Containment answers "how much of this candidate appears in the reference?"
    and is not reduced by extra unrelated reference text, unlike Jaccard. Two
    empty texts are identical (``1.0``); one empty text is ``0.0``.
    """
    size = _validate_n(n)
    kind = _unit(unit)
    left_shingles = _shingles(_text(left, "left"), size, kind)
    right_shingles = _shingles(_text(right, "right"), size, kind)
    smallest = min(len(left_shingles), len(right_shingles))
    if smallest == 0:
        return 1.0 if not left_shingles and not right_shingles else 0.0
    return len(left_shingles & right_shingles) / smallest
