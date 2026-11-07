"""Text similarity gate for reference driven generation.

A candidate text is compared against a text reference with a cheap,
deterministic, offline measure before any judgement is made:

* :func:`~aivia.core.similarity.ngram_overlap` computes the word- or
  character-trigram Jaccard overlap of the candidate against the reference.

When the score is at or above the configured ``text_threshold`` the caller may
regenerate (rewrite) up to ``max_regenerations`` times. If the limit is still
exceeded the candidate is refused rather than shipped as a copy of the
reference.

All scores are in ``[0.0, 1.0]`` where higher means more alike. The gate
triggers at or above the configured limit, so the limit itself is treated as
too similar.
"""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from aivia.core.errors import ValidationError
from aivia.core.similarity import NgramUnit, ngram_overlap

__all__ = [
    "TextGateError",
    "TextGateResult",
    "gate_text_similarity",
    "ngram_similarity",
]


class TextGateError(ValidationError):
    """A candidate is too similar to the reference and cannot be regenerated."""

    code = "text_similarity"
    retryable = False


@dataclass(frozen=True, slots=True)
class TextGateResult:
    """The outcome of one text similarity gate decision."""

    passed: bool
    """Whether the candidate cleared the gate (similarity below the limit)."""

    score: float
    """The n-gram overlap similarity score in ``[0.0, 1.0]``."""

    method: str
    """Which shingle unit contributed: ``word_ngram`` or ``character_ngram``."""

    regenerated: int
    """How many times the regenerator was called before the decision."""

    refused: bool
    """Whether the gate refused the candidate after exhausting regenerations."""

    reason: str | None
    """A human readable reason when the candidate was refused, else ``None``."""


def ngram_similarity(
    reference: str,
    candidate: str,
    *,
    n: int = 3,
    unit: NgramUnit | str = NgramUnit.WORD,
) -> float:
    """Return the *n*-gram overlap of *candidate* against *reference* in ``[0.0, 1.0]``.

    Uses Jaccard overlap of the folded shingles, so added unrelated text on
    either side lowers the score. Two empty texts are identical (``1.0``);
    one empty text is ``0.0``.
    """
    return ngram_overlap(reference, candidate, n=n, unit=unit)


async def gate_text_similarity(
    *,
    reference: str,
    candidate: str,
    threshold: float,
    max_regenerations: int = 2,
    n: int = 3,
    unit: NgramUnit | str = NgramUnit.WORD,
    regenerate: Callable[[], Awaitable[str]] | None = None,
    run_id: str = "",
) -> TextGateResult:
    """Gate one candidate text against a text reference.

    Compares *candidate* to *reference* with n-gram overlap and, when the
    score is at or above *threshold*, calls *regenerate* (when supplied) to
    produce a rewritten replacement and re-checks. This repeats up to
    *max_regenerations* times. If the limit is still exceeded the result is
    marked refused.

    ``threshold`` must be in ``[0, 1]``. ``max_regenerations`` of ``0`` means
    the first candidate above the limit is refused immediately.
    """
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError("similarity limit must be a number")
    level = float(threshold)
    if not math.isfinite(level) or level < 0.0 or level > 1.0:
        raise ValueError("similarity limit must be between 0 and 1")
    if isinstance(max_regenerations, bool) or not isinstance(max_regenerations, int):
        raise TypeError("max_regenerations must be an integer")
    count = max_regenerations
    if count < 0:
        raise ValueError("max_regenerations must not be negative")
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    kind = unit if isinstance(unit, NgramUnit) else NgramUnit(unit)

    current = candidate
    regenerated = 0
    reason: str | None = None
    method = f"{kind.value}_ngram"

    while True:
        score = ngram_similarity(reference, current, n=n, unit=kind)
        if score < level:
            return TextGateResult(
                passed=True,
                score=score,
                method=method,
                regenerated=regenerated,
                refused=False,
                reason=None,
            )
        if regenerated >= max_regenerations:
            reason = (
                f"text similarity {score:.3f} is at or above the "
                f"limit {threshold:g} after {max_regenerations} regeneration(s)"
            )
            return TextGateResult(
                passed=False,
                score=score,
                method=method,
                regenerated=regenerated,
                refused=True,
                reason=reason,
            )
        if regenerate is None:
            reason = (
                f"text similarity {score:.3f} is at or above the "
                f"limit {threshold:g} and no regenerator is configured"
            )
            return TextGateResult(
                passed=False,
                score=score,
                method=method,
                regenerated=regenerated,
                refused=True,
                reason=reason,
            )
        current = await regenerate()
        regenerated += 1
