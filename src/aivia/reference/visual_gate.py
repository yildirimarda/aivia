"""Visual similarity gate for reference driven generation.

A candidate output is compared against a visual reference with two cheap,
deterministic measures before any judgement is made:

* a perceptual (average) hash of the pixels, compared by Hamming distance, and
* an optional embedding similarity computed by an injected callable so a real
  image embedding model can be plugged in without this module importing a vendor.

When the combined score is at or above the configured ``visual_threshold`` the
caller may regenerate up to ``max_regenerations`` times. If the limit is still
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
from typing import Any

from PIL import Image, ImageOps

from aivia.core.errors import ValidationError
from aivia.core.similarity import cosine_similarity
from aivia.media.image_utils import open_image

__all__ = [
    "DEFAULT_HASH_SIZE",
    "VisualGateError",
    "VisualGateResult",
    "average_hash",
    "embedding_similarity",
    "gate_visual_similarity",
    "hamming_similarity",
    "perceptual_similarity",
    "visual_similarity",
]

DEFAULT_HASH_SIZE = 8
"""The side length, in pixels, of the average hash grid."""


class VisualGateError(ValidationError):
    """A candidate is too similar to the reference and cannot be regenerated."""

    code = "visual_similarity"
    retryable = False


@dataclass(frozen=True, slots=True)
class VisualGateResult:
    """The outcome of one visual similarity gate decision."""

    passed: bool
    """Whether the candidate cleared the gate (similarity below the limit)."""

    score: float
    """The combined similarity score in ``[0.0, 1.0]``."""

    method: str
    """Which measure contributed: ``perceptual_hash``, ``embedding`` or
    ``combined``."""

    regenerated: int
    """How many times the regenerator was called before the decision."""

    refused: bool
    """Whether the gate refused the candidate after exhausting regenerations."""

    reason: str | None
    """A human readable reason when the candidate was refused, else ``None``."""


def _to_image(source: Any) -> Image.Image:
    """Open *source* as a detached, exif corrected RGB image."""
    if isinstance(source, Image.Image):
        return ImageOps.exif_transpose(source.copy()).convert("RGB")
    return open_image(source).convert("RGB")


def _hash_bits(value: str) -> int:
    """Return the integer bits of one hex hash string."""
    return int(value, 16)


def average_hash(image: Any, size: int = DEFAULT_HASH_SIZE) -> str:
    """Return the average hash of *image* as a lowercase hex string.

    *image* may be a :class:`~PIL.Image.Image`, raw bytes, a path or a URL.
    The image is converted to grayscale, resized to ``size x size`` and each
    pixel is compared to the mean: brighter than average becomes ``1``, the
    rest ``0``. The bits are packed left to right, top to bottom.
    """
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("hash size must be a positive integer")
    image = _to_image(image)
    small = image.convert("L").resize((size, size), Image.Resampling.BILINEAR)
    pixels = list(small.getdata())
    mean = sum(pixels) / len(pixels)
    bits = 0
    for pixel in pixels:
        bits = (bits << 1) | (1 if pixel > mean else 0)
    width = size * size
    return format(bits, f"0{width // 4}x")


def hamming_similarity(left: str, right: str) -> float:
    """Return the similarity of two equal length hex hashes in ``[0.0, 1.0]``.

    Identical hashes score ``1.0``; completely different hashes approach
    ``0.0``. Hashes of different lengths are padded to the longer side so a
    size change never crashes the comparison.
    """
    if not isinstance(left, str) or not isinstance(right, str):
        raise TypeError("hashes must be hex strings")
    width = max(len(left), len(right))
    left_bits = _hash_bits(left.zfill(width))
    right_bits = _hash_bits(right.zfill(width))
    distance = bin(left_bits ^ right_bits).count("1")
    total = width * 4
    if total == 0:
        return 1.0
    return max(0.0, 1.0 - distance / total)


def perceptual_similarity(
    reference: Any,
    candidate: Any,
    *,
    size: int = DEFAULT_HASH_SIZE,
) -> float:
    """Return the average hash similarity of two images in ``[0.0, 1.0]``."""
    left = average_hash(_to_image(reference), size=size)
    right = average_hash(_to_image(candidate), size=size)
    return hamming_similarity(left, right)


async def embedding_similarity(
    embed: Callable[[Image.Image], Awaitable[list[float]]],
    reference: Any,
    candidate: Any,
) -> float:
    """Return the cosine similarity of two embedded images in ``[0.0, 1.0]``.

    Negative cosine is treated as unrelated (``0.0``) so orthogonal or opposed
    vectors never look like a match.
    """
    reference_vector = await embed(_to_image(reference))
    candidate_vector = await embed(_to_image(candidate))
    score = cosine_similarity(reference_vector, candidate_vector)
    return max(0.0, min(1.0, score))


async def visual_similarity(
    reference: Any,
    candidate: Any,
    *,
    embed: Callable[[Image.Image], Awaitable[list[float]]] | None = None,
    size: int = DEFAULT_HASH_SIZE,
) -> tuple[float, str]:
    """Return ``(score, method)`` for *candidate* against *reference*.

    The score is the maximum of the perceptual hash similarity and, when
    *embed* is supplied, the embedding similarity. Taking the maximum is
    conservative: either measure flagging the pair as alike treats them as a
    near copy. *method* is ``perceptual_hash``, ``embedding`` or ``combined``.
    """
    score = perceptual_similarity(reference, candidate, size=size)
    method = "perceptual_hash"
    if embed is not None:
        embedded = await embedding_similarity(embed, reference, candidate)
        score = max(score, embedded)
        method = "combined"
    return score, method


async def gate_visual_similarity(
    *,
    reference: Any,
    candidate: Any,
    threshold: float,
    max_regenerations: int = 2,
    embed: Callable[[Image.Image], Awaitable[list[float]]] | None = None,
    regenerate: Callable[[], Awaitable[Any]] | None = None,
    run_id: str = "",
) -> VisualGateResult:
    """Gate one candidate against a visual reference.

    Compares *candidate* to *reference* and, when the score is at or above
    *threshold*, calls *regenerate* (when supplied) to produce a replacement and
    re-checks. This repeats up to *max_regenerations* times. If the limit is
    still exceeded the result is marked refused.

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

    current = candidate
    regenerated = 0
    reason: str | None = None

    while True:
        score, method = await visual_similarity(reference, current, embed=embed)
        if score < level:
            return VisualGateResult(
                passed=True,
                score=score,
                method=method,
                regenerated=regenerated,
                refused=False,
                reason=None,
            )
        if regenerated >= max_regenerations:
            reason = (
                f"visual similarity {score:.3f} is at or above the "
                f"limit {threshold:g} after {max_regenerations} regeneration(s)"
            )
            return VisualGateResult(
                passed=False,
                score=score,
                method=method,
                regenerated=regenerated,
                refused=True,
                reason=reason,
            )
        if regenerate is None:
            reason = (
                f"visual similarity {score:.3f} is at or above the "
                f"limit {threshold:g} and no regenerator is configured"
            )
            return VisualGateResult(
                passed=False,
                score=score,
                method=method,
                regenerated=regenerated,
                refused=True,
                reason=reason,
            )
        current = await regenerate()
        regenerated += 1
