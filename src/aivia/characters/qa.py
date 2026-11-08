"""Character consistency QA: vision features and embedding similarity with retry.

A generated asset is checked against a character's reference images using two
signals:

* vision feature checks: a vision scorer compares the generated image to each
  reference image and keeps the best score
* embedding similarity: text descriptions of the generated image and references
  are embedded and compared with cosine similarity

Both scores must clear their configured thresholds for the asset to pass. A
failed check retries the same strategy up to ``max_retries`` times; when every
attempt fails the strategy declines and the consistency chain moves to the next
strategy.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

from aivia.core.config.character import CharacterQAConfig
from aivia.core.errors import ValidationError
from aivia.core.similarity import cosine_similarity
from aivia.providers.protocols.embeddings import (
    EmbeddingProvider,
    EmbeddingRequest,
)

__all__ = [
    "CharacterQA",
    "CharacterQAResult",
    "VisionScorer",
    "check_character_consistency",
]

VisionScorer = Callable[[Any, Any], float | Coroutine[Any, Any, float]]
"""Compare two images and return a similarity score in ``[0, 1]``."""


@dataclass(frozen=True, slots=True)
class CharacterQAResult:
    """Outcome of one character consistency check."""

    passed: bool
    vision_score: float | None
    embedding_score: float | None
    overall_score: float
    reason: str
    checks: int = 1

    @property
    def score(self) -> float:
        """Alias for :attr:`overall_score`."""
        return self.overall_score


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _combine(vision: float | None, embedding: float | None) -> float:
    present = [score for score in (vision, embedding) if score is not None]
    if not present:
        return 0.0
    return sum(present) / len(present)


def _passes(
    overall: float,
    vision: float | None,
    embedding: float | None,
    config: CharacterQAConfig,
) -> bool:
    if overall < config.min_score:
        return False
    if vision is not None and vision < config.vision_threshold:
        return False
    return not (embedding is not None and embedding < config.embedding_threshold)


class CharacterQA:
    """Check character consistency with vision features and embedding similarity.

    ``vision_scorer`` compares two images and returns a similarity score. It may
    be synchronous or asynchronous. ``embedding_provider`` embeds text
    descriptions and the vectors are compared with cosine similarity.

    A generated asset passes when its overall score reaches ``min_score`` and
    each available signal clears its own threshold. On failure the same strategy
    is retried up to ``max_retries`` times; when every attempt fails the
    strategy declines and the consistency chain moves to the next strategy.
    """

    def __init__(
        self,
        *,
        config: CharacterQAConfig | None = None,
        vision_scorer: VisionScorer | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self._config = config or CharacterQAConfig()
        self._vision_scorer = vision_scorer
        self._embedding_provider = embedding_provider

    @property
    def config(self) -> CharacterQAConfig:
        """The QA thresholds and retry budget this instance applies."""
        return self._config

    @property
    def max_retries(self) -> int:
        """How many retries are allowed before declining to the next strategy."""
        return self._config.max_retries

    @property
    def has_vision(self) -> bool:
        """Whether a vision scorer is configured."""
        return self._vision_scorer is not None

    @property
    def has_embedding(self) -> bool:
        """Whether an embedding provider is configured."""
        return self._embedding_provider is not None

    async def _vision(self, asset: Any, references: list[Any]) -> float | None:
        if self._vision_scorer is None or not references:
            return None
        best = 0.0
        for reference in references:
            value = await _maybe_await(self._vision_scorer(asset, reference))
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("vision scorer must return a number")
            value = float(value)
            if not 0.0 <= value <= 1.0:
                raise ValueError("vision score must be between 0 and 1")
            if value > best:
                best = value
        return best

    async def _embedding(
        self,
        asset_description: str,
        reference_descriptions: list[str],
    ) -> float | None:
        if self._embedding_provider is None or not reference_descriptions:
            return None
        texts = [asset_description, *reference_descriptions]
        request = EmbeddingRequest(inputs=texts, normalize=True)
        response = await self._embedding_provider.embed(request)
        vectors = response.vectors
        if len(vectors) != len(texts):
            raise ValidationError(
                "embedding provider returned the wrong number of vectors",
                details={"expected": len(texts), "received": len(vectors)},
            )
        asset_vector = vectors[0]
        best = -1.0
        for reference_vector in vectors[1:]:
            score = cosine_similarity(asset_vector, reference_vector)
            if score > best:
                best = score
        return (best + 1.0) / 2.0

    async def check(
        self,
        asset: Any,
        reference_images: list[Any],
        *,
        asset_description: str = "",
        reference_descriptions: list[str] | None = None,
    ) -> CharacterQAResult:
        """Check *asset* against *reference_images* for character consistency.

        Returns a :class:`CharacterQAResult` with the vision and embedding
        scores, the combined overall score and whether the asset passes.
        """
        reference_images = list(reference_images)
        reference_descriptions = list(reference_descriptions or ())

        vision = await self._vision(asset, reference_images)
        embedding = await self._embedding(asset_description, reference_descriptions)
        overall = _combine(vision, embedding)
        passed = _passes(overall, vision, embedding, self._config)

        parts: list[str] = []
        if vision is not None:
            parts.append(f"vision={vision:.2f}")
        if embedding is not None:
            parts.append(f"embedding={embedding:.2f}")
        if not parts:
            parts.append("no signals")
        reason = "passed" if passed else "failed: " + ", ".join(parts)

        return CharacterQAResult(
            passed=passed,
            vision_score=vision,
            embedding_score=embedding,
            overall_score=overall,
            reason=reason,
            checks=1,
        )

    async def run(
        self,
        generate: Callable[[str], Any | Coroutine[Any, Any, Any]],
        scene: str,
        reference_images: list[Any],
        *,
        asset_description: str = "",
        reference_descriptions: list[str] | None = None,
    ) -> Any | None:
        """Run *generate* for *scene* and retry until the asset passes consistency.

        Returns the accepted asset, or ``None`` when every attempt fails, in
        which case the consistency chain should move to the next strategy.
        """
        reference_images = list(reference_images)
        reference_descriptions = list(reference_descriptions or ())

        for _attempt in range(self._config.max_retries + 1):
            asset = await _maybe_await(generate(scene))
            result = await self.check(
                asset,
                reference_images,
                asset_description=asset_description,
                reference_descriptions=reference_descriptions,
            )
            if result.passed:
                return asset
        return None


def check_character_consistency(
    asset: Any,
    reference_images: list[Any],
    *,
    config: CharacterQAConfig | None = None,
    vision_scorer: VisionScorer | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    asset_description: str = "",
    reference_descriptions: list[str] | None = None,
) -> CharacterQAResult:
    """Run one character consistency check and return the result.

    A convenience wrapper around :class:`CharacterQA.check` for synchronous
    callers.
    """
    qa = CharacterQA(
        config=config,
        vision_scorer=vision_scorer,
        embedding_provider=embedding_provider,
    )
    return asyncio.run(
        qa.check(
            asset,
            reference_images,
            asset_description=asset_description,
            reference_descriptions=reference_descriptions,
        )
    )
