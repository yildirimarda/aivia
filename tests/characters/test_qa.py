"""Character consistency QA: vision features, embedding similarity, retry.

A generated asset is checked against a character's reference images using a
vision scorer and an embedding provider. A failed check retries up to
``max_retries`` times; when every attempt fails the strategy declines and the
consistency chain moves to the next strategy.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

import pytest

from aivia.characters.qa import (
    CharacterQA,
    CharacterQAResult,
    check_character_consistency,
)
from aivia.core.config.character import CharacterQAConfig
from aivia.core.errors import ValidationError
from aivia.providers.protocols.embeddings import (
    Embedding,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
)


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine)


class FakeVisionScorer:
    """Return one score per call from a fixed sequence."""

    def __init__(self, scores: list[float]) -> None:
        self.scores = list(scores)
        self.calls: list[tuple[Any, Any]] = []

    def __call__(self, asset: Any, reference: Any) -> float:
        self.calls.append((asset, reference))
        return self.scores.pop(0) if self.scores else 1.0


class AsyncVisionScorer:
    """Async vision scorer returning one score per call."""

    def __init__(self, scores: list[float]) -> None:
        self.scores = list(scores)
        self.calls: list[tuple[Any, Any]] = []

    async def __call__(self, asset: Any, reference: Any) -> float:
        self.calls.append((asset, reference))
        return self.scores.pop(0) if self.scores else 1.0


class FakeEmbeddingProvider:
    """Return fixed vectors for embedded inputs."""

    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = list(vectors)
        self.requests: list[Any] = []

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self.requests.append(request)
        embeddings = [
            Embedding(index=index, vector=vector)
            for index, vector in enumerate(self.vectors)
        ]
        return EmbeddingResponse(
            embeddings=embeddings,
            model="local",
            provider="fake",
            usage=EmbeddingUsage(),
        )


def _asset(payload: Any = b"generated") -> dict[str, Any]:
    return {"bytes": payload}


def test_vision_and_embedding_both_pass() -> None:
    qa: CharacterQA = CharacterQA(
        vision_scorer=FakeVisionScorer([0.9]),
        embedding_provider=FakeEmbeddingProvider([[1.0, 0.0], [1.0, 0.0]]),  # type: ignore[arg-type]
    )
    result: CharacterQAResult = _run(
        qa.check(
            _asset(),
            ["ref.png"],
            asset_description="the mascot",
            reference_descriptions=["the mascot"],
        )
    )

    assert result.passed is True
    assert result.vision_score == pytest.approx(0.9)
    assert result.embedding_score == pytest.approx(1.0)
    assert result.overall_score == pytest.approx(0.95)
    assert result.score == pytest.approx(0.95)


def test_vision_below_threshold_fails() -> None:
    qa: CharacterQA = CharacterQA(
        vision_scorer=FakeVisionScorer([0.5]),
        embedding_provider=FakeEmbeddingProvider([[1.0, 0.0], [0.9, 0.1]]),  # type: ignore[arg-type]
    )
    result: CharacterQAResult = _run(
        qa.check(
            _asset(),
            ["ref.png"],
            asset_description="the mascot",
            reference_descriptions=["the mascot"],
        )
    )

    assert result.passed is False
    assert result.vision_score == pytest.approx(0.5)
    assert "vision=0.50" in result.reason


def test_embedding_below_threshold_fails() -> None:
    qa: CharacterQA = CharacterQA(
        vision_scorer=FakeVisionScorer([0.9]),
        embedding_provider=FakeEmbeddingProvider([[1.0, 0.0], [0.0, 1.0]]),  # type: ignore[arg-type]
    )
    result: CharacterQAResult = _run(
        qa.check(
            _asset(),
            ["ref.png"],
            asset_description="the mascot",
            reference_descriptions=["the mascot"],
        )
    )

    assert result.passed is False
    assert result.embedding_score == pytest.approx(0.5)
    assert "embedding=0.50" in result.reason


def test_overall_below_min_score_fails() -> None:
    qa: CharacterQA = CharacterQA(
        vision_scorer=FakeVisionScorer([0.6]),
        embedding_provider=FakeEmbeddingProvider([[1.0, 0.0], [0.0, 1.0]]),  # type: ignore[arg-type]
    )
    result: CharacterQAResult = _run(
        qa.check(
            _asset(),
            ["ref.png"],
            asset_description="the mascot",
            reference_descriptions=["the mascot"],
        )
    )

    assert result.passed is False
    assert result.overall_score == pytest.approx(0.55)


def test_vision_only_uses_vision_score() -> None:
    qa: CharacterQA = CharacterQA(vision_scorer=FakeVisionScorer([0.85]))
    result: CharacterQAResult = _run(qa.check(_asset(), ["ref.png"]))

    assert result.passed is True
    assert result.vision_score == pytest.approx(0.85)
    assert result.embedding_score is None
    assert result.overall_score == pytest.approx(0.85)


def test_embedding_only_uses_embedding_score() -> None:
    qa: CharacterQA = CharacterQA(
        embedding_provider=FakeEmbeddingProvider([[1.0, 0.0], [0.8, 0.6]]),  # type: ignore[arg-type]
    )
    result: CharacterQAResult = _run(
        qa.check(
            _asset(),
            [],
            asset_description="the mascot",
            reference_descriptions=["the mascot"],
        )
    )

    assert result.passed is True
    assert result.vision_score is None
    assert result.embedding_score == pytest.approx(0.9)
    assert result.overall_score == pytest.approx(0.9)


def test_no_signals_fails() -> None:
    qa: CharacterQA = CharacterQA()
    result: CharacterQAResult = _run(qa.check(_asset(), []))

    assert result.passed is False
    assert result.vision_score is None
    assert result.embedding_score is None
    assert result.overall_score == 0.0
    assert "no signals" in result.reason


def test_no_reference_images_skips_vision() -> None:
    qa: CharacterQA = CharacterQA(vision_scorer=FakeVisionScorer([0.9]))
    result: CharacterQAResult = _run(qa.check(_asset(), []))

    assert result.vision_score is None
    assert result.passed is False


def test_no_reference_descriptions_skips_embedding() -> None:
    qa: CharacterQA = CharacterQA(
        embedding_provider=FakeEmbeddingProvider([[1.0, 0.0]]),  # type: ignore[arg-type]
    )
    result: CharacterQAResult = _run(
        qa.check(_asset(), [], asset_description="the mascot")
    )

    assert result.embedding_score is None
    assert result.passed is False


def test_vision_scorer_out_of_range_raises() -> None:
    qa: CharacterQA = CharacterQA(vision_scorer=FakeVisionScorer([1.5]))

    with pytest.raises(ValueError, match="vision score must be between 0 and 1"):
        _run(qa.check(_asset(), ["ref.png"]))


def test_vision_scorer_non_number_raises() -> None:
    class BadScorer:
        def __call__(self, asset: Any, reference: Any) -> str:
            return "not a number"

    qa: CharacterQA = CharacterQA(vision_scorer=BadScorer())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="vision scorer must return a number"):
        _run(qa.check(_asset(), ["ref.png"]))


def test_embedding_wrong_vector_count_raises() -> None:
    provider = FakeEmbeddingProvider([[1.0, 0.0]])
    qa: CharacterQA = CharacterQA(embedding_provider=provider)  # type: ignore[arg-type]

    with pytest.raises(ValidationError, match="wrong number of vectors"):
        _run(
            qa.check(
                _asset(),
                [],
                asset_description="the mascot",
                reference_descriptions=["a", "b"],
            )
        )


def test_async_vision_scorer_is_supported() -> None:
    qa: CharacterQA = CharacterQA(vision_scorer=AsyncVisionScorer([0.9]))
    result: CharacterQAResult = _run(qa.check(_asset(), ["ref.png"]))

    assert result.passed is True
    assert result.vision_score == pytest.approx(0.9)
    scorer: Any = qa._vision_scorer
    assert len(scorer.calls) == 1


def test_retry_until_pass_returns_asset() -> None:
    calls: list[str] = []

    def generate(scene: str) -> dict[str, Any]:
        calls.append(scene)
        return _asset(b"attempt-" + str(len(calls)).encode())

    qa: CharacterQA = CharacterQA(
        vision_scorer=FakeVisionScorer([0.4, 0.9]),
        config=CharacterQAConfig(max_retries=2),
    )
    asset = _run(
        qa.run(
            generate,
            "scene one",
            ["ref.png"],
            asset_description="the mascot",
            reference_descriptions=["the mascot"],
        )
    )

    assert asset == _asset(b"attempt-2")
    assert len(calls) == 2


def test_retry_exhausted_returns_none() -> None:
    calls: list[str] = []

    def generate(scene: str) -> dict[str, Any]:
        calls.append(scene)
        return _asset(b"attempt")

    qa: CharacterQA = CharacterQA(
        vision_scorer=FakeVisionScorer([0.4, 0.4, 0.4]),
        config=CharacterQAConfig(max_retries=2),
    )
    asset = _run(
        qa.run(
            generate,
            "scene one",
            ["ref.png"],
            asset_description="the mascot",
            reference_descriptions=["the mascot"],
        )
    )

    assert asset is None
    assert len(calls) == 3


def test_max_retries_zero_single_attempt_then_none() -> None:
    def generate(scene: str) -> dict[str, Any]:
        return _asset()

    qa: CharacterQA = CharacterQA(
        vision_scorer=FakeVisionScorer([0.4]),
        config=CharacterQAConfig(max_retries=0),
    )
    asset = _run(qa.run(generate, "scene one", ["ref.png"]))

    assert asset is None


def test_async_generate_is_supported() -> None:
    async def generate(scene: str) -> dict[str, Any]:
        return _asset()

    qa: CharacterQA = CharacterQA(
        vision_scorer=FakeVisionScorer([0.9]),
        config=CharacterQAConfig(max_retries=1),
    )
    asset = _run(qa.run(generate, "scene one", ["ref.png"]))

    assert asset == _asset()


def test_multiple_reference_images_keeps_best_vision() -> None:
    qa: CharacterQA = CharacterQA(vision_scorer=FakeVisionScorer([0.4, 0.9]))
    result: CharacterQAResult = _run(qa.check(_asset(), ["ref-a.png", "ref-b.png"]))

    assert result.passed is True
    assert result.vision_score == pytest.approx(0.9)
    scorer: Any = qa._vision_scorer
    assert len(scorer.calls) == 2


def test_multiple_reference_descriptions_keeps_best_embedding() -> None:
    provider = FakeEmbeddingProvider([[1.0, 0.0], [0.0, 1.0], [0.8, 0.6]])
    qa: CharacterQA = CharacterQA(embedding_provider=provider)  # type: ignore[arg-type]
    result: CharacterQAResult = _run(
        qa.check(
            _asset(),
            [],
            asset_description="the mascot",
            reference_descriptions=["other", "the mascot"],
        )
    )

    assert result.passed is True
    assert result.embedding_score == pytest.approx(0.9)


def test_check_character_consistency_convenience_function() -> None:
    result: CharacterQAResult = check_character_consistency(
        _asset(),
        ["ref.png"],
        vision_scorer=FakeVisionScorer([0.9]),
        asset_description="the mascot",
        reference_descriptions=["the mascot"],
    )

    assert result.passed is True


def test_config_defaults_are_used() -> None:
    qa: CharacterQA = CharacterQA()
    assert qa.config.min_score == 0.8
    assert qa.config.vision_threshold == 0.8
    assert qa.config.embedding_threshold == 0.75
    assert qa.max_retries == 2


def test_explicit_config_overrides_defaults() -> None:
    config = CharacterQAConfig(
        min_score=0.9,
        vision_threshold=0.85,
        embedding_threshold=0.7,
        max_retries=4,
    )
    qa: CharacterQA = CharacterQA(config=config)
    assert qa.config.min_score == 0.9
    assert qa.config.vision_threshold == 0.85
    assert qa.config.embedding_threshold == 0.7
    assert qa.max_retries == 4


def test_result_reason_describes_failure() -> None:
    qa: CharacterQA = CharacterQA(
        vision_scorer=FakeVisionScorer([0.5]),
        embedding_provider=FakeEmbeddingProvider([[1.0, 0.0], [0.0, 1.0]]),  # type: ignore[arg-type]
    )
    result: CharacterQAResult = _run(
        qa.check(
            _asset(),
            ["ref.png"],
            asset_description="the mascot",
            reference_descriptions=["the mascot"],
        )
    )

    assert result.passed is False
    assert "vision=0.50" in result.reason
    assert "embedding=0.50" in result.reason
