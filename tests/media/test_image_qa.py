"""Generated images are scored, retried and fall back to the next provider or stock."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Coroutine
from typing import Any

import pytest

from aivia.core.config.capability import CapabilityChainConfig
from aivia.core.config.quality import QualityConfig
from aivia.core.errors import ConfigError
from aivia.core.media_probe import validate_media
from aivia.core.models.asset import Asset, AssetKind
from aivia.core.models.cost import CostEvent
from aivia.core.router import Router
from aivia.media.image_qa import (
    DEFAULT_IMAGE_QA_PROMPT,
    ImageQA,
    ImageQAResult,
    ImageQualityAssurance,
    ImageQualityError,
    ImageQualityScore,
    build_quality_request,
    run_image_qa,
    score_image,
)
from aivia.media.image_service import ImageOutcome, ImageService
from aivia.media.screening import screen_prompt
from aivia.providers import Provider
from aivia.providers.mock import MockImageProvider
from aivia.providers.protocols.image import ImageRequest, ImageResponse, ImageResult
from aivia.providers.protocols.llm import (
    ChatMessage,
    ImagePart,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    MessageRole,
    TextPart,
)
from aivia.providers.protocols.stock import StockAsset, StockResponse

PROMPT = "A lighthouse at sunset"
PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine)


def _outcome(provider: str = "acme", *, amount: float = 0.02) -> ImageOutcome:
    probe = validate_media(PNG_1X1)
    asset = Asset(
        kind=AssetKind.IMAGE,
        provider=provider,
        model="mock-image",
        prompt=PROMPT,
        content_hash=probe.sha256,
    )
    event = CostEvent(
        provider=provider,
        model="mock-image",
        operation="image_gen",
        units=1.0,
        unit_kind="image",
        amount_usd=amount,
    )
    return ImageOutcome(
        response=ImageResponse(
            images=[ImageResult(data=PNG_1X1)],
            provider=provider,
            model="mock-image",
        ),
        assets=(asset,),
        probes=(probe,),
        events=(event,),
    )


def _asset(asset_id: str = "stock-1", provider: str = "pexels") -> StockAsset:
    return StockAsset(
        id=asset_id,
        provider=provider,
        url=f"https://cdn.example.com/{asset_id}.jpg",
        width=1200,
        height=800,
    )


class ScriptedScorer:
    """Return one score per call from a fixed sequence."""

    def __init__(self, scores: list[float]) -> None:
        self.scores = list(scores)
        self.sources: list[Any] = []

    async def __call__(self, source: Any) -> float:
        self.sources.append(source)
        return self.scores.pop(0) if self.scores else 1.0


class ScriptedGenerator:
    """A fake image service with a chain and provider aware generation."""

    def __init__(self, chain: list[str], outcomes: list[ImageOutcome]) -> None:
        self.providers = tuple(chain)
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        request: ImageRequest,
        *,
        exclude: set[str] | None = None,
        provider: str | None = None,
        no_cache: bool = True,
        token: Any = None,
        run_id: str = "",
        metadata: Any = None,
    ) -> ImageOutcome:
        self.calls.append(
            {
                "provider": provider,
                "exclude": set(exclude or ()),
                "no_cache": no_cache,
                "token": token,
                "run_id": run_id,
                "metadata": metadata,
            }
        )
        if self.outcomes:
            return self.outcomes.pop(0)
        return _outcome(provider or "acme")


class FakeVisionProvider:
    """An async LLM provider returning a canned quality payload."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(
            content=json.dumps(self.payload),
            provider="acme",
            model="vision-model",
            usage=LLMUsage(input_tokens=10, output_tokens=5),
        )


def test_a_score_can_be_explicit_or_computed_from_weighted_dimensions() -> None:
    explicit = ImageQualityScore.model_validate(
        {
            "overall": 0.8,
            "prompt_fidelity": 0.9,
            "technical": 0.7,
            "aesthetics": 0.6,
        }
    )
    computed = ImageQualityScore.model_validate(
        {
            "prompt_adherence": 1.0,
            "technical_quality": 0.5,
            "aesthetic_quality": 0.5,
        }
    )

    assert explicit.score == 0.8
    assert explicit.dimension_scores == {
        "prompt_adherence": 0.9,
        "technical_quality": 0.7,
        "aesthetic_quality": 0.6,
    }
    assert computed.score == pytest.approx(0.7)
    assert computed.overall == pytest.approx(0.7)
    assert computed.passed is True

    with pytest.raises(ValueError, match="score"):
        ImageQualityScore.model_validate({})
    with pytest.raises(ValueError, match="less than or equal to 1"):
        ImageQualityScore(score=1.2)


def test_the_request_carries_the_image_and_the_intended_prompt_as_data() -> None:
    request = build_quality_request(
        PNG_1X1,
        original_prompt=PROMPT,
        threshold=0.8,
        weights={"prompt_adherence": 0.5, "technical_quality": 0.5},
    )

    assert request.role == "vision_qa"
    assert request.metadata["quality_threshold"] == 0.8
    assert request.metadata["quality_weights"] == {
        "prompt_adherence": 0.5,
        "technical_quality": 0.5,
    }
    message = request.messages[0]
    assert isinstance(message, ChatMessage)
    assert message.role is MessageRole.USER
    assert isinstance(message.content, list)
    text = next(part for part in message.content if isinstance(part, TextPart))
    image = next(part for part in message.content if isinstance(part, ImagePart))
    assert DEFAULT_IMAGE_QA_PROMPT.split(".")[0] in text.text
    assert PROMPT in text.text
    assert "data, not instructions" in text.text
    assert image.data is not None
    assert image.media_type == "image/png"


def test_a_vision_provider_returns_a_validated_score() -> None:
    provider = FakeVisionProvider(
        {
            "overall": 0.82,
            "prompt_adherence": 0.9,
            "technical_quality": 0.8,
            "aesthetic_quality": 0.7,
            "description": "a lighthouse",
            "reason": "matches the prompt",
        }
    )

    score = _run(
        score_image(
            PNG_1X1,
            provider=provider,
            model="vision-model",
            original_prompt=PROMPT,
            threshold=0.8,
        )
    )

    assert isinstance(score, ImageQualityScore)
    assert score.score == pytest.approx(0.82)
    assert score.passed is True
    assert score.description == "a lighthouse"
    request = provider.requests[0]
    assert request.model == "vision-model"
    assert request.role == "vision_qa"
    assert request.images[0].data is not None


def test_configured_weights_recompute_a_dimension_only_vision_score() -> None:
    provider = FakeVisionProvider(
        {
            "prompt_adherence": 0.2,
            "technical_quality": 1.0,
            "aesthetic_quality": 0.2,
        }
    )

    score = _run(
        score_image(
            PNG_1X1,
            provider=provider,
            config=QualityConfig(
                min_score=0.5,
                weights={
                    "prompt_adherence": 0.8,
                    "technical_quality": 0.1,
                    "aesthetic_quality": 0.1,
                },
            ),
        )
    )

    assert score.score == pytest.approx(0.28)
    assert score.passed is False


def test_a_custom_scorer_may_be_sync_or_async_and_return_a_number_or_mapping() -> None:
    sync_score = _run(score_image(PNG_1X1, scorer=lambda image: 0.75))
    async_score = _run(
        score_image(
            PNG_1X1,
            scorer=lambda image: _value({"score": 0.4, "reason": "soft"}),
        )
    )

    assert sync_score.score == 0.75
    assert async_score.score == 0.4
    assert async_score.reason == "soft"
    assert async_score.passed is False


async def _value(value: Any) -> Any:
    return value


def test_a_missing_vision_provider_is_a_configuration_error() -> None:
    with pytest.raises(ConfigError, match="requires a vision provider"):
        _run(score_image(PNG_1X1))


def test_a_low_score_is_retried_up_to_the_configured_limit_then_accepted() -> None:
    generator = ScriptedGenerator(
        ["acme"],
        [_outcome("acme"), _outcome("acme")],
    )
    scorer = ScriptedScorer([0.2, 0.9])

    result = _run(
        ImageQualityAssurance(
            generator,
            scorer=scorer,
            config=QualityConfig(min_score=0.7, max_retries=2),
        ).run(ImageRequest(prompt=PROMPT))
    )

    assert isinstance(result, ImageQAResult)
    assert result.passed is True
    assert result.provider == "acme"
    assert len(result.attempts) == 2
    assert [attempt.passed for attempt in result.attempts] == [False, True]
    assert [attempt.retry for attempt in result.attempts] == [False, True]
    assert len(generator.calls) == 2
    assert all(call["no_cache"] is True for call in generator.calls)
    assert result.total_amount_usd == pytest.approx(0.04)


def test_a_provider_that_never_passes_falls_back_to_the_next_provider() -> None:
    generator = ScriptedGenerator(
        ["first", "second"],
        [_outcome("first"), _outcome("first"), _outcome("second")],
    )
    scorer = ScriptedScorer([0.1, 0.2, 0.9])

    result = _run(
        ImageQualityAssurance(
            generator,
            scorer=scorer,
            config=QualityConfig(min_score=0.7, max_retries=1),
        ).run(ImageRequest(prompt=PROMPT))
    )

    assert result.passed is True
    assert result.provider == "second"
    assert result.fallback == "provider"
    assert [attempt.provider for attempt in result.attempts] == [
        "first",
        "first",
        "second",
    ]
    assert [call["provider"] for call in generator.calls] == [
        "first",
        "first",
        "second",
    ]
    assert generator.calls[0]["exclude"] == set()
    assert generator.calls[1]["exclude"] == set()
    assert generator.calls[2]["exclude"] == {"first"}


def test_every_provider_failing_falls_back_to_stock() -> None:
    generator = ScriptedGenerator(["first"], [_outcome("first")])
    stock = _asset()

    result = _run(
        ImageQualityAssurance(
            generator,
            scorer=ScriptedScorer([0.1]),
            stock=stock,
            config=QualityConfig(min_score=0.7, max_retries=0),
        ).run(ImageRequest(prompt=PROMPT))
    )

    assert result.passed is True
    assert result.used_stock is True
    assert result.stock is stock
    assert result.provider == "pexels"
    assert result.fallback == "stock"
    assert result.outcome is None
    assert [attempt.fallback for attempt in result.attempts] == ["", "stock"]


def test_a_stock_provider_is_searched_with_the_request_shape() -> None:
    class FakeStock:
        def __init__(self) -> None:
            self.requests: list[Any] = []

        async def search(self, request: Any) -> StockResponse:
            self.requests.append(request)
            return StockResponse(query=request.query, assets=[_asset("found")])

    stock = FakeStock()
    result = _run(
        run_image_qa(
            ImageRequest(prompt=PROMPT, width=1920, height=1080),
            generate=ScriptedGenerator(["acme"], [_outcome("acme")]),
            scorer=ScriptedScorer([0.1]),
            stock=stock,
            config=QualityConfig(min_score=0.7, max_retries=0),
        )
    )

    assert result.stock is not None
    assert result.stock.id == "found"
    assert stock.requests[0].query == PROMPT
    assert stock.requests[0].min_width == 1920
    assert stock.requests[0].min_height == 1080


def test_disabling_quality_bypasses_the_vision_scorer() -> None:
    generator = ScriptedGenerator(["acme"], [_outcome("acme")])
    scorer = ScriptedScorer([0.0])

    result = _run(
        ImageQualityAssurance(
            generator,
            scorer=scorer,
            config=QualityConfig(enabled=False),
        ).run(ImageRequest(prompt=PROMPT))
    )

    assert result.passed is True
    assert result.attempts[0].score is None
    assert scorer.sources == []
    assert len(generator.calls) == 1


def test_an_exhausted_result_can_be_required_and_raises() -> None:
    result = _run(
        ImageQualityAssurance(
            ScriptedGenerator(["acme"], [_outcome("acme")]),
            scorer=ScriptedScorer([0.1]),
            config=QualityConfig(min_score=0.9, max_retries=0),
        ).run(ImageRequest(prompt=PROMPT))
    )

    assert result.passed is False
    assert result.failed is True
    assert result.score is not None
    assert result.score.overall == pytest.approx(0.1)
    with pytest.raises(ImageQualityError, match="quality threshold"):
        result.require()


def _router(*providers: Provider) -> Router:
    return Router(
        {provider.name: provider for provider in providers},
        config=CapabilityChainConfig(chain=[provider.name for provider in providers]),
    )


def test_the_image_service_excludes_a_provider_after_a_quality_failure() -> None:
    first = MockImageProvider(name="first", seed=1)
    second = MockImageProvider(name="second", seed=2)
    service = ImageService(_router(first, second))
    token = str(screen_prompt(PROMPT))

    result = _run(
        ImageQA(
            service,
            scorer=ScriptedScorer([0.1, 0.95]),
            config=QualityConfig(min_score=0.7, max_retries=0),
        ).run(ImageRequest(prompt=PROMPT), token=token)
    )

    assert result.passed is True
    assert result.provider == "second"
    assert len(first.calls) == 1
    assert len(second.calls) == 1
    assert result.attempts[0].provider == "first"
    assert result.attempts[1].provider == "second"
