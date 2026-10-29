"""Embedding adapters, batching, dimensions, cost tracking and caching."""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import Callable, Coroutine, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy import Engine, inspect

from aivia.core.config.capability import CapabilityChainConfig
from aivia.core.cost import CostTracker
from aivia.core.db import (
    EMBEDDING_CACHE_PROVIDER_INDEX_NAME,
    create_database_engine,
    upgrade_database,
)
from aivia.core.errors import (
    AuthError,
    ConfigError,
    InvalidRequestError,
    RateLimitError,
    RetryableError,
    ValidationError,
)
from aivia.core.errors import TimeoutError as AiviaTimeoutError
from aivia.core.models import ProviderMeta, ProviderPrice
from aivia.core.router import Router
from aivia.providers import (
    LocalEmbeddingProvider,
    OpenAIEmbeddingProvider,
    VoyageEmbeddingProvider,
    provider_registry,
)
from aivia.providers.base import Provider
from aivia.providers.embeddings import (
    EMBEDDING_OPERATION,
    CachedEmbeddingProvider,
    EmbeddingCache,
    EmbeddingCacheKey,
    EmbeddingCostCalculator,
    EmbeddingService,
    batch_inputs,
    normalize_vector,
)
from aivia.providers.protocols import (
    CAPABILITY_EMBEDDINGS,
    Embedding,
    EmbeddingInputType,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
)
from aivia.providers.protocols.embeddings import EmbeddingProvider

API_KEY = "test-key"
MODEL = "embed-model"


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine)


def _embedding_body(
    count: int,
    *,
    dimensions: int = 3,
    tokens: int = 6,
    model: str = MODEL,
    request_id: str = "req-1",
) -> dict[str, Any]:
    return {
        "id": request_id,
        "model": model,
        "data": [
            {
                "index": index,
                "embedding": [float(index + 1)] * dimensions,
            }
            for index in range(count)
        ],
        "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
    }


def _openai(
    handler: Callable[[httpx.Request], httpx.Response],
    **kwargs: Any,
) -> OpenAIEmbeddingProvider:
    return OpenAIEmbeddingProvider(
        api_key=API_KEY,
        transport=httpx.MockTransport(handler),
        default_model=MODEL,
        **kwargs,
    )


def _voyage(
    handler: Callable[[httpx.Request], httpx.Response],
    **kwargs: Any,
) -> VoyageEmbeddingProvider:
    return VoyageEmbeddingProvider(
        api_key=API_KEY,
        transport=httpx.MockTransport(handler),
        default_model=MODEL,
        **kwargs,
    )


def _capture(
    body: dict[str, Any] | Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    *,
    status: int = 200,
) -> tuple[list[dict[str, Any]], Callable[[httpx.Request], httpx.Response]]:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent = json.loads(request.content)
        bodies.append(sent)
        payload = body(sent) if callable(body) else body
        if payload is None:
            payload = _embedding_body(len(sent["input"]))
        return httpx.Response(status, json=payload)

    return bodies, handler


def test_the_builtin_embedding_adapters_are_registered() -> None:
    assert provider_registry.resolve("openai-embeddings") is OpenAIEmbeddingProvider
    assert provider_registry.resolve("voyage") is VoyageEmbeddingProvider
    assert provider_registry.resolve("local-embeddings") is LocalEmbeddingProvider
    assert OpenAIEmbeddingProvider.default_metadata is not None
    assert VoyageEmbeddingProvider.default_metadata is not None
    assert LocalEmbeddingProvider.default_metadata is not None


def test_openai_batches_inputs_and_sums_usage() -> None:
    def body(sent: dict[str, Any]) -> dict[str, Any]:
        return _embedding_body(
            len(sent["input"]),
            tokens=len(sent["input"]) * 3,
        )

    bodies, handler = _capture(body)
    provider = _openai(handler, dimensions=3, batch_size=2)

    response = _run(provider.embed(EmbeddingRequest(inputs=["a", "b", "c", "d", "e"])))

    assert [len(body["input"]) for body in bodies] == [2, 2, 1]
    assert all(body["model"] == MODEL for body in bodies)
    assert all(body["dimensions"] == 3 for body in bodies)
    assert [embedding.index for embedding in response.embeddings] == [0, 1, 2, 3, 4]
    assert response.usage.input_tokens == 15
    assert response.model == MODEL
    assert response.provider == "openai-embeddings"
    assert response.metadata["batches"] == 3


def test_openai_uses_the_request_model_and_dimension_override() -> None:
    def body(sent: dict[str, Any]) -> dict[str, Any]:
        return _embedding_body(
            len(sent["input"]),
            dimensions=sent.get("dimensions", 3),
            model=sent["model"],
        )

    bodies, handler = _capture(body)
    provider = _openai(handler, dimensions=3)

    response = _run(
        provider.embed(
            EmbeddingRequest(inputs=["a"], model="other-model", dimensions=8)
        )
    )

    assert bodies[0]["model"] == "other-model"
    assert bodies[0]["dimensions"] == 8
    assert response.dimensions == 8


def test_openai_rejects_a_response_with_the_wrong_dimension() -> None:
    _, handler = _capture(_embedding_body(1, dimensions=3))
    provider = _openai(handler, dimensions=3)

    with pytest.raises(ValidationError):
        _run(provider.embed(EmbeddingRequest(inputs=["a"], dimensions=4)))


def test_openai_requires_a_model() -> None:
    provider = OpenAIEmbeddingProvider(
        api_key=API_KEY,
        transport=httpx.MockTransport(lambda request: httpx.Response(200)),
    )

    with pytest.raises(InvalidRequestError):
        _run(provider.embed(EmbeddingRequest(inputs=["a"])))


@pytest.mark.parametrize(
    ("status", "payload", "expected"),
    [
        (401, {"error": {"type": "invalid_api_key"}}, AuthError),
        (429, {"error": {"type": "rate_limit_error"}}, RateLimitError),
        (500, {"error": {"type": "server_error"}}, RetryableError),
        (400, {"error": {"type": "invalid_request_error"}}, InvalidRequestError),
    ],
)
def test_openai_maps_errors_to_the_shared_taxonomy(
    status: int,
    payload: dict[str, Any],
    expected: type[Exception],
) -> None:
    provider = _openai(
        lambda request: httpx.Response(status, json=payload),
        dimensions=3,
    )

    with pytest.raises(expected):
        _run(provider.embed(EmbeddingRequest(inputs=["a"])))


def test_openai_rejects_a_malformed_response() -> None:
    provider = _openai(
        lambda request: httpx.Response(200, json={"data": []}),
        dimensions=3,
    )

    with pytest.raises(ValidationError):
        _run(provider.embed(EmbeddingRequest(inputs=["a"])))


def test_voyage_maps_input_type_dimension_and_truncation() -> None:
    def body(sent: dict[str, Any]) -> dict[str, Any]:
        return {
            "model": sent["model"],
            "data": [
                {
                    "index": index,
                    "embedding": [float(index + 1)] * sent["output_dimension"],
                }
                for index in range(len(sent["input"]))
            ],
            "usage": {"total_tokens": len(sent["input"]) * 2},
        }

    bodies, handler = _capture(body)
    provider = _voyage(handler, dimensions=4, batch_size=2)

    response = _run(
        provider.embed(
            EmbeddingRequest(
                inputs=["first", "second", "query"],
                model="voyage-large",
                input_type=EmbeddingInputType.QUERY,
                truncate=False,
            )
        )
    )

    assert [len(body["input"]) for body in bodies] == [2, 1]
    assert all(body["model"] == "voyage-large" for body in bodies)
    assert all(body["input_type"] == "query" for body in bodies)
    assert all(body["output_dimension"] == 4 for body in bodies)
    assert all(body["truncation"] is False for body in bodies)
    assert response.usage.input_tokens == 6
    assert response.dimensions == 4
    assert response.provider == "voyage"


def test_voyage_maps_document_input_and_error_status() -> None:
    bodies, handler = _capture(
        lambda sent: {
            "model": sent["model"],
            "data": [{"index": 0, "embedding": [1.0, 2.0, 3.0]}],
            "usage": {"total_tokens": 1},
        }
    )
    provider = _voyage(handler)

    _run(
        provider.embed(
            EmbeddingRequest(
                inputs=["stored"],
                input_type=EmbeddingInputType.DOCUMENT,
            )
        )
    )

    assert bodies[0]["input_type"] == "document"
    assert bodies[0]["truncation"] is True

    failing = _voyage(lambda request: httpx.Response(429, json={"detail": "slow"}))
    with pytest.raises(RateLimitError):
        _run(failing.embed(EmbeddingRequest(inputs=["a"])))


def test_local_embeddings_are_deterministic_and_normalized() -> None:
    provider = LocalEmbeddingProvider(dimensions=32, batch_size=2)

    response = _run(
        provider.embed(
            EmbeddingRequest(
                inputs=["alpha beta", "alpha beta", "gamma delta"],
                normalize=True,
            )
        )
    )

    assert response.dimensions == 32
    assert response.model == "local-hash"
    assert [embedding.index for embedding in response.embeddings] == [0, 1, 2]
    assert response.embeddings[0].vector == response.embeddings[1].vector
    assert all(
        math.isclose(
            math.sqrt(sum(value * value for value in embedding.vector)),
            1.0,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        for embedding in response.embeddings
    )
    assert response.usage.input_tokens > 0

    again = _run(provider.embed(EmbeddingRequest(inputs=["alpha beta"])))
    assert again.embeddings[0].vector == response.embeddings[0].vector


def test_local_embeddings_use_the_requested_dimension() -> None:
    provider = LocalEmbeddingProvider(dimensions=32)

    response = _run(provider.embed(EmbeddingRequest(inputs=["a b c"], dimensions=8)))

    assert response.dimensions == 8


def test_local_similar_text_is_closer_than_unrelated_text() -> None:
    provider = LocalEmbeddingProvider(dimensions=128)

    response = _run(
        provider.embed(
            EmbeddingRequest(inputs=["alpha beta gamma", "alpha beta delta", "x y z"])
        )
    )
    first, similar, unrelated = (embedding.vector for embedding in response.embeddings)

    assert sum(a * b for a, b in zip(first, similar, strict=False)) > sum(
        a * b for a, b in zip(first, unrelated, strict=False)
    )


def test_local_accepts_an_injected_encoder() -> None:
    def encoder(texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0, 0.0] for text in texts]

    provider = LocalEmbeddingProvider(encoder=encoder, dimensions=3)

    response = _run(provider.embed(EmbeddingRequest(inputs=["abcd"])))

    assert response.embeddings[0].vector == [4.0, 1.0, 0.0]


def test_batch_inputs_and_normalize_vector_helpers() -> None:
    assert batch_inputs(["a", "b", "c"], 2) == [["a", "b"], ["c"]]
    assert normalize_vector([3.0, 4.0]) == [0.6, 0.8]
    assert normalize_vector([0.0, 0.0]) == [0.0, 0.0]
    with pytest.raises(ValueError):
        batch_inputs(["a"], 0)


class FakeEmbeddingProvider(Provider):
    """A provider with configurable vectors, usage, price and failure."""

    def __init__(
        self,
        name: str = "fake",
        *,
        model: str = MODEL,
        dimensions: int = 3,
        error: Exception | None = None,
        price: float = 0.5,
    ) -> None:
        super().__init__(
            ProviderMeta(
                name=name,
                capabilities=[CAPABILITY_EMBEDDINGS],
                pricing={
                    CAPABILITY_EMBEDDINGS: ProviderPrice(
                        amount_usd=price,
                        unit="million_tokens",
                    )
                },
            )
        )
        self.default_model = model
        self.default_dimensions = dimensions
        self.error = error
        self.requests: list[EmbeddingRequest] = []

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return EmbeddingResponse(
            embeddings=[
                Embedding(
                    index=index,
                    vector=[float(len(text)), float(index), 1.0],
                )
                for index, text in enumerate(request.inputs)
            ],
            model=request.model or self.default_model,
            provider=self.name,
            usage=EmbeddingUsage(input_tokens=len(request.inputs) * 100),
        )


@pytest.fixture
def engine() -> Iterator[Engine]:
    created = create_database_engine(url="sqlite+pysqlite:///:memory:")
    upgrade_database(engine=created)
    try:
        yield created
    finally:
        created.dispose()


def test_cost_uses_the_explicit_model_price(engine: Engine) -> None:
    tracker = CostTracker(engine)
    provider = FakeEmbeddingProvider()
    response = _run(provider.embed(EmbeddingRequest(inputs=["a", "b"])))
    calculator = EmbeddingCostCalculator({MODEL: 2.0})

    event = calculator.record(
        provider,
        response,
        tracker,
        run_id="run_1",
        metadata={"stage": "research"},
    )

    assert event.provider == "fake"
    assert event.model == MODEL
    assert event.operation == EMBEDDING_OPERATION
    assert event.units == 200.0
    assert event.unit_kind == "token"
    assert event.amount_usd == pytest.approx(200 * 2.0 / 1_000_000)
    assert event.run_id == "run_1"
    assert event.metadata["input_tokens"] == 200
    assert event.metadata["stage"] == "research"
    assert tracker.total() == pytest.approx(event.amount_usd)


def test_cost_falls_back_to_provider_metadata() -> None:
    provider = FakeEmbeddingProvider(price=0.5)
    response = _run(provider.embed(EmbeddingRequest(inputs=["a"])))
    calculator = EmbeddingCostCalculator()

    event = calculator.cost_event(provider, response)

    assert event.amount_usd == pytest.approx(100 * 0.5 / 1_000_000)


def test_a_cache_key_changes_with_every_embedded_input_dimension() -> None:
    base = EmbeddingCacheKey(
        provider="p",
        model="m",
        text="hello",
        dimensions=3,
        input_type="query",
        normalize=False,
        truncate=True,
    )

    assert base.digest == base.cache_key
    assert len(base.digest) == 64
    for changed in (
        EmbeddingCacheKey("q", "m", "hello", 3, "query", False, True),
        EmbeddingCacheKey("p", "n", "hello", 3, "query", False, True),
        EmbeddingCacheKey("p", "m", "other", 3, "query", False, True),
        EmbeddingCacheKey("p", "m", "hello", 4, "query", False, True),
        EmbeddingCacheKey("p", "m", "hello", 3, "document", False, True),
        EmbeddingCacheKey("p", "m", "hello", 3, "query", True, True),
        EmbeddingCacheKey("p", "m", "hello", 3, "query", False, False),
    ):
        assert changed.digest != base.digest


def test_the_cache_key_uses_the_provider_default_dimension() -> None:
    provider = LocalEmbeddingProvider(dimensions=8)
    cached = CachedEmbeddingProvider(provider, EmbeddingCache())

    key = cached.key_for(EmbeddingRequest(inputs=["a"]), "a")

    assert key.dimensions == 8


def test_the_cache_serves_a_repeated_request_without_calling_the_provider() -> None:
    provider = FakeEmbeddingProvider()
    cache = EmbeddingCache()
    cached = CachedEmbeddingProvider(provider, cache)
    request = EmbeddingRequest(inputs=["a", "b"])

    first = _run(cached.embed(request))
    second = _run(cached.embed(request))

    assert len(provider.requests) == 1
    assert first.metadata["cache"] == "miss"
    assert first.metadata["cache_misses"] == 2
    assert second.metadata["cache"] == "hit"
    assert second.metadata["cache_hits"] == 2
    assert second.usage.input_tokens == 0
    assert second.vectors == first.vectors
    assert cache.stats().hits == 2
    assert cache.stats().misses == 2
    assert len(cache) == 2


def test_the_cache_only_sends_missing_inputs_and_preserves_order() -> None:
    provider = FakeEmbeddingProvider()
    cache = EmbeddingCache()
    cached = CachedEmbeddingProvider(provider, cache)

    first = _run(cached.embed(EmbeddingRequest(inputs=["a", "b"])))
    second = _run(cached.embed(EmbeddingRequest(inputs=["a", "c"])))

    assert len(provider.requests) == 2
    assert provider.requests[1].inputs == ["c"]
    assert second.metadata["cache"] == "partial"
    assert second.metadata["cache_hits"] == 1
    assert second.metadata["cache_misses"] == 1
    assert [embedding.index for embedding in second.embeddings] == [0, 1]
    assert second.embeddings[0].vector == first.embeddings[0].vector
    assert second.usage.input_tokens == 100


def test_the_cache_can_be_disabled_and_is_bounded() -> None:
    provider = FakeEmbeddingProvider()
    cached = CachedEmbeddingProvider(
        provider,
        EmbeddingCache(max_entries=1),
        no_cache=True,
    )

    _run(cached.embed(EmbeddingRequest(inputs=["a"])))
    _run(cached.embed(EmbeddingRequest(inputs=["a"])))

    assert len(provider.requests) == 2
    assert len(cached.cache) == 0


def test_the_cache_survives_a_new_cache_instance(engine: Engine) -> None:
    provider = FakeEmbeddingProvider()
    first = CachedEmbeddingProvider(provider, EmbeddingCache(engine=engine))
    request = EmbeddingRequest(inputs=["a", "b"])

    stored = _run(first.embed(request))
    restarted = CachedEmbeddingProvider(provider, EmbeddingCache(engine=engine))
    served = _run(restarted.embed(request))

    assert len(provider.requests) == 1
    assert served.metadata["cache"] == "hit"
    assert served.metadata["cache_hits"] == 2
    assert served.usage.input_tokens == 0
    assert served.vectors == stored.vectors


def test_a_persistent_cache_serves_a_vector_after_a_restart(engine: Engine) -> None:
    key = EmbeddingCacheKey("fake", MODEL, "hello", 3, "", False, True)
    first = EmbeddingCache(engine=engine)

    first.put(key, [1.0, 2.0, 3.0])

    restarted = EmbeddingCache(engine=engine)
    assert restarted.persistent is True
    assert restarted.repository is not None
    assert restarted.get(key) == [1.0, 2.0, 3.0]
    assert len(restarted) == 1
    assert restarted.stats().entries == 1


def test_a_file_database_keeps_vectors_across_an_engine_reopen(
    tmp_path: Path,
) -> None:
    database = tmp_path / "aivia.db"
    key = EmbeddingCacheKey("fake", MODEL, "hello", 3)
    first_engine = create_database_engine(database)
    upgrade_database(engine=first_engine)
    try:
        EmbeddingCache(engine=first_engine).put(key, [4.0, 5.0])
    finally:
        first_engine.dispose()

    second_engine = create_database_engine(database)
    try:
        assert EmbeddingCache(engine=second_engine).get(key) == [4.0, 5.0]
    finally:
        second_engine.dispose()


def test_a_persistent_cache_keeps_only_the_most_recent_vectors(engine: Engine) -> None:
    first = EmbeddingCache(max_entries=2, engine=engine)
    keys = [EmbeddingCacheKey("fake", MODEL, text, 3) for text in ("a", "b", "c")]

    for index, key in enumerate(keys):
        first.put(key, [float(index)])
        time.sleep(0.01)

    restarted = EmbeddingCache(max_entries=2, engine=engine)
    assert len(restarted) == 2
    assert restarted.get(keys[0]) is None
    assert restarted.get(keys[1]) == [1.0]
    assert restarted.get(keys[2]) == [2.0]


def test_a_disabled_persistent_cache_writes_nothing(engine: Engine) -> None:
    key = EmbeddingCacheKey("fake", MODEL, "hello", 3)
    disabled = EmbeddingCache(engine=engine, no_cache=True)

    disabled.put(key, [1.0, 2.0, 3.0])

    assert len(disabled) == 0
    assert EmbeddingCache(engine=engine).get(key) is None


def test_the_embedding_cache_table_carries_a_provider_index(
    engine: Engine,
) -> None:
    cache = EmbeddingCache(engine=engine)
    key = EmbeddingCacheKey("fake", MODEL, "hello", 3)
    cache.put(key, [1.0, 2.0, 3.0])

    columns = {
        column["name"] for column in inspect(engine).get_columns("embedding_cache")
    }
    indexes = {
        index["name"] for index in inspect(engine).get_indexes("embedding_cache")
    }
    assert {
        "cache_key",
        "provider",
        "model",
        "text",
        "dimensions",
        "input_type",
        "normalize",
        "truncate",
        "vector",
        "hit_count",
    } <= columns
    assert EMBEDDING_CACHE_PROVIDER_INDEX_NAME in indexes


def test_the_service_routes_caches_and_records_the_actual_cost(
    engine: Engine,
) -> None:
    tracker = CostTracker(engine)
    provider = FakeEmbeddingProvider(price=0.5)
    router = Router(
        {"fake": provider},
        config=CapabilityChainConfig(chain=["fake"]),
    )
    service = EmbeddingService(router, tracker=tracker, cache=EmbeddingCache())

    response = _run(
        service.embed(
            EmbeddingRequest(inputs=["a", "b"]),
            run_id="run_1",
            metadata={"stage": "research"},
        )
    )

    assert len(provider.requests) == 1
    assert response.provider == "fake"
    event = service.last_event
    assert event is not None
    assert event.provider == "fake"
    assert event.operation == EMBEDDING_OPERATION
    assert event.unit_kind == "token"
    assert event.units == 200.0
    assert event.amount_usd == pytest.approx(200 * 0.5 / 1_000_000)
    assert event.run_id == "run_1"
    assert event.metadata["cache"] == "miss"
    assert event.metadata["stage"] == "research"
    assert tracker.total() == pytest.approx(event.amount_usd)

    second = _run(service.embed(EmbeddingRequest(inputs=["a", "b"])))
    assert len(provider.requests) == 1
    assert second.usage.input_tokens == 0
    assert service.last_event is not None
    assert service.last_event.amount_usd == 0.0
    assert service.last_event.metadata["cache"] == "hit"


def test_the_service_falls_back_and_prices_the_provider_that_served_it(
    engine: Engine,
) -> None:
    tracker = CostTracker(engine)
    failing = FakeEmbeddingProvider("failing", error=AiviaTimeoutError("slow"))
    serving = FakeEmbeddingProvider("serving", price=0.25)
    router = Router(
        {"failing": failing, "serving": serving},
        config=CapabilityChainConfig(chain=["failing", "serving"]),
    )
    service = EmbeddingService(router, tracker=tracker)

    response = _run(service.embed(EmbeddingRequest(inputs=["a"])))

    assert len(failing.requests) == 1
    assert len(serving.requests) == 1
    assert response.provider == "serving"
    assert service.last_selection is not None
    assert service.last_selection.name == "serving"
    assert service.last_event is not None
    assert service.last_event.provider == "serving"
    assert service.last_event.amount_usd == pytest.approx(100 * 0.25 / 1_000_000)


def test_the_service_uses_router_pricing_overrides(engine: Engine) -> None:
    tracker = CostTracker(engine)
    provider = FakeEmbeddingProvider(price=0.5)
    router = Router(
        {"fake": provider},
        config=CapabilityChainConfig(
            chain=["fake"],
            pricing_overrides={"fake": 1.0},
        ),
    )
    service = EmbeddingService(router, tracker=tracker)

    _run(service.embed(EmbeddingRequest(inputs=["a"])))

    assert service.last_event is not None
    assert service.last_event.amount_usd == pytest.approx(100 * 1.0 / 1_000_000)


def test_a_service_without_a_tracker_still_returns_vectors() -> None:
    provider = FakeEmbeddingProvider()
    router = Router(
        {"fake": provider},
        config=CapabilityChainConfig(chain=["fake"]),
    )
    service = EmbeddingService(router)

    response = _run(service.embed(EmbeddingRequest(inputs=["a", "b"])))

    assert response.dimensions == 3
    assert service.last_event is None
    assert service.last_price is None


def test_embedding_protocol_is_satisfied_by_the_builtin_adapters() -> None:
    provider = LocalEmbeddingProvider(dimensions=4)
    assert isinstance(provider, EmbeddingProvider)


def test_invalid_provider_configuration_is_rejected() -> None:
    with pytest.raises(ConfigError):
        OpenAIEmbeddingProvider(base_url="not a url", api_key=API_KEY)
    with pytest.raises(ConfigError):
        VoyageEmbeddingProvider(api_key="")
    with pytest.raises(ValueError):
        LocalEmbeddingProvider(default_model="")
