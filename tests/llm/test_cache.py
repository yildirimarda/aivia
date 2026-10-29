"""LLM response cache keys, hits, zero cost and the no cache flag."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine, Iterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, Field
from sqlalchemy import Engine, inspect

from aivia.core.config.llm import LLMConfig
from aivia.core.cost import CostTracker
from aivia.core.db import (
    LLM_CACHE_PROVIDER_INDEX_NAME,
    LLMCacheRepository,
    create_database_engine,
    upgrade_database,
)
from aivia.llm import (
    CACHE_HIT_OPERATION,
    CachedLLMProvider,
    LLMCache,
    LLMCacheKey,
    StructuredOutputHelper,
    llm_cache_key,
)
from aivia.providers.protocols.llm import (
    ChatMessage,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    MessageRole,
)

MODEL = "model-a"
PROVIDER = "acme"


class Item(BaseModel):
    title: str = Field(min_length=1)
    count: int = Field(ge=0)


class FakeProvider:
    """An async LLM provider that returns one canned response."""

    name = PROVIDER

    def __init__(self, content: str = '{"title": "hello", "count": 2}') -> None:
        self.content = content
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(
            content=self.content,
            model=MODEL,
            provider=PROVIDER,
            usage=LLMUsage(input_tokens=1_000, output_tokens=500),
        )


@pytest.fixture
def engine() -> Iterator[Engine]:
    created = create_database_engine(url="sqlite+pysqlite:///:memory:")
    upgrade_database(engine=created)
    try:
        yield created
    finally:
        created.dispose()


@pytest.fixture
def cache(engine: Engine) -> LLMCache:
    return LLMCache(engine)


def _messages(text: str = "make an item") -> list[ChatMessage]:
    return [ChatMessage(role=MessageRole.USER, content=text)]


def _response(content: str = '{"title": "hello", "count": 2}') -> LLMResponse:
    return LLMResponse(
        content=content,
        model=MODEL,
        provider=PROVIDER,
        usage=LLMUsage(input_tokens=1_000, output_tokens=500),
    )


def _request(text: str = "make an item") -> LLMRequest:
    return LLMRequest(messages=_messages(text), model=MODEL, prompt_version="v1")


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine)


def _config() -> LLMConfig:
    return LLMConfig.from_mapping(
        {
            "pricing": {
                MODEL: {
                    "input_per_million": 3.0,
                    "output_per_million": 15.0,
                }
            }
        }
    )


def test_the_key_is_stable_across_equivalent_inputs() -> None:
    first = llm_cache_key(
        " acme ",
        " model-a ",
        " v1 ",
        messages=[
            ChatMessage(role=MessageRole.SYSTEM, content=" be brief "),
            ChatMessage(role=MessageRole.USER, content=" make an item "),
        ],
        parameters={"temperature": 0.2, "max_tokens": 500},
    )
    second = LLMCacheKey(
        provider=PROVIDER,
        model=MODEL,
        prompt_version="v1",
        messages=[
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "make an item"},
        ],
        parameters={"max_tokens": 500, "temperature": 0.2},
    )

    assert first == second
    assert first.digest == second.digest
    assert len(first.digest) == 64
    assert first.digest == first.digest.lower()


def test_every_request_dimension_changes_the_key() -> None:
    base = llm_cache_key(
        PROVIDER,
        MODEL,
        "v1",
        _messages(),
        parameters={"temperature": 0.2},
    )
    same = llm_cache_key(
        PROVIDER,
        MODEL,
        "v1",
        _messages(),
        parameters={"temperature": 0.2},
    )

    assert base.digest == same.digest
    for changed in (
        llm_cache_key(
            "other", MODEL, "v1", _messages(), parameters={"temperature": 0.2}
        ),
        llm_cache_key(
            PROVIDER, "model-b", "v1", _messages(), parameters={"temperature": 0.2}
        ),
        llm_cache_key(
            PROVIDER, MODEL, "v2", _messages(), parameters={"temperature": 0.2}
        ),
        llm_cache_key(
            PROVIDER, MODEL, "v1", _messages("another"), parameters={"temperature": 0.2}
        ),
        llm_cache_key(
            PROVIDER, MODEL, "v1", _messages(), parameters={"temperature": 0.3}
        ),
    ):
        assert changed.digest != base.digest


def test_a_miss_returns_none_and_records_nothing(cache: LLMCache) -> None:
    key = cache.key(PROVIDER, MODEL, "v1", _messages())

    assert cache.get(key) is None
    assert cache.lookup(key) is None
    assert cache.cost.count() == 0


def test_storing_then_looking_up_records_a_zero_cost_hit(
    cache: LLMCache,
) -> None:
    key = cache.key(
        PROVIDER,
        MODEL,
        "v1",
        _messages(),
        parameters={"temperature": 0.2},
    )
    response = _response()
    stored = cache.store(response, key)

    assert stored.provider == PROVIDER
    assert stored.model == MODEL
    assert stored.prompt_version == "v1"
    assert stored.messages == [
        message.model_dump(mode="json", exclude_none=True) for message in _messages()
    ]
    assert stored.parameters == {"temperature": 0.2}
    assert stored.response == response.model_dump(mode="json")
    assert cache.get(key) == response

    hit = cache.lookup(key, run_id="run_2")

    assert hit == response
    events = cache.cost.events()
    assert len(events) == 1
    event = events[0]
    assert event.provider == PROVIDER
    assert event.model == MODEL
    assert event.operation == CACHE_HIT_OPERATION
    assert event.amount_usd == 0.0
    assert event.units == 0.0
    assert event.estimated is False
    assert event.run_id == "run_2"
    assert event.metadata == {
        "cache": "hit",
        "cache_key": key.digest,
        "prompt_version": "v1",
    }
    assert cache.cost.total() == 0.0
    assert cache.cost.count() == 1

    entry = LLMCacheRepository(cache.repository.engine).get(key.digest)
    assert entry is not None
    assert entry.hit_count == 1


def test_concurrent_hits_do_not_lose_increments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = create_database_engine(tmp_path / "aivia.db")
    upgrade_database(engine=engine)
    try:
        cache = LLMCache(engine)
        key = cache.key(PROVIDER, MODEL, "v1", _messages())
        response = _response()
        cache.store(response, key)

        workers = 8
        barrier = threading.Barrier(workers)
        original_get = cache.repository.get

        def synchronized_get(*args: Any, **kwargs: Any) -> Any:
            entry = original_get(*args, **kwargs)
            barrier.wait(timeout=5)
            return entry

        monkeypatch.setattr(cache.repository, "get", synchronized_get)
        errors: list[BaseException] = []
        lock = threading.Lock()

        def hit() -> None:
            try:
                cache.record_hit(key, response)
            except BaseException as exc:
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=hit) for _ in range(workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)

        assert not any(thread.is_alive() for thread in threads)
        assert errors == []
        entry = LLMCacheRepository(engine).get(key.digest)
        assert entry is not None
        assert entry.hit_count == workers
        assert cache.cost.count() == workers
    finally:
        engine.dispose()


def test_a_hit_can_skip_cost_recording(cache: LLMCache) -> None:
    key = cache.key(PROVIDER, MODEL, "v1", _messages())
    cache.store(_response(), key)

    assert cache.lookup(key, record_hit=False) is not None
    assert cache.cost.count() == 0
    entry = cache.entry(key)
    assert entry is not None
    assert entry.hit_count == 0


def test_no_cache_turns_every_lookup_into_a_miss(engine: Engine) -> None:
    cache = LLMCache(engine, no_cache=True)
    key = cache.key(PROVIDER, MODEL, "v1", _messages())
    cache.store(_response(), key)

    assert cache.get(key) is None
    assert cache.lookup(key) is None
    assert cache.cost.count() == 0
    assert cache.repository.list() == []


def test_storing_an_equivalent_response_twice_returns_the_first_entry(
    cache: LLMCache,
) -> None:
    key = cache.key(PROVIDER, MODEL, "v1", _messages())
    first = cache.store(_response(), key)
    second = cache.store(_response("different"), key)

    assert second.cache_key == first.cache_key
    assert second.response == first.response
    assert len(cache.repository.list()) == 1


def test_a_store_can_derive_the_key_from_a_request(cache: LLMCache) -> None:
    provider = FakeProvider()
    request = _request()

    stored = cache.store(_response(), request=request, provider=provider)

    assert stored.cache_key == cache.key_for_request(provider, request).digest
    assert cache.get(cache.key_for_request(provider, request)) == _response()


def test_the_cache_survives_a_new_cache_instance(engine: Engine) -> None:
    first = LLMCache(engine)
    key = first.key(PROVIDER, MODEL, "v1", _messages())
    stored = first.store(_response(), key)

    second = LLMCache(engine)

    assert second.entry(key) == stored
    assert second.lookup(key) == _response()
    assert second.cost.count() == 1


def test_the_cache_table_carries_a_provider_index(engine: Engine) -> None:
    columns = {column["name"] for column in inspect(engine).get_columns("llm_cache")}
    indexes = {index["name"] for index in inspect(engine).get_indexes("llm_cache")}

    assert {
        "cache_key",
        "provider",
        "model",
        "prompt_version",
        "messages",
        "parameters",
        "response",
        "hit_count",
    } <= columns
    assert LLM_CACHE_PROVIDER_INDEX_NAME in indexes


def test_the_structured_helper_serves_the_second_identical_call_from_cache(
    engine: Engine,
) -> None:
    provider = FakeProvider()
    tracker = CostTracker(engine)
    cache = LLMCache(engine)
    helper = StructuredOutputHelper(
        provider,
        Item,
        config=_config(),
        tracker=tracker,
        cache=cache,
        run_id="run_1",
    )

    first = _run(helper.complete(_request()))
    second = _run(helper.complete(_request()))

    assert first.cache_hit is False
    assert first.cache_key is not None
    assert first.amount_usd == pytest.approx(0.0105)
    assert second.cache_hit is True
    assert second.cache_key == first.cache_key
    assert second.value == first.value
    assert second.amount_usd == 0.0
    assert len(provider.requests) == 1
    assert tracker.count() == 2
    assert tracker.total() == pytest.approx(0.0105)


def test_the_helper_honours_no_cache(engine: Engine) -> None:
    provider = FakeProvider()
    cache = LLMCache(engine)
    helper = StructuredOutputHelper(provider, Item, cache=cache, no_cache=True)

    _run(helper.complete(_request()))
    _run(helper.complete(_request()))

    assert len(provider.requests) == 2
    assert cache.repository.list() == []


def test_a_cached_provider_wrapper_avoids_the_second_call(cache: LLMCache) -> None:
    provider = FakeProvider()
    wrapped = CachedLLMProvider(provider, cache, prompt_version="v1")
    request = _request()

    first = _run(wrapped.complete(request))
    second = _run(wrapped.complete(request))

    assert first == second
    assert len(provider.requests) == 1
    assert cache.cost.count() == 1
    assert cache.cost.total() == 0.0
