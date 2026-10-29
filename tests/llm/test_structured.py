"""Structured output validation with token and cost recording."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine, Iterator
from typing import Any

import pytest
from pydantic import BaseModel, Field
from sqlalchemy import Engine

from aivia.core.config.llm import LLMConfig
from aivia.core.cost import CostTracker
from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.errors import ConfigError
from aivia.llm import (
    StructuredOutput,
    StructuredOutputError,
    StructuredOutputHelper,
    StructuredOutputResult,
    complete_structured,
    parse_structured,
)
from aivia.providers.protocols.llm import (
    ChatMessage,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    MessageRole,
)

MODEL = "model-a"


class Item(BaseModel):
    title: str = Field(min_length=1)
    count: int = Field(ge=0)


class FakeProvider:
    """An async LLM provider that returns one canned response."""

    def __init__(
        self,
        content: str = '{"title": "hello", "count": 2}',
        *,
        error: Exception | None = None,
    ) -> None:
        self.content = content
        self.error = error
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return LLMResponse(
            content=self.content,
            model=MODEL,
            provider="acme",
            usage=LLMUsage(
                input_tokens=1_000,
                output_tokens=500,
                cached_input_tokens=100,
            ),
        )


@pytest.fixture
def engine() -> Iterator[Engine]:
    created = create_database_engine(url="sqlite+pysqlite:///:memory:")
    upgrade_database(engine=created)
    try:
        yield created
    finally:
        created.dispose()


def _config(
    *,
    input_per_million: float = 3.0,
    output_per_million: float = 15.0,
) -> LLMConfig:
    return LLMConfig.from_mapping(
        {
            "pricing": {
                MODEL: {
                    "input_per_million": input_per_million,
                    "output_per_million": output_per_million,
                }
            }
        }
    )


def _request(text: str = "make an item") -> LLMRequest:
    return LLMRequest(
        messages=[ChatMessage(role=MessageRole.USER, content=text)],
        model=MODEL,
    )


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine)


def test_a_response_is_validated_and_its_tokens_and_cost_are_recorded(
    engine: Engine,
) -> None:
    provider = FakeProvider()
    tracker = CostTracker(engine)
    helper = StructuredOutputHelper(
        provider,
        Item,
        config=_config(),
        tracker=tracker,
        pricing_version="2026-01",
        run_id="run_1",
        metadata={"stage": "write"},
    )

    result = _run(helper.complete(_request()))

    assert isinstance(result, StructuredOutputResult)
    assert result.value == Item(title="hello", count=2)
    assert result.output == result.value
    assert result.data == result.value
    assert result.provider == "acme"
    assert result.model == MODEL
    assert result.input_tokens == 1_000
    assert result.output_tokens == 500
    assert result.total_tokens == 1_500
    assert result.cost is not None
    assert result.amount_usd == pytest.approx(0.0105)
    assert result.cost_usd == pytest.approx(0.0105)
    assert result.cost.pricing_version == "2026-01"
    assert result.event is result.cost_event
    assert result.event is not None
    assert result.event.run_id == "run_1"
    assert result.event.operation == "llm"
    assert result.event.metadata == {
        "input_tokens": 1_000,
        "output_tokens": 500,
        "cached_input_tokens": 100,
        "input_cost_usd": pytest.approx(0.003),
        "output_cost_usd": pytest.approx(0.0075),
        "stage": "write",
    }
    assert tracker.count() == 1
    assert tracker.total() == pytest.approx(0.0105)


def test_the_request_is_given_the_pydantic_schema_as_a_json_schema() -> None:
    provider = FakeProvider()
    helper = StructuredOutputHelper(provider, Item)

    _run(helper.complete(_request()))

    assert provider.requests[0].response_schema == Item.model_json_schema()


def test_an_explicit_request_schema_is_not_overwritten() -> None:
    provider = FakeProvider()
    helper = StructuredOutputHelper(provider, Item)
    request = _request().model_copy(
        update={"response_schema": {"type": "object", "title": "Narrow"}}
    )

    _run(helper.complete(request))

    assert provider.requests[0].response_schema == {
        "type": "object",
        "title": "Narrow",
    }


def test_invalid_json_raises_a_structured_output_error_and_still_records_cost(
    engine: Engine,
) -> None:
    tracker = CostTracker(engine)
    helper = StructuredOutputHelper(
        FakeProvider("not json"),
        Item,
        config=_config(),
        tracker=tracker,
    )

    with pytest.raises(StructuredOutputError) as raised:
        _run(helper.complete(_request()))

    assert raised.value.code == "structured_output"
    assert raised.value.content == "not json"
    assert raised.value.details["content"] == "not json"
    assert tracker.count() == 1
    assert tracker.total() == pytest.approx(0.0105)


def test_a_schema_mismatch_reports_the_pydantic_errors(engine: Engine) -> None:
    tracker = CostTracker(engine)
    helper = StructuredOutputHelper(
        FakeProvider('{"title": "", "count": -1}'),
        Item,
        config=_config(),
        tracker=tracker,
    )

    with pytest.raises(StructuredOutputError) as raised:
        _run(helper.complete(_request()))

    assert raised.value.errors
    assert {error["loc"] for error in raised.value.errors} == {
        ("title",),
        ("count",),
    }
    assert tracker.count() == 1


def test_json_inside_a_markdown_fence_is_accepted() -> None:
    provider = FakeProvider('```json\n{"title": "fenced", "count": 1}\n```')

    result = _run(complete_structured(provider, _request(), Item))

    assert result.value == Item(title="fenced", count=1)


def test_json_surrounded_by_prose_is_accepted() -> None:
    provider = FakeProvider('Here it is: {"title": "prose", "count": 0} thanks')

    result = _run(complete_structured(provider, _request(), Item))

    assert result.value == Item(title="prose", count=0)


def test_a_decoded_mapping_can_be_validated_without_a_response() -> None:
    assert parse_structured({"title": "direct", "count": 3}, Item) == Item(
        title="direct",
        count=3,
    )


def test_a_response_without_pricing_is_validated_but_not_priced() -> None:
    result = _run(complete_structured(FakeProvider(), _request(), Item))

    assert result.value == Item(title="hello", count=2)
    assert result.cost is None
    assert result.event is None
    assert result.amount_usd == 0.0


def test_the_helper_can_use_a_call_time_schema_and_overrides() -> None:
    provider = FakeProvider()
    helper = StructuredOutputHelper(provider, config=_config())

    result = _run(
        helper.complete(
            _request(),
            Item,
            run_id="run_2",
            operation="extract",
        )
    )

    assert result.value == Item(title="hello", count=2)
    assert result.event is not None
    assert result.event.run_id == "run_2"
    assert result.event.operation == "extract"


def test_a_tracker_without_pricing_is_rejected(engine: Engine) -> None:
    with pytest.raises(ConfigError, match="calculator or pricing"):
        StructuredOutputHelper(FakeProvider(), Item, tracker=CostTracker(engine))


def test_a_non_pydantic_schema_is_rejected_before_the_call() -> None:
    provider = FakeProvider()
    with pytest.raises(TypeError, match="pydantic BaseModel"):
        StructuredOutputHelper(provider, dict)  # type: ignore[arg-type]

    assert provider.requests == []


def test_a_provider_error_propagates_and_records_no_cost(engine: Engine) -> None:
    tracker = CostTracker(engine)
    provider = FakeProvider(error=RuntimeError("provider down"))
    helper = StructuredOutputHelper(
        provider,
        Item,
        config=_config(),
        tracker=tracker,
    )

    with pytest.raises(RuntimeError, match="provider down"):
        _run(helper.complete(_request()))

    assert tracker.count() == 0


def test_the_class_alias_is_the_helper() -> None:
    assert StructuredOutput is StructuredOutputHelper
