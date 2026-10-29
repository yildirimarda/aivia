"""Repair loop for structured output that re prompts with validation errors."""

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
    RepairLoop,
    StructuredOutputError,
    StructuredOutputResult,
    build_repair_prompt,
    repair_structured,
)
from aivia.providers.protocols.llm import (
    ChatMessage,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    MessageRole,
)

MODEL = "model-a"
VALID = '{"title": "fixed", "count": 1}'
INVALID = '{"title": "", "count": -1}'


class Item(BaseModel):
    title: str = Field(min_length=1)
    count: int = Field(ge=0)


class SequenceProvider:
    """An async LLM provider returning one canned response per call."""

    def __init__(
        self,
        responses: list[str] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.responses = list(responses or [VALID])
        self.error = error
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        index = min(len(self.requests) - 1, len(self.responses) - 1)
        return LLMResponse(
            content=self.responses[index],
            model=MODEL,
            provider="acme",
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


def _config(max_repairs: int = 2) -> LLMConfig:
    return LLMConfig.from_mapping(
        {
            "max_repairs": max_repairs,
            "pricing": {
                MODEL: {
                    "input_per_million": 3.0,
                    "output_per_million": 15.0,
                }
            },
        }
    )


def _request(text: str = "make an item") -> LLMRequest:
    return LLMRequest(
        messages=[ChatMessage(role=MessageRole.USER, content=text)],
        model=MODEL,
    )


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine)


def test_a_valid_first_response_is_returned_without_a_repair() -> None:
    provider = SequenceProvider()
    loop = RepairLoop(provider, Item, config=_config())

    result = _run(loop.complete(_request()))

    assert result.value == Item(title="fixed", count=1)
    assert result.attempts == 1
    assert result.repairs == 0
    assert len(provider.requests) == 1
    assert len(provider.requests[0].messages) == 1


def test_a_validation_failure_is_repaired_with_the_validation_error(
    engine: Engine,
) -> None:
    provider = SequenceProvider([INVALID, VALID])
    tracker = CostTracker(engine)
    loop = RepairLoop(provider, Item, config=_config(), tracker=tracker)

    result = _run(loop.complete(_request()))

    assert isinstance(result, StructuredOutputResult)
    assert result.value == Item(title="fixed", count=1)
    assert result.attempts == 2
    assert result.repairs == 1
    assert len(provider.requests) == 2

    repair = provider.requests[1]
    assert len(repair.messages) == 3
    assert repair.messages[0].role is MessageRole.USER
    assert repair.messages[1].role is MessageRole.ASSISTANT
    assert repair.messages[1].content == INVALID
    assert repair.messages[2].role is MessageRole.USER
    assert "failed validation" in repair.messages[2].content
    assert "title" in repair.messages[2].content
    assert "Required JSON schema" in repair.messages[2].content
    assert tracker.count() == 2
    assert tracker.total() == pytest.approx(0.021)


def test_the_loop_raises_a_structured_output_error_at_the_limit(
    engine: Engine,
) -> None:
    provider = SequenceProvider([INVALID])
    tracker = CostTracker(engine)
    loop = RepairLoop(provider, Item, config=_config(max_repairs=2), tracker=tracker)

    with pytest.raises(StructuredOutputError) as raised:
        _run(loop.complete(_request()))

    error = raised.value
    assert error.code == "structured_output"
    assert "repair limit reached" in str(error)
    assert error.attempts == 3
    assert error.repairs == 2
    assert error.max_repairs == 2
    assert error.content == INVALID
    assert error.errors
    assert error.details["attempts"] == 3
    assert error.details["repairs"] == 2
    assert len(provider.requests) == 3
    assert tracker.count() == 3
    assert tracker.total() == pytest.approx(0.0315)


def test_zero_repairs_means_the_first_failure_is_final() -> None:
    provider = SequenceProvider([INVALID, VALID])
    loop = RepairLoop(provider, Item, config=_config(max_repairs=0))

    with pytest.raises(StructuredOutputError):
        _run(loop.complete(_request()))

    assert len(provider.requests) == 1


def test_an_explicit_limit_overrides_the_configured_one() -> None:
    provider = SequenceProvider([INVALID, VALID])
    loop = RepairLoop(provider, Item, config=_config(max_repairs=0))

    result = _run(loop.complete(_request(), max_repairs=1))

    assert result.value == Item(title="fixed", count=1)
    assert result.attempts == 2
    assert len(provider.requests) == 2


def test_invalid_json_is_repaired_and_the_prompt_quotes_the_parse_error() -> None:
    provider = SequenceProvider(["not json", VALID])
    loop = RepairLoop(provider, Item, config=_config())

    _run(loop.complete(_request()))

    prompt = provider.requests[1].messages[-1].content
    assert isinstance(prompt, str)
    assert "not valid JSON" in prompt
    assert "not json" in prompt


def test_a_provider_error_is_not_repaired() -> None:
    provider = SequenceProvider(error=RuntimeError("provider down"))
    loop = RepairLoop(provider, Item, config=_config())

    with pytest.raises(RuntimeError, match="provider down"):
        _run(loop.complete(_request()))

    assert len(provider.requests) == 1


def test_a_mapping_config_supplies_the_repair_limit() -> None:
    provider = SequenceProvider([INVALID, VALID])
    loop = RepairLoop(
        provider,
        Item,
        config={
            "max_repairs": 1,
            "pricing": {
                MODEL: {
                    "input_per_million": 3.0,
                    "output_per_million": 15.0,
                }
            },
        },
    )

    result = _run(loop.complete(_request()))

    assert result.attempts == 2
    assert len(provider.requests) == 2


def test_conflicting_explicit_limits_are_rejected() -> None:
    with pytest.raises(ConfigError, match="disagree"):
        RepairLoop(
            SequenceProvider(),
            Item,
            config=_config(),
            max_repairs=1,
            repair_limit=2,
        )


def test_an_initial_failure_with_empty_content_still_produces_a_repair_prompt() -> None:
    provider = SequenceProvider(["", VALID])
    loop = RepairLoop(provider, Item, config=_config())

    result = _run(loop.complete(_request()))

    assert result.value == Item(title="fixed", count=1)
    repair = provider.requests[1]
    assert len(repair.messages) == 2
    assert repair.messages[-1].role is MessageRole.USER
    assert "empty response" in repair.messages[-1].content


def test_the_one_shot_helper_runs_the_loop() -> None:
    provider = SequenceProvider([INVALID, VALID])

    result = _run(repair_structured(provider, _request(), Item, config=_config()))

    assert result.value == Item(title="fixed", count=1)
    assert result.repairs == 1


def test_the_repair_prompt_can_be_replaced() -> None:
    provider = SequenceProvider([INVALID, VALID])
    seen: list[StructuredOutputError] = []

    def prompt(error: StructuredOutputError, schema: type[BaseModel] | None) -> str:
        seen.append(error)
        return "try again please"

    loop = RepairLoop(provider, Item, config=_config(), prompt_builder=prompt)

    _run(loop.complete(_request()))

    assert len(seen) == 1
    assert provider.requests[1].messages[-1].content == "try again please"


def test_the_prompt_builder_quotes_the_error_and_schema() -> None:
    error = StructuredOutputError(
        "response does not match Item",
        content=INVALID,
        errors=[{"loc": ("title",), "msg": "too short"}],
    )

    prompt = build_repair_prompt(error, Item)

    assert INVALID in prompt
    assert "too short" in prompt
    assert "Required JSON schema" in prompt
