"""Convert reported LLM token usage into priced cost events."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import Engine

from aivia.core.config.llm import LLMConfig, TokenPricing
from aivia.core.cost import CostTracker
from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.errors import ConfigError
from aivia.providers.llm import (
    LLMCostCalculator,
    LLMTokenCost,
    TokenCostCalculator,
    cost_event_for_response,
    token_cost_for_usage,
)
from aivia.providers.protocols.llm import LLMResponse, LLMUsage

MODEL = "model-a"


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


def _response(
    *,
    model: str = MODEL,
    input_tokens: int = 1_000,
    output_tokens: int = 500,
    cached_input_tokens: int = 100,
) -> LLMResponse:
    return LLMResponse(
        content="hello",
        model=model,
        provider="acme",
        usage=LLMUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
        ),
    )


def test_reported_usage_is_priced_with_the_configured_per_million_prices() -> None:
    calculator = LLMCostCalculator(_config(), pricing_version="2026-01")

    cost = calculator.cost_for(_response())

    assert isinstance(cost, LLMTokenCost)
    assert cost.provider == "acme"
    assert cost.model == MODEL
    assert cost.input_tokens == 1_000
    assert cost.output_tokens == 500
    assert cost.cached_input_tokens == 100
    assert cost.total_tokens == 1_500
    assert cost.input_cost_usd == pytest.approx(0.003)
    assert cost.output_cost_usd == pytest.approx(0.0075)
    assert cost.amount_usd == pytest.approx(0.0105)
    assert cost.cost_usd == pytest.approx(0.0105)
    assert cost.pricing_version == "2026-01"


def test_the_cost_event_carries_the_split_amount_and_pricing_version() -> None:
    calculator = LLMCostCalculator(_config(), pricing_version="2026-01")

    event = calculator.cost_event(
        _response(),
        run_id="run_1",
        operation="write",
        metadata={"stage": "write"},
    )

    assert event.provider == "acme"
    assert event.model == MODEL
    assert event.operation == "write"
    assert event.units == 1_500.0
    assert event.unit_kind == "token"
    assert event.amount_usd == pytest.approx(0.0105)
    assert event.estimated is False
    assert event.run_id == "run_1"
    assert event.pricing_version == "2026-01"
    assert event.metadata == {
        "input_tokens": 1_000,
        "output_tokens": 500,
        "cached_input_tokens": 100,
        "input_cost_usd": pytest.approx(0.003),
        "output_cost_usd": pytest.approx(0.0075),
        "stage": "write",
    }


def test_the_calculator_records_an_actual_cost_event(engine: Engine) -> None:
    tracker = CostTracker(engine)
    calculator = LLMCostCalculator(_config())

    event = calculator.record(_response(), tracker, run_id="run_1")

    assert tracker.repository.get(event.id) == event
    assert event.estimated is False
    assert tracker.total() == pytest.approx(0.0105)


def test_an_estimate_can_be_recorded_for_a_pre_call_quote(engine: Engine) -> None:
    tracker = CostTracker(engine)
    calculator = LLMCostCalculator(_config())

    event = calculator.record(_response(), tracker, estimated=True)

    assert event.estimated is True
    assert event.amount_kind == "estimated"
    assert tracker.repository.get(event.id) == event


def test_zero_prices_produce_a_zero_cost_event() -> None:
    calculator = LLMCostCalculator(
        _config(input_per_million=0.0, output_per_million=0.0)
    )

    event = calculator.cost_event(_response())

    assert event.amount_usd == 0.0
    assert event.units == 1_500.0


def test_a_model_without_configured_pricing_is_a_config_error(engine: Engine) -> None:
    tracker = CostTracker(engine)
    calculator = LLMCostCalculator(_config())

    with pytest.raises(ConfigError) as raised:
        calculator.record(_response(model="model-b"), tracker)

    assert raised.value.details["model"] == "model-b"
    assert tracker.count() == 0


def test_a_response_without_a_model_id_is_a_config_error() -> None:
    calculator = LLMCostCalculator(_config())

    with pytest.raises(ConfigError, match="without a model id"):
        calculator.cost_for(_response(model="   "))


def test_a_pricing_mapping_can_configure_the_calculator() -> None:
    calculator = LLMCostCalculator(
        {MODEL: TokenPricing(input_per_million=1.0, output_per_million=2.0)}
    )

    cost = calculator.cost_for(
        _response(input_tokens=1_000_000, output_tokens=1_000_000)
    )

    assert cost.amount_usd == pytest.approx(3.0)
    assert TokenCostCalculator is LLMCostCalculator


def test_token_cost_for_usage_prices_an_already_resolved_price() -> None:
    usage = LLMUsage(input_tokens=2_000_000, output_tokens=500_000)

    cost = token_cost_for_usage(
        usage,
        TokenPricing(input_per_million=1.5, output_per_million=6.0),
        provider="acme",
        model=MODEL,
        pricing_version="2026-02",
    )

    assert cost.input_cost_usd == pytest.approx(3.0)
    assert cost.output_cost_usd == pytest.approx(3.0)
    assert cost.amount_usd == pytest.approx(6.0)
    assert cost.pricing_version == "2026-02"


def test_cost_event_for_response_is_a_shortcut_for_the_calculator() -> None:
    event = cost_event_for_response(
        _response(),
        _config(),
        pricing_version="2026-01",
        run_id="run_2",
    )

    assert event.run_id == "run_2"
    assert event.operation == "llm"
    assert event.pricing_version == "2026-01"
    assert event.amount_usd == pytest.approx(0.0105)
