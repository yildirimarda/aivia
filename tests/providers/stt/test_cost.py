"""Per-minute speech to text cost calculation."""

from __future__ import annotations

import pytest

from aivia.core.errors import ConfigError
from aivia.core.models.provider import ProviderMeta, ProviderPrice
from aivia.core.pricing import MissingPriceWarning, PriceSource
from aivia.providers import Provider
from aivia.providers.protocols import (
    CAPABILITY_STT,
    STTResponse,
    TranscriptSegment,
    WordTimestamp,
)
from aivia.providers.stt import (
    MINUTE_UNIT,
    STT_OPERATION,
    STTCostCalculator,
    stt_cost_event,
)


class _PricedSTT(Provider):
    default_metadata = ProviderMeta(
        name="priced-stt",
        capabilities=[CAPABILITY_STT],
        pricing={
            CAPABILITY_STT: ProviderPrice(amount_usd=0.006, unit=MINUTE_UNIT),
        },
    )


class _PerSecondSTT(Provider):
    default_metadata = ProviderMeta(
        name="per-second-stt",
        capabilities=[CAPABILITY_STT],
        pricing={
            CAPABILITY_STT: ProviderPrice(amount_usd=0.0001, unit="second"),
        },
    )


class _UnpricedSTT(Provider):
    default_metadata = ProviderMeta(name="unpriced-stt", capabilities=[CAPABILITY_STT])


def test_metadata_price_is_applied_per_minute() -> None:
    provider = _PricedSTT()
    calculator = STTCostCalculator(pricing_version="2026-01")

    price = calculator.resolve(provider, 2.5)

    assert price.provider == "priced-stt"
    assert price.operation == STT_OPERATION
    assert price.units == 2.5
    assert price.unit_kind == MINUTE_UNIT
    assert price.billed_units == 2.5
    assert price.amount_usd == pytest.approx(0.015)
    assert price.source is PriceSource.METADATA
    assert price.pricing_version == "2026-01"


def test_a_per_second_price_is_converted_to_minutes() -> None:
    provider = _PerSecondSTT()

    price = STTCostCalculator().resolve(provider, 1.0)

    assert price.unit_kind == MINUTE_UNIT
    assert price.price_unit == "second"
    assert price.billed_units == pytest.approx(60.0)
    assert price.amount_usd == pytest.approx(0.006)


def test_a_provider_override_wins_over_metadata() -> None:
    provider = _PricedSTT()
    calculator = STTCostCalculator({"priced-stt": 0.01})

    price = calculator.resolve(provider, 3.0)

    assert price.amount_usd == pytest.approx(0.03)
    assert price.source is PriceSource.OVERRIDE


def test_a_response_supplies_duration_provenance_and_word_count() -> None:
    provider = _PricedSTT()
    response = STTResponse(
        text="hello there",
        language="en",
        duration_seconds=90.0,
        words=[
            WordTimestamp(word="hello", start_seconds=0.0, end_seconds=0.5),
            WordTimestamp(word="there", start_seconds=0.5, end_seconds=1.0),
        ],
        model="whisper-1",
        provider="priced-stt",
        request_id="req-1",
    )

    event = STTCostCalculator().cost_event(provider, response=response, run_id="run-1")

    assert event.provider == "priced-stt"
    assert event.model == "whisper-1"
    assert event.operation == STT_OPERATION
    assert event.units == pytest.approx(1.5)
    assert event.unit_kind == MINUTE_UNIT
    assert event.amount_usd == pytest.approx(0.009)
    assert event.run_id == "run-1"
    assert event.metadata["minutes"] == pytest.approx(1.5)
    assert event.metadata["duration_seconds"] == 90.0
    assert event.metadata["language"] == "en"
    assert event.metadata["words"] == 2
    assert event.metadata["request_id"] == "req-1"


def test_duration_is_inferred_from_segments_when_not_reported() -> None:
    provider = _PricedSTT()
    response = STTResponse(
        text="one phrase",
        segments=[
            TranscriptSegment(
                text="one phrase",
                start_seconds=0.0,
                end_seconds=30.0,
                words=[
                    WordTimestamp(word="one", start_seconds=0.0, end_seconds=0.4),
                    WordTimestamp(word="phrase", start_seconds=0.4, end_seconds=0.8),
                ],
            )
        ],
        provider="priced-stt",
    )

    event = STTCostCalculator().cost_event(provider, response=response)

    assert event.units == pytest.approx(0.5)
    assert event.amount_usd == pytest.approx(0.003)


def test_a_missing_duration_is_rejected() -> None:
    provider = _PricedSTT()
    response = STTResponse(text="no timing", provider="priced-stt")

    with pytest.raises(ConfigError, match="duration"):
        STTCostCalculator().cost_event(provider, response=response)
    with pytest.raises(ConfigError, match="duration or response"):
        STTCostCalculator().cost_event(provider)
    with pytest.raises(ConfigError, match="minutes"):
        STTCostCalculator().cost_event(provider, minutes=-1)


def test_a_missing_price_is_recorded_as_zero_with_a_warning() -> None:
    provider = _UnpricedSTT()
    calculator = STTCostCalculator()

    with pytest.warns(MissingPriceWarning):
        event = calculator.cost_event(provider, minutes=2.0)

    assert event.amount_usd == 0.0
    assert event.metadata["missing_price"] is True
    assert event.metadata["price_source"] == PriceSource.MISSING.value


def test_the_convenience_function_builds_the_same_event() -> None:
    provider = _PricedSTT()

    event = stt_cost_event(provider, minutes=0.5)

    assert event.amount_usd == pytest.approx(0.003)
    assert event.unit_kind == MINUTE_UNIT
