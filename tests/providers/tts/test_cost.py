"""Per-character text to speech cost calculation."""

from __future__ import annotations

import pytest

from aivia.core.errors import ConfigError
from aivia.core.models.provider import ProviderMeta, ProviderPrice
from aivia.core.pricing import MissingPriceWarning, PriceSource
from aivia.providers import Provider
from aivia.providers.protocols import CAPABILITY_TTS, TTSResponse
from aivia.providers.tts import (
    CHARACTER_UNIT,
    TTS_OPERATION,
    TTSCostCalculator,
    tts_cost_event,
)


class _PricedTTS(Provider):
    default_metadata = ProviderMeta(
        name="priced-tts",
        capabilities=[CAPABILITY_TTS],
        pricing={
            CAPABILITY_TTS: ProviderPrice(amount_usd=0.0001, unit=CHARACTER_UNIT),
        },
    )


class _PerThousandTTS(Provider):
    default_metadata = ProviderMeta(
        name="per-thousand-tts",
        capabilities=[CAPABILITY_TTS],
        pricing={
            CAPABILITY_TTS: ProviderPrice(amount_usd=0.01, unit="1k_chars"),
        },
    )


class _UnpricedTTS(Provider):
    default_metadata = ProviderMeta(name="unpriced-tts", capabilities=[CAPABILITY_TTS])


def test_metadata_price_is_applied_per_character() -> None:
    provider = _PricedTTS()
    calculator = TTSCostCalculator(pricing_version="2026-01")

    price = calculator.resolve(provider, 250)

    assert price.provider == "priced-tts"
    assert price.operation == TTS_OPERATION
    assert price.units == 250
    assert price.unit_kind == CHARACTER_UNIT
    assert price.billed_units == 250
    assert price.amount_usd == pytest.approx(0.025)
    assert price.source is PriceSource.METADATA
    assert price.pricing_version == "2026-01"


def test_a_per_thousand_character_price_is_converted_to_characters() -> None:
    provider = _PerThousandTTS()

    price = TTSCostCalculator().resolve(provider, 250)

    assert price.unit_kind == CHARACTER_UNIT
    assert price.price_unit == "1k_chars"
    assert price.billed_units == pytest.approx(0.25)
    assert price.amount_usd == pytest.approx(0.0025)


def test_a_provider_override_wins_over_metadata() -> None:
    provider = _PricedTTS()
    calculator = TTSCostCalculator({"priced-tts": 0.0002})

    price = calculator.resolve(provider, 100)

    assert price.amount_usd == pytest.approx(0.02)
    assert price.source is PriceSource.OVERRIDE


def test_a_response_supplies_its_character_count_and_provenance() -> None:
    provider = _PricedTTS()
    response = TTSResponse(
        audio=b"ID3audio",
        voice="voice-1",
        model="tts-model",
        provider="priced-tts",
        request_id="req-1",
        metadata={"characters": 300},
    )

    event = TTSCostCalculator().cost_event(provider, response=response, run_id="run-1")

    assert event.provider == "priced-tts"
    assert event.model == "tts-model"
    assert event.operation == TTS_OPERATION
    assert event.units == 300
    assert event.unit_kind == CHARACTER_UNIT
    assert event.amount_usd == pytest.approx(0.03)
    assert event.run_id == "run-1"
    assert event.metadata["characters"] == 300
    assert event.metadata["voice"] == "voice-1"
    assert event.metadata["request_id"] == "req-1"


def test_a_missing_price_is_recorded_as_zero_with_a_warning() -> None:
    provider = _UnpricedTTS()
    calculator = TTSCostCalculator()

    with pytest.warns(MissingPriceWarning):
        event = calculator.cost_event(provider, characters=120)

    assert event.amount_usd == 0.0
    assert event.metadata["missing_price"] is True
    assert event.metadata["price_source"] == PriceSource.MISSING.value


def test_a_missing_character_count_is_rejected() -> None:
    provider = _PricedTTS()
    response = TTSResponse(audio=b"ID3audio", provider="priced-tts")

    with pytest.raises(ConfigError, match="character count"):
        TTSCostCalculator().cost_event(provider, response=response)
    with pytest.raises(ConfigError, match="character count or response"):
        TTSCostCalculator().cost_event(provider)
    with pytest.raises(ConfigError, match="characters"):
        TTSCostCalculator().cost_event(provider, characters=-1)


def test_the_convenience_function_builds_the_same_event() -> None:
    provider = _PricedTTS()

    event = tts_cost_event(provider, characters=50)

    assert event.amount_usd == pytest.approx(0.005)
    assert event.unit_kind == CHARACTER_UNIT
