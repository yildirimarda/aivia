"""Resolving provider prices from metadata, overrides and unit conversions."""

from __future__ import annotations

import math
import warnings
from typing import Any

import pytest

from aivia.core.errors import ConfigError
from aivia.core.models import ProviderMeta, ProviderPrice
from aivia.core.pricing import (
    MissingPriceWarning,
    PriceResolution,
    PriceResolver,
    PriceSource,
    PricingResolver,
    PricingResult,
    PricingSource,
    ResolvedPrice,
    ResolvedPricing,
    convert_units,
    resolve_price,
    resolve_pricing,
)
from aivia.providers import Provider


class FakeProvider(Provider):
    """A provider with configurable metadata and a recorded name."""

    def __init__(
        self,
        name: str,
        capabilities: list[str],
        pricing: dict[str, Any] | float | None = None,
    ) -> None:
        super().__init__(
            ProviderMeta.model_validate(
                {
                    "name": name,
                    "capabilities": capabilities,
                    "pricing": {} if pricing is None else pricing,
                }
            )
        )


def test_metadata_price_scales_units_and_units_per_price() -> None:
    provider = FakeProvider(
        "acme",
        ["image_gen"],
        pricing={"image_gen": ProviderPrice(amount_usd=0.04, unit="image")},
    )

    result = PricingResolver().resolve(provider, "image_gen", 3)

    assert result.provider == "acme"
    assert result.operation == "image_gen"
    assert result.amount_usd == pytest.approx(0.12)
    assert result.units == 3.0
    assert result.billed_units == 3.0
    assert result.unit_kind == "image"
    assert result.price_unit == "image"
    assert result.units_per_price == 1.0
    assert result.currency == "USD"
    assert result.source is PriceSource.METADATA
    assert result.is_metadata is True
    assert result.missing is False


def test_metadata_price_honours_a_bulk_units_per_price() -> None:
    provider = FakeProvider(
        "acme",
        ["llm"],
        pricing={
            "llm": ProviderPrice(
                amount_usd=0.02,
                unit="token",
                units_per_price=1000,
            )
        },
    )

    result = PricingResolver().resolve(provider, "llm", 2500, unit_kind="tokens")

    assert result.billed_units == pytest.approx(2500)
    assert result.amount_usd == pytest.approx(0.05)


def test_metadata_price_converts_requested_time_units() -> None:
    provider = FakeProvider(
        "acme",
        ["video_gen"],
        pricing={"video_gen": ProviderPrice(amount_usd=0.10, unit="minute")},
    )

    result = PricingResolver().resolve(provider, "video_gen", 90, unit_kind="seconds")

    assert result.units == 90
    assert result.unit_kind == "seconds"
    assert result.price_unit == "minute"
    assert result.billed_units == pytest.approx(1.5)
    assert result.amount_usd == pytest.approx(0.15)


def test_metadata_price_falls_back_to_the_default_price() -> None:
    provider = FakeProvider(
        "acme",
        ["image_gen"],
        pricing={"default": ProviderPrice(amount_usd=0.03)},
    )

    result = PricingResolver().resolve(provider, "unknown_operation")

    assert result.amount_usd == pytest.approx(0.03)
    assert result.source is PriceSource.METADATA


def test_unit_conversion_handles_plurals_abbreviations_and_counts() -> None:
    assert convert_units(2, "minutes", "seconds") == pytest.approx(120)
    assert convert_units(1500, "ms", "s") == pytest.approx(1.5)
    assert convert_units(3, "images", "image") == pytest.approx(3)
    assert convert_units(4, "calls", "image") == pytest.approx(4)


def test_incompatible_and_unknown_units_raise_a_config_error() -> None:
    with pytest.raises(ConfigError, match="cannot convert"):
        convert_units(1, "second", "image")

    with pytest.raises(ConfigError, match="unknown pricing unit"):
        convert_units(1, "furlong", "second")


def test_an_unknown_unit_that_matches_itself_needs_no_conversion() -> None:
    assert convert_units(3, "generation", "generation") == pytest.approx(3)


def test_override_keeps_the_metadata_unit_and_bulk_factor() -> None:
    provider = FakeProvider(
        "acme",
        ["llm"],
        pricing={
            "llm": ProviderPrice(
                amount_usd=0.02,
                unit="token",
                units_per_price=1000,
            )
        },
    )

    result = PricingResolver({"acme": 0.03}).resolve(
        provider,
        "llm",
        2500,
        unit_kind="tokens",
    )

    assert result.amount_usd == pytest.approx(0.075)
    assert result.source is PriceSource.OVERRIDE
    assert result.price_unit == "token"
    assert result.units_per_price == 1000


def test_override_without_metadata_does_not_warn_about_a_missing_price() -> None:
    provider = FakeProvider("acme", ["image_gen"])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = PricingResolver({"acme": 0.02}).resolve(provider, "image_gen", 2)

    assert caught == []
    assert result.amount_usd == pytest.approx(0.04)
    assert result.source is PriceSource.OVERRIDE
    assert result.missing is False
    assert result.price_unit == "call"


def test_override_wins_over_metadata_and_uses_the_configured_name() -> None:
    provider = FakeProvider(
        "metadata_name",
        ["image_gen"],
        pricing={"image_gen": ProviderPrice(amount_usd=0.50)},
    )

    result = PricingResolver({"alias": 0.02}).resolve(
        provider,
        "image_gen",
        2,
        provider_name="alias",
    )

    assert result.provider == "alias"
    assert result.amount_usd == pytest.approx(0.04)
    assert result.source is PriceSource.OVERRIDE
    assert result.is_override is True


def test_a_zero_override_marks_a_free_call_not_a_missing_price() -> None:
    provider = FakeProvider(
        "acme",
        ["image_gen"],
        pricing={"image_gen": ProviderPrice(amount_usd=0.50)},
    )

    result = PricingResolver({"acme": 0.0}).resolve(provider, "image_gen")

    assert result.amount_usd == 0.0
    assert result.source is PriceSource.OVERRIDE
    assert result.missing is False


def test_missing_price_warns_and_returns_a_zero_cost_resolution() -> None:
    provider = FakeProvider("acme", ["image_gen"])

    with pytest.warns(MissingPriceWarning, match="no price") as caught:
        result = PricingResolver().resolve(
            provider,
            "image_gen",
            4,
            unit_kind="image",
        )

    assert len(caught) == 1
    warning = caught[0].message
    assert isinstance(warning, MissingPriceWarning)
    assert warning.provider == "acme"
    assert warning.operation == "image_gen"
    assert result.amount_usd == 0.0
    assert result.units == 4.0
    assert result.billed_units == 4.0
    assert result.unit_kind == "image"
    assert result.source is PriceSource.MISSING
    assert result.is_missing is True
    assert result.missing is True


def test_missing_price_does_not_raise() -> None:
    provider = FakeProvider("acme", ["image_gen"])

    with pytest.warns(MissingPriceWarning):
        result = PricingResolver().resolve(provider, "image_gen")

    assert result.amount_usd == 0.0
    assert result.unit_kind == "call"


def test_pricing_version_is_carried_on_the_resolution() -> None:
    provider = FakeProvider(
        "acme",
        ["image_gen"],
        pricing={"image_gen": ProviderPrice(amount_usd=0.04)},
    )

    result = PricingResolver(pricing_version="2026-01").resolve(
        provider,
        "image_gen",
    )

    assert result.pricing_version == "2026-01"


def test_resolver_rejects_invalid_overrides_and_quantities() -> None:
    with pytest.raises(ConfigError, match="finite and non-negative"):
        PricingResolver({"acme": -1})

    with pytest.raises(ConfigError, match="finite and non-negative"):
        PricingResolver({"acme": math.inf})

    with pytest.raises(ConfigError, match="non-empty"):
        PricingResolver({" ": 1})

    provider = FakeProvider(
        "acme",
        ["image_gen"],
        pricing={"image_gen": ProviderPrice(amount_usd=0.04)},
    )
    with pytest.raises(ConfigError, match="units must be a finite"):
        PricingResolver().resolve(provider, "image_gen", -1)
    with pytest.raises(ConfigError, match="units must be a finite"):
        PricingResolver().resolve(provider, "image_gen", math.nan)
    with pytest.raises(ConfigError, match="operation must be a non-empty"):
        PricingResolver().resolve(provider, " ")


def test_resolver_accepts_provider_metadata_without_a_provider_instance() -> None:
    metadata = ProviderMeta(
        name="acme",
        capabilities=["image_gen"],
        pricing={"image_gen": ProviderPrice(amount_usd=0.03)},
    )

    result = resolve_price(metadata, "image_gen")

    assert result.provider == "acme"
    assert result.amount_usd == pytest.approx(0.03)


def test_module_level_helpers_accept_overrides_and_units() -> None:
    provider = FakeProvider(
        "acme",
        ["video_gen"],
        pricing={"video_gen": ProviderPrice(amount_usd=0.10, unit="second")},
    )

    result = resolve_pricing(
        provider,
        "video_gen",
        2,
        unit_kind="minutes",
        overrides={"acme": 0.20},
    )

    assert result.amount_usd == pytest.approx(24.0)
    assert result.source is PriceSource.OVERRIDE


def test_resolved_price_aliases_and_to_dict() -> None:
    provider = FakeProvider(
        "acme",
        ["image_gen"],
        pricing={"image_gen": ProviderPrice(amount_usd=0.04)},
    )

    result = PricingResolver().resolve(provider, "image_gen", 2)

    assert result.amount == result.amount_usd
    assert result.cost_usd == result.amount_usd
    assert result.price == result.amount_usd
    assert result.price_usd == result.amount_usd
    assert result.unit == result.unit_kind
    payload = result.to_dict()
    assert payload["provider"] == "acme"
    assert payload["operation"] == "image_gen"
    assert payload["amount_usd"] == pytest.approx(0.08)
    assert payload["source"] == "metadata"
    assert payload["missing"] is False
    assert PriceResolver is PricingResolver
    assert PricingSource is PriceSource
    assert PriceResolution is ResolvedPrice
    assert PricingResult is ResolvedPrice
    assert ResolvedPricing is ResolvedPrice
