"""Cost event and provider metadata contracts."""

from __future__ import annotations

import json
import warnings
from typing import Any

import pytest
from pydantic import ValidationError

from aivia.core.ids import is_ulid
from aivia.core.models import (
    AssetLicense,
    CostEvent,
    CostEventModel,
    CostEventRecord,
    CostKind,
    ProviderMeta,
    ProviderMetadata,
    ProviderPrice,
    ProviderPricing,
    RateLimit,
    RateLimits,
    UnknownFieldWarning,
)


def _event(**overrides: Any) -> CostEvent:
    values: dict[str, Any] = {
        "provider": "acme",
        "model": "image-1",
        "operation": "image_gen",
        "units": 2.0,
        "unit_kind": "image",
        "amount_usd": 0.08,
        "run_id": "run_1",
        "pricing_version": "2026-01",
    }
    values.update(overrides)
    return CostEvent(**values)


def test_cost_event_defaults_to_a_free_actual_call() -> None:
    event = CostEvent()

    assert event.id.startswith("cost_")
    assert is_ulid(event.id.removeprefix("cost_"))
    assert event.provider == ""
    assert event.model == ""
    assert event.operation == ""
    assert event.units == 0.0
    assert event.unit_kind == "call"
    assert event.amount_usd == 0.0
    assert event.estimated is False
    assert event.run_id == ""
    assert event.pricing_version == ""
    assert event.metadata == {}
    assert event.capability == ""
    assert event.unit == "call"
    assert event.amount == 0.0
    assert event.cost == 0.0
    assert event.cost_usd == 0.0
    assert event.is_estimated is False
    assert event.is_actual is True
    assert event.amount_kind == CostKind.ACTUAL


def test_cost_event_ids_are_unique_and_time_sortable() -> None:
    first = CostEvent()
    second = CostEvent()

    assert first.id != second.id
    assert first.id < second.id


def test_cost_event_accepts_aliases_and_strips_text() -> None:
    event = CostEvent.model_validate(
        {
            "cost_id": "cost_1",
            "provider_name": "  acme  ",
            "model_id": " image-1 ",
            "capability": " image_gen ",
            "quantity": 3,
            "unit": " image ",
            "cost_usd": 0.12,
            "run": " run_1 ",
            "price_version": " 2026-01 ",
            "estimated": True,
            "details": {"stage": "visuals"},
        }
    )

    assert event.id == "cost_1"
    assert event.provider == "acme"
    assert event.model == "image-1"
    assert event.operation == "image_gen"
    assert event.capability == "image_gen"
    assert event.units == 3.0
    assert event.unit_kind == "image"
    assert event.amount_usd == 0.12
    assert event.run_id == "run_1"
    assert event.pricing_version == "2026-01"
    assert event.estimated is True
    assert event.metadata == {"stage": "visuals"}


@pytest.mark.parametrize(
    ("amount_kind", "estimated"),
    [("estimated", True), ("  ESTIMATE ", True), ("actual", False), ("Actual", False)],
)
def test_cost_event_accepts_an_amount_kind_word(
    amount_kind: str, estimated: bool
) -> None:
    event = CostEvent.model_validate({"amount_kind": amount_kind})

    assert event.estimated is estimated
    assert event.amount_kind == (CostKind.ESTIMATED if estimated else CostKind.ACTUAL)


def test_cost_event_rejects_an_unknown_amount_kind() -> None:
    with pytest.raises(ValidationError):
        CostEvent.model_validate({"amount_kind": "probably"})
    with pytest.raises(ValidationError):
        CostEvent.model_validate({"cost_kind": 3})


def test_cost_event_explicit_estimated_flag_wins_over_amount_kind() -> None:
    event = CostEvent.model_validate({"amount_kind": "estimated", "estimated": False})

    assert event.estimated is False


def test_cost_event_rejects_negative_and_non_finite_amounts() -> None:
    with pytest.raises(ValidationError):
        CostEvent(units=-1)
    with pytest.raises(ValidationError):
        CostEvent(amount_usd=-0.01)
    with pytest.raises(ValidationError):
        CostEvent(units=float("inf"))
    with pytest.raises(ValidationError):
        CostEvent(amount_usd=float("nan"))


def test_cost_event_normalizes_missing_metadata_to_an_empty_mapping() -> None:
    event = CostEvent.model_validate({"metadata": None})

    assert event.metadata == {}


def test_cost_event_round_trips_through_canonical_json() -> None:
    event = _event(metadata={"stage": "visuals"})

    payload = json.loads(event.to_json())
    restored = CostEvent.from_dict(payload)

    assert payload["id"] == event.id
    assert payload["amount_usd"] == 0.08
    assert payload["metadata"] == {"stage": "visuals"}
    assert restored == event
    assert restored.to_json() == event.to_json()


def test_cost_event_aliases_are_exported() -> None:
    assert CostEventModel is CostEvent
    assert CostEventRecord is CostEvent


def test_cost_event_warns_about_unknown_fields_but_keeps_them() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        event = CostEvent.from_dict({"provider": "acme", "surprise": 1})

    assert event.provider == "acme"
    assert event.model_extra == {"surprise": 1}
    assert any(issubclass(item.category, UnknownFieldWarning) for item in caught)


def test_provider_price_defaults_and_cost_for() -> None:
    price = ProviderPrice()

    assert price.amount_usd == 0.0
    assert price.unit == "call"
    assert price.units_per_price == 1.0
    assert price.currency == "USD"
    assert price.price_usd == 0.0
    assert price.price == 0.0
    assert price.cost_for(3) == 0.0


def test_provider_price_accepts_aliases_and_scales_units() -> None:
    price = ProviderPrice.model_validate(
        {"price": 0.04, "unit_type": " image ", "currency": "usd"}
    )

    assert price.amount_usd == 0.04
    assert price.unit == "image"
    assert price.currency == "USD"
    assert price.cost_for(3) == pytest.approx(0.12)


def test_provider_price_honours_units_per_price() -> None:
    price = ProviderPrice(amount_usd=0.02, units_per_price=1000)

    assert price.cost_for(2500) == pytest.approx(0.05)


def test_provider_price_rejects_invalid_values() -> None:
    with pytest.raises(ValidationError):
        ProviderPrice(amount_usd=-1)
    with pytest.raises(ValidationError):
        ProviderPrice(amount_usd=float("inf"))
    with pytest.raises(ValidationError):
        ProviderPrice(units_per_price=0)
    with pytest.raises(ValueError):
        ProviderPrice(amount_usd=1).cost_for(-1)
    with pytest.raises(ValueError):
        ProviderPrice(amount_usd=1).cost_for(float("inf"))


def test_rate_limit_defaults() -> None:
    limit = RateLimit()

    assert limit.requests_per_minute == 60
    assert limit.burst == 0
    assert limit.effective_burst == 60
    assert limit.max_concurrency == 0
    assert limit.requests_per_second == 1.0
    assert limit.is_unlimited is False


def test_rate_limit_accepts_aliases_and_a_bare_number() -> None:
    limit = RateLimit.model_validate({"rpm": 120, "max_parallel": 4, "burst_size": 10})

    assert limit.requests_per_minute == 120
    assert limit.burst == 10
    assert limit.effective_burst == 10
    assert limit.max_concurrency == 4
    assert limit.requests_per_second == 2.0

    assert (
        ProviderMeta.model_validate({"rate_limit": 30}).rate_limit.requests_per_minute
        == 30
    )


def test_rate_limit_zero_means_unlimited() -> None:
    limit = RateLimit(requests_per_minute=0)

    assert limit.is_unlimited is True
    assert limit.effective_burst == 0
    assert limit.requests_per_second == 0.0


def test_rate_limit_rejects_negative_values() -> None:
    with pytest.raises(ValidationError):
        RateLimit(requests_per_minute=-1)
    with pytest.raises(ValidationError):
        RateLimit(burst=-1)
    with pytest.raises(ValidationError):
        RateLimit(max_concurrency=-1)


def test_provider_meta_defaults() -> None:
    meta = ProviderMeta()

    assert meta.name == ""
    assert meta.capabilities == []
    assert meta.capability == ""
    assert meta.primary_capability == ""
    assert meta.supported_options == {}
    assert meta.pricing == {}
    assert meta.rate_limit.requests_per_minute == 60
    assert meta.rate_limit_per_minute == 60
    assert meta.quality_rank is None
    assert meta.has_quality_rank is False
    assert meta.license.name == ""
    assert meta.is_commercial is False
    assert meta.commercial_use_notes == ""


def test_provider_meta_accepts_aliases_and_coerces_nested_values() -> None:
    meta = ProviderMeta.model_validate(
        {
            "provider": "  acme  ",
            "capability": " image_gen ",
            "options": {"image_gen": " 1024x1024 "},
            "prices": {"image_gen": 0.04},
            "limits": 30,
            "rank": 4,
            "licensing": "proprietary",
            "commercial_notes": " commercial use allowed ",
        }
    )

    assert meta.name == "acme"
    assert meta.capabilities == ["image_gen"]
    assert meta.capability == "image_gen"
    assert meta.supported_options == {"image_gen": ["1024x1024"]}
    assert meta.pricing["image_gen"].amount_usd == 0.04
    assert meta.rate_limit.requests_per_minute == 30
    assert meta.quality_rank == 4
    assert meta.has_quality_rank is True
    assert meta.license.name == "proprietary"
    assert meta.commercial_use_notes == "commercial use allowed"


def test_provider_meta_deduplicates_capabilities_and_options() -> None:
    meta = ProviderMeta(
        capabilities=[" image_gen ", "image_gen", "video_gen"],
        supported_options={"image_gen": [" square ", "square", "wide"]},
    )

    assert meta.capabilities == ["image_gen", "video_gen"]
    assert meta.supported_options == {"image_gen": ["square", "wide"]}


def test_provider_meta_rejects_empty_capabilities_and_option_keys() -> None:
    with pytest.raises(ValidationError):
        ProviderMeta(capabilities=[""])
    with pytest.raises(ValidationError):
        ProviderMeta(capabilities=[1])  # type: ignore[list-item]
    with pytest.raises(ValidationError):
        ProviderMeta(supported_options={"": ["square"]})
    with pytest.raises(ValidationError):
        ProviderMeta(supported_options={"image_gen": [""]})


def test_provider_meta_supports_and_resolves_pricing_with_a_default() -> None:
    meta = ProviderMeta(
        capabilities=["image_gen", "image_edit"],
        pricing={
            "image_gen": ProviderPrice(amount_usd=0.04, unit="image"),
            "default": ProviderPrice(amount_usd=0.01, unit="call"),
        },
    )

    assert meta.supports("image_gen") is True
    assert meta.supports("  image_edit ") is True
    assert meta.supports("video_gen") is False
    assert meta.pricing_for("image_gen") is not None
    assert meta.pricing_for("image_gen").amount_usd == 0.04  # type: ignore[union-attr]
    assert meta.pricing_for("video_gen") is not None
    assert meta.pricing_for("video_gen").amount_usd == 0.01  # type: ignore[union-attr]
    assert meta.price_for("video_gen") == meta.pricing_for("video_gen")


def test_provider_meta_pricing_accepts_a_scalar_default() -> None:
    meta = ProviderMeta.model_validate({"pricing": 0.5})

    assert meta.pricing["default"].amount_usd == 0.5
    assert meta.pricing_for("anything") == meta.pricing["default"]


def test_provider_meta_pricing_rejects_empty_keys() -> None:
    with pytest.raises(ValidationError):
        ProviderMeta(pricing={" ": ProviderPrice(amount_usd=1)})


def test_provider_meta_rejects_a_negative_quality_rank() -> None:
    with pytest.raises(ValidationError):
        ProviderMeta(quality_rank=-1)


def test_provider_meta_license_tracks_commercial_use() -> None:
    meta = ProviderMeta(
        license=AssetLicense(name="CC0", commercial_use=True),
        commercial_use_notes="No attribution required",
    )

    assert meta.license.name == "CC0"
    assert meta.license.commercial_use is True
    assert meta.is_commercial is True
    assert meta.commercial_use_notes == "No attribution required"


def test_provider_meta_round_trips_through_canonical_json() -> None:
    meta = ProviderMeta(
        name="acme",
        capabilities=["image_gen"],
        supported_options={"image_gen": ["square"]},
        pricing={"image_gen": ProviderPrice(amount_usd=0.04, unit="image")},
        rate_limit=RateLimit(requests_per_minute=120, burst=10),
        quality_rank=4,
        license=AssetLicense(name="proprietary", commercial_use=True),
        commercial_use_notes="Read the terms",
    )

    payload = json.loads(meta.to_json())
    restored = ProviderMeta.from_dict(payload)

    assert payload["name"] == "acme"
    assert payload["pricing"]["image_gen"]["amount_usd"] == 0.04
    assert payload["rate_limit"]["requests_per_minute"] == 120
    assert payload["quality_rank"] == 4
    assert payload["license"]["name"] == "proprietary"
    assert restored == meta
    assert restored.to_json() == meta.to_json()


def test_provider_meta_aliases_are_exported() -> None:
    assert ProviderMetadata is ProviderMeta
    assert ProviderPricing is ProviderPrice
    assert RateLimits is RateLimit
