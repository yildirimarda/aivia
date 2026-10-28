"""Cost events for every paid provider call.

A :class:`CostEvent` is the immutable record of one billable operation: which
provider and model served it, the operation and capability, how many units were
consumed, the unit kind, the amount in USD, the run it belongs to and the
pricing version used to compute it. ``estimated`` distinguishes a quote made
before a call from an amount reported after it.

The model only describes data. It never resolves a price, calls a provider or
enforces a budget; those operations belong to the media service and the cost
tracker.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from pydantic import AliasChoices, Field, field_validator, model_validator

from aivia.core.ids import new_cost_id
from aivia.core.models.base import ModelBase

__all__ = [
    "CostEvent",
    "CostEventModel",
    "CostEventRecord",
    "CostKind",
]

_LINK_KEYS = (
    "replaces",
    "replaces_id",
    "replaces_event_id",
    "replaces_estimate_id",
    "replaced_event_id",
    "estimate_id",
    "estimate_event_id",
    "supersedes",
    "superseded_event_id",
)


class CostKind(StrEnum):
    """The accepted values for the estimated/actual flag."""

    ESTIMATED = "estimated"
    ACTUAL = "actual"


def _strip(value: Any) -> Any:
    return value.strip() if isinstance(value, str) else value


def _coerce_metadata(value: Any) -> Any:
    if value is None:
        return {}
    return value


def _coerce_link(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    identifier = getattr(value, "id", None)
    if isinstance(identifier, str):
        return identifier.strip()
    raise ValueError("a replacement link must be a cost event id or a cost event")


class CostEvent(ModelBase):
    """One paid call, with the units, amount and pricing version that produced it.

    ``units`` is the quantity consumed in ``unit_kind`` units, such as seconds,
    images, characters or tokens. ``amount_usd`` is the resulting cost and may
    be zero for a cached or free call. ``estimated`` is ``True`` for a quote
    made before the call and ``False`` once the provider reports the real
    amount.
    """

    id: str = Field(
        default_factory=new_cost_id,
        min_length=1,
        validation_alias=AliasChoices(
            "id",
            "cost_event_id",
            "cost_id",
            "event_id",
        ),
    )
    provider: str = Field(
        default="",
        validation_alias=AliasChoices("provider", "provider_name"),
    )
    model: str = Field(
        default="",
        validation_alias=AliasChoices("model", "model_id"),
    )
    operation: str = Field(
        default="",
        validation_alias=AliasChoices("operation", "capability", "stage"),
    )
    units: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices("units", "quantity", "unit_count"),
    )
    unit_kind: str = Field(
        default="call",
        validation_alias=AliasChoices("unit_kind", "unit", "unit_type"),
    )
    amount_usd: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices(
            "amount_usd",
            "cost_usd",
            "amount",
            "cost",
            "usd",
        ),
    )
    estimated: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "estimated",
            "is_estimated",
            "estimate",
        ),
    )
    run_id: str = Field(
        default="",
        validation_alias=AliasChoices("run_id", "run"),
    )
    pricing_version: str = Field(
        default="",
        validation_alias=AliasChoices(
            "pricing_version",
            "price_version",
            "pricing_id",
        ),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("metadata", "details"),
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_amount_kind(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        amount_kind = values.pop("amount_kind", None)
        if amount_kind is None:
            amount_kind = values.pop("cost_kind", None)
        if amount_kind is None:
            return values
        if not isinstance(amount_kind, str):
            raise ValueError("amount_kind must be 'estimated' or 'actual'")
        normalized = amount_kind.strip().lower()
        if normalized not in {
            CostKind.ESTIMATED,
            "estimate",
            CostKind.ACTUAL,
        }:
            raise ValueError("amount_kind must be 'estimated' or 'actual'")
        if "estimated" not in values:
            values["estimated"] = normalized in {CostKind.ESTIMATED, "estimate"}
        return values

    @model_validator(mode="before")
    @classmethod
    def _accept_replacement_link(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        metadata = values.get("metadata")
        details = dict(metadata) if isinstance(metadata, Mapping) else {}
        link = ""
        for key in _LINK_KEYS:
            if key in values:
                link = _coerce_link(values.pop(key))
                break
        if not link:
            for key in _LINK_KEYS:
                if key in details:
                    link = _coerce_link(details[key])
                    break
        if link:
            details.setdefault("replaces", link)
            details.setdefault("estimate_event_id", link)
            values["metadata"] = details
        return values

    @field_validator("units", "amount_usd", mode="before")
    @classmethod
    def _validate_finite(cls, value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("cost values must be finite")
        return value

    @field_validator(
        "provider",
        "model",
        "operation",
        "unit_kind",
        "run_id",
        "pricing_version",
        mode="before",
    )
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _strip(value)

    @field_validator("metadata", mode="before")
    @classmethod
    def _normalize_metadata(cls, value: Any) -> Any:
        return _coerce_metadata(value)

    @property
    def capability(self) -> str:
        """Alias for :attr:`operation`."""
        return self.operation

    @property
    def unit(self) -> str:
        """Alias for :attr:`unit_kind`."""
        return self.unit_kind

    @property
    def amount(self) -> float:
        """Alias for :attr:`amount_usd`."""
        return self.amount_usd

    @property
    def cost(self) -> float:
        """Alias for :attr:`amount_usd`."""
        return self.amount_usd

    @property
    def cost_usd(self) -> float:
        """Alias for :attr:`amount_usd`."""
        return self.amount_usd

    @property
    def is_estimated(self) -> bool:
        """Whether this event is a pre-call estimate."""
        return self.estimated

    @property
    def is_actual(self) -> bool:
        """Whether this event holds a reported amount."""
        return not self.estimated

    @property
    def amount_kind(self) -> str:
        """The estimated/actual flag as a word."""
        return CostKind.ESTIMATED if self.estimated else CostKind.ACTUAL

    @property
    def replaces(self) -> str:
        """The id of the estimate this event replaces, if any."""
        link = self.metadata.get("replaces")
        if not link:
            link = self.metadata.get("estimate_event_id")
        return str(link or "")

    @property
    def replaces_event_id(self) -> str:
        """Alias for :attr:`replaces`."""
        return self.replaces

    @property
    def estimate_event_id(self) -> str:
        """Alias for :attr:`replaces`."""
        return self.replaces


CostEventModel = CostEvent
"""Alias for :class:`CostEvent` for callers that prefer the record wording."""

CostEventRecord = CostEvent
"""Alias for :class:`CostEvent` for callers that prefer the record wording."""
