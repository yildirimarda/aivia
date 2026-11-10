"""Privacy controls for personalized orders.

Orders carry personal data: the recipient name, the message, the occasion, the
language and the timezone, plus any occasion variables the customer supplies.
This module minimizes what an order keeps, redacts personal data when it
leaves the process, replaces it with placeholders in placeholder mode and keeps
a data processing record of the providers and fields that touched each order.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from aivia.core.config.sections import PrivacyConfig
from aivia.core.models import utcnow
from aivia.core.redact import REDACTED, redact

__all__ = [
    "DataProcessingEntry",
    "DataProcessingRecord",
    "apply_placeholders",
    "minimize_order",
    "placeholder_for",
    "redact_order",
]

_PERSONAL_ORDER_FIELDS = frozenset(
    {"name", "message", "occasion", "language", "timezone"}
)
"""Order fields that carry personal data about the recipient."""

_STANDARD_ORDER_FIELDS = frozenset(
    {
        "occasion",
        "name",
        "message",
        "language",
        "profile",
        "level",
        "timezone",
        "currency",
        "amount_usd",
        "outputs",
        "variables",
        "pricing",
        "metadata",
        "idempotency_key",
        "run_id",
        "state",
        "budget_usd",
        "scheduled_for",
        "expires_at",
    }
)
"""Top-level fields an order may lawfully keep; anything else is minimized away."""

_PLACEHOLDERS = {
    "name": "[NAME]",
    "message": "[MESSAGE]",
    "occasion": "[OCCASION]",
    "language": "[LANGUAGE]",
    "timezone": "[TIMEZONE]",
}


def placeholder_for(field: str) -> str:
    """Return the placeholder that stands in for *field* in placeholder mode."""
    return _PLACEHOLDERS.get(field, f"[{field.upper()}]")


def minimize_order(
    values: Mapping[str, Any], *, declared: Sequence[str] = ()
) -> dict[str, Any]:
    """Return *values* with only the fields an order may lawfully keep.

    Personal data is minimized to the standard order fields plus the occasion
    variables named in *declared*. Anything else is dropped, so an order never
    carries data it does not need. The original mapping is not mutated.
    """
    allowed = set(_STANDARD_ORDER_FIELDS) | set(declared)
    return {key: value for key, value in values.items() if key in allowed}


def _redact_value(value: Any, privacy: PrivacyConfig) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, Mapping):
        return {key: _redact_value(item, privacy) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_value(item, privacy) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(item, privacy) for item in value)
    return value


def redact_order(order: Mapping[str, Any], privacy: PrivacyConfig) -> dict[str, Any]:
    """Return a copy of *order* with personal data redacted per *privacy*.

    In placeholder mode personal fields are replaced with placeholders. When
    personal data is not allowed they are replaced with :data:`REDACTED`.
    Otherwise they are kept with secrets and token-like strings masked. When
    ``redact_logs`` is set every string value is passed through the redactor.
    The original mapping is not mutated.
    """
    policy = privacy or PrivacyConfig()
    result: dict[str, Any] = dict(order)

    for field in _PERSONAL_ORDER_FIELDS:
        if field not in result:
            continue
        if policy.placeholder_mode:
            result[field] = placeholder_for(field)
        elif not policy.allow_personal_data:
            result[field] = REDACTED
        elif isinstance(result[field], str):
            result[field] = redact(result[field])

    if "variables" in result and isinstance(result["variables"], Mapping):
        result["variables"] = _redact_value(result["variables"], policy)

    if policy.redact_logs:
        for key, value in result.items():
            if key in _PERSONAL_ORDER_FIELDS or key == "variables":
                continue
            if isinstance(value, str):
                result[key] = redact(value)

    return result


def apply_placeholders(order: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of *order* with personal fields replaced by placeholders.

    This is the explicit form of placeholder mode: real personal values are
    replaced with stable placeholders so the order can be shown or logged
    without revealing the recipient's data.
    """
    result = dict(order)
    for field in _PERSONAL_ORDER_FIELDS:
        if field in result:
            result[field] = placeholder_for(field)
    if "variables" in result and isinstance(result["variables"], Mapping):
        result["variables"] = {
            key: placeholder_for("variables") for key in result["variables"]
        }
    return result


@dataclass(frozen=True, slots=True)
class DataProcessingEntry:
    """One record that a provider processed a set of order fields."""

    occurred_at: datetime
    provider: str
    fields: tuple[str, ...]
    purpose: str = ""
    legal_basis: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping of the entry."""
        return {
            "occurred_at": self.occurred_at.isoformat(),
            "provider": self.provider,
            "fields": list(self.fields),
            "purpose": self.purpose,
            "legal_basis": self.legal_basis,
        }


class DataProcessingRecord:
    """A record of the providers and fields that processed an order.

    This is the order's processing record: every time a provider touches a set
    of order fields the call is logged with the provider, the fields, the
    purpose and an optional legal basis. Nothing here identifies the recipient.
    """

    def __init__(self) -> None:
        self._entries: list[DataProcessingEntry] = []

    def add(
        self,
        provider: str,
        fields: Sequence[str],
        *,
        purpose: str = "",
        legal_basis: str = "",
        occurred_at: datetime | None = None,
    ) -> DataProcessingEntry:
        """Record that *provider* processed *fields* and return the entry."""
        normalized = tuple(dict.fromkeys(field for field in fields if field))
        entry = DataProcessingEntry(
            occurred_at=occurred_at if occurred_at is not None else utcnow(),
            provider=provider.strip(),
            fields=normalized,
            purpose=purpose,
            legal_basis=legal_basis,
        )
        self._entries.append(entry)
        return entry

    @property
    def entries(self) -> list[DataProcessingEntry]:
        """The recorded entries in the order they were added."""
        return list(self._entries)

    def to_dict(self) -> list[dict[str, Any]]:
        """Return every entry as JSON-compatible mappings."""
        return [entry.to_dict() for entry in self._entries]

    def clear(self) -> None:
        """Remove every recorded entry."""
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)
