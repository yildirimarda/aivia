"""Reading time estimates and target length ranges for prose and verse.

A writing mode accepts a length as a preset name, a target number of seconds or
an explicit range. The helpers here turn any of those shapes into a
:class:`LengthRange`, count the words in a script and estimate how long it takes
to read. The estimate is deterministic and offline so the same text always
yields the same duration.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from aivia.core.errors import ConfigError

__all__ = [
    "DEFAULT_WORDS_PER_SECOND",
    "LENGTH_PRESETS",
    "LengthRange",
    "coerce_length_range",
    "count_words",
    "reading_time_seconds",
    "resolve_length_range",
]

DEFAULT_WORDS_PER_SECOND = 2.5
"""Fallback reading speed in words per second when no rate is supplied."""

LENGTH_PRESETS: Mapping[str, tuple[float, float]] = MappingProxyType(
    {
        "micro": (0.0, 10.0),
        "short": (0.0, 30.0),
        "medium": (15.0, 60.0),
        "long": (45.0, 180.0),
        "epic": (90.0, 600.0),
    }
)
"""Named target ranges in seconds, from shortest to longest."""

_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_ZERO = 0.0
_MIN_KEYS = ("min_seconds", "min", "minimum")
_MAX_KEYS = ("max_seconds", "max", "maximum")
_TARGET_KEYS = ("target_seconds", "target", "seconds", "duration", "length")
_TOLERANCE_KEYS = ("tolerance", "tolerance_seconds")


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ConfigError(
            f"{field} must be a number",
            details={"field": field, "value": repr(value)},
        )
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            raise ConfigError(
                f"{field} must be a number",
                details={"field": field, "value": value},
            ) from None
    else:
        raise ConfigError(
            f"{field} must be a number",
            details={"field": field, "value": repr(value)},
        )
    if not math.isfinite(number) or number < 0:
        raise ConfigError(
            f"{field} must be a non-negative number",
            details={"field": field, "value": number},
        )
    return number


def _first(value: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in value:
            return value[key]
    return None


@dataclass(frozen=True, slots=True)
class LengthRange:
    """An inclusive range of reading times in seconds."""

    min_seconds: float
    max_seconds: float

    def __post_init__(self) -> None:
        if self.min_seconds < _ZERO:
            raise ValueError("minimum reading time must not be negative")
        if self.max_seconds < self.min_seconds:
            raise ValueError("maximum reading time must not be below the minimum")

    @property
    def minimum(self) -> float:
        """Alias for :attr:`min_seconds`."""
        return self.min_seconds

    @property
    def maximum(self) -> float:
        """Alias for :attr:`max_seconds`."""
        return self.max_seconds

    def contains(self, seconds: float) -> bool:
        """Return whether *seconds* falls inside the inclusive range."""
        return self.min_seconds <= seconds <= self.max_seconds

    def to_dict(self) -> dict[str, float]:
        """Return the range as a JSON-compatible mapping."""
        return {"min_seconds": self.min_seconds, "max_seconds": self.max_seconds}


def _from_target(center: float, tolerance: float) -> LengthRange:
    span = center * tolerance
    return LengthRange(max(0.0, center - span), center + span)


def _from_mapping(value: Mapping[str, Any]) -> LengthRange:
    minimum = _first(value, _MIN_KEYS)
    maximum = _first(value, _MAX_KEYS)
    target = _first(value, _TARGET_KEYS)
    tolerance = _first(value, _TOLERANCE_KEYS)
    if target is not None:
        center = _number(target, "length")
        span = (
            _number(tolerance, "length tolerance")
            if tolerance is not None
            else center * 0.2
        )
        low = _number(minimum, "min_seconds") if minimum is not None else center - span
        high = _number(maximum, "max_seconds") if maximum is not None else center + span
        return LengthRange(max(0.0, low), high)
    if minimum is None and maximum is None:
        raise ConfigError(
            "length must define min_seconds and max_seconds or a target",
            details={"length": dict(value)},
        )
    low = _number(minimum, "min_seconds") if minimum is not None else 0.0
    high = _number(maximum, "max_seconds") if maximum is not None else math.inf
    return LengthRange(low, high)


def coerce_length_range(value: Any) -> LengthRange:
    """Turn a preset name, target, pair or mapping into a :class:`LengthRange`.

    A bare number is a target in seconds with a twenty percent tolerance, while
    a two-item sequence is an explicit ``(min, max)`` pair.
    """
    if isinstance(value, LengthRange):
        return value
    if isinstance(value, Mapping):
        return _from_mapping(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ConfigError("length must not be empty")
        preset = LENGTH_PRESETS.get(text.casefold())
        if preset is not None:
            return LengthRange(*preset)
        try:
            return _from_target(float(text), 0.2)
        except ValueError:
            raise ConfigError(
                f"unknown length preset: {value!r} "
                f"(known: {', '.join(sorted(LENGTH_PRESETS))})",
                details={"length": value},
            ) from None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _from_target(float(value), 0.2)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = list(value)
        if len(items) != 2:
            raise ConfigError(
                "length range must have exactly two values",
                details={"length": value},
            )
        return LengthRange(
            _number(items[0], "min_seconds"),
            _number(items[1], "max_seconds"),
        )
    raise ConfigError(
        f"cannot read a length range from {type(value).__name__}",
        details={"length": repr(value)},
    )


def resolve_length_range(
    value: Any = None,
    *,
    min_seconds: float | None = None,
    max_seconds: float | None = None,
) -> LengthRange | None:
    """Resolve an optional length with explicit bound overrides.

    ``None`` means no target was requested, so the caller should estimate the
    reading time without rejecting the script. An explicit bound overrides the
    matching side of the base range while keeping the other side.
    """
    if value is None and min_seconds is None and max_seconds is None:
        return None
    base = coerce_length_range(value) if value is not None else None
    if min_seconds is None and max_seconds is None:
        return base
    low = (
        _number(min_seconds, "min_seconds")
        if min_seconds is not None
        else (base.min_seconds if base is not None else 0.0)
    )
    high = (
        _number(max_seconds, "max_seconds")
        if max_seconds is not None
        else (base.max_seconds if base is not None else math.inf)
    )
    return LengthRange(low, high)


def count_words(text: str | None) -> int:
    """Return the number of words in *text*, ignoring punctuation."""
    if not isinstance(text, str):
        return 0
    return len(_WORD.findall(text))


def reading_time_seconds(
    text: str | None,
    *,
    words_per_second: float | None = None,
) -> float:
    """Estimate how long *text* takes to read, in seconds."""
    rate = (
        DEFAULT_WORDS_PER_SECOND
        if words_per_second is None
        else float(words_per_second)
    )
    if not math.isfinite(rate) or rate <= 0:
        raise ConfigError(
            "words_per_second must be a positive number",
            details={"words_per_second": words_per_second},
        )
    return count_words(text) / rate
