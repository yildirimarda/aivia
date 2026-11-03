"""Shared time-window parsing for community topic sources.

Reddit and Hacker News both accept a coarse window such as a day or a week.
This module keeps the accepted names and aliases in one place so the two
adapters cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

from aivia.core.errors import ConfigError

__all__ = [
    "DEFAULT_WINDOW",
    "WINDOW_NAMES",
    "WINDOW_SECONDS",
    "derive_window",
    "normalize_window",
    "window_bounds",
    "window_from_metadata",
]

WINDOW_NAMES = ("hour", "day", "week", "month", "year", "all")
"""Every window a community source accepts, shortest first."""

DEFAULT_WINDOW = "day"
"""The default window for a trending source."""

WINDOW_SECONDS = {
    "hour": 3600,
    "day": 86400,
    "week": 604800,
    "month": 2592000,
    "year": 31536000,
}
"""Approximate durations for the coarse windows, used to derive API filters."""

_WINDOW_ALIASES = {
    "24h": "day",
    "7d": "week",
    "30d": "month",
    "365d": "year",
    "all-time": "all",
    "all_time": "all",
    "daily": "day",
    "forever": "all",
    "hourly": "hour",
    "monthly": "month",
    "today": "day",
    "weekly": "week",
    "yearly": "year",
}
_WINDOW_KEYS = (
    "window",
    "timeframe",
    "time_range",
    "t",
    "sort_window",
    "since_window",
)


def normalize_window(
    value: Any,
    *,
    default: str = DEFAULT_WINDOW,
) -> str:
    """Return a validated window name or raise :class:`ConfigError`.

    ``None`` and an empty string select *default*. Common aliases such as
    ``daily``, ``7d`` and ``all-time`` are accepted because config files tend
    to use the shorter names.
    """
    if value is None:
        return default
    if not isinstance(value, str):
        raise ConfigError(
            "window must be a string",
            details={"field": "window"},
        )
    text = value.strip().lower().replace("_", "-")
    if not text:
        return default
    text = _WINDOW_ALIASES.get(text, text)
    if text not in WINDOW_NAMES:
        allowed = ", ".join(WINDOW_NAMES)
        raise ConfigError(
            f"unknown window: {value!r} (allowed: {allowed})",
            details={"field": "window"},
        )
    return text


def window_from_metadata(
    metadata: Mapping[str, Any],
    *,
    default: str = DEFAULT_WINDOW,
) -> str | None:
    """Return a window named in request metadata, or ``None`` when absent."""
    for key in _WINDOW_KEYS:
        if key in metadata and metadata[key] is not None:
            return normalize_window(metadata[key], default=default)
    return None


def derive_window(
    start: datetime | None,
    end: datetime | None,
    *,
    default: str = DEFAULT_WINDOW,
    now: datetime | None = None,
) -> str:
    """Return the smallest coarse window covering *start* through *end*.

    A request that names no bound keeps *default*. A request with only an end
    bound cannot be narrowed from a start, so ``all`` is returned.
    """
    if start is None and end is None:
        return default
    if start is None:
        return "all"
    finish = end or now or datetime.now(UTC)
    span = (finish - start).total_seconds()
    if span <= 0:
        return "hour"
    for name in ("hour", "day", "week", "month", "year"):
        if span <= WINDOW_SECONDS[name]:
            return name
    return "all"


def window_bounds(
    window: str,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    now: datetime | None = None,
) -> tuple[datetime | None, datetime | None]:
    """Return the exact bounds for *window* unless the request supplies them."""
    if start is not None or end is not None:
        return start, end
    if window == "all":
        return None, None
    finish = end or now or datetime.now(UTC)
    return finish - timedelta(seconds=WINDOW_SECONDS[window]), finish
