"""Normalize raw platform metrics into a canonical shape.

A platform reports metrics under its own names. :func:`collect` maps a raw
metrics mapping onto the canonical set used across the analytics module:
views, watch time, likes, shares, comments and followers. Any metric that is
not part of the canonical set is unknown and maps to ``None``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["CANONICAL_METRICS", "collect"]

#: Canonical metric names every platform report is normalized to.
CANONICAL_METRICS: tuple[str, ...] = (
    "views",
    "watch_time",
    "likes",
    "shares",
    "comments",
    "followers",
)


def collect(raw: Mapping[str, Any]) -> dict[str, Any | None]:
    """Return *raw* metrics normalized to the canonical set.

    Each canonical metric is reported with its value from *raw* when present,
    or ``None`` when absent. Metrics in *raw* that are not part of the
    canonical set are unknown and are not collected, so they do not appear in
    the result.
    """
    return {name: raw.get(name) for name in CANONICAL_METRICS}
