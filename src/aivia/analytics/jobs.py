"""Snapshot a published post's metrics at the 24h and 7d marks.

The analytics module is optional: the core never imports it. A snapshot is
stored once per (post, mark), so retrying a partially failed fetch is safe, and
any fetch error is raised as a retryable error so a job worker can attempt it
again instead of dead-lettering the snapshot.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any

from sqlalchemy import Engine

from aivia.analytics.collect import collect
from aivia.core.db import PostMetricRepository
from aivia.core.errors import RetryableError

__all__ = [
    "MARKS",
    "MARK_SEVEN_DAYS",
    "MARK_TWENTY_FOUR_HOURS",
    "snapshot_post_metrics",
]

#: Time marks at which a published post's metrics are snapshotted.
MARK_TWENTY_FOUR_HOURS = "24h"
MARK_SEVEN_DAYS = "7d"
MARKS = (MARK_TWENTY_FOUR_HOURS, MARK_SEVEN_DAYS)

#: Canonical metric names (from :func:`collect`) to the columns they populate.
_CANONICAL_TO_COLUMN: Mapping[str, str] = {
    "views": "views",
    "watch_time": "watch_time_seconds",
    "likes": "likes",
    "shares": "shares",
    "comments": "comments",
}

#: Every NOT NULL numeric column on ``post_metrics`` and its insert default.
_COLUMN_DEFAULTS: Mapping[str, Any] = {
    "impressions": 0,
    "reach": 0,
    "views": 0,
    "likes": 0,
    "comments": 0,
    "shares": 0,
    "saves": 0,
    "clicks": 0,
    "watch_time_seconds": 0.0,
    "engagement_rate": 0.0,
}


def snapshot_post_metrics(
    engine: Engine,
    post_id: str,
    platform: str,
    mark: str,
    captured_at: datetime,
    *,
    fetch_metrics: Callable[[], Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Snapshot *post*'s metrics at *mark* and store them idempotently.

    A snapshot is stored once per (post, mark): when one already exists it is
    returned unchanged, so retrying after a partial failure cannot duplicate a
    row. ``fetch_metrics`` returns the platform's raw report; the canonical
    metrics are normalized with :func:`aivia.analytics.collect.collect` and the
    full report is kept in the ``raw`` column. Any error raised while fetching
    is wrapped in :class:`~aivia.core.errors.RetryableError` so a job worker
    retries the snapshot instead of dead-lettering it.
    """
    if mark not in MARKS:
        raise ValueError(f"unknown mark {mark!r}; expected one of {MARKS!r}")

    repository = PostMetricRepository(engine)

    existing = repository.find_by_post_and_mark(post_id, mark)
    if existing is not None:
        return existing

    try:
        raw = fetch_metrics()
    except RetryableError:
        raise
    except Exception as error:
        raise RetryableError(
            f"could not fetch metrics for post {post_id!r} at {mark!r}",
            details={"post_id": post_id, "mark": mark},
        ) from error

    normalized = collect(raw)
    values: dict[str, Any] = dict(_COLUMN_DEFAULTS)
    for canonical, column in _CANONICAL_TO_COLUMN.items():
        value = normalized.get(canonical)
        if value is not None:
            values[column] = value

    return repository.create(
        post_id=post_id,
        platform=platform,
        captured_at=captured_at,
        metadata={"mark": mark},
        raw=dict(raw),
        **values,
    )
