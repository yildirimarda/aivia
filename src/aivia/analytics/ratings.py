"""Capture and look up human ratings for providers.

A rating is a single score a human gives a provider, stored in the ``ratings``
table keyed by ``(subject_kind='provider', subject_id=<provider name>)``. The
unique subject index keeps exactly one rating per provider, so capturing a
rating upserts it.

The router consumes these scores through a mapping or lookup callable so the
``best_quality`` and ``weighted_ab`` strategies are driven by human judgement
instead of static ranks and configured weights alone. The analytics module owns
capture; the core never imports it.
"""

from __future__ import annotations

import math
from typing import Any

from sqlalchemy import Engine

from aivia.core.db import RatingRepository


def capture_rating(
    engine: Engine,
    provider: str,
    score: float,
    *,
    rater: str = "",
    comment: str = "",
    run_id: str = "",
) -> dict[str, Any]:
    """Upsert a human *rating* for *provider* and return the stored row.

    A provider carries at most one rating: when one already exists it is
    replaced, otherwise a new row is inserted. ``score`` must be a finite
    number.
    """
    value = float(score)
    if not math.isfinite(value):
        raise ValueError(f"rating score must be finite, got {score!r}")
    repository = RatingRepository(engine)
    existing = repository.list(subject_kind="provider", subject_id=provider)
    values = {
        "subject_kind": "provider",
        "subject_id": provider,
        "score": value,
        "rater": rater,
        "comment": comment,
        "run_id": run_id,
    }
    if existing:
        updated = repository.update(existing[0]["id"], **values)
        assert updated is not None
        return updated
    return repository.create(values)


def rating_for(engine: Engine, provider: str) -> float | None:
    """Return the stored score for *provider*, or ``None`` when it is unrated."""
    repository = RatingRepository(engine)
    rows = repository.list(subject_kind="provider", subject_id=provider)
    if not rows:
        return None
    score = rows[0]["score"]
    return None if score is None else float(score)


def all_ratings(engine: Engine, kind: str = "provider") -> dict[str, float]:
    """Return every rating for *subject_kind*, keyed by subject id."""
    repository = RatingRepository(engine)
    ratings: dict[str, float] = {}
    for row in repository.list(subject_kind=kind):
        score = row["score"]
        subject = row["subject_id"]
        if score is not None and subject:
            ratings[subject] = float(score)
    return ratings
