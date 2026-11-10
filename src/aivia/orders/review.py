"""Order review and revision.

A reviewer can leave feedback on an order, tagging the stages the feedback
applies to. A revision re-runs only those affected stages instead of repeating
the whole pipeline, so a name correction only re-runs ``name_overlay`` and a
music complaint only re-runs ``music``.

Feedback is stored on the order's ``metadata`` column so no new table or
migration is required. Stage results are reset to ``pending`` for the affected
stages so the next run re-executes exactly them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from aivia.core.db import OrderRepository, StageRunRepository
from aivia.core.models import utcnow

__all__ = [
    "OrderFeedback",
    "OrderReview",
    "affected_stages",
    "feedback_for_order",
    "record_feedback",
    "revision_stages",
]

_FEEDBACK_KEY = "feedback"


@dataclass(frozen=True, slots=True)
class OrderFeedback:
    """One piece of feedback on an order, tied to the stages it affects."""

    order_id: str
    stage: str
    comment: str
    actor: str = ""
    occurred_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping of the entry."""
        payload: dict[str, Any] = {
            "order_id": self.order_id,
            "stage": self.stage,
            "comment": self.comment,
            "actor": self.actor,
        }
        if self.occurred_at is not None:
            payload["occurred_at"] = self.occurred_at.isoformat()
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OrderFeedback:
        """Build an entry from a mapping stored in order metadata."""
        occurred_at: datetime | None = None
        raw = data.get("occurred_at")
        if isinstance(raw, str) and raw.strip():
            occurred_at = datetime.fromisoformat(raw.strip())
        return cls(
            order_id=str(data.get("order_id", "") or ""),
            stage=str(data.get("stage", "") or ""),
            comment=str(data.get("comment", "") or ""),
            actor=str(data.get("actor", "") or ""),
            occurred_at=occurred_at,
        )


def _read_feedback(order: Mapping[str, Any]) -> list[OrderFeedback]:
    """Return the feedback entries stored on *order*'s metadata."""
    metadata = order.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        return []
    entries = metadata.get(_FEEDBACK_KEY)
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        return []
    feedbacks: list[OrderFeedback] = []
    for entry in entries:
        if isinstance(entry, Mapping):
            feedbacks.append(OrderFeedback.from_dict(entry))
    return feedbacks


def _write_feedback(
    order: Mapping[str, Any], feedbacks: Sequence[OrderFeedback]
) -> dict[str, Any]:
    """Return *order* with *feedbacks* written into its metadata."""
    metadata = dict(order.get("metadata") or {})
    metadata[_FEEDBACK_KEY] = [entry.to_dict() for entry in feedbacks]
    return dict(order, metadata=metadata)


def record_feedback(
    order: Mapping[str, Any],
    stage: str,
    comment: str,
    *,
    actor: str = "",
    occurred_at: datetime | None = None,
) -> tuple[OrderFeedback, dict[str, Any]]:
    """Append feedback for *stage* on *order*.

    Returns the new entry together with the order whose metadata now carries it.
    The order is not persisted by this function; pass the returned mapping to an
    order repository to store it.
    """
    text_stage = stage.strip()
    if not text_stage:
        raise ValueError("feedback stage must be a non-empty name")
    text_comment = comment.strip()
    if not text_comment:
        raise ValueError("feedback comment must be non-empty")
    entry = OrderFeedback(
        order_id=str(order.get("id", "")),
        stage=text_stage,
        comment=text_comment,
        actor=actor.strip(),
        occurred_at=occurred_at if occurred_at is not None else utcnow(),
    )
    feedbacks = _read_feedback(order)
    feedbacks.append(entry)
    return entry, _write_feedback(order, feedbacks)


def feedback_for_order(order: Mapping[str, Any]) -> tuple[OrderFeedback, ...]:
    """Return every feedback entry recorded on *order*, in record order."""
    return tuple(_read_feedback(order))


def affected_stages(order: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the distinct stages with feedback, in record order.

    These are the stages a revision must re-run; every other stage keeps its
    existing result.
    """
    seen: set[str] = set()
    stages: list[str] = []
    for entry in _read_feedback(order):
        if entry.stage not in seen:
            seen.add(entry.stage)
            stages.append(entry.stage)
    return tuple(stages)


revision_stages = affected_stages
"""The stages a revision re-runs: exactly the stages carrying feedback."""


class OrderReview:
    """Manage feedback on an order and the stages a revision must re-run.

    *orders* persists the order records. When *stage_runs* is supplied,
    :meth:`mark_for_revision` resets the affected stage results to
    ``pending`` so they are re-executed on the next run.
    """

    def __init__(
        self,
        orders: OrderRepository,
        stage_runs: StageRunRepository | None = None,
    ) -> None:
        self.orders = orders
        self.stage_runs = stage_runs

    def record(
        self,
        order_id: str,
        stage: str,
        comment: str,
        *,
        actor: str = "",
        occurred_at: datetime | None = None,
    ) -> OrderFeedback:
        """Record feedback for *stage* on *order* and persist the order.

        Raises :class:`ValueError` when *stage* or *comment* is empty and
        :class:`KeyError` when the order does not exist.
        """
        order = self.orders.get(order_id)
        if order is None:
            raise KeyError(f"order not found: {order_id}")
        entry, updated = record_feedback(
            order,
            stage,
            comment,
            actor=actor,
            occurred_at=occurred_at,
        )
        persisted = self.orders.update(order_id, metadata=updated["metadata"])
        if persisted is None:
            raise KeyError(f"order not found: {order_id}")
        return entry

    def feedback(self, order_id: str) -> tuple[OrderFeedback, ...]:
        """Return every feedback entry recorded on *order*, in record order."""
        order = self.orders.get(order_id)
        if order is None:
            raise KeyError(f"order not found: {order_id}")
        return feedback_for_order(order)

    def affected_stages(self, order_id: str) -> tuple[str, ...]:
        """Return the distinct stages with feedback, in record order."""
        order = self.orders.get(order_id)
        if order is None:
            raise KeyError(f"order not found: {order_id}")
        return affected_stages(order)

    def revision_stages(self, order_id: str) -> tuple[str, ...]:
        """The stages a revision re-runs: exactly the stages carrying feedback."""
        return self.affected_stages(order_id)

    def mark_for_revision(self, order_id: str) -> tuple[str, ...]:
        """Reset the affected stage results to ``pending`` so they re-run.

        Returns the stages that were marked. Orders without a linked run or
        without a configured stage repository are left unchanged.
        """
        stages = self.revision_stages(order_id)
        if not stages or self.stage_runs is None:
            return stages
        order = self.orders.get(order_id)
        if order is None:
            return stages
        run_id = order.get("run_id") or ""
        if not run_id:
            return stages
        for stage in stages:
            self.stage_runs.update(run_id, stage, status="pending")
        return stages
