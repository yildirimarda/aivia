"""The legal lifecycle of an order, persisted with an audit log of every change.

An order starts ``draft``. It may be submitted for payment, paid, processed and
delivered. It may fail or be cancelled from any non-terminal state, and a paid
order may be refunded. Terminal states never leave themselves.

The mapping in :data:`LEGAL_ORDER_TRANSITIONS` is the single source of truth.
:func:`transition_order` applies one transition to a mapping without touching
the database, while :class:`OrderStateMachine` reads the current record from an
:class:`~aivia.core.db.repos.OrderRepository`, validates the transition, writes
the result back and records every accepted change in the order audit log. An
illegal transition raises :class:`IllegalOrderTransition` and persists nothing.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from sqlalchemy import Engine

from aivia.core.db.repos import (
    AssetRepository,
    CompositionRepository,
    DeliveryRepository,
    OrderRepository,
    RunRepository,
)
from aivia.core.errors import AiviaError
from aivia.core.logging import get_logger
from aivia.core.models import utcnow

__all__ = [
    "LEGAL_ORDER_TRANSITIONS",
    "ORDER_STATE_TRANSITIONS",
    "IllegalOrderTransition",
    "InvalidOrderTransition",
    "OrderAuditEntry",
    "OrderAuditLog",
    "OrderAuditSink",
    "OrderNotFoundError",
    "OrderState",
    "OrderStateError",
    "OrderStateMachine",
    "OrderStateStore",
    "can_transition",
    "is_legal_transition",
    "legal_transitions",
    "transition_order",
]

_ORDER_STATE_LOGGER = "aivia.orders.state"


class OrderStateError(AiviaError):
    """Base class for order lifecycle errors."""

    code = "order_state"


class OrderNotFoundError(OrderStateError):
    """The requested order does not exist in the repository."""

    code = "order_not_found"


class IllegalOrderTransition(OrderStateError, ValueError):
    """A transition the state machine does not allow.

    The error is also a :class:`ValueError` so callers that validate a
    transition before applying it can use the usual exception handling.
    """

    code = "illegal_order_transition"

    def __init__(self, current: OrderState, target: OrderState) -> None:
        super().__init__(
            f"cannot transition an order from {current.value} to {target.value}",
            details={
                "current": current.value,
                "target": target.value,
                "allowed": sorted(
                    state.value for state in LEGAL_ORDER_TRANSITIONS[current]
                ),
            },
        )
        self.current = current
        self.target = target


InvalidOrderTransition = IllegalOrderTransition
"""Alias for :class:`IllegalOrderTransition`."""


class OrderState(StrEnum):
    """The lifecycle state of an order."""

    DRAFT = "draft"
    SUBMITTED = "submitted"
    PAID = "paid"
    PROCESSING = "processing"
    DELIVERED = "delivered"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"


_ORDER_STATE_ALIASES = {
    "created": OrderState.DRAFT,
    "new": OrderState.DRAFT,
    "open": OrderState.DRAFT,
    "pending": OrderState.SUBMITTED,
    "awaiting_payment": OrderState.SUBMITTED,
    "unpaid": OrderState.SUBMITTED,
    "settled": OrderState.PAID,
    "cleared": OrderState.PAID,
    "producing": OrderState.PROCESSING,
    "in_progress": OrderState.PROCESSING,
    "working": OrderState.PROCESSING,
    "complete": OrderState.DELIVERED,
    "completed": OrderState.DELIVERED,
    "success": OrderState.DELIVERED,
    "done": OrderState.DELIVERED,
    "error": OrderState.FAILED,
    "failure": OrderState.FAILED,
    "canceled": OrderState.CANCELLED,
    "aborted": OrderState.CANCELLED,
    "reversed": OrderState.REFUNDED,
}


_TERMINAL_ORDER_STATES = frozenset(
    {
        OrderState.DELIVERED,
        OrderState.FAILED,
        OrderState.CANCELLED,
        OrderState.REFUNDED,
    }
)

LEGAL_ORDER_TRANSITIONS: Mapping[OrderState, frozenset[OrderState]] = MappingProxyType(
    {
        OrderState.DRAFT: frozenset(
            {OrderState.SUBMITTED, OrderState.CANCELLED, OrderState.FAILED}
        ),
        OrderState.SUBMITTED: frozenset(
            {OrderState.PAID, OrderState.CANCELLED, OrderState.FAILED}
        ),
        OrderState.PAID: frozenset(
            {
                OrderState.PROCESSING,
                OrderState.CANCELLED,
                OrderState.FAILED,
                OrderState.REFUNDED,
            }
        ),
        OrderState.PROCESSING: frozenset(
            {OrderState.DELIVERED, OrderState.CANCELLED, OrderState.FAILED}
        ),
        OrderState.DELIVERED: frozenset(),
        OrderState.FAILED: frozenset(),
        OrderState.CANCELLED: frozenset(),
        OrderState.REFUNDED: frozenset(),
    }
)
"""Every order state mapped to the states it may move to."""

_ORDER_STATE_TRANSITIONS = LEGAL_ORDER_TRANSITIONS
"""Alias for :data:`LEGAL_ORDER_TRANSITIONS`."""

ORDER_STATE_TRANSITIONS = LEGAL_ORDER_TRANSITIONS
"""Alias for :data:`LEGAL_ORDER_TRANSITIONS`."""

_PERSISTED_FIELDS = (
    "state",
    "updated_at",
    "delivered_at",
    "error_code",
    "error_message",
)


def _state(value: OrderState | str) -> OrderState:
    if isinstance(value, OrderState):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        return _ORDER_STATE_ALIASES.get(text, OrderState(text))
    return value


def legal_transitions(state: OrderState | str) -> frozenset[OrderState]:
    """Return the states an order in *state* may move to."""
    return LEGAL_ORDER_TRANSITIONS[_state(state)]


def can_transition(source: OrderState | str, target: OrderState | str) -> bool:
    """Whether an order may move from *source* to *target*."""
    return _state(target) in LEGAL_ORDER_TRANSITIONS[_state(source)]


is_legal_transition = can_transition
"""Alias for :func:`can_transition`."""


def _moment(value: datetime | None) -> datetime:
    return value if value is not None else utcnow()


def transition_order(
    order: Mapping[str, Any],
    target: OrderState | str,
    *,
    now: datetime | None = None,
    error_code: str = "",
    error_message: str = "",
) -> dict[str, Any]:
    """Return *order* moved to *target*, or raise when the move is illegal.

    The original mapping is never mutated. ``updated_at`` is stamped on every
    transition and ``delivered_at`` when an order reaches ``delivered``. Failure
    and cancellation record the optional error code and message.
    """
    current = _state(order.get("state", OrderState.DRAFT))
    destination = _state(target)
    if destination not in LEGAL_ORDER_TRANSITIONS[current]:
        raise IllegalOrderTransition(current, destination)

    moment = _moment(now)
    changes: dict[str, Any] = {"state": destination, "updated_at": moment}

    if destination is OrderState.DELIVERED:
        changes["delivered_at"] = moment
    elif destination in {OrderState.FAILED, OrderState.CANCELLED}:
        changes["error_code"] = error_code.strip()
        changes["error_message"] = error_message.strip()

    return dict(order, **changes)


OrderAuditSink = Callable[["OrderAuditEntry"], None]
"""Receives each recorded entry when an :class:`OrderAuditLog` uses a custom sink."""


@dataclass(frozen=True, slots=True)
class OrderAuditEntry:
    """One order state change: who moved which order from which state to which."""

    order_id: str
    from_state: OrderState
    to_state: OrderState
    occurred_at: datetime
    actor: str = ""
    reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping of the entry."""
        payload: dict[str, Any] = {
            "order_id": self.order_id,
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "occurred_at": self.occurred_at.isoformat(),
            "actor": self.actor,
            "reason": self.reason,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


class OrderAuditLog:
    """Record the state change of every order.

    The log is a data object plus a sink. By default it writes one JSON object
    per change through the ``aivia.orders.state`` logger; tests and alternate
    exporters can pass a callable sink instead. Nothing here touches a provider.
    """

    def __init__(
        self,
        *,
        logger: logging.Logger | None = None,
        sink: OrderAuditSink | None = None,
    ) -> None:
        if sink is not None and not callable(sink):
            raise TypeError("sink must be callable")
        self._logger = logger or get_logger(_ORDER_STATE_LOGGER)
        self._sink = sink

    @property
    def sink(self) -> OrderAuditSink | None:
        """The custom sink, when one was configured."""
        return self._sink

    def record(
        self,
        order_id: str,
        from_state: OrderState | str,
        to_state: OrderState | str,
        *,
        occurred_at: datetime | None = None,
        actor: str = "",
        reason: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> OrderAuditEntry:
        """Record one change and return the entry that was emitted."""
        entry = OrderAuditEntry(
            order_id=_text(order_id),
            from_state=_state(from_state),
            to_state=_state(to_state),
            occurred_at=_moment(occurred_at),
            actor=_text(actor),
            reason=_text(reason),
            metadata=dict(metadata or {}),
        )
        if self._sink is not None:
            self._sink(entry)
        else:
            self._logger.info(
                "%s",
                json.dumps(
                    entry.to_dict(),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                    default=str,
                ),
            )
        return entry


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _remove_path(path: str | Path | None) -> None:
    """Remove one file or directory, ignoring missing paths and errors."""
    text = str(path).strip() if path is not None else ""
    if not text:
        return
    target = Path(text)
    try:
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        else:
            target.unlink(missing_ok=True)
    except OSError:
        pass


class OrderStateMachine:
    """Move orders through legal states, persist them and audit every change.

    *repository* may be an :class:`OrderRepository` or an SQLAlchemy engine. The
    machine reads the latest record before each transition, so a stale record
    held by a caller cannot be used to skip a state. ``clock`` supplies the
    timestamp for the transition and defaults to
    :func:`~aivia.core.models.base.utc`. Every accepted transition is recorded
    in *audit_log*; when none is supplied a logger-based log is used.
    """

    def __init__(
        self,
        repository: OrderRepository | Engine,
        *,
        clock: Callable[[], datetime] = utcnow,
        audit_log: OrderAuditLog | None = None,
    ) -> None:
        if isinstance(repository, Engine):
            repository = OrderRepository(repository)
        self.repository = repository
        self.clock = clock
        self.audit_log = audit_log or OrderAuditLog()

    @property
    def transitions(self) -> Mapping[OrderState, frozenset[OrderState]]:
        """The legal transition mapping used by this machine."""
        return LEGAL_ORDER_TRANSITIONS

    def _state_of(self, order: Mapping[str, Any] | OrderState | str) -> OrderState:
        if isinstance(order, Mapping):
            return _state(order.get("state", OrderState.DRAFT))
        return _state(order)

    def legal_transitions(
        self, order: Mapping[str, Any] | OrderState | str
    ) -> frozenset[OrderState]:
        """Return the states the given order or state may move to."""
        return LEGAL_ORDER_TRANSITIONS[self._state_of(order)]

    def can_transition(
        self, order: Mapping[str, Any] | OrderState | str, target: OrderState | str
    ) -> bool:
        """Whether *order* may move to *target*."""
        return _state(target) in LEGAL_ORDER_TRANSITIONS[self._state_of(order)]

    def create(
        self, order: Mapping[str, Any] | None = None, **values: Any
    ) -> dict[str, Any]:
        """Persist a new order and return the stored record."""
        return self.repository.create(order, **values)

    def get(self, order_id: str) -> dict[str, Any] | None:
        """Return the stored order with *order_id*, if any."""
        return self.repository.get(order_id)

    def transition(
        self,
        order: Mapping[str, Any] | str,
        target: OrderState | str,
        *,
        now: datetime | None = None,
        error_code: str = "",
        error_message: str = "",
    ) -> dict[str, Any]:
        """Persist one legal transition, audit it and return the stored record."""
        current = self._current(order)
        updated = transition_order(
            current,
            target,
            now=now if now is not None else self.clock(),
            error_code=error_code,
            error_message=error_message,
        )
        changes = {
            field: updated[field]
            for field in _PERSISTED_FIELDS
            if updated.get(field) != current.get(field)
        }
        persisted = self.repository.update(current["id"], **changes)
        if persisted is None:
            raise OrderNotFoundError(
                f"order not found: {current['id']}",
                details={"order_id": current["id"]},
            )
        self.audit_log.record(
            current["id"],
            current.get("state", OrderState.DRAFT),
            updated["state"],
            occurred_at=updated.get("updated_at"),
            metadata={"changed": sorted(changes)},
        )
        return persisted

    def submit(
        self, order: Mapping[str, Any] | str, *, now: datetime | None = None
    ) -> dict[str, Any]:
        """Move a draft order to ``submitted``."""
        return self.transition(order, OrderState.SUBMITTED, now=now)

    def pay(
        self, order: Mapping[str, Any] | str, *, now: datetime | None = None
    ) -> dict[str, Any]:
        """Move a submitted order to ``paid``."""
        return self.transition(order, OrderState.PAID, now=now)

    def process(
        self, order: Mapping[str, Any] | str, *, now: datetime | None = None
    ) -> dict[str, Any]:
        """Move a paid order to ``processing``."""
        return self.transition(order, OrderState.PROCESSING, now=now)

    def deliver(
        self, order: Mapping[str, Any] | str, *, now: datetime | None = None
    ) -> dict[str, Any]:
        """Move a processing order to ``delivered``."""
        return self.transition(order, OrderState.DELIVERED, now=now)

    def fail(
        self,
        order: Mapping[str, Any] | str,
        *,
        error_code: str = "",
        error_message: str = "",
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Move a non-terminal order to ``failed``."""
        return self.transition(
            order,
            OrderState.FAILED,
            error_code=error_code,
            error_message=error_message,
            now=now,
        )

    def cancel(
        self,
        order: Mapping[str, Any] | str,
        *,
        error_code: str = "",
        error_message: str = "",
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Move a non-terminal order to ``cancelled``."""
        return self.transition(
            order,
            OrderState.CANCELLED,
            error_code=error_code,
            error_message=error_message,
            now=now,
        )

    def refund(
        self, order: Mapping[str, Any] | str, *, now: datetime | None = None
    ) -> dict[str, Any]:
        """Move a paid order to ``refunded``."""
        return self.transition(order, OrderState.REFUNDED, now=now)

    def delete(
        self,
        order_id: str,
        *,
        delete_final_outputs: bool = False,
    ) -> dict[str, Any] | None:
        """Delete an order and its files, auditing without personal data.

        Delivery files are always removed. Non-pinned asset and composition
        files are removed; pinned finals are kept unless
        ``delete_final_outputs`` is set. Run directories (inputs and
        intermediates) are removed. The order row is deleted from the
        database, cascading to deliveries, posts, runs and their children.
        Assets and compositions lack cascade on run_id, so they are removed
        explicitly. An audit entry is recorded with no personal data.
        """
        try:
            current = self._current(order_id)
        except OrderNotFoundError:
            return None
        engine = self.repository.engine

        run_repo = RunRepository(engine)
        asset_repo = AssetRepository(engine)
        composition_repo = CompositionRepository(engine)
        delivery_repo = DeliveryRepository(engine)

        runs = run_repo.list(order_id=order_id)
        run_ids = [run.id for run in runs]

        assets: list[Any] = []
        compositions: list[Any] = []
        for run_id in run_ids:
            assets.extend(asset_repo.list(run_id=run_id))
            compositions.extend(composition_repo.list(run_id=run_id))

        deliveries = delivery_repo.list(order_id=order_id)

        # Remove files on disk before the database rows.
        for delivery in deliveries:
            _remove_path(delivery.path)
            for file_path in delivery.files:
                _remove_path(file_path)

        for asset in assets:
            if not asset.pinned or delete_final_outputs:
                _remove_path(asset.path)

        for composition in compositions:
            if not composition.pinned or delete_final_outputs:
                _remove_path(composition.path)

        for run in runs:
            _remove_path(run.run_dir)

        # Delete the order row; FK cascade removes deliveries, posts, runs and
        # their children. Assets and compositions lack cascade on run_id, so
        # remove them explicitly, but only when their file was removed.
        self.repository.delete(order_id)
        for asset in assets:
            if not asset.pinned or delete_final_outputs:
                asset_repo.delete(asset.id)
        for composition in compositions:
            if not composition.pinned or delete_final_outputs:
                composition_repo.delete(composition.id)

        # Record the audit entry with no personal data.
        state = current.get("state", OrderState.DRAFT)
        self.audit_log.record(
            current["id"],
            state,
            state,
            reason="deleted",
            metadata={"scope": "order"},
        )
        return current

    def _current(self, order: Mapping[str, Any] | str) -> dict[str, Any]:
        order_id = order["id"] if isinstance(order, Mapping) else order
        current = self.repository.get(order_id)
        if current is None:
            if isinstance(order, Mapping):
                return dict(order)
            raise OrderNotFoundError(
                f"order not found: {order_id}",
                details={"order_id": order_id},
            )
        return current


OrderStateStore = OrderStateMachine
"""Alias for :class:`OrderStateMachine`."""
