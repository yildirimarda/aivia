"""The order lifecycle: legal state transitions and an audit log of every change."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy import Engine, inspect

from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.db.order_state import (
    LEGAL_ORDER_TRANSITIONS,
    IllegalOrderTransition,
    OrderAuditEntry,
    OrderAuditLog,
    OrderState,
    OrderStateMachine,
    legal_transitions,
    transition_order,
)
from aivia.core.db.repos import OrderRepository

ORDER_STATES = {
    "draft",
    "submitted",
    "paid",
    "processing",
    "delivered",
    "failed",
    "cancelled",
    "refunded",
}


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    database = tmp_path / "data" / "aivia.db"
    upgrade_database(database)
    created = create_database_engine(database)
    try:
        yield created
    finally:
        created.dispose()


@pytest.fixture
def repository(engine: Engine) -> OrderRepository:
    return OrderRepository(engine)


def test_order_states_cover_the_lifecycle() -> None:
    assert {state.value for state in OrderState} == ORDER_STATES
    assert OrderState.DRAFT.value == "draft"
    assert OrderState.DELIVERED.value == "delivered"


def test_legal_transitions_map_each_state_to_its_successors() -> None:
    assert set(LEGAL_ORDER_TRANSITIONS) == set(OrderState)
    assert OrderState.SUBMITTED in LEGAL_ORDER_TRANSITIONS[OrderState.DRAFT]
    assert OrderState.PAID in LEGAL_ORDER_TRANSITIONS[OrderState.SUBMITTED]
    assert OrderState.PROCESSING in LEGAL_ORDER_TRANSITIONS[OrderState.PAID]
    assert OrderState.DELIVERED in LEGAL_ORDER_TRANSITIONS[OrderState.PROCESSING]
    assert not LEGAL_ORDER_TRANSITIONS[OrderState.DELIVERED]
    assert not LEGAL_ORDER_TRANSITIONS[OrderState.FAILED]
    assert not LEGAL_ORDER_TRANSITIONS[OrderState.CANCELLED]
    assert not LEGAL_ORDER_TRANSITIONS[OrderState.REFUNDED]


def test_transition_order_applies_a_legal_transition_without_mutating() -> None:
    order = {"id": "order_1", "state": "draft", "occasion": "birthday"}
    original = dict(order)

    updated = transition_order(order, OrderState.SUBMITTED)

    assert order == original
    assert updated["state"] == "submitted"
    assert updated["id"] == "order_1"
    assert updated["occasion"] == "birthday"
    assert "updated_at" in updated


def test_transition_order_stamps_delivered_at_on_delivery() -> None:
    order = {"id": "order_1", "state": "processing"}
    updated = transition_order(order, OrderState.DELIVERED)

    assert updated["state"] == "delivered"
    assert updated["delivered_at"] is not None


def test_transition_order_records_errors_on_failure_and_cancellation() -> None:
    failed = transition_order(
        {"id": "order_1", "state": "paid"},
        OrderState.FAILED,
        error_code="payment_declined",
        error_message="card expired",
    )
    assert failed["state"] == "failed"
    assert failed["error_code"] == "payment_declined"
    assert failed["error_message"] == "card expired"

    cancelled = transition_order(
        {"id": "order_2", "state": "submitted"},
        OrderState.CANCELLED,
        error_code="customer_request",
        error_message="changed their mind",
    )
    assert cancelled["state"] == "cancelled"
    assert cancelled["error_code"] == "customer_request"


def test_transition_order_rejects_an_illegal_transition_without_mutating() -> None:
    order = {"id": "order_1", "state": "draft"}
    original = dict(order)

    with pytest.raises(IllegalOrderTransition):
        transition_order(order, OrderState.DELIVERED)

    assert order == original


def test_transition_order_rejects_a_transition_from_a_terminal_state() -> None:
    with pytest.raises(IllegalOrderTransition):
        transition_order({"id": "order_1", "state": "delivered"}, OrderState.PROCESSING)

    with pytest.raises(IllegalOrderTransition):
        transition_order({"id": "order_2", "state": "failed"}, OrderState.PAID)


def test_state_machine_persists_a_legal_transition(
    engine: Engine, repository: OrderRepository
) -> None:
    stored = repository.create(id="order_1", state="draft", occasion="birthday")
    machine = OrderStateMachine(engine)

    updated = machine.transition(stored, OrderState.SUBMITTED)

    assert updated["state"] == "submitted"
    current = repository.get("order_1")
    assert current is not None
    assert current["state"] == "submitted"


def test_state_machine_records_an_audit_entry_for_every_change(engine: Engine) -> None:
    entries: list[OrderAuditEntry] = []
    repository = OrderRepository(engine)
    machine = OrderStateMachine(engine, audit_log=OrderAuditLog(sink=entries.append))
    stored = repository.create(id="order_1", state="draft", occasion="birthday")

    machine.transition(stored, OrderState.SUBMITTED)
    machine.transition(stored, OrderState.PAID)
    machine.transition(stored, OrderState.PROCESSING)
    machine.transition(stored, OrderState.DELIVERED)

    assert len(entries) == 4
    assert entries[0].order_id == "order_1"
    assert entries[0].from_state == OrderState.DRAFT
    assert entries[0].to_state == OrderState.SUBMITTED
    assert entries[1].from_state == OrderState.SUBMITTED
    assert entries[1].to_state == OrderState.PAID
    assert entries[2].from_state == OrderState.PAID
    assert entries[2].to_state == OrderState.PROCESSING
    assert entries[3].from_state == OrderState.PROCESSING
    assert entries[3].to_state == OrderState.DELIVERED
    for entry in entries:
        assert isinstance(entry.occurred_at, datetime)


def test_audit_entry_records_actor_and_reason(engine: Engine) -> None:
    entries: list[OrderAuditEntry] = []
    repository = OrderRepository(engine)
    machine = OrderStateMachine(engine, audit_log=OrderAuditLog(sink=entries.append))
    stored = repository.create(id="order_1", state="draft")

    machine.fail(stored, error_code="payment_declined", error_message="card expired")

    assert len(entries) == 1
    assert entries[0].from_state == OrderState.DRAFT
    assert entries[0].to_state == OrderState.FAILED


def test_state_machine_rejects_an_illegal_transition_and_persists_nothing(
    engine: Engine,
) -> None:
    entries: list[OrderAuditEntry] = []
    repository = OrderRepository(engine)
    machine = OrderStateMachine(engine, audit_log=OrderAuditLog(sink=entries.append))
    stored = repository.create(id="order_1", state="draft")

    with pytest.raises(IllegalOrderTransition):
        machine.transition(stored, OrderState.DELIVERED)

    current = repository.get("order_1")
    assert current is not None
    assert current["state"] == "draft"
    assert entries == []


def test_terminal_states_cannot_transition(engine: Engine) -> None:
    repository = OrderRepository(engine)
    machine = OrderStateMachine(engine)

    for terminal in ("delivered", "failed", "cancelled", "refunded"):
        stored = repository.create(id=f"order_{terminal}", state=terminal)
        with pytest.raises(IllegalOrderTransition):
            machine.transition(stored, OrderState.PROCESSING)


def test_convenience_methods_advance_through_the_lifecycle(engine: Engine) -> None:
    repository = OrderRepository(engine)
    machine = OrderStateMachine(engine)
    stored = repository.create(id="order_1", state="draft")

    submitted = machine.submit(stored)
    assert submitted["state"] == "submitted"

    paid = machine.pay(stored)
    assert paid["state"] == "paid"

    processing = machine.process(stored)
    assert processing["state"] == "processing"

    delivered = machine.deliver(stored)
    assert delivered["state"] == "delivered"
    assert delivered["delivered_at"] is not None


def test_fail_and_cancel_record_errors(engine: Engine) -> None:
    repository = OrderRepository(engine)
    machine = OrderStateMachine(engine)

    failed = machine.fail(
        repository.create(id="order_1", state="paid"),
        error_code="payment_declined",
        error_message="card expired",
    )
    assert failed["state"] == "failed"
    assert failed["error_code"] == "payment_declined"

    cancelled = machine.cancel(
        repository.create(id="order_2", state="submitted"),
        error_code="customer_request",
    )
    assert cancelled["state"] == "cancelled"
    assert cancelled["error_code"] == "customer_request"


def test_refund_moves_a_paid_order_to_refunded(engine: Engine) -> None:
    repository = OrderRepository(engine)
    machine = OrderStateMachine(engine)
    stored = repository.create(id="order_1", state="paid")

    refunded = machine.refund(stored)

    assert refunded["state"] == "refunded"
    current = repository.get("order_1")
    assert current is not None
    assert current["state"] == "refunded"


def test_refund_is_rejected_from_a_non_paid_state(engine: Engine) -> None:
    repository = OrderRepository(engine)
    machine = OrderStateMachine(engine)
    stored = repository.create(id="order_1", state="submitted")

    with pytest.raises(IllegalOrderTransition):
        machine.refund(stored)

    current = repository.get("order_1")
    assert current is not None
    assert current["state"] == "submitted"


def test_transitions_property_returns_the_legal_mapping() -> None:
    assert set(LEGAL_ORDER_TRANSITIONS) == set(OrderState)


def test_can_transition_and_legal_transitions(engine: Engine) -> None:
    repository = OrderRepository(engine)
    machine = OrderStateMachine(engine)
    stored = repository.create(id="order_1", state="draft")

    assert machine.can_transition(stored, OrderState.SUBMITTED)
    assert not machine.can_transition(stored, OrderState.DELIVERED)
    assert OrderState.SUBMITTED in machine.legal_transitions(stored)
    assert OrderState.DELIVERED not in machine.legal_transitions(stored)
    assert OrderState.SUBMITTED in legal_transitions("draft")
    assert OrderState.DELIVERED not in legal_transitions("draft")


def test_orders_table_carries_the_state_column(engine: Engine) -> None:
    columns = {column["name"] for column in inspect(engine).get_columns("orders")}
    assert "state" in columns
    assert "updated_at" in columns
    assert "delivered_at" in columns
    assert "error_code" in columns
    assert "error_message" in columns
