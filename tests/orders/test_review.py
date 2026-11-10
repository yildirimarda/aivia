"""Order review and revision.

Feedback left on an order is tagged with the stages it applies to. A revision
re-runs only those affected stages instead of repeating the whole pipeline.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
from sqlalchemy import Engine

from aivia.core.db import (
    OrderRepository,
    RunRepository,
    StageRunRepository,
    create_database_engine,
    upgrade_database,
)
from aivia.core.db.order_state import OrderState
from aivia.core.models import StageResult
from aivia.orders import (
    OrderFeedback,
    OrderReview,
    affected_stages,
    feedback_for_order,
    record_feedback,
)


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
def orders(engine: Engine) -> OrderRepository:
    return OrderRepository(engine)


@pytest.fixture
def stage_runs(engine: Engine) -> StageRunRepository:
    return StageRunRepository(engine)


def _make_order(repository: OrderRepository, **values: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "occasion": "birthday",
        "name": "Ada",
        "message": "Happy birthday",
        "language": "en",
        "profile": "",
        "level": "standard",
        "timezone": "UTC",
        "currency": "USD",
        "amount_usd": 0.0,
        "budget_usd": None,
        "idempotency_key": "",
    }
    base.update(values)
    return repository.create(**cast("Mapping[str, Any]", base))


def _create_order_with_run(
    orders: OrderRepository, stage_runs: StageRunRepository
) -> dict[str, Any]:
    order = _make_order(orders, state=OrderState.DELIVERED.value)
    run: dict[str, Any] = {
        "kind": "order",
        "state": "succeeded",
        "order_id": order["id"],
        "profile": "",
        "environment": "",
        "run_dir": "",
        "cost_usd": 0.0,
    }
    runs = RunRepository(orders.engine)
    stored_run = runs.create(**cast("Mapping[str, Any]", run))
    orders.update(order["id"], run_id=stored_run.id)

    stages = ["write", "music", "voice", "visuals", "name_overlay", "render"]
    for index, stage in enumerate(stages):
        stage_runs.create(
            run_id=stored_run.id,
            stage=stage,
            status="succeeded",
            index=index,
            input_hash=f"hash-{stage}",
        )
    stored = orders.get(order["id"])
    assert stored is not None
    return stored


def test_record_feedback_appends_an_entry_to_order_metadata(
    orders: OrderRepository,
) -> None:
    order = _make_order(orders)
    entry, updated = record_feedback(
        order, "name_overlay", "the name is misspelled", actor="reviewer"
    )
    assert entry.order_id == order["id"]
    assert entry.stage == "name_overlay"
    assert entry.comment == "the name is misspelled"
    assert entry.actor == "reviewer"
    assert isinstance(entry.occurred_at, datetime)

    orders.update(order["id"], metadata=updated["metadata"])
    stored = orders.get(order["id"])
    assert stored is not None
    assert stored["metadata"] == updated["metadata"]
    feedbacks = stored["metadata"]["feedback"]
    assert len(feedbacks) == 1
    assert feedbacks[0]["stage"] == "name_overlay"


def test_record_feedback_rejects_empty_stage_and_comment(
    orders: OrderRepository,
) -> None:
    order = _make_order(orders)
    with pytest.raises(ValueError, match="feedback stage must be a non-empty name"):
        record_feedback(order, "   ", "something")
    with pytest.raises(ValueError, match="feedback comment must be non-empty"):
        record_feedback(order, "music", "   ")


def test_feedback_for_order_returns_entries_in_record_order(
    orders: OrderRepository,
) -> None:
    order = _make_order(orders)
    _first, updated = record_feedback(order, "music", "too loud", actor="alice")
    _second, updated = record_feedback(
        updated, "name_overlay", "wrong spelling", actor="bob"
    )
    _third, updated = record_feedback(updated, "music", "wrong genre", actor="alice")

    feedbacks = feedback_for_order(updated)
    assert [f.stage for f in feedbacks] == ["music", "name_overlay", "music"]
    assert [f.comment for f in feedbacks] == [
        "too loud",
        "wrong spelling",
        "wrong genre",
    ]
    assert feedbacks[0].actor == "alice"
    assert feedbacks[1].actor == "bob"


def test_affected_stages_returns_distinct_stages_in_record_order(
    orders: OrderRepository,
) -> None:
    order = _make_order(orders)
    _, updated = record_feedback(order, "music", "too loud")
    _, updated = record_feedback(updated, "name_overlay", "wrong spelling")
    _, updated = record_feedback(updated, "music", "wrong genre")
    _, updated = record_feedback(updated, "render", "too dark")

    assert affected_stages(updated) == ("music", "name_overlay", "render")


def test_affected_stages_is_empty_without_feedback(
    orders: OrderRepository,
) -> None:
    order = _make_order(orders)
    assert affected_stages(order) == ()
    assert feedback_for_order(order) == ()


class TestOrderReview:
    def test_record_persists_feedback_on_the_order(
        self,
        orders: OrderRepository,
        stage_runs: StageRunRepository,
    ) -> None:
        review = OrderReview(orders, stage_runs)
        order = _make_order(orders, state=OrderState.DELIVERED.value)
        entry = review.record(order["id"], "name_overlay", "misspelled", actor="rev")

        assert entry.stage == "name_overlay"
        assert entry.actor == "rev"
        stored = orders.get(order["id"])
        assert stored is not None
        feedbacks = stored["metadata"]["feedback"]
        assert len(feedbacks) == 1
        assert feedbacks[0]["stage"] == "name_overlay"

    def test_record_raises_on_unknown_order(
        self, orders: OrderRepository, stage_runs: StageRunRepository
    ) -> None:
        review = OrderReview(orders, stage_runs)
        with pytest.raises(KeyError, match="order not found"):
            review.record("no_such_order", "music", "too loud")

    def test_feedback_returns_recorded_entries(
        self,
        orders: OrderRepository,
        stage_runs: StageRunRepository,
    ) -> None:
        review = OrderReview(orders, stage_runs)
        order = _make_order(orders)
        review.record(order["id"], "music", "too loud")
        review.record(order["id"], "name_overlay", "wrong spelling")

        feedbacks = review.feedback(order["id"])
        assert [f.stage for f in feedbacks] == ["music", "name_overlay"]

    def test_affected_stages_are_distinct_and_ordered(
        self,
        orders: OrderRepository,
        stage_runs: StageRunRepository,
    ) -> None:
        review = OrderReview(orders, stage_runs)
        order = _make_order(orders)
        review.record(order["id"], "render", "too dark")
        review.record(order["id"], "music", "too loud")
        review.record(order["id"], "render", "still too dark")

        assert review.affected_stages(order["id"]) == ("render", "music")
        assert review.revision_stages(order["id"]) == ("render", "music")

    def test_mark_for_revision_resets_only_affected_stages(
        self,
        orders: OrderRepository,
        stage_runs: StageRunRepository,
    ) -> None:
        review = OrderReview(orders, stage_runs)
        order = _create_order_with_run(orders, stage_runs)
        run_id = order["run_id"]

        before = {
            stage: stage_runs.get(run_id, stage)
            for stage in ("write", "music", "name_overlay", "render")
        }
        for stage_result in before.values():
            assert stage_result is not None
            assert stage_result.status == "succeeded"

        review.record(order["id"], "name_overlay", "misspelled")
        review.record(order["id"], "render", "too dark")
        marked = review.mark_for_revision(order["id"])

        assert marked == ("name_overlay", "render")
        assert cast(StageResult, stage_runs.get(run_id, "write")).status == "succeeded"
        assert cast(StageResult, stage_runs.get(run_id, "music")).status == "succeeded"
        assert cast(StageResult, stage_runs.get(run_id, "voice")).status == "succeeded"
        assert (
            cast(StageResult, stage_runs.get(run_id, "visuals")).status == "succeeded"
        )
        assert (
            cast(StageResult, stage_runs.get(run_id, "name_overlay")).status
            == "pending"
        )
        assert cast(StageResult, stage_runs.get(run_id, "render")).status == "pending"

    def test_mark_for_revision_without_a_run_does_nothing(
        self,
        orders: OrderRepository,
        stage_runs: StageRunRepository,
    ) -> None:
        review = OrderReview(orders, stage_runs)
        order = _make_order(orders, state=OrderState.DELIVERED.value)
        review.record(order["id"], "music", "too loud")
        assert review.mark_for_revision(order["id"]) == ("music",)

    def test_review_without_stage_repository_does_not_mutate_stage_runs(
        self,
        orders: OrderRepository,
        stage_runs: StageRunRepository,
    ) -> None:
        review = OrderReview(orders)
        order = _create_order_with_run(orders, stage_runs)
        run_id = order["run_id"]
        review.record(order["id"], "name_overlay", "misspelled")
        review.mark_for_revision(order["id"])
        stage_result = stage_runs.get(run_id, "name_overlay")
        assert stage_result is not None
        assert stage_result.status == "succeeded"

    def test_feedback_and_affected_stages_raise_on_unknown_order(
        self,
        orders: OrderRepository,
        stage_runs: StageRunRepository,
    ) -> None:
        review = OrderReview(orders, stage_runs)
        with pytest.raises(KeyError, match="order not found"):
            review.feedback("no_such_order")
        with pytest.raises(KeyError, match="order not found"):
            review.affected_stages("no_such_order")

    def test_multiple_orders_keep_separate_feedback(
        self,
        orders: OrderRepository,
        stage_runs: StageRunRepository,
    ) -> None:
        review = OrderReview(orders, stage_runs)
        first = _make_order(orders)
        second = _make_order(orders)
        review.record(first["id"], "music", "too loud")
        review.record(second["id"], "name_overlay", "misspelled")

        assert [f.stage for f in review.feedback(first["id"])] == ["music"]
        assert [f.stage for f in review.feedback(second["id"])] == ["name_overlay"]
        assert review.affected_stages(first["id"]) == ("music",)
        assert review.affected_stages(second["id"]) == ("name_overlay",)


def test_order_feedback_round_trip_through_dict() -> None:
    now = datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC)
    entry = OrderFeedback(
        order_id="order_1",
        stage="voice",
        comment="too quiet",
        actor="rev",
        occurred_at=now,
    )
    restored = OrderFeedback.from_dict(entry.to_dict())
    assert restored == entry
    assert restored.occurred_at == now


def test_order_feedback_from_dict_handles_missing_timestamp() -> None:
    entry = OrderFeedback.from_dict(
        {"order_id": "order_1", "stage": "music", "comment": "loud", "actor": ""}
    )
    assert entry.occurred_at is None
    assert entry.stage == "music"
