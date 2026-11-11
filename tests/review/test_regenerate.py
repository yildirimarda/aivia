"""Re-running only the affected stages when a reviewer regenerates.

A reviewer leaves feedback on the stages a revision applies to (script,
visuals, voice, song). Regenerating re-runs exactly those affected stages;
every other stage keeps its existing result.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, cast

import pytest
from sqlalchemy import Engine

from aivia.core.config.ops import ReviewConfig
from aivia.core.db import (
    OrderRepository,
    RunRepository,
    StageRunRepository,
    create_database_engine,
    upgrade_database,
)
from aivia.core.db.order_state import OrderState
from aivia.core.models import StageResult
from aivia.orders import OrderReview
from aivia.review.review_state import ReviewState, ReviewStateMachine

STAGES = ("script", "visuals", "voice", "song", "render")


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

    for index, stage in enumerate(STAGES):
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


def _machine() -> ReviewStateMachine:
    return ReviewStateMachine(
        settings={
            "daily": ReviewConfig(
                approval_mode="ask",
                timeout_seconds=60,
                on_timeout="reject",
                allow_regenerate=True,
                max_regenerations=3,
            )
        }
    )


def test_regenerate_re_runs_only_affected_stages(
    orders: OrderRepository, stage_runs: StageRunRepository
) -> None:
    review = OrderReview(orders, stage_runs)
    order = _create_order_with_run(orders, stage_runs)
    run_id = order["run_id"]

    before = {stage: stage_runs.get(run_id, stage) for stage in STAGES}
    for result in before.values():
        assert result is not None
        assert result.status == "succeeded"

    # The reviewer asks for a regeneration of the affected media.
    machine = _machine()
    regenerated = machine.regenerate(
        machine.create("r1", "daily"),
        actor="reviewer",
        reason="redo the voice and song",
    )
    assert regenerated.state is ReviewState.REGENERATED
    assert regenerated.regenerated_count == 1

    # Feedback tags the stages the revision applies to.
    review.record(order["id"], "voice", "too quiet", actor="reviewer")
    review.record(order["id"], "song", "wrong genre", actor="reviewer")

    assert review.affected_stages(order["id"]) == ("voice", "song")
    assert review.revision_stages(order["id"]) == ("voice", "song")

    marked = review.mark_for_revision(order["id"])
    assert marked == ("voice", "song")

    # Only the affected stages are reset to pending so they re-run.
    assert cast(StageResult, stage_runs.get(run_id, "voice")).status == "pending"
    assert cast(StageResult, stage_runs.get(run_id, "song")).status == "pending"
    assert cast(StageResult, stage_runs.get(run_id, "script")).status == "succeeded"
    assert cast(StageResult, stage_runs.get(run_id, "visuals")).status == "succeeded"
    assert cast(StageResult, stage_runs.get(run_id, "render")).status == "succeeded"


def test_regenerate_without_feedback_re_runs_nothing(
    orders: OrderRepository, stage_runs: StageRunRepository
) -> None:
    review = OrderReview(orders, stage_runs)
    order = _create_order_with_run(orders, stage_runs)
    run_id = order["run_id"]

    machine = _machine()
    regenerated = machine.regenerate(machine.create("r2", "daily"))
    assert regenerated.state is ReviewState.REGENERATED

    assert review.affected_stages(order["id"]) == ()
    assert review.mark_for_revision(order["id"]) == ()

    for stage in STAGES:
        result = cast(StageResult, stage_runs.get(run_id, stage))
        assert result is not None
        assert result.status == "succeeded"


def test_regenerate_review_recorded_on_the_order(
    orders: OrderRepository, stage_runs: StageRunRepository
) -> None:
    review = OrderReview(orders, stage_runs)
    order = _make_order(orders, state=OrderState.DELIVERED.value)

    entry = review.record(
        order["id"], "visuals", "the palette is off", actor="reviewer"
    )
    assert entry.stage == "visuals"
    assert review.affected_stages(order["id"]) == ("visuals",)

    stored = orders.get(order["id"])
    assert stored is not None
    feedbacks = stored["metadata"]["feedback"]
    assert len(feedbacks) == 1
    assert feedbacks[0]["stage"] == "visuals"
