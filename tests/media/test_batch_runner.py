"""The reusable batch runner caps concurrency, stops on budget and records items."""

from __future__ import annotations

import asyncio
import json
import math
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine

from aivia.core.budget import BudgetGuard
from aivia.core.config.budget import BudgetConfig
from aivia.core.cost import CostTracker
from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.errors import BudgetError
from aivia.media.batch_runner import (
    BatchItemStatus,
    BatchManifest,
    BatchRunner,
    run_batch,
)


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


@pytest.fixture
def engine() -> Iterator[Engine]:
    created = create_database_engine(url="sqlite+pysqlite:///:memory:")
    upgrade_database(engine=created)
    try:
        yield created
    finally:
        created.dispose()


def test_concurrency_cap_limits_simultaneous_items() -> None:
    active = 0
    peak = 0
    lock = asyncio.Lock()

    async def handler(item: int) -> dict[str, Any]:
        nonlocal active, peak
        async with lock:
            active += 1
            peak = max(peak, active)
        await asyncio.sleep(0.01)
        async with lock:
            active -= 1
        return {"value": item, "cost_usd": 0.0}

    result = _run(run_batch(range(12), handler, concurrency=3))

    assert peak == 3
    assert [item.key for item in result.items] == [str(index) for index in range(12)]
    assert all(item.succeeded for item in result.items)
    assert result.ok


def test_sync_handlers_also_respect_the_concurrency_cap() -> None:
    active = 0
    peak = 0
    lock = threading.Lock()

    def handler(item: int) -> int:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return item

    result = _run(run_batch(range(8), handler, concurrency=2))

    assert peak == 2
    assert [item.value for item in result.items] == list(range(8))


def test_batch_budget_stop_marks_remaining_items(tmp_path: Path) -> None:
    calls: list[int] = []
    manifest = tmp_path / "batch.json"

    async def handler(item: int) -> dict[str, Any]:
        calls.append(item)
        return {"value": item, "cost_usd": 0.02}

    runner = BatchRunner(
        handler,
        concurrency=1,
        max_cost_usd=0.05,
        manifest=manifest,
    )
    result = _run(runner.run(range(10)))

    assert calls == [0, 1, 2]
    assert len(result.succeeded) == 3
    assert len(result.budget_stopped) == 7
    assert result.stopped_for_budget
    assert math.isclose(result.total_cost_usd, 0.06)
    assert [item.status for item in result.items[3:]] == [
        BatchItemStatus.BUDGET_STOPPED
    ] * 7

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    statuses = [entry["status"] for entry in payload["items"].values()]
    assert statuses.count("succeeded") == 3
    assert statuses.count("budget_stopped") == 7
    assert payload["items"]["0"]["value"] == {"value": 0, "cost_usd": 0.02}


def test_budget_guard_stops_batch_before_an_over_budget_item(
    engine: Engine,
) -> None:
    tracker = CostTracker(engine)
    guard = BudgetGuard(BudgetConfig(per_run=0.05), tracker)

    async def handler(item: int) -> dict[str, Any]:
        tracker.record(
            provider="acme",
            operation="batch",
            units=1.0,
            unit_kind="item",
            amount_usd=0.02,
            run_id="run-1",
        )
        return {"cost_usd": 0.02}

    result = _run(
        BatchRunner(
            handler,
            concurrency=1,
            budget=guard,
            run_id="run-1",
            estimate=0.02,
        ).run(range(10))
    )

    assert len(result.succeeded) == 2
    assert len(result.budget_stopped) == 8
    assert result.stopped_for_budget
    assert math.isclose(tracker.total(run_id="run-1"), 0.04)


def test_budget_error_from_handler_stops_the_whole_batch() -> None:
    calls: list[int] = []

    async def handler(item: int) -> dict[str, Any]:
        calls.append(item)
        if item == 1:
            raise BudgetError("the provider would exceed the budget")
        return {"cost_usd": 0.01}

    result = _run(BatchRunner(handler, concurrency=1).run(range(5)))

    assert calls == [0, 1]
    assert result.items[0].succeeded
    assert result.items[1].failed
    assert result.items[1].error_code == "budget"
    assert all(item.budget_stopped for item in result.items[2:])
    assert result.stopped_for_budget


def test_manifest_records_each_item_and_resumes_successes(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"

    async def first(item: int) -> dict[str, Any]:
        if item == 2:
            raise RuntimeError("boom")
        return {"value": item, "cost_usd": 0.01}

    first_result = _run(BatchRunner(first, manifest=manifest).run(range(4)))

    assert [item.status for item in first_result.items] == [
        BatchItemStatus.SUCCEEDED,
        BatchItemStatus.SUCCEEDED,
        BatchItemStatus.FAILED,
        BatchItemStatus.SUCCEEDED,
    ]
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["items"]["2"]["status"] == "failed"
    assert payload["items"]["2"]["error"] == "boom"

    calls: list[int] = []

    async def second(item: int) -> dict[str, Any]:
        calls.append(item)
        return {"value": item, "cost_usd": 0.01}

    resumed = _run(BatchRunner(second, manifest=manifest, resume=True).run(range(4)))

    assert calls == [2]
    assert [item.resumed for item in resumed.items] == [True, True, False, True]
    assert all(item.succeeded for item in resumed.items)
    assert math.isclose(resumed.total_cost_usd, 0.04)


def test_a_new_run_resets_a_previous_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "items": {
                    "0": {
                        "key": "0",
                        "index": 0,
                        "status": "succeeded",
                        "value": "stale",
                        "cost_usd": 1.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    async def handler(item: int) -> int:
        return item

    result = _run(BatchRunner(handler, manifest=manifest).run([5]))

    assert len(result.items) == 1
    assert result.items[0].value == 5
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert list(payload["items"]) == ["0"]
    assert payload["items"]["0"]["value"] == 5


def test_duplicate_keys_and_invalid_options_are_rejected() -> None:
    with pytest.raises(ValueError):
        BatchRunner(concurrency=0)
    with pytest.raises(ValueError):
        BatchRunner(max_cost_usd=0.1, budget_usd=0.2)

    runner = BatchRunner(lambda item: item)
    with pytest.raises(ValueError):
        _run(runner.run([{"id": "same"}, {"id": "same"}]))


def test_manifest_can_be_loaded_directly(tmp_path: Path) -> None:
    path = tmp_path / "batch-manifest.json"
    manifest = BatchManifest(path)
    manifest.load()
    assert manifest.path == path
    assert len(manifest) == 0
