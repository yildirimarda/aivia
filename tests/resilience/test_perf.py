"""Concurrency and performance tests for parallel renders, queue load, large
inputs and memory ceilings, all asserting freedom from deadlocks.

These tests are slow: they run real threads against the job queue, render many
frames with the Pillow renderer and trace memory across repeated renders. None
of them touch the network, spend money or call a provider.
"""

from __future__ import annotations

import asyncio
import threading
import tracemalloc
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from sqlalchemy import Engine

from aivia.compose import Composition, PillowStillRenderer
from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.jobs import JobQueue
from aivia.core.locks import ParallelRunLimitError, RunLockManager
from aivia.core.ratelimit import ConcurrencyLimiter

pytestmark = pytest.mark.slow

RENDER_THREADS = 8
RENDERS_PER_THREAD = 6
JOIN_TIMEOUT_SECONDS = 30.0
QUEUE_WORKERS = 8
QUEUE_JOBS = 64
JOIN_TIMEOUT_QUEUE_SECONDS = 60.0
LARGE_CANVAS = "1280x720"
LARGE_LAYERS = 24
LARGE_REPEATS = 12
MEMORY_CEILING_BYTES = 256 * 1024 * 1024


def _solid(
    color: tuple[int, int, int, int], size: tuple[int, int] = (4, 4)
) -> Image.Image:
    return Image.new("RGBA", size, color)


def _composition(
    *,
    layers: list[dict[str, Any]] | None = None,
    canvas: str = LARGE_CANVAS,
    duration_seconds: float = 2.0,
    fps: int = 30,
) -> Composition:
    values: dict[str, Any] = {
        "canvas": canvas,
        "duration_seconds": duration_seconds,
        "fps": fps,
    }
    if layers is not None:
        values["layers"] = layers
    return Composition.model_validate(values)


def _image_layers(count: int) -> tuple[list[dict[str, Any]], dict[str, Image.Image]]:
    layers: list[dict[str, Any]] = []
    assets: dict[str, Image.Image] = {}
    for index in range(count):
        source = f"layer-{index}.png"
        color = (index * 17 % 256, index * 31 % 256, index * 73 % 256, 255)
        layers.append(
            {"id": f"layer-{index}", "type": "image", "source": source, "fit": "cover"}
        )
        assets[source] = _solid(color, (8, 8))
    return layers, assets


# ---------------------------------------------------------------------------
# Parallel renders
# ---------------------------------------------------------------------------


def test_parallel_renders_complete_without_deadlock() -> None:
    """Many threads render concurrently; every thread joins, output is deterministic."""

    layers, assets = _image_layers(6)
    composition = _composition(layers=layers)
    expected = PillowStillRenderer(assets).render_still(composition)

    results: list[Image.Image | None] = [None] * RENDER_THREADS
    errors: list[BaseException] = []
    barrier = threading.Barrier(RENDER_THREADS)

    def render(index: int) -> None:
        barrier.wait(timeout=10)
        try:
            results[index] = PillowStillRenderer(assets).render_still(composition)
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [
        threading.Thread(target=render, args=(index,))
        for index in range(RENDER_THREADS)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(JOIN_TIMEOUT_SECONDS)

    assert not any(thread.is_alive() for thread in threads), (
        "render thread hung (deadlock)"
    )
    assert errors == []
    assert all(result is not None for result in results)
    for result in results:
        assert result is not None
        assert result.size == expected.size
        assert result.tobytes() == expected.tobytes()


def test_concurrent_renders_match_their_single_threaded_render() -> None:
    """Each concurrent render equals the same composition rendered alone."""

    compositions: list[Composition] = []
    assets_list: list[dict[str, Image.Image]] = []
    for _ in range(RENDER_THREADS):
        layers, assets = _image_layers(4)
        compositions.append(_composition(layers=layers))
        assets_list.append(assets)

    expected = [
        PillowStillRenderer(assets).render_still(composition)
        for composition, assets in zip(compositions, assets_list, strict=True)
    ]

    results: list[Image.Image | None] = [None] * RENDER_THREADS
    errors: list[BaseException] = []
    barrier = threading.Barrier(RENDER_THREADS)

    def render(index: int) -> None:
        barrier.wait(timeout=10)
        try:
            results[index] = PillowStillRenderer(assets_list[index]).render_still(
                compositions[index]
            )
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [
        threading.Thread(target=render, args=(index,))
        for index in range(RENDER_THREADS)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(JOIN_TIMEOUT_SECONDS)

    assert not any(thread.is_alive() for thread in threads), (
        "render thread hung (deadlock)"
    )
    assert errors == []
    for index in range(RENDER_THREADS):
        rendered = results[index]
        assert rendered is not None
        assert rendered.tobytes() == expected[index].tobytes()


# ---------------------------------------------------------------------------
# Queue load
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> Iterator[Engine]:
    created = create_database_engine(url="sqlite+pysqlite:///:memory:")
    upgrade_database(engine=created)
    try:
        yield created
    finally:
        created.dispose()


def test_queue_load_concurrent_workers_never_double_claim(tmp_path: Path) -> None:
    """Many workers claim jobs from one queue; no job is claimed twice and none hang."""

    database = tmp_path / "aivia.db"
    setup = create_database_engine(database)
    upgrade_database(engine=setup)
    queue = JobQueue(setup)
    job_ids = {queue.enqueue(kind="render").id for _ in range(QUEUE_JOBS)}
    setup.dispose()

    barrier = threading.Barrier(QUEUE_WORKERS)
    claimed: list[str] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker(index: int) -> None:
        engine = create_database_engine(database)
        try:
            worker_queue = JobQueue(engine)
            barrier.wait(timeout=10)
            while True:
                job = worker_queue.claim(f"worker-{index}")
                if job is None:
                    break
                with lock:
                    claimed.append(job.id)
                worker_queue.complete(job.id, f"worker-{index}")
        except BaseException as exc:  # pragma: no cover - reported below
            with lock:
                errors.append(exc)
        finally:
            engine.dispose()

    threads = [
        threading.Thread(target=worker, args=(index,)) for index in range(QUEUE_WORKERS)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(JOIN_TIMEOUT_QUEUE_SECONDS)

    assert not any(thread.is_alive() for thread in threads), (
        "worker thread hung (deadlock)"
    )
    assert errors == []
    assert len(claimed) == len(set(claimed)) == QUEUE_JOBS
    assert set(claimed) == job_ids


def test_concurrent_enqueues_all_succeed_and_match_depth(tmp_path: Path) -> None:
    """Many threads enqueue concurrently; every enqueue lands and depth matches."""

    database = tmp_path / "aivia.db"
    setup = create_database_engine(database)
    upgrade_database(engine=setup)
    setup.dispose()

    barrier = threading.Barrier(QUEUE_WORKERS)
    enqueued: list[str] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker(index: int) -> None:
        engine = create_database_engine(database)
        try:
            worker_queue = JobQueue(engine)
            barrier.wait(timeout=10)
            for _ in range(QUEUE_JOBS // QUEUE_WORKERS):
                job = worker_queue.enqueue(kind="render")
                with lock:
                    enqueued.append(job.id)
        except BaseException as exc:  # pragma: no cover - reported below
            with lock:
                errors.append(exc)
        finally:
            engine.dispose()

    threads = [
        threading.Thread(target=worker, args=(index,)) for index in range(QUEUE_WORKERS)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(JOIN_TIMEOUT_QUEUE_SECONDS)

    assert not any(thread.is_alive() for thread in threads), (
        "enqueue thread hung (deadlock)"
    )
    assert errors == []
    assert len(enqueued) == len(set(enqueued)) == QUEUE_JOBS

    final = create_database_engine(database)
    try:
        assert JobQueue(final).depth() == QUEUE_JOBS
    finally:
        final.dispose()


# ---------------------------------------------------------------------------
# Large inputs
# ---------------------------------------------------------------------------


def test_large_input_render_completes_with_expected_size() -> None:
    """A large composition with many layers renders to the full canvas size."""

    layers, assets = _image_layers(LARGE_LAYERS)
    composition = _composition(layers=layers, canvas=LARGE_CANVAS)

    image = PillowStillRenderer(assets).render_still(composition)

    assert image.mode == "RGBA"
    assert image.size == (1280, 720)


def test_large_input_render_many_variants_without_deadlock() -> None:
    """Many large compositions render concurrently and every thread joins."""

    variants = [
        _composition(layers=_image_layers(LARGE_LAYERS)[0], canvas=LARGE_CANVAS)
        for _ in range(RENDER_THREADS)
    ]
    assets_list = [_image_layers(LARGE_LAYERS)[1] for _ in range(RENDER_THREADS)]

    results: list[Image.Image | None] = [None] * RENDER_THREADS
    errors: list[BaseException] = []
    barrier = threading.Barrier(RENDER_THREADS)

    def render(index: int) -> None:
        barrier.wait(timeout=10)
        try:
            results[index] = PillowStillRenderer(assets_list[index]).render_still(
                variants[index]
            )
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [
        threading.Thread(target=render, args=(index,))
        for index in range(RENDER_THREADS)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(JOIN_TIMEOUT_SECONDS)

    assert not any(thread.is_alive() for thread in threads), (
        "render thread hung (deadlock)"
    )
    assert errors == []
    for result in results:
        assert result is not None
        assert result.size == (1280, 720)


# ---------------------------------------------------------------------------
# Memory ceilings
# ---------------------------------------------------------------------------


def test_memory_stays_bounded_across_repeated_renders() -> None:
    """Repeated renders of a large composition stay under a memory ceiling."""

    layers, assets = _image_layers(LARGE_LAYERS)
    composition = _composition(layers=layers, canvas=LARGE_CANVAS)

    tracemalloc.start()
    try:
        for _ in range(LARGE_REPEATS):
            image = PillowStillRenderer(assets).render_still(composition)
            assert image.size == (1280, 720)
            del image
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < MEMORY_CEILING_BYTES, (
        f"peak memory {peak} exceeds ceiling {MEMORY_CEILING_BYTES}"
    )


# ---------------------------------------------------------------------------
# Concurrency caps and run limits
# ---------------------------------------------------------------------------


async def _peak_concurrency(limit: int, workers: int) -> int:
    limiter = ConcurrencyLimiter(max_concurrency=limit)
    active = 0
    peak = 0

    async def worker() -> None:
        nonlocal active, peak
        async with limiter:
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.005)
            active -= 1

    await asyncio.gather(*[worker() for _ in range(workers)])
    return peak


def test_concurrency_limiter_enforces_cap_without_deadlock() -> None:
    """A concurrency limiter never admits more callers than its cap."""

    peak = asyncio.run(_peak_concurrency(limit=3, workers=40))
    assert peak <= 3


def test_parallel_run_limit_refuses_extra_runs_and_frees_slots(tmp_path: Path) -> None:
    """The global parallel run limit blocks beyond its slots and releases on done."""

    lock_dir = tmp_path / "locks"
    manager = RunLockManager(max_parallel=2, lock_dir=lock_dir, per_profile=False)

    first = manager.acquire("run-1", wait=False)
    second = manager.acquire("run-2", wait=False)
    assert first is not None
    assert second is not None

    with pytest.raises(ParallelRunLimitError):
        manager.acquire("run-3", wait=False)

    first.release()
    third = manager.acquire("run-3", wait=False)
    assert third is not None
    third.release()
    second.release()
