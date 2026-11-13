"""Asset cache keys, hits and zero cost cost events."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine, inspect

from aivia.core.asset_cache import AssetCache, AssetCacheKey, asset_cache_key
from aivia.core.db import (
    ASSET_CACHE_ASSET_INDEX_NAME,
    AssetCacheRepository,
    AssetRepository,
    create_database_engine,
    upgrade_database,
)
from aivia.core.models import Asset, compute_prompt_hash

REFERENCE_A = "a" * 64
REFERENCE_B = "b" * 64


@pytest.fixture
def engine() -> Iterator[Engine]:
    created = create_database_engine(url="sqlite+pysqlite:///:memory:")
    upgrade_database(engine=created)
    try:
        yield created
    finally:
        created.dispose()


@pytest.fixture
def cache(engine: Engine) -> AssetCache:
    return AssetCache(engine)


def _asset(**overrides: Any) -> Asset:
    values: dict[str, Any] = {
        "provider": "acme",
        "model": "image-1",
        "prompt": "A bridge at dusk",
        "path": "/runs/run_1/bridge.png",
        "content_hash": "c" * 64,
    }
    values.update(overrides)
    return Asset(**values)


def test_the_key_is_stable_across_equivalent_inputs() -> None:
    first = asset_cache_key(
        " acme ",
        " image-1 ",
        "  A bridge at dusk  ",
        parameters={"seed": 7, "size": "1080x1920"},
        reference_hashes=[REFERENCE_B, REFERENCE_A, REFERENCE_B],
    )
    second = AssetCacheKey(
        provider="acme",
        model="image-1",
        prompt="A bridge at dusk",
        parameters={"size": "1080x1920", "seed": 7},
        reference_hashes=(REFERENCE_A, REFERENCE_B),
    )

    assert first == second
    assert first.digest == second.digest
    assert len(first.digest) == 64
    assert first.digest == first.digest.lower()
    assert first.reference_hashes == (REFERENCE_A, REFERENCE_B)


def test_every_request_dimension_changes_the_key() -> None:
    base = asset_cache_key("acme", "image-1", "A bridge", parameters={"seed": 7})
    same = asset_cache_key("acme", "image-1", "A bridge", parameters={"seed": 7})

    assert base.digest == same.digest
    for changed in (
        asset_cache_key("other", "image-1", "A bridge", parameters={"seed": 7}),
        asset_cache_key("acme", "image-2", "A bridge", parameters={"seed": 7}),
        asset_cache_key("acme", "image-1", "Another bridge", parameters={"seed": 7}),
        asset_cache_key("acme", "image-1", "A bridge", parameters={"seed": 8}),
        asset_cache_key(
            "acme",
            "image-1",
            "A bridge",
            parameters={"seed": 7},
            reference_hashes=[REFERENCE_A],
        ),
    ):
        assert changed.digest != base.digest


def test_a_miss_returns_none_and_records_nothing(cache: AssetCache) -> None:
    key = cache.key("acme", "image-1", "A bridge")

    assert cache.get(key) is None
    assert cache.lookup(key) is None
    assert cache.cost.count() == 0


def test_storing_then_looking_up_records_a_zero_cost_hit(
    cache: AssetCache,
) -> None:
    key = cache.key(
        "acme",
        "image-1",
        "A bridge at dusk",
        parameters={"seed": 7},
        reference_hashes=[REFERENCE_A],
    )
    stored = cache.store(_asset(), key=key)

    assert stored.prompt_hash == compute_prompt_hash("A bridge at dusk")
    assert cache.get(key) == stored

    hit = cache.lookup(key, run_id="run_2")

    assert hit == stored
    events = cache.cost.events()
    assert len(events) == 1
    event = events[0]
    assert event.provider == "acme"
    assert event.model == "image-1"
    assert event.operation == "asset_gen"
    assert event.amount_usd == 0.0
    assert event.units == 0.0
    assert event.estimated is False
    assert event.run_id == "run_2"
    assert event.metadata == {
        "cache": "hit",
        "cache_key": key.digest,
        "asset_id": stored.id,
    }
    assert cache.cost.total() == 0.0
    assert cache.cost.count() == 1

    entry = AssetCacheRepository(cache.repository.engine).get(key.digest)
    assert entry is not None
    assert entry.asset_id == stored.id
    assert entry.provider == "acme"
    assert entry.model == "image-1"
    assert entry.prompt_hash == stored.prompt_hash
    assert entry.parameters == {"seed": 7}
    assert entry.reference_hashes == [REFERENCE_A]
    assert entry.hit_count == 1


def test_a_hit_can_skip_cost_recording(cache: AssetCache) -> None:
    key = cache.key("acme", "image-1", "A bridge")
    cache.store(_asset(), key=key)

    assert cache.lookup(key, record_hit=False) is not None
    assert cache.cost.count() == 0
    entry = cache.entry(key)
    assert entry is not None
    assert entry.hit_count == 0


def test_no_cache_turns_every_lookup_into_a_miss(engine: Engine) -> None:
    cache = AssetCache(engine, no_cache=True)
    key = cache.key("acme", "image-1", "A bridge")
    cache.store(_asset(), key=key)

    assert cache.get(key) is None
    assert cache.lookup(key) is None
    assert cache.cost.count() == 0


def test_storing_an_equivalent_asset_twice_returns_the_first_record(
    cache: AssetCache,
) -> None:
    key = cache.key("acme", "image-1", "A bridge", parameters={"seed": 7})
    first = cache.store(_asset(), key=key)
    second = cache.store(_asset(path="/runs/run_2/bridge.png"), key=key)

    assert second.id == first.id
    assert second.path == first.path
    assert len(cache.assets.list()) == 1
    assert len(cache.repository.list()) == 1


def test_concurrent_identical_stores_create_one_entry(tmp_path: Path) -> None:
    engine = create_database_engine(tmp_path / "aivia.db")
    upgrade_database(engine=engine)
    try:
        cache = AssetCache(engine)
        key = cache.key("acme", "image-1", "A bridge", parameters={"seed": 7})
        workers = 8
        barrier = threading.Barrier(workers)
        stored: list[Asset] = []
        errors: list[BaseException] = []
        lock = threading.Lock()

        def store_one(index: int) -> None:
            barrier.wait()
            try:
                asset = cache.store(
                    _asset(path=f"/runs/run_{index}/bridge.png"),
                    key=key,
                )
            except BaseException as exc:
                with lock:
                    errors.append(exc)
            else:
                with lock:
                    stored.append(asset)

        threads = [
            threading.Thread(target=store_one, args=(index,))
            for index in range(workers)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)

        assert not any(thread.is_alive() for thread in threads)
        assert errors == []
        assert len(stored) == workers
        assert len({asset.id for asset in stored}) == 1
        assert len(cache.assets.list()) == 1
        assert len(cache.repository.list()) == 1
        entry = cache.entry(key)
        assert entry is not None
        assert entry.asset_id == stored[0].id
        assert cache.get(key) == stored[0]
    finally:
        engine.dispose()


def test_concurrent_hits_do_not_lose_increments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = create_database_engine(tmp_path / "aivia.db")
    upgrade_database(engine=engine)
    try:
        cache = AssetCache(engine)
        key = cache.key("acme", "image-1", "A bridge at dusk")
        stored = cache.store(_asset(), key=key)

        workers = 8
        barrier = threading.Barrier(workers)
        original_get = cache.repository.get

        def synchronized_get(*args: Any, **kwargs: Any) -> Any:
            entry = original_get(*args, **kwargs)
            barrier.wait(timeout=5)
            return entry

        monkeypatch.setattr(cache.repository, "get", synchronized_get)
        errors: list[BaseException] = []
        lock = threading.Lock()

        def hit() -> None:
            try:
                cache.record_hit(key, stored)
            except BaseException as exc:
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=hit) for _ in range(workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)

        assert not any(thread.is_alive() for thread in threads)
        assert errors == []
        entry = AssetCacheRepository(engine).get(key.digest)
        assert entry is not None
        assert entry.hit_count == workers
        assert cache.cost.count() == workers
    finally:
        engine.dispose()


def test_a_different_reference_hash_is_a_different_cache_entry(
    cache: AssetCache,
) -> None:
    without = cache.key("acme", "image-1", "A bridge")
    with_a = cache.key(
        "acme",
        "image-1",
        "A bridge",
        reference_hashes=[REFERENCE_A],
    )
    cache.store(_asset(), key=without)

    assert cache.get(with_a) is None
    cache.store(_asset(path="/runs/run_2/bridge.png"), key=with_a)
    assert len(cache.assets.list()) == 2
    assert len(cache.repository.list()) == 2


def test_the_cache_survives_a_new_cache_instance(engine: Engine) -> None:
    first = AssetCache(engine)
    key = first.key("acme", "image-1", "A bridge", parameters={"seed": 7})
    stored = first.store(_asset(), key=key)

    second = AssetCache(engine)

    assert second.get(key) == stored
    hit = second.lookup(key)
    assert hit == stored
    assert second.cost.count() == 1


def test_the_cache_table_carries_a_lookup_index(engine: Engine) -> None:
    cache = AssetCache(engine)
    key = cache.key("acme", "image-1", "A bridge")
    cache.store(_asset(), key=key)

    columns = {column["name"] for column in inspect(engine).get_columns("asset_cache")}
    indexes = {index["name"] for index in inspect(engine).get_indexes("asset_cache")}
    assert {
        "cache_key",
        "asset_id",
        "provider",
        "model",
        "prompt_hash",
        "parameters",
        "reference_hashes",
        "hit_count",
    } <= columns
    assert ASSET_CACHE_ASSET_INDEX_NAME in indexes
    entry = cache.entry(key)
    assert entry is not None
    assert AssetRepository(engine).get(entry.asset_id) == cache.get(key)
