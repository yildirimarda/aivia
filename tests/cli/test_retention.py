"""``aivia retention``: remove stale intermediates, cache, orders, deliveries,
logs and hits.

Final outputs are pinned and never removed unless final output deletion is
enabled, and ``--dry-run`` lists what would be deleted without removing it.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from aivia.cli import main
from aivia.core.db import (
    AssetCacheRepository,
    AssetRepository,
    DeliveryRepository,
    OrderRepository,
    RunRepository,
    WebCardHitRepository,
    create_database_engine,
    upgrade_database,
)
from aivia.core.models import utcnow


def _old() -> datetime:
    return datetime(2020, 1, 1, tzinfo=UTC)


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.lstrip(), encoding="utf-8")
    return path


@pytest.fixture
def config(tmp_path: Path) -> Path:
    return write_config(tmp_path, "schema_version: 1\n")


@pytest.fixture
def database(tmp_path: Path) -> Path:
    path = tmp_path / "data" / "aivia.db"
    upgrade_database(path)
    return path


def write_file(path: Path, content: bytes = b"data") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def touch_old(path: Path, when: datetime | None = None) -> None:
    """Set a file's mtime to *when* so it looks old to retention."""
    timestamp = (when or _old()).timestamp()
    os.utime(path, (timestamp, timestamp))


def run_command(capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, str, str]:
    code = main(["retention", *args])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def common_args(config: Path, database: Path) -> list[str]:
    return ["--config", str(config), "--database", str(database)]


def stored_order_ids(database: Path) -> list[str]:
    engine = create_database_engine(database)
    try:
        return [order["id"] for order in OrderRepository(engine).list()]
    finally:
        engine.dispose()


def stored_run_dirs(database: Path) -> dict[str, str]:
    engine = create_database_engine(database)
    try:
        return {run.id: run.run_dir for run in RunRepository(engine).list()}
    finally:
        engine.dispose()


def stored_asset(path: Path, database: Path) -> object | None:
    engine = create_database_engine(database)
    try:
        return AssetRepository(engine).get(path)
    finally:
        engine.dispose()


def test_retention_deletes_old_intermediates_and_keeps_new(
    config: Path, database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = database.parent
    engine = create_database_engine(database)
    order_repo = OrderRepository(engine)
    run_repo = RunRepository(engine)
    order_repo.create(id="order_old", state="delivered")
    order_repo.create(id="order_new", state="delivered")
    old_dir = base / "runs" / "run_old"
    new_dir = base / "runs" / "run_new"
    write_file(old_dir / "intermediate.png")
    write_file(new_dir / "intermediate.png")
    run_repo.create(
        id="run_old",
        order_id="order_old",
        run_dir=str(old_dir),
        state="succeeded",
        created_at=_old(),
    )
    run_repo.create(
        id="run_new",
        order_id="order_new",
        run_dir=str(new_dir),
        state="succeeded",
        created_at=utcnow(),
    )
    engine.dispose()

    code, _out, err = run_command(capsys, *common_args(config, database))

    assert code == 0
    assert err == ""
    assert not (old_dir / "intermediate.png").exists()
    assert not old_dir.exists()
    assert (new_dir / "intermediate.png").exists()
    assert new_dir.exists()


def test_retention_deletes_old_cache_and_keeps_new(
    config: Path, database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = database.parent
    engine = create_database_engine(database)
    asset_repo = AssetRepository(engine)
    cache_repo = AssetCacheRepository(engine)
    old_file = base / "cache" / "old.png"
    new_file = base / "cache" / "new.png"
    write_file(old_file)
    write_file(new_file)
    asset_repo.create(
        id="asset_old",
        path=str(old_file),
        pinned=False,
        kind="video",
        created_at=_old(),
    )
    asset_repo.create(
        id="asset_new",
        path=str(new_file),
        pinned=False,
        kind="video",
        created_at=utcnow(),
    )
    cache_repo.create(
        cache_key="key_old",
        asset_id="asset_old",
        created_at=_old(),
    )
    cache_repo.create(
        cache_key="key_new",
        asset_id="asset_new",
        created_at=utcnow(),
    )
    engine.dispose()

    code, _out, err = run_command(capsys, *common_args(config, database))

    assert code == 0
    assert err == ""
    assert not old_file.exists()
    assert (
        AssetRepository(create_database_engine(database)).get("asset_old") is not None
    )
    assert new_file.exists()
    assert cache_repo is not None


def test_retention_keeps_pinned_final_outputs_by_default(
    config: Path, database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = database.parent
    engine = create_database_engine(database)
    order_repo = OrderRepository(engine)
    run_repo = RunRepository(engine)
    asset_repo = AssetRepository(engine)
    order_repo.create(id="order_old", state="delivered", created_at=_old())
    run_repo.create(
        id="run_old",
        order_id="order_old",
        run_dir="",
        state="succeeded",
        created_at=_old(),
    )
    output_dir = base / "outputs"
    write_file(output_dir / "final.mp4")
    asset_repo.create(
        id="asset_old",
        run_id="run_old",
        path=str(output_dir / "final.mp4"),
        pinned=True,
        kind="video",
        created_at=_old(),
    )
    engine.dispose()

    code, _out, err = run_command(capsys, *common_args(config, database))

    assert code == 0
    assert err == ""
    # The order is expired and removed.
    assert OrderRepository(create_database_engine(database)).get("order_old") is None
    # The pinned final output survives.
    assert (output_dir / "final.mp4").exists()
    assert (
        AssetRepository(create_database_engine(database)).get("asset_old") is not None
    )


def test_retention_removes_pinned_finals_when_enabled(
    config: Path, database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = database.parent
    engine = create_database_engine(database)
    order_repo = OrderRepository(engine)
    run_repo = RunRepository(engine)
    asset_repo = AssetRepository(engine)
    order_repo.create(id="order_old", state="delivered", created_at=_old())
    run_repo.create(
        id="run_old",
        order_id="order_old",
        run_dir="",
        state="succeeded",
        created_at=_old(),
    )
    output_dir = base / "outputs"
    write_file(output_dir / "final.mp4")
    asset_repo.create(
        id="asset_old",
        run_id="run_old",
        path=str(output_dir / "final.mp4"),
        pinned=True,
        kind="video",
        created_at=_old(),
    )
    engine.dispose()

    code, _out, err = run_command(
        capsys,
        "--set",
        "retention.delete_final_outputs=true",
        *common_args(config, database),
    )

    assert code == 0
    assert err == ""
    assert not (output_dir / "final.mp4").exists()
    assert AssetRepository(create_database_engine(database)).get("asset_old") is None


def test_retention_deletes_old_deliveries_and_keeps_new(
    config: Path, database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = database.parent
    engine = create_database_engine(database)
    order_repo = OrderRepository(engine)
    delivery_repo = DeliveryRepository(engine)
    order_repo.create(id="order_old", state="delivered", created_at=_old())
    order_repo.create(id="order_new", state="delivered", created_at=utcnow())
    old_dir = base / "deliveries" / "old"
    new_dir = base / "deliveries" / "new"
    old_file = old_dir / "final.mp4"
    new_file = new_dir / "final.mp4"
    write_file(old_file)
    write_file(new_file)
    delivery_repo.create(
        id="delivery_old",
        order_id="order_old",
        state="delivered",
        path=str(old_file),
        files=[str(old_file)],
        created_at=_old(),
    )
    delivery_repo.create(
        id="delivery_new",
        order_id="order_new",
        state="delivered",
        path=str(new_file),
        files=[str(new_file)],
        created_at=utcnow(),
    )
    engine.dispose()

    code, _out, err = run_command(capsys, *common_args(config, database))

    assert code == 0
    assert err == ""
    assert not old_file.exists()
    assert (
        DeliveryRepository(create_database_engine(database)).get("delivery_old") is None
    )
    assert new_file.exists()


def test_retention_deletes_old_logs_and_keeps_new(
    config: Path, database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    logs_dir = database.parent / "logs"
    old_log = logs_dir / "old.log"
    new_log = logs_dir / "new.log"
    write_file(old_log, b"old log")
    write_file(new_log, b"new log")
    touch_old(old_log)

    code, _out, err = run_command(capsys, *common_args(config, database))

    assert code == 0
    assert err == ""
    assert not old_log.exists()
    assert new_log.exists()


def test_retention_deletes_old_hits_and_keeps_new(
    config: Path, database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    engine = create_database_engine(database)
    order_repo = OrderRepository(engine)
    hit_repo = WebCardHitRepository(engine)
    order_repo.create(id="order_old", state="delivered", created_at=_old())
    order_repo.create(id="order_new", state="delivered", created_at=utcnow())
    hit_repo.create(
        id="hit_old",
        order_id="order_old",
        card_id="card",
        visitor_hash="visitor",
        day="2020-01-01",
        created_at=_old(),
    )
    hit_repo.create(
        id="hit_new",
        order_id="order_new",
        card_id="card",
        visitor_hash="visitor",
        day="2020-01-02",
        created_at=utcnow(),
    )
    engine.dispose()

    code, _out, err = run_command(capsys, *common_args(config, database))

    assert code == 0
    assert err == ""
    assert WebCardHitRepository(create_database_engine(database)).get("hit_old") is None
    assert (
        WebCardHitRepository(create_database_engine(database)).get("hit_new")
        is not None
    )


def test_retention_dry_run_lists_without_deleting(
    config: Path, database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = database.parent
    engine = create_database_engine(database)
    order_repo = OrderRepository(engine)
    run_repo = RunRepository(engine)
    order_repo.create(id="order_old", state="delivered")
    run_repo.create(
        id="run_old",
        order_id="order_old",
        run_dir=str(base / "runs" / "run_old"),
        state="succeeded",
        created_at=_old(),
    )
    write_file(base / "runs" / "run_old" / "intermediate.png")
    engine.dispose()

    code, out, err = run_command(capsys, "--dry-run", *common_args(config, database))

    assert code == 0
    assert err == ""
    assert "dry run" in out
    assert "run_old" in out
    # Nothing was actually removed.
    assert (base / "runs" / "run_old" / "intermediate.png").exists()
    assert (
        OrderRepository(create_database_engine(database)).get("order_old") is not None
    )


def test_retention_dry_run_nothing_to_delete(
    config: Path, database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = run_command(capsys, "--dry-run", *common_args(config, database))

    assert code == 0
    assert err == ""
    assert "nothing to delete" in out


def test_retention_keeps_recent_records(
    config: Path, database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = database.parent
    engine = create_database_engine(database)
    order_repo = OrderRepository(engine)
    run_repo = RunRepository(engine)
    order_repo.create(id="order_new", state="delivered")
    run_repo.create(
        id="run_new",
        order_id="order_new",
        run_dir=str(base / "runs" / "run_new"),
        state="succeeded",
        created_at=utcnow(),
    )
    write_file(base / "runs" / "run_new" / "intermediate.png")
    engine.dispose()

    code, out, err = run_command(capsys, *common_args(config, database))

    assert code == 0
    assert err == ""
    assert "deleted 0 item(s)" in out
    assert (base / "runs" / "run_new" / "intermediate.png").exists()
    assert (
        OrderRepository(create_database_engine(database)).get("order_new") is not None
    )
