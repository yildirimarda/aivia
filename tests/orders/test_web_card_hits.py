"""Anonymous web card view counting with a daily salted hash and a retention
window, shown on order show.

Web card views are counted anonymously: a visitor identifier is combined with
a per-day salt and hashed, so the same visitor yields a different hash on a
different day and cannot be linked across days. Hit counts are shown on
``aivia order show`` within the configured retention window
(retention.web_card_hits_days).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from sqlalchemy import Engine

from aivia.cli import main
from aivia.core.db import (
    WebCardHitRepository,
    create_database_engine,
    upgrade_database,
)
from aivia.core.db.tables import web_card_hits
from aivia.orders import today, visitor_hash


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.lstrip(), encoding="utf-8")
    return path


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
def hits(engine: Engine) -> WebCardHitRepository:
    return WebCardHitRepository(engine)


def _make_order(engine: Engine, **values: Any) -> str:
    from aivia.core.db.repos import OrderRepository

    base = {
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
    stored = OrderRepository(engine).create(**cast("Mapping[str, Any]", base))
    return cast("str", stored["id"])


def _insert_hit(
    engine: Engine,
    *,
    order_id: str,
    card_id: str = "card_1",
    visitor_hash_value: str = "visitor_hash_1",
    day: str = "2024-01-15",
    hit_at: datetime | None = None,
    user_agent: str = "Mozilla/5.0",
    referrer: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> str:
    if hit_at is None:
        hit_at = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
    with engine.begin() as connection:
        result = connection.execute(
            web_card_hits.insert().values(
                id="hit_" + visitor_hash_value,
                order_id=order_id,
                card_id=card_id,
                day=day,
                visitor_hash=visitor_hash_value,
                hit_at=hit_at,
                user_agent=user_agent,
                referrer=referrer,
                metadata=dict(metadata or {}),
            )
        )
    pk = result.inserted_primary_key[0]  # type: ignore[index]
    return str(pk)


def test_visitor_hash_is_deterministic_for_the_same_day() -> None:
    day = today()
    first = visitor_hash("salt", day, "visitor-1")
    second = visitor_hash("salt", day, "visitor-1")
    assert first == second
    assert len(first) == 64


def test_visitor_hash_changes_across_days() -> None:
    salt = "salt"
    visitor = "visitor-1"
    day_one = "2024-01-15"
    day_two = "2024-01-16"
    assert visitor_hash(salt, day_one, visitor) != visitor_hash(salt, day_two, visitor)


def test_visitor_hash_changes_with_a_different_salt() -> None:
    day = "2024-01-15"
    visitor = "visitor-1"
    assert visitor_hash("salt-a", day, visitor) != visitor_hash("salt-b", day, visitor)


def test_visitor_hash_changes_with_a_different_visitor() -> None:
    day = "2024-01-15"
    assert visitor_hash("salt", day, "visitor-a") != visitor_hash(
        "salt", day, "visitor-b"
    )


def test_record_hit_stores_all_fields(
    engine: Engine, hits: WebCardHitRepository
) -> None:
    order_id = _make_order(engine)
    hit_at = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
    stored = hits.record_hit(
        order_id=order_id,
        card_id="card_1",
        visitor_hash="abc123",
        day="2024-01-15",
        hit_at=hit_at,
        user_agent="Mozilla/5.0",
        referrer="https://example.com/",
        metadata={"source": "card"},
    )
    assert stored["order_id"] == order_id
    assert stored["card_id"] == "card_1"
    assert stored["day"] == "2024-01-15"
    assert stored["visitor_hash"] == "abc123"
    assert stored["hit_at"].replace(tzinfo=None) == hit_at.replace(tzinfo=None)
    assert stored["user_agent"] == "Mozilla/5.0"
    assert stored["referrer"] == "https://example.com/"
    assert stored["metadata"] == {"source": "card"}


def test_record_hit_allows_null_order_id(
    engine: Engine, hits: WebCardHitRepository
) -> None:
    hit_at = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
    stored = hits.record_hit(
        order_id=None,
        card_id="card_1",
        visitor_hash="abc123",
        day="2024-01-15",
        hit_at=hit_at,
    )
    assert stored["order_id"] is None


def test_count_hits_for_order(engine: Engine, hits: WebCardHitRepository) -> None:
    order_id = _make_order(engine)
    now = datetime.now(UTC)
    _insert_hit(engine, order_id=order_id, visitor_hash_value="hash-1", hit_at=now)
    _insert_hit(engine, order_id=order_id, visitor_hash_value="hash-2", hit_at=now)
    _insert_hit(engine, order_id=order_id, visitor_hash_value="hash-3", hit_at=now)
    other_order = _make_order(engine)
    _insert_hit(engine, order_id=other_order, visitor_hash_value="hash-4", hit_at=now)

    assert hits.count_hits_for_order(order_id) == 3
    assert hits.count_hits_for_order(other_order) == 1


def test_count_hits_for_order_returns_zero_when_none(
    engine: Engine, hits: WebCardHitRepository
) -> None:
    order_id = _make_order(engine)
    assert hits.count_hits_for_order(order_id) == 0


def test_count_hits_respects_retention_window(
    engine: Engine, hits: WebCardHitRepository
) -> None:
    order_id = _make_order(engine)
    now = datetime.now(UTC)
    _insert_hit(engine, order_id=order_id, visitor_hash_value="recent", hit_at=now)
    _insert_hit(
        engine,
        order_id=order_id,
        visitor_hash_value="old",
        hit_at=now - timedelta(days=40),
    )

    assert hits.count_hits_for_order(order_id, days=30) == 1
    assert hits.count_hits_for_order(order_id, days=60) == 2


def test_count_hits_respects_since(engine: Engine, hits: WebCardHitRepository) -> None:
    order_id = _make_order(engine)
    now = datetime.now(UTC)
    cutoff = now - timedelta(days=10)
    _insert_hit(engine, order_id=order_id, visitor_hash_value="recent", hit_at=now)
    _insert_hit(
        engine,
        order_id=order_id,
        visitor_hash_value="old",
        hit_at=cutoff - timedelta(days=1),
    )

    assert hits.count_hits_for_order(order_id, since=cutoff) == 1


def test_order_show_prints_web_card_hits(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path, "schema_version: 1\n")
    database = tmp_path / "data" / "aivia.db"
    upgrade_database(database)
    engine = create_database_engine(database)

    order_id = _make_order(engine)
    now = datetime.now(UTC)
    _insert_hit(engine, order_id=order_id, visitor_hash_value="hash-1", hit_at=now)
    _insert_hit(engine, order_id=order_id, visitor_hash_value="hash-2", hit_at=now)
    engine.dispose()

    code = main(
        [
            "order",
            "show",
            order_id,
            "--config",
            str(config),
            "--database",
            str(database),
        ]
    )
    captured = capsys.readouterr()
    out = captured.out
    err = captured.err

    assert code == 0
    assert err == ""
    assert f"Order: {order_id}" in out
    assert "Web card hits: 2" in out


def test_order_show_prints_zero_hits_when_none(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path, "schema_version: 1\n")
    database = tmp_path / "data" / "aivia.db"
    upgrade_database(database)
    engine = create_database_engine(database)

    order_id = _make_order(engine)
    engine.dispose()

    code = main(
        [
            "order",
            "show",
            order_id,
            "--config",
            str(config),
            "--database",
            str(database),
        ]
    )
    captured = capsys.readouterr()
    out = captured.out
    err = captured.err

    assert code == 0
    assert err == ""
    assert f"Order: {order_id}" in out
    assert "Web card hits: 0" in out
