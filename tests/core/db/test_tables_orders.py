"""The order, delivery, post, post metric and web card hit tables."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine, inspect, select
from sqlalchemy.exc import IntegrityError

from aivia.core.db import (
    DELIVERY_CREATED_AT_INDEX_NAME,
    DELIVERY_ORDER_INDEX_NAME,
    DELIVERY_SCHEDULED_FOR_INDEX_NAME,
    DELIVERY_STATE_INDEX_NAME,
    ORDER_CREATED_AT_INDEX_NAME,
    ORDER_IDEMPOTENCY_INDEX_NAME,
    ORDER_OCCASION_INDEX_NAME,
    ORDER_RUN_INDEX_NAME,
    ORDER_SCHEDULED_FOR_INDEX_NAME,
    ORDER_STATE_INDEX_NAME,
    POST_CREATED_AT_INDEX_NAME,
    POST_METRIC_CAPTURED_AT_INDEX_NAME,
    POST_METRIC_POST_INDEX_NAME,
    POST_PLATFORM_INDEX_NAME,
    POST_RUN_INDEX_NAME,
    POST_STATE_INDEX_NAME,
    WEB_CARD_HIT_CREATED_AT_INDEX_NAME,
    WEB_CARD_HIT_DAY_INDEX_NAME,
    WEB_CARD_HIT_ORDER_INDEX_NAME,
    WEB_CARD_HIT_VISITOR_INDEX_NAME,
    create_database_engine,
    deliveries,
    metadata,
    orders,
    post_metrics,
    posts,
    upgrade_database,
    web_card_hits,
)

ORDER_TABLES = {"orders", "deliveries", "posts", "post_metrics", "web_card_hits"}


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    database = tmp_path / "data" / "aivia.db"
    upgrade_database(database)
    created = create_database_engine(database)
    try:
        yield created
    finally:
        created.dispose()


def _indexes(engine: Engine, table: str) -> dict[str, Any]:
    return {
        index["name"]: index
        for index in inspect(engine).get_indexes(table)
        if index["name"] is not None
    }


def _columns(engine: Engine, table: str) -> set[str]:
    return {column["name"] for column in inspect(engine).get_columns(table)}


def test_the_migration_creates_the_order_tables(engine: Engine) -> None:
    assert set(inspect(engine).get_table_names()) >= ORDER_TABLES


def test_the_order_tables_are_registered_on_the_shared_metadata() -> None:
    assert set(metadata.tables) >= ORDER_TABLES


def test_orders_carry_the_order_columns(engine: Engine) -> None:
    assert {
        "id",
        "state",
        "occasion",
        "name",
        "message",
        "language",
        "profile",
        "level",
        "timezone",
        "variables",
        "outputs",
        "pricing",
        "currency",
        "amount_usd",
        "tax_usd",
        "cost_usd",
        "budget_usd",
        "run_id",
        "idempotency_key",
        "scheduled_for",
        "delivered_at",
        "expires_at",
        "error_code",
        "error_message",
        "metadata",
        "schema_version",
        "created_at",
        "updated_at",
    } <= _columns(engine, "orders")


def test_deliveries_carry_the_delivery_columns(engine: Engine) -> None:
    assert {
        "id",
        "order_id",
        "run_id",
        "provider",
        "state",
        "target",
        "path",
        "url",
        "email",
        "files",
        "scheduled_for",
        "delivered_at",
        "expires_at",
        "attempts",
        "max_attempts",
        "idempotency_key",
        "confirmation",
        "error_code",
        "error_message",
        "cost_usd",
        "attestation",
        "metadata",
        "schema_version",
        "created_at",
        "updated_at",
    } <= _columns(engine, "deliveries")


def test_posts_carry_the_post_columns(engine: Engine) -> None:
    assert {
        "id",
        "order_id",
        "platform",
        "account",
        "profile",
        "language",
        "format",
        "title",
        "caption",
        "hashtags",
        "mentions",
        "media",
        "media_type",
        "visibility",
        "state",
        "remote_id",
        "url",
        "scheduled_for",
        "published_at",
        "dry_run",
        "idempotency_key",
        "run_id",
        "composition_id",
        "disclosure_text",
        "disclosure_required",
        "disclosure_applied",
        "attestation",
        "error_code",
        "error_message",
        "cost_usd",
        "metadata",
        "schema_version",
        "created_at",
        "updated_at",
    } <= _columns(engine, "posts")


def test_post_metrics_carry_the_metric_columns(engine: Engine) -> None:
    assert {
        "id",
        "post_id",
        "platform",
        "impressions",
        "reach",
        "views",
        "likes",
        "comments",
        "shares",
        "saves",
        "clicks",
        "watch_time_seconds",
        "engagement_rate",
        "captured_at",
        "raw",
        "metadata",
        "schema_version",
        "created_at",
        "updated_at",
    } <= _columns(engine, "post_metrics")


def test_web_card_hits_carry_the_hit_columns(engine: Engine) -> None:
    assert {
        "id",
        "order_id",
        "card_id",
        "day",
        "visitor_hash",
        "hit_at",
        "user_agent",
        "referrer",
        "metadata",
        "schema_version",
        "created_at",
        "updated_at",
    } <= _columns(engine, "web_card_hits")


def test_orders_have_lookup_indexes(engine: Engine) -> None:
    indexes = _indexes(engine, "orders")

    expected = {
        ORDER_STATE_INDEX_NAME: ["state"],
        ORDER_OCCASION_INDEX_NAME: ["occasion"],
        ORDER_RUN_INDEX_NAME: ["run_id"],
        ORDER_IDEMPOTENCY_INDEX_NAME: ["idempotency_key"],
        ORDER_SCHEDULED_FOR_INDEX_NAME: ["scheduled_for"],
        ORDER_CREATED_AT_INDEX_NAME: ["created_at"],
    }
    for name, columns in expected.items():
        assert name in indexes
        assert indexes[name]["column_names"] == columns
        assert not indexes[name]["unique"]


def test_deliveries_have_lookup_indexes(engine: Engine) -> None:
    indexes = _indexes(engine, "deliveries")

    expected = {
        DELIVERY_ORDER_INDEX_NAME: ["order_id"],
        DELIVERY_STATE_INDEX_NAME: ["state"],
        DELIVERY_SCHEDULED_FOR_INDEX_NAME: ["scheduled_for"],
        DELIVERY_CREATED_AT_INDEX_NAME: ["created_at"],
    }
    for name, columns in expected.items():
        assert name in indexes
        assert indexes[name]["column_names"] == columns
        assert not indexes[name]["unique"]


def test_posts_have_lookup_indexes(engine: Engine) -> None:
    indexes = _indexes(engine, "posts")

    expected = {
        POST_STATE_INDEX_NAME: ["state"],
        POST_PLATFORM_INDEX_NAME: ["platform"],
        POST_RUN_INDEX_NAME: ["run_id"],
        POST_CREATED_AT_INDEX_NAME: ["created_at"],
    }
    for name, columns in expected.items():
        assert name in indexes
        assert indexes[name]["column_names"] == columns
        assert not indexes[name]["unique"]


def test_post_metrics_have_lookup_indexes(engine: Engine) -> None:
    indexes = _indexes(engine, "post_metrics")

    expected = {
        POST_METRIC_POST_INDEX_NAME: ["post_id"],
        POST_METRIC_CAPTURED_AT_INDEX_NAME: ["captured_at"],
    }
    for name, columns in expected.items():
        assert name in indexes
        assert indexes[name]["column_names"] == columns
        assert not indexes[name]["unique"]


def test_web_card_hits_have_lookup_indexes(engine: Engine) -> None:
    indexes = _indexes(engine, "web_card_hits")

    expected = {
        WEB_CARD_HIT_ORDER_INDEX_NAME: ["order_id"],
        WEB_CARD_HIT_DAY_INDEX_NAME: ["day"],
        WEB_CARD_HIT_VISITOR_INDEX_NAME: ["visitor_hash"],
        WEB_CARD_HIT_CREATED_AT_INDEX_NAME: ["created_at"],
    }
    for name, columns in expected.items():
        assert name in indexes
        assert indexes[name]["column_names"] == columns
        assert not indexes[name]["unique"]


def test_a_minimal_order_uses_the_record_defaults(engine: Engine) -> None:
    with engine.begin() as connection:
        connection.execute(orders.insert().values(id="order_1"))

    with engine.connect() as connection:
        row = connection.execute(select(orders).where(orders.c.id == "order_1")).one()

    assert row.state == "draft"
    assert row.occasion == ""
    assert row.name == ""
    assert row.message == ""
    assert row.language == ""
    assert row.level == ""
    assert row.timezone == "UTC"
    assert row.variables == {}
    assert row.outputs == []
    assert row.pricing == {}
    assert row.currency == "USD"
    assert row.amount_usd == 0
    assert row.tax_usd == 0
    assert row.cost_usd == 0
    assert row.budget_usd is None
    assert row.run_id == ""
    assert row.idempotency_key == ""
    assert row.metadata == {}


def test_a_minimal_delivery_uses_the_record_defaults(engine: Engine) -> None:
    with engine.begin() as connection:
        connection.execute(deliveries.insert().values(id="delivery_1"))

    with engine.connect() as connection:
        row = connection.execute(
            select(deliveries).where(deliveries.c.id == "delivery_1")
        ).one()

    assert row.order_id is None
    assert row.provider == "folder"
    assert row.state == "pending"
    assert row.files == []
    assert row.attempts == 0
    assert row.max_attempts == 3
    assert row.idempotency_key == ""
    assert row.attestation == {}
    assert row.metadata == {}


def test_a_minimal_post_uses_the_record_defaults(engine: Engine) -> None:
    with engine.begin() as connection:
        connection.execute(posts.insert().values(id="post_1"))

    with engine.connect() as connection:
        row = connection.execute(select(posts).where(posts.c.id == "post_1")).one()

    assert row.order_id is None
    assert row.platform == ""
    assert row.hashtags == []
    assert row.mentions == []
    assert row.media == []
    assert row.state == "draft"
    assert row.dry_run is False
    assert row.disclosure_required is False
    assert row.disclosure_applied is False
    assert row.attestation == {}
    assert row.metadata == {}


def test_a_minimal_post_metric_uses_the_record_defaults(engine: Engine) -> None:
    with engine.begin() as connection:
        connection.execute(post_metrics.insert().values(id="metric_1"))

    with engine.connect() as connection:
        row = connection.execute(
            select(post_metrics).where(post_metrics.c.id == "metric_1")
        ).one()

    assert row.post_id is None
    assert row.impressions == 0
    assert row.reach == 0
    assert row.views == 0
    assert row.likes == 0
    assert row.comments == 0
    assert row.shares == 0
    assert row.saves == 0
    assert row.clicks == 0
    assert row.watch_time_seconds == 0
    assert row.engagement_rate == 0
    assert row.raw == {}
    assert row.metadata == {}


def test_a_minimal_web_card_hit_uses_the_record_defaults(engine: Engine) -> None:
    with engine.begin() as connection:
        connection.execute(web_card_hits.insert().values(id="hit_1"))

    with engine.connect() as connection:
        row = connection.execute(
            select(web_card_hits).where(web_card_hits.c.id == "hit_1")
        ).one()

    assert row.order_id is None
    assert row.card_id == ""
    assert row.day == ""
    assert row.visitor_hash == ""
    assert row.metadata == {}


def test_orders_can_be_selected_by_state(engine: Engine) -> None:
    with engine.begin() as connection:
        connection.execute(
            orders.insert().values(id="order_draft", state="draft", occasion="birthday")
        )
        connection.execute(
            orders.insert().values(id="order_paid", state="paid", occasion="birthday")
        )
        connection.execute(
            orders.insert().values(
                id="order_done", state="delivered", occasion="wedding"
            )
        )

    with engine.connect() as connection:
        waiting = (
            connection.execute(
                select(orders.c.id)
                .where(orders.c.state == "paid")
                .order_by(orders.c.id)
            )
            .scalars()
            .all()
        )
        birthdays = (
            connection.execute(
                select(orders.c.id)
                .where(orders.c.occasion == "birthday")
                .order_by(orders.c.id)
            )
            .scalars()
            .all()
        )

    assert waiting == ["order_paid"]
    assert birthdays == ["order_draft", "order_paid"]


def test_an_order_rejects_a_delivery_that_names_a_missing_order(
    engine: Engine,
) -> None:
    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(
            deliveries.insert().values(id="delivery_1", order_id="missing")
        )


def test_deleting_an_order_deletes_its_deliveries_posts_and_hits(
    engine: Engine,
) -> None:
    with engine.begin() as connection:
        connection.execute(orders.insert().values(id="order_1"))
        connection.execute(
            deliveries.insert().values(id="delivery_1", order_id="order_1")
        )
        connection.execute(posts.insert().values(id="post_1", order_id="order_1"))
        connection.execute(
            post_metrics.insert().values(id="metric_1", post_id="post_1")
        )
        connection.execute(
            web_card_hits.insert().values(id="hit_1", order_id="order_1")
        )

    with engine.begin() as connection:
        connection.execute(orders.delete().where(orders.c.id == "order_1"))

    with engine.connect() as connection:
        assert connection.execute(select(deliveries)).all() == []
        assert connection.execute(select(posts)).all() == []
        assert connection.execute(select(post_metrics)).all() == []
        assert connection.execute(select(web_card_hits)).all() == []


def test_deleting_a_post_deletes_its_metrics(engine: Engine) -> None:
    with engine.begin() as connection:
        connection.execute(posts.insert().values(id="post_1"))
        connection.execute(
            post_metrics.insert().values(id="metric_1", post_id="post_1")
        )
        connection.execute(
            post_metrics.insert().values(id="metric_2", post_id="post_1")
        )

    with engine.begin() as connection:
        connection.execute(posts.delete().where(posts.c.id == "post_1"))

    with engine.connect() as connection:
        assert connection.execute(select(post_metrics)).all() == []


def test_the_downgrade_removes_the_order_tables(engine: Engine) -> None:
    from alembic import command

    from aivia.core.db import alembic_config

    with engine.begin() as connection:
        config = alembic_config(connection=connection)
        command.downgrade(config, "0004")

    remaining = set(inspect(engine).get_table_names())

    assert not (ORDER_TABLES & remaining)
