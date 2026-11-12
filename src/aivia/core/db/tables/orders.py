"""Tables for personalized orders, deliveries, posts and their analytics.

An order is the durable record of one personalized request: the occasion, the
recipient details, the chosen personalization level, the requested outputs and
its pricing. Deliveries and posts are the two ways an order leaves the system,
and both cascade when their order is deleted. Post metrics are periodic
snapshots of a published post, while web card hits count anonymous views of a
delivered card.

The indexes answer the questions the orders module asks most often: which orders
are waiting in a state, which deliveries are due, which post belongs to a run,
which metrics were captured in a time window and how many hits a card collected
per day.
"""

from __future__ import annotations

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Table,
    Text,
    func,
    text,
)

from aivia.core.db.schema import metadata

__all__ = [
    "DELIVERY_CREATED_AT_INDEX_NAME",
    "DELIVERY_ORDER_INDEX_NAME",
    "DELIVERY_SCHEDULED_FOR_INDEX_NAME",
    "DELIVERY_STATE_INDEX_NAME",
    "ORDER_CREATED_AT_INDEX_NAME",
    "ORDER_IDEMPOTENCY_INDEX_NAME",
    "ORDER_OCCASION_INDEX_NAME",
    "ORDER_RUN_INDEX_NAME",
    "ORDER_SCHEDULED_FOR_INDEX_NAME",
    "ORDER_STATE_INDEX_NAME",
    "POST_CREATED_AT_INDEX_NAME",
    "POST_METRIC_CAPTURED_AT_INDEX_NAME",
    "POST_METRIC_POST_INDEX_NAME",
    "POST_PLATFORM_INDEX_NAME",
    "POST_RUN_INDEX_NAME",
    "POST_STATE_INDEX_NAME",
    "WEB_CARD_HIT_CREATED_AT_INDEX_NAME",
    "WEB_CARD_HIT_DAY_INDEX_NAME",
    "WEB_CARD_HIT_ORDER_INDEX_NAME",
    "WEB_CARD_HIT_VISITOR_INDEX_NAME",
    "deliveries",
    "orders",
    "post_metrics",
    "posts",
    "web_card_hits",
]

ORDER_STATE_INDEX_NAME = "ix_orders_state"
"""Find orders awaiting approval, delivery or completion."""

ORDER_OCCASION_INDEX_NAME = "ix_orders_occasion"
"""Group orders by occasion for reporting."""

ORDER_RUN_INDEX_NAME = "ix_orders_run_id"
"""List the orders attached to one run."""

ORDER_IDEMPOTENCY_INDEX_NAME = "ix_orders_idempotency_key"
"""Return the existing order for a repeated create request."""

ORDER_SCHEDULED_FOR_INDEX_NAME = "ix_orders_scheduled_for"
"""Find orders whose delivery time is due."""

ORDER_CREATED_AT_INDEX_NAME = "ix_orders_created_at"
"""Scan orders by creation time for reports and retention."""

DELIVERY_ORDER_INDEX_NAME = "ix_deliveries_order_id"
"""List the deliveries belonging to one order."""

DELIVERY_STATE_INDEX_NAME = "ix_deliveries_state"
"""Find deliveries that are pending, scheduled or failed."""

DELIVERY_SCHEDULED_FOR_INDEX_NAME = "ix_deliveries_scheduled_for"
"""Find deliveries whose scheduled time has arrived."""

DELIVERY_CREATED_AT_INDEX_NAME = "ix_deliveries_created_at"
"""Scan deliveries by creation time for retention."""

POST_STATE_INDEX_NAME = "ix_posts_state"
"""Find posts that are drafts, scheduled or publishing."""

POST_PLATFORM_INDEX_NAME = "ix_posts_platform"
"""Group posts by platform for reporting."""

POST_RUN_INDEX_NAME = "ix_posts_run_id"
"""List the posts produced by one run."""

POST_CREATED_AT_INDEX_NAME = "ix_posts_created_at"
"""Scan posts by creation time for retention."""

POST_METRIC_POST_INDEX_NAME = "ix_post_metrics_post_id"
"""Collect the metric snapshots of one post."""

POST_METRIC_CAPTURED_AT_INDEX_NAME = "ix_post_metrics_captured_at"
"""Scan metric snapshots by capture time."""

WEB_CARD_HIT_ORDER_INDEX_NAME = "ix_web_card_hits_order_id"
"""Count the hits collected by one order."""

WEB_CARD_HIT_DAY_INDEX_NAME = "ix_web_card_hits_day"
"""Aggregate anonymous card views per day."""

WEB_CARD_HIT_VISITOR_INDEX_NAME = "ix_web_card_hits_visitor_hash"
"""Find the daily hash of one anonymous visitor."""

WEB_CARD_HIT_CREATED_AT_INDEX_NAME = "ix_web_card_hits_created_at"
"""Scan card hits by creation time for retention."""


orders = Table(
    "orders",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("state", String(32), nullable=False, server_default=text("'draft'")),
    Column("occasion", String(128), nullable=False, server_default=text("''")),
    Column("name", String(255), nullable=False, server_default=text("''")),
    Column("message", Text, nullable=False, server_default=text("''")),
    Column("language", String(32), nullable=False, server_default=text("''")),
    Column("profile", String(64), nullable=False, server_default=text("''")),
    Column("level", String(64), nullable=False, server_default=text("''")),
    Column("timezone", String(64), nullable=False, server_default=text("'UTC'")),
    Column("variables", JSON, nullable=False, server_default=text("'{}'")),
    Column("outputs", JSON, nullable=False, server_default=text("'[]'")),
    Column("pricing", JSON, nullable=False, server_default=text("'{}'")),
    Column("currency", String(8), nullable=False, server_default=text("'USD'")),
    Column("amount_usd", Float, nullable=False, server_default=text("0")),
    Column("tax_usd", Float, nullable=False, server_default=text("0")),
    Column("cost_usd", Float, nullable=False, server_default=text("0")),
    Column("budget_usd", Float, nullable=True),
    Column("run_id", String(64), nullable=False, server_default=text("''")),
    Column(
        "idempotency_key",
        String(255),
        nullable=False,
        server_default=text("''"),
    ),
    Column(
        "character",
        String(128),
        nullable=False,
        server_default=text("''"),
    ),
    Column("owner", String(255), nullable=False, server_default=text("''")),
    Column("scheduled_for", DateTime(timezone=True), nullable=True),
    Column("delivered_at", DateTime(timezone=True), nullable=True),
    Column("expires_at", DateTime(timezone=True), nullable=True),
    Column("error_code", String(64), nullable=False, server_default=text("''")),
    Column("error_message", Text, nullable=False, server_default=text("''")),
    Column("metadata", JSON, nullable=False, server_default=text("'{}'")),
    Column("schema_version", Integer, nullable=False, server_default=text("1")),
    Column(
        "created_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    ),
    Column(
        "updated_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    ),
)

Index(ORDER_STATE_INDEX_NAME, orders.c.state)
Index(ORDER_OCCASION_INDEX_NAME, orders.c.occasion)
Index(ORDER_RUN_INDEX_NAME, orders.c.run_id)
Index(ORDER_IDEMPOTENCY_INDEX_NAME, orders.c.idempotency_key)
Index(ORDER_SCHEDULED_FOR_INDEX_NAME, orders.c.scheduled_for)
Index(ORDER_CREATED_AT_INDEX_NAME, orders.c.created_at)


deliveries = Table(
    "deliveries",
    metadata,
    Column("id", String(64), primary_key=True),
    Column(
        "order_id",
        String(64),
        ForeignKey("orders.id", ondelete="CASCADE"),
        nullable=True,
    ),
    Column("run_id", String(64), nullable=False, server_default=text("''")),
    Column("provider", String(64), nullable=False, server_default=text("'folder'")),
    Column("state", String(32), nullable=False, server_default=text("'pending'")),
    Column("target", Text, nullable=False, server_default=text("''")),
    Column("path", Text, nullable=False, server_default=text("''")),
    Column("url", Text, nullable=False, server_default=text("''")),
    Column("email", String(320), nullable=False, server_default=text("''")),
    Column("files", JSON, nullable=False, server_default=text("'[]'")),
    Column("scheduled_for", DateTime(timezone=True), nullable=True),
    Column("delivered_at", DateTime(timezone=True), nullable=True),
    Column("expires_at", DateTime(timezone=True), nullable=True),
    Column("attempts", Integer, nullable=False, server_default=text("0")),
    Column("max_attempts", Integer, nullable=False, server_default=text("3")),
    Column(
        "idempotency_key",
        String(255),
        nullable=False,
        server_default=text("''"),
    ),
    Column("confirmation", Text, nullable=False, server_default=text("''")),
    Column("error_code", String(64), nullable=False, server_default=text("''")),
    Column("error_message", Text, nullable=False, server_default=text("''")),
    Column("cost_usd", Float, nullable=False, server_default=text("0")),
    Column("attestation", JSON, nullable=False, server_default=text("'{}'")),
    Column("metadata", JSON, nullable=False, server_default=text("'{}'")),
    Column("schema_version", Integer, nullable=False, server_default=text("1")),
    Column(
        "created_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    ),
    Column(
        "updated_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    ),
)

Index(DELIVERY_ORDER_INDEX_NAME, deliveries.c.order_id)
Index(DELIVERY_STATE_INDEX_NAME, deliveries.c.state)
Index(DELIVERY_SCHEDULED_FOR_INDEX_NAME, deliveries.c.scheduled_for)
Index(DELIVERY_CREATED_AT_INDEX_NAME, deliveries.c.created_at)


posts = Table(
    "posts",
    metadata,
    Column("id", String(64), primary_key=True),
    Column(
        "order_id",
        String(64),
        ForeignKey("orders.id", ondelete="CASCADE"),
        nullable=True,
    ),
    Column("platform", String(64), nullable=False, server_default=text("''")),
    Column("account", String(128), nullable=False, server_default=text("''")),
    Column("profile", String(64), nullable=False, server_default=text("''")),
    Column("language", String(32), nullable=False, server_default=text("''")),
    Column("format", String(64), nullable=False, server_default=text("''")),
    Column("title", Text, nullable=False, server_default=text("''")),
    Column("caption", Text, nullable=False, server_default=text("''")),
    Column("hashtags", JSON, nullable=False, server_default=text("'[]'")),
    Column("mentions", JSON, nullable=False, server_default=text("'[]'")),
    Column("media", JSON, nullable=False, server_default=text("'[]'")),
    Column("media_type", String(64), nullable=False, server_default=text("''")),
    Column("visibility", String(32), nullable=False, server_default=text("''")),
    Column("state", String(32), nullable=False, server_default=text("'draft'")),
    Column("remote_id", String(255), nullable=False, server_default=text("''")),
    Column("url", Text, nullable=False, server_default=text("''")),
    Column("scheduled_for", DateTime(timezone=True), nullable=True),
    Column("published_at", DateTime(timezone=True), nullable=True),
    Column("dry_run", Boolean, nullable=False, server_default=text("0")),
    Column(
        "idempotency_key",
        String(255),
        nullable=False,
        server_default=text("''"),
    ),
    Column("run_id", String(64), nullable=False, server_default=text("''")),
    Column("owner", String(255), nullable=False, server_default=text("''")),
    Column("composition_id", String(64), nullable=False, server_default=text("''")),
    Column("disclosure_text", Text, nullable=False, server_default=text("''")),
    Column("disclosure_required", Boolean, nullable=False, server_default=text("0")),
    Column("disclosure_applied", Boolean, nullable=False, server_default=text("0")),
    Column("attestation", JSON, nullable=False, server_default=text("'{}'")),
    Column("error_code", String(64), nullable=False, server_default=text("''")),
    Column("error_message", Text, nullable=False, server_default=text("''")),
    Column("cost_usd", Float, nullable=False, server_default=text("0")),
    Column("metadata", JSON, nullable=False, server_default=text("'{}'")),
    Column("schema_version", Integer, nullable=False, server_default=text("1")),
    Column(
        "created_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    ),
    Column(
        "updated_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    ),
)

Index(POST_STATE_INDEX_NAME, posts.c.state)
Index(POST_PLATFORM_INDEX_NAME, posts.c.platform)
Index(POST_RUN_INDEX_NAME, posts.c.run_id)
Index(POST_CREATED_AT_INDEX_NAME, posts.c.created_at)


post_metrics = Table(
    "post_metrics",
    metadata,
    Column("id", String(64), primary_key=True),
    Column(
        "post_id",
        String(64),
        ForeignKey("posts.id", ondelete="CASCADE"),
        nullable=True,
    ),
    Column("platform", String(64), nullable=False, server_default=text("''")),
    Column("impressions", Integer, nullable=False, server_default=text("0")),
    Column("reach", Integer, nullable=False, server_default=text("0")),
    Column("views", Integer, nullable=False, server_default=text("0")),
    Column("likes", Integer, nullable=False, server_default=text("0")),
    Column("comments", Integer, nullable=False, server_default=text("0")),
    Column("shares", Integer, nullable=False, server_default=text("0")),
    Column("saves", Integer, nullable=False, server_default=text("0")),
    Column("clicks", Integer, nullable=False, server_default=text("0")),
    Column("watch_time_seconds", Float, nullable=False, server_default=text("0")),
    Column("engagement_rate", Float, nullable=False, server_default=text("0")),
    Column("captured_at", DateTime(timezone=True), nullable=True),
    Column("raw", JSON, nullable=False, server_default=text("'{}'")),
    Column("metadata", JSON, nullable=False, server_default=text("'{}'")),
    Column("schema_version", Integer, nullable=False, server_default=text("1")),
    Column(
        "created_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    ),
    Column(
        "updated_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    ),
)

Index(POST_METRIC_POST_INDEX_NAME, post_metrics.c.post_id)
Index(POST_METRIC_CAPTURED_AT_INDEX_NAME, post_metrics.c.captured_at)


web_card_hits = Table(
    "web_card_hits",
    metadata,
    Column("id", String(64), primary_key=True),
    Column(
        "order_id",
        String(64),
        ForeignKey("orders.id", ondelete="CASCADE"),
        nullable=True,
    ),
    Column("card_id", String(64), nullable=False, server_default=text("''")),
    Column("day", String(10), nullable=False, server_default=text("''")),
    Column("visitor_hash", String(128), nullable=False, server_default=text("''")),
    Column("hit_at", DateTime(timezone=True), nullable=True),
    Column("user_agent", Text, nullable=False, server_default=text("''")),
    Column("referrer", Text, nullable=False, server_default=text("''")),
    Column("metadata", JSON, nullable=False, server_default=text("'{}'")),
    Column("schema_version", Integer, nullable=False, server_default=text("1")),
    Column(
        "created_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    ),
    Column(
        "updated_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    ),
)

Index(WEB_CARD_HIT_ORDER_INDEX_NAME, web_card_hits.c.order_id)
Index(WEB_CARD_HIT_DAY_INDEX_NAME, web_card_hits.c.day)
Index(WEB_CARD_HIT_VISITOR_INDEX_NAME, web_card_hits.c.visitor_hash)
Index(WEB_CARD_HIT_CREATED_AT_INDEX_NAME, web_card_hits.c.created_at)
