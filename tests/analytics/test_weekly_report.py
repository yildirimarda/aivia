"""Weekly report of spend, outputs, top performers and provider comparison.

The report scopes the cost ledger and the platform metric snapshots to a single
UTC week and summarizes spend, outputs, top performers and provider comparison.
It is delivered through a notifier as a notification message.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Sequence
from datetime import UTC, date, datetime
from typing import Any

import pytest
from sqlalchemy import Engine, Table, insert

from aivia.analytics.weekly import (
    TOP_PERFORMER_LIMIT,
    build_weekly_report,
    deliver_weekly_report,
    weekly_report_message,
)
from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.db.tables import cost_events, post_metrics, posts, runs
from aivia.core.models.provider import ProviderMeta
from aivia.providers.base import Provider
from aivia.providers.protocols.notify import (
    CAPABILITY_NOTIFY,
    NotificationMessage,
    NotificationReceipt,
)

WEEK_START = date(2026, 1, 5)
IN_WEEK = datetime(2026, 1, 6, 10, 0, tzinfo=UTC)
BEFORE_WEEK = datetime(2026, 1, 4, 10, 0, tzinfo=UTC)
AT_START = datetime(2026, 1, 5, 0, 0, tzinfo=UTC)
AT_END = datetime(2026, 1, 12, 0, 0, tzinfo=UTC)
AFTER_WEEK = datetime(2026, 1, 12, 10, 0, tzinfo=UTC)
GENERATED_AT = datetime(2026, 1, 13, 9, 0, tzinfo=UTC)


class FixedNotifier(Provider):
    """A notifier that records every message it is asked to send."""

    default_metadata = ProviderMeta(
        name="fixed-notifier",
        capabilities=[CAPABILITY_NOTIFY],
    )

    def __init__(self) -> None:
        super().__init__()
        self.sent: list[NotificationMessage] = []

    async def send(self, message: NotificationMessage) -> NotificationReceipt:
        self.sent.append(message)
        return NotificationReceipt(
            message_id="fake-1",
            channel="fake",
            provider="fake",
            delivered_at=GENERATED_AT,
        )


@pytest.fixture
def engine() -> Iterator[Engine]:
    created = create_database_engine(url="sqlite+pysqlite:///:memory:")
    upgrade_database(engine=created)
    try:
        yield created
    finally:
        created.dispose()


def _insert_rows(engine: Engine, table: Table, rows: Sequence[dict[str, Any]]) -> None:
    with engine.begin() as connection:
        for row in rows:
            connection.execute(insert(table), row)


def seed_weekly(engine: Engine) -> None:
    """Two providers, four posts and a replaced estimate inside the week."""
    _insert_rows(
        engine,
        runs,
        [
            {"id": "run_morning", "profile": "daily", "format": "short"},
            {"id": "run_evening", "profile": "weekly", "format": "long"},
        ],
    )

    # acme: $0.10 + $0.30 + $0.08 actual (the $0.05 estimate is replaced)
    # other: $0.20
    _insert_rows(
        engine,
        cost_events,
        [
            {
                "id": "evt_acme_image",
                "provider": "acme",
                "model": "gen",
                "operation": "image",
                "amount_usd": 0.10,
                "estimated": False,
                "run_id": "run_morning",
                "metadata": {"mode": "script"},
                "created_at": IN_WEEK,
            },
            {
                "id": "evt_acme_video",
                "provider": "acme",
                "model": "gen",
                "operation": "video",
                "amount_usd": 0.30,
                "estimated": False,
                "run_id": "run_evening",
                "metadata": {"mode": "script"},
                "created_at": IN_WEEK,
            },
            {
                "id": "evt_other_image",
                "provider": "other",
                "model": "gen",
                "operation": "image",
                "amount_usd": 0.20,
                "estimated": False,
                "run_id": "run_morning",
                "metadata": {},
                "created_at": IN_WEEK,
            },
            {
                "id": "evt_acme_audio_est",
                "provider": "acme",
                "model": "gen",
                "operation": "audio",
                "amount_usd": 0.05,
                "estimated": True,
                "run_id": "run_evening",
                "metadata": {},
                "created_at": IN_WEEK,
            },
            {
                "id": "evt_acme_audio_act",
                "provider": "acme",
                "model": "gen",
                "operation": "audio",
                "amount_usd": 0.08,
                "estimated": False,
                "run_id": "run_evening",
                "metadata": {"replaces": "evt_acme_audio_est"},
                "created_at": IN_WEEK,
            },
        ],
    )

    _insert_rows(
        engine,
        posts,
        [
            {
                "id": "post_top",
                "run_id": "run_morning",
                "platform": "tiktok",
                "language": "en",
                "created_at": IN_WEEK,
            },
            {
                "id": "post_second",
                "run_id": "run_evening",
                "platform": "youtube",
                "language": "en",
                "created_at": IN_WEEK,
            },
            {
                "id": "post_third",
                "run_id": "run_morning",
                "platform": "tiktok",
                "language": "es",
                "created_at": IN_WEEK,
            },
            {
                "id": "post_small",
                "run_id": "run_evening",
                "platform": "instagram",
                "language": "en",
                "created_at": IN_WEEK,
            },
        ],
    )

    _insert_rows(
        engine,
        post_metrics,
        [
            {
                "id": "metric_top",
                "post_id": "post_top",
                "views": 2000,
                "likes": 100,
                "comments": 20,
                "shares": 5,
            },
            {
                "id": "metric_second",
                "post_id": "post_second",
                "views": 1000,
                "likes": 50,
                "comments": 10,
                "shares": 2,
            },
            {
                "id": "metric_third",
                "post_id": "post_third",
                "views": 500,
                "likes": 25,
                "comments": 5,
                "shares": 1,
            },
            {
                "id": "metric_small",
                "post_id": "post_small",
                "views": 100,
                "likes": 5,
                "comments": 1,
                "shares": 0,
            },
        ],
    )


def test_weekly_report_sums_spend_excluding_replaced_estimates(
    engine: Engine,
) -> None:
    seed_weekly(engine)
    report = build_weekly_report(engine, WEEK_START, now=GENERATED_AT)
    # 0.10 + 0.30 + 0.20 + 0.08 (the 0.05 estimate is replaced by 0.08)
    assert report.spend_usd == pytest.approx(0.68)
    assert report.outputs == 4
    assert report.week_start == WEEK_START
    assert report.week_end == date(2026, 1, 12)
    assert report.generated_at == GENERATED_AT


def test_weekly_report_ranks_top_performers_by_views(engine: Engine) -> None:
    seed_weekly(engine)
    report = build_weekly_report(engine, WEEK_START)
    performers = report.top_performers
    assert [p.post_id for p in performers] == [
        "post_top",
        "post_second",
        "post_third",
        "post_small",
    ]
    assert performers[0].views == 2000
    assert performers[0].likes == 100
    assert performers[0].comments == 20
    assert performers[0].shares == 5
    assert performers[0].platform == "tiktok"
    assert performers[-1].views == 100


def test_weekly_report_compares_providers(engine: Engine) -> None:
    seed_weekly(engine)
    report = build_weekly_report(engine, WEEK_START)
    comparisons = report.provider_comparison
    assert [c.provider for c in comparisons] == ["acme", "other"]
    acme = comparisons[0]
    assert acme.spend_usd == pytest.approx(0.48)  # 0.10 + 0.30 + 0.08
    assert acme.calls == 3
    other = comparisons[1]
    assert other.spend_usd == pytest.approx(0.20)
    assert other.calls == 1


def test_weekly_report_includes_start_boundary_excludes_end(engine: Engine) -> None:
    seed_weekly(engine)
    _insert_rows(
        engine,
        cost_events,
        [
            {
                "id": "evt_start",
                "provider": "boundary",
                "model": "gen",
                "operation": "image",
                "amount_usd": 0.01,
                "estimated": False,
                "run_id": "run_morning",
                "metadata": {},
                "created_at": AT_START,
            },
            {
                "id": "evt_end",
                "provider": "boundary",
                "model": "gen",
                "operation": "image",
                "amount_usd": 0.02,
                "estimated": False,
                "run_id": "run_morning",
                "metadata": {},
                "created_at": AT_END,
            },
        ],
    )
    report = build_weekly_report(engine, WEEK_START)
    # 0.68 + 0.01 (start included, end excluded)
    assert report.spend_usd == pytest.approx(0.69)


def test_weekly_report_ignores_events_outside_the_week(engine: Engine) -> None:
    seed_weekly(engine)
    _insert_rows(
        engine,
        cost_events,
        [
            {
                "id": "evt_before",
                "provider": "before",
                "model": "gen",
                "operation": "image",
                "amount_usd": 9.00,
                "estimated": False,
                "run_id": "run_morning",
                "metadata": {},
                "created_at": BEFORE_WEEK,
            },
            {
                "id": "evt_after",
                "provider": "after",
                "model": "gen",
                "operation": "image",
                "amount_usd": 9.00,
                "estimated": False,
                "run_id": "run_morning",
                "metadata": {},
                "created_at": AFTER_WEEK,
            },
        ],
    )
    report = build_weekly_report(engine, WEEK_START)
    assert report.spend_usd == pytest.approx(0.68)


def test_empty_week_reports_nothing(engine: Engine) -> None:
    report = build_weekly_report(engine, WEEK_START)
    assert report.spend_usd == 0.0
    assert report.outputs == 0
    assert report.top_performers == ()
    assert report.provider_comparison == ()


def test_weekly_report_message_renders_all_sections(engine: Engine) -> None:
    seed_weekly(engine)
    report = build_weekly_report(engine, WEEK_START)
    message = weekly_report_message(report)
    assert message.title == "Weekly report"
    assert message.level.value == "info"
    assert f"Spend: ${report.spend_usd:,.2f}" in message.text
    assert "Outputs: 4 posts" in message.text
    assert "Top performers:" in message.text
    assert "post_top (tiktok): 2000 views" in message.text
    assert "Provider comparison:" in message.text
    assert "acme: $0.48 (3 calls)" in message.text
    assert "other: $0.20 (1 call)" in message.text


def test_top_performers_are_capped(engine: Engine) -> None:
    seed_weekly(engine)
    # Add more posts than the limit so the cap is exercised.
    extra_posts = [
        {
            "id": f"post_extra_{i}",
            "run_id": "run_morning",
            "platform": "tiktok",
            "language": "en",
            "created_at": IN_WEEK,
        }
        for i in range(TOP_PERFORMER_LIMIT + 5)
    ]
    _insert_rows(engine, posts, extra_posts)
    extra_metrics = [
        {
            "id": f"metric_extra_{i}",
            "post_id": f"post_extra_{i}",
            "views": i,
            "likes": 0,
            "comments": 0,
            "shares": 0,
        }
        for i in range(TOP_PERFORMER_LIMIT + 5)
    ]
    _insert_rows(engine, post_metrics, extra_metrics)
    report = build_weekly_report(engine, WEEK_START)
    assert len(report.top_performers) == TOP_PERFORMER_LIMIT
    # The seeded posts have higher views than the extras (0..14), so they lead.
    assert report.top_performers[0].views == 2000


def test_deliver_weekly_report_sends_a_message(engine: Engine) -> None:
    seed_weekly(engine)
    notifier = FixedNotifier()
    receipt = asyncio.run(
        deliver_weekly_report(engine, WEEK_START, notifier=notifier, now=GENERATED_AT)
    )
    assert isinstance(receipt, NotificationReceipt)
    assert receipt.delivered
    assert len(notifier.sent) == 1
    message = notifier.sent[0]
    assert message.title == "Weekly report"
    assert f"Spend: ${0.68:,.2f}" in message.text
