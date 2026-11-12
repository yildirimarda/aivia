"""Performance reporting by profile, mode, category, language, platform,
format, hook style and visual strategy plus cost per thousand views.

The report combines the cost ledger with the platform metric snapshots so
spend can be viewed beside the views it bought. A run's spend is the sum of its
effective cost events (an estimate replaced by an actual is not counted), and a
run's views are the sum of the views snapshotted for the run's posts. Each run
contributes its spend and views to each dimension value it carries, so a group's
cost per thousand views is spend divided by views times one thousand, or zero
when the group has no views.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import UTC, datetime

import pytest
from sqlalchemy import Engine, text

from aivia.analytics.reports import (
    PerformanceBucket,
    PerformanceReport,
    build_performance_report,
    performance_by,
)
from aivia.core.db import create_database_engine, upgrade_database

NOW = datetime(2026, 2, 3, 10, 0, tzinfo=UTC)


@pytest.fixture
def engine() -> Iterator[Engine]:
    created = create_database_engine(url="sqlite+pysqlite:///:memory:")
    upgrade_database(engine=created)
    try:
        yield created
    finally:
        created.dispose()


def _insert(
    engine: Engine,
    table: str,
    columns: list[str],
    rows: Sequence[tuple[object, ...]],
) -> None:
    placeholders = ", ".join(f":{name}" for name in columns)
    column_list = ", ".join(columns)
    sql = f"INSERT INTO {table} ({column_list}) VALUES ({placeholders})"
    parameters = [dict(zip(columns, row, strict=True)) for row in rows]
    with engine.begin() as connection:
        for parameter in parameters:
            connection.execute(text(sql), parameter)


def seed_performance(engine: Engine) -> None:
    """Two profiles, three runs, one replaced estimate and a viewless run."""
    runs = [
        ("run_daily_short", "daily", "short"),
        ("run_weekly_long", "weekly", "long"),
        ("run_daily_viewless", "daily", "short"),
        ("run_no_cost", "daily", "short"),
    ]
    _insert(
        engine,
        "runs",
        ["id", "profile", "format"],
        runs,
    )

    # run_daily_short: $0.10 actual, 1000 views on tiktok in English
    # run_weekly_long:  $0.20 actual, 2000 views on youtube in English
    # run_daily_viewless: estimate $0.05 replaced by actual $0.08, 500 views
    # on tiktok in Spanish
    # run_no_cost: no cost events, no posts
    cost_events = [
        # id, provider, model, operation, amount_usd, estimated, run_id, metadata
        (
            "evt_daily",
            "acme",
            "gen",
            "image",
            0.10,
            0,
            "run_daily_short",
            '{"mode": "script", "category": "news", "hook_style": "question", '
            '"visual_strategy": "hybrid"}',
        ),
        (
            "evt_weekly",
            "acme",
            "gen",
            "video",
            0.20,
            0,
            "run_weekly_long",
            '{"mode": "script", "category": "news", "hook_style": "question", '
            '"visual_strategy": "image"}',
        ),
        (
            "evt_viewless_est",
            "acme",
            "gen",
            "audio",
            0.05,
            1,
            "run_daily_viewless",
            "{}",
        ),
        (
            "evt_viewless_act",
            "acme",
            "gen",
            "audio",
            0.08,
            0,
            "run_daily_viewless",
            '{"mode": "draft", "category": "tutorial", "hook_style": "list", '
            '"visual_strategy": "stock", "replaces": "evt_viewless_est"}',
        ),
    ]
    _insert(
        engine,
        "cost_events",
        [
            "id",
            "provider",
            "model",
            "operation",
            "amount_usd",
            "estimated",
            "run_id",
            "metadata",
        ],
        cost_events,
    )

    posts = [
        # id, run_id, platform, language
        ("post_daily", "run_daily_short", "tiktok", "en"),
        ("post_weekly", "run_weekly_long", "youtube", "en"),
        ("post_viewless", "run_daily_viewless", "tiktok", "es"),
    ]
    _insert(
        engine,
        "posts",
        ["id", "run_id", "platform", "language"],
        posts,
    )

    metrics = [
        # id, post_id, views
        ("metric_daily", "post_daily", 1000),
        ("metric_weekly", "post_weekly", 2000),
        ("metric_viewless", "post_viewless", 500),
    ]
    _insert(
        engine,
        "post_metrics",
        ["id", "post_id", "views"],
        metrics,
    )


def _by_key(buckets: tuple[PerformanceBucket, ...]) -> dict[str, PerformanceBucket]:
    return {bucket.key: bucket for bucket in buckets}


def test_performance_by_profile_groups_spend_and_views(engine: Engine) -> None:
    seed_performance(engine)
    buckets = performance_by(engine, "profile")
    by_key = _by_key(buckets)
    assert set(by_key) == {"daily", "weekly"}
    daily = by_key["daily"]
    # $0.10 + $0.08 (the $0.05 estimate is replaced by the $0.08 actual)
    assert daily.cost_usd == pytest.approx(0.18)
    assert daily.views == 1500
    assert daily.cpm_usd == pytest.approx(0.18 / 1500 * 1000)
    weekly = by_key["weekly"]
    assert weekly.cost_usd == pytest.approx(0.20)
    assert weekly.views == 2000
    assert weekly.cpm_usd == pytest.approx(0.20 / 2000 * 1000)


def test_performance_by_format_groups_spend_and_views(engine: Engine) -> None:
    seed_performance(engine)
    buckets = performance_by(engine, "format")
    by_key = _by_key(buckets)
    assert set(by_key) == {"short", "long"}
    assert by_key["short"].cost_usd == pytest.approx(0.18)
    assert by_key["short"].views == 1500
    assert by_key["short"].cpm_usd == pytest.approx(0.12)
    assert by_key["long"].cost_usd == pytest.approx(0.20)
    assert by_key["long"].views == 2000
    assert by_key["long"].cpm_usd == pytest.approx(0.10)


def test_performance_by_platform_groups_spend_and_views(engine: Engine) -> None:
    seed_performance(engine)
    buckets = performance_by(engine, "platform")
    by_key = _by_key(buckets)
    assert set(by_key) == {"tiktok", "youtube"}
    assert by_key["tiktok"].cost_usd == pytest.approx(0.18)
    assert by_key["tiktok"].views == 1500
    assert by_key["tiktok"].cpm_usd == pytest.approx(0.12)
    assert by_key["youtube"].cost_usd == pytest.approx(0.20)
    assert by_key["youtube"].views == 2000
    assert by_key["youtube"].cpm_usd == pytest.approx(0.10)


def test_performance_by_language_groups_spend_and_views(engine: Engine) -> None:
    seed_performance(engine)
    buckets = performance_by(engine, "language")
    by_key = _by_key(buckets)
    assert set(by_key) == {"en", "es"}
    assert by_key["en"].cost_usd == pytest.approx(0.30)
    assert by_key["en"].views == 3000
    assert by_key["en"].cpm_usd == pytest.approx(0.10)
    assert by_key["es"].cost_usd == pytest.approx(0.08)
    assert by_key["es"].views == 500
    assert by_key["es"].cpm_usd == pytest.approx(0.16)


def test_performance_by_mode_category_hook_style_and_visual_strategy(
    engine: Engine,
) -> None:
    seed_performance(engine)
    # mode: script (daily+weekly), draft (viewless)
    by_mode = _by_key(performance_by(engine, "mode"))
    assert set(by_mode) == {"script", "draft"}
    assert by_mode["script"].cost_usd == pytest.approx(0.30)
    assert by_mode["script"].views == 3000
    assert by_mode["draft"].cost_usd == pytest.approx(0.08)
    assert by_mode["draft"].views == 500

    # category: news (daily+weekly), tutorial (viewless)
    by_category = _by_key(performance_by(engine, "category"))
    assert set(by_category) == {"news", "tutorial"}
    assert by_category["news"].cost_usd == pytest.approx(0.30)
    assert by_category["tutorial"].cost_usd == pytest.approx(0.08)

    # hook_style: question (daily+weekly), list (viewless)
    by_hook = _by_key(performance_by(engine, "hook_style"))
    assert set(by_hook) == {"question", "list"}
    assert by_hook["question"].cost_usd == pytest.approx(0.30)
    assert by_hook["list"].cost_usd == pytest.approx(0.08)

    # visual_strategy: hybrid, image, stock
    by_visual = _by_key(performance_by(engine, "visual_strategy"))
    assert set(by_visual) == {"hybrid", "image", "stock"}
    assert by_visual["hybrid"].cost_usd == pytest.approx(0.10)
    assert by_visual["hybrid"].views == 1000
    assert by_visual["image"].cost_usd == pytest.approx(0.20)
    assert by_visual["stock"].cost_usd == pytest.approx(0.08)


def test_performance_buckets_are_ordered_by_key(engine: Engine) -> None:
    seed_performance(engine)
    buckets = performance_by(engine, "platform")
    assert [bucket.key for bucket in buckets] == ["tiktok", "youtube"]


def test_a_run_with_no_views_reports_zero_cpm(engine: Engine) -> None:
    seed_performance(engine)
    # run_daily_viewless has spend but no cost-free... it has 500 views. Add a
    # spend-only run without posts by inserting a run and a cost event only.
    _insert(
        engine,
        "runs",
        ["id", "profile", "format"],
        [("run_spend_only", "daily", "short")],
    )
    _insert(
        engine,
        "cost_events",
        [
            "id",
            "provider",
            "model",
            "operation",
            "amount_usd",
            "estimated",
            "run_id",
            "metadata",
        ],
        [
            (
                "evt_spend_only",
                "acme",
                "gen",
                "image",
                0.05,
                0,
                "run_spend_only",
                '{"mode": "script", "category": "news", "hook_style": "question", '
                '"visual_strategy": "hybrid"}',
            )
        ],
    )
    buckets = performance_by(engine, "platform")
    by_key = _by_key(buckets)
    # spend_only run has no posts -> no platform -> "(none)" bucket, spend 0.05, views 0
    none_bucket = by_key["(none)"]
    assert none_bucket.cost_usd == pytest.approx(0.05)
    assert none_bucket.views == 0
    assert none_bucket.cpm_usd == 0.0


def test_unknown_database_reports_nothing(engine: Engine) -> None:
    report = build_performance_report(engine)
    assert report.by_profile == ()
    assert report.by_mode == ()
    assert report.by_category == ()
    assert report.by_language == ()
    assert report.by_platform == ()
    assert report.by_format == ()
    assert report.by_hook_style == ()
    assert report.by_visual_strategy == ()
    assert report.total_cost_usd == 0.0
    assert report.total_views == 0
    assert report.overall_cpm_usd == 0.0


def test_build_performance_report_totals_match_the_buckets(engine: Engine) -> None:
    seed_performance(engine)
    report = build_performance_report(engine)
    # daily = $0.10 + $0.08 (the $0.05 estimate is replaced), weekly = $0.20
    assert report.total_cost_usd == pytest.approx(0.38)
    assert report.total_views == 1000 + 2000 + 500
    assert report.overall_cpm_usd == pytest.approx(0.38 / 3500 * 1000)
    assert isinstance(report, PerformanceReport)
    assert report.generated_at.tzinfo is not None


def test_unknown_dimension_is_rejected(engine: Engine) -> None:
    with pytest.raises(ValueError, match="unknown performance dimension"):
        performance_by(engine, "occasion")


def test_bucket_fields_are_typed(engine: Engine) -> None:
    seed_performance(engine)
    bucket = performance_by(engine, "platform")[0]
    assert isinstance(bucket.dimension, str)
    assert isinstance(bucket.key, str)
    assert isinstance(bucket.cost_usd, float)
    assert isinstance(bucket.views, int)
    assert isinstance(bucket.cpm_usd, float)
