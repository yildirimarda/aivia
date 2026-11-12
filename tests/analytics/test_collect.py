"""Collecting platform metrics into the canonical shape.

Known metrics pass through their values, absent known metrics are ``None``,
and unknown metrics map to ``None`` (they are not collected).
"""

from __future__ import annotations

from aivia.analytics.collect import CANONICAL_METRICS, collect


def test_collect_returns_every_canonical_metric() -> None:
    result = collect({})

    assert set(result) == set(CANONICAL_METRICS)
    assert all(value is None for value in result.values())


def test_collect_passes_through_all_known_metrics() -> None:
    raw = {
        "views": 1_000,
        "watch_time": 3600,
        "likes": 50,
        "shares": 5,
        "comments": 10,
        "followers": 100,
    }

    assert collect(raw) == raw


def test_unknown_metrics_map_to_null() -> None:
    raw = {"views": 1, "impressions": 2, "reach": 3, "saves": 4}

    result = collect(raw)

    assert result["views"] == 1
    assert "impressions" not in result
    assert "reach" not in result
    assert "saves" not in result
    assert result["watch_time"] is None
    assert result["likes"] is None
    assert result["shares"] is None
    assert result["comments"] is None
    assert result["followers"] is None


def test_partial_known_metrics_are_null_for_the_rest() -> None:
    raw = {"likes": 7, "comments": 2}

    result = collect(raw)

    assert result["likes"] == 7
    assert result["comments"] == 2
    assert result["views"] is None
    assert result["watch_time"] is None
    assert result["shares"] is None
    assert result["followers"] is None
