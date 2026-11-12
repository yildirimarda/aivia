"""Feeding performance back into category weights.

Category weights are adjusted from measured performance: categories that
deliver views cheaply gain weight, expensive ones lose it. The learning rate
bounds how far a weight may move in one update, a manual override pins a
weight so performance cannot move it, and every change is recorded in a change
log.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import Engine
from tests.analytics.test_reports import seed_performance

from aivia.analytics.learning import (
    DEFAULT_WEIGHT,
    MAX_WEIGHT,
    MIN_WEIGHT,
    CategoryWeightChange,
    CategoryWeights,
    CategoryWeightStore,
    apply_performance_to_category_weights,
    apply_performance_to_report,
    category_weights_path,
    load_category_weights,
    save_category_weights,
)
from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.errors import ConfigError
from aivia.core.performance import PerformanceBucket, performance_by


@pytest.fixture
def engine() -> Engine:
    created = create_database_engine(url="sqlite+pysqlite:///:memory:")
    upgrade_database(engine=created)
    try:
        yield created
    finally:
        created.dispose()


def _bucket(category: str, cost_usd: float, views: int) -> PerformanceBucket:
    cpm_usd = cost_usd / views * 1000.0 if views else 0.0
    return PerformanceBucket(
        dimension="category",
        key=category,
        cost_usd=cost_usd,
        views=views,
        cpm_usd=cpm_usd,
    )


def test_efficient_category_gains_weight_inefficient_loses() -> None:
    # news: 3000 views / $0.30 = 10000 views per dollar
    # tutorial: 500 views / $0.08 = 6250 views per dollar
    buckets = (
        _bucket("news", 0.30, 3000),
        _bucket("tutorial", 0.08, 500),
    )
    weights, changes = apply_performance_to_category_weights(
        {"news": 1.0, "tutorial": 1.0}, buckets
    )
    assert weights["news"] > 1.0
    assert weights["tutorial"] < 1.0
    assert len(changes) == 2
    assert changes[0].category == "news"
    assert changes[0].reason == "performance"
    assert changes[1].category == "tutorial"
    assert changes[1].reason == "performance"


def test_learning_rate_bounds_each_change() -> None:
    buckets = (
        _bucket("news", 0.30, 3000),
        _bucket("tutorial", 0.08, 500),
    )
    _weights, changes = apply_performance_to_category_weights(
        {"news": 1.0, "tutorial": 1.0}, buckets, learning_rate=0.5
    )
    # The target is efficiency relative to the mean efficiency; the move is
    # exactly learning_rate of the gap, so it is bounded by the learning rate.
    efficiencies = {
        "news": 3000 / 0.30,
        "tutorial": 500 / 0.08,
    }
    mean = (efficiencies["news"] + efficiencies["tutorial"]) / 2
    for change in changes:
        target = efficiencies[change.category] / mean
        assert abs(change.new_weight - change.old_weight) == pytest.approx(
            0.5 * abs(target - change.old_weight)
        )


def test_custom_learning_rate_scales_the_move() -> None:
    buckets = (
        _bucket("news", 0.30, 3000),
        _bucket("tutorial", 0.08, 500),
    )
    small, _ = apply_performance_to_category_weights(
        {"news": 1.0, "tutorial": 1.0}, buckets, learning_rate=0.1
    )
    large, _ = apply_performance_to_category_weights(
        {"news": 1.0, "tutorial": 1.0}, buckets, learning_rate=0.9
    )
    assert abs(large["news"] - DEFAULT_WEIGHT) > abs(small["news"] - DEFAULT_WEIGHT)


def test_category_without_performance_keeps_its_weight() -> None:
    buckets = (_bucket("news", 0.30, 3000), _bucket("tutorial", 0.08, 500))
    weights, changes = apply_performance_to_category_weights(
        {"news": 1.0, "tutorial": 2.0, "evergreen": 3.0}, buckets
    )
    assert weights["news"] != 1.0
    assert weights["tutorial"] != 2.0
    assert weights["evergreen"] == 3.0
    assert all(change.category != "evergreen" for change in changes)


def test_unknown_category_starts_at_default_weight() -> None:
    # Two categories are needed so the mean differs from one category's own
    # efficiency; an unknown category then starts at the default and moves.
    buckets = (
        _bucket("news", 0.30, 3000),
        _bucket("tutorial", 0.08, 500),
    )
    weights, _ = apply_performance_to_category_weights({}, buckets)
    # news starts at the default weight then moves toward its target
    assert weights["news"] != DEFAULT_WEIGHT
    assert weights["tutorial"] != DEFAULT_WEIGHT


def test_single_reported_category_changes_nothing() -> None:
    # One category is its own mean, so its target is 1.0 and nothing moves.
    buckets = (_bucket("news", 0.30, 3000),)
    weights, changes = apply_performance_to_category_weights({"news": 1.0}, buckets)
    assert weights["news"] == DEFAULT_WEIGHT
    assert changes == []


def test_zero_learning_rate_changes_nothing() -> None:
    buckets = (
        _bucket("news", 0.30, 3000),
        _bucket("tutorial", 0.08, 500),
    )
    weights, changes = apply_performance_to_category_weights(
        {"news": 1.0, "tutorial": 1.0}, buckets, learning_rate=0.0
    )
    assert weights == {"news": 1.0, "tutorial": 1.0}
    assert changes == []


def test_negative_learning_rate_is_rejected() -> None:
    buckets = (_bucket("news", 0.30, 3000),)
    with pytest.raises(ConfigError, match="learning rate"):
        apply_performance_to_category_weights(
            {"news": 1.0}, buckets, learning_rate=-0.1
        )


def test_non_finite_learning_rate_is_rejected() -> None:
    buckets = (_bucket("news", 0.30, 3000),)
    for bad in (float("inf"), float("nan")):
        with pytest.raises(ConfigError, match="learning rate"):
            apply_performance_to_category_weights(
                {"news": 1.0}, buckets, learning_rate=bad
            )


def test_invalid_weight_is_rejected() -> None:
    buckets = (_bucket("news", 0.30, 3000), _bucket("tutorial", 0.08, 500))
    with pytest.raises(ConfigError, match="category weight"):
        apply_performance_to_category_weights({"news": -1.0, "tutorial": 1.0}, buckets)


def test_weights_are_clamped_at_bounds() -> None:
    # A zero-efficiency category next to an efficient one targets zero; with a
    # full learning rate it decays exactly to the minimum bound.
    buckets = (
        _bucket("dead", 0.30, 0),
        _bucket("news", 0.30, 3000),
    )
    weights, _ = apply_performance_to_category_weights(
        {"dead": 1.0, "news": 1.0}, buckets, learning_rate=1.0
    )
    assert weights["dead"] == MIN_WEIGHT
    assert MIN_WEIGHT <= weights["news"] <= MAX_WEIGHT


def test_change_log_records_old_and_new_weights() -> None:
    buckets = (
        _bucket("news", 0.30, 3000),
        _bucket("tutorial", 0.08, 500),
    )
    _, changes = apply_performance_to_category_weights(
        {"news": 1.0, "tutorial": 1.0}, buckets
    )
    assert len(changes) == 2
    for change in changes:
        assert isinstance(change, CategoryWeightChange)
        assert change.old_weight == 1.0
        assert change.new_weight != 1.0
        assert change.reason == "performance"
        assert change.recorded_at.tzinfo is not None


def test_change_log_entry_is_immutable() -> None:
    buckets = (_bucket("news", 0.30, 3000), _bucket("tutorial", 0.08, 500))
    _, changes = apply_performance_to_category_weights(
        {"news": 1.0, "tutorial": 1.0}, buckets
    )
    # Frozen models reject reassignment.
    with pytest.raises((ValueError, TypeError)):
        changes[0].new_weight = 999.0  # type: ignore[misc]


def test_manual_override_pins_weight_against_performance() -> None:
    buckets = (
        _bucket("news", 0.30, 3000),
        _bucket("tutorial", 0.08, 500),
    )
    # news performs better and would gain weight, but it is overridden.
    weights, changes = apply_performance_to_category_weights(
        {"news": 5.0, "tutorial": 1.0},
        buckets,
        overrides={"news": 5.0},
    )
    assert weights["news"] == 5.0
    assert weights["tutorial"] != 1.0
    assert all(change.category != "news" for change in changes)


def test_effective_weight_respects_override() -> None:
    weights = CategoryWeights(weights={"news": 1.0}, overrides={"news": 7.0})
    assert weights.effective("news") == 7.0
    assert weights.effective("tutorial") == DEFAULT_WEIGHT
    assert weights.effective("missing") == DEFAULT_WEIGHT


def test_apply_performance_to_report_uses_the_database(engine: Engine) -> None:
    seed_performance(engine)
    weights, changes = apply_performance_to_report(
        engine, {"news": 1.0, "tutorial": 1.0}
    )
    # news: $0.30, 3000 views; tutorial: $0.08, 500 views -> news gains
    assert weights["news"] > 1.0
    assert weights["tutorial"] < 1.0
    assert len(changes) == 2


def test_unknown_dimension_is_rejected(engine: Engine) -> None:
    seed_performance(engine)
    with pytest.raises(ValueError, match="unknown performance dimension"):
        performance_by(engine, "occasion")


def test_save_and_load_round_trip(tmp_path) -> None:
    path = tmp_path / "analytics" / "category_weights.json"
    recorded_at = datetime(2026, 2, 3, 10, 0, tzinfo=UTC)
    weights = CategoryWeights.model_validate(
        {
            "weights": {"news": 1.5},
            "overrides": {"tutorial": 0.5},
            "changes": [
                {
                    "category": "news",
                    "old_weight": 1.0,
                    "new_weight": 1.5,
                    "reason": "performance",
                    "recorded_at": recorded_at.isoformat(),
                }
            ],
        }
    )
    saved = save_category_weights(weights, path=path)
    assert saved is weights
    assert path.exists()
    loaded = load_category_weights(path=path)
    assert loaded is not None
    assert loaded.weights == {"news": 1.5}
    assert loaded.overrides == {"tutorial": 0.5}
    assert len(loaded.changes) == 1
    assert loaded.changes[0].category == "news"
    assert loaded.changes[0].old_weight == 1.0
    assert loaded.changes[0].new_weight == 1.5
    assert loaded.changes[0].reason == "performance"
    assert loaded.changes[0].recorded_at == recorded_at


def test_load_returns_none_when_absent(tmp_path) -> None:
    assert load_category_weights(path=tmp_path / "missing.json") is None


def test_category_weights_path_shape() -> None:
    path = category_weights_path(data_dir="/tmp/x")
    assert path.parent.name == "analytics"
    assert path.name == "category_weights.json"


def test_override_blocks_learning() -> None:
    # news performs better and would gain weight, but it is overridden to 9.0.
    buckets = (
        _bucket("news", 0.30, 3000),
        _bucket("tutorial", 0.08, 500),
    )
    weights, changes = apply_performance_to_category_weights(
        {"news": 9.0, "tutorial": 1.0},
        buckets,
        overrides={"news": 9.0},
    )
    assert weights["news"] == 9.0
    assert weights["tutorial"] != 1.0
    assert all(change.category != "news" for change in changes)


def test_store_set_override_blocks_learning_and_persists(tmp_path, engine) -> None:
    seed_performance(engine)
    path = tmp_path / "analytics" / "category_weights.json"
    store = CategoryWeightStore(path=path)
    store.set_override("news", 9.0)
    loaded = store.load()
    assert loaded is not None
    assert loaded.overrides == {"news": 9.0}
    assert loaded.effective("news") == 9.0
    # Feeding performance into the store leaves the overridden weight pinned
    # while still moving non-overridden categories.
    result = store.apply_performance(engine)
    assert result.effective("news") == 9.0
    assert result.effective("tutorial") != DEFAULT_WEIGHT
    # The override and its change log entry are persisted.
    stored = load_category_weights(path=path)
    assert stored is not None
    assert stored.overrides == {"news": 9.0}
    assert any(
        change.category == "news" and change.reason == "override"
        for change in stored.changes
    )


def test_store_remove_override(tmp_path, engine) -> None:
    path = tmp_path / "analytics" / "category_weights.json"
    store = CategoryWeightStore(path=path)
    store.set_override("news", 9.0)
    assert store.load().overrides == {"news": 9.0}
    store.remove_override("news")
    assert store.load().overrides == {}


def test_store_apply_performance_persists_change_log(tmp_path, engine) -> None:
    seed_performance(engine)
    path = tmp_path / "analytics" / "category_weights.json"
    store = CategoryWeightStore(path=path)
    result = store.apply_performance(engine)
    assert result.weights["news"] > 1.0
    assert result.weights["tutorial"] < 1.0
    # The change log is persisted alongside the weights.
    stored = load_category_weights(path=store.path)
    assert stored is not None
    assert len(stored.changes) == 2
    assert {change.category for change in stored.changes} == {"news", "tutorial"}
    assert all(change.reason == "performance" for change in stored.changes)
