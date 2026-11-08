"""Character consistency chain: the configured strategies run per scene and
the strategy that produced each asset is recorded.
"""

from __future__ import annotations

from typing import Any

import pytest

from aivia.characters.chain import Asset, SceneResult, run_strategy_chain
from aivia.core.config.character import (
    CharacterConfig,
    ConsistencyStrategy,
)

DEFAULT_STRATEGIES = (
    ConsistencyStrategy.COMPOSITE,
    ConsistencyStrategy.REFERENCE,
    ConsistencyStrategy.ADAPTER,
    ConsistencyStrategy.PROMPT_ONLY,
)


def _config(*strategies: ConsistencyStrategy) -> CharacterConfig:
    return CharacterConfig(consistency_strategies=list(strategies))


def _asset(path: str) -> Asset:
    return {"path": path, "bytes": b"fake-asset"}


def test_first_acceptable_strategy_wins_and_later_strategies_are_skipped() -> None:
    calls: list[ConsistencyStrategy] = []

    def composite(scene: str) -> Asset | None:
        calls.append(ConsistencyStrategy.COMPOSITE)
        return _asset("composite.png")

    def reference(scene: str) -> Asset | None:
        calls.append(ConsistencyStrategy.REFERENCE)
        return _asset("reference.png")

    strategies: dict[ConsistencyStrategy, Any] = {
        ConsistencyStrategy.COMPOSITE: composite,
        ConsistencyStrategy.REFERENCE: reference,
    }
    results = run_strategy_chain(
        ["scene one", "scene two"],
        config=_config(*DEFAULT_STRATEGIES),
        strategies=strategies,
    )

    assert [r.strategy for r in results] == [
        ConsistencyStrategy.COMPOSITE,
        ConsistencyStrategy.COMPOSITE,
    ]
    assert [r.asset for r in results] == [
        _asset("composite.png"),
        _asset("composite.png"),
    ]
    assert calls == [
        ConsistencyStrategy.COMPOSITE,
        ConsistencyStrategy.COMPOSITE,
    ]


def test_strategies_are_tried_in_configured_order() -> None:
    calls: list[ConsistencyStrategy] = []

    def composite(scene: str) -> Asset | None:
        calls.append(ConsistencyStrategy.COMPOSITE)
        return None

    def reference(scene: str) -> Asset | None:
        calls.append(ConsistencyStrategy.REFERENCE)
        return _asset("reference.png")

    def adapter(scene: str) -> Asset | None:
        calls.append(ConsistencyStrategy.ADAPTER)
        return _asset("adapter.png")

    strategies: dict[ConsistencyStrategy, Any] = {
        ConsistencyStrategy.COMPOSITE: composite,
        ConsistencyStrategy.REFERENCE: reference,
        ConsistencyStrategy.ADAPTER: adapter,
    }
    results = run_strategy_chain(
        ["scene one"],
        config=_config(*DEFAULT_STRATEGIES),
        strategies=strategies,
    )

    assert results[0].strategy is ConsistencyStrategy.REFERENCE
    assert results[0].asset == _asset("reference.png")
    assert calls == [
        ConsistencyStrategy.COMPOSITE,
        ConsistencyStrategy.REFERENCE,
    ]


def test_prompt_only_is_the_last_resort() -> None:
    calls: list[ConsistencyStrategy] = []

    def composite(scene: str) -> Asset | None:
        calls.append(ConsistencyStrategy.COMPOSITE)
        return None

    def reference(scene: str) -> Asset | None:
        calls.append(ConsistencyStrategy.REFERENCE)
        return None

    def adapter(scene: str) -> Asset | None:
        calls.append(ConsistencyStrategy.ADAPTER)
        return None

    def prompt_only(scene: str) -> Asset | None:
        calls.append(ConsistencyStrategy.PROMPT_ONLY)
        return _asset("prompt_only.png")

    strategies: dict[ConsistencyStrategy, Any] = {
        ConsistencyStrategy.COMPOSITE: composite,
        ConsistencyStrategy.REFERENCE: reference,
        ConsistencyStrategy.ADAPTER: adapter,
        ConsistencyStrategy.PROMPT_ONLY: prompt_only,
    }
    results = run_strategy_chain(
        ["scene one"],
        config=_config(*DEFAULT_STRATEGIES),
        strategies=strategies,
    )

    assert results[0].strategy is ConsistencyStrategy.PROMPT_ONLY
    assert results[0].asset == _asset("prompt_only.png")
    assert calls == list(DEFAULT_STRATEGIES)


def test_all_strategies_decline_record_none() -> None:
    calls: list[ConsistencyStrategy] = []

    def composite(scene: str) -> Asset | None:
        calls.append(ConsistencyStrategy.COMPOSITE)
        return None

    strategies: dict[ConsistencyStrategy, Any] = {
        ConsistencyStrategy.COMPOSITE: composite,
    }
    results = run_strategy_chain(
        ["scene one"],
        config=_config(
            ConsistencyStrategy.COMPOSITE,
            ConsistencyStrategy.PROMPT_ONLY,
        ),
        strategies=strategies,
    )

    assert results[0].strategy is None
    assert results[0].asset is None
    assert calls == [ConsistencyStrategy.COMPOSITE]


def test_each_scene_is_recorded_independently() -> None:
    def composite(scene: str) -> Asset | None:
        return _asset("composite.png")

    def adapter(scene: str) -> Asset | None:
        return _asset("adapter.png")

    strategies: dict[ConsistencyStrategy, Any] = {
        ConsistencyStrategy.COMPOSITE: composite,
        ConsistencyStrategy.ADAPTER: adapter,
    }
    results = run_strategy_chain(
        ["scene one", "scene two", "scene three"],
        config=_config(
            ConsistencyStrategy.COMPOSITE,
            ConsistencyStrategy.REFERENCE,
            ConsistencyStrategy.ADAPTER,
        ),
        strategies=strategies,
    )

    assert [r.scene_index for r in results] == [0, 1, 2]
    assert [r.strategy for r in results] == [
        ConsistencyStrategy.COMPOSITE,
        ConsistencyStrategy.COMPOSITE,
        ConsistencyStrategy.COMPOSITE,
    ]


def test_configured_strategy_without_a_callable_is_skipped() -> None:
    def composite(scene: str) -> Asset | None:
        return _asset("composite.png")

    strategies: dict[ConsistencyStrategy, Any] = {
        ConsistencyStrategy.COMPOSITE: composite,
        # REFERENCE and ADAPTER are configured but have no callable.
    }
    results = run_strategy_chain(
        ["scene one"],
        config=_config(
            ConsistencyStrategy.COMPOSITE,
            ConsistencyStrategy.REFERENCE,
            ConsistencyStrategy.ADAPTER,
            ConsistencyStrategy.PROMPT_ONLY,
        ),
        strategies=strategies,
    )

    assert results[0].strategy is ConsistencyStrategy.COMPOSITE
    assert results[0].asset == _asset("composite.png")


def test_empty_scenes_return_empty_results() -> None:
    results = run_strategy_chain(
        [],
        config=_config(*DEFAULT_STRATEGIES),
        strategies={},
    )

    assert results == []


def test_scene_result_fields_are_exposed() -> None:
    result = SceneResult(
        scene_index=3,
        strategy=ConsistencyStrategy.REFERENCE,
        asset=_asset("reference.png"),
    )
    assert result.scene_index == 3
    assert result.strategy is ConsistencyStrategy.REFERENCE
    assert result.asset == _asset("reference.png")


def test_config_rejects_prompt_only_not_last() -> None:
    with pytest.raises(ValueError, match="prompt_only must be the last"):
        _config(
            ConsistencyStrategy.PROMPT_ONLY,
            ConsistencyStrategy.COMPOSITE,
        )
