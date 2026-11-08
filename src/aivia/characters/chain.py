"""Character consistency chain: run the configured strategies per scene.

A run selects one character and the character config names an ordered chain of
consistency strategies (``composite -> reference -> adapter -> prompt_only``).
For every scene the strategies are tried in that order. The first strategy
that returns an asset wins and its name is recorded on the result, so a run
can replay exactly which strategy produced each asset. Strategies that return
``None`` decline the scene and the next strategy is tried; when every strategy
declines the scene's strategy is recorded as ``None``.

This module is the chain runner only: the concrete strategies (composite,
reference, adapter, prompt only) are plugged in by the caller.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from aivia.core.config.character import CharacterConfig, ConsistencyStrategy

__all__ = [
    "Asset",
    "Scene",
    "SceneResult",
    "Strategy",
    "run_strategy_chain",
]

Scene = str
"""One scene: the prompt describing what to render for that scene."""

Asset = Any
"""An asset produced by a strategy; its shape is strategy specific."""

Strategy = Callable[[Scene], Asset | None]
"""A consistency strategy for one scene.

Returns the asset it produced, or ``None`` to decline and let the chain try the
next strategy.
"""


@dataclass(frozen=True, slots=True)
class SceneResult:
    """Outcome of running the strategy chain for one scene.

    ``strategy`` is the configured strategy that produced ``asset``, or
    ``None`` when every strategy declined the scene.
    """

    scene_index: int
    strategy: ConsistencyStrategy | None
    asset: Asset | None


def run_strategy_chain(
    scenes: Sequence[Scene],
    *,
    config: CharacterConfig,
    strategies: Mapping[ConsistencyStrategy, Strategy],
) -> list[SceneResult]:
    """Run the configured strategy chain for every scene.

    Strategies are tried in ``config.consistency_strategies`` order. The first
    strategy that returns a non-``None`` asset wins for that scene and its name
    is recorded. Strategies with no registered callable are skipped.
    """
    results: list[SceneResult] = []
    for index, scene in enumerate(scenes):
        winner: ConsistencyStrategy | None = None
        produced: Asset | None = None
        for strategy in config.consistency_strategies:
            attempt = strategies.get(strategy)
            if attempt is None:
                continue
            produced = attempt(scene)
            if produced is not None:
                winner = strategy
                break
        results.append(SceneResult(scene_index=index, strategy=winner, asset=produced))
    return results
