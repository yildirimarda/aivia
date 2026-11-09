"""End-to-end: one mock character run per strategy, strategy recorded.

Each configured consistency strategy is exercised once through ``aivia
character test --dry-run`` with only that strategy enabled. The mock provider
renders every scene, so the single configured strategy wins each scene and its
name is recorded on a cost event with provider ``character-test`` and
``operation`` equal to the strategy value.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aivia.cli import main
from aivia.cli.character import _DEFAULT_SCENES
from aivia.core.config.character import ConsistencyStrategy
from aivia.core.db import (
    CostEventRepository,
    create_database_engine,
    upgrade_database,
)
from aivia.core.models.cost import CostEvent


def _db_path(tmp_path: Path) -> Path:
    return tmp_path / "character-test.db"


def _recorded_cost_events(db_path: Path) -> list[CostEvent]:
    engine = create_database_engine(path=db_path)
    try:
        upgrade_database(engine=engine)
        return CostEventRepository(engine).list(
            provider="character-test", run_id="character-test"
        )
    finally:
        engine.dispose()


def _run_one_strategy(
    tmp_path: Path, strategy: ConsistencyStrategy, capsys: pytest.CaptureFixture[str]
) -> tuple[int, str, str]:
    code = main(
        [
            "--dry-run",
            "character",
            "test",
            "--database",
            str(_db_path(tmp_path)),
            "--set",
            f"character.consistency_strategies=[{strategy.value}]",
        ]
    )
    captured = capsys.readouterr()
    return code, captured.out, captured.err


@pytest.mark.parametrize("strategy", list(ConsistencyStrategy))
def test_one_mock_run_per_strategy_records_the_strategy(
    tmp_path: Path,
    strategy: ConsistencyStrategy,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, out, err = _run_one_strategy(tmp_path, strategy, capsys)

    assert code == 0, err
    assert err == ""
    assert strategy.value in out

    events = _recorded_cost_events(_db_path(tmp_path))
    assert len(events) == len(_DEFAULT_SCENES)
    for event in events:
        assert event.provider == "character-test"
        assert event.run_id == "character-test"
        assert event.operation == strategy.value
        assert event.unit_kind == "image"
        assert event.units == 1.0
        assert event.amount_usd == 0.02


def test_all_strategies_recorded_across_separate_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every strategy is recorded exactly once per scene across separate runs."""
    recorded: dict[ConsistencyStrategy, int] = {}
    for strategy in ConsistencyStrategy:
        db_path = tmp_path / f"{strategy.value}.db"
        code = main(
            [
                "--dry-run",
                "character",
                "test",
                "--database",
                str(db_path),
                "--set",
                f"character.consistency_strategies=[{strategy.value}]",
            ]
        )
        out = capsys.readouterr().out
        assert code == 0, out
        events = _recorded_cost_events(db_path)
        recorded[strategy] = len(events)

    assert set(recorded) == set(ConsistencyStrategy)
    for count in recorded.values():
        assert count == len(_DEFAULT_SCENES)
