"""``aivia run --idempotency-key``: repeat a request without paying twice."""

from __future__ import annotations

from pathlib import Path

import pytest

from aivia.cli import main
from aivia.core.db import (
    JobRepository,
    RunRepository,
    create_database_engine,
    upgrade_database,
)
from aivia.core.models import Run, RunState, StageResult
from aivia.core.stages import Stage, StageContext


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.lstrip(), encoding="utf-8")
    return path


@pytest.fixture
def config(tmp_path: Path) -> Path:
    return write_config(
        tmp_path,
        """
providers:
  text:
    type: mock-llm
formats:
  sample:
    description: A small priced recipe.
    stages:
      - stage: write
    capabilities: [llm]
    output_type: video
    handler: save
  other:
    description: A second priced recipe.
    stages:
      - stage: write
    capabilities: [llm]
    output_type: video
    handler: save
""",
    )


@pytest.fixture
def database(tmp_path: Path) -> Path:
    path = tmp_path / "data" / "aivia.db"
    upgrade_database(path)
    return path


def run_command(
    capsys: pytest.CaptureFixture[str],
    *args: str,
) -> tuple[int, str, str]:
    code = main(["run", *args])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def run_id_from(output: str) -> str:
    line = next(line for line in output.splitlines() if line.startswith("Run: "))
    return line.removeprefix("Run: ").strip()


def common_args(config: Path, database: Path) -> list[str]:
    return ["--config", str(config), "--database", str(database)]


def test_a_repeated_key_returns_the_existing_run_and_job(
    config: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = [
        "--format",
        "sample",
        "--enqueue",
        "--idempotency-key",
        "  request-42  ",
        *common_args(config, database),
    ]

    first_code, first_out, first_err = run_command(capsys, *args)
    second_code, second_out, second_err = run_command(capsys, *args)

    assert first_code == 0
    assert first_err == ""
    assert second_code == 0
    assert second_err == ""
    run_id = run_id_from(first_out)
    assert run_id_from(second_out) == run_id
    assert "State: queued" in second_out

    engine = create_database_engine(database)
    try:
        runs = RunRepository(engine).list()
        assert [run.id for run in runs] == [run_id]
        assert runs[0].idempotency_key == "request-42"
        assert runs[0].idempotency_payload_hash
        assert len(JobRepository(engine).list(run_id=run_id)) == 1
    finally:
        engine.dispose()


def test_a_different_payload_with_the_same_key_conflicts(
    config: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    first_code, first_out, first_err = run_command(
        capsys,
        "--format",
        "sample",
        "--enqueue",
        "--idempotency-key",
        "request-42",
        *common_args(config, database),
    )
    assert first_code == 0
    assert first_err == ""
    run_id = run_id_from(first_out)

    code, out, err = run_command(
        capsys,
        "--format",
        "other",
        "--enqueue",
        "--idempotency-key",
        "request-42",
        *common_args(config, database),
    )

    assert code == 1
    assert out == ""
    assert "idempotency key 'request-42'" in err
    assert f"run {run_id}" in err
    assert "different payload" in err

    engine = create_database_engine(database)
    try:
        assert [run.id for run in RunRepository(engine).list()] == [run_id]
        assert len(JobRepository(engine).list(run_id=run_id)) == 1
    finally:
        engine.dispose()


def test_repeating_a_key_after_execution_does_not_execute_again(
    config: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class Write(Stage):
        name = "write"
        artifacts = ("script.md",)

        async def run(self, context: StageContext) -> StageResult:
            calls.append(context.run_dir.name)
            (context.run_dir / "script.md").write_text("script", encoding="utf-8")
            return StageResult(stage=self.name, artifacts=["script.md"])

    monkeypatch.setattr(
        "aivia.cli.run.build_stages",
        lambda recipe: [Write() for _stage in recipe.stages],
    )

    first_code, first_out, first_err = run_command(
        capsys,
        "--format",
        "sample",
        "--idempotency-key",
        "request-42",
        *common_args(config, database),
    )
    assert first_code == 0
    assert first_err == ""
    run_id = run_id_from(first_out)
    assert len(calls) == 1

    second_code, second_out, second_err = run_command(
        capsys,
        "--format",
        "sample",
        "--idempotency-key",
        "request-42",
        *common_args(config, database),
    )

    assert second_code == 0
    assert second_err == ""
    assert run_id_from(second_out) == run_id
    assert "State: succeeded" in second_out
    assert len(calls) == 1

    engine = create_database_engine(database)
    try:
        stored = RunRepository(engine).get(run_id)
        assert stored is not None
        assert stored.state is RunState.SUCCEEDED
        assert [run.id for run in RunRepository(engine).list()] == [run_id]
    finally:
        engine.dispose()


def test_runs_without_a_key_are_not_deduplicated(
    config: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    first_code, first_out, _ = run_command(
        capsys,
        "--format",
        "sample",
        "--enqueue",
        *common_args(config, database),
    )
    second_code, second_out, _ = run_command(
        capsys,
        "--format",
        "sample",
        "--enqueue",
        *common_args(config, database),
    )

    assert first_code == 0
    assert second_code == 0
    assert run_id_from(first_out) != run_id_from(second_out)

    engine = create_database_engine(database)
    try:
        assert len(RunRepository(engine).list()) == 2
    finally:
        engine.dispose()


def test_a_run_without_a_stored_payload_hash_is_still_matched(
    config: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    engine = create_database_engine(database)
    try:
        seeded = RunRepository(engine).create(
            Run(
                id="run_seeded",
                state=RunState.SUCCEEDED,
                idempotency_key="request-42",
                format="sample",
            )
        )
    finally:
        engine.dispose()

    code, out, err = run_command(
        capsys,
        "--format",
        "sample",
        "--enqueue",
        "--idempotency-key",
        "request-42",
        *common_args(config, database),
    )

    assert code == 0
    assert err == ""
    assert run_id_from(out) == seeded.id
