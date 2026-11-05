"""``aivia visuals``: report the per beat decision, provider, cost and quality."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from aivia.cli import main
from aivia.core.db import (
    AssetRepository,
    RunRepository,
    create_database_engine,
    upgrade_database,
)
from aivia.core.models import (
    Asset,
    AssetCostLink,
    AssetKind,
    Run,
    RunState,
    StageResult,
)
from aivia.core.run_dirs import RunDirectories
from aivia.core.stages import Stage, StageContext


def _write_config(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
formats:
  sample:
    description: A small visual recipe.
    stages:
      - stage: visuals
    capabilities: [image_gen]
    output_type: video
    handler: save
""".lstrip(),
        encoding="utf-8",
    )
    return path


def _seed_run(
    database: Path,
    tmp_path: Path,
    *,
    run_id: str = "run_visuals",
) -> Run:
    engine = create_database_engine(database)
    try:
        run = Run(id=run_id, format="sample", state=RunState.QUEUED)
        RunDirectories(tmp_path / "data").create(run)
        RunRepository(engine).create(run)
        return run
    finally:
        engine.dispose()


def _write_manifest(run: Run, name: str, payload: dict[str, Any]) -> Path:
    path = Path(run.run_dir) / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _visuals(
    capsys: pytest.CaptureFixture[str],
    *args: str,
) -> tuple[int, str, str]:
    code = main(["visuals", *args])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _manifests(run: Run) -> None:
    _write_manifest(
        run,
        "image.json",
        {
            "version": 1,
            "stage": "image",
            "beats": [
                {
                    "index": 0,
                    "request": {"index": 0, "visual_type": "image"},
                    "provider": "fake-image",
                    "model": "image-1",
                    "cost_usd": 0.02,
                    "asset": {
                        "id": "asset_0",
                        "kind": "image",
                        "provider": "fake-image",
                        "model": "image-1",
                        "quality_score": 0.7,
                    },
                }
            ],
        },
    )
    _write_manifest(
        run,
        "video.json",
        {
            "version": 1,
            "stage": "video",
            "beats": [
                {
                    "index": 1,
                    "request": {"index": 1, "visual_type": "video"},
                    "provider": "fake-video",
                    "model": "video-1",
                    "cost_usd": 0.5,
                    "asset": {
                        "id": "asset_1",
                        "kind": "video",
                        "provider": "fake-video",
                        "model": "video-1",
                        "quality_score": 0.65,
                    },
                }
            ],
        },
    )
    _write_manifest(
        run,
        "visual-qa.json",
        {
            "version": 1,
            "stage": "visual_qa",
            "beats": [
                {
                    "index": 0,
                    "kind": "image",
                    "provider": "fake-image",
                    "model": "image-1",
                    "score": 0.93,
                    "cost_usd": 0.04,
                },
                {
                    "index": 1,
                    "kind": "video",
                    "provider": "fake-video",
                    "model": "video-1",
                    "score": 0.81,
                    "cost_usd": 0.6,
                },
            ],
        },
    )


def test_visuals_prints_the_per_beat_decision_provider_cost_and_quality(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = tmp_path / "data" / "aivia.db"
    upgrade_database(database)
    run = _seed_run(database, tmp_path)
    _manifests(run)
    config = _write_config(tmp_path)

    code, out, err = _visuals(
        capsys,
        run.id,
        "--config",
        str(config),
        "--database",
        str(database),
    )

    assert code == 0, err
    assert err == ""
    assert "Run: run_visuals" in out
    assert "Beats: 2" in out
    assert "Cost: $0.640000" in out
    assert "Beat: 0" in out
    assert "Decision: image" in out
    assert "Provider: fake-image" in out
    assert "Cost: $0.040000" in out
    assert "Quality: 0.930" in out
    assert "Beat: 1" in out
    assert "Decision: video" in out
    assert "Provider: fake-video" in out
    assert "Cost: $0.600000" in out
    assert "Quality: 0.810" in out


def test_visuals_can_print_the_per_beat_report_as_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = tmp_path / "data" / "aivia.db"
    upgrade_database(database)
    run = _seed_run(database, tmp_path)
    _manifests(run)
    config = _write_config(tmp_path)

    code, out, err = _visuals(
        capsys,
        run.id,
        "--json",
        "--config",
        str(config),
        "--database",
        str(database),
    )

    assert code == 0, err
    payload = json.loads(out)
    assert payload["run"] == "run_visuals"
    assert payload["cost_usd"] == pytest.approx(0.64)
    assert [beat["index"] for beat in payload["beats"]] == [0, 1]
    assert payload["beats"][0]["decision"] == "image"
    assert payload["beats"][0]["provider"] == "fake-image"
    assert payload["beats"][0]["cost_usd"] == pytest.approx(0.04)
    assert payload["beats"][0]["quality_score"] == pytest.approx(0.93)


class FakeVisual(Stage):
    """A visual stage that records a result and one priced provider call."""

    name = "visuals"
    artifacts = ("visuals.json",)

    def __init__(self) -> None:
        super().__init__(name="visuals", artifacts=("visuals.json",))
        self.results: list[Any] = []

    async def run(self, context: StageContext) -> StageResult:
        assert context.run is not None
        assert context.cost is not None
        event = context.cost.record(
            provider="fake",
            model="fake-image",
            operation="image_gen",
            units=1.0,
            unit_kind="image",
            amount_usd=0.03,
            run_id=context.run.id,
        )
        self.results = [
            SimpleNamespace(
                index=0,
                request=SimpleNamespace(visual_type="image"),
                provider="fake",
                model="fake-image",
                cost_usd=0.03,
                score=0.88,
                cached=False,
                fallback="",
                error="",
                asset=None,
            )
        ]
        (context.run_dir / "visuals.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "stage": "visuals",
                    "beats": [
                        {
                            "index": 0,
                            "request": {"visual_type": "image"},
                            "provider": "fake",
                            "model": "fake-image",
                            "cost_usd": 0.03,
                            "score": 0.88,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return StageResult(
            run_id=context.run.id,
            stage=self.name,
            artifacts=["visuals.json"],
            cost_usd=event.amount_usd,
        )


def test_visuals_runs_concrete_stages_and_reports_their_results(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "data" / "aivia.db"
    upgrade_database(database)
    run = _seed_run(database, tmp_path)
    config = _write_config(tmp_path)
    stage = FakeVisual()
    monkeypatch.setattr(
        "aivia.cli.visuals.build_stages",
        lambda _recipe: [stage],
    )

    code, out, err = _visuals(
        capsys,
        run.id,
        "--config",
        str(config),
        "--database",
        str(database),
    )

    assert code == 0, err
    assert "Beat: 0" in out
    assert "Decision: image" in out
    assert "Provider: fake" in out
    assert "Cost: $0.030000" in out
    assert "Quality: 0.880" in out
    assert stage.results

    engine = create_database_engine(database)
    try:
        stored = RunRepository(engine).get(run.id)
        assert stored is not None
        assert stored.state is RunState.SUCCEEDED
        assert stored.cost_usd == pytest.approx(0.03)
    finally:
        engine.dispose()


def test_visuals_falls_back_to_asset_rows_when_no_manifest_exists(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = tmp_path / "data" / "aivia.db"
    upgrade_database(database)
    run = _seed_run(database, tmp_path)
    config = _write_config(tmp_path)
    engine = create_database_engine(database)
    try:
        AssetRepository(engine).create(
            Asset(
                kind=AssetKind.IMAGE,
                provider="fake-image",
                model="image-1",
                quality_score=0.77,
                cost_links=[AssetCostLink(cost_event_id="cost_1", amount_usd=0.05)],
                run_id=run.id,
            )
        )
    finally:
        engine.dispose()

    code, out, err = _visuals(
        capsys,
        run.id,
        "--config",
        str(config),
        "--database",
        str(database),
    )

    assert code == 0, err
    assert "Beat: 0" in out
    assert "Decision: image" in out
    assert "Provider: fake-image" in out
    assert "Cost: $0.050000" in out
    assert "Quality: 0.770" in out
