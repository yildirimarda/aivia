"""``aivia sfx``: place effects on a run timeline and report placements and cost."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from aivia.audio import SfxStage
from aivia.cli import main
from aivia.core.db import (
    RunRepository,
    create_database_engine,
    upgrade_database,
)
from aivia.core.models import Run, RunState
from aivia.core.run_dirs import RunDirectories
from aivia.core.stages import Stage
from aivia.providers.protocols.sfx import SFXRequest


class FakeSFX:
    """Return a stored path, measured duration and cost for each effect."""

    def __init__(self, *, cost_usd: float = 0.01) -> None:
        self.cost_usd = cost_usd
        self.requests: list[SFXRequest] = []

    async def __call__(self, request: SFXRequest) -> SimpleNamespace:
        self.requests.append(request)
        slug = request.prompt.replace(" ", "-").casefold()
        return SimpleNamespace(
            path=f"sfx/{slug}.mp3",
            duration_seconds=0.5,
            cost_usd=self.cost_usd,
            provider="fake",
            model="fake-sfx",
        )


def _write_config(
    tmp_path: Path,
    *,
    rule_files: list[str] | None = None,
    volume_db: float = -6.0,
    placement: str = "start",
) -> Path:
    values: dict[str, Any] = {
        "sfx": {
            "rule_files": rule_files or [],
            "volume_db": volume_db,
            "placement": placement,
        },
        "formats": {
            "sample": {
                "description": "A small sound effect recipe.",
                "stages": [{"stage": "sfx"}],
                "capabilities": ["sfx"],
                "output_type": "video",
                "handler": "save",
            }
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return path


def _write_rules(tmp_path: Path, rules: list[dict[str, Any]]) -> Path:
    path = tmp_path if tmp_path.suffix else tmp_path / "rules.yaml"
    path.write_text(yaml.safe_dump({"rules": rules}, sort_keys=False), encoding="utf-8")
    return path


def _timeline_payload() -> dict[str, Any]:
    return {
        "id": "run_sfx",
        "duration_seconds": 6.0,
        "beats": [
            {
                "id": "beat_0",
                "text": "A door slams shut.",
                "start_seconds": 0.0,
                "end_seconds": 2.0,
            },
            {
                "id": "beat_1",
                "text": "The camera clicks twice.",
                "start_seconds": 2.0,
                "end_seconds": 4.0,
            },
        ],
        "languages": [
            {
                "language": "en",
                "voices": [
                    {
                        "id": "voice_0",
                        "source": "audio/voice.wav",
                        "start_seconds": 0.0,
                        "end_seconds": 4.0,
                    }
                ],
            }
        ],
    }


def _seed_run(database: Path, tmp_path: Path, *, run_id: str = "run_sfx") -> Run:
    engine = create_database_engine(database)
    try:
        run = Run(id=run_id, format="sample", state=RunState.QUEUED)
        RunDirectories(tmp_path / "data").create(run)
        RunRepository(engine).create(run)
        return run
    finally:
        engine.dispose()


def _write_timeline(run: Run) -> Path:
    path = Path(run.run_dir) / "timeline.json"
    path.write_text(json.dumps(_timeline_payload()), encoding="utf-8")
    return path


def _sfx(
    capsys: pytest.CaptureFixture[str],
    *args: str,
) -> tuple[int, str, str]:
    code = main(["sfx", *args])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_sfx_places_configured_rules_and_prints_each_placement_and_cost(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "data" / "aivia.db"
    upgrade_database(database)
    run = _seed_run(database, tmp_path)
    _write_timeline(run)
    rules = _write_rules(
        tmp_path,
        [
            {
                "id": "door",
                "match": "door",
                "prompt": "heavy door creaks",
                "position": "start",
                "volume_db": -12.0,
            },
            {
                "id": "click",
                "beat": "beat_1",
                "prompt": "camera shutter",
                "position": "end",
                "duration_seconds": 0.5,
                "volume_db": -18.0,
            },
        ],
    )
    config = _write_config(tmp_path, rule_files=[str(rules)])
    fake = FakeSFX()
    monkeypatch.setattr(
        "aivia.cli.sfx.build_stages",
        lambda _recipe: [SfxStage(sfx=fake)],
    )

    code, out, err = _sfx(
        capsys,
        run.id,
        "--config",
        str(config),
        "--database",
        str(database),
    )

    assert code == 0, err
    assert err == ""
    assert "Rules: 2" in out
    assert "Placements: 2" in out
    assert "Cost: $0.020000" in out
    assert "Duration: 6.000s" in out
    assert "Duration unchanged: yes" in out
    assert "Timeline: timeline.json" in out
    assert "Manifest: sfx.json" in out
    assert "Rule: door" in out
    assert "Beat: beat_0" in out
    assert "Start: 0.000s" in out
    assert "End: 0.500s" in out
    assert "Volume: -12.0 dB" in out
    assert "Source: sfx/heavy-door-creaks.mp3" in out
    assert "Rule: click" in out
    assert "Beat: beat_1" in out
    assert "Start: 3.500s" in out
    assert "End: 4.000s" in out
    assert "Volume: -18.0 dB" in out
    assert [request.prompt for request in fake.requests] == [
        "heavy door creaks",
        "camera shutter",
    ]

    timeline = json.loads(
        (Path(run.run_dir) / "timeline.json").read_text(encoding="utf-8")
    )
    assert timeline["duration_seconds"] == pytest.approx(6.0)
    assert [track["role"] for track in timeline["music"]] == ["sfx", "sfx"]
    manifest = json.loads((Path(run.run_dir) / "sfx.json").read_text(encoding="utf-8"))
    assert manifest["placements"] == 2
    assert manifest["cost_usd"] == pytest.approx(0.02)


def test_sfx_flags_override_rules_volume_and_placement_and_support_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "data" / "aivia.db"
    upgrade_database(database)
    run = _seed_run(database, tmp_path)
    _write_timeline(run)
    configured = _write_rules(
        tmp_path,
        [{"id": "door", "match": "door", "prompt": "ignored"}],
    )
    override = _write_rules(
        tmp_path / "override.yaml",
        [{"id": "click", "beat": "beat_1", "prompt": "camera shutter"}],
    )
    config = _write_config(tmp_path, rule_files=[str(configured)])
    fake = FakeSFX(cost_usd=0.03)
    monkeypatch.setattr(
        "aivia.cli.sfx.build_stages",
        lambda _recipe: [SfxStage(sfx=fake)],
    )

    code, out, err = _sfx(
        capsys,
        run.id,
        "--rules",
        str(override),
        "--volume-db",
        "-3.0",
        "--placement",
        "center",
        "--json",
        "--config",
        str(config),
        "--database",
        str(database),
    )

    assert code == 0, err
    payload = json.loads(out)
    assert payload["placements"] == 1
    assert payload["cost_usd"] == pytest.approx(0.03)
    [placement] = payload["tracks"]
    assert placement["rule"] == "click"
    assert placement["beat"] == "beat_1"
    assert placement["volume_db"] == pytest.approx(-3.0)
    assert placement["start_seconds"] == pytest.approx(2.75)
    assert placement["end_seconds"] == pytest.approx(3.25)
    assert placement["source"] == "sfx/camera-shutter.mp3"
    assert [request.prompt for request in fake.requests] == ["camera shutter"]


def test_sfx_reports_a_missing_run_and_a_recipe_without_an_sfx_stage(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "data" / "aivia.db"
    upgrade_database(database)
    config = _write_config(tmp_path)

    code, out, err = _sfx(
        capsys,
        "run_missing",
        "--config",
        str(config),
        "--database",
        str(database),
    )
    assert code == 1
    assert out == ""
    assert "run not found: run_missing" in err

    run = _seed_run(database, tmp_path, run_id="run_render")
    render_config = tmp_path / "render.yaml"
    render_config.write_text(
        yaml.safe_dump(
            {
                "formats": {
                    "sample": {
                        "description": "Rendering only.",
                        "stages": [{"stage": "render"}],
                        "output_type": "image",
                        "handler": "save",
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "aivia.cli.sfx.build_stages",
        lambda _recipe: [Stage(name="render")],
    )

    code, out, err = _sfx(
        capsys,
        run.id,
        "--config",
        str(render_config),
        "--database",
        str(database),
    )
    assert code == 1
    assert out == ""
    assert "format has no sfx stage" in err


def test_sfx_requires_a_run_id(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _write_config(tmp_path)

    code, out, err = _sfx(capsys, "--config", str(config))

    assert code == 2
    assert out == ""
    assert "a run id is required" in err
