"""``aivia music``: select a bed and report its source, license and ducking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from aivia.audio import MusicSelectStage
from aivia.cli import main
from aivia.cli import music as music_cli
from aivia.core.db import (
    RunRepository,
    create_database_engine,
    upgrade_database,
)
from aivia.core.models import Run, RunState
from aivia.core.run_dirs import RunDirectories

CC0 = {
    "name": "CC0-1.0",
    "url": "https://creativecommons.org/publicdomain/zero/1.0/",
    "commercial_use": True,
}
NON_COMMERCIAL = {
    "name": "Test Non Commercial",
    "url": "https://example.com/non-commercial",
    "commercial_use": False,
}


def _track(
    entry_id: str,
    *,
    mood: str,
    genre: str = "ambient",
    tempo_bpm: float = 90.0,
    duration_seconds: float = 30.0,
    tags: tuple[str, ...] = (),
    license: dict[str, Any] = CC0,
) -> dict[str, Any]:
    return {
        "id": entry_id,
        "title": entry_id.replace("-", " ").title(),
        "path": f"music/{entry_id}.wav",
        "duration_seconds": duration_seconds,
        "mood": mood,
        "genre": genre,
        "tempo_bpm": tempo_bpm,
        "license": license,
        "tags": list(tags),
    }


LIBRARY = {
    "tracks": [
        _track("calm-ambient", mood="calm", tags=("loop",)),
        _track(
            "dark-cinematic",
            mood="dark",
            genre="cinematic",
            tempo_bpm=70.0,
            license=NON_COMMERCIAL,
        ),
    ]
}


def _write_library(tmp_path: Path) -> Path:
    path = tmp_path / "library.yaml"
    path.write_text(yaml.safe_dump(LIBRARY, sort_keys=False), encoding="utf-8")
    return path


def _write_config(
    tmp_path: Path,
    *,
    library: Path | None = None,
    mood: str = "calm",
    source: str = "library",
    ducking: dict[str, Any] | None = None,
    formats: bool = False,
) -> Path:
    values: dict[str, Any] = {
        "music": {
            "source": source,
            "library_path": "" if library is None else str(library),
            "mood": mood,
            "volume_db": -9.0,
            "ducking": ducking
            or {
                "enabled": True,
                "threshold_db": -21.0,
                "ratio": 4.0,
                "attack_ms": 12.0,
                "release_ms": 250.0,
            },
        }
    }
    if formats:
        values["formats"] = {
            "sample": {
                "description": "A small music recipe.",
                "stages": [{"stage": "music"}],
                "capabilities": ["music_gen"],
                "output_type": "video",
                "handler": "save",
            }
        }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return path


def _music(
    capsys: pytest.CaptureFixture[str],
    *args: str,
) -> tuple[int, str, str]:
    code = main(["music", *args])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_music_selects_the_configured_library_bed_and_prints_its_rights(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _write_config(tmp_path, library=_write_library(tmp_path))

    code, out, err = _music(capsys, "--config", str(config))

    assert code == 0, err
    assert err == ""
    assert "Source: library" in out
    assert "Path:" in out
    assert "calm-ambient.wav" in out
    assert "Track: calm-ambient" in out
    assert "License: CC0-1.0" in out
    assert "Commercial use: yes" in out
    assert "Ducking: enabled" in out
    assert "threshold -21.0 dB" in out
    assert "ratio 4" in out
    assert "attack 12.0 ms" in out
    assert "release 250.0 ms" in out


def test_music_flags_override_the_configured_mood_and_ducking(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _write_config(tmp_path, library=_write_library(tmp_path))

    code, out, err = _music(
        capsys,
        "--config",
        str(config),
        "--mood",
        "dark",
        "--no-duck",
    )

    assert code == 0, err
    assert "Track: dark-cinematic" in out
    assert "License: Test Non Commercial" in out
    assert "Commercial use: no" in out
    assert "Ducking: disabled" in out


def test_music_can_print_the_selection_as_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _write_config(tmp_path, library=_write_library(tmp_path))

    code, out, err = _music(capsys, "--config", str(config), "--json")

    assert code == 0, err
    payload = json.loads(out)
    assert payload["source"] == "library"
    assert payload["id"] == "calm-ambient"
    assert payload["license_name"] == "CC0-1.0"
    assert payload["ducking"]["enabled"] is True
    assert payload["ducking"]["threshold_db"] == pytest.approx(-21.0)


def test_music_rejects_a_provider_bed_without_a_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _write_config(tmp_path, library=_write_library(tmp_path))

    code, _out, err = _music(
        capsys,
        "--config",
        str(config),
        "--source",
        "provider",
    )

    assert code == 1
    assert "run id" in err


def test_music_needs_a_library_when_none_is_configured(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _write_config(tmp_path)

    code, _out, err = _music(capsys, "--config", str(config))

    assert code == 1
    assert "library" in err


def _seed_run(database: Path, tmp_path: Path, *, run_id: str = "run_music") -> Run:
    engine = create_database_engine(database)
    try:
        run = Run(id=run_id, format="sample", state=RunState.QUEUED)
        RunDirectories(tmp_path / "data").create(run)
        RunRepository(engine).create(run)
        return run
    finally:
        engine.dispose()


def _write_timeline(run: Run) -> None:
    path = Path(run.run_dir) / "timeline.json"
    path.write_text(
        json.dumps(
            {
                "id": run.id,
                "duration_seconds": 6.0,
                "languages": [
                    {
                        "language": "en",
                        "voices": [
                            {
                                "id": "voice_0",
                                "source": "audio/voice.wav",
                                "start_seconds": 0.0,
                                "end_seconds": 6.0,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_music_runs_the_recipe_stage_for_an_existing_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "data" / "aivia.db"
    upgrade_database(database)
    run = _seed_run(database, tmp_path)
    _write_timeline(run)
    config = _write_config(
        tmp_path,
        library=_write_library(tmp_path),
        formats=True,
    )
    monkeypatch.setattr(
        music_cli,
        "build_stages",
        lambda _recipe: [MusicSelectStage()],
    )

    code, out, err = _music(
        capsys,
        run.id,
        "--config",
        str(config),
        "--database",
        str(database),
    )

    assert code == 0, err
    assert "Source: library" in out
    assert "Track: calm-ambient" in out
    assert "License: CC0-1.0" in out
    assert "Ducking: enabled" in out
    assert "Ducking verified: yes" in out
    assert "Cost: $0.000000" in out
