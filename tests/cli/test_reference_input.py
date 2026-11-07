"""Repeatable reference inputs are validated by extension and by content probe."""

from __future__ import annotations

import json
import struct
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from aivia.api import create_app
from aivia.cli import main
from aivia.core.config import ApiConfig
from aivia.core.db import (
    ReferenceRepository,
    RunRepository,
    create_database_engine,
    upgrade_database,
)
from aivia.core.errors import ValidationError
from aivia.reference import ReferenceKind, load_reference_inputs


def png(width: int = 2, height: int = 2) -> bytes:
    header = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0dIHDR"
    body = struct.pack(">II", width, height) + b"\x08\x02\x00\x00\x00"
    return header + body + b"\x00\x00\x00\x00"


def mp4() -> bytes:
    return struct.pack(">I", 24) + b"ftypisom" + b"\x00" * 12


def wav() -> bytes:
    return b"RIFF" + struct.pack("<I", 36) + b"WAVEfmt " + b"\x00" * 28


def probe_json(kind: str) -> str:
    if kind == "video":
        streams = [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264",
                "width": 640,
                "height": 360,
                "duration": "2.0",
                "avg_frame_rate": "30/1",
            }
        ]
        format_name = "mov,mp4,m4a,3gp,3g2,mj2"
    else:
        streams = [
            {
                "index": 0,
                "codec_type": "audio",
                "codec_name": "pcm_s16le",
                "duration": "2.0",
                "sample_rate": "44100",
                "channels": 2,
            }
        ]
        format_name = "wav"
    return json.dumps(
        {
            "streams": streams,
            "format": {
                "format_name": format_name,
                "duration": "2.0",
                "size": "1234",
            },
        }
    )


def runner_for(kind: str) -> Any:
    def runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            list(argv),
            0,
            stdout=probe_json(kind),
            stderr="",
        )

    return runner


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
formats:
  blank:
    description: A recipe with no stages.
    stages: []
    output_type: image
    handler: save
""",
    )


@pytest.fixture
def database(tmp_path: Path) -> Path:
    path = tmp_path / "data" / "aivia.db"
    upgrade_database(path)
    return path


def run_command(capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, str, str]:
    code = main(["run", *args])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_repeatable_cli_references_are_probed_and_stored(
    config: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    image = config.parent / "look.png"
    image.write_bytes(png())
    notes = config.parent / "notes.md"
    notes.write_text("# A calm opening\n\nShort cuts, warm light.\n", encoding="utf-8")

    code, _out, err = run_command(
        capsys,
        "--format",
        "blank",
        "--reference",
        str(image),
        "--reference",
        str(notes),
        "--config",
        str(config),
        "--database",
        str(database),
    )

    assert code == 0
    assert err == ""

    engine = create_database_engine(database)
    try:
        rows = ReferenceRepository(engine).list()
        assert {row["kind"] for row in rows} == {"image", "text"}
        assert {row["path"] for row in rows} == {str(image), str(notes)}
        for row in rows:
            assert row["content_hash"]
            assert row["size_bytes"] > 0
            assert row["mime_type"]
        image_row = next(row for row in rows if row["kind"] == "image")
        assert image_row["mime_type"] == "image/png"
        assert image_row["metadata"]["probe"] == "content"
        assert len(RunRepository(engine).list()) == 1
    finally:
        engine.dispose()


def test_cli_rejects_a_reference_whose_extension_lies_about_its_content(
    config: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    misleading = config.parent / "clip.mp4"
    misleading.write_bytes(png())

    code, _out, err = run_command(
        capsys,
        "--format",
        "blank",
        "--reference",
        str(misleading),
        "--config",
        str(config),
        "--database",
        str(database),
    )

    assert code == 1
    assert "says video" in err
    assert "is image" in err

    engine = create_database_engine(database)
    try:
        assert ReferenceRepository(engine).list() == []
        assert RunRepository(engine).list() == []
    finally:
        engine.dispose()


def test_cli_rejects_missing_and_unrecognizable_references(
    config: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = config.parent / "missing.png"
    code, _out, err = run_command(
        capsys,
        "--format",
        "blank",
        "--reference",
        str(missing),
        "--config",
        str(config),
        "--database",
        str(database),
    )
    assert code == 1
    assert "not found" in err

    unknown = config.parent / "mystery.bin"
    unknown.write_bytes(b"\x00\x01\x02\x03")
    code, _out, err = run_command(
        capsys,
        "--format",
        "blank",
        "--reference",
        str(unknown),
        "--config",
        str(config),
        "--database",
        str(database),
    )
    assert code == 1
    assert "not recognizable" in err


def test_references_are_estimated_without_writing_any_rows(
    config: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    image = config.parent / "look.png"
    image.write_bytes(png())

    code, _out, err = run_command(
        capsys,
        "--format",
        "blank",
        "--estimate",
        "--reference",
        str(image),
        "--config",
        str(config),
        "--database",
        str(database),
    )

    assert code == 0
    assert err == ""
    engine = create_database_engine(database)
    try:
        assert ReferenceRepository(engine).list() == []
    finally:
        engine.dispose()


def test_load_reference_inputs_probes_video_and_audio(
    tmp_path: Path,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(mp4())
    audio = tmp_path / "song.wav"
    audio.write_bytes(wav())

    video_spec, audio_spec = load_reference_inputs(
        [video, audio],
        ffprobe_runner=lambda argv: (
            runner_for("video")(argv)
            if str(video) in argv
            else runner_for("audio")(argv)
        ),
    )

    assert video_spec.kind is ReferenceKind.VIDEO
    assert video_spec.mime_type == "video/mp4"
    assert audio_spec.kind is ReferenceKind.AUDIO
    assert audio_spec.mime_type == "audio/wav"


def test_load_reference_inputs_rejects_a_cross_family_probe(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(wav())

    with pytest.raises(ValidationError) as raised:
        load_reference_inputs([video], ffprobe_runner=runner_for("audio"))

    assert raised.value.details["expected_kind"] == "video"
    assert raised.value.details["detected_kind"] == "audio"


def test_api_accepts_repeatable_references_and_records_them(tmp_path: Path) -> None:
    image = tmp_path / "look.png"
    image.write_bytes(png())
    notes = tmp_path / "notes.txt"
    notes.write_text("A calm opening with short cuts.\n", encoding="utf-8")
    engine = create_database_engine(path=tmp_path / "aivia.db")
    upgrade_database(engine=engine)
    client = TestClient(create_app(ApiConfig(require_auth=False), engine, environ={}))
    try:
        response = client.post(
            "/api/v1/runs",
            json={"kind": "run", "references": [str(image), str(notes)]},
        )

        assert response.status_code == 202
        body = response.json()
        assert len(body["job"]["payload"]["references"]) == 2
        assert len(body["job"]["payload"]["reference_ids"]) == 2
        rows = ReferenceRepository(engine).list()
        assert {row["kind"] for row in rows} == {"image", "text"}

        rejected = client.post(
            "/api/v1/runs",
            json={"kind": "run", "references": [str(image), str(notes)]},
        )
        assert rejected.status_code == 202
        assert len(ReferenceRepository(engine).list()) == 2
    finally:
        engine.dispose()


def test_api_rejects_a_reference_that_does_not_match_its_extension(
    tmp_path: Path,
) -> None:
    misleading = tmp_path / "clip.mp4"
    misleading.write_bytes(png())
    engine = create_database_engine(path=tmp_path / "aivia.db")
    upgrade_database(engine=engine)
    client = TestClient(create_app(ApiConfig(require_auth=False), engine, environ={}))
    try:
        response = client.post(
            "/api/v1/runs",
            json={"kind": "run", "references": [str(misleading)]},
        )

        assert response.status_code == 422
        detail = response.json()["detail"]
        assert detail["expected_kind"] == "video"
        assert detail["detected_kind"] == "image"
        assert ReferenceRepository(engine).list() == []
        assert RunRepository(engine).list() == []
    finally:
        engine.dispose()
