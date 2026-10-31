"""``aivia media generate``: one prompt or file in, one artifact plus cost out."""

from __future__ import annotations

import os
import wave
from pathlib import Path

import pytest

from aivia.cli import main
from aivia.core.db import CostEventRepository, create_database_engine


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text.lstrip(), encoding="utf-8")
    return path


@pytest.fixture
def config(tmp_path: Path) -> Path:
    return write_config(
        tmp_path,
        """
providers:
  mock:
    type: mock
storage:
  backend: local
  local_dir: media
""",
    )


@pytest.fixture
def fake_ffprobe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Put a deterministic ffprobe on PATH for the WAV outputs of the mocks."""
    directory = tmp_path / "bin"
    directory.mkdir()
    script = directory / "ffprobe"
    script.write_text(
        """#!/bin/sh
cat <<'JSON'
{"streams":[{"index":0,"codec_type":"audio","codec_name":"pcm_s16le","duration":"1.000","sample_rate":"44100","channels":1}],"format":{"format_name":"wav","format_long_name":"WAV","duration":"1.000","size":"88200"}}
JSON
""",
        encoding="utf-8",
    )
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{directory}{os.pathsep}{os.environ['PATH']}")
    return script


def run_cli(
    capsys: pytest.CaptureFixture[str],
    *args: str,
) -> tuple[int, str, str]:
    code = main(list(args))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def write_wav(path: Path, *, duration: float = 1.0) -> Path:
    sample_rate = 8000
    frames = int(sample_rate * duration)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(b"\x00\x00" * frames)
    return path


def output_path(output: str) -> Path:
    line = next(line for line in output.splitlines() if line.startswith("Output: "))
    return Path(line.removeprefix("Output: ").strip())


@pytest.mark.parametrize(
    ("kind", "capability"),
    [
        ("image", "image_gen"),
        ("video", "video_gen"),
        ("speech", "tts"),
        ("transcription", "stt"),
        ("music", "music_gen"),
        ("sfx", "sfx"),
    ],
)
def test_estimate_prices_every_kind_without_a_paid_call(
    config: Path,
    capsys: pytest.CaptureFixture[str],
    kind: str,
    capability: str,
) -> None:
    code, out, err = run_cli(
        capsys,
        "media",
        "generate",
        kind,
        "--config",
        str(config),
        "--estimate",
    )

    assert code == 0
    assert err == ""
    assert f"Format: {kind}" in out
    assert capability in out
    assert "TOTAL" in out
    assert "$" in out
    assert not (config.parent / "data").exists()


def test_estimate_honours_an_explicit_provider(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = write_config(
        tmp_path,
        """
providers:
  first:
    type: mock-image
  second:
    type: mock-image
""",
    )

    code, out, err = run_cli(
        capsys,
        "media",
        "generate",
        "image",
        "--prompt",
        "a lighthouse",
        "--provider",
        "second",
        "--config",
        str(config),
        "--estimate",
    )

    assert code == 0
    assert err == ""
    assert "second" in out
    assert "first" not in out


def test_prompt_file_is_read_and_the_image_is_stored(
    config: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("A lighthouse at sunset\n", encoding="utf-8")
    destination = tmp_path / "out.png"

    code, out, err = run_cli(
        capsys,
        "media",
        "generate",
        "image",
        "--prompt-file",
        str(prompt),
        "--output",
        str(destination),
        "--config",
        str(config),
    )

    assert code == 0
    assert err == ""
    assert "Provider: mock" in out
    assert "Cost: $" in out
    assert destination.is_file()
    assert destination.read_bytes().startswith(b"\x89PNG")


def test_a_missing_prompt_is_reported_without_running(
    config: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, _out, err = run_cli(
        capsys,
        "media",
        "generate",
        "image",
        "--config",
        str(config),
    )

    assert code == 1
    assert "prompt is required" in err


def test_an_unknown_provider_is_rejected(
    config: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, _out, err = run_cli(
        capsys,
        "media",
        "generate",
        "image",
        "--prompt",
        "a lighthouse",
        "--provider",
        "missing",
        "--config",
        str(config),
        "--estimate",
    )

    assert code == 1
    assert "unknown provider" in err
    assert "missing" in err


def test_no_configured_provider_is_reported(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = write_config(tmp_path, "schema_version: 1\n")

    code, _out, err = run_cli(
        capsys,
        "media",
        "generate",
        "image",
        "--prompt",
        "a lighthouse",
        "--config",
        str(config),
    )

    assert code == 1
    assert "no providers are configured" in err


def test_dry_run_uses_the_mock_provider(
    config: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    destination = tmp_path / "dry.png"

    code, out, err = run_cli(
        capsys,
        "media",
        "generate",
        "image",
        "--prompt",
        "a lighthouse",
        "--dry-run",
        "--output",
        str(destination),
        "--config",
        str(config),
    )

    assert code == 0
    assert err == ""
    assert "Provider: mock" in out
    assert destination.is_file()


@pytest.mark.parametrize(
    ("kind", "extra"),
    [
        ("speech", ("--voice", "narrator", "--format", "wav")),
        ("music", ("--instrumental", "--duration", "1", "--format", "wav")),
        ("sfx", ("--duration", "1", "--loop", "--format", "wav")),
    ],
)
def test_audio_kinds_write_an_audio_file_and_record_their_cost(
    config: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    fake_ffprobe: Path,
    kind: str,
    extra: tuple[str, ...],
) -> None:
    destination = tmp_path / f"{kind}.wav"

    code, out, err = run_cli(
        capsys,
        "media",
        "generate",
        kind,
        "--prompt",
        "A calm test tone",
        *extra,
        "--output",
        str(destination),
        "--config",
        str(config),
    )

    assert code == 0
    assert err == ""
    assert destination.is_file()
    assert destination.read_bytes().startswith(b"RIFF")
    assert "Cost: $" in out
    assert "Cached: no" in out

    engine = create_database_engine(config.parent / "data" / "aivia.db")
    try:
        events = CostEventRepository(engine).list()
        assert len(events) == 1
        operation = {"speech": "tts", "music": "music_gen", "sfx": "sfx"}[kind]
        assert events[0].operation == operation
        assert events[0].amount_usd > 0
    finally:
        engine.dispose()


def test_transcription_reads_a_file_and_prints_the_transcript(
    config: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    audio = write_wav(tmp_path / "speech.wav")
    destination = tmp_path / "transcript.txt"

    code, out, err = run_cli(
        capsys,
        "media",
        "generate",
        "transcription",
        str(audio),
        "--prompt",
        "hello world",
        "--output",
        str(destination),
        "--config",
        str(config),
    )

    assert code == 0
    assert err == ""
    assert destination.read_text(encoding="utf-8") == "hello world"
    assert "Provider: mock" in out
    assert "Cost: $" in out


def test_transcription_without_an_audio_file_is_reported(
    config: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, _out, err = run_cli(
        capsys,
        "media",
        "generate",
        "transcription",
        "--config",
        str(config),
    )

    assert code == 1
    assert "audio or video file is required" in err
