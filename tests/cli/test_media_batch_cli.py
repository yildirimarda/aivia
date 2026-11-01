"""``aivia media batch``: CSV or YAML rows run through the resumable runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

import aivia.cli.media as media_module
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


def run_cli(
    capsys: pytest.CaptureFixture[str],
    *args: str,
) -> tuple[int, str, str]:
    code = main(list(args))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_csv_batch_maps_row_flags_and_writes_each_output(
    config: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    manifest = tmp_path / "manifest.json"
    batch = tmp_path / "items.csv"
    batch.write_text(
        "id,prompt,width,height,seed,output\n"
        f"one,A lighthouse,32,24,7,{first}\n"
        f"two,A castle,16,12,9,{second}\n",
        encoding="utf-8",
    )

    code, out, err = run_cli(
        capsys,
        "media",
        "batch",
        str(batch),
        "--kind",
        "image",
        "--manifest",
        str(manifest),
        "--config",
        str(config),
    )

    assert code == 0
    assert err == ""
    assert "Succeeded: 2" in out
    assert "Total cost: $" in out
    assert first.is_file()
    assert second.is_file()
    with Image.open(first) as image:
        assert image.size == (32, 24)
    with Image.open(second) as image:
        assert image.size == (16, 12)

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert set(payload["items"]) == {"one", "two"}
    assert payload["items"]["one"]["value"]["outputs"] == [str(first)]

    engine = create_database_engine(config.parent / "data" / "aivia.db")
    try:
        assert len(CostEventRepository(engine).list()) == 2
    finally:
        engine.dispose()


def test_yaml_batch_uses_defaults_and_per_item_overrides(
    config: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "out"
    batch = tmp_path / "items.yaml"
    batch.write_text(
        """
defaults:
  kind: image
  width: 20
  height: 10
items:
  - id: alpha
    prompt: A lighthouse
  - id: beta
    prompt: A castle
    width: 40
""".lstrip(),
        encoding="utf-8",
    )

    code, out, err = run_cli(
        capsys,
        "media",
        "batch",
        str(batch),
        "--output-dir",
        str(output_dir),
        "--config",
        str(config),
    )

    assert code == 0
    assert err == ""
    assert "Succeeded: 2" in out
    with Image.open(output_dir / "alpha.png") as image:
        assert image.size == (20, 10)
    with Image.open(output_dir / "beta.png") as image:
        assert image.size == (40, 10)


def test_completed_items_resume_from_the_manifest(
    config: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class FakeOutcome:
        provider = "fake"
        model = "fake"
        amount_usd = 0.02
        cached = False

    async def fake_generate(runtime: Any, args: Any, kind: str) -> FakeOutcome:
        calls.append(args.prompt)
        return FakeOutcome()

    monkeypatch.setattr(media_module, "_generate", fake_generate)
    manifest = tmp_path / "manifest.json"
    batch = tmp_path / "items.csv"
    batch.write_text(
        "id,prompt\none,A lighthouse\ntwo,A castle\n",
        encoding="utf-8",
    )
    arguments = (
        "media",
        "batch",
        str(batch),
        "--manifest",
        str(manifest),
        "--config",
        str(config),
    )

    first_code, first_out, first_err = run_cli(capsys, *arguments)

    assert first_code == 0
    assert first_err == ""
    assert calls == ["A lighthouse", "A castle"]
    assert "Total cost: $0.040000" in first_out

    second_code, second_out, second_err = run_cli(capsys, *arguments)

    assert second_code == 0
    assert second_err == ""
    assert calls == ["A lighthouse", "A castle"]
    assert second_out.count("resumed") == 2
    assert "Total cost: $0.040000" in second_out

    batch.write_text(
        "id,prompt\none,A lighthouse\ntwo,A castle\nthree,A bridge\n",
        encoding="utf-8",
    )
    third_code, third_out, third_err = run_cli(capsys, *arguments)

    assert third_code == 0
    assert third_err == ""
    assert calls == ["A lighthouse", "A castle", "A bridge"]
    assert third_out.count("resumed") == 2
    assert "Total cost: $0.060000" in third_out


def test_changed_row_content_invalidates_a_resumed_item(
    config: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class FakeOutcome:
        provider = "fake"
        model = "fake"
        amount_usd = 0.02
        cached = False

    async def fake_generate(runtime: Any, args: Any, kind: str) -> FakeOutcome:
        calls.append(args.prompt)
        return FakeOutcome()

    monkeypatch.setattr(media_module, "_generate", fake_generate)
    manifest = tmp_path / "manifest.json"
    batch = tmp_path / "items.csv"
    batch.write_text(
        "id,prompt\none,A lighthouse\ntwo,A castle\n",
        encoding="utf-8",
    )
    arguments = (
        "media",
        "batch",
        str(batch),
        "--manifest",
        str(manifest),
        "--config",
        str(config),
    )

    first_code, first_out, _first_err = run_cli(capsys, *arguments)

    assert first_code == 0
    assert calls == ["A lighthouse", "A castle"]
    assert "Total cost: $0.040000" in first_out
    first_payload = json.loads(manifest.read_text(encoding="utf-8"))
    first_fingerprint = first_payload["items"]["one"]["fingerprint"]
    assert first_fingerprint

    batch.write_text(
        "id,prompt\none,A bright lighthouse\ntwo,A castle\n",
        encoding="utf-8",
    )
    second_code, second_out, second_err = run_cli(capsys, *arguments)

    assert second_code == 0
    assert second_err == ""
    assert calls == ["A lighthouse", "A castle", "A bright lighthouse"]
    assert second_out.count("resumed") == 1
    assert "Total cost: $0.040000" in second_out
    second_payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert second_payload["items"]["one"]["fingerprint"] != first_fingerprint

    third_code, third_out, third_err = run_cli(capsys, *arguments)

    assert third_code == 0
    assert third_err == ""
    assert calls == ["A lighthouse", "A castle", "A bright lighthouse"]
    assert third_out.count("resumed") == 2
    assert "Total cost: $0.040000" in third_out


def test_changed_cli_default_invalidates_a_resumed_item(
    config: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int | None] = []

    class FakeOutcome:
        provider = "fake"
        model = "fake"
        amount_usd = 0.02
        cached = False

    async def fake_generate(runtime: Any, args: Any, kind: str) -> FakeOutcome:
        calls.append(args.seed)
        return FakeOutcome()

    monkeypatch.setattr(media_module, "_generate", fake_generate)
    manifest = tmp_path / "manifest.json"
    batch = tmp_path / "items.csv"
    batch.write_text(
        "id,prompt\none,A lighthouse\ntwo,A castle\n",
        encoding="utf-8",
    )
    arguments = (
        "media",
        "batch",
        str(batch),
        "--manifest",
        str(manifest),
        "--config",
        str(config),
    )

    first_code, _first_out, first_err = run_cli(capsys, *arguments, "--seed", "7")

    assert first_code == 0
    assert first_err == ""
    assert calls == [7, 7]

    second_code, second_out, second_err = run_cli(capsys, *arguments, "--seed", "7")

    assert second_code == 0
    assert second_err == ""
    assert calls == [7, 7]
    assert second_out.count("resumed") == 2

    third_code, third_out, third_err = run_cli(capsys, *arguments, "--seed", "8")

    assert third_code == 0
    assert third_err == ""
    assert calls == [7, 7, 8, 8]
    assert third_out.count("resumed") == 0


def test_changed_output_destination_invalidates_a_resumed_item(
    config: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    source = tmp_path / "artifact.png"
    source.write_bytes(b"image")

    class FakeAsset:
        path = source.as_uri()

    class FakeOutcome:
        provider = "fake"
        model = "fake"
        amount_usd = 0.02
        cached = False
        assets = (FakeAsset(),)

    async def fake_generate(runtime: Any, args: Any, kind: str) -> FakeOutcome:
        calls.append(args.prompt)
        return FakeOutcome()

    monkeypatch.setattr(media_module, "_generate", fake_generate)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    manifest = tmp_path / "manifest.json"
    batch = tmp_path / "items.csv"
    batch.write_text("id,prompt\none,A lighthouse\n", encoding="utf-8")
    arguments = (
        "media",
        "batch",
        str(batch),
        "--manifest",
        str(manifest),
        "--config",
        str(config),
    )

    first_code, _first_out, first_err = run_cli(
        capsys, *arguments, "--output-dir", str(first_dir)
    )

    assert first_code == 0
    assert first_err == ""
    assert calls == ["A lighthouse"]
    assert (first_dir / "one.png").is_file()
    first_payload = json.loads(manifest.read_text(encoding="utf-8"))
    first_fingerprint = first_payload["items"]["one"]["fingerprint"]

    second_code, second_out, second_err = run_cli(
        capsys, *arguments, "--output-dir", str(first_dir)
    )

    assert second_code == 0
    assert second_err == ""
    assert calls == ["A lighthouse"]
    assert second_out.count("resumed") == 1

    third_code, third_out, third_err = run_cli(
        capsys, *arguments, "--output-dir", str(second_dir)
    )

    assert third_code == 0
    assert third_err == ""
    assert calls == ["A lighthouse", "A lighthouse"]
    assert third_out.count("resumed") == 0
    assert (second_dir / "one.png").is_file()
    third_payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert third_payload["items"]["one"]["fingerprint"] != first_fingerprint
    assert third_payload["items"]["one"]["value"]["outputs"] == [
        str(second_dir / "one.png")
    ]


def test_changed_cli_prompt_file_invalidates_a_resumed_item(
    config: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class FakeOutcome:
        provider = "fake"
        model = "fake"
        amount_usd = 0.02
        cached = False

    async def fake_generate(runtime: Any, args: Any, kind: str) -> FakeOutcome:
        calls.append(Path(args.prompt_file).read_text(encoding="utf-8"))
        return FakeOutcome()

    monkeypatch.setattr(media_module, "_generate", fake_generate)
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("A lighthouse", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    batch = tmp_path / "items.csv"
    batch.write_text("id\none\ntwo\n", encoding="utf-8")
    arguments = (
        "media",
        "batch",
        str(batch),
        "--manifest",
        str(manifest),
        "--prompt-file",
        str(prompt_file),
        "--config",
        str(config),
    )

    first_code, _first_out, first_err = run_cli(capsys, *arguments)

    assert first_code == 0
    assert first_err == ""
    assert calls == ["A lighthouse", "A lighthouse"]

    second_code, second_out, second_err = run_cli(capsys, *arguments)

    assert second_code == 0
    assert second_err == ""
    assert calls == ["A lighthouse", "A lighthouse"]
    assert second_out.count("resumed") == 2

    prompt_file.write_text("A bright lighthouse", encoding="utf-8")
    third_code, third_out, third_err = run_cli(capsys, *arguments)

    assert third_code == 0
    assert third_err == ""
    assert calls == [
        "A lighthouse",
        "A lighthouse",
        "A bright lighthouse",
        "A bright lighthouse",
    ]
    assert third_out.count("resumed") == 0


def test_changed_referenced_input_files_invalidate_only_their_rows(
    config: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class FakeOutcome:
        provider = "fake"
        model = "fake"
        amount_usd = 0.02
        cached = False

    async def fake_generate(runtime: Any, args: Any, kind: str) -> FakeOutcome:
        calls.append(kind)
        return FakeOutcome()

    monkeypatch.setattr(media_module, "_generate", fake_generate)
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("A lighthouse", encoding="utf-8")
    lyrics_file = tmp_path / "lyrics.txt"
    lyrics_file.write_text("la la", encoding="utf-8")
    audio_file = tmp_path / "clip.wav"
    audio_file.write_bytes(b"RIFF-audio")
    manifest = tmp_path / "manifest.json"
    batch = tmp_path / "items.csv"
    batch.write_text(
        "id,kind,prompt,prompt_file,lyrics_file,audio_file\n"
        f"image,image,,{prompt_file},,\n"
        f"music,music,A song,,{lyrics_file},\n"
        f"transcript,transcription,,,,{audio_file}\n",
        encoding="utf-8",
    )
    arguments = (
        "media",
        "batch",
        str(batch),
        "--manifest",
        str(manifest),
        "--config",
        str(config),
    )

    first_code, _first_out, first_err = run_cli(capsys, *arguments)

    assert first_code == 0
    assert first_err == ""
    assert calls == ["image", "music", "transcription"]

    second_code, second_out, second_err = run_cli(capsys, *arguments)

    assert second_code == 0
    assert second_err == ""
    assert calls == ["image", "music", "transcription"]
    assert second_out.count("resumed") == 3

    prompt_file.write_text("A bright lighthouse", encoding="utf-8")
    third_code, third_out, third_err = run_cli(capsys, *arguments)

    assert third_code == 0
    assert third_err == ""
    assert calls == ["image", "music", "transcription", "image"]
    assert third_out.count("resumed") == 2

    lyrics_file.write_text("la la la", encoding="utf-8")
    fourth_code, fourth_out, fourth_err = run_cli(capsys, *arguments)

    assert fourth_code == 0
    assert fourth_err == ""
    assert calls == ["image", "music", "transcription", "image", "music"]
    assert fourth_out.count("resumed") == 2

    audio_file.write_bytes(b"RIFF-other-audio")
    fifth_code, fifth_out, fifth_err = run_cli(capsys, *arguments)

    assert fifth_code == 0
    assert fifth_err == ""
    assert calls == [
        "image",
        "music",
        "transcription",
        "image",
        "music",
        "transcription",
    ]
    assert fifth_out.count("resumed") == 2


def test_a_bad_batch_file_is_reported_without_running(
    config: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    unknown = tmp_path / "unknown.csv"
    unknown.write_text("prompt,nope\nA lighthouse,x\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"

    code, _out, err = run_cli(
        capsys,
        "media",
        "batch",
        str(unknown),
        "--manifest",
        str(manifest),
        "--config",
        str(config),
    )

    assert code == 1
    assert "unknown field: nope" in err
    assert not manifest.exists()

    wrong = tmp_path / "items.txt"
    wrong.write_text("prompt\nA lighthouse\n", encoding="utf-8")

    code, _out, err = run_cli(
        capsys,
        "media",
        "batch",
        str(wrong),
        "--config",
        str(config),
    )

    assert code == 1
    assert ".csv, .yaml or .yml" in err
