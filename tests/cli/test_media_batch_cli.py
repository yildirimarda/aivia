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
