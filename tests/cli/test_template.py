"""Template extract, run and CSV batch with per-row cost and isolated bad rows."""

from __future__ import annotations

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


def run_cli(capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, str, str]:
    code = main(list(args))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


class FakeAsset:
    path = ""


class FakeOutcome:
    provider = "mock"
    model = "mock-image"
    amount_usd = 0.02
    cached = False
    assets = (FakeAsset(),)


def _monkeypatch_generate(
    monkeypatch: pytest.MonkeyPatch, boom: str | None = None
) -> None:
    async def fake_generate(runtime: Any, args: Any, kind: str) -> FakeOutcome:
        if boom is not None and boom in getattr(args, "prompt_option", ""):
            raise RuntimeError(f"boom: {boom}")
        return FakeOutcome()

    monkeypatch.setattr(media_module, "_generate", fake_generate)


def test_template_extract_suggests_variables_and_asset_slots(
    config: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    session = tmp_path / "session.yaml"
    session.write_text(
        """
name: my session
kind: image
assets:
  - label: background
    path: assets/bg.png
    kind: image
  - label: logo
    path: assets/logo.png
    kind: image
current:
  layers:
    - id: title
      type: text
      text: "Hello {{ name }} in {{ place }}"
""".lstrip(),
        encoding="utf-8",
    )
    template = tmp_path / "card.yaml"

    code, out, err = run_cli(
        capsys,
        "template",
        "extract",
        str(session),
        "--output",
        str(template),
        "--config",
        str(config),
    )

    assert code == 0
    assert err == ""
    assert "Suggested variables:" in out
    assert "name" in out
    assert "place" in out
    assert "Asset slots:" in out
    assert "background" in out
    assert "logo" in out
    assert template.is_file()

    import yaml

    payload = yaml.safe_load(template.read_text(encoding="utf-8"))
    assert payload["kind"] == "image"
    assert {variable["name"] for variable in payload["variables"]} == {"name", "place"}
    assert {slot["asset_label"] for slot in payload["asset_slots"]} == {
        "background",
        "logo",
    }


def test_template_run_reports_cost(
    config: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    template = tmp_path / "card.yaml"
    template.write_text(
        """
name: card
kind: image
prompt: "A {{ subject }} {{ style }}"
variables:
  - name: subject
    default: lighthouse
  - name: style
    default: oil painting
options:
  width: 32
  height: 24
""".lstrip(),
        encoding="utf-8",
    )
    output = tmp_path / "out.png"

    code, out, err = run_cli(
        capsys,
        "template",
        "run",
        str(template),
        "--var",
        "subject=castle",
        "--output",
        str(output),
        "--config",
        str(config),
    )

    assert code == 0
    assert err == ""
    assert "Provider: mock" in out
    assert "Cost: $0.020000" in out
    assert output.is_file()
    with Image.open(output) as image:
        assert image.size == (32, 24)

    engine = create_database_engine(config.parent / "data" / "aivia.db")
    try:
        assert len(CostEventRepository(engine).list()) == 1
    finally:
        engine.dispose()


def test_template_batch_per_row_cost_and_isolated_bad_rows(
    config: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _monkeypatch_generate(monkeypatch, boom="boom")
    template = tmp_path / "card.yaml"
    template.write_text(
        """
name: card
kind: image
prompt: "A {{ subject }} {{ style }}"
variables:
  - name: subject
    default: lighthouse
  - name: style
    default: oil painting
options:
  width: 16
  height: 12
""".lstrip(),
        encoding="utf-8",
    )
    source = tmp_path / "source.png"
    Image.new("RGB", (16, 12)).save(source)
    FakeAsset.path = source.as_uri()
    FakeOutcome.assets = (FakeAsset(),)
    batch = tmp_path / "items.csv"
    batch.write_text(
        "id,subject,style\n"
        "one,A lighthouse,oil painting\n"
        "two,A castle,watercolor\n"
        "three,A boom,sketch\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    code, out, err = run_cli(
        capsys,
        "template",
        "batch",
        str(batch),
        "--template",
        str(template),
        "--output-dir",
        str(output_dir),
        "--config",
        str(config),
    )

    assert code == 1
    assert "Succeeded: 2" in out
    assert "Total cost: $0.040000" in out
    assert "Error: boom" in err
    assert (output_dir / "one.png").is_file()
    assert (output_dir / "two.png").is_file()
    assert not (output_dir / "three.png").is_file()
