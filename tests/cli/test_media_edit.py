"""``aivia media edit``: instruction, background removal, upscale and outpaint."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

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
def rembg_config(tmp_path: Path) -> Path:
    return write_config(
        tmp_path,
        """
providers:
  cutout:
    type: rembg
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


def write_png(
    path: Path,
    *,
    size: tuple[int, int] = (40, 30),
    color: tuple[int, int, int] = (200, 40, 40),
) -> Path:
    Image.new("RGB", size, color).save(path)
    return path


def write_flat_png(path: Path, *, size: tuple[int, int] = (32, 24)) -> Path:
    image = Image.new("RGB", size, (255, 255, 255))
    for x in range(8, 24):
        for y in range(6, 18):
            image.putpixel((x, y), (200, 20, 20))
    image.save(path)
    return path


def output_path(output: str) -> Path:
    line = next(line for line in output.splitlines() if line.startswith("Output: "))
    return Path(line.removeprefix("Output: ").strip())


def test_instruction_edit_writes_an_image_and_records_its_cost(
    config: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = write_png(tmp_path / "source.png")
    destination = tmp_path / "edited.png"

    code, out, err = run_cli(
        capsys,
        "media",
        "edit",
        "instruction",
        str(source),
        "--prompt",
        "make the subject wear a blue hat",
        "--output",
        str(destination),
        "--config",
        str(config),
    )

    assert code == 0
    assert err == ""
    assert "Provider: mock" in out
    assert "Cost: $" in out
    assert "Cached: no" in out
    assert destination.read_bytes().startswith(b"\x89PNG")

    engine = create_database_engine(config.parent / "data" / "aivia.db")
    try:
        events = CostEventRepository(engine).list()
        assert len(events) == 1
        assert events[0].operation == "image_edit"
        assert events[0].amount_usd > 0
    finally:
        engine.dispose()


def test_background_removal_uses_the_local_provider_and_writes_alpha_png(
    rembg_config: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = write_flat_png(tmp_path / "flat.png")
    destination = tmp_path / "cutout.png"

    code, out, err = run_cli(
        capsys,
        "media",
        "edit",
        "remove-background",
        str(source),
        "--output",
        str(destination),
        "--config",
        str(rembg_config),
    )

    assert code == 0
    assert err == ""
    assert "Provider: cutout" in out
    assert "Cost: $0.000000" in out
    with Image.open(destination) as edited:
        assert edited.mode == "RGBA"
        assert edited.size == (32, 24)
        corner = edited.getpixel((0, 0))
        center = edited.getpixel((16, 12))
        assert isinstance(corner, tuple)
        assert isinstance(center, tuple)
        assert corner[3] == 0
        assert center[3] == 255


def test_upscale_doubles_the_source_dimensions(
    config: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = write_png(tmp_path / "source.png", size=(40, 30))
    destination = tmp_path / "upscaled.png"

    code, _out, err = run_cli(
        capsys,
        "media",
        "edit",
        "upscale",
        str(source),
        "--scale",
        "2",
        "--output",
        str(destination),
        "--config",
        str(config),
    )

    assert code == 0
    assert err == ""
    with Image.open(destination) as edited:
        assert edited.size == (80, 60)


def test_outpaint_uses_the_requested_target_size(
    config: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = write_png(tmp_path / "source.png", size=(40, 30))
    destination = tmp_path / "outpainted.png"

    code, _out, err = run_cli(
        capsys,
        "media",
        "edit",
        "outpaint",
        str(source),
        "--width",
        "80",
        "--height",
        "60",
        "--prompt",
        "extend the scene",
        "--output",
        str(destination),
        "--config",
        str(config),
    )

    assert code == 0
    assert err == ""
    with Image.open(destination) as edited:
        assert edited.size == (80, 60)


def test_the_operation_may_be_selected_with_a_flag(
    config: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = write_png(tmp_path / "source.png", size=(20, 10))
    destination = tmp_path / "upscaled.png"

    code, _out, err = run_cli(
        capsys,
        "media",
        "edit",
        str(source),
        "--operation",
        "upscale",
        "--scale",
        "4",
        "--output",
        str(destination),
        "--config",
        str(config),
    )

    assert code == 0
    assert err == ""
    with Image.open(destination) as edited:
        assert edited.size == (80, 40)


def test_a_second_identical_edit_is_served_from_cache(
    config: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = write_png(tmp_path / "source.png")
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"

    args = (
        "media",
        "edit",
        "instruction",
        str(source),
        "--prompt",
        "add snow",
        "--config",
        str(config),
    )
    code, out, err = run_cli(capsys, *args, "--output", str(first))
    assert code == 0
    assert err == ""
    assert "Cached: no" in out

    code, out, err = run_cli(capsys, *args, "--output", str(second))
    assert code == 0
    assert err == ""
    assert "Cached: yes" in out
    assert first.read_bytes() == second.read_bytes()


def test_edit_estimate_prices_image_edit_without_a_paid_call(
    config: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, out, err = run_cli(
        capsys,
        "media",
        "edit",
        "--estimate",
        "--config",
        str(config),
    )

    assert code == 0
    assert err == ""
    assert "Format: edit" in out
    assert "image_edit" in out
    assert "$" in out
    assert not (config.parent / "data").exists()


def test_a_missing_source_image_is_reported(
    config: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, _out, err = run_cli(
        capsys,
        "media",
        "edit",
        "instruction",
        "--prompt",
        "brighten it",
        "--config",
        str(config),
    )

    assert code == 1
    assert "source image is required" in err


def test_an_instruction_edit_without_a_prompt_is_reported(
    config: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = write_png(tmp_path / "source.png")

    code, _out, err = run_cli(
        capsys,
        "media",
        "edit",
        "instruction",
        str(source),
        "--config",
        str(config),
    )

    assert code == 1
    assert "requires a prompt" in err


def test_an_outpaint_without_a_target_size_is_reported(
    config: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = write_png(tmp_path / "source.png")

    code, _out, err = run_cli(
        capsys,
        "media",
        "edit",
        "outpaint",
        str(source),
        "--width",
        "80",
        "--config",
        str(config),
    )

    assert code == 1
    assert "outpaint requires both --width and --height" in err


def test_an_upscale_factor_of_one_is_rejected(
    config: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = write_png(tmp_path / "source.png")

    code, _out, err = run_cli(
        capsys,
        "media",
        "edit",
        "upscale",
        str(source),
        "--scale",
        "1",
        "--config",
        str(config),
    )

    assert code == 1
    assert "scale" in err


def test_background_removal_rejects_a_lossy_output_format(
    rembg_config: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = write_flat_png(tmp_path / "flat.png")

    code, _out, err = run_cli(
        capsys,
        "media",
        "edit",
        "remove-background",
        str(source),
        "--format",
        "jpeg",
        "--config",
        str(rembg_config),
    )

    assert code == 1
    assert "PNG with alpha" in err


def test_background_removal_skips_a_provider_that_cannot_do_it(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = write_config(
        tmp_path,
        """
providers:
  editor:
    type: aggregator
    options:
      api_key: test-key
      models:
        edit: vendor/instruction-edit
  cutout:
    type: rembg
storage:
  backend: local
  local_dir: media
""",
    )
    source = write_flat_png(tmp_path / "flat.png")
    destination = tmp_path / "cutout.png"

    code, out, err = run_cli(
        capsys,
        "media",
        "edit",
        "remove-background",
        str(source),
        "--output",
        str(destination),
        "--config",
        str(config),
    )

    assert code == 0
    assert err == ""
    assert "Provider: cutout" in out
