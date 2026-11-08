"""``aivia build poses``: generate, check and store a standard pose set."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from aivia.characters import Pose, PoseLibrary
from aivia.cli import main
from aivia.core.db import (
    CostEventRepository,
    create_database_engine,
    upgrade_database,
)
from aivia.core.models.cost import CostEvent


def write_png(path: Path, size: tuple[int, int] = (16, 16), color: str = "red") -> Path:
    Image.new("RGB", size, color).save(path, format="PNG")
    return path


def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "aivia" / "data"


def database_path(tmp_path: Path) -> Path:
    return data_dir(tmp_path) / "aivia.db"


def recorded_cost(tmp_path: Path) -> list[CostEvent]:
    engine = create_database_engine(path=database_path(tmp_path))
    try:
        upgrade_database(engine=engine)
        return CostEventRepository(engine).list(provider="rembg")
    finally:
        engine.dispose()


def run_cli(capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, str, str]:
    code = main(list(args))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_build_poses_generates_checks_and_stores_each_pose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    idle = tmp_path / "idle.png"
    happy = tmp_path / "happy.png"
    write_png(idle)
    write_png(happy, color="blue")

    code, out, err = run_cli(
        capsys,
        "build",
        "poses",
        f"idle={idle}",
        f"happy={happy}",
    )

    assert code == 0, err
    assert err == ""

    poses_dir = data_dir(tmp_path) / "characters" / "poses"
    assert (poses_dir / "idle.yaml").is_file()
    assert (poses_dir / "idle.png").is_file()
    assert (poses_dir / "happy.yaml").is_file()
    assert (poses_dir / "happy.png").is_file()

    library = PoseLibrary(data_dir=data_dir(tmp_path))
    idle_pose = library.load("idle")
    assert isinstance(idle_pose, Pose)
    assert idle_pose.name == "idle"
    assert idle_pose.image == "idle.png"
    assert idle_pose.alpha is True
    assert idle_pose.edge_cleanup == "rembg"
    happy_pose = library.load("happy")
    assert happy_pose.name == "happy"
    assert happy_pose.alpha is True

    events = recorded_cost(tmp_path)
    assert len(events) == 2
    for event in events:
        assert event.provider == "rembg"
        assert event.model == "rembg"
        assert event.operation == "remove_background"
        assert event.units == 1.0
        assert event.unit_kind == "image"
        assert event.amount_usd == 0.0
        assert event.estimated is False

    assert "Built 2 pose(s)" in out


def test_build_poses_scoped_to_a_character(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    image = tmp_path / "idle.png"
    write_png(image)

    code, _, err = run_cli(
        capsys, "build", "poses", "--character", "mascot", f"idle={image}"
    )
    assert code == 0, err

    poses_dir = data_dir(tmp_path) / "characters" / "poses" / "mascot"
    assert (poses_dir / "idle.yaml").is_file()
    assert (poses_dir / "idle.png").is_file()

    library = PoseLibrary(data_dir=data_dir(tmp_path), character="mascot")
    assert library.load("idle").name == "idle"


def test_build_poses_refuses_to_overwrite_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    image = tmp_path / "idle.png"
    write_png(image)

    code, _, _ = run_cli(capsys, "build", "poses", f"idle={image}")
    assert code == 0

    code, _, err = run_cli(capsys, "build", "poses", f"idle={image}")
    assert code == 1
    assert "already exists" in err or "exists" in err

    library = PoseLibrary(data_dir=data_dir(tmp_path))
    assert library.load("idle").image == "idle.png"
    assert len(recorded_cost(tmp_path)) == 1


def test_build_poses_force_overwrites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    first = tmp_path / "idle.png"
    second = tmp_path / "idle.png"
    write_png(first, color="red")
    write_png(second, color="blue")

    assert run_cli(capsys, "build", "poses", f"idle={first}")[0] == 0
    assert run_cli(capsys, "build", "poses", "--force", f"idle={second}")[0] == 0

    library = PoseLibrary(data_dir=data_dir(tmp_path))
    assert library.load("idle").image == "idle.png"
    assert len(recorded_cost(tmp_path)) == 2


def test_build_poses_dry_run_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    image = tmp_path / "idle.png"
    write_png(image)

    code, out, _ = run_cli(capsys, "build", "poses", "--dry-run", f"idle={image}")
    assert code == 0
    assert "Would build the following poses:" in out
    assert f"idle <- {image}" in out

    poses_dir = data_dir(tmp_path) / "characters" / "poses"
    assert not poses_dir.exists()
    assert recorded_cost(tmp_path) == []


def test_build_poses_rejects_a_missing_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    missing = tmp_path / "does-not-exist.png"

    code, _, err = run_cli(capsys, "build", "poses", f"idle={missing}")
    assert code == 1
    assert "not found" in err or "cannot read" in err

    poses_dir = data_dir(tmp_path) / "characters" / "poses"
    assert not poses_dir.exists()
    assert recorded_cost(tmp_path) == []
