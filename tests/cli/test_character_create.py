"""``aivia character create``: validate supplied images and store version one."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from PIL import Image

from aivia.characters import (
    CURRENT_CHARACTER_SCHEMA_VERSION,
    CharacterLibrary,
    load_character_file,
)
from aivia.cli import main


def write_png(path: Path, size: tuple[int, int] = (16, 16), color: str = "red") -> Path:
    Image.new("RGB", size, color).save(path, format="PNG")
    return path


def character_dir(tmp_path: Path) -> Path:
    return tmp_path / "aivia" / "data" / "characters"


def test_character_create_stores_version_one_with_reference_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    face = tmp_path / "face.png"
    turnaround = tmp_path / "turnaround.png"
    write_png(face)
    write_png(turnaround, color="blue")

    assert (
        main(
            [
                "character",
                "create",
                "mascot",
                "--image",
                str(face),
                "--image",
                str(turnaround),
            ]
        )
        == 0
    )

    characters = character_dir(tmp_path)
    character_file = characters / "mascot.yaml"
    assert character_file.is_file()
    assert (characters / "face.png").is_file()
    assert (characters / "turnaround.png").is_file()

    document = yaml.safe_load(character_file.read_text(encoding="utf-8"))
    assert document["name"] == "mascot"
    assert document["version"] == CURRENT_CHARACTER_SCHEMA_VERSION
    assert document["reference_images"] == ["face.png", "turnaround.png"]
    assert document.get("pose_library", {}) == {}
    assert document.get("voice", "") == ""

    output = capsys.readouterr().out
    assert f"Created character 'mascot' at {character_file}" in output
    assert f"(version {CURRENT_CHARACTER_SCHEMA_VERSION})" in output

    # The stored character loads cleanly through the loader.
    library = CharacterLibrary(data_dir=tmp_path / "aivia" / "data")
    character = library.load("mascot")
    assert character.name == "mascot"
    assert character.version == CURRENT_CHARACTER_SCHEMA_VERSION
    assert character.reference_images == ["face.png", "turnaround.png"]


def test_character_create_stores_pose_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    idle = tmp_path / "idle.png"
    wave = tmp_path / "wave.png"
    write_png(idle)
    write_png(wave)

    assert (
        main(
            [
                "character",
                "create",
                "mascot",
                "--pose",
                f"idle={idle}",
                "--pose",
                f"wave={wave}",
            ]
        )
        == 0
    )

    document = yaml.safe_load(
        (character_dir(tmp_path) / "mascot.yaml").read_text(encoding="utf-8")
    )
    assert document["pose_library"] == {"idle": "idle.png", "wave": "wave.png"}
    assert (character_dir(tmp_path) / "idle.png").is_file()
    assert (character_dir(tmp_path) / "wave.png").is_file()


def test_character_create_stores_voice_art_style_brand_kit_and_negatives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    image = tmp_path / "face.png"
    write_png(image)

    assert (
        main(
            [
                "character",
                "create",
                "mascot",
                "--image",
                str(image),
                "--voice",
                "narrator-en",
                "--art-style",
                "flat vector",
                "--brand-kit",
                "default",
                "--negatives",
                "photorealistic",
                "--negatives",
                "blurry",
            ]
        )
        == 0
    )

    document = yaml.safe_load(
        (character_dir(tmp_path) / "mascot.yaml").read_text(encoding="utf-8")
    )
    assert document["voice"] == "narrator-en"
    assert document["art_style"] == "flat vector"
    assert document["brand_kit"] == "default"
    assert document["negatives"] == ["photorealistic", "blurry"]


def test_character_create_rejects_a_non_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    not_an_image = tmp_path / "notes.txt"
    not_an_image.write_text("not an image", encoding="utf-8")

    assert main(["character", "create", "mascot", "--image", str(not_an_image)]) == 1

    assert not (character_dir(tmp_path) / "mascot.yaml").exists()
    captured = capsys.readouterr()
    assert (
        "is not an image" in captured.err or "could not be recognised" in captured.err
    )


def test_character_create_refuses_to_overwrite_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    image = tmp_path / "face.png"
    write_png(image)
    assert main(["character", "create", "mascot", "--image", str(image)]) == 0

    character_file = character_dir(tmp_path) / "mascot.yaml"
    original = character_file.read_text(encoding="utf-8")

    assert main(["character", "create", "mascot", "--image", str(image)]) == 1
    assert character_file.read_text(encoding="utf-8") == original
    captured = capsys.readouterr()
    assert "already exists" in captured.err
    assert "--force" in captured.err


def test_character_create_force_overwrites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    image = tmp_path / "face.png"
    write_png(image)
    assert main(["character", "create", "mascot", "--image", str(image)]) == 0

    other = tmp_path / "other.png"
    write_png(other, color="green")
    assert (
        main(["character", "create", "mascot", "--force", "--image", str(other)]) == 0
    )

    document = yaml.safe_load(
        (character_dir(tmp_path) / "mascot.yaml").read_text(encoding="utf-8")
    )
    assert document["reference_images"] == ["other.png"]


def test_character_create_dry_run_validates_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    image = tmp_path / "face.png"
    write_png(image)

    assert (
        main(["character", "create", "mascot", "--image", str(image), "--dry-run"]) == 0
    )

    assert not character_dir(tmp_path).exists()


def test_character_create_rejects_an_escaping_character_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    assert main(["character", "create", "../secret"]) == 1
    captured = capsys.readouterr()
    assert "characters directory" in captured.err or "escape" in captured.err


def test_character_create_loads_via_load_character_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    image = tmp_path / "face.png"
    write_png(image)
    assert main(["character", "create", "mascot", "--image", str(image)]) == 0

    character = load_character_file(character_dir(tmp_path) / "mascot.yaml")
    assert character.name == "mascot"
    assert character.version == CURRENT_CHARACTER_SCHEMA_VERSION
    assert character.schema_version == 1
    assert character.reference_images == ["face.png"]
