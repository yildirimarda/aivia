"""Pose library: named cutouts with alpha, anchors, tags and edge cleanup.

A pose cutout is a named image that carries an alpha flag, named anchor points
for compositing and tags for lookup. Edge cleanup removes the background so the
cutout sits cleanly on any canvas. Poses are stored as YAML next to their cutout
images in the user data directory.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import cast

import pytest
from PIL import Image, ImageDraw
from pydantic import ValidationError as PydanticValidationError

from aivia.characters import (
    POSE_LIBRARY_SCHEMA_VERSION,
    Pose,
    PoseError,
    PoseLibrary,
    PoseNotFoundError,
    PoseSchemaError,
    cleanup_edges,
    load_pose_library,
)
from aivia.core.errors import ValidationError


def write_png(
    path: Path,
    size: tuple[int, int] = (32, 32),
    color: str = "red",
) -> Path:
    Image.new("RGB", size, color).save(path, format="PNG")
    return path


def _cutout_image(path: Path) -> Path:
    """A red foreground on a white background, suitable for edge cleanup."""
    image = Image.new("RGB", (64, 48), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((16, 12, 48, 36), fill=(220, 20, 20))
    image.save(path, format="PNG")
    return path


def _load_rgba(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as image:
        return image.convert("RGBA").copy()


def test_pose_schema_version_is_current() -> None:
    assert POSE_LIBRARY_SCHEMA_VERSION == 1


def test_pose_model_parses_every_field() -> None:
    pose = Pose.model_validate(
        {
            "name": "idle",
            "image": "idle.png",
            "alpha": True,
            "anchors": {"hand": [0.5, 0.8], "foot": [0.5, 0.1]},
            "tags": ["standing", "neutral"],
            "edge_cleanup": "rembg",
        }
    )

    assert pose.name == "idle"
    assert pose.image == "idle.png"
    assert pose.alpha is True
    assert pose.anchors == {"hand": (0.5, 0.8), "foot": (0.5, 0.1)}
    assert pose.tags == ["standing", "neutral"]
    assert pose.edge_cleanup == "rembg"


def test_pose_model_defaults_are_empty() -> None:
    pose = Pose.model_validate({"name": "idle", "image": "idle.png"})

    assert pose.name == "idle"
    assert pose.image == "idle.png"
    assert pose.alpha is False
    assert pose.anchors == {}
    assert pose.tags == []
    assert pose.edge_cleanup == ""


def test_pose_round_trips_through_json() -> None:
    pose = Pose.model_validate(
        {
            "name": "wave",
            "image": "wave.png",
            "alpha": True,
            "anchors": {"hand": [0.9, 0.5]},
            "tags": ["waving"],
            "edge_cleanup": "rembg",
        }
    )
    restored = Pose.model_validate(pose.model_dump(mode="json"))

    assert restored == pose


def test_pose_rejects_blank_name_and_image() -> None:
    with pytest.raises(PydanticValidationError):
        Pose.model_validate({"name": "   ", "image": "idle.png"})
    with pytest.raises(PydanticValidationError):
        Pose.model_validate({"name": "idle", "image": "   "})


def test_pose_rejects_out_of_range_and_malformed_anchors() -> None:
    with pytest.raises(PydanticValidationError) as caught:
        Pose.model_validate(
            {"name": "idle", "image": "idle.png", "anchors": {"hand": [1.5, 0.5]}}
        )
    assert "within 0-1" in str(caught.value)

    with pytest.raises(PydanticValidationError) as caught:
        Pose.model_validate(
            {"name": "idle", "image": "idle.png", "anchors": {"hand": [0.5]}}
        )
    assert "(x, y) pair" in str(caught.value)


def test_pose_rejects_blank_tags() -> None:
    with pytest.raises(PydanticValidationError) as caught:
        Pose.model_validate(
            {"name": "idle", "image": "idle.png", "tags": ["standing", "  "]}
        )
    assert "blank entries" in str(caught.value)


def test_pose_rejects_unsupported_edge_cleanup() -> None:
    with pytest.raises(PydanticValidationError) as caught:
        Pose.model_validate(
            {"name": "idle", "image": "idle.png", "edge_cleanup": "photoshop"}
        )
    assert "unsupported edge cleanup method" in str(caught.value)


def test_pose_library_store_and_load(tmp_path: Path) -> None:
    library = PoseLibrary(data_dir=tmp_path / "data", character="mascot")
    image = tmp_path / "idle.png"
    write_png(image)
    image_bytes = image.read_bytes()

    pose = library.store(
        Pose(name="idle", image="idle.png", tags=["standing", "neutral"]),
        image_bytes=image_bytes,
    )

    assert pose.name == "idle"
    assert pose.image == "idle.png"
    assert pose.tags == ["standing", "neutral"]
    assert library.exists("idle")
    assert library.names() == ("idle",)

    loaded = library.load("idle")
    assert loaded.name == "idle"
    assert loaded.image == "idle.png"
    assert loaded.tags == ["standing", "neutral"]
    assert (library.poses_dir / "idle.png").is_file()
    assert (library.poses_dir / "idle.yaml").is_file()


def test_pose_library_load_is_a_pure_file_read(tmp_path: Path) -> None:
    library = PoseLibrary(data_dir=tmp_path / "data", character="mascot")
    image = tmp_path / "idle.png"
    write_png(image)
    library.store(Pose(name="idle", image="idle.png"), image_bytes=image.read_bytes())

    # Reloading through a fresh library is a pure file read.
    other = PoseLibrary(data_dir=tmp_path / "data", character="mascot")
    loaded = other.load("idle")
    assert loaded.image == "idle.png"


def test_pose_library_unknown_pose_names_the_file(tmp_path: Path) -> None:
    library = PoseLibrary(data_dir=tmp_path / "data", character="mascot")

    with pytest.raises(PoseNotFoundError) as caught:
        library.load("missing")

    assert caught.value.code == "pose_not_found"
    assert caught.value.details["name"] == "missing"
    assert "poses" in caught.value.details["file"]
    assert "unknown pose: missing" in str(caught.value)


def test_pose_library_missing_asset_names_file_and_field(tmp_path: Path) -> None:
    library = PoseLibrary(data_dir=tmp_path / "data", character="mascot")
    path = library.poses_dir / "idle.yaml"
    library.poses_dir.mkdir(parents=True, exist_ok=True)
    path.write_text("name: idle\nimage: missing.png\n", encoding="utf-8")

    with pytest.raises(PoseSchemaError) as caught:
        library.load("idle")

    assert caught.value.code == "pose_schema"
    assert caught.value.details["field"] == "image"
    assert caught.value.details["asset"] == "missing.png"
    assert "missing.png" in str(caught.value)


def test_pose_library_duplicate_pose_is_rejected(tmp_path: Path) -> None:
    library = PoseLibrary(data_dir=tmp_path / "data", character="mascot")
    image_bytes = write_png(tmp_path / "idle.png").read_bytes()

    library.store(Pose(name="idle", image="idle.png"), image_bytes=image_bytes)

    with pytest.raises(PoseError) as caught:
        library.store(Pose(name="idle", image="idle.png"), image_bytes=image_bytes)
    assert "already exists" in str(caught.value)


def test_pose_library_find_by_tag(tmp_path: Path) -> None:
    library = PoseLibrary(data_dir=tmp_path / "data", character="mascot")
    image_bytes = write_png(tmp_path / "idle.png").read_bytes()

    library.store(
        Pose(name="idle", image="idle.png", tags=["standing", "neutral"]),
        image_bytes=image_bytes,
    )
    library.store(
        Pose(name="wave", image="wave.png", tags=["waving"]),
        image_bytes=image_bytes,
    )

    standing = library.find_by_tag("standing")
    assert [pose.name for pose in standing] == ["idle"]
    waving = library.find_by_tag("waving")
    assert [pose.name for pose in waving] == ["wave"]
    assert library.find_by_tag("dancing") == ()


def test_pose_library_default_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    library = load_pose_library(character="mascot")

    assert library.data_dir == tmp_path / "aivia" / "data"
    assert library.poses_dir == (
        tmp_path / "aivia" / "data" / "characters" / "poses" / "mascot"
    )


def test_cleanup_edges_removes_background(tmp_path: Path) -> None:
    path = tmp_path / "cutout.png"
    _cutout_image(path)
    data = path.read_bytes()

    result = cleanup_edges(data)
    image = _load_rgba(result)

    assert image.mode == "RGBA"
    width, height = image.size
    # Background corners are transparent.
    assert cast("tuple[int, ...]", image.getpixel((0, 0)))[3] == 0
    assert cast("tuple[int, ...]", image.getpixel((width - 1, height - 1)))[3] == 0
    # Foreground centre is opaque.
    assert cast("tuple[int, ...]", image.getpixel((32, 24)))[3] == 255


def test_store_with_edge_cleanup_writes_alpha_png(tmp_path: Path) -> None:
    library = PoseLibrary(data_dir=tmp_path / "data", character="mascot")
    path = tmp_path / "cutout.png"
    _cutout_image(path)
    image_bytes = path.read_bytes()

    pose = library.store(
        Pose(name="idle", image="idle.png"),
        image_bytes=image_bytes,
        edge_cleanup=True,
    )

    assert pose.alpha is True
    assert pose.edge_cleanup == "rembg"
    stored = (library.poses_dir / "idle.png").read_bytes()
    image = _load_rgba(stored)
    assert image.mode == "RGBA"
    assert cast("tuple[int, ...]", image.getpixel((0, 0)))[3] == 0


def test_store_rejects_an_escaping_image_name(tmp_path: Path) -> None:
    library = PoseLibrary(data_dir=tmp_path / "data", character="mascot")

    with pytest.raises(ValidationError):
        library.store(
            Pose(name="idle", image="../secret.png"),
            image_bytes=b"data",
        )


def test_library_loads_via_load_pose_library(tmp_path: Path) -> None:
    library = load_pose_library(data_dir=tmp_path / "data", character="mascot")
    assert isinstance(library, PoseLibrary)
    assert library.character == "mascot"
