"""Asset registry: labels, content hashes, read-only originals, duplicate rejection."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from PIL import Image

from aivia.assistant import AssetRegistry, DuplicateLabelError
from aivia.core.models.asset import AssetKind


def _write(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


def _image(path: Path, size: tuple[int, int] = (8, 8)) -> Path:
    image = Image.new("RGB", size, (255, 0, 0))
    image.save(path, format="PNG")
    return path


def test_registry_stores_asset_by_label(tmp_path: Path) -> None:
    registry = AssetRegistry(root=tmp_path / "registry")
    source = _write(tmp_path / "hero.png", b"PNG-IMAGE-BYTES")
    asset = registry.add(source, label="hero")

    assert asset.label == "hero"
    assert registry.get("hero") is asset
    assert "hero" in registry
    assert len(registry) == 1


def test_labels_are_sorted(tmp_path: Path) -> None:
    registry = AssetRegistry(root=tmp_path / "registry")
    registry.add(_write(tmp_path / "c.png", b"c"), label="zeta")
    registry.add(_write(tmp_path / "a.png", b"a"), label="alpha")
    registry.add(_write(tmp_path / "b.png", b"b"), label="beta")

    assert registry.labels == ("alpha", "beta", "zeta")


def test_content_hash_is_recorded(tmp_path: Path) -> None:
    registry = AssetRegistry(root=tmp_path / "registry")
    data = b"deterministic-bytes"
    source = _write(tmp_path / "file.png", data)
    asset = registry.add(source, label="file")

    expected = hashlib.sha256(data).hexdigest()
    assert asset.content_hash == expected
    assert registry.hashes == {"file": expected}
    assert registry.hashes["file"] == expected


def test_default_label_is_filename_stem(tmp_path: Path) -> None:
    registry = AssetRegistry(root=tmp_path / "registry")
    source = _write(tmp_path / "my_hero.png", b"bytes")
    asset = registry.add(source)

    assert asset.label == "my_hero"


def test_empty_label_is_rejected(tmp_path: Path) -> None:
    registry = AssetRegistry(root=tmp_path / "registry")
    source = _write(tmp_path / "file.png", b"bytes")

    with pytest.raises(ValueError, match="label must not be empty"):
        registry.add(source, label="")


def test_duplicate_label_is_rejected(tmp_path: Path) -> None:
    registry = AssetRegistry(root=tmp_path / "registry")
    registry.add(_write(tmp_path / "a.png", b"a"), label="hero")

    with pytest.raises(DuplicateLabelError):
        registry.add(_write(tmp_path / "b.png", b"b"), label="hero")


def test_duplicate_label_error_is_a_value_error(tmp_path: Path) -> None:
    registry = AssetRegistry(root=tmp_path / "registry")
    registry.add(_write(tmp_path / "a.png", b"a"), label="hero")

    with pytest.raises(ValueError):
        registry.add(_write(tmp_path / "b.png", b"b"), label="hero")


def test_original_is_stored_read_only(tmp_path: Path) -> None:
    registry = AssetRegistry(root=tmp_path / "registry")
    source = _write(tmp_path / "hero.png", b"bytes")
    asset = registry.add(source, label="hero")

    stored = Path(asset.path)
    assert stored.exists()
    assert stored.parent == registry.root
    assert stored.stat().st_mode & 0o222 == 0  # no write bits


def test_original_file_is_not_modified_in_place(tmp_path: Path) -> None:
    registry = AssetRegistry(root=tmp_path / "registry")
    source = _write(tmp_path / "hero.png", b"bytes")
    asset = registry.add(source, label="hero")

    assert source.read_bytes() == b"bytes"
    assert source.stat().st_mode & 0o222 != 0  # source stays writable
    assert Path(asset.path).read_bytes() == b"bytes"


def test_asset_kind_is_detected_from_extension(tmp_path: Path) -> None:
    registry = AssetRegistry(root=tmp_path / "registry")
    image = _image(tmp_path / "photo.png")
    asset = registry.add(image, label="photo")

    assert asset.kind is AssetKind.IMAGE


def test_asset_kind_can_be_supplied(tmp_path: Path) -> None:
    registry = AssetRegistry(root=tmp_path / "registry")
    source = _write(tmp_path / "track.dat", b"audio")
    asset = registry.add(source, label="track", kind=AssetKind.AUDIO)

    assert asset.kind is AssetKind.AUDIO


def test_unknown_file_missing_raises(tmp_path: Path) -> None:
    registry = AssetRegistry(root=tmp_path / "registry")

    with pytest.raises(FileNotFoundError):
        registry.add(tmp_path / "does-not-exist.png", label="missing")


def test_same_content_stored_once(tmp_path: Path) -> None:
    registry = AssetRegistry(root=tmp_path / "registry")
    data = b"shared-content"
    first = _write(tmp_path / "first.png", data)
    second = _write(tmp_path / "second.png", data)

    asset_a = registry.add(first, label="one")
    asset_b = registry.add(second, label="two")

    assert asset_a.content_hash == asset_b.content_hash
    assert Path(asset_a.path) == Path(asset_b.path)
    assert len(list(registry.root.iterdir())) == 1


def test_get_returns_none_for_unknown_label(tmp_path: Path) -> None:
    registry = AssetRegistry(root=tmp_path / "registry")

    assert registry.get("nothing") is None


def test_registry_iterates_labels(tmp_path: Path) -> None:
    registry = AssetRegistry(root=tmp_path / "registry")
    registry.add(_write(tmp_path / "a.png", b"a"), label="zeta")
    registry.add(_write(tmp_path / "b.png", b"b"), label="alpha")

    assert list(registry) == ["alpha", "zeta"]
