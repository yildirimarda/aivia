"""Parsing @ labels in instructions, resolving them to assets."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from aivia.assistant import AssetRegistry, UnknownLabelError, resolve_labels


def _image(path: Path, size: tuple[int, int] = (8, 8)) -> Path:
    image = Image.new("RGB", size, (255, 0, 0))
    image.save(path, format="PNG")
    return path


def test_resolves_known_labels(tmp_path: Path) -> None:
    registry = AssetRegistry(root=tmp_path / "registry")
    registry.add(_image(tmp_path / "hero.png"), label="hero")
    registry.add(_image(tmp_path / "side.png"), label="side")

    resolved = resolve_labels("make a scene with @hero and @side", registry)

    assert set(resolved) == {"hero", "side"}
    assert resolved["hero"].label == "hero"
    assert resolved["side"].label == "side"
    assert resolved["hero"] is registry.get("hero")
    assert resolved["side"] is registry.get("side")


def test_no_labels_returns_empty(tmp_path: Path) -> None:
    registry = AssetRegistry(root=tmp_path / "registry")

    assert resolve_labels("just plain text", registry) == {}
    assert resolve_labels("", registry) == {}


def test_repeated_label_resolved_once(tmp_path: Path) -> None:
    registry = AssetRegistry(root=tmp_path / "registry")
    registry.add(_image(tmp_path / "hero.png"), label="hero")

    resolved = resolve_labels("@hero plus @hero again", registry)

    assert resolved == {"hero": registry.get("hero")}


def test_unknown_label_fails_clearly(tmp_path: Path) -> None:
    registry = AssetRegistry(root=tmp_path / "registry")

    with pytest.raises(UnknownLabelError) as exc_info:
        resolve_labels("use @missing here", registry)

    assert isinstance(exc_info.value, ValueError)
    assert "missing" in str(exc_info.value)
    assert "unknown asset label" in str(exc_info.value)


def test_unknown_label_names_the_first_offender(tmp_path: Path) -> None:
    registry = AssetRegistry(root=tmp_path / "registry")
    registry.add(_image(tmp_path / "hero.png"), label="hero")

    with pytest.raises(UnknownLabelError) as exc_info:
        resolve_labels("@hero and @ghost", registry)

    assert "ghost" in str(exc_info.value)
    assert "hero" not in str(exc_info.value)


def test_label_characters_are_flexible(tmp_path: Path) -> None:
    registry = AssetRegistry(root=tmp_path / "registry")
    registry.add(_image(tmp_path / "file.png"), label="my.hero-2")

    resolved = resolve_labels("use @my.hero-2 now", registry)

    assert set(resolved) == {"my.hero-2"}
