"""Registered assets are analyzed for dimensions, colors, vision
description, detected text and face regions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from aivia.assistant import AssetRegistry


def _image(path: Path, size: tuple[int, int] = (8, 8)) -> Path:
    image = Image.new("RGB", size, (255, 0, 0))
    image.save(path, format="PNG")
    return path


class _FakeAnalyzer:
    """A synchronous-analyzer stand-in that returns a canned payload."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls = 0

    async def analyze(self, source: Any) -> dict[str, Any]:
        self.calls += 1
        return dict(self.payload)


def test_registered_image_is_analyzed_offline(tmp_path: Path) -> None:
    registry = AssetRegistry(root=tmp_path / "registry", analyzer=_FakeAnalyzer({}))
    source = _image(tmp_path / "hero.png", (100, 50))
    asset = registry.add(source, label="hero")

    assert asset.analysis.width == 100
    assert asset.analysis.height == 50
    assert asset.analysis.colors == ["#ff0000"]
    assert asset.analysis.description == ""
    assert asset.analysis.detected_text == ""
    assert asset.analysis.faces == []


def test_registered_image_carries_vision_fields(tmp_path: Path) -> None:
    analyzer = _FakeAnalyzer(
        {
            "description": "A red square beside a blue square.",
            "detected_text": "SALE",
            "faces": [{"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4}],
        }
    )
    registry = AssetRegistry(root=tmp_path / "registry", analyzer=analyzer)
    source = _image(tmp_path / "hero.png", (100, 50))
    asset = registry.add(source, label="hero")

    assert analyzer.calls == 1
    assert asset.analysis.width == 100
    assert asset.analysis.height == 50
    assert asset.analysis.colors == ["#ff0000"]
    assert asset.analysis.description == "A red square beside a blue square."
    assert asset.analysis.detected_text == "SALE"
    assert len(asset.analysis.faces) == 1
    assert asset.analysis.faces[0].box == (10.0, 10.0, 30.0, 20.0)
    assert asset.analysis.has_faces is True
    assert asset.analysis.is_analyzed is True


def test_analysis_records_provider_and_model(tmp_path: Path) -> None:
    class _NamedAnalyzer:
        name = "acme"

        async def analyze(self, source: Any) -> dict[str, Any]:
            return {"description": "seen", "model_id": "vision-model"}

    registry = AssetRegistry(root=tmp_path / "registry", analyzer=_NamedAnalyzer())
    source = _image(tmp_path / "hero.png")
    asset = registry.add(source, label="hero")

    assert asset.analysis.provider == "acme"


def test_non_image_asset_is_not_analyzed(tmp_path: Path) -> None:
    analyzer = _FakeAnalyzer({"description": "should not run"})
    registry = AssetRegistry(root=tmp_path / "registry", analyzer=analyzer)
    source = tmp_path / "note.txt"
    source.write_text("hello")
    asset = registry.add(source, label="note")

    assert analyzer.calls == 0
    assert asset.analysis.width is None
    assert asset.analysis.height is None
    assert asset.analysis.colors == []
    assert asset.analysis.description == ""
    assert asset.analysis.detected_text == ""
    assert asset.analysis.faces == []
    assert asset.analysis.is_analyzed is False


def test_asset_without_analyzer_is_not_analyzed(tmp_path: Path) -> None:
    registry = AssetRegistry(root=tmp_path / "registry")
    source = _image(tmp_path / "hero.png")
    asset = registry.add(source, label="hero")

    assert asset.analysis.width is None
    assert asset.analysis.colors == []
    assert asset.analysis.is_analyzed is False
