"""Render cache keys and artifacts are stable and invalidate on every input."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from aivia.compose import (
    RENDER_CACHE_VERSION,
    Composition,
    RenderCache,
    RenderCacheKey,
    asset_hash_map,
    asset_hashes,
    cache_render,
    cached_render,
    composition_hash,
    render_cache_key,
)
from aivia.compose import render_cache as render_cache_module
from aivia.core.models import ToolchainFingerprint

RENDERER = "renderer-1"
FFMPEG = "6.1.1"
FONTS = "fonts-abc"


def _composition(**overrides: Any) -> Composition:
    values: dict[str, Any] = {
        "canvas": "4x4",
        "duration_seconds": 3.0,
        "fps": 30,
    }
    values.update(overrides)
    return Composition.model_validate(values)


def _image_composition() -> Composition:
    return _composition(
        layers=[{"id": "still", "type": "image", "source": "background.png"}]
    )


def _key(composition: Composition, **overrides: Any) -> RenderCacheKey:
    values: dict[str, Any] = {
        "assets": {"background.png": b"first image"},
        "renderer_version": RENDERER,
        "ffmpeg_version": FFMPEG,
        "font_fingerprint": FONTS,
    }
    values.update(overrides)
    return render_cache_key(composition, **values)


def test_the_composition_hash_ignores_formatting_but_not_content() -> None:
    first = _composition(
        canvas={"width": 4, "height": 4, "background": "#000000"},
    )
    equivalent = Composition.model_validate(
        {
            "canvas": {"width": 4, "height": 4, "background": "#000000"},
            "duration_seconds": 3.0,
            "fps": 30,
        }
    )
    changed = _composition(
        canvas={"width": 4, "height": 4, "background": "#ffffff"},
    )

    assert composition_hash(first) == composition_hash(equivalent)
    assert composition_hash(first) != composition_hash(changed)
    assert len(composition_hash(first)) == 64


def test_asset_hashes_change_when_asset_bytes_change() -> None:
    composition = _image_composition()

    first = asset_hash_map(composition, {"background.png": b"first image"})
    second = asset_hash_map(composition, {"background.png": b"second image"})

    assert first != second
    assert asset_hashes(composition, {"background.png": b"first image"}) == (
        first["background.png"],
    )


def test_the_key_is_stable_across_equivalent_inputs() -> None:
    composition = _image_composition()
    first = _key(composition)
    second = RenderCacheKey(
        composition_hash=composition_hash(composition),
        asset_hashes=[first.asset_hashes[0], first.asset_hashes[0]],
        renderer_version=RENDERER,
        ffmpeg_version=FFMPEG,
        font_set=FONTS,
    )

    assert first == second
    assert first.digest == second.digest
    assert first.cache_key == first.hash == first.hexdigest
    assert first.font_set == first.font_fingerprint == FONTS
    assert first.toolchain.renderer_version == RENDERER
    assert first.toolchain.ffmpeg_version == FFMPEG
    assert first.toolchain.font_set == FONTS


def test_every_render_input_changes_the_key() -> None:
    composition = _image_composition()
    base = _key(composition)

    variants = (
        _key(
            _composition(layers=[{"id": "still", "type": "image", "source": "b.png"}])
        ),
        _key(composition, assets={"background.png": b"other image"}),
        _key(composition, renderer_version="renderer-2"),
        _key(composition, ffmpeg_version="7.0"),
        _key(composition, font_fingerprint="fonts-def"),
    )

    for variant in variants:
        assert variant.digest != base.digest


def test_a_supplied_toolchain_fills_missing_components() -> None:
    composition = _composition()
    key = RenderCacheKey(
        composition=composition,
        asset_hashes=(),
        toolchain=ToolchainFingerprint(
            renderer_version=RENDERER,
            ffmpeg_version=FFMPEG,
            font_set=FONTS,
        ),
    )

    assert key.renderer_version == RENDERER
    assert key.ffmpeg_version == FFMPEG
    assert key.font_fingerprint == FONTS
    assert key.composition_hash == composition_hash(composition)


def test_missing_versions_are_detected_locally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fingerprint = ToolchainFingerprint(
        renderer_version="detected-renderer",
        ffmpeg_version="detected-ffmpeg",
        font_set="detected-fonts",
    )
    seen: dict[str, Any] = {}

    def fake_detect_toolchain(**kwargs: Any) -> ToolchainFingerprint:
        seen.update(kwargs)
        return fingerprint

    monkeypatch.setattr(render_cache_module, "detect_toolchain", fake_detect_toolchain)

    key = render_cache_key(_composition(), asset_hashes=())

    assert key.renderer_version == "detected-renderer"
    assert key.ffmpeg_version == "detected-ffmpeg"
    assert key.font_fingerprint == "detected-fonts"
    assert seen["renderer_version"] == render_cache_module.RENDERER_VERSION


def test_a_stored_render_is_reused_by_a_new_cache_instance(tmp_path: Path) -> None:
    composition = _image_composition()
    assets = {"background.png": b"first image"}
    cache = RenderCache(
        tmp_path / "renders",
        renderer_version=RENDERER,
        ffmpeg_version=FFMPEG,
        font_fingerprint=FONTS,
    )
    key = cache.key(composition, assets=assets)

    assert cache.get(key) is None
    assert cache.lookup(key) is None

    stored = cache.store(key, b"rendered bytes")

    assert stored.path.is_file()
    assert stored.read_bytes() == b"rendered bytes"
    assert stored.size_bytes == len(b"rendered bytes")
    assert stored.digest == key.digest
    assert stored.metadata == {}

    reopened = RenderCache(
        tmp_path / "renders",
        renderer_version=RENDERER,
        ffmpeg_version=FFMPEG,
        font_fingerprint=FONTS,
    )
    hit = reopened.get(key)
    assert hit is not None
    assert hit.path == stored.path
    assert hit == b"rendered bytes"
    assert hit.key == key
    assert hit.exists is True
    assert hit.is_file() is True
    assert Path(hit) == stored.path


def test_a_changed_input_misses_the_stored_render(tmp_path: Path) -> None:
    composition = _image_composition()
    cache = RenderCache(
        tmp_path,
        renderer_version=RENDERER,
        ffmpeg_version=FFMPEG,
        font_fingerprint=FONTS,
    )
    key = cache.key(composition, assets={"background.png": b"first image"})
    cache.store(key, b"rendered bytes")

    changed_asset = cache.key(composition, assets={"background.png": b"new image"})
    changed_renderer = cache.key(
        composition,
        assets={"background.png": b"first image"},
        renderer_version="renderer-2",
    )

    assert cache.get(changed_asset) is None
    assert cache.get(changed_renderer) is None


def test_a_missing_artifact_or_metadata_is_a_miss(tmp_path: Path) -> None:
    cache = RenderCache(
        tmp_path,
        renderer_version=RENDERER,
        ffmpeg_version=FFMPEG,
        font_fingerprint=FONTS,
    )
    key = cache.key(_composition(), asset_hashes=())
    entry = cache.store(key, b"rendered bytes")

    entry.path.unlink()
    assert cache.get(key) is None

    cache.store(key, b"rendered bytes")
    cache.metadata_path(key).write_text("not json", encoding="utf-8")
    assert cache.get(key) is None


def test_metadata_for_another_key_is_a_miss(tmp_path: Path) -> None:
    cache = RenderCache(
        tmp_path,
        renderer_version=RENDERER,
        ffmpeg_version=FFMPEG,
        font_fingerprint=FONTS,
    )
    key = cache.key(_composition(), asset_hashes=())
    cache.store(key, b"rendered bytes")
    cache.metadata_path(key).write_text(
        '{"cache_version": "'
        + RENDER_CACHE_VERSION
        + '", "cache_key": "'
        + "0" * 64
        + '"}',
        encoding="utf-8",
    )

    assert cache.get(key) is None


def test_metadata_from_another_cache_layout_is_a_miss(tmp_path: Path) -> None:
    cache = RenderCache(
        tmp_path,
        renderer_version=RENDERER,
        ffmpeg_version=FFMPEG,
        font_fingerprint=FONTS,
    )
    key = cache.key(_composition(), asset_hashes=())
    cache.store(key, b"rendered bytes")
    cache.metadata_path(key).write_text(
        '{"cache_version": "0", "cache_key": "' + key.digest + '"}',
        encoding="utf-8",
    )

    assert cache.get(key) is None


def test_a_decoded_image_is_stored_as_png(tmp_path: Path) -> None:
    cache = RenderCache(
        tmp_path,
        renderer_version=RENDERER,
        ffmpeg_version=FFMPEG,
        font_fingerprint=FONTS,
    )
    key = cache.key(_composition(), asset_hashes=())
    image = Image.new("RGBA", (2, 2), (10, 20, 30, 255))

    entry = cache.store(key, image)

    assert entry.extension == ".png"
    with Image.open(entry.path) as loaded:
        assert loaded.convert("RGBA").getpixel((0, 0)) == (10, 20, 30, 255)


def test_compositions_can_be_passed_without_building_a_key(tmp_path: Path) -> None:
    composition = _image_composition()
    assets = {"background.png": b"first image"}
    cache = RenderCache(
        tmp_path,
        assets=assets,
        renderer_version=RENDERER,
        ffmpeg_version=FFMPEG,
        font_fingerprint=FONTS,
    )

    assert cache.get(composition) is None
    entry = cache.store(composition, b"rendered bytes")
    hit = cache.get(composition)

    assert hit is not None
    assert hit.path == entry.path
    assert cached_render(cache, composition) == entry
    stored = cache_render(cache, composition, b"second render")
    assert stored.read_bytes() == b"second render"
    assert cache.get(composition) == stored


def test_no_cache_disables_lookups_but_still_stores(tmp_path: Path) -> None:
    cache = RenderCache(
        tmp_path,
        renderer_version=RENDERER,
        ffmpeg_version=FFMPEG,
        font_fingerprint=FONTS,
        no_cache=True,
    )
    key = cache.key(_composition(), asset_hashes=())

    entry = cache.store(key, b"rendered bytes")

    assert entry.path.is_file()
    assert cache.get(key) is None
    assert cache.contains(key) is False


def test_clear_removes_every_entry(tmp_path: Path) -> None:
    cache = RenderCache(
        tmp_path,
        renderer_version=RENDERER,
        ffmpeg_version=FFMPEG,
        font_fingerprint=FONTS,
    )
    first = cache.key(_composition(fps=30), asset_hashes=())
    second = cache.key(_composition(fps=24), asset_hashes=())
    cache.store(first, b"one")
    cache.store(second, b"two")

    assert cache.clear() == 2
    assert cache.get(first) is None
    assert cache.get(second) is None


def test_an_unsupported_artifact_does_not_leave_an_entry(tmp_path: Path) -> None:
    cache = RenderCache(
        tmp_path,
        renderer_version=RENDERER,
        ffmpeg_version=FFMPEG,
        font_fingerprint=FONTS,
    )
    key = cache.key(_composition(), asset_hashes=())

    with pytest.raises(ValueError):
        cache.store(key, object())

    assert cache.get(key) is None


def test_the_on_disk_metadata_records_the_key_and_layout(tmp_path: Path) -> None:
    cache = RenderCache(
        tmp_path,
        renderer_version=RENDERER,
        ffmpeg_version=FFMPEG,
        font_fingerprint=FONTS,
    )
    key = cache.key(_composition(), asset_hashes=())
    cache.store(key, b"rendered bytes", metadata={"format": "mp4"})

    text = cache.metadata_path(key).read_text(encoding="utf-8")

    assert f'"cache_version": "{RENDER_CACHE_VERSION}"' in text
    assert f'"cache_key": "{key.digest}"' in text
    assert '"format": "mp4"' in text
