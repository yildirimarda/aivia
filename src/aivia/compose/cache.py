"""Alias for :mod:`aivia.compose.render_cache`."""

from __future__ import annotations

from aivia.compose.render_cache import (
    RENDER_CACHE_VERSION,
    RENDERER_VERSION,
    RenderCache,
    RenderCacheEntry,
    RenderCacheError,
    RenderCacheKey,
    asset_hash_map,
    asset_hashes,
    asset_hashes_for,
    cache_render,
    cached_render,
    composition_hash,
    detect_toolchain,
    ffmpeg_version,
    font_fingerprint,
    font_set_identity,
    hash_assets,
    load_cached_render,
    render_cache_key,
    store_render,
)

__all__ = [
    "RENDERER_VERSION",
    "RENDER_CACHE_VERSION",
    "RenderCache",
    "RenderCacheEntry",
    "RenderCacheError",
    "RenderCacheKey",
    "asset_hash_map",
    "asset_hashes",
    "asset_hashes_for",
    "cache_render",
    "cached_render",
    "composition_hash",
    "detect_toolchain",
    "ffmpeg_version",
    "font_fingerprint",
    "font_set_identity",
    "hash_assets",
    "load_cached_render",
    "render_cache_key",
    "store_render",
]
