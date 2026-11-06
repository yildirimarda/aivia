"""Cache rendered compositions by every input that can change their pixels.

A render is reusable only when the composition, the bytes of every asset it
references, the renderer implementation, the installed ffmpeg build and the
font set are all identical. :class:`RenderCacheKey` folds those values into one
stable SHA-256 digest and :class:`RenderCache` stores the rendered artifact in
a directory named by that digest.

The cache is local and read-only with respect to its inputs: hashing a file or
an already decoded image never calls a provider and never touches the network.
A lookup that finds metadata without its artifact, or metadata that does not
match the requested key, is a miss rather than a corrupt hit.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from aivia.core import paths
from aivia.core import toolchain as _toolchain
from aivia.core.models.base import canonical_json
from aivia.core.models.run import ToolchainFingerprint

from .io import composition_to_dict
from .render import PILLOW_STILL_RENDERER_VERSION, RENDERER_VERSION
from .schema import Composition
from .urls import local_source_path

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

RENDER_CACHE_VERSION = "1"
"""The on-disk layout version of a cached render entry."""

ARTIFACT_NAME = "artifact"
"""The file name of the cached render inside its digest directory."""

METADATA_NAME = "entry.json"
"""The file name of the cache entry metadata inside its digest directory."""

_HEX_DIGEST = re.compile(r"^[0-9a-fA-F]{64}$")
_ASSET_NAMES = ("source", "path", "file", "asset", "asset_id")

type CompositionLike = Composition | Mapping[str, Any]
type AssetResolver = Mapping[str, Any] | Callable[[str], Any]
type HashInput = Sequence[str] | Mapping[str, Any] | str | None


class RenderCacheError(ValueError):
    """A render could not be stored in or read from the cache."""


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _value_for(obj: Any, *names: str) -> Any:
    if isinstance(obj, Mapping):
        for name in names:
            if name in obj:
                return obj[name]
        return None
    extras = getattr(obj, "model_extra", None) or {}
    for name in names:
        if isinstance(extras, Mapping) and name in extras:
            return extras[name]
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def _as_composition(value: CompositionLike) -> Composition:
    if isinstance(value, Composition):
        return value
    if isinstance(value, Mapping):
        return Composition.model_validate(dict(value))
    raise TypeError("expected a Composition or a mapping")


def _digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def composition_hash(composition: CompositionLike) -> str:
    """Return the stable SHA-256 of *composition*.

    The digest covers the canonical JSON form of the validated composition, so
    key order, aliases and formatting in a source document do not matter while
    every render-affecting field does.
    """
    payload = composition_to_dict(_as_composition(composition))
    return _digest_text(canonical_json(payload))


_composition_hash = composition_hash


def _composition_sources(composition: Composition) -> list[str]:
    sources: list[str] = []
    for layer in composition.layers:
        source = _value_for(layer, *_ASSET_NAMES)
        if isinstance(source, str) and source.strip():
            sources.append(source.strip())
        mask = _value_for(layer, "mask", "image_mask")
        if isinstance(mask, Mapping):
            mask_source = _value_for(mask, *_ASSET_NAMES)
        else:
            mask_source = _value_for(mask, *_ASSET_NAMES) if mask is not None else None
        if isinstance(mask_source, str) and mask_source.strip():
            sources.append(mask_source.strip())
    for track in composition.audio_tracks:
        source = _value_for(track, *_ASSET_NAMES)
        if isinstance(source, str) and source.strip():
            sources.append(source.strip())
    if composition.subtitle_file:
        sources.append(composition.subtitle_file)
    return sources


def _resolve_asset(
    source: str,
    assets: AssetResolver | None,
    source_root: str | Path | None,
) -> Any:
    if callable(assets):
        return assets(source)
    if isinstance(assets, Mapping) and source in assets:
        return assets[source]
    path = local_source_path(source)
    if source_root is not None and not path.is_absolute():
        path = Path(source_root) / path
    return path


def _hash_image(image: Image.Image) -> str:
    digest = hashlib.sha256()
    digest.update(image.mode.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(image.size).encode("ascii"))
    digest.update(b"\0")
    digest.update(image.tobytes())
    return digest.hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_asset(value: Any, source: str) -> str:
    if value is None:
        return _digest_text(source)
    if isinstance(value, Mapping):
        for name in ("content_hash", "sha256", "digest", "hash"):
            recorded = value.get(name)
            if isinstance(recorded, str) and recorded.strip():
                return recorded.strip().lower()
    for name in ("content_hash", "sha256", "digest", "hash"):
        recorded = getattr(value, name, None)
        if isinstance(recorded, str) and recorded.strip():
            return recorded.strip().lower()
    if isinstance(value, Image.Image):
        return _hash_image(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _digest_bytes(bytes(value))
    if isinstance(value, Path):
        return _hash_file(value) if value.is_file() else _digest_text(str(value))
    if isinstance(value, str):
        text = value.strip()
        if _HEX_DIGEST.fullmatch(text):
            return text.lower()
        path = local_source_path(text)
        if path.is_file():
            return _hash_file(path)
        return _digest_text(text)
    reader = getattr(value, "read", None)
    if callable(reader):
        data = reader()
        if isinstance(data, str):
            return _digest_text(data)
        return _digest_bytes(bytes(data))
    return _digest_text(str(value))


def asset_hash_map(
    composition: CompositionLike,
    assets: AssetResolver | None = None,
    *,
    source_root: str | Path | None = None,
) -> dict[str, str]:
    """Return one content digest for every asset *composition* references.

    Sources are resolved through *assets* (a mapping or callable), then through
    *source_root* or the local filesystem. An unresolved source hashes its own
    name, so a composition that points at a missing asset still has a stable
    key and becomes a different key once the asset appears.
    """
    model = _as_composition(composition)
    hashes: dict[str, str] = {}
    for source in _composition_sources(model):
        value = _resolve_asset(source, assets, source_root)
        hashes[source] = _hash_asset(value, source)
    return hashes


def asset_hashes(
    composition: CompositionLike,
    assets: AssetResolver | None = None,
    *,
    source_root: str | Path | None = None,
) -> tuple[str, ...]:
    """Return the sorted, de-duplicated asset digests of *composition*."""
    return tuple(
        sorted(
            set(asset_hash_map(composition, assets, source_root=source_root).values())
        )
    )


hash_assets = asset_hashes
"""Alias for :func:`asset_hashes`."""


def detect_toolchain(
    *,
    fonts_dir: str | Path | None = None,
    executable: str = "ffmpeg",
    renderer_version: str | None = None,
) -> ToolchainFingerprint:
    """Return the toolchain fingerprint used by this cache."""
    return _toolchain.detect_toolchain(
        fonts_dir=fonts_dir,
        executable=executable,
        renderer_version=renderer_version,
    )


def ffmpeg_version(executable: str = "ffmpeg") -> str:
    """Return the version reported by *executable*, or an empty string."""
    return _toolchain.ffmpeg_version(executable)


def font_set_identity(fonts_dir: str | Path | None = None) -> str:
    """Return a stable digest of the font files below *fonts_dir*."""
    return _toolchain.font_set_identity(fonts_dir)


font_fingerprint = font_set_identity
"""Alias for :func:`font_set_identity`."""


def _normalise_hashes(value: HashInput) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items: Sequence[Any] = (value,)
    elif isinstance(value, Mapping):
        items = list(value.values())
    else:
        items = value
    cleaned = {_text(item) for item in items if _text(item)}
    return tuple(sorted(cleaned))


def _coerce_toolchain(
    value: ToolchainFingerprint | Mapping[str, Any] | None,
) -> ToolchainFingerprint | None:
    if value is None:
        return None
    return ToolchainFingerprint.model_validate(value)


@dataclass(frozen=True, slots=True, init=False)
class RenderCacheKey:
    """Every input that makes two renders the same render.

    ``composition_hash`` and ``asset_hashes`` identify the document and the
    bytes it references. The three toolchain components identify the software
    that turns them into pixels. ``font_set`` is accepted as an alias for
    ``font_fingerprint`` and a ``toolchain`` object supplies any missing
    component, so callers can pass either representation.
    """

    composition_hash: str
    asset_hashes: tuple[str, ...]
    renderer_version: str
    ffmpeg_version: str
    font_fingerprint: str

    def __init__(
        self,
        composition_hash: str = "",
        asset_hashes: HashInput = None,
        renderer_version: str = "",
        ffmpeg_version: str = "",
        font_fingerprint: str = "",
        *,
        composition: CompositionLike | None = None,
        composition_digest: str | None = None,
        content_hash: str | None = None,
        assets: AssetResolver | None = None,
        source_root: str | Path | None = None,
        font_set: str | None = None,
        font_identity: str | None = None,
        fonts_hash: str | None = None,
        renderer: str | None = None,
        ffmpeg: str | None = None,
        toolchain: ToolchainFingerprint | Mapping[str, Any] | None = None,
    ) -> None:
        fingerprint = _coerce_toolchain(toolchain)
        document_hash = _text(
            composition_digest
            or content_hash
            or composition_hash
            or ("" if composition is None else _composition_hash(composition))
        )
        if asset_hashes is None and composition is not None:
            resolved = asset_hashes_for(composition, assets, source_root=source_root)
        else:
            resolved = _normalise_hashes(asset_hashes)
        renderer_text = _text(renderer_version or renderer)
        if not renderer_text and fingerprint is not None:
            renderer_text = fingerprint.renderer_version
        ffmpeg_text = _text(ffmpeg_version or ffmpeg)
        if not ffmpeg_text and fingerprint is not None:
            ffmpeg_text = fingerprint.ffmpeg_version
        font_text = _text(font_fingerprint or font_set or font_identity or fonts_hash)
        if not font_text and fingerprint is not None:
            font_text = fingerprint.font_set
        object.__setattr__(self, "composition_hash", document_hash)
        object.__setattr__(self, "asset_hashes", resolved)
        object.__setattr__(self, "renderer_version", renderer_text)
        object.__setattr__(self, "ffmpeg_version", ffmpeg_text)
        object.__setattr__(self, "font_fingerprint", font_text)

    def as_mapping(self) -> dict[str, Any]:
        """Return the key components as a JSON-compatible mapping."""
        return {
            "composition_hash": self.composition_hash,
            "asset_hashes": list(self.asset_hashes),
            "renderer_version": self.renderer_version,
            "ffmpeg_version": self.ffmpeg_version,
            "font_fingerprint": self.font_fingerprint,
        }

    to_dict = as_mapping

    @property
    def identity(self) -> str:
        """The canonical representation of the key components."""
        return canonical_json(self.as_mapping())

    @property
    def digest(self) -> str:
        """The stable SHA-256 used as the cache directory name."""
        return _digest_text(self.identity)

    @property
    def cache_key(self) -> str:
        """Alias for :attr:`digest`."""
        return self.digest

    @property
    def hash(self) -> str:
        """Alias for :attr:`digest`."""
        return self.digest

    @property
    def hexdigest(self) -> str:
        """Alias for :attr:`digest`."""
        return self.digest

    @property
    def font_set(self) -> str:
        """Alias for :attr:`font_fingerprint`."""
        return self.font_fingerprint

    @property
    def assets(self) -> tuple[str, ...]:
        """Alias for :attr:`asset_hashes`."""
        return self.asset_hashes

    @property
    def toolchain(self) -> ToolchainFingerprint:
        """The toolchain components as a fingerprint record."""
        return ToolchainFingerprint(
            renderer_version=self.renderer_version,
            ffmpeg_version=self.ffmpeg_version,
            font_set=self.font_fingerprint,
        )

    def __str__(self) -> str:
        return self.digest


def asset_hashes_for(
    composition: CompositionLike,
    assets: AssetResolver | None = None,
    *,
    source_root: str | Path | None = None,
) -> tuple[str, ...]:
    """Internal-friendly alias for :func:`asset_hashes`."""
    return asset_hashes(composition, assets, source_root=source_root)


def render_cache_key(
    composition: CompositionLike | None = None,
    *,
    assets: AssetResolver | None = None,
    asset_hashes: HashInput = None,
    source_root: str | Path | None = None,
    composition_hash: str | None = None,
    renderer_version: str | None = None,
    ffmpeg_version: str | None = None,
    font_fingerprint: str | None = None,
    font_set: str | None = None,
    toolchain: ToolchainFingerprint | Mapping[str, Any] | None = None,
    fonts_dir: str | Path | None = None,
    executable: str = "ffmpeg",
) -> RenderCacheKey:
    """Build a :class:`RenderCacheKey` for *composition*.

    Any missing toolchain component is detected locally. Explicit values and a
    supplied ``toolchain`` take precedence over detection, which keeps the
    function deterministic in tests.
    """
    fingerprint = _coerce_toolchain(toolchain)
    if fingerprint is None:
        fingerprint = detect_toolchain(
            fonts_dir=fonts_dir,
            executable=executable,
            renderer_version=renderer_version or RENDERER_VERSION,
        )
    document_hash = composition_hash
    if document_hash is None and composition is not None:
        document_hash = _composition_hash(composition)
    hashes = asset_hashes
    if hashes is None and composition is not None:
        hashes = asset_hashes_for(composition, assets, source_root=source_root)
    return RenderCacheKey(
        composition_hash=document_hash or "",
        asset_hashes=hashes,
        renderer_version=renderer_version or fingerprint.renderer_version,
        ffmpeg_version=ffmpeg_version or fingerprint.ffmpeg_version,
        font_fingerprint=(font_fingerprint or font_set or fingerprint.font_set),
    )


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return datetime.now(UTC)
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True, eq=False)
class RenderCacheEntry:
    """One cached render artifact and the key that produced it."""

    key: RenderCacheKey
    path: Path
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: Mapping[str, Any] = field(default_factory=dict)
    extension: str = ""

    @property
    def digest(self) -> str:
        """The digest of :attr:`key`."""
        return self.key.digest

    @property
    def data(self) -> bytes:
        """The cached artifact bytes."""
        return self.path.read_bytes()

    @property
    def size_bytes(self) -> int:
        """The size of the cached artifact."""
        return self.path.stat().st_size

    @property
    def exists(self) -> bool:
        """Whether the artifact is still on disk."""
        return self.path.is_file()

    def is_file(self) -> bool:
        """Whether the artifact is still on disk."""
        return self.exists

    def read_bytes(self) -> bytes:
        """Read the cached artifact."""
        return self.path.read_bytes()

    def read_text(self, encoding: str = "utf-8") -> str:
        """Read the cached artifact as text."""
        return self.path.read_text(encoding=encoding)

    def __fspath__(self) -> str:
        return str(self.path)

    def __str__(self) -> str:
        return str(self.path)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, RenderCacheEntry):
            return self.key == other.key and self.path == other.path
        if isinstance(other, (str, Path)):
            return self.path == Path(other)
        if isinstance(other, (bytes, bytearray, memoryview)):
            return self.path.is_file() and self.read_bytes() == bytes(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.key, self.path))


def _artifact_bytes(data: Any) -> tuple[bytes, str]:
    if isinstance(data, Image.Image):
        buffer = BytesIO()
        data.save(buffer, format="PNG")
        return buffer.getvalue(), ".png"
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data), ""
    if isinstance(data, (str, Path)):
        path = Path(data)
        if not path.is_file():
            raise RenderCacheError(f"render artifact not found: {path}")
        return path.read_bytes(), path.suffix
    for name in ("path", "output_path", "file"):
        recorded = getattr(data, name, None)
        if isinstance(recorded, (str, Path)):
            path = Path(recorded)
            if path.is_file():
                return path.read_bytes(), path.suffix
    reader = getattr(data, "read", None)
    if callable(reader):
        value = reader()
        if isinstance(value, str):
            return value.encode("utf-8"), ""
        return bytes(value), ""
    raise RenderCacheError("a render artifact must be bytes, a path or an image")


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


class RenderCache:
    """Store rendered artifacts under a directory named by their cache key.

    Pass a root directory, or omit it to use ``data/cache/renders`` below the
    user data directory. The constructor detects the local toolchain once; a
    supplied ``toolchain`` or explicit component overrides detection.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        renderer: Any = None,
        renderer_version: str | None = None,
        ffmpeg_version: str | None = None,
        font_fingerprint: str | None = None,
        font_set: str | None = None,
        toolchain: ToolchainFingerprint | Mapping[str, Any] | None = None,
        assets: AssetResolver | None = None,
        source_root: str | Path | None = None,
        fonts_dir: str | Path | None = None,
        executable: str = "ffmpeg",
        no_cache: bool = False,
    ) -> None:
        self.root = (
            Path(root).expanduser()
            if root is not None
            else paths.data_dir() / "cache" / "renders"
        )
        self._assets = assets
        self._source_root = None if source_root is None else Path(source_root)
        self.no_cache = bool(no_cache)
        version = (
            renderer_version
            or getattr(renderer, "version", None)
            or PILLOW_STILL_RENDERER_VERSION
        )
        fingerprint = _coerce_toolchain(toolchain)
        if fingerprint is None:
            fingerprint = detect_toolchain(
                fonts_dir=fonts_dir,
                executable=executable,
                renderer_version=str(version),
            )
        self._toolchain = fingerprint
        self._renderer_version = (
            _text(renderer_version)
            or _text(getattr(renderer, "version", None))
            or fingerprint.renderer_version
            or str(version)
        )
        self._ffmpeg_version = _text(ffmpeg_version) or fingerprint.ffmpeg_version
        self._font_fingerprint = (
            _text(font_fingerprint) or _text(font_set) or fingerprint.font_set
        )

    @property
    def toolchain(self) -> ToolchainFingerprint:
        """The fingerprint recorded with every cached render."""
        return self._toolchain

    @property
    def renderer_version(self) -> str:
        """The renderer version component of the key."""
        return self._renderer_version

    @property
    def ffmpeg_version(self) -> str:
        """The ffmpeg version component of the key."""
        return self._ffmpeg_version

    @property
    def font_fingerprint(self) -> str:
        """The font fingerprint component of the key."""
        return self._font_fingerprint

    @property
    def font_set(self) -> str:
        """Alias for :attr:`font_fingerprint`."""
        return self._font_fingerprint

    @property
    def directory(self) -> Path:
        """Alias for :attr:`root`."""
        return self.root

    def key(
        self,
        composition: CompositionLike,
        *,
        assets: AssetResolver | None = None,
        asset_hashes: HashInput = None,
        source_root: str | Path | None = None,
        renderer_version: str | None = None,
        ffmpeg_version: str | None = None,
        font_fingerprint: str | None = None,
        font_set: str | None = None,
    ) -> RenderCacheKey:
        """Build the cache key for *composition* with this cache's toolchain."""
        return render_cache_key(
            composition,
            assets=self._assets if assets is None else assets,
            asset_hashes=asset_hashes,
            source_root=(self._source_root if source_root is None else source_root),
            renderer_version=renderer_version or self._renderer_version,
            ffmpeg_version=ffmpeg_version or self._ffmpeg_version,
            font_fingerprint=(font_fingerprint or font_set or self._font_fingerprint),
        )

    def path_for(self, key: RenderCacheKey) -> Path:
        """Return the artifact path for *key*, whether or not it exists."""
        return self.root / key.digest / ARTIFACT_NAME

    def metadata_path(self, key: RenderCacheKey) -> Path:
        """Return the metadata path for *key*, whether or not it exists."""
        return self.root / key.digest / METADATA_NAME

    def _coerce_key(
        self,
        key: RenderCacheKey | CompositionLike | None,
        *,
        composition: CompositionLike | None = None,
        assets: AssetResolver | None = None,
        asset_hashes: HashInput = None,
        source_root: str | Path | None = None,
    ) -> RenderCacheKey:
        if isinstance(key, RenderCacheKey):
            return key
        value = key if key is not None else composition
        if value is None:
            raise TypeError("a RenderCacheKey or a composition is required")
        return self.key(
            value,
            assets=assets,
            asset_hashes=asset_hashes,
            source_root=source_root,
        )

    def _load(self, key: RenderCacheKey) -> RenderCacheEntry | None:
        artifact = self.path_for(key)
        metadata_file = self.metadata_path(key)
        if not artifact.is_file() or not metadata_file.is_file():
            return None
        try:
            payload = json.loads(metadata_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(payload, Mapping):
            return None
        if payload.get("cache_version") != RENDER_CACHE_VERSION:
            return None
        if payload.get("cache_key") != key.digest:
            return None
        metadata = payload.get("metadata")
        if not isinstance(metadata, Mapping):
            metadata = {}
        extension = payload.get("extension")
        return RenderCacheEntry(
            key=key,
            path=artifact,
            created_at=_parse_timestamp(payload.get("created_at")),
            metadata=dict(metadata),
            extension=extension if isinstance(extension, str) else "",
        )

    def get(
        self,
        key: RenderCacheKey | CompositionLike | None = None,
        *,
        composition: CompositionLike | None = None,
        assets: AssetResolver | None = None,
        asset_hashes: HashInput = None,
        source_root: str | Path | None = None,
    ) -> RenderCacheEntry | None:
        """Return the cached render for *key*, or ``None`` on a miss.

        A composition may be passed instead of a key; it is hashed with this
        cache's toolchain and the optional per-call asset resolver.
        """
        if self.no_cache:
            return None
        cache_key = self._coerce_key(
            key,
            composition=composition,
            assets=assets,
            asset_hashes=asset_hashes,
            source_root=source_root,
        )
        return self._load(cache_key)

    lookup = get
    load = get
    read = get

    def contains(
        self,
        key: RenderCacheKey | CompositionLike | None = None,
        *,
        composition: CompositionLike | None = None,
        assets: AssetResolver | None = None,
        asset_hashes: HashInput = None,
        source_root: str | Path | None = None,
    ) -> bool:
        """Whether a complete cached render exists for *key*."""
        return (
            self.get(
                key,
                composition=composition,
                assets=assets,
                asset_hashes=asset_hashes,
                source_root=source_root,
            )
            is not None
        )

    def store(
        self,
        key: RenderCacheKey | CompositionLike | None = None,
        data: Any = None,
        *,
        composition: CompositionLike | None = None,
        assets: AssetResolver | None = None,
        asset_hashes: HashInput = None,
        source_root: str | Path | None = None,
        extension: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RenderCacheEntry:
        """Write *data* as the render for *key* and return its entry.

        *data* may be encoded bytes, a local file path or a decoded Pillow
        image. The artifact and its metadata are written atomically so an
        interrupted store leaves a miss rather than a partial hit.
        """
        cache_key = self._coerce_key(
            key,
            composition=composition,
            assets=assets,
            asset_hashes=asset_hashes,
            source_root=source_root,
        )
        payload, detected_extension = _artifact_bytes(data)
        suffix = extension if extension is not None else detected_extension
        artifact = self.path_for(cache_key)
        created_at = datetime.now(UTC)
        entry_metadata = dict(metadata or {})
        record = {
            "cache_version": RENDER_CACHE_VERSION,
            "cache_key": cache_key.digest,
            "key": cache_key.as_mapping(),
            "artifact": ARTIFACT_NAME,
            "extension": suffix,
            "created_at": created_at.isoformat(),
            "metadata": entry_metadata,
        }
        _write_atomic(artifact, payload)
        _write_atomic(
            self.metadata_path(cache_key),
            json.dumps(record, sort_keys=True, ensure_ascii=False).encode("utf-8"),
        )
        return RenderCacheEntry(
            key=cache_key,
            path=artifact,
            created_at=created_at,
            metadata=entry_metadata,
            extension=suffix,
        )

    put = store
    save = store
    write = store

    def clear(self) -> int:
        """Remove every cached render and return how many entries were removed."""
        if not self.root.is_dir():
            return 0
        entries = [path for path in self.root.iterdir() if path.is_dir()]
        shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        return len(entries)

    def prune(self, max_entries: int) -> int:
        """Keep only the *max_entries* most recently written renders."""
        if isinstance(max_entries, bool) or not isinstance(max_entries, int):
            raise TypeError("max_entries must be an integer")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        if not self.root.is_dir():
            return 0
        entries = [path for path in self.root.iterdir() if path.is_dir()]
        if len(entries) <= max_entries:
            return 0
        ordered = sorted(
            entries,
            key=lambda path: (
                (path / METADATA_NAME).stat().st_mtime
                if (path / METADATA_NAME).is_file()
                else 0.0,
                path.name,
            ),
            reverse=True,
        )
        removed = 0
        for path in ordered[max_entries:]:
            shutil.rmtree(path, ignore_errors=True)
            removed += 1
        return removed


def cached_render(
    cache: RenderCache,
    composition: CompositionLike,
    *,
    assets: AssetResolver | None = None,
    asset_hashes: HashInput = None,
    source_root: str | Path | None = None,
) -> RenderCacheEntry | None:
    """Look up *composition* in *cache* with optional per-call assets."""
    key = cache.key(
        composition,
        assets=assets,
        asset_hashes=asset_hashes,
        source_root=source_root,
    )
    return cache.get(key)


def cache_render(
    cache: RenderCache,
    composition: CompositionLike,
    data: Any,
    *,
    assets: AssetResolver | None = None,
    asset_hashes: HashInput = None,
    source_root: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RenderCacheEntry:
    """Store *data* for *composition* in *cache*."""
    key = cache.key(
        composition,
        assets=assets,
        asset_hashes=asset_hashes,
        source_root=source_root,
    )
    return cache.store(key, data, metadata=metadata)


load_cached_render = cached_render
"""Alias for :func:`cached_render`."""

store_render = cache_render
"""Alias for :func:`cache_render`."""
