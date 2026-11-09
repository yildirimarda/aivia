"""Asset registry for one session or run.

The registry holds files with labels, content hashes and analysis. Originals
are copied into the registry directory and marked read-only, so they can never
be modified in place. Duplicate labels are rejected.

The media kind is detected from the file extension when it is not supplied.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from aivia.core.ids import new_asset_id
from aivia.core.models.asset import Asset, AssetKind

__all__ = [
    "AssetRegistry",
    "DuplicateLabelError",
    "UnknownLabelError",
    "resolve_labels",
]

_EXTENSION_KINDS: dict[str, AssetKind] = {
    ".png": AssetKind.IMAGE,
    ".jpg": AssetKind.IMAGE,
    ".jpeg": AssetKind.IMAGE,
    ".gif": AssetKind.IMAGE,
    ".webp": AssetKind.IMAGE,
    ".bmp": AssetKind.IMAGE,
    ".tiff": AssetKind.IMAGE,
    ".mp4": AssetKind.VIDEO,
    ".mov": AssetKind.VIDEO,
    ".webm": AssetKind.VIDEO,
    ".mkv": AssetKind.VIDEO,
    ".wav": AssetKind.AUDIO,
    ".mp3": AssetKind.AUDIO,
    ".aac": AssetKind.AUDIO,
    ".ogg": AssetKind.AUDIO,
    ".flac": AssetKind.AUDIO,
    ".txt": AssetKind.TEXT,
    ".md": AssetKind.TEXT,
    ".json": AssetKind.TEXT,
    ".yaml": AssetKind.TEXT,
    ".yml": AssetKind.TEXT,
    ".ttf": AssetKind.FONT,
    ".otf": AssetKind.FONT,
    ".woff": AssetKind.FONT,
    ".woff2": AssetKind.FONT,
}

_READ_ONLY_MODE = 0o444


class DuplicateLabelError(ValueError):
    """Raised when an asset is added with a label already in the registry."""


class UnknownLabelError(ValueError):
    """Raised when an instruction references an asset label that is not registered."""

    code: str = "unknown_asset_label"


_LABEL_RE = re.compile(r"@([A-Za-z0-9_.-]+)")


def resolve_labels(instruction: str, registry: AssetRegistry) -> dict[str, Asset]:
    """Resolve every ``@label`` reference in *instruction* to its asset.

    Returns a mapping of label to :class:`Asset`. Each referenced label is
    resolved once, so repeated references collapse to a single entry. Raises
    :class:`UnknownLabelError` (a :class:`ValueError`) naming the first label
    that is not registered, so the failure is clear and actionable.
    """
    found = _LABEL_RE.findall(instruction)
    resolved: dict[str, Asset] = {}
    for label in found:
        if label in resolved:
            continue
        asset = registry.get(label)
        if asset is None:
            raise UnknownLabelError(f"unknown asset label: {label!r}")
        resolved[label] = asset
    return resolved


def _kind_from_extension(path: Path) -> AssetKind:
    return _EXTENSION_KINDS.get(path.suffix.lower(), AssetKind.OTHER)


def _content_hash(data: bytes) -> str:
    """Return the SHA-256 digest of *data*."""
    return hashlib.sha256(data).hexdigest()


class AssetRegistry:
    """A collection of assets for one session or run.

    Assets are keyed by label. The file is copied into the registry root and
    marked read-only, so the stored original can never be modified in place.
    Duplicate labels are rejected with :class:`DuplicateLabelError`.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        import tempfile

        if root is None:
            root = Path(tempfile.mkdtemp(prefix="aivia-assets-"))
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._assets: dict[str, Asset] = {}

    @property
    def root(self) -> Path:
        """The directory holding the read-only originals."""
        return self._root

    @property
    def labels(self) -> tuple[str, ...]:
        """All registered labels in sorted order."""
        return tuple(sorted(self._assets))

    @property
    def hashes(self) -> dict[str, str]:
        """The content hash of every registered asset, keyed by label."""
        return {label: asset.content_hash for label, asset in self._assets.items()}

    def add(
        self,
        path: str | Path,
        *,
        label: str | None = None,
        kind: AssetKind | None = None,
    ) -> Asset:
        """Register *path* under *label* and return its asset record.

        The file is copied into the registry root and marked read-only. The
        content hash is computed over the file bytes. When *label* is omitted
        it defaults to the file name stem. Raises :class:`DuplicateLabelError`
        when *label* is already registered.
        """
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"asset file not found: {source}")
        data = source.read_bytes()
        content_hash = _content_hash(data)

        resolved_label = label.strip() if isinstance(label, str) else source.stem
        if not resolved_label:
            raise ValueError("asset label must not be empty")
        if resolved_label in self._assets:
            raise DuplicateLabelError(f"label already registered: {resolved_label!r}")

        resolved_kind: AssetKind = (
            _kind_from_extension(source) if kind is None else kind
        )

        stored_name = f"{content_hash}{source.suffix}"
        stored_path = self._root / stored_name
        if not stored_path.exists():
            stored_path.write_bytes(data)
            stored_path.chmod(_READ_ONLY_MODE)

        asset = Asset(
            id=new_asset_id(),
            label=resolved_label,
            kind=resolved_kind,
            path=str(stored_path),
            content_hash=content_hash,
        )
        self._assets[resolved_label] = asset
        return asset

    def get(self, label: str) -> Asset | None:
        """Return the asset for *label*, or ``None`` when not registered."""
        return self._assets.get(label.strip() if isinstance(label, str) else label)

    def __contains__(self, label: object) -> bool:
        return isinstance(label, str) and label.strip() in self._assets

    def __len__(self) -> int:
        return len(self._assets)

    def __iter__(self) -> Any:
        return iter(self.labels)
