"""Resolve local file URLs to filesystem paths.

Composition sources are usually relative asset names, but stored visuals are
addressable as ``file://`` URLs. Rendering and hashing must open the same local
file the URL names rather than treat the URL text as a path.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlsplit

__all__ = ["file_url_path", "local_source_path"]


def file_url_path(source: str) -> Path | None:
    """Return the local path named by a ``file://`` URL, or ``None``."""
    try:
        parsed = urlsplit(source)
    except ValueError:
        return None
    if parsed.scheme.lower() != "file":
        return None
    if parsed.netloc and parsed.netloc.lower() not in {"", "localhost"}:
        return None
    path = unquote(parsed.path)
    return Path(path) if path else None


def local_source_path(source: str | Path) -> Path:
    """Return the local path for *source*, resolving a ``file://`` URL.

    Other strings, including relative names and unknown schemes, are returned
    as a plain :class:`Path` so existing lookup behaviour is unchanged.
    """
    if isinstance(source, Path):
        return source
    local = file_url_path(source)
    return Path(source) if local is None else local
