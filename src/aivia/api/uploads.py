"""Upload filename sanitization for the local API.

Multipart filenames are attacker-controlled data. They are reduced to a single
safe path component before the rest of the application sees them, so no caller
can traverse directories, hide files or smuggle control characters.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = ["sanitize_filename"]

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
_REPEATED_UNDERSCORE = re.compile(r"_{2,}")
_WINDOWS_RESERVED = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{number}" for number in range(1, 10)),
        *(f"LPT{number}" for number in range(1, 10)),
    }
)


def sanitize_filename(
    name: str | None,
    *,
    fallback: str = "upload",
    max_length: int = 128,
) -> str:
    """Return a safe, single-component filename derived from *name*.

    Path separators are discarded, control characters and anything outside
    ``[A-Za-z0-9._-]`` become underscores, leading dots are stripped and the
    result is truncated while preserving its extension. Reserved device names
    are prefixed so they stay ordinary files on Windows.
    """
    if max_length <= 0:
        raise ValueError("max_length must be positive")

    if name is None:
        return fallback

    candidate = unicodedata.normalize("NFKC", name)
    candidate = candidate.replace("\\", "/").rsplit("/", maxsplit=1)[-1]
    candidate = "".join(
        character
        for character in candidate
        if character.isprintable() and character not in {"\x00"}
    )
    candidate = _UNSAFE.sub("_", candidate)
    candidate = _REPEATED_UNDERSCORE.sub("_", candidate)
    candidate = candidate.strip(" ._-")

    if not candidate or candidate in {".", ".."}:
        return fallback

    stem, extension = _split_extension(candidate)
    if stem.upper() in _WINDOWS_RESERVED:
        stem = f"_{stem}"
    candidate = f"{stem}{extension}"

    if len(candidate) > max_length:
        extension = extension[:16]
        keep = max(1, max_length - len(extension))
        stem = stem[:keep].rstrip(" ._-")
        candidate = f"{stem or fallback}{extension}"

    return candidate


def _split_extension(name: str) -> tuple[str, str]:
    stem, dot, extension = name.rpartition(".")
    if not dot or not stem:
        return name, ""
    return stem, f".{extension}"
