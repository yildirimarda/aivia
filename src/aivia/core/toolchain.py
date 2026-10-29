"""Detect the renderer toolchain recorded with runs and replays.

The toolchain fingerprint names the three inputs that decide whether a render
is reproducible: the renderer implementation version, the installed ffmpeg
build and the identity of the font set. Detection is local and read-only: it
hashes the packaged font files and asks ffmpeg for its version, never touching
the network.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from aivia import __version__
from aivia.core import paths
from aivia.core.models.run import ToolchainFingerprint

__all__ = [
    "COMPONENTS",
    "ToolchainDrift",
    "detect_toolchain",
    "diff_toolchain",
    "ffmpeg_version",
    "font_set_identity",
]

COMPONENTS = ("renderer_version", "ffmpeg_version", "font_set")
"""The fingerprint components in report order."""

_FFMPEG_VERSION = re.compile(r"version\s+(\S+)")
_FFMPEG_DEADLINE = 10


@lru_cache(maxsize=8)
def font_set_identity(fonts_dir: str | Path | None = None) -> str:
    """Return a stable digest of every font file below *fonts_dir*.

    The digest covers each file's path relative to the root and its bytes, so
    adding, removing or replacing a font changes the identity. The packaged
    font directory is used when *fonts_dir* is omitted. A missing or empty
    directory has no identity and returns an empty string.
    """
    root = Path(fonts_dir) if fonts_dir is not None else paths.package_fonts_dir()
    if not root.is_dir():
        return ""
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        return ""
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@lru_cache(maxsize=8)
def ffmpeg_version(executable: str = "ffmpeg") -> str:
    """Return the version reported by *executable*, or an empty string.

    A missing ffmpeg is not an error here: the fingerprint simply records that
    the build is unknown so a caller can decide whether that matters.
    """
    try:
        result = subprocess.run(
            [executable, "-version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=_FFMPEG_DEADLINE,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    first_line = result.stdout.splitlines()[0] if result.stdout else ""
    match = _FFMPEG_VERSION.search(first_line)
    if match is not None:
        return match.group(1)
    return first_line.strip()


def detect_toolchain(
    *,
    fonts_dir: str | Path | None = None,
    executable: str = "ffmpeg",
    renderer_version: str | None = None,
) -> ToolchainFingerprint:
    """Return the fingerprint of the renderer available to this process.

    *renderer_version* overrides the packaged version, which is the renderer
    identity until a renderer module supplies its own. *fonts_dir* and
    *executable* make detection testable without touching the real toolchain.
    """
    return ToolchainFingerprint(
        renderer_version=__version__ if renderer_version is None else renderer_version,
        ffmpeg_version=ffmpeg_version(executable),
        font_set=font_set_identity(fonts_dir),
    )


@dataclass(frozen=True, slots=True)
class ToolchainDrift:
    """One fingerprint component that differs between two runs."""

    component: str
    original: str
    replay: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable representation of the difference."""
        return {
            "component": self.component,
            "original": self.original,
            "replay": self.replay,
        }


def diff_toolchain(
    recorded: ToolchainFingerprint,
    current: ToolchainFingerprint,
) -> tuple[ToolchainDrift, ...]:
    """Compare two fingerprints component by component.

    An unknown recorded fingerprint cannot establish a baseline, so it yields
    no drift. A component the original never recorded is likewise skipped;
    a component that was recorded and has since changed or disappeared is
    reported.
    """
    if not recorded.is_known:
        return ()
    drifts: list[ToolchainDrift] = []
    for component in COMPONENTS:
        before = getattr(recorded, component)
        after = getattr(current, component)
        if before and before != after:
            drifts.append(
                ToolchainDrift(component=component, original=before, replay=after)
            )
    return tuple(drifts)
