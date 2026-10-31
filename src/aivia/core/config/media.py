"""Packaged defaults for downloading and validating generated media.

These values live with the rest of the tunable limits rather than inside the
probe helper. Callers can override any of them per download with
:class:`aivia.core.media_probe.MediaLimits`; the constants here are only the
safe defaults used when no override is supplied.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_MEDIA_ANIMATED_FRAME_RATE",
    "DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS",
    "DEFAULT_MEDIA_FFPROBE_TIMEOUT_SECONDS",
    "DEFAULT_MEDIA_MAX_BYTES",
    "DEFAULT_MEDIA_MAX_PIXELS",
    "DEFAULT_MEDIA_MIN_ANIMATED_FRAME_RATE",
]

DEFAULT_MEDIA_MAX_BYTES = 100 * 1024 * 1024
"""Default cap for one downloaded asset, in bytes."""

DEFAULT_MEDIA_MAX_PIXELS = 100_000_000
"""Default decompression-bomb guard: pixels an image may declare."""

DEFAULT_MEDIA_FFPROBE_TIMEOUT_SECONDS = 10.0
"""Seconds ffprobe may run before it is treated as a validation failure."""

DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS = 300.0
"""Seconds a deterministic ffmpeg transform may run before it fails."""

DEFAULT_MEDIA_ANIMATED_FRAME_RATE = 15.0
"""Default frame rate for GIF and animated WebP output."""

DEFAULT_MEDIA_MIN_ANIMATED_FRAME_RATE = 8.0
"""Lowest frame rate a size-capped animated output may fall back to."""
