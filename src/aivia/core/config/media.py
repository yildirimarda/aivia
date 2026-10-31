"""Packaged defaults for downloading and validating generated media.

These values live with the rest of the tunable limits rather than inside the
probe helper. Callers can override any of them per download with
:class:`aivia.core.media_probe.MediaLimits`; the constants here are only the
safe defaults used when no override is supplied.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_MEDIA_ALIGNMENT_SIMILARITY_THRESHOLD",
    "DEFAULT_MEDIA_ANIMATED_FRAME_RATE",
    "DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS",
    "DEFAULT_MEDIA_FFPROBE_TIMEOUT_SECONDS",
    "DEFAULT_MEDIA_MAX_BYTES",
    "DEFAULT_MEDIA_MAX_PIXELS",
    "DEFAULT_MEDIA_MIN_ANIMATED_FRAME_RATE",
    "DEFAULT_MEDIA_SUBTITLE_MAX_CHARS",
    "DEFAULT_MEDIA_SUBTITLE_MAX_DURATION_SECONDS",
    "DEFAULT_MEDIA_SUBTITLE_MAX_GAP_SECONDS",
    "DEFAULT_MUSIC_CLIPPING_PEAK_DB",
    "DEFAULT_MUSIC_DURATION_TOLERANCE_SECONDS",
    "DEFAULT_MUSIC_LIBRARY_TEMPO_BUCKET_BPM",
    "DEFAULT_MUSIC_LIBRARY_TEMPO_TOLERANCE_BPM",
    "DEFAULT_MUSIC_MAX_CLIPPING_RATIO",
    "DEFAULT_MUSIC_MAX_LOUDNESS_LUFS",
    "DEFAULT_MUSIC_MAX_SILENCE_RATIO",
    "DEFAULT_MUSIC_MIN_DURATION_SECONDS",
    "DEFAULT_MUSIC_MIN_LOUDNESS_LUFS",
    "DEFAULT_MUSIC_SILENCE_MIN_SECONDS",
    "DEFAULT_MUSIC_SILENCE_NOISE_DB",
    "MAX_MEDIA_SIMILARITY",
    "MIN_MEDIA_SIMILARITY",
    "MIN_MUSIC_LIBRARY_LIMIT",
    "MUSIC_LIBRARY_DURATION_WEIGHT",
    "MUSIC_LIBRARY_GENRE_WEIGHT",
    "MUSIC_LIBRARY_LICENSE_WEIGHT",
    "MUSIC_LIBRARY_MOOD_WEIGHT",
    "MUSIC_LIBRARY_TAG_WEIGHT",
    "MUSIC_LIBRARY_TEMPO_WEIGHT",
]

DEFAULT_MEDIA_ALIGNMENT_SIMILARITY_THRESHOLD = 0.8
"""Word similarity at or above this pairs a script word with a transcript word."""

MAX_MEDIA_SIMILARITY = 1.0
"""The highest possible normalized similarity score."""

MIN_MEDIA_SIMILARITY = 0.0
"""The lowest possible normalized similarity score."""

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

DEFAULT_MEDIA_SUBTITLE_MAX_CHARS = 42
"""Longest cue text before the next word starts a new subtitle cue."""

DEFAULT_MEDIA_SUBTITLE_MAX_DURATION_SECONDS = 6.0
"""Longest cue duration before the next word starts a new subtitle cue."""

DEFAULT_MEDIA_SUBTITLE_MAX_GAP_SECONDS = 1.0
"""A pause longer than this starts a new subtitle cue."""

DEFAULT_MUSIC_MIN_DURATION_SECONDS = 0.0
"""The shortest music candidate accepted when no target duration is supplied."""

DEFAULT_MUSIC_DURATION_TOLERANCE_SECONDS = 2.0
"""How far a candidate may differ from a requested duration."""

DEFAULT_MUSIC_MAX_SILENCE_RATIO = 0.35
"""The largest share of a music candidate that may be silence."""

DEFAULT_MUSIC_MAX_CLIPPING_RATIO = 0.001
"""The largest share of samples at full scale before a candidate is clipped."""

DEFAULT_MUSIC_MIN_LOUDNESS_LUFS = -30.0
"""The quietest integrated loudness a music candidate may have."""

DEFAULT_MUSIC_MAX_LOUDNESS_LUFS = -6.0
"""The loudest integrated loudness a music candidate may have."""

DEFAULT_MUSIC_SILENCE_NOISE_DB = -50.0
"""The noise floor below which ffmpeg counts a stretch as silence."""

DEFAULT_MUSIC_SILENCE_MIN_SECONDS = 0.5
"""The shortest pause ffmpeg reports as silence."""

DEFAULT_MUSIC_CLIPPING_PEAK_DB = 0.0
"""The peak level at or above which a sample counts as clipped."""

DEFAULT_MUSIC_LIBRARY_TEMPO_TOLERANCE_BPM = 5.0
"""How far a library track tempo may differ from a requested tempo, in BPM."""

DEFAULT_MUSIC_LIBRARY_TEMPO_BUCKET_BPM = 10.0
"""The width of one tempo bucket in the library tempo index, in BPM."""

MIN_MUSIC_LIBRARY_LIMIT = 1
"""The smallest number of ranked library matches a caller may request."""

MUSIC_LIBRARY_MOOD_WEIGHT = 4.0
"""The rank bonus for a matching library mood."""

MUSIC_LIBRARY_GENRE_WEIGHT = 3.0
"""The rank bonus for a matching library genre."""

MUSIC_LIBRARY_TAG_WEIGHT = 2.0
"""The rank bonus for each matching library tag."""

MUSIC_LIBRARY_TEMPO_WEIGHT = 2.0
"""The maximum rank bonus for a matching library tempo."""

MUSIC_LIBRARY_DURATION_WEIGHT = 3.0
"""The maximum rank bonus for a matching library duration."""

MUSIC_LIBRARY_LICENSE_WEIGHT = 1.0
"""The rank bonus for a matching library license filter."""
