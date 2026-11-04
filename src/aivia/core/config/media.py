"""Packaged defaults for downloading and validating generated media.

These values live with the rest of the tunable limits rather than inside the
probe helper. Callers can override any of them per download with
:class:`aivia.core.media_probe.MediaLimits`; the constants here are only the
safe defaults used when no override is supplied.
"""

from __future__ import annotations

__all__ = [
    "AUDIO_QA_TIME_TOLERANCE_SECONDS",
    "DEFAULT_AUDIO_ANALYSIS_MAX_BPM",
    "DEFAULT_AUDIO_ANALYSIS_MIN_BPM",
    "DEFAULT_AUDIO_DUCKING_ATTACK_MS",
    "DEFAULT_AUDIO_DUCKING_RATIO",
    "DEFAULT_AUDIO_DUCKING_RELEASE_MS",
    "DEFAULT_AUDIO_DUCKING_THRESHOLD_DB",
    "DEFAULT_AUDIO_MIX_CHANNELS",
    "DEFAULT_AUDIO_MIX_SAMPLE_RATE",
    "DEFAULT_AUDIO_QA_EXPECTED_WORDS_PER_SECOND",
    "DEFAULT_AUDIO_QA_MAX_INTERNAL_SILENCE_SECONDS",
    "DEFAULT_AUDIO_QA_MAX_LEADING_SILENCE_SECONDS",
    "DEFAULT_AUDIO_QA_MAX_MISPRONUNCIATION_RATIO",
    "DEFAULT_AUDIO_QA_MAX_SILENCE_RATIO",
    "DEFAULT_AUDIO_QA_MAX_TRAILING_SILENCE_SECONDS",
    "DEFAULT_AUDIO_QA_MIN_DURATION_RATIO",
    "DEFAULT_AUDIO_QA_MIN_WORD_CONFIDENCE",
    "DEFAULT_AUDIO_QA_SILENT_RATIO",
    "DEFAULT_MEDIA_ALIGNMENT_SIMILARITY_THRESHOLD",
    "DEFAULT_MEDIA_ANIMATED_FRAME_RATE",
    "DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS",
    "DEFAULT_MEDIA_FFPROBE_TIMEOUT_SECONDS",
    "DEFAULT_MEDIA_LOOP_SIMILARITY_THRESHOLD",
    "DEFAULT_MEDIA_MAX_ANIMATED_FRAMES",
    "DEFAULT_MEDIA_MAX_ANIMATED_PIXELS",
    "DEFAULT_MEDIA_MAX_BYTES",
    "DEFAULT_MEDIA_MAX_PIXELS",
    "DEFAULT_MEDIA_MIN_ANIMATED_FRAME_RATE",
    "DEFAULT_MEDIA_PREVIEW_MAX_FPS",
    "DEFAULT_MEDIA_PREVIEW_SCALE",
    "DEFAULT_MEDIA_SUBTITLE_MAX_CHARS",
    "DEFAULT_MEDIA_SUBTITLE_MAX_DURATION_SECONDS",
    "DEFAULT_MEDIA_SUBTITLE_MAX_GAP_SECONDS",
    "DEFAULT_MEDIA_VOICE_GAP_SECONDS",
    "DEFAULT_MUSIC_BACKGROUND_FADE_IN_SECONDS",
    "DEFAULT_MUSIC_BACKGROUND_FADE_OUT_SECONDS",
    "DEFAULT_MUSIC_BACKGROUND_VOLUME_DB",
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
    "MAX_AUDIO_QA_CONFIDENCE",
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

DEFAULT_AUDIO_ANALYSIS_MIN_BPM = 40.0
"""The slowest tempo the audio analyser reports, in beats per minute."""

DEFAULT_AUDIO_ANALYSIS_MAX_BPM = 240.0
"""The fastest tempo the audio analyser reports, in beats per minute."""

DEFAULT_AUDIO_DUCKING_THRESHOLD_DB = -25.0
"""The voice level at which the sidechain compressor starts ducking."""

DEFAULT_AUDIO_DUCKING_RATIO = 8.0
"""How strongly the music bed is compressed while a voice is present."""

DEFAULT_AUDIO_DUCKING_ATTACK_MS = 20.0
"""Milliseconds the ducking compressor takes to react to a voice."""

DEFAULT_AUDIO_DUCKING_RELEASE_MS = 300.0
"""Milliseconds the music bed takes to return after a voice stops."""

DEFAULT_AUDIO_MIX_SAMPLE_RATE = 48_000
"""Default sample rate for a mixed audio output."""

DEFAULT_AUDIO_MIX_CHANNELS = 2
"""Default channel count for a mixed audio output."""

DEFAULT_MUSIC_BACKGROUND_VOLUME_DB = -6.0
"""Default gain applied to a background music bed before ducking."""

DEFAULT_MUSIC_BACKGROUND_FADE_IN_SECONDS = 1.0
"""Default fade in for a background music bed."""

DEFAULT_MUSIC_BACKGROUND_FADE_OUT_SECONDS = 2.0
"""Default fade out for a background music bed."""

AUDIO_QA_TIME_TOLERANCE_SECONDS = 0.05
"""How far a word timing may pass a segment end before it is an error."""

MAX_AUDIO_QA_CONFIDENCE = 1.0
"""The highest word confidence a provider may report."""

DEFAULT_AUDIO_QA_MAX_INTERNAL_SILENCE_SECONDS = 1.0
"""An internal pause longer than this is suspicious."""

DEFAULT_AUDIO_QA_MAX_LEADING_SILENCE_SECONDS = 0.75
"""A voice segment should normally start speaking before this."""

DEFAULT_AUDIO_QA_MAX_TRAILING_SILENCE_SECONDS = 1.0
"""A voice segment should normally stop speaking before this."""

DEFAULT_AUDIO_QA_MAX_SILENCE_RATIO = 0.5
"""Above this share of silence the segment is flagged."""

DEFAULT_AUDIO_QA_SILENT_RATIO = 0.9
"""At or above this share of silence the segment is treated as silent."""

DEFAULT_AUDIO_QA_MIN_WORD_CONFIDENCE = 0.5
"""A word below this provider confidence may be mispronounced."""

DEFAULT_AUDIO_QA_MAX_MISPRONUNCIATION_RATIO = 0.5
"""Above this share of wrong words the finding becomes an error."""

DEFAULT_AUDIO_QA_EXPECTED_WORDS_PER_SECOND = 2.5
"""A fallback speaking rate used when no word timing is available."""

DEFAULT_AUDIO_QA_MIN_DURATION_RATIO = 0.4
"""Below this share of the expected duration the audio is truncated."""

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

DEFAULT_MEDIA_MAX_ANIMATED_FRAMES = 500
"""Default cap on the number of frames an animated image may contain."""

DEFAULT_MEDIA_MAX_ANIMATED_PIXELS = 250_000_000
"""Default cap on decoded pixels summed across every animation frame."""

DEFAULT_MEDIA_FFPROBE_TIMEOUT_SECONDS = 10.0
"""Seconds ffprobe may run before it is treated as a validation failure."""

DEFAULT_MEDIA_LOOP_SIMILARITY_THRESHOLD = 0.98
"""First and last loop frames at or above this similarity form a seamless seam."""

DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS = 300.0
"""Seconds a deterministic ffmpeg transform may run before it fails."""

DEFAULT_MEDIA_ANIMATED_FRAME_RATE = 15.0
"""Default frame rate for GIF and animated WebP output."""

DEFAULT_MEDIA_MIN_ANIMATED_FRAME_RATE = 8.0
"""Lowest frame rate a size-capped animated output may fall back to."""

DEFAULT_MEDIA_PREVIEW_SCALE = 0.5
"""Fraction applied to preview canvas dimensions and frame rate."""

DEFAULT_MEDIA_PREVIEW_MAX_FPS = 15
"""Frame rate a preview never exceeds unless the caller asks for more."""

DEFAULT_MEDIA_SUBTITLE_MAX_CHARS = 42
"""Longest cue text before the next word starts a new subtitle cue."""

DEFAULT_MEDIA_SUBTITLE_MAX_DURATION_SECONDS = 6.0
"""Longest cue duration before the next word starts a new subtitle cue."""

DEFAULT_MEDIA_SUBTITLE_MAX_GAP_SECONDS = 1.0
"""A pause longer than this starts a new subtitle cue."""

DEFAULT_MEDIA_VOICE_GAP_SECONDS = 0.0
"""Silence inserted between voiced beats unless a caller configures one."""

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
