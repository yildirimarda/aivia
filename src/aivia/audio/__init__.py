"""Voice synthesis, subtitles, timeline building and the audio stages.

The package owns the voice stage that turns written scripts into measured
speech and a timeline file, and the subtitle stage that turns that timeline
into one styled ASS file per language. It consumes the core script model, the
compose timeline schema and the media services; it does not select providers
or write cost events itself.
"""

from aivia.audio.subtitle_stage import (
    DEFAULT_SUBTITLE_FILENAME_PATTERN,
    SUBTITLE_STAGE_NAME,
    Subtitle,
    SubtitlesStage,
    SubtitleStage,
    SubtitleStageResult,
    build_subtitle_stage,
    run_subtitle_stage,
    subtitle_stage,
)
from aivia.audio.voice_stage import (
    DEFAULT_GAP_SECONDS,
    DEFAULT_SCRIPT_FILENAME_PATTERN,
    DEFAULT_TIMELINE_FILENAME,
    SYNTHESIS_MODES,
    VOICE_STAGE_NAME,
    Voice,
    VoiceStage,
    VoiceStageResult,
    build_voice_stage,
    run_voice_stage,
    voice_stage,
)

__all__ = [
    "DEFAULT_GAP_SECONDS",
    "DEFAULT_SCRIPT_FILENAME_PATTERN",
    "DEFAULT_SUBTITLE_FILENAME_PATTERN",
    "DEFAULT_TIMELINE_FILENAME",
    "SUBTITLE_STAGE_NAME",
    "SYNTHESIS_MODES",
    "VOICE_STAGE_NAME",
    "Subtitle",
    "SubtitleStage",
    "SubtitleStageResult",
    "SubtitlesStage",
    "Voice",
    "VoiceStage",
    "VoiceStageResult",
    "build_subtitle_stage",
    "build_voice_stage",
    "run_subtitle_stage",
    "run_voice_stage",
    "subtitle_stage",
    "voice_stage",
]
