"""Voice synthesis, timeline building and the audio stages.

The package owns the voice stage that turns written scripts into measured
speech and a timeline file. It consumes the core script model, the compose
timeline schema and the media speech service; it does not select providers or
write cost events itself.
"""

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
    "DEFAULT_TIMELINE_FILENAME",
    "SYNTHESIS_MODES",
    "VOICE_STAGE_NAME",
    "Voice",
    "VoiceStage",
    "VoiceStageResult",
    "build_voice_stage",
    "run_voice_stage",
    "voice_stage",
]
