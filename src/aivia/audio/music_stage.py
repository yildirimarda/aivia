"""Compatibility import for the background music selection stage.

The implementation lives in :mod:`aivia.audio.music_select`; this module keeps
the stage naming used by the voice and subtitle stages importable as well.
"""

from __future__ import annotations

from aivia.audio.music_select import (
    DEFAULT_MUSIC_FILENAME,
    DEFAULT_TIMELINE_FILENAME,
    MUSIC_STAGE_NAME,
    SOURCE_MODES,
    BackgroundMusic,
    MusicChoice,
    MusicSelection,
    MusicSelectStage,
    MusicStage,
    SelectedMusic,
    apply_ducking,
    build_music_prompt,
    build_music_prompt_from_profile,
    build_music_stage,
    build_profile_music_prompt,
    music_prompt,
    music_prompt_for_profile,
    music_prompt_from_profile,
    music_stage,
    profile_music_prompt,
    run_music_stage,
    select_library_music,
    select_music,
    verify_timeline_ducking,
)

__all__ = [
    "DEFAULT_MUSIC_FILENAME",
    "DEFAULT_TIMELINE_FILENAME",
    "MUSIC_STAGE_NAME",
    "SOURCE_MODES",
    "BackgroundMusic",
    "MusicChoice",
    "MusicSelectStage",
    "MusicSelection",
    "MusicStage",
    "SelectedMusic",
    "apply_ducking",
    "build_music_prompt",
    "build_music_prompt_from_profile",
    "build_music_stage",
    "build_profile_music_prompt",
    "music_prompt",
    "music_prompt_for_profile",
    "music_prompt_from_profile",
    "music_stage",
    "profile_music_prompt",
    "run_music_stage",
    "select_library_music",
    "select_music",
    "verify_timeline_ducking",
]
