"""A timeline becomes one exact composition per language and platform."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aivia.compose import (
    AudioTrackRole,
    ImageLayer,
    Timeline,
    TimelineError,
    VideoLayer,
    compositions_by_language_and_platform,
    compositions_from_timeline,
    timeline_to_compositions,
    validate_composition,
)
from aivia.core.config import VideoConfig


def _video_config() -> VideoConfig:
    return VideoConfig.from_mapping(
        {
            "resolution": "1080x1920",
            "fps": 30,
            "loudness_lufs": -14.0,
            "duration_seconds": {"min": 20, "max": 45},
            "platforms": {
                "tiktok": {
                    "resolution": "720x1280",
                    "fps": 24,
                    "loudness_lufs": -16.0,
                    "duration_seconds": {"min": 1, "max": 5},
                },
                "youtube": {
                    "resolution": "1920x1080",
                    "fps": 60,
                    "loudness_lufs": -12.0,
                    "duration_seconds": {"min": 1, "max": 10},
                },
            },
        }
    )


def _timeline() -> Timeline:
    return Timeline.model_validate(
        {
            "id": "launch",
            "platforms": ["tiktok", "youtube"],
            "beats": [
                {
                    "id": "hook",
                    "start": 0,
                    "end": 2.5,
                    "asset": "hook",
                    "asset_kind": "image",
                },
                {
                    "id": "demo",
                    "start": 2.5,
                    "end": 6.0,
                    "asset": "demo",
                    "asset_kind": "video",
                },
            ],
            "assets": {
                "hook": "assets/hook.png",
                "demo": {"source": "assets/demo.mp4", "kind": "video"},
            },
            "music": [
                {
                    "id": "bed",
                    "source": "audio/bed.mp3",
                    "volume_db": -12,
                    "fade_in": 0.5,
                    "fade_out": 1.0,
                    "loop": True,
                }
            ],
            "languages": {
                "en": {
                    "voices": [
                        {
                            "id": "en_hook",
                            "beat": "hook",
                            "source": "audio/en_hook.wav",
                            "start": 0,
                            "end": 2.5,
                            "words": [
                                {"word": "Hello", "start": 0.1, "end": 0.4},
                            ],
                        },
                        {
                            "id": "en_demo",
                            "beat": "demo",
                            "source": "audio/en_demo.wav",
                            "start": 2.5,
                            "end": 6.0,
                        },
                    ],
                    "subtitles": {
                        "cues": [{"start": 0.1, "end": 2.4, "text": "Hello"}]
                    },
                },
                "de": {
                    "voices": [
                        {
                            "id": "de_hook",
                            "beat": "hook",
                            "source": "audio/de_hook.wav",
                            "start": 0,
                            "end": 2.5,
                        },
                        {
                            "id": "de_demo",
                            "beat": "demo",
                            "source": "audio/de_demo.wav",
                            "start": 2.5,
                            "end": 6.5,
                        },
                    ],
                    "subtitles": {
                        "cues": [{"start": 0.1, "end": 2.4, "text": "Hallo"}]
                    },
                },
            },
        }
    )


def test_one_composition_per_language_and_platform_keeps_the_exact_duration() -> None:
    compositions = timeline_to_compositions(_timeline(), video=_video_config())

    assert len(compositions) == 4
    indexed = compositions_by_language_and_platform(compositions)
    assert set(indexed) == {
        ("en", "tiktok"),
        ("en", "youtube"),
        ("de", "tiktok"),
        ("de", "youtube"),
    }

    en_tiktok = indexed[("en", "tiktok")]
    de_tiktok = indexed[("de", "tiktok")]
    en_youtube = indexed[("en", "youtube")]

    assert en_tiktok.duration_seconds == pytest.approx(6.0)
    assert de_tiktok.duration_seconds == pytest.approx(6.5)
    assert en_youtube.duration_seconds == pytest.approx(6.0)
    assert en_tiktok.id == "launch_en_tiktok"
    assert en_tiktok.language == en_tiktok.lang == "en"
    assert en_tiktok.platform == "tiktok"

    assert (en_tiktok.width, en_tiktok.height) == (720, 1280)
    assert en_tiktok.fps == 24
    assert en_tiktok.loudness_lufs == pytest.approx(-16.0)
    assert (en_youtube.width, en_youtube.height) == (1920, 1080)
    assert en_youtube.fps == 60
    assert en_youtube.loudness_lufs == pytest.approx(-12.0)


def test_beats_assets_voice_subtitles_and_music_become_composition_values() -> None:
    composition = compositions_by_language_and_platform(
        timeline_to_compositions(_timeline(), video=_video_config())
    )[("en", "tiktok")]

    assert [layer.id for layer in composition.layers] == ["hook", "demo"]
    hook, demo = composition.layers
    assert isinstance(hook, ImageLayer)
    assert isinstance(demo, VideoLayer)
    assert hook.source == "assets/hook.png"
    assert hook.start_seconds == 0.0
    assert hook.end_seconds == pytest.approx(2.5)
    assert demo.source == "assets/demo.mp4"
    assert demo.start_seconds == pytest.approx(2.5)
    assert demo.end_seconds == pytest.approx(6.0)
    assert demo.muted is True

    assert [track.id for track in composition.audio_tracks] == [
        "en_hook",
        "en_demo",
        "bed",
    ]
    first, second, music = composition.audio_tracks
    assert first.role is AudioTrackRole.VOICE
    assert first.source == "audio/en_hook.wav"
    assert first.start_seconds == 0.0
    assert first.duration_seconds == pytest.approx(2.5)
    assert second.role is AudioTrackRole.VOICE
    assert second.start_seconds == pytest.approx(2.5)
    assert second.duration_seconds == pytest.approx(3.5)
    assert music.role is AudioTrackRole.MUSIC
    assert music.volume_db == pytest.approx(-12.0)
    assert music.start_seconds == 0.0
    assert music.duration_seconds == pytest.approx(6.0)
    assert music.loop is True

    assert composition.subtitles is not None
    assert composition.subtitles.language == "en"
    assert composition.subtitles.text == "Hello"

    issues = validate_composition(composition, asset_exists=lambda source: True)
    assert issues.ok


def test_a_language_voice_can_extend_the_shared_visual_timeline() -> None:
    indexed = compositions_by_language_and_platform(
        timeline_to_compositions(_timeline(), video=_video_config())
    )

    assert indexed[("de", "tiktok")].duration_seconds == pytest.approx(6.5)
    assert indexed[("de", "tiktok")].audio_tracks[-1].duration_seconds == pytest.approx(
        6.5
    )
    assert indexed[("de", "youtube")].duration_seconds == pytest.approx(6.5)


def test_a_single_language_timeline_derives_subtitles_from_timed_words() -> None:
    timeline = Timeline.model_validate(
        {
            "id": "quote",
            "duration": 8,
            "language": "tr",
            "beats": [{"id": "only", "start": 0, "end": 4, "asset": "card"}],
            "assets": {"card": "assets/card.png"},
            "voices": [
                {
                    "beat": "only",
                    "source": "audio/tr.wav",
                    "start": 0,
                    "end": 4,
                    "words": [
                        {"word": "Merhaba", "start": 0.2, "end": 0.9},
                        {"word": "dünya", "start": 1.0, "end": 1.6},
                    ],
                }
            ],
            "music": [{"source": "audio/bed.mp3"}],
        }
    )

    compositions = compositions_from_timeline(timeline)
    assert len(compositions) == 1
    composition = compositions[0]

    assert composition.duration_seconds == pytest.approx(8.0)
    assert composition.language == "tr"
    assert composition.platform == ""
    assert composition.subtitles is not None
    assert composition.subtitles.language == "tr"
    assert composition.subtitles.text == "Merhaba dünya"
    assert composition.audio_tracks[-1].duration_seconds == pytest.approx(8.0)
    assert composition.id == "quote_tr"


def test_platform_argument_overrides_the_timeline_targets() -> None:
    compositions = timeline_to_compositions(
        _timeline(),
        video=_video_config(),
        platforms=["youtube"],
    )

    assert [(item.language, item.platform) for item in compositions] == [
        ("en", "youtube"),
        ("de", "youtube"),
    ]


def test_an_explicit_duration_shorter_than_the_content_is_rejected() -> None:
    timeline = Timeline.model_validate(
        {
            "id": "short",
            "duration": 2,
            "beats": [{"id": "one", "start": 0, "end": 3}],
            "voices": [
                {
                    "beat": "one",
                    "source": "audio/voice.wav",
                    "start": 0,
                    "end": 3,
                }
            ],
        }
    )

    with pytest.raises(TimelineError, match="shorter than its content"):
        timeline_to_compositions(timeline)


def test_a_voice_must_reference_an_existing_beat() -> None:
    with pytest.raises(ValidationError, match="unknown beat"):
        Timeline.model_validate(
            {
                "beats": [{"id": "one", "start": 0, "end": 1}],
                "voices": [{"beat": "missing", "source": "audio/voice.wav", "end": 1}],
            }
        )


def test_duplicate_language_entries_are_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate language"):
        Timeline.model_validate(
            {
                "languages": [
                    {"language": "en"},
                    {"language": "en"},
                ]
            }
        )


def test_a_timeline_without_timing_has_no_composition() -> None:
    with pytest.raises(TimelineError, match="no beats"):
        timeline_to_compositions(Timeline.model_validate({"id": "empty"}))
