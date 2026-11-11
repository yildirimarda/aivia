"""Description chapters from a transcript, validated against platform rules.

Every platform specific value (chapter cap, minimum chapter duration, chapter
title length and the description limit) is read from the packaged
``platforms.yaml`` through :func:`load_platforms`. The test code contains no
platform data literals.
"""

from __future__ import annotations

from collections.abc import Sequence

from aivia.core.errors import ContentPolicyError
from aivia.core.platforms import Platform, PlatformCatalog, load_platforms
from aivia.providers.protocols.audio import WordTimestamp
from aivia.providers.protocols.stt import STTResponse
from aivia.publish import ChapterGenerator, ChapterSet

PLATFORMS = ("x", "youtube", "instagram", "tiktok", "telegram")


def _transcript(starts: Sequence[float], duration_seconds: float) -> STTResponse:
    words = [
        WordTimestamp(
            word=f"word {index}", start_seconds=start, end_seconds=start + 0.5
        )
        for index, start in enumerate(starts)
    ]
    return STTResponse(
        text=" ".join(f"word {index}" for index in range(len(starts))),
        words=words,
        duration_seconds=duration_seconds,
    )


def _platform(name: str) -> Platform:
    return load_platforms().require(name)


def _catalog(
    *,
    max_description_length: int | None = None,
    **chapter_rules: object,
) -> PlatformCatalog:
    caption_limits: dict[str, object] = {}
    if max_description_length is not None:
        caption_limits["max_description_length"] = max_description_length
    payload: dict[str, object] = {
        "platforms": {
            "test": {
                "name": "Test",
                "media_types": ["video"],
                "caption_limits": caption_limits,
                "chapter_rules": chapter_rules,
            }
        }
    }
    return PlatformCatalog.from_mapping(payload)


def test_chapter_rules_are_loaded_from_platform_data() -> None:
    for name in PLATFORMS:
        rules = _platform(name).chapter_rules
        assert rules.enabled is True
        assert rules.max_chapters is not None and rules.max_chapters > 0
        assert rules.max_chapter_title_length is not None


def test_generates_chapters_from_timed_transcript() -> None:
    platform = _platform("youtube")
    transcript = _transcript([0.0, 60.0, 120.0], 180.0)
    result = ChapterGenerator(platform).build(transcript)

    assert isinstance(result, ChapterSet)
    assert result.platform == "youtube"
    assert [chapter.start_seconds for chapter in result.chapters] == [0.0, 60.0, 120.0]
    assert all(chapter.title for chapter in result.chapters)
    assert result.description
    assert "0:00" in result.description
    assert not result.truncated


def test_empty_transcript_yields_no_chapters() -> None:
    platform = _platform("youtube")
    transcript = _transcript([], 10.0)
    result = ChapterGenerator(platform).build(transcript)

    assert result.chapters == []
    assert result.description == ""
    assert not result.truncated


def test_chapter_count_respects_platform_max_chapters() -> None:
    platform = _catalog(max_chapters=2).require("test")
    transcript = _transcript([0.0, 60.0, 120.0], 180.0)
    result = ChapterGenerator(platform).build(transcript)

    assert len(result.chapters) <= 2
    assert "chapters capped to max_chapters" in result.warnings
    assert result.truncated


def test_chapter_count_violation_raises_in_strict() -> None:
    platform = _catalog(max_chapters=2).require("test")
    transcript = _transcript([0.0, 60.0, 120.0], 180.0)
    generator = ChapterGenerator(platform)

    try:
        generator.build_strict(transcript)
    except ContentPolicyError as exc:
        assert "too many chapters" in str(exc).lower()
        assert exc.details["platform"] == "test"
    else:
        raise AssertionError("build_strict did not raise for too many chapters")


def test_min_chapter_duration_drops_short_chapters() -> None:
    # YouTube defaults to a 30 second minimum chapter duration.
    platform = _platform("youtube")
    transcript = _transcript([0.0, 60.0], 61.0)
    result = ChapterGenerator(platform).build(transcript)

    # The final chapter is only one second long and is dropped.
    assert len(result.chapters) == 1
    assert result.chapters[0].start_seconds == 0.0


def test_min_chapter_duration_violation_raises_in_strict() -> None:
    platform = _platform("youtube")
    transcript = _transcript([0.0, 60.0], 61.0)
    generator = ChapterGenerator(platform)

    try:
        generator.build_strict(transcript)
    except ContentPolicyError as exc:
        assert "minimum duration" in str(exc).lower()
    else:
        raise AssertionError("build_strict did not raise for short chapter")


def test_chapter_titles_respect_the_platform_title_limit() -> None:
    platform = _catalog(max_chapter_title_length=5).require("test")
    long_title = "this is a very long opening sentence for the chapter"
    transcript = STTResponse(
        text=long_title,
        words=[WordTimestamp(word=long_title, start_seconds=0.0, end_seconds=1.0)],
        duration_seconds=60.0,
    )
    result = ChapterGenerator(platform).build(transcript)

    assert len(result.chapters) == 1
    assert len(result.chapters[0].title) <= 5


def test_description_fits_the_platform_description_limit() -> None:
    platform = _catalog(max_description_length=40).require("test")
    transcript = _transcript([0.0, 30.0, 60.0, 90.0], 120.0)
    result = ChapterGenerator(platform).build(transcript)

    assert len(result.description) <= 40


def test_description_limit_violation_raises_in_strict() -> None:
    platform = _catalog(max_description_length=5).require("test")
    transcript = _transcript([0.0], 10.0)
    generator = ChapterGenerator(platform)

    try:
        generator.build_strict(transcript)
    except ContentPolicyError as exc:
        assert "description" in str(exc).lower()
    else:
        raise AssertionError("build_strict did not raise for long description")


def test_disabled_chapter_rules_yield_no_chapters() -> None:
    platform = _catalog(enabled=False).require("test")
    transcript = _transcript([0.0, 60.0], 120.0)
    result = ChapterGenerator(platform).build(transcript)

    assert result.chapters == []
    assert result.description == ""
