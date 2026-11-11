"""Captions, hashtags, titles and disclosure formatted from platform data.

Every platform specific value (caption limits, disclosure text, hashtag cap) is
read from the packaged ``platforms.yaml`` through :func:`load_platforms`. The
test code contains no platform data literals.
"""

from __future__ import annotations

from aivia.core.platforms import Platform, load_platforms
from aivia.writing import (
    Beat,
    CaptionSet,
    FactsPayload,
    Script,
    ScriptFact,
    build_captions,
)

PLATFORMS = ("x", "youtube", "instagram", "tiktok", "telegram")


def _platform(name: str) -> Platform:
    return load_platforms().require(name)


def _script(*, title: str = "The secret life of bees") -> Script:
    return Script(
        title=title,
        language="en",
        hook="Bees can recognise human faces.",
        beats=[
            Beat(index=0, text="Honey never spoils."),
            Beat(index=1, text="Bees can recognise human faces."),
        ],
        outro="That is the story of bees.",
        payload=FactsPayload(
            facts=[
                ScriptFact(
                    text="Honey never spoils.",
                    source_urls=["https://example.com/honey"],
                ),
                ScriptFact(
                    text="Bees can recognise human faces.",
                    source_urls=[
                        "https://example.com/faces",
                        "https://example.com/honey",
                    ],
                ),
            ],
            source_urls=["https://example.com/honey", "https://example.com/faces"],
        ),
    )


def test_caption_title_description_and_hashtags_are_limited_by_platform_data() -> None:
    for name in PLATFORMS:
        platform = _platform(name)
        result = build_captions(_script(), name)

        assert isinstance(result, CaptionSet)
        assert result.platform == name
        assert result.disclosure_required is platform.disclosure.required
        assert result.disclosure == platform.disclosure.text
        if platform.disclosure.text:
            assert result.disclosure in result.caption
        limits = platform.caption_limits
        if limits.max_length is not None:
            assert len(result.caption) <= limits.max_length
        if limits.max_title_length is not None:
            assert len(result.title) <= limits.max_title_length
        if limits.max_description_length is not None:
            assert len(result.description) <= limits.max_description_length
        if limits.max_hashtags is not None:
            assert len(result.hashtags) <= limits.max_hashtags


def test_limits_recorded_on_the_result_come_from_platform_data() -> None:
    for name in PLATFORMS:
        platform = _platform(name)
        result = build_captions(_script(), name)

        assert result.limits["max_length"] == platform.caption_limits.max_length
        assert result.limits["max_hashtags"] == platform.caption_limits.max_hashtags
        assert (
            result.limits["max_title_length"]
            == platform.caption_limits.max_title_length
        )
        assert (
            result.limits["max_description_length"]
            == platform.caption_limits.max_description_length
        )


def test_hashtag_count_respects_the_platform_limit_from_data() -> None:
    for name in PLATFORMS:
        platform = _platform(name)
        limit = platform.caption_limits.max_hashtags
        if limit is None:
            continue
        result = build_captions(
            _script(),
            name,
            hashtags=[f"tag {i}" for i in range(limit + 5)],
        )
        assert len(result.hashtags) <= limit


def test_title_truncation_uses_the_platform_limit_from_data() -> None:
    name = next(
        (
            platform_id
            for platform_id in load_platforms().ids()
            if load_platforms().require(platform_id).caption_limits.max_title_length
        ),
        None,
    )
    assert name is not None
    platform = _platform(name)
    limit = platform.caption_limits.max_title_length
    assert limit is not None
    long_title = "x" * (limit + 10)
    result = build_captions(_script(title=long_title), name)
    assert len(result.title) <= limit
    assert result.title != long_title
    assert "title truncated" in result.warnings


def test_caption_truncation_uses_the_platform_limit_from_data() -> None:
    platform = _platform("x")
    limit = platform.caption_limits.max_length
    assert limit is not None
    long_title = "x" * (limit + 100)
    result = build_captions(_script(title=long_title), "x")
    assert len(result.caption) <= limit
    assert "caption truncated" in result.warnings
