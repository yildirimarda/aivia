"""Personalized order lyrics: name in chorus and hook, optional age and interests."""

from __future__ import annotations

import pytest

from aivia.core.errors import ConfigError
from aivia.writing import (
    LyricsPayload,
    Script,
    generate_personalized_lyrics,
)


def _payload(script: Script) -> LyricsPayload:
    assert isinstance(script.payload, LyricsPayload)
    return script.payload


def _chorus_lines(script: Script) -> list[str]:
    payload = _payload(script)
    for section in payload.sections:
        if section.name.casefold() == "chorus":
            return list(section.lines)
    raise AssertionError("no chorus section")


def _all_lines(script: Script) -> list[str]:
    return [line for section in _payload(script).sections for line in section.lines]


def test_name_appears_in_chorus_and_hook() -> None:
    script = generate_personalized_lyrics("Ava")

    assert "Ava" in script.hook
    assert script.hook == "Ava, Ava, hear the cheer"
    assert any("Ava" in line for line in _chorus_lines(script))
    assert all("Ava" in line for line in _chorus_lines(script))


def test_name_appears_in_the_verse_as_well() -> None:
    script = generate_personalized_lyrics("Ava")

    assert any("Ava" in line for line in _all_lines(script))


def test_optional_age_is_woven_into_the_verse() -> None:
    with_age = generate_personalized_lyrics("Ava", age=30)
    without_age = generate_personalized_lyrics("Ava")

    with_lines = _all_lines(with_age)
    without_lines = _all_lines(without_age)
    assert any("30" in line for line in with_lines)
    assert not any("30" in line for line in without_lines)
    assert len(with_lines) == len(without_lines) + 1


def test_optional_interests_are_woven_into_the_verse() -> None:
    script = generate_personalized_lyrics("Ava", interests=["dancing", "music"])

    lines = _all_lines(script)
    assert any("dancing" in line and "music" in line for line in lines)

    plain = generate_personalized_lyrics("Ava")
    plain_lines = _all_lines(plain)
    assert not any("dancing" in line for line in plain_lines)


def test_language_is_recorded_on_the_script() -> None:
    script = generate_personalized_lyrics("Ava", language="de")

    assert script.language == "de"


def test_tone_shapes_the_opening_line() -> None:
    cheerful = generate_personalized_lyrics("Ava", tone="cheerful")
    calm = generate_personalized_lyrics("Ava", tone="calm")
    energetic = generate_personalized_lyrics("Ava", tone="energetic")

    cheerful_lines = _all_lines(cheerful)
    calm_lines = _all_lines(calm)
    energetic_lines = _all_lines(energetic)
    assert any("bright day" in line for line in cheerful_lines)
    assert any("quiet day" in line for line in calm_lines)
    assert any("big day" in line for line in energetic_lines)
    assert cheerful_lines[0] != calm_lines[0]


def test_duration_sets_a_per_line_syllable_budget() -> None:
    short = generate_personalized_lyrics("Ava", duration_seconds=15)
    long_ = generate_personalized_lyrics("Ava", duration_seconds=60)
    none_ = generate_personalized_lyrics("Ava")

    short_budget = _payload(short).syllable_budget
    long_budget = _payload(long_).syllable_budget
    assert short_budget is not None
    assert long_budget is not None
    assert long_budget > short_budget
    assert _payload(none_).syllable_budget is None


def test_zero_or_negative_duration_leaves_no_budget() -> None:
    for duration in (0, -1, -5.5):
        script = generate_personalized_lyrics("Ava", duration_seconds=duration)
        assert _payload(script).syllable_budget is None


def test_empty_name_is_rejected() -> None:
    for name in ("", "   ", "\t\n"):
        with pytest.raises(ConfigError, match="name must be non-empty"):
            generate_personalized_lyrics(name)


def test_output_is_deterministic() -> None:
    first = generate_personalized_lyrics(
        "Ava",
        age=30,
        interests=["dancing", "music"],
        language="de",
        tone="cheerful",
        duration_seconds=30,
    )
    second = generate_personalized_lyrics(
        "Ava",
        age=30,
        interests=["dancing", "music"],
        language="de",
        tone="cheerful",
        duration_seconds=30,
    )

    assert first.mode == second.mode
    assert first.title == second.title
    assert first.language == second.language
    assert first.hook == second.hook
    assert first.outro == second.outro
    assert [beat.text for beat in first.beats] == [beat.text for beat in second.beats]
    assert first.payload == second.payload


def test_output_round_trips_through_json() -> None:
    script = generate_personalized_lyrics("Ava", age=25, interests=["art"])

    round_tripped = Script.model_validate_json(script.to_json())
    assert round_tripped == script
    assert round_tripped.hook == script.hook
    assert round_tripped.language == script.language


def test_chorus_and_hook_are_distinct() -> None:
    script = generate_personalized_lyrics("Ava")

    assert script.hook != ""
    assert script.hook not in _chorus_lines(script)


def test_hook_contains_the_name_twice() -> None:
    script = generate_personalized_lyrics("Ava")

    assert script.hook.count("Ava") == 2
