"""The poem and story modes honour length, form, tone and reading time."""

from __future__ import annotations

import math

import pytest

from aivia.core.errors import ConfigError
from aivia.writing import (
    LENGTH_PRESETS,
    BuiltinWritingMode,
    LengthRange,
    Poem,
    PoemMode,
    PoemPayload,
    PoemWritingMode,
    Script,
    Story,
    StoryMode,
    StoryPayload,
    StoryWritingMode,
    WritingMode,
    build_poem_script,
    build_story_script,
    builtin_writing_mode,
    coerce_length_range,
    coerce_poem,
    coerce_story,
    count_words,
    extract_poem,
    extract_story,
    normalise_poem,
    normalise_story,
    normalize_poem,
    normalize_story,
    parse_poem,
    parse_story,
    poem_script,
    reading_time_seconds,
    resolve_mode,
    story_script,
)


def _poem_payload(script: Script) -> PoemPayload:
    assert isinstance(script.payload, PoemPayload)
    return script.payload


def _story_payload(script: Script) -> StoryPayload:
    assert isinstance(script.payload, StoryPayload)
    return script.payload


def _duration(script: Script) -> float:
    assert script.duration_seconds is not None
    return script.duration_seconds


def _seconds(payload: PoemPayload | StoryPayload) -> tuple[float, float]:
    assert payload.min_seconds is not None
    assert payload.max_seconds is not None
    return payload.min_seconds, payload.max_seconds


def test_poem_mode_replaces_the_builtin_placeholder() -> None:
    assert resolve_mode("poem") is PoemMode
    assert builtin_writing_mode(WritingMode.POEM) is PoemMode
    assert issubclass(PoemMode, BuiltinWritingMode)

    mode = PoemMode()

    assert mode.name == "poem"
    assert mode.mode == "poem"
    assert mode.payload_model is PoemPayload
    assert PoemWritingMode is PoemMode
    assert Poem is PoemMode


def test_story_mode_replaces_the_builtin_placeholder() -> None:
    assert resolve_mode("story") is StoryMode
    assert builtin_writing_mode(WritingMode.STORY) is StoryMode
    assert issubclass(StoryMode, BuiltinWritingMode)

    mode = StoryMode()

    assert mode.name == "story"
    assert mode.mode == "story"
    assert mode.payload_model is StoryPayload
    assert StoryWritingMode is StoryMode
    assert Story is StoryMode


def test_poem_mapping_builds_hook_beats_payload_and_duration() -> None:
    script = PoemMode().build_script(
        {
            "title": "Night",
            "language": "de",
            "form": "haiku",
            "tone": "calm",
            "lines": [
                "Silver moon above",
                "The quiet river answers",
                "A heron takes flight",
            ],
            "stanzas": [
                [
                    "Silver moon above",
                    "The quiet river answers",
                    "A heron takes flight",
                ]
            ],
            "rhyme_scheme": "abc",
            "syllable_budget": 17,
            "length": "short",
        }
    )

    assert isinstance(script, Script)
    assert script.mode is WritingMode.POEM
    assert script.title == "Night"
    assert script.language == "de"
    assert script.hook == "Night: Silver moon above"
    assert [beat.text for beat in script.beats] == [
        "Silver moon above",
        "The quiet river answers",
        "A heron takes flight",
    ]
    assert [beat.index for beat in script.beats] == [0, 1, 2]
    assert [line.speaker for line in script.dialogue] == ["narrator"] * 3
    assert script.outro == ""

    payload = _poem_payload(script)
    assert payload.form == "haiku"
    assert payload.tone == "calm"
    assert payload.rhyme_scheme == "abc"
    assert payload.syllable_budget == 17
    assert payload.length == "short"
    assert payload.stanzas == [
        [
            "Silver moon above",
            "The quiet river answers",
            "A heron takes flight",
        ]
    ]
    assert payload.min_seconds == LENGTH_PRESETS["short"][0]
    assert payload.max_seconds == LENGTH_PRESETS["short"][1]
    assert payload.reading_time_seconds == pytest.approx(11 / 2.5)
    assert payload.reading_time == payload.reading_time_seconds
    assert script.duration_seconds == pytest.approx(11 / 2.5)
    low, high = _seconds(payload)
    assert low <= _duration(script) <= high
    assert Script.model_validate_json(script.to_json()) == script


def test_poem_text_separates_stanzas_and_keeps_form_and_tone() -> None:
    script = PoemMode().build_script(
        "First line\nSecond line\n\nThird line\nFourth line",
        form="free_verse",
        tone="wistful",
    )

    payload = _poem_payload(script)
    assert payload.lines == [
        "First line",
        "Second line",
        "Third line",
        "Fourth line",
    ]
    assert payload.stanzas == [
        ["First line", "Second line"],
        ["Third line", "Fourth line"],
    ]
    assert payload.form == "free_verse"
    assert payload.tone == "wistful"
    assert script.hook == "First line"
    assert [beat.text for beat in script.beats] == payload.lines
    assert script.duration_seconds == pytest.approx(8 / 2.5)


def test_poem_explicit_options_override_the_built_defaults() -> None:
    script = PoemMode().build_script(
        {
            "form": "sonnet",
            "tone": "solemn",
            "lines": ["One line here."],
            "length": "long",
        },
        title="A short verse",
        language="fr",
        form="haiku",
        tone="playful",
        length="short",
        rhyme_scheme="aab",
        syllable_budget=5,
        hook="Open softly.",
        outro="Close softly.",
        speaker="bard",
    )

    payload = _poem_payload(script)
    assert script.title == "A short verse"
    assert script.language == "fr"
    assert script.hook == "Open softly."
    assert script.outro == "Close softly."
    assert payload.form == "haiku"
    assert payload.tone == "playful"
    assert payload.length == "short"
    assert payload.rhyme_scheme == "aab"
    assert payload.syllable_budget == 5
    assert [line.speaker for line in script.dialogue] == ["bard"]


def test_poem_length_preset_keeps_the_reading_time_in_range() -> None:
    lines = ["one two three four five six seven eight nine ten"] * 3
    script = PoemMode().build_script(lines, length="short")

    assert script.duration_seconds == pytest.approx(30 / 2.5)
    payload = _poem_payload(script)
    assert payload.min_seconds == LENGTH_PRESETS["short"][0]
    assert payload.max_seconds == LENGTH_PRESETS["short"][1]
    low, high = _seconds(payload)
    assert low <= _duration(script) <= high


def test_poem_rejects_a_reading_time_outside_the_target() -> None:
    too_long = ["one two three four five six seven eight nine ten"] * 10

    with pytest.raises(ConfigError, match="outside the target range") as refused:
        PoemMode().build_script(too_long, length="short")

    details = refused.value.details
    assert details["mode"] == "poem"
    assert details["measured_seconds"] == pytest.approx(40.0)
    assert details["min_seconds"] == LENGTH_PRESETS["short"][0]
    assert details["max_seconds"] == LENGTH_PRESETS["short"][1]


def test_poem_explicit_second_bounds_override_the_preset() -> None:
    value = {"lines": ["one two three four five"]}

    inside = PoemMode().build_script(value, min_seconds=1, max_seconds=5)
    assert inside.duration_seconds == pytest.approx(2.0)
    assert _poem_payload(inside).min_seconds == 1
    assert _poem_payload(inside).max_seconds == 5

    with pytest.raises(ConfigError, match="outside the target range"):
        PoemMode().build_script(value, min_seconds=10, max_seconds=20)


def test_poem_requires_at_least_one_line() -> None:
    with pytest.raises(ConfigError, match="at least one line"):
        PoemMode().build_script("")
    with pytest.raises(ConfigError, match="at least one line"):
        PoemMode().build_script([])
    with pytest.raises(ConfigError, match="at least one line"):
        PoemMode().build_script(None)


def test_story_mapping_builds_scenes_payload_and_duration() -> None:
    script = StoryMode().build_script(
        {
            "title": "The Lantern",
            "form": "fable",
            "tone": "warm",
            "scenes": [
                "Ana found a lantern in the attic.",
                "She carried it into the storm and the village lights returned.",
            ],
            "characters": ["Ana"],
            "setting": "A coastal village",
            "point_of_view": "third",
            "moral": "Small courage lights the dark.",
            "length": "short",
        }
    )

    assert isinstance(script, Script)
    assert script.mode is WritingMode.STORY
    assert script.title == "The Lantern"
    assert script.language == "en"
    assert script.hook == "The Lantern: Ana found a lantern in the attic."
    assert [beat.text for beat in script.beats] == [
        "Ana found a lantern in the attic.",
        "She carried it into the storm and the village lights returned.",
    ]
    assert [beat.index for beat in script.beats] == [0, 1]
    assert [line.speaker for line in script.dialogue] == ["narrator"] * 2
    assert script.outro == "Small courage lights the dark."

    payload = _story_payload(script)
    assert payload.form == "fable"
    assert payload.tone == "warm"
    assert payload.characters == ["Ana"]
    assert payload.setting == "A coastal village"
    assert payload.point_of_view == "third"
    assert payload.moral == "Small courage lights the dark."
    assert payload.length == "short"
    assert payload.reading_time_seconds == pytest.approx(18 / 2.5)
    assert payload.reading_time == payload.reading_time_seconds
    assert script.duration_seconds == pytest.approx(18 / 2.5)
    low, high = _seconds(payload)
    assert low <= _duration(script) <= high
    assert Script.model_validate_json(script.to_json()) == script


def test_story_text_separates_scenes_and_accepts_tone() -> None:
    script = StoryMode().build_script(
        "First scene.\n\nSecond scene.",
        tone="tense",
        characters="Ana, Ben",
    )

    payload = _story_payload(script)
    assert payload.scenes == ["First scene.", "Second scene."]
    assert payload.tone == "tense"
    assert payload.characters == ["Ana", "Ben"]
    assert script.hook == "First scene."
    assert [beat.text for beat in script.beats] == payload.scenes
    assert script.duration_seconds == pytest.approx(4 / 2.5)


def test_story_explicit_options_override_the_built_defaults() -> None:
    script = StoryMode().build_script(
        {
            "form": "myth",
            "tone": "cold",
            "scenes": ["One scene."],
            "characters": ["Old"],
            "setting": "A city",
            "point_of_view": "first",
            "moral": "The old moral.",
            "length": "long",
        },
        title="A new story",
        language="de",
        form="fable",
        tone="warm",
        characters="Ana, Ben",
        setting="A forest",
        point_of_view="third",
        moral="The new moral.",
        length="short",
        hook="Open here.",
        outro="Close here.",
        speaker="guide",
    )

    payload = _story_payload(script)
    assert script.title == "A new story"
    assert script.language == "de"
    assert script.hook == "Open here."
    assert script.outro == "Close here."
    assert payload.form == "fable"
    assert payload.tone == "warm"
    assert payload.characters == ["Ana", "Ben"]
    assert payload.setting == "A forest"
    assert payload.point_of_view == "third"
    assert payload.moral == "The new moral."
    assert payload.length == "short"
    assert [line.speaker for line in script.dialogue] == ["guide"]


def test_story_length_range_keeps_the_reading_time_in_range() -> None:
    script = StoryMode().build_script(
        {"scenes": ["one two three four five six seven eight nine ten"] * 3},
        length="short",
    )

    assert script.duration_seconds == pytest.approx(30 / 2.5)
    payload = _story_payload(script)
    low, high = _seconds(payload)
    assert low <= _duration(script) <= high

    too_long = {"scenes": ["one two three four five six seven eight nine ten"] * 10}
    with pytest.raises(ConfigError, match="outside the target range") as refused:
        StoryMode().build_script(too_long, length="short")

    assert refused.value.details["mode"] == "story"
    assert refused.value.details["measured_seconds"] == pytest.approx(40.0)


def test_story_requires_at_least_one_scene() -> None:
    with pytest.raises(ConfigError, match="at least one scene"):
        StoryMode().build_script("")
    with pytest.raises(ConfigError, match="at least one scene"):
        StoryMode().build_script([])
    with pytest.raises(ConfigError, match="at least one scene"):
        StoryMode().build_script(None)


def test_individual_builders_can_be_used_on_their_own() -> None:
    poem = PoemMode()
    poem_value = {"lines": ["One line.", "Two line."]}

    assert poem.build_hook(poem_value) == "One line."
    assert poem.build_hook(poem_value, title="Topic") == "Topic: One line."
    assert [beat.text for beat in poem.build_beats(poem_value)] == [
        "One line.",
        "Two line.",
    ]
    assert poem.build_beats(poem_value, max_beats=1)[0].text == "One line."
    assert poem.build_outro(poem_value) == ""

    story = StoryMode()
    story_value = {"scenes": ["One scene.", "Two scene."], "moral": "Keep going."}

    assert story.build_hook(story_value) == "One scene."
    assert story.build_hook(story_value, title="Topic") == "Topic: One scene."
    assert [beat.text for beat in story.build_beats(story_value)] == [
        "One scene.",
        "Two scene.",
    ]
    assert story.build_beats(story_value, max_beats=1)[0].text == "One scene."
    assert story.build_outro(story_value) == "Keep going."


def test_extract_aliases_accept_every_input_shape() -> None:
    assert coerce_poem is extract_poem
    assert parse_poem is extract_poem
    assert normalize_poem is extract_poem
    assert normalise_poem is extract_poem
    assert coerce_story is extract_story
    assert parse_story is extract_story
    assert normalize_story is extract_story
    assert normalise_story is extract_story

    assert extract_poem("One line\nTwo line").lines == ["One line", "Two line"]
    assert extract_poem(("One line", "Two line")).lines == ["One line", "Two line"]
    assert extract_poem({"poem": "One line\n\nTwo line"}).stanzas == [
        ["One line"],
        ["Two line"],
    ]
    assert (
        extract_poem({"payload": {"lines": ["One line"]}, "tone": "calm"}).tone
        == "calm"
    )
    assert extract_poem(PoemPayload(lines=["One line"], form="haiku")).form == "haiku"

    assert extract_story("One scene.\n\nTwo scene.").scenes == [
        "One scene.",
        "Two scene.",
    ]
    assert extract_story(("One scene.", "Two scene.")).scenes == [
        "One scene.",
        "Two scene.",
    ]
    assert extract_story({"story": "One scene.\n\nTwo scene."}).scenes == [
        "One scene.",
        "Two scene.",
    ]
    assert extract_story({"story": "One scene.", "kind": "fable"}).form == "fable"
    assert (
        extract_story({"payload": {"scenes": ["One scene."]}, "tone": "warm"}).tone
        == "warm"
    )
    assert (
        extract_story(StoryPayload(scenes=["One scene."], tone="warm")).tone == "warm"
    )


def test_module_level_builders_and_aliases() -> None:
    assert poem_script is build_poem_script
    assert story_script is build_story_script

    poem = build_poem_script("One line.")
    story = build_story_script("One scene.")

    assert poem.mode is WritingMode.POEM
    assert poem.hook == "One line."
    assert story.mode is WritingMode.STORY
    assert story.hook == "One scene."


def test_building_is_deterministic() -> None:
    poem_value = {
        "title": "Stable",
        "form": "haiku",
        "tone": "calm",
        "lines": ["One line.", "Two line."],
        "length": "short",
    }
    story_value = {
        "title": "Stable",
        "tone": "warm",
        "scenes": ["One scene.", "Two scene."],
        "length": "short",
    }

    def poem_structure(script: Script) -> tuple[object, ...]:
        return (
            script.mode,
            script.title,
            script.hook,
            script.outro,
            script.duration_seconds,
            _poem_payload(script),
            tuple(beat.text for beat in script.beats),
        )

    def story_structure(script: Script) -> tuple[object, ...]:
        return (
            script.mode,
            script.title,
            script.hook,
            script.outro,
            script.duration_seconds,
            _story_payload(script),
            tuple(beat.text for beat in script.beats),
        )

    assert poem_structure(PoemMode().build_script(poem_value)) == poem_structure(
        PoemMode().build_script(poem_value)
    )
    assert story_structure(StoryMode().build_script(story_value)) == story_structure(
        StoryMode().build_script(story_value)
    )


def test_length_helpers_estimate_reading_time_and_parse_ranges() -> None:
    assert count_words("one two, three!") == 3
    assert reading_time_seconds("one two three four five") == pytest.approx(2.0)
    assert reading_time_seconds(
        "one two three four five", words_per_second=5.0
    ) == pytest.approx(1.0)
    assert coerce_length_range("short") == LengthRange(*LENGTH_PRESETS["short"])
    assert coerce_length_range(10) == LengthRange(8.0, 12.0)
    assert coerce_length_range((1, 2)).contains(1.5)
    assert coerce_length_range({"min_seconds": 2, "max_seconds": 4}).contains(3.0)
    assert math.isinf(coerce_length_range({"min_seconds": 2}).max_seconds)

    with pytest.raises(ConfigError, match="unknown length preset"):
        coerce_length_range("not-a-length")
