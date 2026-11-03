"""Mode aware script schema: typed payloads, dialogue and a stable round trip."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from aivia.writing import (
    CURRENT_SCRIPT_SCHEMA_VERSION,
    MODE_PAYLOADS,
    SCRIPT_SCHEMA_VERSION,
    Beat,
    CustomPayload,
    DialogueLine,
    FactsPayload,
    JokePayload,
    LyricsPayload,
    LyricsSection,
    PoemPayload,
    QuotePayload,
    RiddlePayload,
    Script,
    ScriptConfig,
    ScriptModel,
    ScriptSchema,
    ScriptSpec,
    StoryPayload,
    WritingMode,
    WritingScript,
    build_payload,
    coerce_payload,
    parse_payload,
    payload_for_mode,
    payload_model_for_mode,
    validate_payload,
)

_SOURCE = "https://example.com/source"

_PAYLOADS: dict[WritingMode, dict[str, Any]] = {
    WritingMode.FACTS: {
        "facts": ["One fact.", {"text": "Two facts.", "sources": [_SOURCE]}],
        "angle": "warm",
    },
    WritingMode.QUOTE: {
        "quote": "To be or not to be",
        "attribution": "William Shakespeare",
        "source_urls": [_SOURCE],
    },
    WritingMode.POEM: {
        "form": "haiku",
        "tone": "calm",
        "lines": ["first", "second", "third"],
        "stanzas": [["first"], ["second", "third"]],
        "rhyme_scheme": "abc",
        "syllable_budget": 17,
    },
    WritingMode.STORY: {
        "tone": "warm",
        "scenes": ["Opening", "Closing"],
        "characters": ["Ana"],
        "setting": "A forest",
        "point_of_view": "third",
        "moral": "Keep going",
    },
    WritingMode.JOKE: {
        "setup": "Why did the chicken cross the road?",
        "punchline": "To get to the other side.",
        "style": "one-liner",
        "tags": ["short"],
    },
    WritingMode.RIDDLE: {
        "question": "What has keys but no locks?",
        "answer": "A piano",
        "hint": "It makes music",
        "difficulty": "easy",
    },
    WritingMode.LYRICS: {
        "sections": [
            {
                "kind": "verse",
                "lines": ["A line", "Another line"],
                "rhyme_scheme": "aa",
                "syllable_budget": 8,
            },
            {"name": "chorus", "lines": ["The hook"]},
        ],
        "section_counts": {"verse": 1, "chorus": 1},
        "rhyme_scheme": "aa",
        "syllable_budget": 8,
        "keywords": ["night", "city"],
        "genre": "pop",
        "tempo": "fast",
    },
    WritingMode.CUSTOM: {
        "name": "my-mode",
        "instructions": "Follow the house style",
        "extra": {"keep": True},
    },
}

_PAYLOAD_TYPES: dict[WritingMode, type[Any]] = {
    WritingMode.FACTS: FactsPayload,
    WritingMode.QUOTE: QuotePayload,
    WritingMode.POEM: PoemPayload,
    WritingMode.STORY: StoryPayload,
    WritingMode.JOKE: JokePayload,
    WritingMode.RIDDLE: RiddlePayload,
    WritingMode.LYRICS: LyricsPayload,
    WritingMode.CUSTOM: CustomPayload,
}


def _script(mode: WritingMode, **overrides: Any) -> Script:
    values: dict[str, Any] = {
        "mode": mode,
        "title": "A script",
        "language": "en",
        "beats": [
            Beat(
                text="The opening line",
                dialogue=[
                    DialogueLine(speaker="narrator", text="The opening line"),
                    DialogueLine(speaker="guide", text="Welcome"),
                ],
            )
        ],
        "payload": _PAYLOADS[mode],
    }
    values.update(overrides)
    return Script.model_validate(values)


def test_script_defaults_to_facts_mode_with_a_typed_payload() -> None:
    script = Script()

    assert script.mode is WritingMode.FACTS
    assert isinstance(script.payload, FactsPayload)
    assert script.payload.facts == []
    assert script.beats == []
    assert script.dialogue == []
    assert script.title == ""
    assert script.language == "en"
    assert script.schema_version == CURRENT_SCRIPT_SCHEMA_VERSION
    assert CURRENT_SCRIPT_SCHEMA_VERSION == SCRIPT_SCHEMA_VERSION == 1


@pytest.mark.parametrize("mode", list(WritingMode))
def test_every_mode_selects_its_payload_class(mode: WritingMode) -> None:
    script = Script(mode=mode)

    assert script.mode is mode
    assert isinstance(script.payload, _PAYLOAD_TYPES[mode])
    assert MODE_PAYLOADS[mode] is _PAYLOAD_TYPES[mode]


@pytest.mark.parametrize("mode", list(WritingMode))
def test_payload_model_for_mode_resolves_built_in_and_unknown_modes(
    mode: WritingMode,
) -> None:
    assert payload_model_for_mode(mode) is _PAYLOAD_TYPES[mode]
    assert payload_model_for_mode(mode.value.upper()) is _PAYLOAD_TYPES[mode]
    assert payload_model_for_mode("not-a-built-in-mode") is CustomPayload


def test_payload_aliases_all_build_the_same_typed_payload() -> None:
    for builder in (
        coerce_payload,
        parse_payload,
        payload_for_mode,
        validate_payload,
        build_payload,
    ):
        payload = builder(WritingMode.JOKE, {"setup": "Why?", "punchline": "Because"})

        assert isinstance(payload, JokePayload)
        assert payload.setup == "Why?"
        assert payload.punchline == "Because"


def test_joke_payload_aliases_setup_and_punchline() -> None:
    payload = JokePayload.model_validate(
        {"question": "  Why?  ", "answer": "  Because.  ", "kind": " pun "}
    )

    assert payload.setup == "Why?"
    assert payload.punchline == "Because."
    assert payload.style == "pun"


def test_quote_payload_carries_attribution_and_source_urls() -> None:
    payload = QuotePayload.model_validate(
        {
            "text": "  A quotation.  ",
            "author": "  Someone  ",
            "sources": [_SOURCE, _SOURCE],
        }
    )

    assert payload.quote == "A quotation."
    assert payload.attribution == "Someone"
    assert payload.source_urls == [_SOURCE]


def test_facts_payload_accepts_text_and_sourced_mappings() -> None:
    payload = FactsPayload.model_validate(
        {
            "facts": [
                "  A plain fact.  ",
                {"statement": "A sourced fact.", "sources": [_SOURCE]},
            ],
            "source_urls": [_SOURCE],
        }
    )

    assert [fact.text for fact in payload.facts] == [
        "A plain fact.",
        "A sourced fact.",
    ]
    assert payload.facts[0].source_urls == []
    assert payload.facts[1].source_urls == [_SOURCE]
    assert payload.source_urls == [_SOURCE]


def test_lyrics_payload_parses_sections_and_keywords() -> None:
    payload = LyricsPayload.model_validate(
        {
            "sections": [
                {"type": " verse ", "lines": "One line"},
                {"name": "chorus", "lines": ["The hook"]},
            ],
            "keywords": "night",
            "syllable_budget": 8,
        }
    )

    assert [section.name for section in payload.sections] == ["verse", "chorus"]
    assert payload.sections[0].kind == "verse"
    assert payload.sections[0].lines == ["One line"]
    assert payload.keywords == ["night"]
    assert payload.syllable_budget == 8
    assert all(isinstance(section, LyricsSection) for section in payload.sections)


@pytest.mark.parametrize("alias", ["mode_payload", "mode_data", "data", "options"])
def test_script_accepts_payload_aliases(alias: str) -> None:
    script = Script.model_validate(
        {
            "writing_mode": "riddle",
            alias: {"question": "What?", "answer": "That"},
        }
    )

    assert script.mode is WritingMode.RIDDLE
    assert isinstance(script.payload, RiddlePayload)
    assert script.payload.question == "What?"
    assert script.payload.answer == "That"


def test_script_dialogue_is_grouped_per_speaker_across_beats() -> None:
    intro = DialogueLine(speaker="narrator", text="Intro")
    welcome = DialogueLine(speaker="guide", text="Welcome")
    closing = DialogueLine(speaker="narrator", text="Closing")
    script = Script.model_validate(
        {
            "mode": "story",
            "dialogue": [intro],
            "beats": [
                {"dialogue": [welcome]},
                {"dialogue": [closing]},
            ],
        }
    )

    assert script.dialogue == [intro, welcome, closing]
    assert script.speakers == ["narrator", "guide"]
    assert script.dialogue_by_speaker == {
        "narrator": [intro, closing],
        "guide": [welcome],
    }
    assert script.lines_by_speaker == script.dialogue_by_speaker
    assert script.beats[0].dialogue_by_speaker == {"guide": [welcome]}


@pytest.mark.parametrize("mode", list(WritingMode))
def test_script_round_trips_every_mode_without_losing_payload(
    mode: WritingMode,
) -> None:
    script = _script(mode)

    text = script.to_json()
    restored = Script.model_validate_json(text)

    assert restored == script
    assert restored.to_json() == text
    assert isinstance(restored.payload, _PAYLOAD_TYPES[mode])
    expected = coerce_payload(mode, _PAYLOADS[mode]).model_dump(mode="json")
    assert restored.payload.model_dump(mode="json") == expected


def test_script_round_trip_is_stable_across_repeated_reads() -> None:
    script = _script(WritingMode.LYRICS, hook="A hook", outro="An outro")

    first = script.to_json()
    second = Script.model_validate_json(first).to_json()
    third = Script.model_validate_json(second).to_json()

    assert first == second == third
    decoded = json.loads(first)
    assert list(decoded) == sorted(decoded)
    assert decoded["mode"] == "lyrics"
    assert decoded["payload"]["sections"][0]["name"] == "verse"
    assert decoded["beats"][0]["dialogue"][0]["speaker"] == "narrator"


def test_script_payload_dict_matches_the_typed_payload() -> None:
    script = _script(WritingMode.POEM)

    assert script.typed_payload is script.payload
    assert script.payload_dict == script.payload.model_dump(mode="json")


def test_custom_payload_keeps_extra_fields() -> None:
    script = _script(WritingMode.CUSTOM)

    assert isinstance(script.payload, CustomPayload)
    assert script.payload.name == "my-mode"
    assert script.payload.model_extra == {"extra": {"keep": True}}
    assert Script.model_validate_json(script.to_json()) == script


def test_script_rejects_an_unknown_mode_and_a_non_mapping_payload() -> None:
    with pytest.raises(ValidationError):
        Script.model_validate({"mode": "interpretive_dance"})

    with pytest.raises(ValidationError):
        Script.model_validate({"mode": "joke", "payload": "not a mapping"})


def test_script_schema_aliases_are_exported() -> None:
    assert ScriptSchema is Script
    assert ScriptSpec is Script
    assert ScriptConfig is Script
    assert ScriptModel is Script
    assert WritingScript is Script
    assert ScriptModel.__name__ == "Script"
