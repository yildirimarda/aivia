"""The joke and riddle modes keep a setup and punchline or question and answer."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

import pytest

import aivia.writing.humor as humor_module
from aivia.core.errors import ConfigError
from aivia.writing import (
    BuiltinWritingMode,
    Joke,
    JokeMode,
    JokePayload,
    JokeWritingMode,
    Riddle,
    RiddleMode,
    RiddlePayload,
    RiddleWritingMode,
    Script,
    WritingMode,
    build_joke_script,
    build_riddle_script,
    builtin_writing_mode,
    coerce_joke,
    coerce_riddle,
    extract_joke,
    extract_riddle,
    joke_script,
    normalise_joke,
    normalise_riddle,
    normalize_joke,
    normalize_riddle,
    parse_joke,
    parse_riddle,
    resolve_mode,
    riddle_script,
    validate_joke,
    validate_riddle,
)


def _joke_payload(script: Script) -> JokePayload:
    assert isinstance(script.payload, JokePayload)
    return script.payload


def _riddle_payload(script: Script) -> RiddlePayload:
    assert isinstance(script.payload, RiddlePayload)
    return script.payload


@dataclass
class JokeRecord:
    """A structural joke object without importing a provider or module."""

    setup: str
    punchline: str
    style: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class RiddleRecord:
    """A structural riddle object without importing a provider or module."""

    question: str
    answer: str
    hint: str = ""
    difficulty: str = ""


def test_joke_mode_replaces_the_builtin_placeholder() -> None:
    assert resolve_mode("joke") is JokeMode
    assert builtin_writing_mode(WritingMode.JOKE) is JokeMode
    assert issubclass(JokeMode, BuiltinWritingMode)

    mode = JokeMode()

    assert mode.name == "joke"
    assert mode.mode == "joke"
    assert mode.payload_model is JokePayload
    assert JokeWritingMode is JokeMode
    assert Joke is JokeMode


def test_riddle_mode_replaces_the_builtin_placeholder() -> None:
    assert resolve_mode("riddle") is RiddleMode
    assert builtin_writing_mode(WritingMode.RIDDLE) is RiddleMode
    assert issubclass(RiddleMode, BuiltinWritingMode)

    mode = RiddleMode()

    assert mode.name == "riddle"
    assert mode.mode == "riddle"
    assert mode.payload_model is RiddlePayload
    assert RiddleWritingMode is RiddleMode
    assert Riddle is RiddleMode


def test_humor_mode_does_not_import_research() -> None:
    source = Path(humor_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)

    assert not [name for name in imported if name.startswith("aivia.research")]


def test_joke_mapping_builds_hook_beats_payload_and_round_trips() -> None:
    script = JokeMode().build_script(
        {
            "title": "Chicken",
            "language": "de",
            "setup": "  Why did the chicken cross the road?  ",
            "punchline": "  To get to the other side.  ",
            "style": "one-liner",
            "tags": ["animals", "classic"],
        }
    )

    assert isinstance(script, Script)
    assert script.mode is WritingMode.JOKE
    assert script.title == "Chicken"
    assert script.language == "de"
    assert script.hook == "Chicken: Why did the chicken cross the road?"
    assert [beat.text for beat in script.beats] == [
        "Why did the chicken cross the road?",
        "To get to the other side.",
    ]
    assert [beat.index for beat in script.beats] == [0, 1]
    assert [line.speaker for line in script.dialogue] == ["narrator"] * 2
    assert script.outro == ""
    payload = _joke_payload(script)
    assert payload.setup == "Why did the chicken cross the road?"
    assert payload.punchline == "To get to the other side."
    assert payload.style == "one-liner"
    assert payload.tags == ["animals", "classic"]
    assert Script.model_validate_json(script.to_json()) == script


def test_riddle_mapping_builds_hook_beats_payload_and_round_trips() -> None:
    script = RiddleMode().build_script(
        {
            "title": "Keys",
            "language": "fr",
            "question": "  What has keys but no locks?  ",
            "answer": "  A piano.  ",
            "hint": "It makes music.",
            "difficulty": "easy",
        }
    )

    assert isinstance(script, Script)
    assert script.mode is WritingMode.RIDDLE
    assert script.title == "Keys"
    assert script.language == "fr"
    assert script.hook == "Keys: What has keys but no locks?"
    assert [beat.text for beat in script.beats] == [
        "What has keys but no locks?",
        "A piano.",
    ]
    assert [beat.index for beat in script.beats] == [0, 1]
    assert [line.speaker for line in script.dialogue] == ["narrator"] * 2
    assert script.outro == ""
    payload = _riddle_payload(script)
    assert payload.question == "What has keys but no locks?"
    assert payload.answer == "A piano."
    assert payload.hint == "It makes music."
    assert payload.difficulty == "easy"
    assert Script.model_validate_json(script.to_json()) == script


def test_joke_text_splits_on_a_blank_line_or_newline_and_strips_labels() -> None:
    blank_line = JokeMode().build_script("Why?\n\nBecause.")
    newline = JokeMode().build_script("Why?\nBecause.")
    labelled = JokeMode().build_script("Setup: Why?\nPunchline: Because.")

    for script in (blank_line, newline, labelled):
        payload = _joke_payload(script)
        assert payload.setup == "Why?"
        assert payload.punchline == "Because."
        assert script.hook == "Why?"


def test_riddle_text_splits_on_a_blank_line_or_newline_and_strips_labels() -> None:
    blank_line = RiddleMode().build_script("What has keys?\n\nA piano.")
    newline = RiddleMode().build_script("What has keys?\nA piano.")
    labelled = RiddleMode().build_script("Question: What has keys?\nAnswer: A piano.")

    for script in (blank_line, newline, labelled):
        payload = _riddle_payload(script)
        assert payload.question == "What has keys?"
        assert payload.answer == "A piano."
        assert script.hook == "What has keys?"


def test_joke_explicit_options_override_the_built_defaults() -> None:
    script = JokeMode().build_script(
        {
            "setup": "The old setup.",
            "punchline": "The old punchline.",
            "style": "old",
            "tags": ["old"],
            "hook": "The old hook.",
            "outro": "The old outro.",
        },
        title="A new joke",
        language="de",
        setup="The new setup.",
        punchline="The new punchline.",
        style="pun",
        tags=["new"],
        hook="Open here.",
        outro="Close here.",
        speaker="comedian",
        max_beats=1,
    )

    payload = _joke_payload(script)
    assert script.title == "A new joke"
    assert script.language == "de"
    assert script.hook == "Open here."
    assert script.outro == "Close here."
    assert payload.setup == "The new setup."
    assert payload.punchline == "The new punchline."
    assert payload.style == "pun"
    assert payload.tags == ["new"]
    assert [beat.text for beat in script.beats] == ["The new setup."]
    assert [line.speaker for line in script.dialogue] == ["comedian"]


def test_riddle_explicit_options_override_the_built_defaults() -> None:
    script = RiddleMode().build_script(
        {
            "question": "The old question.",
            "answer": "The old answer.",
            "hint": "The old hint.",
            "difficulty": "hard",
            "hook": "The old hook.",
            "outro": "The old outro.",
        },
        title="A new riddle",
        language="de",
        question="The new question.",
        answer="The new answer.",
        hint="The new hint.",
        difficulty="easy",
        hook="Open here.",
        outro="Close here.",
        speaker="host",
        max_beats=1,
    )

    payload = _riddle_payload(script)
    assert script.title == "A new riddle"
    assert script.language == "de"
    assert script.hook == "Open here."
    assert script.outro == "Close here."
    assert payload.question == "The new question."
    assert payload.answer == "The new answer."
    assert payload.hint == "The new hint."
    assert payload.difficulty == "easy"
    assert [beat.text for beat in script.beats] == ["The new question."]
    assert [line.speaker for line in script.dialogue] == ["host"]


def test_structural_humor_objects_are_read_without_importing_them() -> None:
    joke = JokeMode().build_script(
        JokeRecord(
            setup="A setup.",
            punchline="A punchline.",
            style="dry",
            tags=["short"],
        )
    )
    riddle = RiddleMode().build_script(
        RiddleRecord(
            question="A question?",
            answer="An answer.",
            hint="A hint.",
            difficulty="medium",
        )
    )

    assert _joke_payload(joke).setup == "A setup."
    assert _joke_payload(joke).punchline == "A punchline."
    assert _joke_payload(joke).style == "dry"
    assert _joke_payload(joke).tags == ["short"]
    assert _riddle_payload(riddle).question == "A question?"
    assert _riddle_payload(riddle).answer == "An answer."
    assert _riddle_payload(riddle).hint == "A hint."
    assert _riddle_payload(riddle).difficulty == "medium"


def test_joke_requires_a_setup_and_a_punchline() -> None:
    with pytest.raises(ConfigError, match="needs a setup and a punchline") as empty:
        JokeMode().build_script("")

    assert empty.value.details["missing"] == ["setup", "punchline"]

    with pytest.raises(ConfigError, match="needs a punchline"):
        JokeMode().build_script({"setup": "Only a setup."})

    with pytest.raises(ConfigError, match="needs a setup"):
        JokeMode().build_script({"punchline": "Only a punchline."})


def test_riddle_requires_a_question_and_an_answer() -> None:
    with pytest.raises(ConfigError, match="needs a question and an answer") as empty:
        RiddleMode().build_script("")

    assert empty.value.details["missing"] == ["question", "answer"]

    with pytest.raises(ConfigError, match="needs an answer"):
        RiddleMode().build_script({"question": "Only a question?"})

    with pytest.raises(ConfigError, match="needs a question"):
        RiddleMode().build_script({"answer": "Only an answer."})


def test_joke_accepts_question_and_answer_aliases() -> None:
    script = JokeMode().build_script(
        {"question": "Why?", "answer": "Because.", "kind": "pun"}
    )

    payload = _joke_payload(script)
    assert payload.setup == "Why?"
    assert payload.punchline == "Because."
    assert payload.style == "pun"


def test_riddle_accepts_setup_and_punchline_aliases() -> None:
    script = RiddleMode().build_script(
        {"setup": "What?", "punchline": "That.", "clue": "Think."}
    )

    payload = _riddle_payload(script)
    assert payload.question == "What?"
    assert payload.answer == "That."
    assert payload.hint == "Think."


def test_validate_helpers_return_payloads_and_check_both_halves() -> None:
    assert validate_joke({"setup": "Why?", "punchline": "Because."}).setup == "Why?"
    assert validate_riddle({"question": "What?", "answer": "That."}).answer == "That."

    with pytest.raises(ConfigError, match="needs a punchline"):
        validate_joke({"setup": "Why?"})

    with pytest.raises(ConfigError, match="needs an answer"):
        validate_riddle({"question": "What?"})


def test_build_payload_keeps_the_typed_contract() -> None:
    joke = JokeMode().build_payload(
        {"setup": "Why?", "payoff": "Because.", "kind": "pun"}
    )
    riddle = RiddleMode().build_payload(
        {"prompt": "What?", "solution": "That.", "level": "easy"}
    )

    assert isinstance(joke, JokePayload)
    assert joke.setup == "Why?"
    assert joke.punchline == "Because."
    assert joke.style == "pun"
    partial_joke = JokeMode().build({"setup": "Only."})
    assert isinstance(partial_joke, JokePayload)
    assert partial_joke.punchline == ""
    assert isinstance(riddle, RiddlePayload)
    assert riddle.question == "What?"
    assert riddle.answer == "That."
    assert riddle.difficulty == "easy"
    partial_riddle = RiddleMode().build({"question": "Only?"})
    assert isinstance(partial_riddle, RiddlePayload)
    assert partial_riddle.answer == ""


def test_extract_aliases_accept_every_input_shape() -> None:
    assert coerce_joke is extract_joke
    assert parse_joke is extract_joke
    assert normalize_joke is extract_joke
    assert normalise_joke is extract_joke
    assert coerce_riddle is extract_riddle
    assert parse_riddle is extract_riddle
    assert normalize_riddle is extract_riddle
    assert normalise_riddle is extract_riddle

    assert extract_joke("Why?\nBecause.").punchline == "Because."
    assert extract_joke(("Why?", "Because.")).setup == "Why?"
    assert extract_joke({"joke": "Why?\n\nBecause."}).punchline == "Because."
    assert (
        extract_joke(
            {
                "payload": {"setup": "Why?", "punchline": "Because."},
                "style": "pun",
            }
        ).style
        == "pun"
    )
    assert (
        extract_joke(JokePayload(setup="Why?", punchline="Because.")).punchline
        == "Because."
    )

    assert extract_riddle("What?\nThat.").answer == "That."
    assert extract_riddle(("What?", "That.")).question == "What?"
    assert extract_riddle({"riddle": "What?\n\nThat."}).answer == "That."
    assert (
        extract_riddle(
            {
                "payload": {"question": "What?", "answer": "That."},
                "hint": "Think.",
            }
        ).hint
        == "Think."
    )
    assert (
        extract_riddle(RiddlePayload(question="What?", answer="That.")).question
        == "What?"
    )


def test_individual_builders_can_be_used_on_their_own() -> None:
    joke = JokeMode()
    joke_value = {"setup": "Why?", "punchline": "Because."}

    assert joke.build_hook(joke_value) == "Why?"
    assert joke.build_hook(joke_value, title="Topic") == "Topic: Why?"
    assert [beat.text for beat in joke.build_beats(joke_value)] == [
        "Why?",
        "Because.",
    ]
    assert joke.build_beats(joke_value, max_beats=1)[0].text == "Why?"
    assert joke.build_beats(joke_value, max_beats=0) == []
    assert joke.build_outro(joke_value) == ""

    riddle = RiddleMode()
    riddle_value = {"question": "What?", "answer": "That."}

    assert riddle.build_hook(riddle_value) == "What?"
    assert riddle.build_hook(riddle_value, title="Topic") == "Topic: What?"
    assert [beat.text for beat in riddle.build_beats(riddle_value)] == [
        "What?",
        "That.",
    ]
    assert riddle.build_beats(riddle_value, max_beats=1)[0].text == "What?"
    assert riddle.build_beats(riddle_value, max_beats=0) == []
    assert riddle.build_outro(riddle_value) == ""


def test_module_level_builders_and_aliases() -> None:
    assert joke_script is build_joke_script
    assert riddle_script is build_riddle_script

    joke = build_joke_script({"setup": "Why?", "punchline": "Because."})
    riddle = build_riddle_script({"question": "What?", "answer": "That."})

    assert joke.mode is WritingMode.JOKE
    assert joke.hook == "Why?"
    assert riddle.mode is WritingMode.RIDDLE
    assert riddle.hook == "What?"


def test_building_is_deterministic() -> None:
    joke_value = {
        "title": "Stable",
        "setup": "Why?",
        "punchline": "Because.",
        "style": "pun",
        "tags": ["short"],
    }
    riddle_value = {
        "title": "Stable",
        "question": "What?",
        "answer": "That.",
        "hint": "Think.",
        "difficulty": "easy",
    }

    def joke_structure(script: Script) -> tuple[object, ...]:
        return (
            script.mode,
            script.title,
            script.hook,
            script.outro,
            _joke_payload(script),
            tuple(beat.text for beat in script.beats),
        )

    def riddle_structure(script: Script) -> tuple[object, ...]:
        return (
            script.mode,
            script.title,
            script.hook,
            script.outro,
            _riddle_payload(script),
            tuple(beat.text for beat in script.beats),
        )

    assert joke_structure(JokeMode().build_script(joke_value)) == joke_structure(
        JokeMode().build_script(joke_value)
    )
    assert riddle_structure(RiddleMode().build_script(riddle_value)) == (
        riddle_structure(RiddleMode().build_script(riddle_value))
    )
