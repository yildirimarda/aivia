"""Mode aware script, beat, media prompt and dialogue models."""

from __future__ import annotations

import json
import warnings
from typing import Any

import pytest
from pydantic import ValidationError

from aivia.core.models import (
    Beat,
    BeatModel,
    Dialogue,
    DialogueLine,
    ImagePrompt,
    MediaKind,
    MediaPrompt,
    MediaPromptKind,
    MediaType,
    Mode,
    MotionPrompt,
    Script,
    ScriptBeat,
    ScriptMode,
    ScriptModel,
    UnknownFieldWarning,
    VisualType,
    WritingMode,
    WritingModeName,
)


def _line(speaker: str = "narrator", text: str = "A line") -> DialogueLine:
    return DialogueLine(speaker=speaker, text=text)


def _prompt(kind: MediaKind = MediaKind.IMAGE, text: str = "A scene") -> MediaPrompt:
    return MediaPrompt(kind=kind, prompt=text)


def test_script_defaults_to_facts_mode_and_an_empty_structure() -> None:
    script = Script()

    assert script.mode is WritingMode.FACTS
    assert script.title == ""
    assert script.language == "en"
    assert script.beats == []
    assert script.dialogue == []
    assert script.payload == {}
    assert script.hook == ""
    assert script.outro == ""
    assert script.duration_seconds is None


@pytest.mark.parametrize(
    "mode",
    [
        "facts",
        "quote",
        "poem",
        "story",
        "joke",
        "riddle",
        "lyrics",
        "custom",
    ],
)
def test_script_accepts_every_built_in_writing_mode(mode: str) -> None:
    script = Script.model_validate({"mode": mode})

    assert script.mode == mode
    assert isinstance(script.mode, WritingMode)


def test_script_mode_is_normalized_and_aliased() -> None:
    script = Script.model_validate({"writing_mode": "  FACTS  "})

    assert script.mode is WritingMode.FACTS
    assert script.writing_mode is WritingMode.FACTS


def test_script_rejects_an_unknown_mode() -> None:
    with pytest.raises(ValidationError):
        Script.model_validate({"mode": "interpretive_dance"})


def test_script_carries_title_language_beats_payload_and_length() -> None:
    beat = Beat(text="Opening")
    script = Script(
        mode=WritingMode.JOKE,
        title="  A short joke  ",
        language=" de ",
        beats=[beat],
        payload={"setup": "Why did it cross?", "punchline": "To get there"},
        hook="A hook",
        outro="An outro",
        duration_seconds=12.5,
    )

    assert script.title == "A short joke"
    assert script.language == "de"
    assert script.beats == [beat]
    assert script.payload["punchline"] == "To get there"
    assert script.hook == "A hook"
    assert script.outro == "An outro"
    assert script.duration_seconds == 12.5
    assert script.duration == 12.5


def test_script_assigns_beat_indexes_from_their_order() -> None:
    first = Beat(text="one")
    second = Beat(text="two", index=7)
    third = Beat(text="three")

    script = Script(beats=[first, second, third])

    assert [beat.index for beat in script.beats] == [0, 7, 2]
    assert first.order == 0
    assert third.order == 2


def test_script_keeps_an_explicit_zero_index() -> None:
    script = Script(beats=[Beat(text="first", index=0), Beat(text="second")])

    assert [beat.index for beat in script.beats] == [0, 1]


def test_beat_index_defaults_to_zero() -> None:
    assert Beat(text="only").index == 0


def test_dialogue_line_requires_a_speaker_and_text() -> None:
    with pytest.raises(ValidationError):
        DialogueLine(speaker="  ", text="hello")

    with pytest.raises(ValidationError):
        DialogueLine(speaker="narrator", text="  ")


def test_dialogue_line_strips_and_aliases_its_fields() -> None:
    line = DialogueLine.model_validate(
        {
            "character": "  narrator  ",
            "line": "  Hello there  ",
            "mood": "warm",
            "stage_direction": "  smiling  ",
            "voice": "voice-en",
        }
    )

    assert line.speaker == "narrator"
    assert line.text == "Hello there"
    assert line.emotion == "warm"
    assert line.direction == "smiling"
    assert line.voice == "voice-en"
    assert line.character == line.speaker
    assert line.line == line.text
    assert line.mood == line.emotion
    assert line.stage_direction == line.direction


def test_dialogue_line_timing_is_optional_and_ordered() -> None:
    line = DialogueLine.model_validate(
        {"speaker": "narrator", "text": "hello", "start": 1.5, "end": 3.0}
    )

    assert line.start_seconds == 1.5
    assert line.end_seconds == 3.0
    assert line.duration == 1.5

    with pytest.raises(ValidationError):
        DialogueLine(
            speaker="narrator",
            text="hello",
            start_seconds=3.0,
            end_seconds=2.0,
        )


def test_beat_carries_text_dialogue_prompts_and_visual_fields() -> None:
    image = _prompt(MediaKind.IMAGE, "A bridge at dusk")
    motion = _prompt(MediaKind.MOTION, "The bridge slowly pans")
    first = _line()
    second = _line("guide", "Indeed")
    beat = Beat.model_validate(
        {
            "index": 1,
            "narration": "  The bridge opened in 1937  ",
            "dialogue": [first, second],
            "media_prompts": [image, motion],
            "visual_type": "video",
            "transition": "cut",
            "duration_seconds": 4.0,
        }
    )

    assert beat.index == 1
    assert beat.text == "The bridge opened in 1937"
    assert beat.narration == beat.text
    assert beat.voiceover == beat.text
    assert beat.line == beat.text
    assert beat.dialogue == [first, second]
    assert beat.media_prompts == [image, motion]
    assert beat.prompts == beat.media_prompts
    assert beat.visual_type == "video"
    assert beat.visual == "video"
    assert beat.transition == "cut"
    assert beat.duration_seconds == 4.0
    assert beat.duration == 4.0
    assert beat.image_prompt == image
    assert beat.motion_prompt == motion


def test_beat_allows_a_narration_only_or_dialogue_only_unit() -> None:
    line = _line()
    narration = Beat(text="Only narration")
    dialogue = Beat(dialogue=[line])

    assert narration.dialogue == []
    assert narration.media_prompts == []
    assert dialogue.text == ""
    assert dialogue.dialogue == [line]


def test_beat_dialogue_groups_lines_by_speaker_in_order() -> None:
    first = _line("narrator", "One")
    second = _line("guide", "Two")
    third = _line("narrator", "Three")
    beat = Beat(dialogue=[first, second, third])

    assert beat.speakers == ["narrator", "guide"]
    assert beat.dialogue_by_speaker == {
        "narrator": [first, third],
        "guide": [second],
    }
    assert beat.lines_by_speaker == beat.dialogue_by_speaker


def test_beat_accepts_dialogue_as_a_mapping_of_speaker_to_lines() -> None:
    beat = Beat.model_validate(
        {
            "dialogue": {
                "narrator": "Narration",
                "guide": ["First", "Second"],
            }
        }
    )

    assert [line.speaker for line in beat.dialogue] == [
        "narrator",
        "guide",
        "guide",
    ]
    assert [line.text for line in beat.dialogue] == [
        "Narration",
        "First",
        "Second",
    ]


def test_beat_accepts_media_prompt_shorthand() -> None:
    beat = Beat.model_validate(
        {
            "text": "A beat",
            "image_prompt": "A quiet street",
            "motion_prompt": {
                "prompt": "The camera pushes in",
                "style": "cinematic",
            },
        }
    )

    assert len(beat.media_prompts) == 2
    assert beat.image_prompt is not None
    assert beat.image_prompt.prompt == "A quiet street"
    assert beat.image_prompt.kind is MediaKind.IMAGE
    assert beat.motion_prompt is not None
    assert beat.motion_prompt.prompt == "The camera pushes in"
    assert beat.motion_prompt.kind is MediaKind.MOTION
    assert beat.motion_prompt.style == "cinematic"


def test_beat_accepts_media_prompts_as_a_mapping_of_kind_to_prompt() -> None:
    beat = Beat.model_validate(
        {"media_prompts": {"image": "A still", "motion": "A move"}}
    )

    assert [prompt.kind for prompt in beat.media_prompts] == [
        MediaKind.IMAGE,
        MediaKind.MOTION,
    ]
    assert [prompt.prompt for prompt in beat.media_prompts] == ["A still", "A move"]


def test_media_prompt_requires_a_prompt_and_has_optional_fields() -> None:
    with pytest.raises(ValidationError):
        MediaPrompt.model_validate({})

    with pytest.raises(ValidationError):
        MediaPrompt(prompt="   ")

    prompt = MediaPrompt(
        prompt="  A lone tree  ",
        negative_prompt="  text, watermark  ",
        style="watercolour",
        aspect_ratio="9:16",
        seed=42,
        duration_seconds=3.5,
    )

    assert prompt.kind is MediaKind.IMAGE
    assert prompt.prompt == "A lone tree"
    assert prompt.text == prompt.prompt
    assert prompt.negative_prompt == "text, watermark"
    assert prompt.negative == prompt.negative_prompt
    assert prompt.style == "watercolour"
    assert prompt.aspect_ratio == "9:16"
    assert prompt.seed == 42
    assert prompt.duration_seconds == 3.5
    assert prompt.duration == 3.5


def test_media_prompt_kind_accepts_aliases_and_is_normalized() -> None:
    prompt = MediaPrompt.model_validate(
        {"type": "  VISUAL  ", "description": "A scene"}
    )

    assert prompt.kind is MediaKind.IMAGE
    assert prompt.media_type is MediaKind.IMAGE
    assert prompt.target is MediaKind.IMAGE
    assert prompt.prompt == "A scene"


def test_media_prompt_rejects_an_unknown_kind() -> None:
    with pytest.raises(ValidationError):
        MediaPrompt.model_validate({"kind": "hologram", "prompt": "x"})


def test_image_and_motion_prompts_have_distinct_defaults() -> None:
    image = ImagePrompt(prompt="A still")
    motion = MotionPrompt(prompt="A move")

    assert image.kind is MediaKind.IMAGE
    assert motion.kind is MediaKind.MOTION


def test_script_dialogue_groups_lines_across_its_beats() -> None:
    first = _line("narrator", "Intro")
    second = _line("guide", "Welcome")
    third = _line("narrator", "Closing")
    script = Script.model_validate(
        {
            "dialogue": [first],
            "beats": [
                Beat(dialogue=[second]),
                Beat(dialogue=[third]),
            ],
        }
    )

    assert script.dialogue == [first, second, third]
    assert script.all_dialogue == [first, second, third]
    assert script.speakers == ["narrator", "guide"]
    assert script.dialogue_by_speaker == {
        "narrator": [first, third],
        "guide": [second],
    }
    assert script.lines_by_speaker == script.dialogue_by_speaker


def test_script_accepts_aliases_without_unknown_field_warnings() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnknownFieldWarning)
        script = Script.model_validate(
            {
                "writing_mode": "lyrics",
                "name": "Aliased script",
                "lang": "tr",
                "units": [{"order": 0, "narration": "A line"}],
                "mode_data": {"sections": 2},
            }
        )

    assert script.mode is WritingMode.LYRICS
    assert script.title == "Aliased script"
    assert script.language == "tr"
    assert script.beats[0].index == 0
    assert script.beats[0].text == "A line"
    assert script.payload == {"sections": 2}


def test_script_serializes_canonically_with_nested_beats() -> None:
    script = Script(
        mode=WritingMode.FACTS,
        title="Topic",
        beats=[
            Beat(
                text="Narration",
                dialogue=[_line()],
                media_prompts=[_prompt()],
            )
        ],
    )

    payload: dict[str, Any] = json.loads(script.to_json())

    assert list(payload) == sorted(payload)
    assert payload["mode"] == "facts"
    assert payload["beats"][0]["dialogue"][0]["speaker"] == "narrator"
    assert payload["beats"][0]["media_prompts"][0]["kind"] == "image"


def test_script_round_trips_through_json() -> None:
    script = Script(
        mode=WritingMode.STORY,
        title="A story",
        language="en",
        beats=[
            Beat(
                index=0,
                text="Once upon a time",
                dialogue=[_line("narrator", "Once upon a time")],
                media_prompts=[_prompt(MediaKind.IMAGE, "A forest")],
                visual_type="image",
                duration_seconds=2.0,
            )
        ],
        payload={"tone": "warm"},
    )

    restored = Script.model_validate_json(script.to_json())

    assert restored == script
    assert restored.to_json() == script.to_json()


def test_script_round_trips_with_dialogue_only_beats() -> None:
    script = Script(
        mode=WritingMode.QUOTE,
        beats=[Beat.model_validate({"dialogue": {"narrator": "A quoted line"}})],
    )

    restored = Script.model_validate_json(script.to_json())

    assert restored == script
    assert restored.dialogue_by_speaker["narrator"][0].text == "A quoted line"


def test_script_mode_and_model_aliases_are_exported() -> None:
    assert ScriptModel is Script
    assert ScriptBeat is Beat
    assert BeatModel is Beat
    assert Dialogue is DialogueLine
    assert ScriptMode is WritingMode
    assert WritingModeName is WritingMode
    assert Mode is WritingMode
    assert MediaPromptKind is MediaKind
    assert MediaType is MediaKind
    assert VisualType.VIDEO.value == "video"
