"""Character prompt injection: character info reaches only scenes that include
the character.
A character's sheet, art style, negatives and per-scene pose are injected into
the media prompts of a scene only when that character appears in it, determined
by the beat's dialogue speakers. Scenes without the character are left untouched.
"""

from __future__ import annotations

from aivia.characters import Character
from aivia.characters.sheet import CharacterSheet
from aivia.core.models.script import (
    Beat,
    DialogueLine,
    ImagePrompt,
    MediaKind,
    MotionPrompt,
    Script,
)
from aivia.writing import build_media_prompts

STYLE = {
    "name": "house",
    "description": "A calm documentary look.",
    "prompt_suffix": "Muted palette and soft window light.",
}


def _character(
    name: str,
    *,
    art_style: str = "",
    negatives: tuple[str, ...] = (),
    pose_library: dict[str, str] | None = None,
) -> Character:
    return Character.model_validate(
        {
            "name": name,
            "art_style": art_style,
            "negatives": list(negatives),
            "pose_library": dict(pose_library or {}),
        }
    )


def _sheet(
    name: str,
    *,
    description: str = "",
    art_style: str = "",
    pose: str = "",
) -> CharacterSheet:
    return CharacterSheet.model_validate(
        {
            "name": name,
            "description": description,
            "art_style": art_style,
            "pose": pose,
        }
    )


def _script() -> Script:
    return Script(
        title="Show",
        language="en",
        beats=[
            Beat(
                index=0,
                text="The host opens the show.",
                dialogue=[DialogueLine(speaker="alice", text="Hello everyone.")],
            ),
            Beat(
                index=1,
                text="A quiet establishing shot.",
                dialogue=[DialogueLine(speaker="narrator", text="Empty room.")],
            ),
            Beat(
                index=2,
                text="The host returns.",
                dialogue=[DialogueLine(speaker="alice", text="Back again.")],
            ),
        ],
    )


def _prompts(
    characters: dict[str, Character] | None = None,
    sheets: dict[str, CharacterSheet] | None = None,
) -> Script:
    return build_media_prompts(
        _script(),
        style_bible=STYLE,
        characters=characters or {},
        sheets=sheets or {},
    )


def _image(beat: Beat) -> ImagePrompt:
    prompt = beat.image_prompt
    assert isinstance(prompt, ImagePrompt)
    return prompt


def _motion(beat: Beat) -> MotionPrompt:
    prompt = beat.motion_prompt
    assert isinstance(prompt, MotionPrompt)
    return prompt


def test_character_info_injected_only_into_scenes_with_the_character() -> None:
    alice = _character(
        "alice",
        art_style="flat vector",
        negatives=("blurry", "low resolution"),
        pose_library={"0": "idle.png", "2": "wave.png"},
    )
    sheets = {
        "alice": _sheet(
            "alice",
            description="A red mascot with round ears.",
            art_style="flat vector",
            pose="idle",
        )
    }

    script = _prompts(characters={"alice": alice}, sheets=sheets)
    beats = script.beats

    assert [beat.text for beat in beats] == [
        "The host opens the show.",
        "A quiet establishing shot.",
        "The host returns.",
    ]

    # Scenes that include alice (beats 0 and 2) carry her sheet, art style and
    # per-scene pose, plus her negatives in the negative prompt.
    for index in (0, 2):
        prompt = _image(beats[index]).prompt
        assert "A red mascot with round ears." in prompt, prompt
        assert "art style flat vector" in prompt, prompt
        negative = _image(beats[index]).negative_prompt
        assert "blurry" in negative, negative
        assert "low resolution" in negative, negative

    expected_pose = {0: "idle.png", 2: "wave.png"}
    for index, pose in expected_pose.items():
        assert f"pose {pose}" in _image(beats[index]).prompt, _image(
            beats[index]
        ).prompt

    # The scene without alice (beat 1) receives no character info.
    other_prompt = _image(beats[1]).prompt
    for token in ("red mascot", "flat vector", "idle.png", "wave.png", "alice"):
        assert token not in other_prompt, other_prompt
    other_negative = _image(beats[1]).negative_prompt
    assert "blurry" not in other_negative
    assert "low resolution" not in other_negative


def test_motion_prompts_receive_the_same_injection() -> None:
    alice = _character(
        "alice",
        art_style="flat vector",
        pose_library={"0": "idle.png"},
    )
    sheets = {"alice": _sheet("alice", description="A red mascot.")}

    script = _prompts(characters={"alice": alice}, sheets=sheets)
    assert "A red mascot." in _motion(script.beats[0]).prompt
    assert "pose idle.png" in _motion(script.beats[0]).prompt
    assert "A red mascot." not in _motion(script.beats[1]).prompt


def test_no_characters_leaves_prompts_untouched() -> None:
    script = _prompts()
    for beat in script.beats:
        prompt = _image(beat).prompt
        assert prompt
        for token in ("red mascot", "flat vector", "pose", "blurry"):
            assert token not in prompt, prompt


def test_unknown_character_is_never_injected() -> None:
    eve = _character("eve", art_style="oil", pose_library={"0": "eve.png"})
    sheets = {"eve": _sheet("eve", description="An oil painting figure.")}

    script = _prompts(characters={"eve": eve}, sheets=sheets)
    for beat in script.beats:
        prompt = _image(beat).prompt
        assert "oil painting figure" not in prompt, prompt
        assert "oil" not in prompt, prompt
        assert "eve.png" not in prompt, prompt


def test_each_media_prompt_is_image_and_motion() -> None:
    script = _prompts()
    for beat in script.beats:
        assert [prompt.kind for prompt in beat.media_prompts] == [
            MediaKind.IMAGE,
            MediaKind.MOTION,
        ]
