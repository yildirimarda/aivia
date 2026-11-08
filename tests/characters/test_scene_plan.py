"""Character scene planning: one planned scene per beat.

Each beat is mapped to a character scene with an action, pose, expression,
position and size. Position and size are clamped to the composition's safe
area so the character never crosses a margin. Planning is deterministic and
offline: no AI call and no network request happen here.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

from aivia.characters import (
    Character,
    CharacterScene,
    action_of,
    clamp_to_safe_area,
    expression_of,
    plan_scenes,
    pose_of,
)
from aivia.compose.schema import Composition
from aivia.core.models.script import Beat, DialogueLine
from aivia.core.platforms import SafeArea


def _beat(text: str, *, dialogue: list[tuple[str, str]] | None = None) -> Beat:
    lines = [
        DialogueLine(speaker=speaker, text=line) for speaker, line in (dialogue or [])
    ]
    return Beat(text=text, dialogue=lines)


def _character(
    *,
    name: str = "Mascot",
    pose_library: dict[str, str] | None = None,
) -> Character:
    return Character.model_validate(
        {
            "name": name,
            "pose_library": pose_library or {},
        }
    )


def _composition(
    *,
    size: tuple[int, int] = (100, 100),
    safe_area: SafeArea | None = None,
) -> Composition:
    values: dict[str, object] = {
        "canvas": {"width": size[0], "height": size[1], "background": "#ffffff"},
        "duration_seconds": 3.0,
        "fps": 30,
    }
    if safe_area is not None:
        values["safe_area"] = safe_area
    return Composition.model_validate(values)


def test_plan_scenes_produces_one_scene_per_beat() -> None:
    beats = [_beat("one"), _beat("two"), _beat("three")]
    scenes = plan_scenes(
        beats,
        character=_character(pose_library={"idle": "idle.png"}),
        composition=_composition(),
    )

    assert len(scenes) == 3
    assert [scene.beat_index for scene in scenes] == [0, 1, 2]
    assert all(isinstance(scene, CharacterScene) for scene in scenes)


def test_plan_scenes_empty_beats_return_empty_list() -> None:
    scenes = plan_scenes(
        [],
        character=_character(),
        composition=_composition(),
    )

    assert scenes == []


def test_action_derived_from_beat_text() -> None:
    beat = _beat("The mascot waves hello")
    scene = plan_scenes(
        [beat],
        character=_character(pose_library={"wave": "wave.png"}),
        composition=_composition(),
    )[0]

    assert scene.action == "wave"


def test_action_derived_from_dialogue() -> None:
    beat = _beat("", dialogue=[("host", "Now I think about it")])
    scene = plan_scenes(
        [beat],
        character=_character(pose_library={"think": "think.png"}),
        composition=_composition(),
    )[0]

    assert scene.action == "think"


def test_action_defaults_to_idle() -> None:
    beat = _beat("Nothing special happens here")
    scene = plan_scenes(
        [beat],
        character=_character(pose_library={"idle": "idle.png"}),
        composition=_composition(),
    )[0]

    assert scene.action == "idle"


def test_expression_derived_from_beat_text() -> None:
    beat = _beat("The mascot is happy today")
    scene = plan_scenes(
        [beat],
        character=_character(),
        composition=_composition(),
    )[0]

    assert scene.expression == "happy"


def test_expression_defaults_to_neutral() -> None:
    beat = _beat("Just standing there")
    scene = plan_scenes(
        [beat],
        character=_character(),
        composition=_composition(),
    )[0]

    assert scene.expression == "neutral"


def test_pose_selected_from_library_by_action() -> None:
    beat = _beat("The mascot waves")
    scene = plan_scenes(
        [beat],
        character=_character(pose_library={"wave": "wave.png", "idle": "idle.png"}),
        composition=_composition(),
    )[0]

    assert scene.pose == "wave"


def test_pose_falls_back_to_idle() -> None:
    beat = _beat("The mascot dances")
    scene = plan_scenes(
        [beat],
        character=_character(pose_library={"idle": "idle.png", "wave": "wave.png"}),
        composition=_composition(),
    )[0]

    assert scene.pose == "idle"


def test_pose_falls_back_to_first_library_pose() -> None:
    beat = _beat("The mascot dances")
    scene = plan_scenes(
        [beat],
        character=_character(pose_library={"wave": "wave.png"}),
        composition=_composition(),
    )[0]

    assert scene.pose == "wave"


def test_pose_empty_when_library_empty() -> None:
    beat = _beat("The mascot dances")
    scene = plan_scenes(
        [beat],
        character=_character(pose_library={}),
        composition=_composition(),
    )[0]

    assert scene.pose == ""


def test_action_poses_mapping_overrides_pose() -> None:
    beat = _beat("The mascot dances")
    scene = plan_scenes(
        [beat],
        character=_character(pose_library={"spin": "spin.png"}),
        composition=_composition(),
        actions=("dance",),
        action_poses={"dance": "spin"},
    )[0]

    assert scene.action == "dance"
    assert scene.pose == "spin"


def test_position_and_size_are_inside_safe_area() -> None:
    safe_area = SafeArea(top=0.2, right=0.2, bottom=0.2, left=0.2)
    beat = _beat("The mascot waves")
    scene = plan_scenes(
        [beat],
        character=_character(pose_library={"wave": "wave.png"}),
        composition=_composition(size=(100, 100), safe_area=safe_area),
    )[0]

    cx, cy = scene.position
    w, h = scene.size
    # Safe area on a 100x100 canvas with 20% margins is (20, 20, 60, 60).
    assert 20.0 <= cx * 100 <= 80.0
    assert 20.0 <= cy * 100 <= 80.0
    assert cx - w / 2 >= 0.2
    assert cy - h / 2 >= 0.2
    assert cx + w / 2 <= 0.8
    assert cy + h / 2 <= 0.8


def test_position_clamped_to_safe_area_margins() -> None:
    safe_area = SafeArea(top=0.2, right=0.2, bottom=0.2, left=0.2)
    beat = _beat("The mascot waves")
    scene = plan_scenes(
        [beat],
        character=_character(pose_library={"wave": "wave.png"}),
        composition=_composition(size=(100, 100), safe_area=safe_area),
        position=(0.05, 0.05),
        size=(0.2, 0.2),
    )[0]

    cx, cy = scene.position
    # Centered at (0.05, 0.05) with size 0.2 would cross the margin; clamped so
    # the box stays inside the safe area (20..80 on a 100 canvas).
    assert cx == pytest.approx(0.3)
    assert cy == pytest.approx(0.3)


def test_size_shrunk_to_fit_safe_area() -> None:
    safe_area = SafeArea(top=0.2, right=0.2, bottom=0.2, left=0.2)
    beat = _beat("The mascot waves")
    scene = plan_scenes(
        [beat],
        character=_character(pose_library={"wave": "wave.png"}),
        composition=_composition(size=(100, 100), safe_area=safe_area),
        position=(0.5, 0.5),
        size=(1.0, 1.0),
    )[0]

    w, h = scene.size
    # The safe area is 60x60 on a 100 canvas, so the size is capped at 0.6.
    assert w == 0.6
    assert h == 0.6


def test_without_safe_area_position_stays_on_canvas() -> None:
    beat = _beat("The mascot waves")
    scene = plan_scenes(
        [beat],
        character=_character(pose_library={"wave": "wave.png"}),
        composition=_composition(size=(100, 100)),
        position=(0.9, 0.9),
        size=(0.4, 0.4),
    )[0]

    cx, cy = scene.position
    w, h = scene.size
    # No safe area: the full canvas is available, center clamped so the box fits.
    assert 0.2 <= cx <= 0.8
    assert 0.2 <= cy <= 0.8
    assert cx + w / 2 <= 1.0
    assert cy + h / 2 <= 1.0


def test_position_and_size_components_must_be_zero_to_one() -> None:
    with pytest.raises(PydanticValidationError):
        CharacterScene(beat_index=0, position=(1.5, 0.5))
    with pytest.raises(PydanticValidationError):
        CharacterScene(beat_index=0, size=(0.5, -0.1))


def test_beat_index_must_be_non_negative() -> None:
    with pytest.raises(PydanticValidationError):
        CharacterScene(beat_index=-1)


def test_action_of_recognizes_keywords() -> None:
    assert action_of(_beat("The mascot waves")) == "wave"
    assert action_of(_beat("The mascot sits down")) == "sit"
    assert action_of(_beat("Quiet moment")) == "idle"


def test_action_of_accepts_custom_keywords() -> None:
    assert action_of(_beat("The mascot spins"), actions=("spin", "idle")) == "spin"


def test_expression_of_recognizes_keywords() -> None:
    assert expression_of(_beat("The mascot is sad")) == "sad"
    assert expression_of(_beat("All is well")) == "neutral"


def test_expression_of_accepts_custom_expressions() -> None:
    assert (
        expression_of(
            _beat("The mascot is glowing"), expressions=("glowing", "neutral")
        )
        == "glowing"
    )


def test_pose_of_uses_action_when_in_library() -> None:
    character = _character(pose_library={"wave": "wave.png"})
    assert pose_of("wave", character) == "wave"


def test_pose_of_falls_back_to_idle() -> None:
    character = _character(pose_library={"idle": "idle.png", "wave": "wave.png"})
    assert pose_of("dance", character) == "idle"


def test_pose_of_uses_first_pose_when_no_idle() -> None:
    character = _character(pose_library={"wave": "wave.png"})
    assert pose_of("dance", character) == "wave"


def test_pose_of_empty_when_library_empty() -> None:
    character = _character(pose_library={})
    assert pose_of("wave", character) == ""


def test_clamp_to_safe_area_keeps_box_inside() -> None:
    composition = _composition(
        size=(100, 100), safe_area=SafeArea(left=0.2, right=0.2, top=0.2, bottom=0.2)
    )
    position, size = clamp_to_safe_area((0.5, 0.5), (1.0, 1.0), composition)

    assert size == (0.6, 0.6)
    assert position == (0.5, 0.5)


def test_clamp_to_safe_area_moves_center_inside() -> None:
    composition = _composition(
        size=(100, 100), safe_area=SafeArea(left=0.2, right=0.2, top=0.2, bottom=0.2)
    )
    position, size = clamp_to_safe_area((0.1, 0.1), (0.2, 0.2), composition)

    # Center (0.1, 0.1) with size 0.2 crosses the left/top margin; clamped to
    # (0.3, 0.3) so the box sits inside the safe area.
    assert position[0] == pytest.approx(0.3)
    assert position[1] == pytest.approx(0.3)
    assert size == (0.2, 0.2)


def test_scenes_keep_beat_order_and_independent_fields() -> None:
    beats = [_beat("The mascot waves"), _beat("The mascot sits")]
    scenes = plan_scenes(
        beats,
        character=_character(pose_library={"wave": "wave.png", "sit": "sit.png"}),
        composition=_composition(),
    )

    assert scenes[0].action == "wave"
    assert scenes[0].pose == "wave"
    assert scenes[1].action == "sit"
    assert scenes[1].pose == "sit"
    assert scenes[0].beat_index == 0
    assert scenes[1].beat_index == 1


def test_plan_scenes_round_trips_through_model_dump() -> None:
    beat = _beat("The mascot waves")
    scene = plan_scenes(
        [beat],
        character=_character(pose_library={"wave": "wave.png"}),
        composition=_composition(),
    )[0]
    restored = CharacterScene.model_validate(scene.model_dump(mode="json"))

    assert restored == scene
