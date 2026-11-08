"""Character scene planning: one planned scene per beat.

Each beat is mapped to a character scene describing what the character does
(action), which pose cutout to show (pose), the facial expression (expression)
and where and how large to place it (position and size). Position and size are
clamped to the composition's safe area so the character never crosses a margin.

No AI call and no network request happen here: this is deterministic planning
over beat text, the character's pose library and the composition geometry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from aivia.characters.schema import Character
from aivia.compose.safe_area import safe_area_box
from aivia.compose.schema import Composition
from aivia.core.models.script import Beat

__all__ = [
    "CharacterScene",
    "action_of",
    "clamp_to_safe_area",
    "expression_of",
    "plan_scenes",
    "pose_of",
]

_DEFAULT_ACTION_KEYWORDS: tuple[str, ...] = (
    "wave",
    "idle",
    "think",
    "laugh",
    "smile",
    "point",
    "walk",
    "run",
    "stand",
    "sit",
)

_DEFAULT_EXPRESSIONS: tuple[str, ...] = (
    "happy",
    "sad",
    "angry",
    "surprised",
    "neutral",
    "excited",
    "worried",
)

_DEFAULT_POSITION: tuple[float, float] = (0.5, 0.5)
_DEFAULT_SIZE: tuple[float, float] = (0.4, 0.5)


def _beat_text(beat: Beat) -> str:
    parts = [beat.text]
    for line in beat.dialogue:
        parts.append(line.text)
    return " ".join(parts)


def _normalise_float(value: Any) -> float:
    number = float(value)
    if not number == number or number in (float("inf"), float("-inf")):
        raise ValueError("value must be finite")
    return number


class CharacterScene(BaseModel):
    """One planned character scene for a single beat.

    ``position`` is the normalized center (x, y) in 0..1 and ``size`` is the
    normalized (width, height) in 0..1. Both are clamped to the composition's
    safe area by :func:`plan_scenes`.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    beat_index: int = Field(ge=0)
    action: str = ""
    pose: str = ""
    expression: str = "neutral"
    position: tuple[float, float] = Field(default=_DEFAULT_POSITION)
    size: tuple[float, float] = Field(default=_DEFAULT_SIZE)

    @field_validator("position", "size")
    @classmethod
    def _normalise_pair(cls, value: tuple[float, float]) -> tuple[float, float]:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("position and size must be (x, y) pairs")
        x = _normalise_float(value[0])
        y = _normalise_float(value[1])
        if not (0.0 <= x <= 1.0) or not (0.0 <= y <= 1.0):
            raise ValueError("position and size components must be between 0 and 1")
        return (x, y)


def action_of(beat: Beat, *, actions: Sequence[str] | None = None) -> str:
    """Derive the character's action from a beat's text and dialogue.

    The first recognized action keyword found in the beat text (case
    insensitive) wins; otherwise ``"idle"`` is returned.
    """
    keywords = list(actions) if actions is not None else list(_DEFAULT_ACTION_KEYWORDS)
    text = _beat_text(beat).lower()
    for keyword in keywords:
        if keyword in text:
            return keyword
    return "idle"


def expression_of(beat: Beat, *, expressions: Sequence[str] | None = None) -> str:
    """Derive the character's expression from a beat's text and dialogue.

    The first recognized expression keyword found in the beat text (case
    insensitive) wins; otherwise ``"neutral"`` is returned.
    """
    keywords = (
        list(expressions) if expressions is not None else list(_DEFAULT_EXPRESSIONS)
    )
    text = _beat_text(beat).lower()
    for keyword in keywords:
        if keyword in text:
            return keyword
    return "neutral"


def pose_of(
    action: str,
    character: Character,
    *,
    action_poses: Mapping[str, str] | None = None,
) -> str:
    """Pick a pose name from the character's pose library for *action*.

    An explicit ``action_poses`` mapping wins, then the action name itself when
    it is a pose in the library, then ``"idle"`` when present, then the first
    pose in the library. An empty library yields an empty pose.
    """
    if action_poses:
        mapped = action_poses.get(action)
        if isinstance(mapped, str) and mapped.strip():
            return mapped.strip()
    key = action.strip()
    if key and key in character.pose_library:
        return key
    if "idle" in character.pose_library:
        return "idle"
    for name in character.pose_library:
        return name
    return ""


def clamp_to_safe_area(
    position: tuple[float, float],
    size: tuple[float, float],
    composition: Composition,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Clamp *position* and *size* so the character fits the safe area.

    Returns the clamped ``(position, size)``. Size is shrunk to fit the safe
    area and the center is moved so the whole character stays inside it.
    """
    cx, cy = position
    width, height = size
    canvas_w = float(composition.width)
    canvas_h = float(composition.height)
    if canvas_w <= 0.0 or canvas_h <= 0.0:
        return (cx, cy), (width, height)
    left, top, safe_w, safe_h = safe_area_box(composition)
    safe_x = left / canvas_w
    safe_y = top / canvas_h
    safe_w_norm = safe_w / canvas_w
    safe_h_norm = safe_h / canvas_h
    width = min(width, safe_w_norm)
    height = min(height, safe_h_norm)
    half_w = width / 2.0
    half_h = height / 2.0
    cx = min(max(cx, safe_x + half_w), safe_x + safe_w_norm - half_w)
    cy = min(max(cy, safe_y + half_h), safe_y + safe_h_norm - half_h)
    return (cx, cy), (width, height)


def plan_scenes(
    beats: Sequence[Beat],
    *,
    character: Character,
    composition: Composition,
    actions: Sequence[str] | None = None,
    expressions: Sequence[str] | None = None,
    action_poses: Mapping[str, str] | None = None,
    position: tuple[float, float] | None = None,
    size: tuple[float, float] | None = None,
) -> list[CharacterScene]:
    """Plan one :class:`CharacterScene` per beat.

    Each scene's action and expression are derived from the beat text, its pose
    is chosen from the character's pose library and its position and size are
    clamped to the composition's safe area.
    """
    planned: list[CharacterScene] = []
    for index, beat in enumerate(beats):
        action = action_of(beat, actions=actions)
        expression = expression_of(beat, expressions=expressions)
        pose = pose_of(action, character, action_poses=action_poses)
        scene_position = position if position is not None else _DEFAULT_POSITION
        scene_size = size if size is not None else _DEFAULT_SIZE
        clamped_position, clamped_size = clamp_to_safe_area(
            scene_position, scene_size, composition
        )
        planned.append(
            CharacterScene(
                beat_index=index,
                action=action,
                pose=pose,
                expression=expression,
                position=clamped_position,
                size=clamped_size,
            )
        )
    return planned
