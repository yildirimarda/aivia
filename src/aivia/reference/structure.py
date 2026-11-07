"""Reproduce a reference structure: scene count and durations within tolerance.

A reference analysis carries a measured structure: ordered scenes, each with a
duration. This module generates new media that matches that structure: one clip
per scene, each probed for its actual duration and compared against the
reference scene within a tolerance. When the first generation deviates - the
scene count differs or any timed scene's duration falls outside the tolerance -
the whole structure is regenerated once. The result reports whether the
structure matched, how many regenerations were needed and the deviation, if
any.

The module never calls a provider or ffmpeg directly. The caller supplies a
scene generator and a duration probe, both injectable, which keeps the offline
suite free of the network and the real binaries.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from aivia.core.config.reference import (
    DEFAULT_STRUCTURE_MATCH_TOLERANCE_SECONDS,
)
from aivia.core.errors import ValidationError
from aivia.media.video_service import VideoOutcome
from aivia.providers.protocols.video import VideoRequest
from aivia.reference.schema import (
    ReferenceAnalysis,
    ReferenceSpec,
    SceneSpec,
)

__all__ = [
    "DEFAULT_STRUCTURE_MATCH_TOLERANCE_SECONDS",
    "StructureReproductionResult",
    "reproduce_structure",
]


def _reference_scenes(reference: ReferenceAnalysis | ReferenceSpec) -> list[SceneSpec]:
    """Return the ordered scenes carried by *reference*, if any."""
    structure = getattr(reference, "structure", None)
    if structure is None:
        analysis = getattr(reference, "analysis", None)
        if analysis is not None:
            structure = getattr(analysis, "structure", None)
    if structure is None:
        return []
    return list(getattr(structure, "scenes", []) or [])


def _scene_prompt(base_prompt: str, scene: SceneSpec, index: int) -> str:
    """Build the prompt used to render one scene of the reproduced structure."""
    parts = [base_prompt.strip()]
    if scene.description:
        parts.append(scene.description.strip())
    parts.append(f"scene {index + 1}")
    return " ".join(part for part in parts if part)


def _scene_request(request: VideoRequest, scene: SceneSpec, index: int) -> VideoRequest:
    """Clone *request* for one scene, aiming its duration at the reference."""
    update: dict[str, Any] = {
        "first_frame": None,
        "source_video": None,
        "count": 1,
        "prompt": _scene_prompt(request.prompt, scene, index),
    }
    if scene.duration_seconds is not None:
        update["duration_seconds"] = scene.duration_seconds
    return request.model_copy(update=update)


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _deviation(
    reference_scenes: Sequence[SceneSpec],
    actual_durations: Sequence[float | None],
    tolerance_seconds: float,
) -> str | None:
    """Return the first mismatch between the reference and generated durations."""
    if len(actual_durations) != len(reference_scenes):
        return (
            f"scene count {len(actual_durations)} does not match reference "
            f"{len(reference_scenes)}"
        )
    for index, (scene, actual) in enumerate(
        zip(reference_scenes, actual_durations, strict=False)
    ):
        reference = scene.duration_seconds
        if reference is None:
            continue
        if actual is None:
            return f"scene {index} duration was not measured"
        delta = abs(actual - reference)
        if delta > tolerance_seconds:
            return (
                f"scene {index} duration {actual:.3f}s deviates from reference "
                f"{reference:.3f}s by {delta:.3f}s > {tolerance_seconds:.3f}s"
            )
    return None


@dataclass(frozen=True, slots=True)
class StructureReproductionResult:
    """Outcome of reproducing a reference structure.

    ``scenes`` is the reference structure with each scene's actual generated
    duration filled in. ``scene_count`` is the number of scenes generated.
    ``matched`` is true when the scene count equals the reference and every
    timed scene's duration is within ``tolerance_seconds``. ``regenerations``
    is how many times the structure was regenerated. ``deviation`` describes
    the first mismatch, or ``None`` when matched. ``outcomes`` holds every
    generated clip, in order, from the final generation.
    """

    scenes: tuple[SceneSpec, ...]
    scene_count: int
    matched: bool
    regenerations: int
    deviation: str | None
    outcomes: tuple[VideoOutcome, ...]


async def _generate_all(
    generate_one: Callable[[VideoRequest, Any], Awaitable[VideoOutcome]],
    request: VideoRequest,
    scenes: Sequence[SceneSpec],
    *,
    screen_prompt: Callable[[str], Any] | None,
    token: Any,
) -> tuple[VideoOutcome, ...]:
    """Generate one clip per scene, each screened and aimed at its reference."""
    outcomes: list[VideoOutcome] = []
    for index, scene in enumerate(scenes):
        scene_request = _scene_request(request, scene, index)
        active_token = token
        if screen_prompt is not None:
            screened = screen_prompt(scene_request.prompt)
            if inspect.isawaitable(screened):
                screened = await screened
            active_token = screened
        outcome = await generate_one(scene_request, active_token)
        outcomes.append(outcome)
    return tuple(outcomes)


async def reproduce_structure(
    *,
    reference: ReferenceAnalysis | ReferenceSpec | None,
    request: VideoRequest,
    generate_one: Callable[[VideoRequest, Any], Awaitable[VideoOutcome]],
    probe_duration: Callable[[VideoOutcome], Awaitable[float | None]],
    tolerance_seconds: float = DEFAULT_STRUCTURE_MATCH_TOLERANCE_SECONDS,
    max_regenerations: int = 1,
    screen_prompt: Callable[[str], Any] | None = None,
    token: Any = None,
) -> StructureReproductionResult:
    """Generate media matching the reference scene count and durations.

    One clip is generated per reference scene, each aimed at the reference
    scene's duration. Every clip is probed with *probe_duration* and compared
    against its reference scene within *tolerance_seconds*. When the scene
    count differs or any timed scene's duration falls outside the tolerance,
    the whole structure is regenerated, up to *max_regenerations* times.

    Returns a :class:`StructureReproductionResult` reporting whether the
    structure matched, how many regenerations were needed and the deviation.
    """
    if reference is None:
        raise ValidationError("a reference is required to reproduce its structure")
    tolerance = _finite(tolerance_seconds, label="tolerance_seconds")
    if tolerance < 0.0:
        raise ValueError("tolerance_seconds must not be negative")
    if isinstance(max_regenerations, bool) or not isinstance(max_regenerations, int):
        raise TypeError("max_regenerations must be an integer")

    regen_count = 0
    reference_scenes = _reference_scenes(reference)
    if not reference_scenes:
        return StructureReproductionResult(
            scenes=(),
            scene_count=0,
            matched=True,
            regenerations=regen_count,
            deviation=None,
            outcomes=(),
        )

    outcomes: tuple[VideoOutcome, ...] = ()
    durations: tuple[float | None, ...] = ()

    async def generate() -> None:
        nonlocal outcomes, durations, regen_count
        generated = await _generate_all(
            generate_one,
            request,
            reference_scenes,
            screen_prompt=screen_prompt,
            token=token,
        )
        outcomes = generated
        measured = [await probe_duration(outcome) for outcome in generated]
        durations = tuple(measured)

    await generate()
    deviation = _deviation(reference_scenes, durations, tolerance)

    while deviation is not None and regen_count < max_regenerations:
        regen_count += 1
        await generate()
        deviation = _deviation(reference_scenes, durations, tolerance)

    matched = deviation is None
    filled: list[SceneSpec] = []
    for scene, actual in zip(reference_scenes, durations, strict=False):
        if actual is not None:
            filled.append(scene.model_copy(update={"duration_seconds": actual}))
        else:
            filled.append(scene)

    return StructureReproductionResult(
        scenes=tuple(filled),
        scene_count=len(outcomes),
        matched=matched,
        regenerations=regen_count,
        deviation=deviation,
        outcomes=outcomes,
    )
