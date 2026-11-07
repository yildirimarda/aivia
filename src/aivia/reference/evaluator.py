"""Match evaluation between a reference style spec and a candidate.

A candidate style spec is scored against a reference across the style
spec sections - structure, pacing, palette, typography, music mood and
aspect - each producing a score in ``[0.0, 1.0]`` where higher means the
candidate more closely matches the reference. A weighted overall score is
computed from the section scores. When the overall score falls below a
configured threshold the candidate may be regenerated, up to
``max_regenerations`` times, before the match is refused.

The module never calls a provider. The caller supplies an injectable
regenerator that returns a replacement candidate, keeping the offline suite
free of the network.

All section scores are computed deterministically from the extracted spec
fields, so two runs with the same specs always produce the same match
decision.
"""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from aivia.core.errors import ValidationError
from aivia.reference.schema import (
    AspectSpec,
    MusicMoodSpec,
    PacingSpec,
    PaletteSpec,
    ShotType,
    StructureSpec,
    StyleSpec,
    TypographySpec,
)

__all__ = [
    "DEFAULT_MATCH_THRESHOLD",
    "DEFAULT_MATCH_WEIGHTS",
    "MatchEvaluatorError",
    "MatchResult",
    "SectionScore",
    "aspect_score",
    "evaluate_match",
    "music_mood_score",
    "pacing_score",
    "palette_score",
    "structure_score",
    "typography_score",
]

DEFAULT_MATCH_THRESHOLD = 0.7
"""Overall score at or above which a candidate matches the reference."""

_DEFAULT_SECTION_WEIGHTS = {
    "structure": 0.25,
    "pacing": 0.20,
    "palette": 0.15,
    "typography": 0.15,
    "music": 0.15,
    "aspect": 0.10,
}
DEFAULT_MATCH_WEIGHTS = dict(_DEFAULT_SECTION_WEIGHTS)
"""Default weight of each style spec section in the overall match score."""

_SECTION_NAMES = (
    "structure",
    "pacing",
    "palette",
    "typography",
    "music",
    "aspect",
)


class MatchEvaluatorError(ValidationError):
    """A candidate does not match the reference and cannot be regenerated."""

    code = "match_evaluation"
    retryable = False


@dataclass(frozen=True, slots=True)
class SectionScore:
    """The score of one style spec section in ``[0.0, 1.0]``."""

    name: str
    """The section name: ``structure``, ``pacing``, ``palette``,
    ``typography``, ``music`` or ``aspect``."""

    score: float
    """The section score in ``[0.0, 1.0]``. Higher means a closer match."""

    detail: str = ""
    """A human readable note on how the score was derived."""


@dataclass(frozen=True, slots=True)
class MatchResult:
    """The outcome of evaluating one candidate against a reference."""

    matched: bool
    """Whether the overall score reached the threshold."""

    score: float
    """The weighted overall score in ``[0.0, 1.0]``."""

    threshold: float
    """The threshold the score was compared against."""

    section_scores: tuple[SectionScore, ...]
    """The per-section scores that compose the overall score."""

    regenerations: int
    """How many times the regenerator was called before the decision."""

    refused: bool
    """Whether the match was refused after exhausting regenerations."""

    reason: str | None
    """A human readable reason when the match was refused, else ``None``."""


def _finite_ratio(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _clamped_ratio(left: float | None, right: float | None, label: str) -> float | None:
    """Return min/max in ``[0, 1]`` when both numbers are present, else None."""
    if left is None or right is None:
        return None
    if left == 0.0 and right == 0.0:
        return 1.0
    if left == 0.0 or right == 0.0:
        return 0.0
    return min(left, right) / max(left, right)


def _validate_weights(weights: Mapping[str, float], label: str) -> dict[str, float]:
    normalised: dict[str, float] = {}
    for name, weight in weights.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{label} weight names must be non-empty")
        _finite_ratio(weight, f"{label} weight for {name!r}")
        if weight < 0:
            raise ValueError(f"{label} weight for {name!r} must not be negative")
        normalised[name.strip()] = float(weight)
    if not normalised:
        raise ValueError(f"{label} weights must not be empty")
    total = sum(normalised.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{label} weights must sum to 1.0, got {total:g}")
    return normalised


def _validate_threshold(threshold: Any) -> float:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError("match threshold must be a number")
    level = float(threshold)
    if not math.isfinite(level) or level < 0.0 or level > 1.0:
        raise ValueError("match threshold must be between 0 and 1")
    return level


def _jaccard(left: Sequence[Any], right: Sequence[Any]) -> float | None:
    """Jaccard overlap of two sequences, or None when both are empty."""
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return None
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _equal_score(left: Any, right: Any) -> float | None:
    """1.0 when equal, 0.0 when only one is set, None when both empty."""
    left_empty = left is None or left == ""
    right_empty = right is None or right == ""
    if left_empty and right_empty:
        return None
    if left_empty or right_empty:
        return 0.0
    return 1.0 if left == right else 0.0


def _enum_score(left: Any, right: Any) -> float | None:
    """1.0 when equal enums, 0.0 when only one is set, None when both None."""
    if left is None and right is None:
        return None
    if left is None or right is None:
        return 0.0
    return 1.0 if left == right else 0.0


def _mean(scores: list[float]) -> float:
    if not scores:
        return 1.0
    return sum(scores) / len(scores)


def structure_score(reference: StructureSpec, candidate: StructureSpec) -> SectionScore:
    """Score how closely the candidate structure matches the reference."""
    reference_scenes = list(reference.scenes)
    candidate_scenes = list(candidate.scenes)

    if not reference_scenes and not candidate_scenes:
        count_score = 1.0
    elif not reference_scenes or not candidate_scenes:
        count_score = 0.0
    else:
        count_score = min(len(reference_scenes), len(candidate_scenes)) / max(
            len(reference_scenes), len(candidate_scenes)
        )

    reference_durations = [
        scene.duration_seconds
        for scene in reference_scenes
        if scene.duration_seconds is not None
    ]
    candidate_durations = [
        scene.duration_seconds
        for scene in candidate_scenes
        if scene.duration_seconds is not None
    ]
    duration_score = _clamped_ratio(
        sum(reference_durations) / len(reference_durations)
        if reference_durations
        else None,
        sum(candidate_durations) / len(candidate_durations)
        if candidate_durations
        else None,
        "duration",
    )
    if duration_score is None:
        duration_score = (
            1.0 if not reference_durations and not candidate_durations else 0.0
        )

    reference_types = [
        scene.shot_type for scene in reference_scenes if scene.shot_type is not None
    ]
    candidate_types = [
        scene.shot_type for scene in candidate_scenes if scene.shot_type is not None
    ]
    types_score = _jaccard(reference_types, candidate_types)
    if types_score is None:
        types_score = 1.0

    score = _mean([count_score, duration_score, types_score])
    detail = (
        f"scenes {len(reference_scenes)} vs {len(candidate_scenes)}, "
        f"shot types {_types_summary(reference_types)} vs "
        f"{_types_summary(candidate_types)}"
    )
    return SectionScore(name="structure", score=score, detail=detail)


def _types_summary(types: list[ShotType]) -> str:
    return ", ".join(sorted({t.value for t in types})) or "none"


def pacing_score(reference: PacingSpec, candidate: PacingSpec) -> SectionScore:
    """Score how closely the candidate pacing matches the reference."""
    parts: list[float] = []
    details: list[str] = []

    cut_score = _clamped_ratio(
        reference.cuts_per_minute, candidate.cuts_per_minute, "cuts per minute"
    )
    if cut_score is not None:
        parts.append(cut_score)
        details.append(
            f"cpm {reference.cuts_per_minute} vs {candidate.cuts_per_minute}"
        )
    elif reference.cuts_per_minute is None and candidate.cuts_per_minute is None:
        pass
    else:
        parts.append(0.0)

    shot_score = _clamped_ratio(
        reference.average_shot_seconds,
        candidate.average_shot_seconds,
        "average shot seconds",
    )
    if shot_score is not None:
        parts.append(shot_score)
        details.append(
            f"avg shot {reference.average_shot_seconds} vs "
            f"{candidate.average_shot_seconds}s"
        )
    elif (
        reference.average_shot_seconds is None
        and candidate.average_shot_seconds is None
    ):
        pass
    else:
        parts.append(0.0)

    energy_score = _equal_score(reference.energy, candidate.energy)
    if energy_score is not None:
        parts.append(energy_score)
        details.append(f"energy {reference.energy} vs {candidate.energy}")
    elif reference.energy is None and candidate.energy is None:
        pass
    else:
        parts.append(0.0)

    score = _mean(parts)
    return SectionScore(
        name="pacing", score=score, detail="; ".join(details) or "unspecified"
    )


def palette_score(reference: PaletteSpec, candidate: PaletteSpec) -> SectionScore:
    """Score how closely the candidate palette matches the reference."""
    parts: list[float] = []
    details: list[str] = []

    colors_score = _jaccard(reference.colors, candidate.colors)
    if colors_score is not None:
        parts.append(colors_score)
        details.append(f"colors {reference.colors} vs {candidate.colors}")
    elif not reference.colors and not candidate.colors:
        pass
    else:
        parts.append(0.0)

    temperature_score = _enum_score(reference.temperature, candidate.temperature)
    if temperature_score is not None:
        parts.append(temperature_score)
        details.append(
            f"temperature {reference.temperature} vs {candidate.temperature}"
        )
    elif reference.temperature is None and candidate.temperature is None:
        pass
    else:
        parts.append(0.0)

    saturation_score = _clamped_ratio(
        reference.saturation, candidate.saturation, "saturation"
    )
    if saturation_score is not None:
        parts.append(saturation_score)
        details.append(f"saturation {reference.saturation} vs {candidate.saturation}")
    elif reference.saturation is None and candidate.saturation is None:
        pass
    else:
        parts.append(0.0)

    contrast_score = _clamped_ratio(reference.contrast, candidate.contrast, "contrast")
    if contrast_score is not None:
        parts.append(contrast_score)
        details.append(f"contrast {reference.contrast} vs {candidate.contrast}")
    elif reference.contrast is None and candidate.contrast is None:
        pass
    else:
        parts.append(0.0)

    score = _mean(parts)
    return SectionScore(
        name="palette", score=score, detail="; ".join(details) or "unspecified"
    )


def typography_score(
    reference: TypographySpec, candidate: TypographySpec
) -> SectionScore:
    """Score how closely the candidate typography matches the reference."""
    parts: list[float] = []
    details: list[str] = []

    fonts_score = _jaccard(reference.typefaces, candidate.typefaces)
    if fonts_score is not None:
        parts.append(fonts_score)
        details.append(f"fonts {reference.typefaces} vs {candidate.typefaces}")
    elif not reference.typefaces and not candidate.typefaces:
        pass
    else:
        parts.append(0.0)

    size_score = _clamped_ratio(reference.font_size, candidate.font_size, "font size")
    if size_score is not None:
        parts.append(size_score)
        details.append(f"size {reference.font_size} vs {candidate.font_size}px")
    elif reference.font_size is None and candidate.font_size is None:
        pass
    else:
        parts.append(0.0)

    align_score = _enum_score(reference.align, candidate.align)
    if align_score is not None:
        parts.append(align_score)
        details.append(f"align {reference.align} vs {candidate.align}")
    elif reference.align is None and candidate.align is None:
        pass
    else:
        parts.append(0.0)

    case_score = _enum_score(reference.case, candidate.case)
    if case_score is not None:
        parts.append(case_score)
        details.append(f"case {reference.case} vs {candidate.case}")
    elif reference.case is None and candidate.case is None:
        pass
    else:
        parts.append(0.0)

    score = _mean(parts)
    return SectionScore(
        name="typography", score=score, detail="; ".join(details) or "unspecified"
    )


def music_mood_score(
    reference: MusicMoodSpec, candidate: MusicMoodSpec
) -> SectionScore:
    """Score how closely the candidate music mood matches the reference."""
    parts: list[float] = []
    details: list[str] = []

    genre_score = _jaccard(reference.genres, candidate.genres)
    if genre_score is not None:
        parts.append(genre_score)
        details.append(f"genres {reference.genres} vs {candidate.genres}")
    elif not reference.genres and not candidate.genres:
        pass
    else:
        parts.append(0.0)

    mood_score = _jaccard(reference.moods, candidate.moods)
    if mood_score is not None:
        parts.append(mood_score)
        details.append(f"moods {reference.moods} vs {candidate.moods}")
    elif not reference.moods and not candidate.moods:
        pass
    else:
        parts.append(0.0)

    bpm_score = _clamped_ratio(reference.bpm, candidate.bpm, "bpm")
    if bpm_score is not None:
        parts.append(bpm_score)
        details.append(f"bpm {reference.bpm} vs {candidate.bpm}")
    elif reference.bpm is None and candidate.bpm is None:
        pass
    else:
        parts.append(0.0)

    energy_score = _clamped_ratio(reference.energy, candidate.energy, "music energy")
    if energy_score is not None:
        parts.append(energy_score)
        details.append(f"energy {reference.energy} vs {candidate.energy}")
    elif reference.energy is None and candidate.energy is None:
        pass
    else:
        parts.append(0.0)

    valence_score = _clamped_ratio(reference.valence, candidate.valence, "valence")
    if valence_score is not None:
        parts.append(valence_score)
        details.append(f"valence {reference.valence} vs {candidate.valence}")
    elif reference.valence is None and candidate.valence is None:
        pass
    else:
        parts.append(0.0)

    score = _mean(parts)
    return SectionScore(
        name="music", score=score, detail="; ".join(details) or "unspecified"
    )


def aspect_score(reference: AspectSpec, candidate: AspectSpec) -> SectionScore:
    """Score how closely the candidate aspect matches the reference."""
    parts: list[float] = []
    details: list[str] = []

    ratio_score = _equal_score(reference.ratio, candidate.ratio)
    if ratio_score is not None:
        parts.append(ratio_score)
        details.append(
            f"ratio {reference.ratio or 'none'} vs {candidate.ratio or 'none'}"
        )
    elif not reference.ratio and not candidate.ratio:
        pass
    else:
        parts.append(0.0)

    orientation_score = _enum_score(reference.orientation, candidate.orientation)
    if orientation_score is not None:
        parts.append(orientation_score)
        details.append(
            f"orientation {reference.orientation or 'none'} vs "
            f"{candidate.orientation or 'none'}"
        )
    elif reference.orientation is None and candidate.orientation is None:
        pass
    else:
        parts.append(0.0)

    score = _mean(parts)
    return SectionScore(
        name="aspect", score=score, detail="; ".join(details) or "unspecified"
    )


_SECTION_SCORERS: dict[str, Callable[[StyleSpec, StyleSpec], SectionScore]] = {
    "structure": lambda reference, candidate: structure_score(
        reference.structure, candidate.structure
    ),
    "pacing": lambda reference, candidate: pacing_score(
        reference.pacing, candidate.pacing
    ),
    "palette": lambda reference, candidate: palette_score(
        reference.palette, candidate.palette
    ),
    "typography": lambda reference, candidate: typography_score(
        reference.typography, candidate.typography
    ),
    "music": lambda reference, candidate: music_mood_score(
        reference.music, candidate.music
    ),
    "aspect": lambda reference, candidate: aspect_score(
        reference.aspect, candidate.aspect
    ),
}


def _section_scores(
    reference: StyleSpec, candidate: StyleSpec, weights: Mapping[str, float]
) -> tuple[SectionScore, ...]:
    scores: list[SectionScore] = []
    for name in _SECTION_NAMES:
        if name not in weights:
            continue
        scorer = _SECTION_SCORERS[name]
        scores.append(scorer(reference, candidate))
    return tuple(scores)


def _overall_score(
    section_scores: Sequence[SectionScore], weights: Mapping[str, float]
) -> float:
    weighted = 0.0
    total_weight = 0.0
    for section in section_scores:
        weight = weights.get(section.name, 0.0)
        weighted += section.score * weight
        total_weight += weight
    if total_weight <= 0.0:
        return 0.0
    return weighted / total_weight


async def evaluate_match(
    *,
    reference: StyleSpec,
    candidate: StyleSpec,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
    weights: Mapping[str, float] | None = None,
    max_regenerations: int = 2,
    regenerate: Callable[[], Awaitable[StyleSpec]] | None = None,
) -> MatchResult:
    """Evaluate how well *candidate* matches *reference*.

    Computes per-section scores for structure, pacing, palette, typography,
    music mood and aspect, then a weighted overall score in ``[0.0, 1.0]``.
    When the overall score is below *threshold* and *regenerate* is supplied,
    the regenerator is called to produce a replacement candidate and the
    evaluation repeats. This happens at most *max_regenerations* times.

    ``threshold`` must be in ``[0, 1]``. *weights* defaults to
    :data:`DEFAULT_MATCH_WEIGHTS` and must sum to ``1.0``. *max_regenerations*
    of ``0`` means the first candidate below the threshold is refused
    immediately.

    Returns a :class:`MatchResult` reporting whether the candidate matched,
    the overall and per-section scores, and how many regenerations were used.
    """
    level = _validate_threshold(threshold)
    effective_weights = _validate_weights(
        dict(weights) if weights is not None else _DEFAULT_SECTION_WEIGHTS,
        "match",
    )
    if isinstance(max_regenerations, bool) or not isinstance(max_regenerations, int):
        raise TypeError("max_regenerations must be an integer")
    count = max_regenerations
    if count < 0:
        raise ValueError("max_regenerations must not be negative")

    current = candidate
    regenerations = 0

    while True:
        sections = _section_scores(reference, current, effective_weights)
        score = _overall_score(sections, effective_weights)
        if score >= level:
            return MatchResult(
                matched=True,
                score=score,
                threshold=level,
                section_scores=sections,
                regenerations=regenerations,
                refused=False,
                reason=None,
            )
        if regenerations >= count:
            reason = (
                f"match score {score:.3f} is below the threshold "
                f"{level:g} after {count} regeneration(s)"
            )
            return MatchResult(
                matched=False,
                score=score,
                threshold=level,
                section_scores=sections,
                regenerations=regenerations,
                refused=True,
                reason=reason,
            )
        if regenerate is None:
            reason = (
                f"match score {score:.3f} is below the threshold "
                f"{level:g} and no regenerator is configured"
            )
            return MatchResult(
                matched=False,
                score=score,
                threshold=level,
                section_scores=sections,
                regenerations=regenerations,
                refused=True,
                reason=reason,
            )
        current = await regenerate()
        regenerations += 1
