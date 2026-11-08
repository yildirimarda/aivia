"""Evaluating a candidate style spec against a reference scores structure,
pacing, look and mood, and regenerates up to a bounded limit on mismatch."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

import pytest

from aivia.reference import (
    DEFAULT_MATCH_THRESHOLD,
    DEFAULT_MATCH_WEIGHTS,
    AspectOrientation,
    AspectSpec,
    ColorTemperature,
    MatchResult,
    MusicMoodSpec,
    PacingSpec,
    PaletteSpec,
    SceneSpec,
    ShotType,
    StructureSpec,
    StyleSpec,
    TextAlignment,
    TextCase,
    TypographySpec,
    evaluate_match,
)

DEFAULT_WEIGHTS = dict(DEFAULT_MATCH_WEIGHTS)


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine)


def _structure(
    durations: list[float | None] | None = None,
    shot_types: list[ShotType] | None = None,
) -> StructureSpec:
    scenes: list[SceneSpec] = []
    count = 0
    if durations is not None:
        count = max(count, len(durations))
    if shot_types is not None:
        count = max(count, len(shot_types))
    for index in range(count):
        scenes.append(
            SceneSpec(
                index=index,
                duration_seconds=durations[index]
                if durations is not None and index < len(durations)
                else None,
                shot_type=shot_types[index]
                if shot_types is not None and index < len(shot_types)
                else None,
                description=f"scene {index}",
            )
        )
    return StructureSpec(scenes=scenes)


def _pacing(
    cuts_per_minute: float | None = None,
    average_shot_seconds: float | None = None,
    energy: float | None = None,
) -> PacingSpec:
    return PacingSpec(
        cuts_per_minute=cuts_per_minute,
        average_shot_seconds=average_shot_seconds,
        energy=energy,
    )


def _palette(
    colors: list[str] | None = None,
    temperature: ColorTemperature | str | None = None,
    saturation: float | None = None,
    contrast: float | None = None,
) -> PaletteSpec:
    return PaletteSpec(
        colors=colors or [],
        temperature=ColorTemperature(temperature) if temperature is not None else None,
        saturation=saturation,
        contrast=contrast,
    )


def _typography(
    font_family: str = "",
    fonts: list[str] | None = None,
    font_size: float | None = None,
    align: TextAlignment = TextAlignment.CENTER,
    case: TextCase = TextCase.NONE,
) -> TypographySpec:
    return TypographySpec(
        font_family=font_family,
        fonts=fonts or [],
        font_size=font_size,
        align=align,
        case=case,
    )


def _music(
    genres: list[str] | None = None,
    moods: list[str] | None = None,
    bpm: float | None = None,
    energy: float | None = None,
    valence: float | None = None,
) -> MusicMoodSpec:
    return MusicMoodSpec(
        genres=genres or [],
        moods=moods or [],
        bpm=bpm,
        energy=energy,
        valence=valence,
    )


def _aspect(
    ratio: str = "", orientation: AspectOrientation | None = None
) -> AspectSpec:
    return AspectSpec(ratio=ratio, orientation=orientation)


def _spec(
    structure: StructureSpec | None = None,
    pacing: PacingSpec | None = None,
    palette: PaletteSpec | None = None,
    typography: TypographySpec | None = None,
    music: MusicMoodSpec | None = None,
    aspect: AspectSpec | None = None,
) -> StyleSpec:
    return StyleSpec(
        structure=structure or _structure(),
        pacing=pacing or _pacing(),
        palette=palette or _palette(),
        typography=typography or _typography(),
        music=music or _music(),
        aspect=aspect or _aspect(),
    )


def test_identical_specs_match_without_regeneration() -> None:
    spec = _spec(
        structure=_structure([2.0, 3.0], [ShotType.WIDE, ShotType.CLOSE_UP]),
        pacing=_pacing(60.0, 2.0, 0.8),
        palette=_palette(["#112233", "#445566"], "warm", 0.6, 0.7),
        typography=_typography(
            "Serif", font_size=32.0, align=TextAlignment.LEFT, case=TextCase.UPPER
        ),
        music=_music(["jazz"], ["mellow"], 120.0, 0.5, 0.6),
        aspect=_aspect("9:16"),
    )
    result = _run(
        evaluate_match(reference=spec, candidate=spec, weights=DEFAULT_WEIGHTS)
    )
    assert isinstance(result, MatchResult)
    assert result.matched is True
    assert result.score == 1.0
    assert result.threshold == DEFAULT_MATCH_THRESHOLD
    assert result.regenerations == 0
    assert result.refused is False
    assert result.reason is None
    assert {section.name for section in result.section_scores} == {
        "structure",
        "pacing",
        "palette",
        "typography",
        "music",
        "aspect",
    }
    assert all(0.0 <= section.score <= 1.0 for section in result.section_scores)


def test_different_structure_scores_below_one() -> None:
    reference = _spec(structure=_structure([2.0, 2.0], [ShotType.WIDE]))
    candidate = _spec(
        structure=_structure([2.0, 2.0, 2.0], [ShotType.WIDE, ShotType.CLOSE_UP])
    )
    result = _run(
        evaluate_match(reference=reference, candidate=candidate, threshold=0.99)
    )
    assert result.matched is False
    assert result.score < 1.0
    structure = next(s for s in result.section_scores if s.name == "structure")
    assert structure.score < 1.0


def test_different_pacing_scores_below_one() -> None:
    reference = _spec(pacing=_pacing(60.0, 2.0, 0.8))
    candidate = _spec(pacing=_pacing(15.0, 8.0, 0.2))
    result = _run(
        evaluate_match(reference=reference, candidate=candidate, threshold=0.99)
    )
    assert result.matched is False
    pacing = next(s for s in result.section_scores if s.name == "pacing")
    assert pacing.score < 1.0


def test_different_palette_scores_below_one() -> None:
    reference = _spec(palette=_palette(["#112233"], "warm", 0.6, 0.7))
    candidate = _spec(palette=_palette(["#abcdef"], "cool", 0.1, 0.2))
    result = _run(
        evaluate_match(reference=reference, candidate=candidate, threshold=0.99)
    )
    assert result.matched is False
    palette = next(s for s in result.section_scores if s.name == "palette")
    assert palette.score < 1.0


def test_different_typography_scores_below_one() -> None:
    reference = _spec(
        typography=_typography("Serif", font_size=32.0, align=TextAlignment.LEFT)
    )
    candidate = _spec(
        typography=_typography("Mono", font_size=12.0, align=TextAlignment.RIGHT)
    )
    result = _run(
        evaluate_match(reference=reference, candidate=candidate, threshold=0.99)
    )
    assert result.matched is False
    typography = next(s for s in result.section_scores if s.name == "typography")
    assert typography.score < 1.0


def test_different_music_scores_below_one() -> None:
    reference = _spec(music=_music(["jazz"], ["mellow"], 120.0, 0.5, 0.6))
    candidate = _spec(music=_music(["metal"], ["aggressive"], 180.0, 0.9, 0.1))
    result = _run(
        evaluate_match(reference=reference, candidate=candidate, threshold=0.99)
    )
    assert result.matched is False
    music = next(s for s in result.section_scores if s.name == "music")
    assert music.score < 1.0


def test_different_aspect_scores_below_one() -> None:
    reference = _spec(aspect=_aspect("9:16"))
    candidate = _spec(aspect=_aspect("16:9"))
    result = _run(
        evaluate_match(reference=reference, candidate=candidate, threshold=0.99)
    )
    assert result.matched is False
    aspect = next(s for s in result.section_scores if s.name == "aspect")
    assert aspect.score == 0.0


def test_empty_specs_match() -> None:
    result = _run(
        evaluate_match(
            reference=StyleSpec(),
            candidate=StyleSpec(),
            weights=DEFAULT_WEIGHTS,
        )
    )
    assert result.matched is True
    assert result.score == 1.0
    assert result.regenerations == 0


def test_score_at_threshold_matches() -> None:
    reference = _spec(pacing=_pacing(60.0))
    candidate = _spec(pacing=_pacing(30.0))
    result = _run(
        evaluate_match(
            reference=reference,
            candidate=candidate,
            threshold=0.5,
            weights=DEFAULT_WEIGHTS,
        )
    )
    assert result.matched is True
    assert result.score >= 0.5


def test_below_threshold_regenerates_once_and_matches() -> None:
    reference = _spec(
        structure=_structure([2.0, 3.0], [ShotType.WIDE, ShotType.CLOSE_UP]),
        pacing=_pacing(60.0, 2.0, 0.8),
        palette=_palette(["#112233"], "warm", 0.6, 0.7),
        typography=_typography(
            "Serif", font_size=32.0, align=TextAlignment.LEFT, case=TextCase.UPPER
        ),
        music=_music(["jazz"], ["mellow"], 120.0, 0.5, 0.6),
        aspect=_aspect("9:16"),
    )
    calls = 0

    async def regenerate() -> StyleSpec:
        nonlocal calls
        calls += 1
        return _spec(
            structure=_structure([2.0, 3.0], [ShotType.WIDE, ShotType.CLOSE_UP]),
            pacing=_pacing(60.0, 2.0, 0.8),
            palette=_palette(["#112233"], "warm", 0.6, 0.7),
            typography=_typography(
                "Serif", font_size=32.0, align=TextAlignment.LEFT, case=TextCase.UPPER
            ),
            music=_music(["jazz"], ["mellow"], 120.0, 0.5, 0.6),
            aspect=_aspect("9:16"),
        )

    result = _run(
        evaluate_match(
            reference=reference,
            candidate=_spec(
                structure=_structure([9.0], [ShotType.STILL]),
                pacing=_pacing(5.0, 20.0, 0.1),
                palette=_palette(["#fedcba"], "cool", 0.1, 0.2),
                typography=_typography(
                    "Mono",
                    font_size=10.0,
                    align=TextAlignment.RIGHT,
                    case=TextCase.LOWER,
                ),
                music=_music(["metal"], ["aggressive"], 200.0, 0.9, 0.1),
                aspect=_aspect("16:9"),
            ),
            threshold=0.9,
            max_regenerations=2,
            regenerate=regenerate,
        )
    )
    assert result.matched is True
    assert result.score == 1.0
    assert result.regenerations == 1
    assert calls == 1
    assert result.refused is False


def test_below_threshold_refuses_without_regenerator() -> None:
    result = _run(
        evaluate_match(
            reference=_spec(
                music=_music(["jazz"], ["mellow"], 120.0, 0.5, 0.6),
                aspect=_aspect("9:16"),
            ),
            candidate=_spec(
                music=_music(["metal"], ["aggressive"], 200.0, 0.9, 0.1),
                aspect=_aspect("16:9"),
            ),
            threshold=0.9,
            max_regenerations=2,
        )
    )
    assert result.matched is False
    assert result.refused is True
    assert result.regenerations == 0
    assert result.reason is not None
    assert "below the threshold" in result.reason


def test_zero_max_regenerations_refuses_immediately() -> None:
    calls = 0

    async def regenerate() -> StyleSpec:
        nonlocal calls
        calls += 1
        return _spec(music=_music(bpm=120.0))

    result = _run(
        evaluate_match(
            reference=_spec(music=_music(["jazz"], ["mellow"], 120.0, 0.5, 0.6)),
            candidate=_spec(music=_music(["metal"], ["aggressive"], 200.0, 0.9, 0.1)),
            threshold=0.9,
            max_regenerations=0,
            regenerate=regenerate,
        )
    )
    assert result.matched is False
    assert result.refused is True
    assert result.regenerations == 0
    assert calls == 0


def test_refuses_after_exhausting_regenerations() -> None:
    calls = 0

    async def regenerate() -> StyleSpec:
        nonlocal calls
        calls += 1
        return _spec(
            music=_music(["metal"], ["aggressive"], 200.0, 0.9, 0.1),
            aspect=_aspect("16:9"),
        )

    result = _run(
        evaluate_match(
            reference=_spec(
                music=_music(["jazz"], ["mellow"], 120.0, 0.5, 0.6),
                aspect=_aspect("9:16"),
            ),
            candidate=_spec(
                music=_music(["metal"], ["aggressive"], 200.0, 0.9, 0.1),
                aspect=_aspect("16:9"),
            ),
            threshold=0.9,
            max_regenerations=2,
            regenerate=regenerate,
        )
    )
    assert result.matched is False
    assert result.refused is True
    assert result.regenerations == 2
    assert calls == 2
    assert result.reason is not None
    assert "after 2 regeneration(s)" in result.reason


def test_custom_weights_affect_overall_score() -> None:
    weights = {
        "structure": 1.0,
        "pacing": 0.0,
        "palette": 0.0,
        "typography": 0.0,
        "music": 0.0,
        "aspect": 0.0,
    }
    reference = _spec(structure=_structure([2.0]), music=_music(bpm=120.0))
    candidate = _spec(structure=_structure([5.0]), music=_music(bpm=120.0))
    result = _run(
        evaluate_match(reference=reference, candidate=candidate, weights=weights)
    )
    assert result.matched is True
    assert result.score < 1.0


def test_rejects_weights_not_summing_to_one() -> None:
    bad_weights = {"structure": 0.5, "pacing": 0.3}
    with pytest.raises(ValueError, match=r"weights must sum to 1\.0"):
        _run(
            evaluate_match(
                reference=StyleSpec(),
                candidate=StyleSpec(),
                weights=bad_weights,
            )
        )


def test_rejects_negative_weight() -> None:
    bad_weights = dict(DEFAULT_WEIGHTS)
    bad_weights["structure"] = -0.1
    bad_weights["pacing"] = 1.1
    with pytest.raises(ValueError, match="must not be negative"):
        _run(
            evaluate_match(
                reference=StyleSpec(),
                candidate=StyleSpec(),
                weights=bad_weights,
            )
        )


def test_rejects_empty_weights() -> None:
    with pytest.raises(ValueError, match="weights must not be empty"):
        _run(
            evaluate_match(
                reference=StyleSpec(),
                candidate=StyleSpec(),
                weights={},
            )
        )


def test_rejects_bad_threshold() -> None:
    with pytest.raises(TypeError):
        _run(
            evaluate_match(
                reference=StyleSpec(),
                candidate=StyleSpec(),
                threshold="high",  # type: ignore[arg-type]
            )
        )
    with pytest.raises(ValueError):
        _run(
            evaluate_match(reference=StyleSpec(), candidate=StyleSpec(), threshold=1.5)
        )
    with pytest.raises(ValueError):
        _run(
            evaluate_match(reference=StyleSpec(), candidate=StyleSpec(), threshold=-0.1)
        )


def test_rejects_bad_max_regenerations() -> None:
    with pytest.raises(TypeError):
        _run(
            evaluate_match(
                reference=StyleSpec(),
                candidate=StyleSpec(),
                max_regenerations="two",  # type: ignore[arg-type]
            )
        )
    with pytest.raises(ValueError):
        _run(
            evaluate_match(
                reference=StyleSpec(),
                candidate=StyleSpec(),
                max_regenerations=-1,
            )
        )


def test_default_threshold_is_positive() -> None:
    assert DEFAULT_MATCH_THRESHOLD > 0.0


def test_default_weights_sum_to_one() -> None:
    assert sum(DEFAULT_MATCH_WEIGHTS.values()) == pytest.approx(1.0)
    assert set(DEFAULT_MATCH_WEIGHTS) == {
        "structure",
        "pacing",
        "palette",
        "typography",
        "music",
        "aspect",
    }


def test_regenerator_receives_no_arguments_and_returns_spec() -> None:
    produced: list[StyleSpec] = []

    async def regenerate() -> StyleSpec:
        produced.append(
            _spec(
                structure=_structure([2.0, 3.0], [ShotType.WIDE, ShotType.CLOSE_UP]),
                pacing=_pacing(60.0, 2.0, 0.8),
                palette=_palette(["#112233"], "warm", 0.6, 0.7),
                typography=_typography(
                    "Serif",
                    font_size=32.0,
                    align=TextAlignment.LEFT,
                    case=TextCase.UPPER,
                ),
                music=_music(["jazz"], ["mellow"], 120.0, 0.5, 0.6),
                aspect=_aspect("9:16"),
            )
        )
        return produced[-1]

    result = _run(
        evaluate_match(
            reference=_spec(
                structure=_structure([2.0, 3.0], [ShotType.WIDE, ShotType.CLOSE_UP]),
                pacing=_pacing(60.0, 2.0, 0.8),
                palette=_palette(["#112233"], "warm", 0.6, 0.7),
                typography=_typography(
                    "Serif",
                    font_size=32.0,
                    align=TextAlignment.LEFT,
                    case=TextCase.UPPER,
                ),
                music=_music(["jazz"], ["mellow"], 120.0, 0.5, 0.6),
                aspect=_aspect("9:16"),
            ),
            candidate=_spec(
                structure=_structure([9.0], [ShotType.STILL]),
                pacing=_pacing(5.0, 20.0, 0.1),
                palette=_palette(["#fedcba"], "cool", 0.1, 0.2),
                typography=_typography(
                    "Mono",
                    font_size=10.0,
                    align=TextAlignment.RIGHT,
                    case=TextCase.LOWER,
                ),
                music=_music(["metal"], ["aggressive"], 200.0, 0.9, 0.1),
                aspect=_aspect("16:9"),
            ),
            threshold=0.9,
            max_regenerations=1,
            regenerate=regenerate,
        )
    )
    assert result.matched is True
    assert result.score == 1.0
    assert len(produced) == 1
    assert isinstance(produced[0], StyleSpec)
