"""Merge several references: shared traits, conflicts and documented choices."""

from __future__ import annotations

from typing import Any

import pytest

from aivia.core.errors import ValidationError
from aivia.reference import (
    MergeResult,
    MergeStrategy,
    ReferenceAnalysis,
    ReferenceKind,
    ReferenceSpec,
    StyleSpec,
    merge_references,
)


def _analysis(
    reference_id: str,
    *,
    confidence: float | None = 0.8,
    **overrides: Any,
) -> ReferenceAnalysis:
    data: dict[str, Any] = {
        "reference_id": reference_id,
        "kind": "image",
        "confidence": confidence,
        "palette": {"colors": ["#111111", "#222222"], "primary": "#111111"},
        "aspect": "9:16",
        "language": "en",
        "tone": "calm",
        "structure": {"hook": "Open with a question", "beats": ["hook", "body"]},
        "pacing": {"cuts_per_minute": 30.0},
    }
    data.update(overrides)
    return ReferenceAnalysis.model_validate(data)


def test_merge_keeps_traits_shared_by_every_reference() -> None:
    first = _analysis("first", confidence=0.9)
    second = _analysis("second", confidence=0.7)

    result = merge_references([first, second])

    assert result.sources == ["first", "second"]
    assert result.spec.references == ["first", "second"]
    assert result.spec.palette.colors == ["#111111", "#222222"]
    assert result.spec.palette.primary == "#111111"
    assert result.spec.aspect.ratio == "9:16"
    assert result.spec.language == "en"
    assert result.spec.tone == "calm"
    assert result.spec.structure.hook == "Open with a question"
    assert result.spec.structure.beats == ["hook", "body"]
    assert result.spec.pacing.cuts_per_minute == 30.0
    assert result.spec.kind is ReferenceKind.IMAGE
    assert result.spec.confidence == 0.7

    assert result.conflicts == []
    assert result.has_conflicts is False
    assert "palette.colors" in result.common
    assert "language" in result.common
    assert "pacing.cuts_per_minute" in result.common


def test_merge_reports_a_scalar_conflict_and_prefers_confidence() -> None:
    first = _analysis("first", confidence=0.9, pacing={"cuts_per_minute": 60.0})
    second = _analysis("second", confidence=0.4, pacing={"cuts_per_minute": 15.0})

    result = merge_references([first, second])

    conflict = result.conflict_for("pacing.cuts_per_minute")
    assert conflict is not None
    assert conflict.chosen == 60.0
    assert conflict.resolution == "confidence"
    assert conflict.reason == (
        "kept the value from first, which has the highest analyser confidence"
    )
    assert [(item.reference_id, item.value) for item in conflict.values] == [
        ("first", 60.0),
        ("second", 15.0),
    ]
    assert result.spec.pacing.cuts_per_minute == 60.0
    assert result.resolution_for("pacing.cuts_per_minute") == 60.0
    assert result.resolved == {"pacing.cuts_per_minute": 60.0}
    assert result.has_conflicts is True


def test_merge_first_strategy_resolves_by_input_order() -> None:
    first = _analysis("first", confidence=0.1, pacing={"cuts_per_minute": 12.0})
    second = _analysis("second", confidence=0.9, pacing={"cuts_per_minute": 48.0})

    result = merge_references([first, second], strategy=MergeStrategy.FIRST)

    conflict = result.conflict_for("pacing.cuts_per_minute")
    assert conflict is not None
    assert conflict.chosen == 12.0
    assert conflict.resolution == "first"
    assert conflict.reason == "kept the value from first, the first reference"
    assert result.spec.pacing.cuts_per_minute == 12.0


def test_merge_majority_strategy_uses_confidence_to_break_a_tie() -> None:
    first = _analysis("first", confidence=0.1, pacing={"cuts_per_minute": 10.0})
    second = _analysis("second", confidence=0.6, pacing={"cuts_per_minute": 10.0})
    third = _analysis("third", confidence=0.9, pacing={"cuts_per_minute": 90.0})

    result = merge_references([first, second, third], strategy=MergeStrategy.MAJORITY)

    conflict = result.conflict_for("pacing.cuts_per_minute")
    assert conflict is not None
    assert conflict.chosen == 10.0
    assert conflict.resolution == "majority"
    assert conflict.reason == "kept the most common value from second"


def test_merge_intersects_unordered_lists_and_reports_the_conflict() -> None:
    first = _analysis(
        "first",
        confidence=0.9,
        palette={"colors": ["#111111", "#222222"]},
    )
    second = _analysis(
        "second",
        confidence=0.4,
        palette={"colors": ["#222222", "#333333"]},
    )

    result = merge_references([first, second])

    assert result.spec.palette.colors == ["#222222"]
    conflict = result.conflict_for("palette.colors")
    assert conflict is not None
    assert conflict.chosen == ["#222222"]
    assert conflict.resolution == "intersection"
    assert conflict.reason == "kept only the values every reference reported"
    assert "palette.colors" not in result.common


def test_merge_takes_an_ordered_list_whole_from_one_reference() -> None:
    first = _analysis(
        "first",
        confidence=0.9,
        structure={"hook": "same", "beats": ["hook", "body"]},
    )
    second = _analysis(
        "second",
        confidence=0.4,
        structure={"hook": "same", "beats": ["hook", "body", "payoff"]},
    )

    result = merge_references([first, second])

    assert result.spec.structure.beats == ["hook", "body"]
    conflict = result.conflict_for("structure.beats")
    assert conflict is not None
    assert conflict.chosen == ["hook", "body"]
    assert conflict.resolution == "confidence"
    assert "structure.beats" not in result.common


def test_merge_falls_back_to_confidence_when_lists_do_not_overlap() -> None:
    first = _analysis(
        "first",
        confidence=0.9,
        palette={"colors": ["#111111"]},
    )
    second = _analysis(
        "second",
        confidence=0.4,
        palette={"colors": ["#222222"]},
    )

    result = merge_references([first, second])

    assert result.spec.palette.colors == ["#111111"]
    conflict = result.conflict_for("palette.colors")
    assert conflict is not None
    assert conflict.resolution == "confidence"
    assert conflict.chosen == ["#111111"]


def test_missing_values_do_not_create_a_conflict() -> None:
    learned = _analysis("learned", confidence=0.9)
    empty = ReferenceAnalysis.model_validate({"reference_id": "empty"})

    result = merge_references([learned, empty])

    assert result.spec.aspect.ratio == "9:16"
    assert result.spec.palette.colors == ["#111111", "#222222"]
    assert result.spec.pacing.cuts_per_minute == 30.0
    assert result.conflicts == []
    assert "aspect.ratio" in result.common
    assert "palette.colors" in result.common


def test_merge_never_copies_the_subject_or_detected_text() -> None:
    first = _analysis(
        "first",
        subject="The original title",
        detected_text="the original words",
    )
    second = _analysis(
        "second",
        subject="The original title",
        detected_text="the original words",
    )

    result = merge_references([first, second])

    assert result.spec.subject == ""
    assert result.spec.detected_text == ""
    assert result.conflicts == []


def test_merge_preserves_common_metadata_and_records_provenance() -> None:
    first = _analysis("first", metadata={"style": "warm", "word_count": 10})
    second = _analysis("second", metadata={"style": "warm", "word_count": 20})

    result = merge_references([first, second])

    assert result.spec.metadata["style"] == "warm"
    assert "metadata.style" in result.common
    assert "word_count" not in result.spec.metadata
    assert result.spec.metadata["merge"] == {
        "strategy": "confidence",
        "sources": ["first", "second"],
        "conflicts": 0,
    }
    assert result.spec.metadata["kinds"] == ["image"]


def test_merge_accepts_reference_specs_style_specs_and_mappings() -> None:
    analysis = _analysis("analysis", confidence=0.9)
    reference = ReferenceSpec(
        path="examples/reference.png",
        label="image-reference",
        analysis=analysis,
    )
    style = StyleSpec.model_validate(
        {
            "name": "Clean",
            "palette": {"colors": ["#111111", "#222222"]},
            "aspect": "9:16",
        }
    )
    mapping = {
        "reference_id": "mapped",
        "palette": {"colors": ["#111111", "#222222"]},
        "aspect": "9:16",
    }

    result = merge_references([reference, style, mapping])

    assert result.sources == ["image-reference", "reference-2", "mapped"]
    assert result.spec.references == ["image-reference", "reference-2", "mapped"]
    assert result.spec.name == "Clean"
    assert result.spec.palette.colors == ["#111111", "#222222"]
    assert result.spec.aspect.ratio == "9:16"


def test_merge_disambiguates_duplicate_source_ids() -> None:
    first = _analysis("same", confidence=0.9)
    second = _analysis("same", confidence=0.8)

    result = merge_references([first, second])

    assert result.sources == ["same", "same-2"]
    assert result.spec.references == ["same", "same-2"]


def test_merge_drops_a_kind_when_references_disagree() -> None:
    image = _analysis("image", kind="image")
    video = _analysis("video", kind="video")

    result = merge_references([image, video])

    assert result.spec.kind is None


def test_merge_round_trips_canonically() -> None:
    first = _analysis("first", confidence=0.9, pacing={"cuts_per_minute": 60.0})
    second = _analysis("second", confidence=0.4, pacing={"cuts_per_minute": 15.0})

    result = merge_references([first, second])
    restored = MergeResult.model_validate_json(result.to_json())

    assert restored == result
    assert restored.conflict_for("pacing.cuts_per_minute") is not None


def test_merge_requires_at_least_one_reference() -> None:
    with pytest.raises(ValidationError, match="at least one reference"):
        merge_references([])


def test_merge_rejects_an_unknown_strategy() -> None:
    with pytest.raises(ValidationError, match="unknown merge strategy"):
        merge_references([_analysis("first")], strategy="bogus")
