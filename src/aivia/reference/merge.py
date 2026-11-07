"""Merge several reference analyses into one style spec and report conflicts.

A user may supply more than one example for the same output. The references
often agree on the important traits, such as a portrait aspect, a warm palette
or a measured pace, and disagree on the details. This module keeps the shared
traits, resolves each disagreement deterministically and records why the chosen
value won, so a later review can inspect or override the decision.

Unordered lists keep only the values every reference reported. Ordered lists,
such as beats or scenes, are taken whole from one reference because reordering
them would invent a structure no example had. Scalars are chosen by the
configured :class:`MergeStrategy`; the default prefers the analysis with the
highest confidence and breaks ties by input order.

The merged analysis never carries the original subject or the detected
reference text. Those are the parts that must not be copied, so they stay out
of the result even when every reference agrees on them.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, StrEnum
from typing import Any, cast, get_origin

from pydantic import BaseModel, Field
from pydantic import ValidationError as PydanticValidationError
from pydantic.fields import FieldInfo

from aivia.core.errors import ValidationError
from aivia.core.models.base import canonical_json
from aivia.reference.schema import (
    ReferenceAnalysis,
    ReferenceSpec,
    StyleSpec,
    _SpecModel,
)

__all__ = [
    "ConflictValue",
    "MergeConflict",
    "MergeReport",
    "MergeResult",
    "MergeStrategy",
    "merge_analyses",
    "merge_reference_specs",
    "merge_references",
    "merge_style_specs",
]


class MergeStrategy(StrEnum):
    """How a scalar disagreement is resolved."""

    CONFIDENCE = "confidence"
    FIRST = "first"
    LAST = "last"
    MAJORITY = "majority"


class ConflictValue(_SpecModel):
    """One reference's value for a conflicting field."""

    reference_id: str = ""
    value: Any = None


class MergeConflict(_SpecModel):
    """A disagreement between references and the documented resolution.

    ``resolution`` names the rule that chose ``chosen`` and ``reason`` explains
    the choice in words, so the report is useful without reading the code.
    """

    field: str
    values: list[ConflictValue] = Field(default_factory=list)
    chosen: Any = None
    resolution: str = ""
    reason: str = ""


class MergeResult(_SpecModel):
    """The merged analysis plus the report of how conflicts were resolved."""

    spec: ReferenceAnalysis = Field(default_factory=ReferenceAnalysis)
    conflicts: list[MergeConflict] = Field(default_factory=list)
    common: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    strategy: MergeStrategy = MergeStrategy.CONFIDENCE

    @property
    def has_conflicts(self) -> bool:
        """Whether any reference trait disagreed."""
        return bool(self.conflicts)

    def conflict_for(self, field: str) -> MergeConflict | None:
        """Return the conflict recorded for one dotted field path, if any."""
        return next((item for item in self.conflicts if item.field == field), None)

    def resolution_for(self, field: str) -> Any:
        """Return the chosen value for one field, or ``None`` when absent."""
        conflict = self.conflict_for(field)
        return None if conflict is None else conflict.chosen

    @property
    def resolved(self) -> dict[str, Any]:
        """The chosen value for every conflicting field, keyed by path."""
        return {item.field: item.chosen for item in self.conflicts}


MergeReport = MergeResult
"""Alias for :class:`MergeResult`."""


@dataclass(frozen=True)
class _Source:
    reference_id: str
    analysis: ReferenceAnalysis

    @property
    def confidence(self) -> float:
        value = self.analysis.confidence
        return 0.0 if value is None else value


_MISSING = object()

_SKIP_FIELDS = frozenset(
    {
        "schema_version",
        "reference_id",
        "kind",
        "confidence",
        "provider",
        "model",
        "subject",
        "detected_text",
        "references",
        "metadata",
    }
)

_UNORDERED_LISTS = frozenset(
    {
        "shot_types",
        "structure.shot_types",
        "structure.transitions",
        "palette.colors",
        "typography.fonts",
        "music.genres",
        "music.moods",
        "music.instruments",
    }
)


def merge_references(
    references: Sequence[Any] | Any,
    *,
    strategy: MergeStrategy | str = MergeStrategy.CONFIDENCE,
    name: str = "",
    description: str = "",
) -> MergeResult:
    """Merge one or more references and report every disagreement.

    A reference may be a :class:`~aivia.reference.schema.ReferenceSpec`, a
    :class:`~aivia.reference.schema.ReferenceAnalysis`, a
    :class:`~aivia.reference.schema.StyleSpec` or a mapping of any of them. A
    single value is accepted as well as a sequence. ``name`` and
    ``description`` override the merged top-level fields when supplied.
    """
    sources = _coerce_sources(references)
    chosen_strategy = _coerce_strategy(strategy)
    conflicts: list[MergeConflict] = []
    common: list[str] = []
    spec = _merge_analysis(
        sources,
        chosen_strategy,
        name=name,
        description=description,
        conflicts=conflicts,
        common=common,
    )
    conflicts.sort(key=lambda item: item.field)
    return MergeResult(
        spec=spec,
        conflicts=conflicts,
        common=sorted(set(common)),
        sources=[source.reference_id for source in sources],
        strategy=chosen_strategy,
    )


merge_analyses = merge_references
"""Alias for :func:`merge_references`."""

merge_style_specs = merge_references
"""Alias for :func:`merge_references`."""

merge_reference_specs = merge_references
"""Alias for :func:`merge_references`."""


def _coerce_strategy(value: MergeStrategy | str) -> MergeStrategy:
    if isinstance(value, MergeStrategy):
        return value
    try:
        return MergeStrategy(value)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in MergeStrategy)
        raise ValidationError(
            f"unknown merge strategy {value!r} (allowed: {allowed})",
            details={"strategy": str(value)},
        ) from exc


def _entries(values: Sequence[Any] | Any | None) -> list[Any]:
    if values is None:
        return []
    if isinstance(values, (str, bytes, bytearray, Mapping, BaseModel)):
        return [values]
    if isinstance(values, Sequence):
        return list(values)
    return [values]


def _coerce_sources(values: Sequence[Any] | Any | None) -> list[_Source]:
    entries = _entries(values)
    if not entries:
        raise ValidationError("at least one reference is required to merge")
    sources: list[_Source] = []
    seen: dict[str, int] = {}
    for index, entry in enumerate(entries, start=1):
        source = _coerce_source(entry, index)
        identifier = source.reference_id
        count = seen.get(identifier, 0) + 1
        seen[identifier] = count
        if count > 1:
            source = _Source(f"{identifier}-{count}", source.analysis)
        sources.append(source)
    return sources


def _coerce_source(value: Any, index: int) -> _Source:
    if isinstance(value, ReferenceSpec):
        return _Source(
            _reference_source_id(value, index),
            value.analysis.model_copy(deep=True),
        )
    if isinstance(value, ReferenceAnalysis):
        identifier = value.reference_id.strip() or f"reference-{index}"
        return _Source(identifier, value.model_copy(deep=True))
    if isinstance(value, StyleSpec):
        analysis = ReferenceAnalysis.model_validate(value.model_dump(mode="json"))
        identifier = analysis.reference_id.strip() or f"reference-{index}"
        return _Source(identifier, analysis)
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if isinstance(value, Mapping):
        data = dict(value)
        if _looks_like_reference(data):
            try:
                reference = ReferenceSpec.model_validate(data)
            except PydanticValidationError as exc:
                raise ValidationError(
                    "invalid reference for merge",
                    details={"index": index, "error": str(exc)},
                ) from exc
            return _Source(
                _reference_source_id(reference, index),
                reference.analysis.model_copy(deep=True),
            )
        try:
            analysis = ReferenceAnalysis.model_validate(data)
        except PydanticValidationError as exc:
            raise ValidationError(
                "invalid reference analysis for merge",
                details={"index": index, "error": str(exc)},
            ) from exc
        identifier = analysis.reference_id.strip() or f"reference-{index}"
        return _Source(identifier, analysis)
    raise ValidationError(
        "a reference to merge must be a reference, analysis, style spec or mapping, "
        f"got {type(value).__name__}",
        details={"index": index},
    )


def _looks_like_reference(data: Mapping[str, Any]) -> bool:
    return "analysis" in data or any(
        key in data
        for key in (
            "path",
            "url",
            "kind",
            "label",
            "owned",
            "content_hash",
            "mime_type",
        )
    )


def _reference_source_id(reference: ReferenceSpec, index: int) -> str:
    for candidate in (
        reference.label,
        reference.content_hash,
        reference.path,
        reference.url,
        reference.analysis.reference_id,
    ):
        text = candidate.strip()
        if text:
            return text
    return f"reference-{index}"


def _merge_analysis(
    sources: list[_Source],
    strategy: MergeStrategy,
    *,
    name: str,
    description: str,
    conflicts: list[MergeConflict],
    common: list[str],
) -> ReferenceAnalysis:
    entries = [(source, source.analysis) for source in sources]
    updates = _merge_fields(
        ReferenceAnalysis,
        entries,
        "",
        strategy,
        conflicts,
        common,
    )
    identifiers = _unique(source.reference_id for source in sources)
    updates["references"] = identifiers
    updates["metadata"] = _merge_metadata(sources, common)

    kinds = {
        source.analysis.kind for source in sources if source.analysis.kind is not None
    }
    updates["kind"] = next(iter(kinds)) if len(kinds) == 1 else None

    confidences = [
        source.analysis.confidence
        for source in sources
        if source.analysis.confidence is not None
    ]
    updates["confidence"] = min(confidences) if confidences else None

    providers = _unique(source.analysis.provider for source in sources)
    models = _unique(source.analysis.model for source in sources)
    updates["provider"] = providers[0] if len(providers) == 1 else ""
    updates["model"] = models[0] if len(models) == 1 else ""
    updates["subject"] = ""
    updates["detected_text"] = ""
    updates["reference_id"] = ""
    if name:
        updates["name"] = name
    if description:
        updates["description"] = description

    metadata = cast(dict[str, Any], updates["metadata"])
    metadata["merge"] = {
        "strategy": strategy.value,
        "sources": identifiers,
        "conflicts": len(conflicts),
    }
    if providers:
        metadata["providers"] = providers
    if models:
        metadata["models"] = models
    if kinds:
        metadata["kinds"] = sorted(kind.value for kind in kinds)

    return ReferenceAnalysis.model_validate(updates)


def _merge_fields(
    model_cls: type[BaseModel],
    entries: Sequence[tuple[_Source, BaseModel]],
    path: str,
    strategy: MergeStrategy,
    conflicts: list[MergeConflict],
    common: list[str],
) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    for name, info in model_cls.model_fields.items():
        if name in _SKIP_FIELDS:
            continue
        field_path = f"{path}.{name}" if path else name
        values = [(source, getattr(model, name)) for source, model in entries]

        nested = _nested_model(info.annotation)
        if nested is not None:
            nested_entries = [
                (source, cast(BaseModel, value)) for source, value in values
            ]
            updates[name] = _merge_fields(
                nested,
                nested_entries,
                field_path,
                strategy,
                conflicts,
                common,
            )
            continue

        if _is_list_field(info):
            updates[name] = _merge_list(
                field_path,
                values,
                unordered=field_path in _UNORDERED_LISTS,
                strategy=strategy,
                conflicts=conflicts,
                common=common,
            )
            continue

        merged = _merge_scalar(
            field_path,
            values,
            strategy,
            conflicts,
            common,
        )
        if merged is not _MISSING:
            updates[name] = merged
    return updates


def _nested_model(annotation: Any) -> type[BaseModel] | None:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    return None


def _is_list_field(info: FieldInfo) -> bool:
    return get_origin(info.annotation) is list or info.default_factory is list


def _merge_scalar(
    path: str,
    values: list[tuple[_Source, Any]],
    strategy: MergeStrategy,
    conflicts: list[MergeConflict],
    common: list[str],
) -> Any:
    candidates = [(source, value) for source, value in values if not _is_empty(value)]
    if not candidates:
        return _MISSING
    signatures = {_key(value) for _, value in candidates}
    if len(signatures) == 1:
        common.append(path)
        return deepcopy(candidates[0][1])

    source, chosen = _choose(candidates, strategy)
    conflicts.append(
        _conflict(
            path,
            candidates,
            chosen,
            strategy.value,
            reason=_choose_reason(strategy, source),
        )
    )
    return deepcopy(chosen)


def _merge_list(
    path: str,
    values: list[tuple[_Source, Any]],
    *,
    unordered: bool,
    strategy: MergeStrategy,
    conflicts: list[MergeConflict],
    common: list[str],
) -> list[Any]:
    candidates = [(source, list(value)) for source, value in values if value]
    if not candidates:
        return []

    if unordered:
        signatures = [
            frozenset(_key(item) for item in items) for _, items in candidates
        ]
        agreed = all(signature == signatures[0] for signature in signatures)
    else:
        ordered = [tuple(_key(item) for item in items) for _, items in candidates]
        agreed = all(signature == ordered[0] for signature in ordered)

    if agreed:
        common.append(path)
        return deepcopy(candidates[0][1])

    if unordered:
        sets = [{_key(item) for item in items} for _, items in candidates]
        intersection = sets[0].intersection(*sets[1:])
        chosen = [
            deepcopy(item) for item in candidates[0][1] if _key(item) in intersection
        ]
        if chosen:
            conflicts.append(
                _conflict(
                    path,
                    candidates,
                    chosen,
                    "intersection",
                    reason="kept only the values every reference reported",
                )
            )
            return chosen

    source, chosen_list = _choose(candidates, strategy)
    conflicts.append(
        _conflict(
            path,
            candidates,
            chosen_list,
            strategy.value,
            reason=_choose_reason(strategy, source),
        )
    )
    return deepcopy(chosen_list)


def _conflict(
    path: str,
    candidates: list[tuple[_Source, Any]],
    chosen: Any,
    resolution: str,
    *,
    reason: str,
) -> MergeConflict:
    return MergeConflict(
        field=path,
        values=[
            ConflictValue(
                reference_id=source.reference_id,
                value=deepcopy(value),
            )
            for source, value in candidates
        ],
        chosen=deepcopy(chosen),
        resolution=resolution,
        reason=reason,
    )


def _choose(
    candidates: list[tuple[_Source, Any]],
    strategy: MergeStrategy,
) -> tuple[_Source, Any]:
    if strategy is MergeStrategy.FIRST:
        return candidates[0]
    if strategy is MergeStrategy.LAST:
        return candidates[-1]
    if strategy is MergeStrategy.MAJORITY:
        counts = Counter(_key(value) for _, value in candidates)
        best = max(counts.values())
        winners = {signature for signature, count in counts.items() if count == best}
        subset = [
            (source, value) for source, value in candidates if _key(value) in winners
        ]
        return _highest_confidence(subset)
    return _highest_confidence(candidates)


def _highest_confidence(
    candidates: list[tuple[_Source, Any]],
) -> tuple[_Source, Any]:
    best = candidates[0]
    for candidate in candidates[1:]:
        if candidate[0].confidence > best[0].confidence:
            best = candidate
    return best


def _choose_reason(strategy: MergeStrategy, source: _Source) -> str:
    if strategy is MergeStrategy.FIRST:
        return f"kept the value from {source.reference_id}, the first reference"
    if strategy is MergeStrategy.LAST:
        return f"kept the value from {source.reference_id}, the last reference"
    if strategy is MergeStrategy.MAJORITY:
        return f"kept the most common value from {source.reference_id}"
    return (
        f"kept the value from {source.reference_id}, "
        "which has the highest analyser confidence"
    )


def _merge_metadata(sources: list[_Source], common: list[str]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    keys: list[str] = []
    for source in sources:
        for key in source.analysis.metadata:
            if key not in keys:
                keys.append(key)
    for key in keys:
        values = [
            (source, source.analysis.metadata[key])
            for source in sources
            if key in source.analysis.metadata
            and not _is_empty(source.analysis.metadata[key])
        ]
        if len(values) != len(sources):
            continue
        if len({_key(value) for _, value in values}) != 1:
            continue
        merged[key] = deepcopy(values[0][1])
        common.append(f"metadata.{key}")
    return merged


def _unique(values: Any) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in result:
            result.append(text)
    return result


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, dict, set, frozenset)):
        return len(value) == 0
    return False


def _key(value: Any) -> str:
    try:
        return canonical_json(_jsonable(value))
    except (TypeError, ValueError):
        return f"{type(value).__name__}:{value!r}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
