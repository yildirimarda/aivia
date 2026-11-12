"""Feed measured performance back into category weights.

The analytics module is optional: the core never imports it. Performance is
reported per category (spend, views and cost per thousand views) by
:func:`aivia.core.performance.performance_by`. This module turns those
measurements into category weights: categories that deliver views cheaply gain
weight, expensive ones lose it.

Three concerns are separated:

* **Bounded learning rate** -- a category's weight moves toward its
  performance target by at most ``learning_rate`` of the gap in one update, so
  a single run cannot swing the mix.
* **Manual override** -- a category's weight can be pinned by hand; performance
  feedback then leaves it untouched until the override is removed.
* **Change log** -- every weight change records the category, the old and new
  weight, the reason (``"performance"`` or ``"override"``) and the time it was
  recorded, so the evolution of the weights is auditable.

The computation is deterministic and offline: it never calls a provider.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from aivia.core import paths
from aivia.core.errors import ConfigError
from aivia.core.models.base import canonical_json, utcnow
from aivia.core.performance import PerformanceBucket, performance_by

__all__ = [
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_WEIGHT",
    "MAX_WEIGHT",
    "MIN_WEIGHT",
    "CategoryWeight",
    "CategoryWeightChange",
    "CategoryWeights",
    "apply_performance_to_category_weights",
    "apply_performance_to_report",
    "category_weights_path",
    "load_category_weights",
    "save_category_weights",
]

DEFAULT_LEARNING_RATE = 0.2
"""The most a category weight may move toward its target in one update."""
DEFAULT_WEIGHT = 1.0
"""The weight of a category with no measured performance or override."""
MIN_WEIGHT = 0.0
"""Weights are clamped to be non-negative."""
MAX_WEIGHT = 1000.0
"""Weights are clamped to avoid runaway growth."""

DEFAULT_CATEGORY_WEIGHTS_FILENAME = "category_weights.json"
"""The weights sidecar under ``data/analytics``."""

_REASON_PERFORMANCE = "performance"
_REASON_OVERRIDE = "override"


def _validate_learning_rate(learning_rate: Any) -> float:
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)):
        raise ConfigError("learning rate must be a number")
    rate = float(learning_rate)
    if not math.isfinite(rate) or rate < 0:
        raise ConfigError("learning rate must be finite and non-negative")
    return rate


def _validate_weight(category: str, weight: Any) -> float:
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        raise ConfigError(f"category weight for {category!r} must be a number")
    number = float(weight)
    if not math.isfinite(number) or number < 0:
        raise ConfigError(
            f"category weight for {category!r} must be finite and non-negative"
        )
    return number


def _efficiency(bucket: PerformanceBucket) -> float:
    """Views per dollar spent: higher means the category delivers more views."""
    cost = bucket.cost_usd
    if cost <= 0:
        return 0.0
    return bucket.views / cost


def _targets(buckets: Sequence[PerformanceBucket]) -> dict[str, float]:
    """Target weight per category: efficiency relative to the mean efficiency.

    The mean target is 1.0 so the overall scale is preserved. When there is no
    meaningful signal (every reported category has zero efficiency) the result
    is empty, so weights are left unchanged.
    """
    efficiencies = {bucket.key: _efficiency(bucket) for bucket in buckets}
    if not efficiencies:
        return {}
    mean = math.fsum(efficiencies.values()) / len(efficiencies)
    if mean <= 0:
        return {}
    return {key: value / mean for key, value in efficiencies.items()}


class CategoryWeight(BaseModel):
    """One category's effective weight and where it came from."""

    model_config = ConfigDict(frozen=True)

    category: str
    """The category the weight belongs to."""

    weight: float = Field(ge=0.0)
    """The relative weight; higher means the category is favoured."""

    source: str
    """``"default"``, ``"performance"`` or ``"override"``."""


class CategoryWeightChange(BaseModel):
    """One recorded change to a category weight: a change log entry."""

    model_config = ConfigDict(frozen=True)

    category: str
    """The category whose weight changed."""

    old_weight: float
    """The weight before the change."""

    new_weight: float
    """The weight after the change."""

    reason: str
    """``"performance"`` when learning moved it, ``"override"`` when set by hand."""

    recorded_at: datetime
    """When the change was recorded (UTC)."""


class CategoryWeights(BaseModel):
    """Category weights, manual overrides and the append-only change log.

    ``weights`` holds the learned weights. ``overrides`` pins a category's
    weight so performance feedback cannot move it. ``changes`` records every
    weight change so the evolution of the weights is auditable.
    """

    model_config = ConfigDict(extra="ignore")

    weights: dict[str, float] = Field(default_factory=dict)
    """Learned weight per category, keyed by category name."""

    overrides: dict[str, float] = Field(default_factory=dict)
    """Manually pinned weight per category, keyed by category name."""

    changes: tuple[CategoryWeightChange, ...] = ()
    """The append-only change log, newest change not guaranteed to be last."""

    def effective(self, category: str) -> float:
        """Return the effective weight for *category*.

        A manual override wins, then a learned weight, then the default.
        """
        if category in self.overrides:
            return self.overrides[category]
        return self.weights.get(category, DEFAULT_WEIGHT)


def apply_performance_to_category_weights(
    weights: Mapping[str, float],
    buckets: Sequence[PerformanceBucket],
    *,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    overrides: Mapping[str, float] | None = None,
) -> tuple[dict[str, float], list[CategoryWeightChange]]:
    """Adjust *weights* from measured performance, bounded by *learning_rate*.

    Each reported category's target weight is its views-per-dollar efficiency
    relative to the mean efficiency of all reported categories, so the mean
    target is 1.0 and the overall scale is preserved. A category's weight moves
    toward its target by at most ``learning_rate`` of the gap in one update, so
    a single run cannot swing the mix. Categories without a performance bucket
    keep their weight; categories not yet known start at the default weight.

    Categories named in *overrides* are pinned: performance feedback leaves
    their weight untouched and records no change for them.

    Returns the new weights and the change log entries recording every change.
    """
    rate = _validate_learning_rate(learning_rate)
    override_map = {
        str(name): _validate_weight(str(name), value)
        for name, value in (overrides or {}).items()
    }
    targets = _targets(list(buckets))
    new_weights: dict[str, float] = {}
    changes: list[CategoryWeightChange] = []
    known = set(weights) | set(targets) | set(override_map)
    now = utcnow()
    for category in sorted(known):
        old = _validate_weight(category, weights.get(category, DEFAULT_WEIGHT))
        if category in override_map:
            new_weights[category] = old
            continue
        if category not in targets:
            new_weights[category] = old
            continue
        target = targets[category]
        new = old + rate * (target - old)
        new = min(max(new, MIN_WEIGHT), MAX_WEIGHT)
        new = _validate_weight(category, new)
        if new != old:
            changes.append(
                CategoryWeightChange(
                    category=category,
                    old_weight=old,
                    new_weight=new,
                    reason=_REASON_PERFORMANCE,
                    recorded_at=now,
                )
            )
        new_weights[category] = new
    return new_weights, changes


def apply_performance_to_report(
    engine: Any,
    weights: Mapping[str, float],
    *,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    overrides: Mapping[str, float] | None = None,
) -> tuple[dict[str, float], list[CategoryWeightChange]]:
    """Read performance by category from *engine* and adjust *weights*.

    A thin wrapper around :func:`apply_performance_to_category_weights` that
    pulls the category buckets from the cost ledger and metric snapshots via
    :func:`aivia.core.performance.performance_by`.
    """
    buckets = performance_by(engine, "category")
    return apply_performance_to_category_weights(
        weights, buckets, learning_rate=learning_rate, overrides=overrides
    )


def category_weights_path(
    data_dir: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Return the category weights sidecar path under ``data/analytics``."""
    root = paths.data_dir(environ) if data_dir is None else Path(data_dir)
    return Path(root).expanduser() / "analytics" / DEFAULT_CATEGORY_WEIGHTS_FILENAME


def _change_dicts(
    changes: Iterable[CategoryWeightChange | Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Serialize change records to dicts for robust model validation."""
    out: list[dict[str, Any]] = []
    for change in changes:
        if isinstance(change, CategoryWeightChange):
            out.append(
                {
                    "category": change.category,
                    "old_weight": change.old_weight,
                    "new_weight": change.new_weight,
                    "reason": change.reason,
                    "recorded_at": change.recorded_at.isoformat(),
                }
            )
        else:
            out.append(dict(change))
    return out


class CategoryWeightStore:
    """Read and write category weights, overrides and the change log atomically."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        data_dir: str | Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self._path = (
            category_weights_path(data_dir, environ=environ)
            if path is None
            else Path(path).expanduser()
        )

    @property
    def path(self) -> Path:
        """The JSON file backing this store."""
        return self._path

    def load(self) -> CategoryWeights | None:
        """Return the stored weights, or ``None`` when the file is absent."""
        if not self._path.exists():
            return None
        try:
            text = self._path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"cannot read category weights: {self._path}") from exc
        if not text.strip():
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(
                f"invalid category weights JSON: {self._path}: {exc}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise ConfigError(f"category weights root must be a mapping: {self._path}")
        return CategoryWeights.model_validate(dict(payload))

    def save(self, weights: CategoryWeights) -> CategoryWeights:
        """Write *weights* atomically and deterministically."""
        if not isinstance(weights, CategoryWeights):
            raise TypeError("weights must be a CategoryWeights")
        payload: dict[str, Any] = {
            "weights": dict(weights.weights),
            "overrides": dict(weights.overrides),
            "changes": [
                {
                    "category": change.category,
                    "old_weight": change.old_weight,
                    "new_weight": change.new_weight,
                    "reason": change.reason,
                    "recorded_at": change.recorded_at.isoformat(),
                }
                for change in weights.changes
            ],
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{self._path.name}.",
            suffix=".tmp",
            dir=self._path.parent,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(canonical_json(payload, indent=2))
                handle.write("\n")
            os.replace(temporary, self._path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
        return weights

    def apply_performance(
        self,
        engine: Any,
        *,
        learning_rate: float = DEFAULT_LEARNING_RATE,
    ) -> CategoryWeights:
        """Load, feed performance into the weights, persist and return them."""
        current = self.load() or CategoryWeights()
        new_weights, changes = apply_performance_to_category_weights(
            current.weights,
            performance_by(engine, "category"),
            learning_rate=learning_rate,
            overrides=current.overrides,
        )
        updated = CategoryWeights.model_validate(
            {
                "weights": new_weights,
                "overrides": dict(current.overrides),
                "changes": _change_dicts(list(current.changes) + changes),
            }
        )
        return self.save(updated)

    def set_override(
        self,
        category: str,
        weight: float,
    ) -> CategoryWeights:
        """Pin *category* to *weight*, recording the change, and persist."""
        category = _validate_mode_name(category)
        number = _validate_weight(category, weight)
        current = self.load() or CategoryWeights()
        old = current.effective(category)
        if number == old:
            return current
        new_change: dict[str, Any] = {
            "category": category,
            "old_weight": old,
            "new_weight": number,
            "reason": _REASON_OVERRIDE,
            "recorded_at": utcnow().isoformat(),
        }
        updated = CategoryWeights.model_validate(
            {
                "weights": dict(current.weights),
                "overrides": {**current.overrides, category: number},
                "changes": _change_dicts([*list(current.changes), new_change]),
            }
        )
        return self.save(updated)

    def remove_override(self, category: str) -> CategoryWeights:
        """Stop pinning *category* and return the stored weights."""
        category = _validate_mode_name(category)
        current = self.load() or CategoryWeights()
        if category not in current.overrides:
            return current
        updated = CategoryWeights.model_validate(
            {
                "weights": dict(current.weights),
                "overrides": {
                    name: value
                    for name, value in current.overrides.items()
                    if name != category
                },
                "changes": _change_dicts(current.changes),
            }
        )
        return self.save(updated)


def _validate_mode_name(name: object) -> str:
    if not isinstance(name, str):
        raise TypeError("category name must be a string")
    stripped = name.strip()
    if not stripped:
        raise ConfigError("category name must not be empty")
    return stripped


def load_category_weights(
    path: str | Path | None = None,
    *,
    data_dir: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> CategoryWeights | None:
    """Load the stored category weights, or ``None`` when absent."""
    return CategoryWeightStore(path, data_dir=data_dir, environ=environ).load()


def save_category_weights(
    weights: CategoryWeights,
    path: str | Path | None = None,
    *,
    data_dir: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> CategoryWeights:
    """Save *weights* to the sidecar and return them."""
    return CategoryWeightStore(path, data_dir=data_dir, environ=environ).save(weights)
