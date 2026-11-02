"""Cross-object validation for compositions.

The schema validates each object on its own. This module adds the checks that
need the whole composition: unique ids, every timing window inside the
composition duration, referenced assets that exist, ordered keyframes and
keyframe values inside their property ranges. It returns every problem it can
find instead of stopping at the first one.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from aivia.compose.keyframes import Easing, KeyframeTrack, LayerKeyframes
from aivia.compose.safe_area import warn_safe_area_violations
from aivia.compose.schema import Composition, Layer
from aivia.core.errors import ValidationError

__all__ = [
    "AssetCheck",
    "AssetLookup",
    "CompositionIssue",
    "CompositionIssues",
    "CompositionValidationError",
    "validate_composition",
    "validate_composition_or_raise",
]

AssetCheck = Callable[[str], bool]
AssetLookup = Collection[str] | Mapping[str, Any] | AssetCheck | Path

_EPSILON = 1e-9

_KEYFRAME_CONTAINERS = ("keyframes", "animation", "animations")
_KEYFRAME_LAYER_FIELDS = (
    "keyframes",
    "position_keyframes",
    "scale_keyframes",
    "rotation_keyframes",
    "opacity_keyframes",
)
_KEYFRAME_TIME_KEYS = (
    "time_seconds",
    "time",
    "t",
    "at",
    "offset_seconds",
    "offset",
    "start_seconds",
    "start",
)
_KEYFRAME_VALUE_KEYS = ("value", "v", "to", "target")
_KEYFRAME_EASING_KEYS = ("easing", "ease", "timing", "curve")

_KEYFRAME_RANGES: dict[str, tuple[float | None, float | None]] = {
    "alpha": (0.0, 1.0),
    "height": (0.0, None),
    "opacity": (0.0, 1.0),
    "progress": (0.0, 1.0),
    "scale": (0.0, None),
    "scale_x": (0.0, None),
    "scale_y": (0.0, None),
    "width": (0.0, None),
}

_NUMERIC_KEYFRAME_PROPERTIES = frozenset(
    {
        "angle",
        "blur",
        "brightness",
        "contrast",
        "font_size",
        "gain",
        "gain_db",
        "playback_rate",
        "rotation",
        "saturation",
        "speed",
        "volume",
        "volume_db",
        "x",
        "y",
        *_KEYFRAME_RANGES,
    }
)


@dataclass(frozen=True, slots=True)
class CompositionIssue:
    """One problem found while validating a composition."""

    path: str
    message: str
    code: str = "invalid"

    def __str__(self) -> str:
        return f"{self.path}: {self.message}" if self.path else self.message

    def to_dict(self) -> dict[str, str]:
        """Return the issue as a JSON-compatible mapping."""
        return {"path": self.path, "message": self.message, "code": self.code}


class CompositionIssues(list[CompositionIssue]):
    """Every issue found in one validation pass.

    It behaves like a plain list, so callers can iterate, index or compare it
    with ``[]``. The helpers make the common cases explicit.
    """

    @property
    def ok(self) -> bool:
        """Whether the composition has no issues."""
        return not self

    @property
    def errors(self) -> list[CompositionIssue]:
        """The issues as a plain list."""
        return list(self)

    def raise_for_errors(self) -> None:
        """Raise :class:`CompositionValidationError` when any issue exists."""
        if self:
            raise CompositionValidationError(self)


class CompositionValidationError(ValidationError):
    """A composition failed one or more cross-object checks."""

    code = "composition_invalid"

    def __init__(self, issues: Sequence[CompositionIssue]) -> None:
        self.issues: tuple[CompositionIssue, ...] = tuple(issues)
        message = "composition is invalid: " + "; ".join(str(issue) for issue in issues)
        super().__init__(
            message,
            details={"errors": [issue.to_dict() for issue in self.issues]},
        )


def _as_finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _location(loc: Sequence[Any]) -> str:
    parts: list[str] = []
    for part in loc:
        if isinstance(part, int) and not isinstance(part, bool):
            parts.append(f"[{part}]")
        elif parts:
            parts.append(f".{part}")
        else:
            parts.append(str(part))
    return "".join(parts)


def _schema_issue(error: Mapping[str, Any]) -> CompositionIssue:
    loc = error.get("loc", ())
    path = _location(loc) if isinstance(loc, Sequence) else str(loc)
    message = str(error.get("msg", "invalid value"))
    return CompositionIssue(path=path, message=message, code="schema")


def _check_duplicates(
    items: Sequence[Any],
    *,
    kind: str,
    collection: str,
    issues: CompositionIssues,
) -> None:
    seen: set[str] = set()
    for index, item in enumerate(items):
        identifier = item.id
        if identifier in seen:
            issues.append(
                CompositionIssue(
                    path=f"{collection}[{index}].id",
                    message=f"duplicate {kind} id {identifier!r}",
                    code="duplicate_id",
                )
            )
        else:
            seen.add(identifier)


def _check_window(
    *,
    kind: str,
    collection: str,
    index: int,
    start: float,
    end: float | None,
    duration: float,
    issues: CompositionIssues,
) -> None:
    path = f"{collection}[{index}]"
    if start >= duration:
        issues.append(
            CompositionIssue(
                path=path,
                message=(
                    f"{kind} starts at {start:g}s, at or after the composition "
                    f"duration of {duration:g}s"
                ),
                code="timing",
            )
        )
    if end is not None and end > duration + _EPSILON:
        issues.append(
            CompositionIssue(
                path=path,
                message=(
                    f"{kind} ends at {end:g}s, after the composition duration "
                    f"of {duration:g}s"
                ),
                code="timing",
            )
        )


def _reference_source(reference: Any) -> str | None:
    if isinstance(reference, str):
        source = reference
    elif isinstance(reference, Mapping):
        source = None
        for key in ("source", "path", "file", "asset", "asset_id", "mask"):
            value = reference.get(key)
            if isinstance(value, str) and value.strip():
                source = value
                break
    else:
        source = getattr(reference, "source", None)
    if isinstance(source, str) and source.strip():
        return source.strip()
    return None


def _layer_source(layer: Layer) -> str | None:
    source = getattr(layer, "source", None)
    if isinstance(source, str):
        return source.strip()
    extras = layer.model_extra or {}
    for key in ("source", "path", "file", "asset", "asset_id"):
        value = extras.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _mask_source(layer: Layer) -> str | None:
    mask = getattr(layer, "mask", None)
    if mask is None:
        extras = layer.model_extra or {}
        for key in ("mask", "image_mask", "mask_source"):
            if key in extras:
                mask = extras[key]
                break
    return _reference_source(mask)


def _asset_checker(
    assets: AssetLookup | None,
    asset_exists: AssetCheck | None,
) -> AssetCheck | None:
    if asset_exists is not None:
        return asset_exists
    if assets is None:
        return None
    if isinstance(assets, (str, Path)):
        root = Path(assets)

        def check_path(source: str) -> bool:
            candidate = Path(source)
            if candidate.is_absolute():
                return candidate.exists()
            return (root / candidate).exists()

        return check_path
    if isinstance(assets, Mapping):
        known = {str(key) for key in assets}
    elif callable(assets):
        return assets
    else:
        known = {str(item) for item in assets}
    return known.__contains__


def _check_assets(
    composition: Composition,
    checker: AssetCheck,
    issues: CompositionIssues,
) -> None:
    for index, layer in enumerate(composition.layers):
        source = _layer_source(layer)
        if source and not checker(source):
            issues.append(
                CompositionIssue(
                    path=f"layers[{index}].source",
                    message=f"asset {source!r} does not exist",
                    code="missing_asset",
                )
            )
        mask_source = _mask_source(layer)
        if mask_source and not checker(mask_source):
            issues.append(
                CompositionIssue(
                    path=f"layers[{index}].mask.source",
                    message=f"asset {mask_source!r} does not exist",
                    code="missing_asset",
                )
            )
    for index, track in enumerate(composition.audio_tracks):
        if not checker(track.source):
            issues.append(
                CompositionIssue(
                    path=f"audio_tracks[{index}].source",
                    message=f"asset {track.source!r} does not exist",
                    code="missing_asset",
                )
            )


def _keyframe_tracks(layer: Layer, index: int) -> Iterator[tuple[str, str, Any]]:
    fields = type(layer).model_fields
    for name in _KEYFRAME_LAYER_FIELDS:
        if name not in fields:
            continue
        raw = getattr(layer, name, None)
        if raw is None:
            continue
        path = f"layers[{index}].{name}"
        if name == "keyframes":
            if isinstance(raw, LayerKeyframes):
                for prop, track in raw.tracks.items():
                    yield f"{path}.{prop.value}", prop.value, track.keyframes
            elif isinstance(raw, Mapping):
                for prop, frames in raw.items():
                    yield f"{path}.{prop}", str(prop), frames
            else:
                yield path, "value", raw
        else:
            property_name = name[: -len("_keyframes")]
            frames = raw.keyframes if isinstance(raw, KeyframeTrack) else raw
            yield path, property_name, frames

    extras = layer.model_extra or {}
    for key, raw in extras.items():
        if key in _KEYFRAME_LAYER_FIELDS:
            continue
        path = f"layers[{index}].{key}"
        if key in _KEYFRAME_CONTAINERS:
            if isinstance(raw, Mapping):
                for prop, frames in raw.items():
                    yield f"{path}.{prop}", str(prop), frames
            else:
                yield path, "value", raw
        elif key.endswith("_keyframes"):
            yield path, key[: -len("_keyframes")], raw
        elif key.endswith("_animation"):
            yield path, key[: -len("_animation")], raw


def _keyframe_time(frame: Any) -> Any:
    if isinstance(frame, Mapping):
        for key in _KEYFRAME_TIME_KEYS:
            if key in frame:
                return frame[key]
        return None
    if (
        isinstance(frame, Sequence)
        and not isinstance(frame, (str, bytes))
        and len(frame) >= 2
    ):
        return frame[0]
    return None


def _keyframe_value(frame: Any, prop: str) -> tuple[Any, bool]:
    if isinstance(frame, Mapping):
        for key in _KEYFRAME_VALUE_KEYS:
            if key in frame:
                return frame[key], True
        if prop in frame:
            return frame[prop], True
        return None, False
    if (
        isinstance(frame, Sequence)
        and not isinstance(frame, (str, bytes))
        and len(frame) >= 2
    ):
        return frame[1], True
    return None, False


def _normalise_property(prop: str) -> str:
    return prop.strip().lower().replace("-", "_").replace(" ", "_")


def _check_keyframe_easing(
    path: str,
    frame: Any,
    issues: CompositionIssues,
) -> None:
    if not isinstance(frame, Mapping):
        return
    for key in _KEYFRAME_EASING_KEYS:
        if key not in frame:
            continue
        raw = frame[key]
        try:
            Easing(raw)
        except (TypeError, ValueError):
            issues.append(
                CompositionIssue(
                    path=f"{path}.{key}",
                    message=f"unknown keyframe easing {raw!r}",
                    code="keyframe_easing",
                )
            )
        return


def _check_keyframe_value(
    path: str,
    prop: str,
    value: Any,
    issues: CompositionIssues,
) -> None:
    key = _normalise_property(prop)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if key in _KEYFRAME_RANGES:
            issues.append(
                CompositionIssue(
                    path=path,
                    message=f"{prop} keyframe value must be a single number",
                    code="keyframe_range",
                )
            )
            return
        for index, component in enumerate(value):
            if _as_finite(component) is None:
                issues.append(
                    CompositionIssue(
                        path=f"{path}[{index}]",
                        message=f"{prop} keyframe value must be a finite number",
                        code="keyframe_range",
                    )
                )
        return

    number = _as_finite(value)
    if number is None:
        if key in _NUMERIC_KEYFRAME_PROPERTIES:
            issues.append(
                CompositionIssue(
                    path=path,
                    message=f"{prop} keyframe value must be a finite number",
                    code="keyframe_range",
                )
            )
        return

    bounds = _KEYFRAME_RANGES.get(key)
    if bounds is None:
        return
    low, high = bounds
    if low is not None and number < low:
        issues.append(
            CompositionIssue(
                path=path,
                message=(
                    f"{prop} keyframe value {number:g} is below the minimum {low:g}"
                ),
                code="keyframe_range",
            )
        )
    if high is not None and number > high:
        issues.append(
            CompositionIssue(
                path=path,
                message=(
                    f"{prop} keyframe value {number:g} is above the maximum {high:g}"
                ),
                code="keyframe_range",
            )
        )


def _check_keyframe_track(
    *,
    path: str,
    prop: str,
    frames: Any,
    duration: float,
    layer_end: float,
    issues: CompositionIssues,
) -> None:
    if isinstance(frames, Mapping):
        frames = [frames]
    if isinstance(frames, (str, bytes)) or not isinstance(frames, Sequence):
        issues.append(
            CompositionIssue(
                path=path,
                message="keyframes must be a list",
                code="keyframe",
            )
        )
        return

    previous: float | None = None
    for index, frame in enumerate(frames):
        frame_path = f"{path}[{index}]"
        raw_time = _keyframe_time(frame)
        if raw_time is None:
            issues.append(
                CompositionIssue(
                    path=f"{frame_path}.time",
                    message="keyframe needs a time",
                    code="keyframe_time",
                )
            )
        else:
            time = _as_finite(raw_time)
            if time is None:
                issues.append(
                    CompositionIssue(
                        path=f"{frame_path}.time",
                        message="keyframe time must be a finite number",
                        code="keyframe_time",
                    )
                )
            else:
                if time < 0:
                    issues.append(
                        CompositionIssue(
                            path=f"{frame_path}.time",
                            message="keyframe time must not be negative",
                            code="keyframe_time",
                        )
                    )
                window_end = min(layer_end, duration)
                if time > window_end + _EPSILON:
                    if window_end < duration - _EPSILON:
                        message = (
                            f"keyframe time {time:g}s is after the layer window "
                            f"ending at {window_end:g}s"
                        )
                    else:
                        message = (
                            f"keyframe time {time:g}s is after the composition "
                            f"duration of {duration:g}s"
                        )
                    issues.append(
                        CompositionIssue(
                            path=f"{frame_path}.time",
                            message=message,
                            code="keyframe_time",
                        )
                    )
                if previous is not None and time <= previous:
                    issues.append(
                        CompositionIssue(
                            path=f"{frame_path}.time",
                            message=(
                                "keyframes must be sorted by strictly ascending time"
                            ),
                            code="keyframe_order",
                        )
                    )
                previous = time

        value, found = _keyframe_value(frame, prop)
        if not found:
            issues.append(
                CompositionIssue(
                    path=f"{frame_path}.value",
                    message="keyframe needs a value",
                    code="keyframe",
                )
            )
            continue
        _check_keyframe_value(f"{frame_path}.value", prop, value, issues)
        _check_keyframe_easing(frame_path, frame, issues)


def _check_keyframes(composition: Composition, issues: CompositionIssues) -> None:
    for index, layer in enumerate(composition.layers):
        layer_end = layer.end
        if layer_end is None:
            layer_end = composition.duration_seconds
        for path, prop, frames in _keyframe_tracks(layer, index):
            _check_keyframe_track(
                path=path,
                prop=prop,
                frames=frames,
                duration=composition.duration_seconds,
                layer_end=layer_end,
                issues=issues,
            )


def _validate_loaded(
    composition: Composition,
    *,
    assets: AssetLookup | None,
    asset_exists: AssetCheck | None,
) -> CompositionIssues:
    issues = CompositionIssues()
    duration = composition.duration_seconds

    _check_duplicates(
        composition.layers, kind="layer", collection="layers", issues=issues
    )
    _check_duplicates(
        composition.audio_tracks,
        kind="audio track",
        collection="audio_tracks",
        issues=issues,
    )
    _check_duplicates(
        composition.effects, kind="effect", collection="effects", issues=issues
    )

    for index, layer in enumerate(composition.layers):
        _check_window(
            kind="layer",
            collection="layers",
            index=index,
            start=layer.start_seconds,
            end=layer.end,
            duration=duration,
            issues=issues,
        )
    for index, track in enumerate(composition.audio_tracks):
        end = (
            track.start_seconds + track.duration_seconds
            if track.duration_seconds is not None
            else None
        )
        _check_window(
            kind="audio track",
            collection="audio_tracks",
            index=index,
            start=track.start_seconds,
            end=end,
            duration=duration,
            issues=issues,
        )
    for index, effect in enumerate(composition.effects):
        _check_window(
            kind="effect",
            collection="effects",
            index=index,
            start=effect.start_seconds,
            end=effect.end_seconds,
            duration=duration,
            issues=issues,
        )

    checker = _asset_checker(assets, asset_exists)
    if checker is not None:
        _check_assets(composition, checker, issues)

    _check_keyframes(composition, issues)
    return issues


def _load(composition: Composition | Mapping[str, Any]) -> Composition:
    if isinstance(composition, Composition):
        return composition
    return Composition.model_validate(composition)


def validate_composition(
    composition: Composition | Mapping[str, Any],
    *,
    assets: AssetLookup | None = None,
    asset_exists: AssetCheck | None = None,
    warn_safe_area: bool = True,
) -> CompositionIssues:
    """Return every cross-object problem in *composition*.

    ``assets`` may be a collection of known asset ids, a mapping whose keys are
    known ids, a directory to resolve relative paths against, or a callable.
    When neither ``assets`` nor ``asset_exists`` is given, asset existence is
    not checked. A mapping is parsed first; any schema errors are returned as
    issues with their dotted paths instead of raising. Manual text or sticker
    coordinates outside the platform safe area emit a :class:`SafeAreaWarning`
    unless *warn_safe_area* is false.
    """
    try:
        loaded = _load(composition)
    except PydanticValidationError as error:
        return CompositionIssues(_schema_issue(item) for item in error.errors())
    issues = _validate_loaded(loaded, assets=assets, asset_exists=asset_exists)
    if warn_safe_area:
        warn_safe_area_violations(loaded, stacklevel=3)
    return issues


def validate_composition_or_raise(
    composition: Composition | Mapping[str, Any],
    *,
    assets: AssetLookup | None = None,
    asset_exists: AssetCheck | None = None,
    warn_safe_area: bool = True,
) -> Composition:
    """Validate *composition* and return it, raising every issue at once."""
    try:
        loaded = _load(composition)
    except PydanticValidationError as error:
        raise CompositionValidationError(
            [_schema_issue(item) for item in error.errors()]
        ) from error
    issues = _validate_loaded(loaded, assets=assets, asset_exists=asset_exists)
    if warn_safe_area:
        warn_safe_area_violations(loaded, stacklevel=3)
    issues.raise_for_errors()
    return loaded
