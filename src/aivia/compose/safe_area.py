"""Resolve platform safe areas and check manual layer geometry against them.

A composition may carry an explicit safe area or name a platform whose data
file supplies one. Text and sticker layers are laid out inside that area, while
manual coordinates are left untouched and reported as advisory warnings when
they cross a margin. Rendering stays deterministic: this module only does
arithmetic on the composition and the packaged platform data.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

from aivia.core.platforms import Platform, SafeArea, get_platform

from .schema import Composition, Layer, LayerType

__all__ = [
    "SafeAreaViolation",
    "SafeAreaWarning",
    "check_safe_area",
    "effective_safe_area",
    "resolve_safe_area",
    "safe_area_bounds",
    "safe_area_box",
    "safe_area_for",
    "safe_area_insets",
    "safe_area_pixels",
    "safe_area_violations",
    "safe_area_warnings",
    "warn_safe_area_violations",
]


@dataclass(frozen=True, slots=True)
class SafeAreaViolation:
    """One manual layer rectangle that crosses a safe area margin."""

    path: str
    message: str
    code: str = "safe_area"
    layer_id: str = ""

    def __str__(self) -> str:
        return f"{self.path}: {self.message}" if self.path else self.message

    def to_dict(self) -> dict[str, str]:
        """Return the violation as a JSON-compatible mapping."""
        return {
            "path": self.path,
            "message": self.message,
            "code": self.code,
            "layer_id": self.layer_id,
        }


class SafeAreaWarning(UserWarning):
    """Advisory warning for manual coordinates outside a platform safe area."""


def effective_safe_area(
    composition: Composition,
    *,
    platform: Platform | str | None = None,
) -> SafeArea:
    """Return the safe area that applies to *composition*.

    An explicit ``composition.safe_area`` wins. Otherwise the platform named by
    the composition, or by *platform* when given, supplies the margins. With no
    platform the margins are zero.
    """
    if composition.safe_area is not None:
        return composition.safe_area
    target = composition.platform if platform is None else platform
    if isinstance(target, Platform):
        return target.safe_area
    name = "" if target is None else str(target).strip()
    if not name:
        return SafeArea()
    return get_platform(name).safe_area


resolve_safe_area = effective_safe_area
safe_area_for = effective_safe_area


def _finite_margins(area: SafeArea) -> None:
    if not all(math.isfinite(value) for value in area.margins):
        raise ValueError("safe area margins must be finite")


def safe_area_insets(
    composition: Composition,
    *,
    platform: Platform | str | None = None,
) -> tuple[int, int, int, int]:
    """Return the safe area margins as ``(top, right, bottom, left)`` pixels."""
    area = effective_safe_area(composition, platform=platform)
    _finite_margins(area)
    return area.to_pixels(composition.width, composition.height)


safe_area_pixels = safe_area_insets


def safe_area_box(
    composition: Composition,
    *,
    platform: Platform | str | None = None,
) -> tuple[int, int, int, int]:
    """Return the safe area as ``(left, top, width, height)`` pixels."""
    top, right, bottom, left = safe_area_insets(composition, platform=platform)
    width = max(0, composition.width - left - right)
    height = max(0, composition.height - top - bottom)
    return left, top, width, height


safe_area_bounds = safe_area_box


def _value(layer: Layer, *names: str) -> Any:
    extras = layer.model_extra or {}
    for name in names:
        if name in extras:
            return extras[name]
        value = getattr(layer, name, None)
        if value is not None:
            return value
    return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _pair(value: Any) -> tuple[float, float] | None:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        first = _number(value[0])
        second = _number(value[1])
        if first is not None and second is not None:
            return first, second
        return None
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace("x", ",").split(",")]
        if len(parts) == 2:
            try:
                return float(parts[0]), float(parts[1])
            except ValueError:
                return None
    return None


def _text_box(
    layer: Layer,
    composition: Composition,
    area: SafeArea,
) -> tuple[int, int, int, int] | None:
    position = _pair(_value(layer, "position", "pos", "offset"))
    size = _pair(_value(layer, "size", "dimensions"))
    x = _number(_value(layer, "x", "left"))
    y = _number(_value(layer, "y", "top"))
    width = _number(_value(layer, "width", "w"))
    height = _number(_value(layer, "height", "h"))

    if x is None and position is not None:
        x = position[0]
    if y is None and position is not None:
        y = position[1]
    if width is None and size is not None:
        width = size[0]
    if height is None and size is not None:
        height = size[1]

    if x is None and y is None and width is None and height is None:
        return None

    top, right, bottom, left = area.to_pixels(composition.width, composition.height)
    if x is None:
        x = float(left)
    if y is None:
        y = float(top)
    if width is None:
        width = float(max(0, composition.width - left - right))
    if height is None:
        height = float(max(0, composition.height - top - bottom))
    return round(x), round(y), round(width), round(height)


def _overlay_box(layer: Layer) -> tuple[int, int, int, int] | None:
    position = _pair(_value(layer, "position", "pos", "offset"))
    x = _number(_value(layer, "x", "left"))
    y = _number(_value(layer, "y", "top"))
    if x is None and position is not None:
        x = position[0]
    if y is None and position is not None:
        y = position[1]
    if x is None and y is None:
        return None
    width = _number(_value(layer, "width", "w")) or 0.0
    height = _number(_value(layer, "height", "h")) or 0.0
    return round(x or 0.0), round(y or 0.0), round(width), round(height)


def safe_area_violations(
    composition: Composition,
    *,
    platform: Platform | str | None = None,
) -> list[SafeAreaViolation]:
    """Return every manual text or sticker rectangle outside the safe area."""
    area = effective_safe_area(composition, platform=platform)
    if area == SafeArea():
        return []
    top, right, bottom, left = area.to_pixels(composition.width, composition.height)
    safe_right = composition.width - right
    safe_bottom = composition.height - bottom
    violations: list[SafeAreaViolation] = []
    for index, layer in enumerate(composition.layers):
        if not layer.visible:
            continue
        if layer.type is LayerType.TEXT:
            if getattr(layer, "smart_placement", False):
                continue
            box = _text_box(layer, composition, area)
        elif layer.type in (LayerType.STICKER, LayerType.LOGO):
            box = _overlay_box(layer)
        else:
            continue
        if box is None:
            continue
        x, y, width, height = box
        if x < left or y < top or x + width > safe_right or y + height > safe_bottom:
            identifier = layer.id or f"#{index}"
            violations.append(
                SafeAreaViolation(
                    path=f"layers[{index}]",
                    message=(
                        f"{layer.type.value} layer {identifier!r} at "
                        f"({x}, {y}) size {width}x{height} is outside the "
                        f"safe area ({left}, {top}, {safe_right}, {safe_bottom})"
                    ),
                    layer_id=layer.id,
                )
            )
    return violations


check_safe_area = safe_area_violations
safe_area_warnings = safe_area_violations


def warn_safe_area_violations(
    composition: Composition,
    *,
    platform: Platform | str | None = None,
    stacklevel: int = 2,
) -> list[SafeAreaViolation]:
    """Emit :class:`SafeAreaWarning` for each manual layer outside the area."""
    violations = safe_area_violations(composition, platform=platform)
    for violation in violations:
        warnings.warn(str(violation), SafeAreaWarning, stacklevel=stacklevel)
    return violations
