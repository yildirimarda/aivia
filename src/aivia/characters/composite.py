"""Place pose cutouts into compositions.

A cutout becomes an overlay (sticker) layer pinned to a canvas anchor, scaled,
optionally mirrored, optionally given a drop shadow and given a subtle idle
animation. The renderer keeps the layer inside the composition's safe area
based on the anchor and margin, so no explicit coordinates are needed.

Everything here is plain data and deterministic arithmetic: no AI call and no
network request happens here.
"""

from __future__ import annotations

from aivia.compose.keyframes import Easing
from aivia.compose.motion import Motion, MotionType
from aivia.compose.schema import Anchor, FitMode, Flip, LayerType, StickerLayer

__all__ = ["Flip", "PlaceCutoutError", "idle_motion", "place_cutout"]


class PlaceCutoutError(ValueError):
    """A cutout could not be placed into a composition."""


def idle_motion(
    duration_seconds: float,
    *,
    drift: tuple[float, float] = (0.0, 4.0),
    easing: Easing | str = Easing.LINEAR,
) -> Motion:
    """Return a gentle idle animation that floats a cutout across its window.

    The cutout drifts from its anchor position by *drift* over the duration and
    holds there, which reads as a subtle idle float when the layer is shown for
    the whole window.
    """
    return Motion(
        type=MotionType.PARALLAX,
        depth=1.0,
        drift=drift,
        start_pan=(0.0, 0.0),
        easing=Easing(easing),
    )


def place_cutout(
    *,
    source: str,
    duration_seconds: float,
    start_seconds: float = 0.0,
    z_index: int = 0,
    opacity: float = 1.0,
    anchor: Anchor = Anchor.CENTER,
    scale: float = 1.0,
    flip: Flip = Flip.NONE,
    shadow: bool = False,
    shadow_color: str = "#000000",
    shadow_offset: tuple[float, float] = (4.0, 4.0),
    shadow_blur: float = 0.0,
    shadow_opacity: float = 0.5,
    margin: float | tuple[float, float] = 0.0,
    fit: FitMode = FitMode.CONTAIN,
    idle: bool = True,
    idle_drift: tuple[float, float] = (0.0, 4.0),
) -> StickerLayer:
    """Build a :class:`~aivia.compose.schema.StickerLayer` for one cutout.

    The layer is pinned to *anchor*, scaled by *scale*, optionally mirrored by
    *flip*, optionally given a drop shadow and animated with a subtle idle
    float. The renderer keeps it inside the composition's safe area based on
    the anchor and margin, so no explicit coordinates are needed.
    """
    if scale <= 0.0:
        raise PlaceCutoutError("scale must be positive")
    motion = idle_motion(duration_seconds, drift=idle_drift) if idle else None
    return StickerLayer(
        type=LayerType.STICKER,
        source=source,
        start_seconds=start_seconds,
        end_seconds=start_seconds + duration_seconds,
        z_index=z_index,
        opacity=opacity,
        anchor=anchor,
        scale=scale,
        flip=flip,
        shadow=shadow,
        shadow_color=shadow_color,
        shadow_offset=shadow_offset,
        shadow_blur=shadow_blur,
        shadow_opacity=shadow_opacity,
        margin=margin,
        fit=fit,
        motion=motion,
    )
