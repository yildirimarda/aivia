"""Smart text placement over low detail regions.

A text block reads best over a calm part of an image. This module scores
candidate boxes by the amount of local detail under them and optionally keeps
them away from face rectangles supplied by a vision check. The scoring is pure
Pillow work: it reads pixels only and never calls a model or the network.

The renderer uses :func:`find_text_placement` when a text layer asks for smart
placement. A composition builder may run the optional :func:`faces_with_vision`
helper first and store the returned rectangles on the text layer, so a render
still has no AI dependency.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, cast

from PIL import Image, ImageChops, ImageFilter

from .errors import RenderError
from .schema import FaceBox, FitMode

__all__ = [
    "DEFAULT_ANALYSIS_SIZE",
    "FaceRegion",
    "LayerPlacement",
    "LowDetailRegion",
    "Placement",
    "PlacementCandidate",
    "PlacementRegion",
    "TextPlacementRegion",
    "analyze_faces",
    "avoid_faces",
    "check_faces",
    "choose_text_placement",
    "detect_faces_with_vision",
    "detect_low_detail_regions",
    "face_regions",
    "face_regions_from_analysis",
    "faces_from_analysis",
    "faces_with_vision",
    "find_low_detail_regions",
    "find_placement",
    "find_text_placement",
    "low_detail_regions",
    "place_text",
    "rank_text_placements",
    "scale_face_rectangles",
    "scale_faces_for_fit",
    "scale_faces_to_canvas",
    "smart_placement",
    "vision_faces",
]

DEFAULT_ANALYSIS_SIZE = 256
"""The longest edge used while scoring candidates, to keep the scan cheap."""

_MIN_RESULTS = 0
_FIRST_RESULT = 1

type ImageInput = Image.Image | str | Path | bytes
"""A decoded image, a local path or encoded image bytes."""

type FaceInput = Any
"""A face rectangle as a mapping, a sequence or an object with x/y/w/h."""


@dataclass(frozen=True, slots=True)
class LayerPlacement:
    """The destination rectangle of one layer on the canvas, in pixels."""

    x: int
    y: int
    width: int
    height: int

    @property
    def center(self) -> tuple[float, float]:
        """The centre of the rectangle."""
        return self.x + self.width / 2.0, self.y + self.height / 2.0

    @property
    def box(self) -> tuple[int, int, int, int]:
        """The rectangle as ``(x, y, width, height)``."""
        return self.x, self.y, self.width, self.height

    @property
    def right(self) -> int:
        """The first column after the rectangle."""
        return self.x + self.width

    @property
    def bottom(self) -> int:
        """The first row after the rectangle."""
        return self.y + self.height


@dataclass(frozen=True, slots=True)
class PlacementRegion:
    """A scored candidate box for smart text placement."""

    x: int
    y: int
    width: int
    height: int
    detail: float = 0.0
    face_overlap: float = 0.0
    score: float = 0.0

    @property
    def placement(self) -> LayerPlacement:
        """The rectangle as a :class:`LayerPlacement`."""
        return LayerPlacement(self.x, self.y, self.width, self.height)

    @property
    def box(self) -> tuple[int, int, int, int]:
        """The rectangle as ``(x, y, width, height)``."""
        return self.x, self.y, self.width, self.height

    @property
    def center(self) -> tuple[float, float]:
        """The centre of the rectangle."""
        return self.x + self.width / 2.0, self.y + self.height / 2.0


Placement = PlacementRegion
PlacementCandidate = PlacementRegion
LowDetailRegion = PlacementRegion
TextPlacementRegion = PlacementRegion
FaceRegion = FaceBox


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RenderError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise RenderError(f"{name} must be finite")
    return number


def _image(source: ImageInput) -> Image.Image:
    if isinstance(source, Image.Image):
        return source.convert("RGBA")
    try:
        if isinstance(source, bytes):
            with Image.open(BytesIO(source)) as opened:
                return opened.convert("RGBA")
        with Image.open(Path(source)) as opened:
            return opened.convert("RGBA")
    except (OSError, ValueError) as exc:
        raise RenderError("cannot open image for placement") from exc


def _box_size(value: Any) -> tuple[int, int]:
    if isinstance(value, bool):
        raise RenderError("text box size must be a pair of positive integers")
    width: Any
    height: Any
    if isinstance(value, int):
        width = height = value
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        width, height = value
    elif isinstance(value, str) and "x" in value.lower():
        left, right = value.lower().split("x", 1)
        width, height = left.strip(), right.strip()
    else:
        raise RenderError("text box size must be a pair of positive integers")
    if isinstance(width, bool) or isinstance(height, bool):
        raise RenderError("text box size must be a pair of positive integers")
    try:
        box_width = int(width)
        box_height = int(height)
    except (TypeError, ValueError) as exc:
        raise RenderError("text box size must be a pair of positive integers") from exc
    if box_width < 1 or box_height < 1:
        raise RenderError("text box size must be a pair of positive integers")
    return box_width, box_height


def _margin(value: Any) -> tuple[float, float]:
    if value is None:
        return 0.0, 0.0
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise RenderError("margin must be a number or a pair of numbers")
        horizontal = _finite(value[0], "margin")
        vertical = _finite(value[1], "margin")
    else:
        horizontal = vertical = _finite(value, "margin")
    if horizontal < 0.0 or vertical < 0.0:
        raise RenderError("margin must not be negative")
    return horizontal, vertical


def _face_rect(value: FaceInput) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        x = value.get("x", value.get("left", 0.0))
        y = value.get("y", value.get("top", 0.0))
        width = value.get("width", value.get("w", 0.0))
        height = value.get("height", value.get("h", 0.0))
    elif isinstance(value, (tuple, list)) and len(value) == 4:
        x, y, width, height = value
    else:
        x = getattr(value, "x", getattr(value, "left", 0.0))
        y = getattr(value, "y", getattr(value, "top", 0.0))
        width = getattr(value, "width", getattr(value, "w", 0.0))
        height = getattr(value, "height", getattr(value, "h", 0.0))
    try:
        rectangle = (
            _finite(x, "face x"),
            _finite(y, "face y"),
            _finite(width, "face width"),
            _finite(height, "face height"),
        )
    except RenderError:
        return None
    if rectangle[2] <= 0.0 or rectangle[3] <= 0.0:
        return None
    return rectangle


def _face_confidence(value: Any) -> float | None:
    if isinstance(value, Mapping):
        raw = value.get("confidence")
    else:
        raw = getattr(value, "confidence", None)
    if raw is None:
        return None
    try:
        return _finite(raw, "face confidence")
    except RenderError:
        return None


def _face_records(
    faces: Any,
) -> list[tuple[tuple[float, float, float, float], float | None]]:
    if faces is None:
        return []
    if not isinstance(faces, (Mapping, FaceBox, tuple, list)) and (
        hasattr(faces, "faces") or hasattr(faces, "face_regions")
    ):
        faces = faces_from_analysis(faces)
    if isinstance(faces, Mapping) and not any(
        key in faces for key in ("x", "left", "width", "w")
    ):
        faces = faces.get("faces", faces.get("face_regions", []))
    if isinstance(faces, (FaceBox, Mapping, tuple, list)) and not isinstance(
        faces, (list, tuple)
    ):
        faces = [faces]
    if (
        isinstance(faces, tuple)
        and len(faces) == 4
        and all(isinstance(item, (int, float)) for item in faces)
    ):
        faces = [faces]
    if not isinstance(faces, Sequence):
        faces = [faces]
    records = []
    for face in faces:
        rectangle = _face_rect(face)
        if rectangle is not None:
            records.append((rectangle, _face_confidence(face)))
    return records


def _face_list(faces: Any) -> list[tuple[float, float, float, float]]:
    return [rectangle for rectangle, _ in _face_records(faces)]


def faces_from_analysis(analysis: Any) -> list[Any]:
    """Return the face rectangles carried by an analysis result.

    Accepts an :class:`~aivia.media.image_analysis.ImageAnalysis`, an asset
    analysis, a mapping with ``faces``/``face_regions`` or a plain sequence.
    """
    if analysis is None:
        return []
    if isinstance(analysis, Mapping):
        value = analysis.get("faces", analysis.get("face_regions", analysis))
    elif hasattr(analysis, "faces"):
        value = analysis.faces
    elif hasattr(analysis, "face_regions"):
        value = analysis.face_regions
    else:
        value = analysis
    if isinstance(value, (FaceBox, Mapping)) or hasattr(value, "box"):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return [value]


face_regions_from_analysis = faces_from_analysis
face_regions = faces_from_analysis


def _scaled(
    rectangle: tuple[float, float, float, float],
    scale_x: float,
    scale_y: float,
) -> tuple[float, float, float, float]:
    return (
        rectangle[0] * scale_x,
        rectangle[1] * scale_y,
        rectangle[2] * scale_x,
        rectangle[3] * scale_y,
    )


def _overlap(
    box: tuple[float, float, float, float],
    face: tuple[float, float, float, float],
) -> float:
    left = max(box[0], face[0])
    top = max(box[1], face[1])
    right = min(box[0] + box[2], face[0] + face[2])
    bottom = min(box[1] + box[3], face[1] + face[3])
    if right <= left or bottom <= top:
        return 0.0
    return (right - left) * (bottom - top)


def _size_pair(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise RenderError(f"{name} must be a pair of positive numbers")
    width = _finite(value[0], name)
    height = _finite(value[1], name)
    if width <= 0.0 or height <= 0.0:
        raise RenderError(f"{name} must be a pair of positive numbers")
    return width, height


def _target_box(
    target_size: Any,
    placement: Any,
) -> tuple[float, float, float, float]:
    if placement is not None:
        if isinstance(placement, LayerPlacement):
            return (
                float(placement.x),
                float(placement.y),
                float(placement.width),
                float(placement.height),
            )
        if isinstance(placement, (tuple, list)) and len(placement) == 4:
            x = _finite(placement[0], "placement x")
            y = _finite(placement[1], "placement y")
            width = _finite(placement[2], "placement width")
            height = _finite(placement[3], "placement height")
            if width <= 0.0 or height <= 0.0:
                raise RenderError("placement must be a rectangle with positive size")
            return x, y, width, height
        raise RenderError("placement must be a rectangle or LayerPlacement")
    width, height = _size_pair(target_size, "target_size")
    return 0.0, 0.0, width, height


def _fit_transform(
    source_size: tuple[float, float],
    target_box: tuple[float, float, float, float],
    fit: FitMode,
) -> tuple[float, float, float, float]:
    source_width, source_height = source_size
    x, y, width, height = target_box
    if fit is FitMode.STRETCH:
        return width / source_width, height / source_height, x, y
    if fit is FitMode.COVER:
        scale = max(width / source_width, height / source_height)
        return (
            scale,
            scale,
            x + (width - source_width * scale) / 2.0,
            y + (height - source_height * scale) / 2.0,
        )
    if fit is FitMode.CONTAIN:
        fitted_width: float
        fitted_height: float
        if source_width / source_height > width / height:
            fitted_width = width
            fitted_height = round(source_height / source_width * width)
        else:
            fitted_height = height
            fitted_width = round(source_width / source_height * height)
        return (
            fitted_width / source_width,
            fitted_height / source_height,
            x + (width - fitted_width) / 2.0,
            y + (height - fitted_height) / 2.0,
        )
    return (
        1.0,
        1.0,
        x + (width - source_width) / 2.0,
        y + (height - source_height) / 2.0,
    )


def scale_faces_to_canvas(
    faces: Any,
    source_size: tuple[int, int],
    target_size: tuple[int, int] | None = None,
    fit: FitMode | str = FitMode.COVER,
    *,
    canvas_size: tuple[int, int] | None = None,
    placement: LayerPlacement | tuple[int, int, int, int] | None = None,
    offset: tuple[float, float] = (0.0, 0.0),
) -> list[FaceBox]:
    """Map face rectangles from source image pixels to canvas pixels.

    *source_size* is the size of the image the vision check analysed.
    *target_size* (or *canvas_size*) is the canvas, while *placement* names the
    rectangle the background image occupies. *fit* mirrors the renderer's fit
    modes so a cover crop or a contain letterbox lands in the right place.
    """
    records = _face_records(faces)
    if not records:
        return []
    source = _size_pair(source_size, "source_size")
    if target_size is None:
        target_size = canvas_size
    if target_size is None and placement is None:
        raise RenderError("target_size, canvas_size or placement is required")
    box = _target_box(target_size, placement)
    try:
        mode = FitMode(fit)
    except (TypeError, ValueError) as exc:
        raise RenderError(f"unsupported fit mode {fit!r}") from exc
    scale_x, scale_y, dx, dy = _fit_transform(source, box, mode)
    dx += _finite(offset[0], "offset")
    dy += _finite(offset[1], "offset")
    box_left, box_top, box_width, box_height = box
    box_right = box_left + box_width
    box_bottom = box_top + box_height
    scaled: list[FaceBox] = []
    for rectangle, confidence in records:
        left = rectangle[0] * scale_x + dx
        top = rectangle[1] * scale_y + dy
        right = left + rectangle[2] * scale_x
        bottom = top + rectangle[3] * scale_y
        left = max(left, box_left)
        top = max(top, box_top)
        right = min(right, box_right)
        bottom = min(bottom, box_bottom)
        if right <= left or bottom <= top:
            continue
        scaled.append(
            FaceBox(
                x=left,
                y=top,
                width=right - left,
                height=bottom - top,
                confidence=confidence,
            )
        )
    return scaled


scale_faces_for_fit = scale_faces_to_canvas
scale_face_rectangles = scale_faces_to_canvas


def _integral(image: Image.Image) -> list[list[float]]:
    width, height = image.size
    pixels = list(image.get_flattened_data())
    rows: list[list[float]] = [[0.0] * (width + 1) for _ in range(height + 1)]
    for y in range(height):
        running = 0.0
        previous = rows[y]
        current = rows[y + 1]
        offset = y * width
        for x in range(width):
            running += float(cast(int, pixels[offset + x]))
            current[x + 1] = previous[x + 1] + running
    return rows


def _region_sum(
    integral: list[list[float]],
    x: int,
    y: int,
    width: int,
    height: int,
) -> float:
    return (
        integral[y + height][x + width]
        - integral[y][x + width]
        - integral[y + height][x]
        + integral[y][x]
    )


def _positions(start: int, stop: int, step: int) -> list[int]:
    if stop < start:
        return []
    values = list(range(start, stop + 1, step))
    if not values or values[-1] != stop:
        values.append(stop)
    return values


def find_low_detail_regions(
    image: ImageInput,
    box_size: tuple[int, int] | int,
    *,
    faces: Sequence[FaceInput] | None = None,
    avoid_faces: bool = True,
    face_padding: float = 0.0,
    margin: float | tuple[float, float] = 0.0,
    step: int | None = None,
    limit: int | None = 10,
    max_analysis_size: int = DEFAULT_ANALYSIS_SIZE,
) -> list[PlacementRegion]:
    """Return candidate text boxes ordered from calmest to busiest.

    Detail is the mean high-frequency energy under a box, normalised to
    ``0..1``. With *avoid_faces* a box that touches a face rectangle is ranked
    below every box that does not. Ties prefer the canvas centre, then the top
    left corner, so the result is deterministic.
    """
    source = _image(image)
    box_width, box_height = _box_size(box_size)
    horizontal_margin, vertical_margin = _margin(margin)
    if step is not None and (
        isinstance(step, bool) or not isinstance(step, int) or step < 1
    ):
        raise RenderError("step must be a positive integer")
    if limit is not None and limit < _MIN_RESULTS:
        raise RenderError("limit must not be negative")
    if max_analysis_size < 1:
        raise RenderError("max_analysis_size must be positive")
    padding = _finite(face_padding, "face_padding")
    if padding < 0.0:
        raise RenderError("face_padding must not be negative")

    width, height = source.size
    scale = min(1.0, max_analysis_size / max(width, height))
    analysis_width = max(1, round(width * scale))
    analysis_height = max(1, round(height * scale))
    gray = source.convert("L")
    if (analysis_width, analysis_height) != (width, height):
        gray = gray.resize(
            (analysis_width, analysis_height),
            Image.Resampling.BILINEAR,
        )
    blurred = gray.filter(ImageFilter.GaussianBlur(2))
    detail = ImageChops.difference(gray, blurred)
    integral = _integral(detail)

    scale_x = analysis_width / width
    scale_y = analysis_height / height
    target_width = min(analysis_width, max(1, round(box_width * scale_x)))
    target_height = min(analysis_height, max(1, round(box_height * scale_y)))

    margin_x = round(horizontal_margin * scale_x)
    margin_y = round(vertical_margin * scale_y)
    max_x = analysis_width - target_width
    max_y = analysis_height - target_height
    if margin_x * 2 > max_x:
        margin_x = 0
    if margin_y * 2 > max_y:
        margin_y = 0
    start_x = min(max(0, margin_x), max_x)
    start_y = min(max(0, margin_y), max_y)
    stop_x = max(start_x, max_x - margin_x)
    stop_y = max(start_y, max_y - margin_y)

    if step is None:
        step = max(1, min(target_width, target_height) // 8)
    step = max(1, min(step, max(target_width, target_height)))

    face_rectangles = _face_list(faces) if avoid_faces else []
    scaled_faces = [
        (
            (face[0] - padding) * scale_x,
            (face[1] - padding) * scale_y,
            (face[2] + padding * 2) * scale_x,
            (face[3] + padding * 2) * scale_y,
        )
        for face in face_rectangles
    ]
    area = float(target_width * target_height)
    canvas_center_x = analysis_width / 2.0
    canvas_center_y = analysis_height / 2.0

    candidates: list[tuple[tuple[float, ...], PlacementRegion]] = []
    for y in _positions(start_y, stop_y, step):
        for x in _positions(start_x, stop_x, step):
            total = _region_sum(integral, x, y, target_width, target_height)
            mean = total / area / 255.0
            overlap = 0.0
            if scaled_faces:
                box = (float(x), float(y), float(target_width), float(target_height))
                covered = sum(_overlap(box, face) for face in scaled_faces)
                overlap = min(1.0, covered / area)
            score = mean + overlap
            center_x = x + target_width / 2.0
            center_y = y + target_height / 2.0
            distance = abs(center_x - canvas_center_x) + abs(center_y - canvas_center_y)
            key = (
                1.0 if overlap > 0.0 else 0.0,
                round(overlap, 9),
                round(mean, 9),
                round(distance, 6),
                y,
                x,
            )
            full_width = min(width, box_width)
            full_height = min(height, box_height)
            full_x = min(width - full_width, max(0, round(x / scale_x)))
            full_y = min(height - full_height, max(0, round(y / scale_y)))
            candidates.append(
                (
                    key,
                    PlacementRegion(
                        x=full_x,
                        y=full_y,
                        width=full_width,
                        height=full_height,
                        detail=mean,
                        face_overlap=overlap,
                        score=score,
                    ),
                )
            )

    candidates.sort(key=lambda item: item[0])
    regions = [region for _, region in candidates]
    if limit is not None:
        regions = regions[:limit]
    return regions


detect_low_detail_regions = find_low_detail_regions
low_detail_regions = find_low_detail_regions
rank_text_placements = find_low_detail_regions


def find_text_placement(
    image: ImageInput,
    box_size: tuple[int, int] | int,
    *,
    faces: Sequence[FaceInput] | None = None,
    avoid_faces: bool = True,
    face_padding: float = 0.0,
    margin: float | tuple[float, float] = 0.0,
    step: int | None = None,
    max_analysis_size: int = DEFAULT_ANALYSIS_SIZE,
) -> LayerPlacement:
    """Return the calmest text box on *image*, optionally away from faces."""
    regions = find_low_detail_regions(
        image,
        box_size,
        faces=faces,
        avoid_faces=avoid_faces,
        face_padding=face_padding,
        margin=margin,
        step=step,
        limit=_FIRST_RESULT,
        max_analysis_size=max_analysis_size,
    )
    if not regions:
        raise RenderError("the image is too small for the text box")
    return regions[0].placement


find_placement = find_text_placement
choose_text_placement = find_text_placement
smart_placement = find_text_placement
place_text = find_text_placement
avoid_faces = find_text_placement


async def faces_with_vision(
    source: Any,
    *,
    analyzer: Any | None = None,
    helper: Any | None = None,
    provider: Any | None = None,
    **kwargs: Any,
) -> list[Any]:
    """Run the optional vision face check and return face rectangles.

    The media analysis module is imported lazily so importing composition
    stays free of LLM and network dependencies. Rendering never calls this
    function; it consumes rectangles stored on a text layer.
    """
    from aivia.media.image_analysis import detect_faces

    return await detect_faces(
        source,
        analyzer=analyzer,
        helper=helper,
        provider=provider,
        **kwargs,
    )


detect_faces_with_vision = faces_with_vision
vision_faces = faces_with_vision
check_faces = faces_with_vision
analyze_faces = faces_with_vision
