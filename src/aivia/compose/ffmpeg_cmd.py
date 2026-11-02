"""Pure FFmpeg command construction for a composition.

The builder never runs ffmpeg, touches the network or reads an asset. It turns
a validated composition into two inspectable products: a ``filter_complex``
string and an argv list. That separation keeps command construction testable
without an external binary while leaving execution to the caller.

Visual layers are composited over a generated colour background. Image, video,
sticker and logo layers become inputs; their geometry, fit, opacity and time
window become filter nodes. Text layers become transparent inputs with a
``drawtext`` filter so they animate with the same keyframes. Layer entrance
transitions fade or slide a layer in, and extend the outgoing layer through the
transition window. Audio mixing is deliberately left to a later composition
step.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audio_subs import (
    audio_mix_settings,
    build_audio_filtergraph,
    subtitle_filter,
)
from .errors import RenderError
from .fonts import font_manager
from .keyframes import KeyframeTrack
from .motion import (
    MotionTracks,
    pair_expression,
    track_expression,
    tracks_for_layer,
)
from .output_profiles import (
    OutputProfile,
    apply_output_profile,
    resolve_output_profile,
)
from .safe_area import effective_safe_area
from .schema import (
    Anchor,
    BlendMode,
    Composition,
    FilterType,
    FitMode,
    ImageFilter,
    ImageMask,
    Layer,
    LayerType,
    MaskMode,
    TextStyle,
    TextWeight,
)
from .transitions import (
    layer_transition_for,
    outgoing_transition_end,
)

__all__ = [
    "DEFAULT_AUDIO_CODEC",
    "DEFAULT_CRF",
    "DEFAULT_PIXEL_FORMAT",
    "DEFAULT_PRESET",
    "DEFAULT_VIDEO_CODEC",
    "FFmpegCommand",
    "FFmpegCommandError",
    "FFmpegInput",
    "build_ffmpeg_args",
    "build_ffmpeg_command",
    "build_ffmpeg_filtergraph",
    "build_filtergraph",
    "ffmpeg_args",
    "ffmpeg_command",
    "ffmpeg_filtergraph",
]

DEFAULT_VIDEO_CODEC = "libx264"
"""The default video encoder for a deterministic local render."""

DEFAULT_AUDIO_CODEC = "aac"
"""The default audio encoder once audio mixing is attached."""

DEFAULT_PRESET = "medium"
"""The default encoder speed and compression trade-off."""

DEFAULT_CRF = 18
"""The default constant quality value for the video encoder."""

DEFAULT_PIXEL_FORMAT = "yuv420p"
"""The default pixel format for broad player compatibility."""

_IMAGE_TYPES = frozenset(
    {
        LayerType.IMAGE,
        LayerType.STICKER,
        LayerType.LOGO,
    }
)
_OVERLAY_TYPES = frozenset({LayerType.STICKER, LayerType.LOGO})
_UNSUPPORTED_TYPES = frozenset({LayerType.SHAPE})


class FFmpegCommandError(RenderError):
    """An ffmpeg command could not be built from the composition."""

    code = "ffmpeg_command"


@dataclass(frozen=True, slots=True)
class FFmpegInput:
    """One media input plus the options that precede ``-i``."""

    source: str
    options: tuple[str, ...] = ()
    layer_id: str = ""

    def argv(self) -> list[str]:
        """Return this input as argv fragments."""
        return [*self.options, "-i", self.source]


class FFmpegCommand(list[str]):
    """An ffmpeg argument list that also exposes its graph and inputs.

    Subclassing ``list`` keeps the value directly usable as argv while giving
    callers named access to the filtergraph and the ordered input list.
    """

    def __init__(
        self,
        args: Sequence[str],
        *,
        filtergraph: str,
        output: str,
        inputs: Sequence[FFmpegInput] = (),
    ) -> None:
        super().__init__(args)
        self.filtergraph = filtergraph
        self.output = output
        self.inputs = tuple(inputs)

    @property
    def argv(self) -> list[str]:
        """Return a plain copy of the argument list."""
        return list(self)

    @property
    def args(self) -> list[str]:
        """Alias for :attr:`argv`."""
        return list(self)

    @property
    def executable(self) -> str:
        """The executable in the first argument."""
        return self[0]

    @property
    def filter_complex(self) -> str:
        """Alias for :attr:`filtergraph`."""
        return self.filtergraph

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable description of this command."""
        return {
            "executable": self.executable,
            "argv": list(self),
            "filtergraph": self.filtergraph,
            "output": self.output,
            "inputs": [
                {"source": item.source, "options": list(item.options)}
                for item in self.inputs
            ],
        }


def _composition(value: Composition | Mapping[str, Any]) -> Composition:
    if isinstance(value, Composition):
        return value
    return Composition.model_validate(value)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FFmpegCommandError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise FFmpegCommandError(f"{name} must be finite")
    return number


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise FFmpegCommandError(f"{name} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise FFmpegCommandError(f"{name} must be an integer")


def _pair(value: Any, name: str) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        parts = [part for part in value.replace("x", ",").split(",") if part.strip()]
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        parts = list(value)
    else:
        raise FFmpegCommandError(f"{name} must be a pair of integers")
    if len(parts) != 2:
        raise FFmpegCommandError(f"{name} must be a pair of integers")
    return _integer(parts[0], name), _integer(parts[1], name)


def _layer_value(layer: Layer, *names: str) -> Any:
    extras = layer.model_extra or {}
    for name in names:
        if name in extras:
            return extras[name]
        value = getattr(layer, name, None)
        if value is not None:
            return value
    return None


def _seconds(value: float) -> str:
    return f"{value:.6f}"


def _output_size(
    width: Any,
    height: Any,
) -> tuple[int, int] | None:
    if (width is None) != (height is None):
        raise FFmpegCommandError(
            "output_width and output_height must be provided together"
        )
    if width is None:
        return None
    resolved_width = _integer(width, "output_width")
    resolved_height = _integer(height, "output_height")
    if resolved_width <= 0 or resolved_height <= 0:
        raise FFmpegCommandError("output dimensions must be positive")
    return resolved_width, resolved_height


def _ffmpeg_color(value: str) -> str:
    text = value.strip()
    if not text:
        raise FFmpegCommandError("canvas background must not be empty")
    if not text.startswith("#"):
        return text
    digits = text[1:]
    if len(digits) in (3, 4):
        digits = "".join(character * 2 for character in digits)
    if len(digits) == 6:
        return f"0x{digits}"
    if len(digits) == 8:
        rgb, alpha = digits[:6], digits[6:]
        return f"0x{rgb}@{int(alpha, 16) / 255.0:.6f}"
    raise FFmpegCommandError(f"unsupported canvas background {value!r}")


def _layer_end(layer: Layer, composition: Composition) -> float:
    end = layer.end
    if end is None:
        end = composition.duration_seconds
    extended = outgoing_transition_end(layer, composition)
    if extended is not None:
        end = max(end, extended)
    return min(max(end, layer.start_seconds), composition.duration_seconds)


def _active_layers(composition: Composition) -> list[Layer]:
    layers: list[tuple[int, int, Layer]] = []
    for index, layer in enumerate(composition.layers):
        if not layer.visible:
            continue
        if layer.type in _UNSUPPORTED_TYPES:
            raise FFmpegCommandError(
                f"layer type {layer.type.value!r} is not supported by the ffmpeg "
                "command builder yet"
            )
        if layer.type not in _IMAGE_TYPES and layer.type not in (
            LayerType.VIDEO,
            LayerType.TEXT,
        ):
            continue
        end = _layer_end(layer, composition)
        if end <= layer.start_seconds:
            continue
        layers.append((layer.z_index, index, layer))
    layers.sort(key=lambda item: (item[0], item[1]))
    return [layer for _, _, layer in layers]


def _fit_mode(layer: Layer, *, default: FitMode) -> FitMode:
    raw = _layer_value(layer, "fit", "fit_mode", "mode")
    if raw is None:
        return default
    try:
        return FitMode(raw)
    except (TypeError, ValueError) as exc:
        raise FFmpegCommandError(f"unsupported fit mode {raw!r}") from exc


def _box(
    layer: Layer,
    composition: Composition,
    *,
    overlay: bool,
) -> tuple[int | None, int | None, int | None, int | None]:
    position = _pair(_layer_value(layer, "position", "pos", "offset"), "position")
    size = _pair(_layer_value(layer, "size", "dimensions"), "size")
    x = _layer_value(layer, "x", "left")
    y = _layer_value(layer, "y", "top")
    width = _layer_value(layer, "width", "w")
    height = _layer_value(layer, "height", "h")

    if x is None and position is not None:
        x = position[0]
    if y is None and position is not None:
        y = position[1]
    if width is None and size is not None:
        width = size[0]
    if height is None and size is not None:
        height = size[1]

    resolved_x = None if x is None else _integer(x, "layer x")
    resolved_y = None if y is None else _integer(y, "layer y")
    if overlay:
        resolved_width = None if width is None else _integer(width, "layer width")
        resolved_height = None if height is None else _integer(height, "layer height")
    else:
        resolved_width = (
            composition.canvas.width
            if width is None
            else _integer(width, "layer width")
        )
        resolved_height = (
            composition.canvas.height
            if height is None
            else _integer(height, "layer height")
        )
    if resolved_width is not None and resolved_width < 1:
        raise FFmpegCommandError("layer width must be positive")
    if resolved_height is not None and resolved_height < 1:
        raise FFmpegCommandError("layer height must be positive")
    return resolved_x, resolved_y, resolved_width, resolved_height


def _fit_filters(width: int, height: int, fit: FitMode) -> list[str]:
    if fit is FitMode.NONE:
        return []
    if fit is FitMode.STRETCH:
        return [f"scale={width}:{height}"]
    if fit is FitMode.CONTAIN:
        return [
            f"scale={width}:{height}:force_original_aspect_ratio=decrease",
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black@0",
        ]
    return [
        f"scale={width}:{height}:force_original_aspect_ratio=increase",
        f"crop={width}:{height}",
    ]


def _overlay_static_scale(layer: Layer) -> float:
    raw_scale = _layer_value(layer, "scale", "zoom")
    scale = 1.0 if raw_scale is None else _finite(raw_scale, "scale")
    if scale <= 0.0:
        raise FFmpegCommandError("scale must be positive")
    return scale


def _overlay_scale(layer: Layer) -> str | None:
    width = _layer_value(layer, "width", "w")
    height = _layer_value(layer, "height", "h")
    scale = _overlay_static_scale(layer)

    if width is not None and height is not None:
        target_width = max(1, round(_integer(width, "layer width") * scale))
        target_height = max(1, round(_integer(height, "layer height") * scale))
        return f"scale={target_width}:{target_height}"
    if width is not None:
        target_width = max(1, round(_integer(width, "layer width") * scale))
        return f"scale={target_width}:-1"
    if height is not None:
        target_height = max(1, round(_integer(height, "layer height") * scale))
        return f"scale=-1:{target_height}"
    if scale != 1.0:
        return f"scale=iw*{scale:g}:ih*{scale:g}"
    return None


def _anchor(layer: Layer) -> str:
    raw = _layer_value(layer, "anchor", "corner", "alignment")
    if raw is None:
        return Anchor.CENTER.value
    try:
        return Anchor(raw).value
    except (TypeError, ValueError) as exc:
        raise FFmpegCommandError(f"unsupported anchor {raw!r}") from exc


def _margin(layer: Layer) -> tuple[float, float]:
    raw = _layer_value(layer, "margin", "margins", "inset")
    if raw is None:
        return 0.0, 0.0
    if isinstance(raw, (tuple, list)):
        if len(raw) != 2:
            raise FFmpegCommandError("margin must be a number or a pair of numbers")
        horizontal = _finite(raw[0], "margin")
        vertical = _finite(raw[1], "margin")
    else:
        horizontal = vertical = _finite(raw, "margin")
    if horizontal < 0.0 or vertical < 0.0:
        raise FFmpegCommandError("margin must not be negative")
    return horizontal, vertical


def _max_scale(track: KeyframeTrack, index: int = 0) -> float:
    maximum = 1.0
    for frame in track.keyframes:
        value = frame.value
        magnitude = float(value[index]) if isinstance(value, tuple) else float(value)
        maximum = max(maximum, magnitude)
    return maximum


def _overlay_axis_placement(
    *,
    axis: str,
    coordinate: int | None,
    base_size: int | None,
    box_size: int,
    canvas_size: int,
    safe_start: int,
    safe_end: int,
    margin: float,
    position: str,
) -> tuple[str, str]:
    outer = "ow" if axis == "x" else "oh"
    inner = "iw" if axis == "x" else "ih"
    center_pad = f"({outer}-{inner})/2"
    end_pad = f"{outer}-{inner}"
    if coordinate is not None:
        if base_size is None:
            return "0", f"{coordinate}"
        offset = coordinate + base_size / 2 - box_size / 2
        return center_pad, f"{offset:g}"

    if position in ("center", "middle"):
        center = safe_start + (canvas_size - safe_start - safe_end) / 2
        return center_pad, f"{center - box_size / 2:g}"
    if position in ("left", "top"):
        if base_size is None:
            return "0", f"{safe_start + margin:g}"
        center = safe_start + margin + base_size / 2
        return center_pad, f"{center - box_size / 2:g}"
    if base_size is None:
        return end_pad, f"{canvas_size - safe_end - margin - box_size:g}"
    center = canvas_size - safe_end - margin - base_size / 2
    return center_pad, f"{center - box_size / 2:g}"


def _overlay_motion_filters(
    layer: Layer,
    composition: Composition,
    tracks: MotionTracks,
    x: int | None,
    y: int | None,
    width: int | None,
    height: int | None,
    start_seconds: float,
) -> tuple[list[str], str, str, str]:
    if tracks.scale is None:
        raise FFmpegCommandError("overlay motion needs a scale track")
    variable = f"(t+{_seconds(start_seconds)})"
    scale_x_expression, scale_y_expression = _scale_expressions(
        tracks.scale,
        variable,
    )
    scale_x = f"max(({scale_x_expression}),0.0001)"
    scale_y = f"max(({scale_y_expression}),0.0001)"
    static_scale = _overlay_static_scale(layer)
    base_width: int | None = None
    base_height: int | None = None
    if width is not None and height is not None:
        base_width = max(1, round(_integer(width, "layer width") * static_scale))
        base_height = max(1, round(_integer(height, "layer height") * static_scale))
        scale_filter = (
            f"scale=w='{base_width}*({scale_x})':"
            f"h='{base_height}*({scale_y})':eval=frame"
        )
    elif width is not None:
        base_width = max(1, round(_integer(width, "layer width") * static_scale))
        scale_filter = (
            f"scale=w='{base_width}*({scale_x})':"
            f"h='ih*{base_width}/iw*({scale_y})':eval=frame"
        )
    elif height is not None:
        base_height = max(1, round(_integer(height, "layer height") * static_scale))
        scale_filter = (
            f"scale=w='iw*{base_height}/ih*({scale_x})':"
            f"h='{base_height}*({scale_y})':eval=frame"
        )
    else:
        scale_filter = (
            f"scale=w='iw*{static_scale:g}*({scale_x})':"
            f"h='ih*{static_scale:g}*({scale_y})':eval=frame"
        )

    maximum_x = _max_scale(tracks.scale, 0)
    maximum_y = _max_scale(tracks.scale, 1)
    canvas_width = composition.canvas.width
    canvas_height = composition.canvas.height
    box_width = (
        canvas_width
        if base_width is None
        else max(canvas_width, round(base_width * maximum_x))
    )
    box_height = (
        canvas_height
        if base_height is None
        else max(canvas_height, round(base_height * maximum_y))
    )
    safe_top, safe_right, safe_bottom, safe_left = effective_safe_area(
        composition
    ).to_pixels(canvas_width, canvas_height)
    margin_x, margin_y = _margin(layer)
    parts = _anchor(layer).split("_")
    x_position = (
        "left" if "left" in parts else "right" if "right" in parts else "center"
    )
    y_position = (
        "top" if "top" in parts else "bottom" if "bottom" in parts else "middle"
    )
    pad_x, overlay_x = _overlay_axis_placement(
        axis="x",
        coordinate=x,
        base_size=base_width,
        box_size=box_width,
        canvas_size=canvas_width,
        safe_start=safe_left,
        safe_end=safe_right,
        margin=margin_x,
        position=x_position,
    )
    pad_y, overlay_y = _overlay_axis_placement(
        axis="y",
        coordinate=y,
        base_size=base_height,
        box_size=box_height,
        canvas_size=canvas_height,
        safe_start=safe_top,
        safe_end=safe_bottom,
        margin=margin_y,
        position=y_position,
    )
    pad = (
        f"pad=w={box_width}:h={box_height}:"
        f"x='{pad_x}':y='{pad_y}':color=black@0:eval=frame"
    )
    return [scale_filter], pad, overlay_x, overlay_y


def _anchor_position(layer: Layer, composition: Composition) -> tuple[str, str]:
    anchor = _anchor(layer)
    margin_x, margin_y = _margin(layer)
    safe_top, safe_right, safe_bottom, safe_left = effective_safe_area(
        composition
    ).to_pixels(composition.canvas.width, composition.canvas.height)
    left = f"{safe_left + margin_x:g}"
    top = f"{safe_top + margin_y:g}"
    right = f"W-w-{safe_right + margin_x:g}"
    bottom = f"H-h-{safe_bottom + margin_y:g}"
    if safe_left == 0 and safe_right == 0:
        center_x = "(W-w)/2"
    else:
        center_x = f"{safe_left}+(W-w-{safe_left + safe_right})/2"
    if safe_top == 0 and safe_bottom == 0:
        center_y = "(H-h)/2"
    else:
        center_y = f"{safe_top}+(H-h-{safe_top + safe_bottom})/2"
    positions = {
        "top_left": (left, top),
        "top_center": (center_x, top),
        "top_right": (right, top),
        "middle_left": (left, center_y),
        "center": (center_x, center_y),
        "middle_right": (right, center_y),
        "bottom_left": (left, bottom),
        "bottom_center": (center_x, bottom),
        "bottom_right": (right, bottom),
    }
    try:
        return positions[anchor]
    except KeyError as exc:
        raise FFmpegCommandError(f"unsupported anchor {anchor!r}") from exc


def _position(
    layer: Layer,
    composition: Composition,
    *,
    x: int | None,
    y: int | None,
    anchored: bool,
) -> tuple[str, str]:
    if x is not None or y is not None:
        return str(0 if x is None else x), str(0 if y is None else y)
    if anchored or _layer_value(layer, "anchor", "corner", "alignment") is not None:
        return _anchor_position(layer, composition)
    return "0", "0"


def _image_input(layer: Layer, composition: Composition) -> FFmpegInput:
    source = _layer_value(layer, "source", "path", "file", "asset", "asset_id")
    if not isinstance(source, str) or not source.strip():
        raise FFmpegCommandError("image layer has no source")
    duration = _layer_end(layer, composition) - layer.start_seconds
    return FFmpegInput(
        source=source.strip(),
        options=("-loop", "1", "-t", _seconds(duration)),
        layer_id=layer.id,
    )


def _mask_input(layer: Layer, composition: Composition) -> FFmpegInput | None:
    mask = None if layer.type is LayerType.TEXT else _mask_for_layer(layer)
    if mask is None:
        return None
    duration = _layer_end(layer, composition) - layer.start_seconds
    return FFmpegInput(
        source=mask.source,
        options=("-loop", "1", "-t", _seconds(duration)),
        layer_id=f"{layer.id}:mask",
    )


def _video_input(layer: Layer) -> FFmpegInput:
    source = _layer_value(layer, "source", "path", "file", "asset", "asset_id")
    if not isinstance(source, str) or not source.strip():
        raise FFmpegCommandError("video layer has no source")
    loop = bool(_layer_value(layer, "loop", "looped") or False)
    options: tuple[str, ...] = ("-stream_loop", "-1") if loop else ()
    return FFmpegInput(source=source.strip(), options=options, layer_id=layer.id)


def _text_input(layer: Layer, composition: Composition) -> FFmpegInput:
    duration = _layer_end(layer, composition) - layer.start_seconds
    canvas = composition.canvas
    return FFmpegInput(
        source=(
            f"color=c=black@0.0:s={canvas.width}x{canvas.height}:"
            f"r={composition.fps}:d={_seconds(duration)}"
        ),
        options=("-f", "lavfi"),
        layer_id=layer.id,
    )


def _filter_text(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace(":", "\\:")
        .replace("%", "\\%")
        .replace("\n", "\\n")
    )


def _text_filters(layer: Layer, x: int | None, y: int | None) -> list[str]:
    raw_text = _layer_value(layer, "text", "content", "value")
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise FFmpegCommandError("text layer has no text")
    raw_family = _layer_value(layer, "font_family", "font", "family") or ""
    if not isinstance(raw_family, str):
        raise FFmpegCommandError("font_family must be a string")
    raw_size = _layer_value(layer, "font_size", "size", "font_size_px") or 48.0
    font_size = _finite(raw_size, "font_size")
    if font_size <= 0.0:
        raise FFmpegCommandError("font_size must be positive")
    raw_color = _layer_value(layer, "color", "fill", "text_color") or "#ffffff"
    if not isinstance(raw_color, str):
        raise FFmpegCommandError("text colour must be a string")
    weight = _layer_value(layer, "font_weight", "weight", "font-weight")
    style = _layer_value(layer, "font_style", "style", "font-style")
    path = font_manager().resolve(
        raw_family.strip() or None,
        weight=None if weight is None else TextWeight(weight),
        style=None if style is None else TextStyle(style),
    )
    options = [
        f"fontfile='{_filter_text(str(path))}'",
        f"text='{_filter_text(raw_text)}'",
        f"fontsize={font_size:g}",
        f"fontcolor={_ffmpeg_color(raw_color)}",
        f"x={0 if x is None else x}",
        f"y={0 if y is None else y}",
    ]
    raw_stroke = _layer_value(layer, "stroke", "stroke_color", "outline")
    raw_stroke_width = _layer_value(layer, "stroke_width", "outline_width") or 0.0
    stroke_width = _finite(raw_stroke_width, "stroke_width")
    if isinstance(raw_stroke, str) and raw_stroke.strip() and stroke_width > 0.0:
        options.extend(
            (
                f"borderw={stroke_width:g}",
                f"bordercolor={_ffmpeg_color(raw_stroke)}",
            )
        )
    return [f"drawtext={':'.join(options)}"]


def _video_filters(layer: Layer, composition: Composition) -> list[str]:
    visible = _layer_end(layer, composition) - layer.start_seconds
    trim_start = _finite(
        _layer_value(layer, "trim_start_seconds", "trim_start", "source_start") or 0.0,
        "trim_start_seconds",
    )
    trim_end = _layer_value(layer, "trim_end_seconds", "trim_end", "source_end")
    rate = _finite(
        _layer_value(layer, "playback_rate", "speed", "rate") or 1.0,
        "playback_rate",
    )
    if rate <= 0.0:
        raise FFmpegCommandError("playback_rate must be positive")
    if trim_end is None:
        trim_end = trim_start + visible * rate
    else:
        trim_end = _finite(trim_end, "trim_end_seconds")
    if trim_end <= trim_start:
        raise FFmpegCommandError(
            "trim_end_seconds must be greater than trim_start_seconds"
        )
    return [
        f"trim=start={trim_start:.6f}:end={trim_end:.6f}",
        f"setpts=(PTS-STARTPTS)/{rate:g}",
        f"fps={composition.fps}",
    ]


def _scale_expressions(track: KeyframeTrack, variable: str) -> tuple[str, str]:
    first = track.keyframes[0].value
    if isinstance(first, tuple):
        return pair_expression(track, variable=variable)
    expression = track_expression(track, variable=variable)
    return expression, expression


def _pair_scale_motion_filters(
    tracks: MotionTracks,
    composition: Composition,
    width: int | None,
    height: int | None,
    start_seconds: float,
) -> list[str]:
    if tracks.scale is None:
        return []
    box_width = composition.canvas.width if width is None else width
    box_height = composition.canvas.height if height is None else height
    variable = f"(t+{_seconds(start_seconds)})"
    scale_x, scale_y = _scale_expressions(tracks.scale, variable)
    scale_x = f"max(({scale_x}),0.0001)"
    scale_y = f"max(({scale_y}),0.0001)"
    if tracks.pan is None:
        pan_x, pan_y = "0", "0"
    else:
        pan_x, pan_y = pair_expression(tracks.pan, variable=variable)
    maximum_x = _max_scale(tracks.scale, 0)
    maximum_y = _max_scale(tracks.scale, 1)
    pad_width = max(1, math.ceil(box_width * maximum_x))
    pad_height = max(1, math.ceil(box_height * maximum_y))
    return [
        f"fps={composition.fps}",
        f"scale=w='{box_width}*({scale_x})':h='{box_height}*({scale_y})':eval=frame",
        f"pad=w={pad_width}:h={pad_height}:"
        "x='(ow-iw)/2':y='(oh-ih)/2':color=black@0:eval=frame",
        f"crop=w={box_width}:h={box_height}:"
        f"x='(iw-ow)/2+({pan_x})*({scale_x})':"
        f"y='(ih-oh)/2+({pan_y})*({scale_y})'",
    ]


def _motion_filters(
    tracks: MotionTracks,
    composition: Composition,
    width: int | None,
    height: int | None,
    start_seconds: float,
) -> list[str]:
    if tracks.scale is None and tracks.pan is None:
        return []
    if tracks.scale is not None and isinstance(tracks.scale.keyframes[0].value, tuple):
        return _pair_scale_motion_filters(
            tracks,
            composition,
            width,
            height,
            start_seconds,
        )
    box_width = composition.canvas.width if width is None else width
    box_height = composition.canvas.height if height is None else height
    variable = f"(on/{composition.fps}+{_seconds(start_seconds)})"
    zoom = "1"
    if tracks.scale is not None:
        zoom = _scale_expressions(tracks.scale, variable)[0]
    if tracks.pan is None:
        pan_x, pan_y = "0", "0"
    else:
        pan_x, pan_y = pair_expression(tracks.pan, variable=variable)
    x_expression = f"iw/2-(iw/zoom/2)+({pan_x})"
    y_expression = f"ih/2-(ih/zoom/2)+({pan_y})"
    return [
        f"fps={composition.fps}",
        f"zoompan=z='{zoom}':x='{x_expression}':y='{y_expression}':"
        f"d=1:s={box_width}x{box_height}:fps={composition.fps}",
    ]


def _rotation_filters(tracks: MotionTracks, start_seconds: float) -> list[str]:
    if tracks.rotation is None:
        return []
    variable = f"(t+{_seconds(start_seconds)})"
    angle = track_expression(tracks.rotation, variable=variable)
    return [f"rotate=a='({angle})*PI/180':ow=iw:oh=ih:c=none"]


def _transition_alpha(transition: Any) -> str | None:
    if transition is None or not transition.is_crossfade:
        return None
    span = transition.duration_seconds
    if span <= 0.0:
        return None
    return f"clip(T/{_seconds(span)},0,1)"


def _opacity_filters(
    layer: Layer,
    tracks: MotionTracks,
    start_seconds: float,
    transition: Any = None,
) -> list[str]:
    fade = _transition_alpha(transition)
    if tracks.opacity is not None:
        variable = f"(T+{_seconds(start_seconds)})"
        alpha = track_expression(tracks.opacity, variable=variable)
        if fade is not None:
            alpha = f"({alpha})*({fade})"
        return [f"geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='alpha(X,Y)*({alpha})'"]
    opacity = _finite(layer.opacity, "opacity")
    if fade is not None:
        alpha = f"{opacity:.6f}*({fade})"
        return [f"geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='alpha(X,Y)*({alpha})'"]
    if opacity < 1.0:
        return [f"colorchannelmixer=aa={opacity:.6f}"]
    return []


def _blend_mode(layer: Layer) -> BlendMode:
    raw = _layer_value(layer, "blend_mode", "blend", "blending")
    if raw is None:
        return BlendMode.NORMAL
    if isinstance(raw, BlendMode):
        return raw
    try:
        return BlendMode(raw)
    except (TypeError, ValueError) as exc:
        raise FFmpegCommandError(f"unsupported blend mode {raw!r}") from exc


def _ffmpeg_blend_mode(mode: BlendMode) -> str:
    names = {
        BlendMode.NORMAL: "normal",
        BlendMode.MULTIPLY: "multiply",
        BlendMode.SCREEN: "screen",
        BlendMode.OVERLAY: "overlay",
        BlendMode.ADD: "addition",
        BlendMode.SUBTRACT: "subtract",
        BlendMode.DARKEN: "darken",
        BlendMode.LIGHTEN: "lighten",
        BlendMode.DIFFERENCE: "difference",
        BlendMode.HARD_LIGHT: "hardlight",
        BlendMode.SOFT_LIGHT: "softlight",
    }
    try:
        return names[mode]
    except KeyError as exc:
        raise FFmpegCommandError(f"unsupported blend mode {mode.value!r}") from exc


def _image_filter_specs(layer: Layer) -> list[ImageFilter]:
    raw = _layer_value(layer, "filters", "filter", "image_filters")
    if raw is None:
        return []
    specs: list[ImageFilter] = []
    for item in raw:
        if isinstance(item, ImageFilter):
            specs.append(item)
            continue
        try:
            specs.append(ImageFilter.model_validate(item))
        except ValueError as exc:
            raise FFmpegCommandError("invalid image filter") from exc
    return specs


def _image_filter_filters(layer: Layer) -> list[str]:
    filters: list[str] = []
    for spec in _image_filter_specs(layer):
        radius = spec.amount if spec.radius is None else spec.radius
        if spec.type is FilterType.BLUR:
            filters.append(f"gblur=sigma={radius:g}")
        elif spec.type is FilterType.BOX_BLUR:
            filters.append(f"boxblur={radius:g}")
        elif spec.type in (FilterType.SHARPEN, FilterType.UNSHARP_MASK):
            filters.append(f"unsharp=5:5:{spec.amount:g}")
        elif spec.type is FilterType.BRIGHTNESS:
            filters.append(f"eq=brightness={spec.amount - 1.0:.6f}")
        elif spec.type is FilterType.CONTRAST:
            filters.append(f"eq=contrast={spec.amount:.6f}")
        elif spec.type is FilterType.SATURATION:
            filters.append(f"eq=saturation={spec.amount:.6f}")
        elif spec.type is FilterType.GRAYSCALE:
            filters.append("hue=s=0")
        elif spec.type is FilterType.INVERT:
            filters.append("negate")
        elif spec.type is FilterType.AUTOCONTRAST:
            filters.append("normalize")
    return filters


def _mask_for_layer(layer: Layer) -> ImageMask | None:
    raw = _layer_value(layer, "mask", "image_mask")
    if raw is None:
        return None
    if isinstance(raw, ImageMask):
        return raw
    if isinstance(raw, str):
        raw = {"source": raw}
    try:
        return ImageMask.model_validate(raw)
    except ValueError as exc:
        raise FFmpegCommandError("invalid image mask") from exc


def _mask_filters(
    mask: ImageMask,
    width: int | None,
    height: int | None,
) -> list[str]:
    filters: list[str] = []
    if width is not None and height is not None:
        filters.extend(_fit_filters(width, height, mask.fit))
    if mask.mode is MaskMode.ALPHA:
        filters.append("alphaextract")
    else:
        filters.append("format=gray")
    if mask.invert:
        filters.append("negate")
    if mask.feather > 0.0:
        filters.append(f"gblur=sigma={mask.feather:g}")
    return filters


def _input_plan(
    layers: Sequence[Layer],
) -> list[tuple[int, int | None]]:
    plan: list[tuple[int, int | None]] = []
    next_index = 1
    for layer in layers:
        layer_index = next_index
        next_index += 1
        mask_index: int | None = None
        if layer.type is not LayerType.TEXT and _mask_for_layer(layer) is not None:
            mask_index = next_index
            next_index += 1
        plan.append((layer_index, mask_index))
    return plan


def _layer_filters(
    layer: Layer,
    composition: Composition,
    index: int,
    input_index: int,
    mask_index: int | None,
) -> tuple[str, str]:
    overlay = layer.type in _OVERLAY_TYPES
    x, y, width, height = _box(layer, composition, overlay=overlay)
    tracks = tracks_for_layer(
        layer,
        layer.start_seconds,
        _layer_end(layer, composition),
    )
    filters: list[str] = []
    overlay_pad: str | None = None
    overlay_motion_x: str | None = None
    overlay_motion_y: str | None = None

    if layer.type is LayerType.TEXT:
        if x is None or y is None:
            safe_top, _safe_right, _safe_bottom, safe_left = effective_safe_area(
                composition
            ).to_pixels(composition.canvas.width, composition.canvas.height)
            x = safe_left if x is None else x
            y = safe_top if y is None else y
        filters.extend(_text_filters(layer, x, y))
    elif layer.type is LayerType.VIDEO:
        filters.extend(_video_filters(layer, composition))
        fit = _fit_mode(layer, default=FitMode.COVER)
        if width is not None and height is not None:
            filters.extend(_fit_filters(width, height, fit))
    elif overlay:
        if tracks.scale is not None:
            motion_x, motion_y = x, y
            if x is not None or y is not None:
                motion_x = 0 if x is None else x
                motion_y = 0 if y is None else y
            motion_filters, overlay_pad, overlay_motion_x, overlay_motion_y = (
                _overlay_motion_filters(
                    layer,
                    composition,
                    tracks,
                    motion_x,
                    motion_y,
                    width,
                    height,
                    layer.start_seconds,
                )
            )
            filters.extend(motion_filters)
        else:
            scale = _overlay_scale(layer)
            if scale is not None:
                filters.append(scale)
    else:
        fit = _fit_mode(layer, default=FitMode.COVER)
        if width is not None and height is not None:
            filters.extend(_fit_filters(width, height, fit))
    if not overlay:
        filters.extend(
            _motion_filters(
                tracks,
                composition,
                width,
                height,
                layer.start_seconds,
            )
        )
    filters.extend(_rotation_filters(tracks, layer.start_seconds))
    filters.append("setsar=1")
    filters.append("format=rgba")
    if overlay_pad is not None:
        filters.append(overlay_pad)
    filters.extend(_image_filter_filters(layer))
    transition = layer_transition_for(layer)
    tail_filters = _opacity_filters(
        layer,
        tracks,
        layer.start_seconds,
        transition,
    )
    tail_filters.append(f"setpts=PTS-STARTPTS+{layer.start_seconds:.6f}/TB")

    label = f"v{index}"
    mask = None if layer.type is LayerType.TEXT else _mask_for_layer(layer)
    if mask is None:
        node = f"[{input_index}:v]{','.join([*filters, *tail_filters])}[{label}]"
    else:
        if mask_index is None:
            raise FFmpegCommandError("image mask has no ffmpeg input")
        pre_label = f"p{index}"
        mask_label = f"m{index}"
        mask_filters = _mask_filters(mask, width, height)
        node = (
            f"[{input_index}:v]{','.join(filters)}[{pre_label}];"
            f"[{mask_index}:v]{','.join(mask_filters)}[{mask_label}];"
            f"[{pre_label}][{mask_label}]alphamerge,"
            f"{','.join(tail_filters)}[{label}]"
        )
    overlay_x, overlay_y = _position(
        layer,
        composition,
        x=x,
        y=y,
        anchored=overlay,
    )
    if overlay_motion_x is not None and overlay_motion_y is not None:
        overlay_x, overlay_y = overlay_motion_x, overlay_motion_y
    if layer.type is LayerType.TEXT:
        overlay_x, overlay_y = "0", "0"
    if tracks.position is not None:
        position_x, position_y = pair_expression(tracks.position, variable="t")
        overlay_x = f"'({overlay_x})+({position_x})'"
        overlay_y = f"'({overlay_y})+({position_y})'"
    if overlay and tracks.pan is not None:
        pan_x, pan_y = pair_expression(tracks.pan, variable="t")
        overlay_x = f"'({overlay_x})+({pan_x})'"
        overlay_y = f"'({overlay_y})+({pan_y})'"
    if transition is not None and transition.is_slide:
        span = transition.duration_seconds
        progress = f"clip((t-{layer.start_seconds:.6f})/{_seconds(span)},0,1)"
        offset_x, offset_y = transition.expression_offset(
            (composition.canvas.width, composition.canvas.height)
        )
        if offset_x != "0":
            overlay_x = f"'({overlay_x})+({offset_x})*(1-({progress}))'"
        if offset_y != "0":
            overlay_y = f"'({overlay_y})+({offset_y})*(1-({progress}))'"
    end = _layer_end(layer, composition)
    overlay_options = (
        f"x={overlay_x}:y={overlay_y}:"
        "eof_action=pass:repeatlast=0:shortest=0:"
        f"enable='between(t,{layer.start_seconds:.6f},{end:.6f})'"
    )
    mode = _blend_mode(layer)
    if mode is BlendMode.NORMAL:
        overlay_node = (
            f"[base{index}][{label}]overlay={overlay_options}[base{index + 1}]"
        )
    else:
        base_top = f"base{index}a"
        base_bottom = f"base{index}b"
        layer_top = f"{label}a"
        layer_bottom = f"{label}b"
        blended = f"blend{index}"
        alpha = f"alpha{index}"
        composite = f"composite{index}"
        overlay_node = ";".join(
            (
                f"[base{index}]split[{base_top}][{base_bottom}]",
                f"[{label}]split[{layer_top}][{layer_bottom}]",
                f"[{layer_top}][{base_top}]blend="
                f"all_mode={_ffmpeg_blend_mode(mode)}:shortest=1[{blended}]",
                f"[{layer_bottom}]alphaextract[{alpha}]",
                f"[{blended}][{alpha}]alphamerge[{composite}]",
                f"[{base_bottom}][{composite}]overlay="
                f"{overlay_options}[base{index + 1}]",
            )
        )
    return node, overlay_node


def _background_input(composition: Composition) -> FFmpegInput:
    canvas = composition.canvas
    source = (
        f"color=c={_ffmpeg_color(canvas.background)}:"
        f"s={canvas.width}x{canvas.height}:"
        f"r={composition.fps}:"
        f"d={_seconds(composition.duration_seconds)}"
    )
    return FFmpegInput(source=source, options=("-f", "lavfi"))


def _audio_inputs(composition: Composition) -> list[FFmpegInput]:
    inputs: list[FFmpegInput] = []
    for track in composition.audio_tracks:
        if not track.enabled:
            continue
        options = ("-stream_loop", "-1") if track.loop else ()
        inputs.append(
            FFmpegInput(
                source=track.source,
                options=options,
                layer_id=track.id,
            )
        )
    return inputs


def _subtitle_path(
    composition: Composition,
    subtitle_file: str | Path | None,
) -> str | None:
    if subtitle_file is not None:
        path = str(subtitle_file).strip()
        return path or None
    return composition.subtitle_file.strip() or None


def build_ffmpeg_filtergraph(
    composition: Composition | Mapping[str, Any],
    *,
    subtitle_file: str | Path | None = None,
    include_audio: bool = True,
    audio_settings: Any | None = None,
    loudness_lufs: float | None = None,
    normalize_loudness: bool | None = None,
    ducking: Any | None = None,
    output_width: int | None = None,
    output_height: int | None = None,
) -> str:
    """Return the pure ``filter_complex`` graph for *composition*.

    Input zero is always the generated background. Each supported layer adds
    one input and one overlay node, in ``z_index`` order. Enabled audio tracks
    add their mix graph after the video nodes, and a written ASS file adds the
    ``ass`` filter that burns subtitles over the final frame. When
    *output_width* and *output_height* are given, the finished frame is scaled
    to that size before it is mapped to ``[vout]``.
    """
    value = _composition(composition)
    target = _output_size(output_width, output_height)
    layers = _active_layers(value)
    plan = _input_plan(layers)
    parts = [
        f"color=c={_ffmpeg_color(value.canvas.background)}:"
        f"s={value.canvas.width}x{value.canvas.height}:"
        f"r={value.fps}:d={_seconds(value.duration_seconds)}[base0]"
    ]
    for index, layer in enumerate(layers):
        input_index, mask_index = plan[index]
        node, overlay = _layer_filters(
            layer,
            value,
            index,
            input_index,
            mask_index,
        )
        parts.extend((node, overlay))
    video_label = f"base{len(layers)}"
    path = _subtitle_path(value, subtitle_file)
    if value.subtitles is not None and path is None:
        raise FFmpegCommandError(
            "subtitles need a written ASS file; pass subtitle_file"
        )
    if path is not None:
        subtitle_label = "vout" if target is None else "subbed"
        parts.append(f"[{video_label}]{subtitle_filter(path)}[{subtitle_label}]")
        video_label = subtitle_label
    if target is not None:
        parts.append(
            f"[{video_label}]scale={target[0]}:{target[1]}:flags=lanczos[vout]"
        )
    elif path is None:
        parts.append(f"[{video_label}]null[vout]")

    if include_audio and any(track.enabled for track in value.audio_tracks):
        mask_inputs = sum(1 for _index, mask_index in plan if mask_index is not None)
        audio_graph = build_audio_filtergraph(
            value,
            settings=audio_settings,
            input_offset=1 + len(layers) + mask_inputs,
            loudness_lufs=loudness_lufs,
            normalize_loudness=normalize_loudness,
            ducking=ducking,
        )
        if audio_graph:
            parts.append(audio_graph)
    return ";".join(parts)


def _command_inputs(composition: Composition) -> list[FFmpegInput]:
    inputs: list[FFmpegInput] = []
    layers = _active_layers(composition)
    for layer, (_input_index, mask_index) in zip(
        layers,
        _input_plan(layers),
        strict=True,
    ):
        if layer.type is LayerType.VIDEO:
            inputs.append(_video_input(layer))
        elif layer.type is LayerType.TEXT:
            inputs.append(_text_input(layer, composition))
        else:
            inputs.append(_image_input(layer, composition))
        if mask_index is not None:
            mask_input = _mask_input(layer, composition)
            if mask_input is not None:
                inputs.append(mask_input)
    return inputs


def build_ffmpeg_command(
    composition: Composition | Mapping[str, Any],
    output: str | Path = "out.mp4",
    *,
    executable: str = "ffmpeg",
    video_codec: str | None = None,
    audio_codec: str | None = None,
    preset: str = DEFAULT_PRESET,
    crf: int = DEFAULT_CRF,
    bitrate: int | None = None,
    audio_bitrate: int | None = None,
    pixel_format: str = DEFAULT_PIXEL_FORMAT,
    movflags: str | None = "+faststart",
    overwrite: bool = True,
    subtitle_file: str | Path | None = None,
    include_audio: bool = True,
    audio_settings: Any | None = None,
    loudness_lufs: float | None = None,
    normalize_loudness: bool | None = None,
    ducking: Any | None = None,
    output_profile: OutputProfile | Mapping[str, Any] | None = None,
    config: Any | None = None,
    video: Any | None = None,
    platform: str | None = None,
    profile: str | None = None,
    output_width: int | None = None,
    output_height: int | None = None,
) -> FFmpegCommand:
    """Build an ffmpeg argv for *composition* without executing anything.

    ``output`` is only stringified; it does not have to exist and its parent
    directory is never created or inspected. The returned value is a list of
    arguments that also carries the generated filtergraph and inputs. Audio
    tracks are mixed in the same command and a *subtitle_file* is burned with
    the ``ass`` filter; the builder never writes that file itself.

    Passing *config* (or *video*) resolves the video configuration with the
    optional *platform* and *profile* overrides and applies it to the
    composition. An explicit *output_profile* wins over configuration. Codec,
    bitrate and loudness arguments still win over the resolved profile.
    *output_width* and *output_height* scale the finished frame to a smaller
    size, which is how a preview renders at reduced resolution.
    """
    if not isinstance(executable, str) or not executable.strip():
        raise FFmpegCommandError("executable must be a non-empty string")

    value = _composition(composition)
    if output_profile is not None or config is not None or video is not None:
        resolved = (
            output_profile
            if isinstance(output_profile, OutputProfile)
            else (
                OutputProfile.from_mapping(output_profile)
                if output_profile is not None
                else resolve_output_profile(
                    config if config is not None else video,
                    platform=platform,
                    profile=profile,
                )
            )
        )
        value = apply_output_profile(value, resolved)
        if video_codec is None:
            video_codec = resolved.codec
        if audio_codec is None:
            audio_codec = resolved.audio_codec
        bitrate = bitrate if bitrate is not None else resolved.bitrate
        audio_bitrate = (
            audio_bitrate if audio_bitrate is not None else resolved.audio_bitrate
        )
        loudness_lufs = (
            loudness_lufs if loudness_lufs is not None else resolved.loudness_lufs
        )

    video_codec = DEFAULT_VIDEO_CODEC if video_codec is None else video_codec
    audio_codec = DEFAULT_AUDIO_CODEC if audio_codec is None else audio_codec

    if not isinstance(video_codec, str) or not video_codec.strip():
        raise FFmpegCommandError("video_codec must be a non-empty string")
    if not isinstance(audio_codec, str) or not audio_codec.strip():
        raise FFmpegCommandError("audio_codec must be a non-empty string")
    if isinstance(crf, bool) or not isinstance(crf, int):
        raise FFmpegCommandError("crf must be an integer")
    if crf < 0:
        raise FFmpegCommandError("crf must not be negative")
    if bitrate is not None:
        if isinstance(bitrate, bool) or not isinstance(bitrate, int):
            raise FFmpegCommandError("bitrate must be an integer")
        if bitrate <= 0:
            raise FFmpegCommandError("bitrate must be positive")
    if audio_bitrate is not None:
        if isinstance(audio_bitrate, bool) or not isinstance(audio_bitrate, int):
            raise FFmpegCommandError("audio_bitrate must be an integer")
        if audio_bitrate <= 0:
            raise FFmpegCommandError("audio_bitrate must be positive")
    if not isinstance(preset, str) or not preset.strip():
        raise FFmpegCommandError("preset must be a non-empty string")
    if not isinstance(pixel_format, str) or not pixel_format.strip():
        raise FFmpegCommandError("pixel_format must be a non-empty string")
    if movflags is not None and not isinstance(movflags, str):
        raise FFmpegCommandError("movflags must be a string or None")

    inputs = _command_inputs(value)
    audio = _audio_inputs(value) if include_audio else []
    settings = None
    if audio:
        if audio_settings is None:
            settings = audio_mix_settings(
                value,
                loudness_lufs=loudness_lufs,
                normalize_loudness=normalize_loudness,
                ducking=ducking,
            )
        else:
            from aivia.media.audio_mix import AudioMixSettings

            settings = AudioMixSettings.from_value(audio_settings)
    graph = build_ffmpeg_filtergraph(
        value,
        subtitle_file=subtitle_file,
        include_audio=bool(audio),
        audio_settings=settings,
        loudness_lufs=loudness_lufs,
        normalize_loudness=normalize_loudness,
        ducking=ducking,
        output_width=output_width,
        output_height=output_height,
    )
    background = _background_input(value)

    argv: list[str] = [executable.strip(), "-hide_banner", "-loglevel", "error"]
    if overwrite:
        argv.append("-y")
    argv.extend(background.argv())
    for item in (*inputs, *audio):
        argv.extend(item.argv())
    argv.extend(
        [
            "-filter_complex",
            graph,
            "-map",
            "[vout]",
            "-t",
            _seconds(value.duration_seconds),
            "-r",
            str(value.fps),
            "-c:v",
            video_codec.strip(),
            "-preset",
            preset.strip(),
        ]
    )
    if bitrate is None:
        argv.extend(["-crf", str(crf)])
    else:
        argv.extend(["-b:v", f"{bitrate}k"])
    argv.extend(["-pix_fmt", pixel_format.strip()])
    if audio:
        argv.extend(["-map", "[out]", "-c:a", audio_codec.strip()])
        if audio_bitrate is not None:
            argv.extend(["-b:a", f"{audio_bitrate}k"])
        if settings is not None:
            argv.extend(["-ar", str(settings.sample_rate)])
            argv.extend(["-ac", str(settings.channels)])
    else:
        argv.append("-an")
    if movflags:
        argv.extend(["-movflags", movflags])
    argv.append(str(output))
    return FFmpegCommand(
        argv,
        filtergraph=graph,
        output=str(output),
        inputs=(background, *inputs, *audio),
    )


def build_ffmpeg_args(
    composition: Composition | Mapping[str, Any],
    output: str | Path = "out.mp4",
    **options: Any,
) -> list[str]:
    """Return just the argv list for *composition*."""
    return build_ffmpeg_command(composition, output, **options).argv


def build_filtergraph(
    composition: Composition | Mapping[str, Any],
    **options: Any,
) -> str:
    """Alias for :func:`build_ffmpeg_filtergraph`."""
    return build_ffmpeg_filtergraph(composition, **options)


def ffmpeg_filtergraph(
    composition: Composition | Mapping[str, Any],
    **options: Any,
) -> str:
    """Alias for :func:`build_ffmpeg_filtergraph`."""
    return build_ffmpeg_filtergraph(composition, **options)


def ffmpeg_args(
    composition: Composition | Mapping[str, Any],
    output: str | Path = "out.mp4",
    **options: Any,
) -> list[str]:
    """Alias for :func:`build_ffmpeg_args`."""
    return build_ffmpeg_args(composition, output, **options)


def ffmpeg_command(
    composition: Composition | Mapping[str, Any],
    output: str | Path = "out.mp4",
    **options: Any,
) -> FFmpegCommand:
    """Alias for :func:`build_ffmpeg_command`."""
    return build_ffmpeg_command(composition, output, **options)
