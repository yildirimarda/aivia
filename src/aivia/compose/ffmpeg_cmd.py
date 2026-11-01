"""Pure FFmpeg command construction for a composition.

The builder never runs ffmpeg, touches the network or reads an asset. It turns
a validated composition into two inspectable products: a ``filter_complex``
string and an argv list. That separation keeps command construction testable
without an external binary while leaving execution to the caller.

Visual layers are composited over a generated colour background. Image, video,
sticker and logo layers become inputs; their geometry, fit, opacity and time
window become filter nodes. Audio mixing, transitions and text rendering are
deliberately left to later composition steps.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import RenderError
from .keyframes import KeyframeTrack
from .motion import (
    MotionTracks,
    pair_expression,
    track_expression,
    tracks_for_layer,
)
from .schema import Anchor, Composition, FitMode, Layer, LayerType

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
_UNSUPPORTED_TYPES = frozenset({LayerType.TEXT, LayerType.SHAPE})


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
        if layer.type not in _IMAGE_TYPES and layer.type is not LayerType.VIDEO:
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


def _overlay_scale(layer: Layer) -> str | None:
    width = _layer_value(layer, "width", "w")
    height = _layer_value(layer, "height", "h")
    raw_scale = _layer_value(layer, "scale", "zoom")
    scale = 1.0 if raw_scale is None else _finite(raw_scale, "scale")
    if scale <= 0.0:
        raise FFmpegCommandError("scale must be positive")

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


def _anchor_position(layer: Layer) -> tuple[str, str]:
    anchor = _anchor(layer)
    margin_x, margin_y = _margin(layer)
    left = f"{margin_x:g}"
    top = f"{margin_y:g}"
    right = f"W-w-{margin_x:g}"
    bottom = f"H-h-{margin_y:g}"
    center_x = "(W-w)/2"
    center_y = "(H-h)/2"
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
    *,
    x: int | None,
    y: int | None,
    anchored: bool,
) -> tuple[str, str]:
    if x is not None or y is not None:
        return str(0 if x is None else x), str(0 if y is None else y)
    if anchored or _layer_value(layer, "anchor", "corner", "alignment") is not None:
        return _anchor_position(layer)
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


def _video_input(layer: Layer) -> FFmpegInput:
    source = _layer_value(layer, "source", "path", "file", "asset", "asset_id")
    if not isinstance(source, str) or not source.strip():
        raise FFmpegCommandError("video layer has no source")
    loop = bool(_layer_value(layer, "loop", "looped") or False)
    options: tuple[str, ...] = ("-stream_loop", "-1") if loop else ()
    return FFmpegInput(source=source.strip(), options=options, layer_id=layer.id)


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


def _scale_expression(track: KeyframeTrack, variable: str) -> str:
    first = track.keyframes[0].value
    if isinstance(first, tuple):
        x_expression, y_expression = pair_expression(track, variable=variable)
        return f"sqrt(({x_expression})*({y_expression}))"
    return track_expression(track, variable=variable)


def _motion_filters(
    tracks: MotionTracks,
    composition: Composition,
    width: int | None,
    height: int | None,
    start_seconds: float,
) -> list[str]:
    if tracks.scale is None and tracks.pan is None:
        return []
    box_width = composition.canvas.width if width is None else width
    box_height = composition.canvas.height if height is None else height
    variable = f"(on/{composition.fps}+{_seconds(start_seconds)})"
    zoom = "1" if tracks.scale is None else _scale_expression(tracks.scale, variable)
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


def _opacity_filters(
    layer: Layer,
    tracks: MotionTracks,
    start_seconds: float,
) -> list[str]:
    if tracks.opacity is not None:
        variable = f"(T+{_seconds(start_seconds)})"
        alpha = track_expression(tracks.opacity, variable=variable)
        return [f"geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='alpha(X,Y)*({alpha})'"]
    opacity = _finite(layer.opacity, "opacity")
    if opacity < 1.0:
        return [f"colorchannelmixer=aa={opacity:.6f}"]
    return []


def _layer_filters(
    layer: Layer,
    composition: Composition,
    index: int,
) -> tuple[str, str]:
    overlay = layer.type in _OVERLAY_TYPES
    x, y, width, height = _box(layer, composition, overlay=overlay)
    tracks = tracks_for_layer(
        layer,
        layer.start_seconds,
        _layer_end(layer, composition),
    )
    filters: list[str] = []

    if layer.type is LayerType.VIDEO:
        filters.extend(_video_filters(layer, composition))
        fit = _fit_mode(layer, default=FitMode.COVER)
        if width is not None and height is not None:
            filters.extend(_fit_filters(width, height, fit))
    elif overlay:
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
    filters.extend(_opacity_filters(layer, tracks, layer.start_seconds))
    filters.append(f"setpts=PTS-STARTPTS+{layer.start_seconds:.6f}/TB")

    label = f"v{index}"
    node = f"[{index + 1}:v]{','.join(filters)}[{label}]"
    overlay_x, overlay_y = _position(
        layer,
        x=x,
        y=y,
        anchored=overlay,
    )
    if tracks.position is not None:
        position_x, position_y = pair_expression(tracks.position, variable="t")
        overlay_x = f"'({overlay_x})+({position_x})'"
        overlay_y = f"'({overlay_y})+({position_y})'"
    end = _layer_end(layer, composition)
    overlay_node = (
        f"[base{index}][{label}]overlay="
        f"x={overlay_x}:y={overlay_y}:"
        "eof_action=pass:repeatlast=0:shortest=0:"
        f"enable='between(t,{layer.start_seconds:.6f},{end:.6f})'"
        f"[base{index + 1}]"
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


def build_ffmpeg_filtergraph(
    composition: Composition | Mapping[str, Any],
) -> str:
    """Return the pure ``filter_complex`` graph for *composition*.

    Input zero is always the generated background. Each supported layer adds
    one input and one overlay node, in ``z_index`` order.
    """
    value = _composition(composition)
    parts = [
        f"color=c={_ffmpeg_color(value.canvas.background)}:"
        f"s={value.canvas.width}x{value.canvas.height}:"
        f"r={value.fps}:d={_seconds(value.duration_seconds)}[base0]"
    ]
    for index, layer in enumerate(_active_layers(value)):
        node, overlay = _layer_filters(layer, value, index)
        parts.extend((node, overlay))
    parts.append(f"[base{len(_active_layers(value))}]null[vout]")
    return ";".join(parts)


def _command_inputs(composition: Composition) -> list[FFmpegInput]:
    inputs: list[FFmpegInput] = []
    for layer in _active_layers(composition):
        if layer.type is LayerType.VIDEO:
            inputs.append(_video_input(layer))
        else:
            inputs.append(_image_input(layer, composition))
    return inputs


def build_ffmpeg_command(
    composition: Composition | Mapping[str, Any],
    output: str | Path = "out.mp4",
    *,
    executable: str = "ffmpeg",
    video_codec: str = DEFAULT_VIDEO_CODEC,
    audio_codec: str = DEFAULT_AUDIO_CODEC,
    preset: str = DEFAULT_PRESET,
    crf: int = DEFAULT_CRF,
    pixel_format: str = DEFAULT_PIXEL_FORMAT,
    movflags: str | None = "+faststart",
    overwrite: bool = True,
) -> FFmpegCommand:
    """Build an ffmpeg argv for *composition* without executing anything.

    ``output`` is only stringified; it does not have to exist and its parent
    directory is never created or inspected. The returned value is a list of
    arguments that also carries the generated filtergraph and inputs.
    """
    if not isinstance(executable, str) or not executable.strip():
        raise FFmpegCommandError("executable must be a non-empty string")
    if not isinstance(video_codec, str) or not video_codec.strip():
        raise FFmpegCommandError("video_codec must be a non-empty string")
    if isinstance(crf, bool) or not isinstance(crf, int):
        raise FFmpegCommandError("crf must be an integer")
    if crf < 0:
        raise FFmpegCommandError("crf must not be negative")
    if not isinstance(preset, str) or not preset.strip():
        raise FFmpegCommandError("preset must be a non-empty string")
    if not isinstance(pixel_format, str) or not pixel_format.strip():
        raise FFmpegCommandError("pixel_format must be a non-empty string")
    if movflags is not None and not isinstance(movflags, str):
        raise FFmpegCommandError("movflags must be a string or None")

    value = _composition(composition)
    inputs = _command_inputs(value)
    graph = build_ffmpeg_filtergraph(value)
    background = _background_input(value)

    argv: list[str] = [executable.strip(), "-hide_banner", "-loglevel", "error"]
    if overwrite:
        argv.append("-y")
    argv.extend(background.argv())
    for item in inputs:
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
            "-crf",
            str(crf),
            "-pix_fmt",
            pixel_format.strip(),
            "-an",
        ]
    )
    if movflags:
        argv.extend(["-movflags", movflags])
    argv.append(str(output))
    return FFmpegCommand(
        argv,
        filtergraph=graph,
        output=str(output),
        inputs=(background, *inputs),
    )


def build_ffmpeg_args(
    composition: Composition | Mapping[str, Any],
    output: str | Path = "out.mp4",
    **options: Any,
) -> list[str]:
    """Return just the argv list for *composition*."""
    return build_ffmpeg_command(composition, output, **options).argv


def build_filtergraph(composition: Composition | Mapping[str, Any]) -> str:
    """Alias for :func:`build_ffmpeg_filtergraph`."""
    return build_ffmpeg_filtergraph(composition)


def ffmpeg_filtergraph(composition: Composition | Mapping[str, Any]) -> str:
    """Alias for :func:`build_ffmpeg_filtergraph`."""
    return build_ffmpeg_filtergraph(composition)


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
