"""Composition tools for text, asset placement, overlays, animation and audio mixing.

Each tool turns its arguments into one or more patch operations, applies the
patch to a composition and validates the result, so every tool call emits a
validated, reversible patch that can be applied, undone or stored as part of a
plan. No media is generated or rendered here; the tools only edit the
declarative composition.

The tools are bound to the assistant tool framework through
:class:`~aivia.assistant.tool_framework.Tool` and are registered with
:func:`composition_tool_registry`.
"""

from __future__ import annotations

from typing import Any

from pydantic import AliasChoices, Field

from aivia.assistant.tool_framework import (
    Tool,
    ToolArguments,
    ToolError,
    ToolResult,
    ToolValidationError,
)
from aivia.compose.io import composition_to_dict
from aivia.compose.patch import CompositionPatch, PatchOperation, apply_patch
from aivia.compose.schema import (
    AudioTrackRole,
    Composition,
    LayerType,
)

__all__ = [
    "ALL_COMPOSITION_TOOLS",
    "AnimationTool",
    "AudioMixTool",
    "CompositionTool",
    "OverlayTool",
    "PlacementTool",
    "TextTool",
    "composition_tool_registry",
]


def _layer_index(composition: Composition, layer_id: str) -> int | None:
    for index, layer in enumerate(composition.layers):
        if layer.id == layer_id:
            return index
    return None


def _track_index(composition: Composition, track_id: str) -> int | None:
    for index, track in enumerate(composition.audio_tracks):
        if track.id == track_id:
            return index
    return None


def _next_layer_id(composition: Composition, kind: str) -> str:
    existing = {layer.id for layer in composition.layers}
    candidate = 1
    while f"{kind}-{candidate}" in existing:
        candidate += 1
    return f"{kind}-{candidate}"


def _next_effect_id(composition: Composition) -> str:
    existing = {effect.id for effect in composition.effects}
    candidate = 1
    while f"effect-{candidate}" in existing:
        candidate += 1
    return f"effect-{candidate}"


def _next_track_id(composition: Composition) -> str:
    existing = {track.id for track in composition.audio_tracks}
    candidate = 1
    while f"track-{candidate}" in existing:
        candidate += 1
    return f"track-{candidate}"


class CompositionTool(Tool):
    """Base class for composition tools that emit validated patches."""

    @classmethod
    def _build_patch(
        cls, arguments: dict[str, Any], composition: Composition
    ) -> list[PatchOperation]:
        raise NotImplementedError(f"tool {cls.name!r} does not implement _build_patch")

    @classmethod
    def execute(cls, arguments: dict[str, Any]) -> ToolResult:
        raw_composition = arguments.pop("composition", None)
        if raw_composition is None:
            raise ToolError(
                f"tool {cls.name!r} requires a composition argument",
                details={"tool": cls.name},
            )
        composition = Composition.model_validate(raw_composition)
        operations = cls._build_patch(arguments, composition)
        patch = CompositionPatch.from_operations(operations)
        patched = apply_patch(composition, patch, validate=True)
        return ToolResult(
            output={
                "patch": patch.to_dict(),
                "composition": composition_to_dict(patched),
            },
            cost_usd=0.0,
            estimated=False,
        )


class _BaseArgs(ToolArguments):
    composition: dict[str, Any]


class _TextArgs(_BaseArgs):
    text: str
    layer_id: str | None = Field(
        default=None, validation_alias=AliasChoices("layer_id", "id", "name")
    )
    operation: str = Field(
        default="add",
        validation_alias=AliasChoices("operation", "op", "action"),
    )
    start_seconds: float | None = Field(
        default=None, validation_alias=AliasChoices("start_seconds", "start", "at")
    )
    duration_seconds: float | None = Field(
        default=None,
        validation_alias=AliasChoices("duration_seconds", "duration", "length"),
    )
    font_size: float | None = Field(
        default=None, validation_alias=AliasChoices("font_size", "size", "font_size_px")
    )
    color: str | None = Field(
        default=None, validation_alias=AliasChoices("color", "fill", "text_color")
    )
    font_family: str | None = Field(
        default=None, validation_alias=AliasChoices("font_family", "font", "family")
    )
    align: str | None = Field(
        default=None, validation_alias=AliasChoices("align", "text_align", "alignment")
    )
    language: str | None = Field(
        default=None, validation_alias=AliasChoices("language", "lang", "locale")
    )
    direction: str | None = Field(
        default=None,
        validation_alias=AliasChoices("direction", "text_direction", "dir"),
    )
    x: int | None = Field(default=None, validation_alias=AliasChoices("x", "left"))
    y: int | None = Field(default=None, validation_alias=AliasChoices("y", "top"))
    width: int | None = Field(default=None, validation_alias=AliasChoices("width", "w"))
    height: int | None = Field(
        default=None, validation_alias=AliasChoices("height", "h")
    )


class TextTool(CompositionTool):
    """Add or edit a text layer on the composition canvas."""

    name = "composition.text"
    description = "Add or edit a text layer on the composition canvas."
    arguments_model = _TextArgs

    @classmethod
    def _build_patch(
        cls, arguments: dict[str, Any], composition: Composition
    ) -> list[PatchOperation]:
        text = arguments.pop("text")
        layer_id = arguments.pop("layer_id", None)
        operation = arguments.pop("operation", "add")
        layer: dict[str, Any] = {"type": LayerType.TEXT, "text": text}
        if layer_id is None:
            layer_id = _next_layer_id(composition, "text")
        layer["id"] = layer_id
        for field in (
            "start_seconds",
            "duration_seconds",
            "font_size",
            "color",
            "font_family",
            "align",
            "language",
            "direction",
            "x",
            "y",
            "width",
            "height",
        ):
            value = arguments.pop(field, None)
            if value is not None:
                layer[field] = value
        index = _layer_index(composition, layer_id)
        if index is None or operation == "add":
            return [PatchOperation(op="add", path="/layers/-", value=layer)]
        return [PatchOperation(op="replace", path=f"/layers/{index}", value=layer)]


class _PlacementArgs(_BaseArgs):
    source: str
    type: str = Field(
        default="image", validation_alias=AliasChoices("type", "kind", "layer_type")
    )
    layer_id: str | None = Field(
        default=None, validation_alias=AliasChoices("layer_id", "id", "name")
    )
    operation: str = Field(
        default="add",
        validation_alias=AliasChoices("operation", "op", "action"),
    )
    start_seconds: float | None = Field(
        default=None, validation_alias=AliasChoices("start_seconds", "start", "at")
    )
    duration_seconds: float | None = Field(
        default=None,
        validation_alias=AliasChoices("duration_seconds", "duration", "length"),
    )
    x: int | None = Field(default=None, validation_alias=AliasChoices("x", "left"))
    y: int | None = Field(default=None, validation_alias=AliasChoices("y", "top"))
    width: int | None = Field(default=None, validation_alias=AliasChoices("width", "w"))
    height: int | None = Field(
        default=None, validation_alias=AliasChoices("height", "h")
    )
    anchor: str | None = Field(
        default=None, validation_alias=AliasChoices("anchor", "corner", "alignment")
    )
    scale: float | None = Field(
        default=None, validation_alias=AliasChoices("scale", "zoom")
    )
    fit: str | None = Field(
        default=None, validation_alias=AliasChoices("fit", "fit_mode", "mode")
    )


class PlacementTool(CompositionTool):
    """Place an asset as an image, video, sticker or logo layer."""

    name = "composition.place"
    description = "Place an asset as an image, video, sticker or logo layer."
    arguments_model = _PlacementArgs

    @classmethod
    def _build_patch(
        cls, arguments: dict[str, Any], composition: Composition
    ) -> list[PatchOperation]:
        source = arguments.pop("source")
        layer_type = arguments.pop("type", "image")
        try:
            LayerType(layer_type)
        except ValueError as exc:
            raise ToolValidationError(
                f"unsupported layer type {layer_type!r}",
                details={"tool": cls.name, "type": layer_type},
            ) from exc
        layer_id = arguments.pop("layer_id", None)
        operation = arguments.pop("operation", "add")
        layer: dict[str, Any] = {"type": layer_type, "source": source}
        if layer_id is None:
            layer_id = _next_layer_id(composition, layer_type)
        layer["id"] = layer_id
        for field in (
            "start_seconds",
            "duration_seconds",
            "x",
            "y",
            "width",
            "height",
            "anchor",
            "scale",
            "fit",
        ):
            value = arguments.pop(field, None)
            if value is not None:
                layer[field] = value
        index = _layer_index(composition, layer_id)
        if index is None or operation == "add":
            return [PatchOperation(op="add", path="/layers/-", value=layer)]
        return [PatchOperation(op="replace", path=f"/layers/{index}", value=layer)]


class _OverlayArgs(_BaseArgs):
    type: str = Field(
        validation_alias=AliasChoices("type", "kind", "preset", "overlay", "style")
    )
    effect_id: str | None = Field(
        default=None, validation_alias=AliasChoices("effect_id", "id", "name")
    )
    start_seconds: float | None = Field(
        default=None, validation_alias=AliasChoices("start_seconds", "start", "at")
    )
    end_seconds: float | None = Field(
        default=None, validation_alias=AliasChoices("end_seconds", "end", "stop")
    )
    duration_seconds: float | None = Field(
        default=None,
        validation_alias=AliasChoices("duration_seconds", "duration", "length"),
    )
    seed: int | None = Field(
        default=None, validation_alias=AliasChoices("seed", "random_seed", "rng")
    )
    count: int | None = Field(
        default=None, validation_alias=AliasChoices("count", "particles", "amount")
    )
    colors: tuple[str, ...] | None = Field(
        default=None,
        validation_alias=AliasChoices("colors", "colours", "colour", "palette"),
    )
    min_size: float | None = Field(
        default=None, validation_alias=AliasChoices("min_size", "size_min", "smallest")
    )
    max_size: float | None = Field(
        default=None, validation_alias=AliasChoices("max_size", "size_max", "largest")
    )
    speed: float | None = Field(
        default=None, validation_alias=AliasChoices("speed", "velocity", "motion")
    )
    wind: float | None = Field(
        default=None, validation_alias=AliasChoices("wind", "drift", "horizontal_drift")
    )
    opacity: float | None = Field(
        default=None, validation_alias=AliasChoices("opacity", "alpha")
    )
    enabled: bool = Field(
        default=True, validation_alias=AliasChoices("enabled", "visible")
    )


class OverlayTool(CompositionTool):
    """Add a procedural overlay effect (confetti, sparkles, snow or bokeh)."""

    name = "composition.overlay"
    description = "Add a procedural overlay effect (confetti, sparkles, snow or bokeh)."
    arguments_model = _OverlayArgs

    @classmethod
    def _build_patch(
        cls, arguments: dict[str, Any], composition: Composition
    ) -> list[PatchOperation]:
        overlay_type = arguments.pop("type")
        effect_id = arguments.pop("effect_id", None)
        if effect_id is None:
            effect_id = _next_effect_id(composition)
        effect: dict[str, Any] = {
            "id": effect_id,
            "type": overlay_type,
            "enabled": arguments.pop("enabled", True),
        }
        for field in (
            "start_seconds",
            "end_seconds",
            "duration_seconds",
            "seed",
            "count",
            "colors",
            "min_size",
            "max_size",
            "speed",
            "wind",
            "opacity",
        ):
            value = arguments.pop(field, None)
            if value is not None:
                effect[field] = value
        return [PatchOperation(op="add", path="/effects/-", value=effect)]


class _AnimationArgs(_BaseArgs):
    layer_id: str = Field(validation_alias=AliasChoices("layer_id", "id", "name"))
    kind: str = Field(default="entrance", validation_alias=AliasChoices("kind", "side"))
    preset: str = Field(
        default="fade",
        validation_alias=AliasChoices("preset", "type", "kind", "animation"),
    )
    duration: float | None = Field(
        default=None,
        validation_alias=AliasChoices("duration", "duration_seconds", "length"),
    )
    easing: str | None = Field(
        default=None, validation_alias=AliasChoices("easing", "ease", "curve", "timing")
    )
    distance: float | None = Field(
        default=None,
        validation_alias=AliasChoices("distance", "distance_px", "offset_px"),
    )
    offset: tuple[float, float] | None = Field(
        default=None, validation_alias=AliasChoices("offset", "translate", "movement")
    )
    start_scale: float | None = Field(
        default=None,
        validation_alias=AliasChoices("start_scale", "from_scale", "scale_from"),
    )
    end_scale: float | None = Field(
        default=None, validation_alias=AliasChoices("end_scale", "to_scale", "scale_to")
    )
    opacity_from: float | None = Field(
        default=None,
        validation_alias=AliasChoices("opacity_from", "from_opacity", "start_opacity"),
    )
    opacity_to: float | None = Field(
        default=None,
        validation_alias=AliasChoices("opacity_to", "to_opacity", "end_opacity"),
    )
    delay: float | None = Field(
        default=None,
        validation_alias=AliasChoices("delay", "start_delay", "delay_seconds"),
    )


class AnimationTool(CompositionTool):
    """Add an entrance or exit animation to a text layer."""

    name = "composition.animate"
    description = "Add an entrance or exit animation to a text layer."
    arguments_model = _AnimationArgs

    @classmethod
    def _build_patch(
        cls, arguments: dict[str, Any], composition: Composition
    ) -> list[PatchOperation]:
        layer_id = arguments.pop("layer_id")
        kind = arguments.pop("kind", "entrance")
        preset = arguments.pop("preset", "fade")
        index = _layer_index(composition, layer_id)
        if index is None:
            raise ToolValidationError(
                f"layer {layer_id!r} not found in composition",
                details={"tool": cls.name, "layer_id": layer_id},
            )
        layer = composition.layers[index]
        if layer.type != LayerType.TEXT:
            raise ToolValidationError(
                f"layer {layer_id!r} is not a text layer",
                details={
                    "tool": cls.name,
                    "layer_id": layer_id,
                    "type": layer.type.value,
                },
            )
        animation: dict[str, Any] = {"type": preset}
        for field in (
            "duration",
            "easing",
            "distance",
            "offset",
            "start_scale",
            "end_scale",
            "opacity_from",
            "opacity_to",
            "delay",
        ):
            value = arguments.pop(field, None)
            if value is not None:
                animation[field] = value
        target = "entrance" if kind in {"entrance", "enter", "in"} else "exit"
        return [
            PatchOperation(op="set", path=f"/layers/{index}/{target}", value=animation)
        ]


class _AudioArgs(_BaseArgs):
    source: str
    track_id: str | None = Field(
        default=None, validation_alias=AliasChoices("track_id", "id", "name")
    )
    role: str = Field(
        default="music", validation_alias=AliasChoices("role", "kind", "type")
    )
    operation: str = Field(
        default="add",
        validation_alias=AliasChoices("operation", "op", "action"),
    )
    start_seconds: float | None = Field(
        default=None, validation_alias=AliasChoices("start_seconds", "start", "offset")
    )
    duration_seconds: float | None = Field(
        default=None,
        validation_alias=AliasChoices("duration_seconds", "duration", "length"),
    )
    volume_db: float | None = Field(
        default=None,
        validation_alias=AliasChoices("volume_db", "volume", "gain_db", "gain"),
    )
    fade_in_seconds: float | None = Field(
        default=None, validation_alias=AliasChoices("fade_in_seconds", "fade_in")
    )
    fade_out_seconds: float | None = Field(
        default=None, validation_alias=AliasChoices("fade_out_seconds", "fade_out")
    )
    loop: bool | None = Field(
        default=None, validation_alias=AliasChoices("loop", "looped")
    )
    duck: bool | None = Field(
        default=None, validation_alias=AliasChoices("duck", "ducked", "allow_ducking")
    )
    enabled: bool | None = Field(
        default=None, validation_alias=AliasChoices("enabled", "visible")
    )


class AudioMixTool(CompositionTool):
    """Add or edit an audio track in the composition."""

    name = "composition.audio"
    description = "Add or edit an audio track in the composition."
    arguments_model = _AudioArgs

    @classmethod
    def _build_patch(
        cls, arguments: dict[str, Any], composition: Composition
    ) -> list[PatchOperation]:
        source = arguments.pop("source")
        role = arguments.pop("role", "music")
        try:
            AudioTrackRole(role)
        except ValueError as exc:
            raise ToolValidationError(
                f"unsupported audio role {role!r}",
                details={"tool": cls.name, "role": role},
            ) from exc
        track_id = arguments.pop("track_id", None)
        operation = arguments.pop("operation", "add")
        track: dict[str, Any] = {"source": source, "role": role}
        if track_id is None:
            track_id = _next_track_id(composition)
        track["id"] = track_id
        for field in (
            "start_seconds",
            "duration_seconds",
            "volume_db",
            "fade_in_seconds",
            "fade_out_seconds",
            "loop",
            "duck",
            "enabled",
        ):
            value = arguments.pop(field, None)
            if value is not None:
                track[field] = value
        index = _track_index(composition, track_id)
        if index is None or operation == "add":
            return [PatchOperation(op="add", path="/audio_tracks/-", value=track)]
        return [
            PatchOperation(op="replace", path=f"/audio_tracks/{index}", value=track)
        ]


ALL_COMPOSITION_TOOLS: tuple[type[CompositionTool], ...] = (
    TextTool,
    PlacementTool,
    OverlayTool,
    AnimationTool,
    AudioMixTool,
)


def composition_tool_registry() -> Any:
    """Return a registry with every composition tool registered."""
    from aivia.assistant.tool_framework import ToolRegistry

    registry = ToolRegistry()
    for tool in ALL_COMPOSITION_TOOLS:
        registry.register(tool)
    return registry
