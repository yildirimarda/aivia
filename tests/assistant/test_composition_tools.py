"""Composition tools for text, asset placement, overlays, animation and audio mixing.

Each tool is exercised through the
:class:`~aivia.assistant.tool_framework.ToolFramework`, so arguments are
validated against the JSON schema, the allowlist is enforced and every call
emits a validated, reversible patch that can be applied, undone or stored as
part of a plan. No media is generated or rendered here.
"""

from __future__ import annotations

from typing import Any

import pytest

from aivia.assistant import (
    ToolFramework,
    ToolNotAllowedError,
    ToolUnknownError,
    ToolValidationError,
)
from aivia.assistant.composition_tools import (
    ALL_COMPOSITION_TOOLS,
    composition_tool_registry,
)
from aivia.compose import Composition, composition_hash
from aivia.compose.patch import CompositionPatch, apply_patch_reversible
from aivia.core.config.planner import PlannerConfig


def _composition() -> dict[str, Any]:
    return {
        "canvas": {"width": 1280, "height": 720},
        "duration_seconds": 5.0,
        "layers": [],
        "audio_tracks": [],
        "effects": [],
    }


def _framework() -> ToolFramework:
    return ToolFramework(composition_tool_registry())


def test_all_composition_tools_are_registered() -> None:
    registry = composition_tool_registry()

    assert registry.names() == (
        "composition.animate",
        "composition.audio",
        "composition.overlay",
        "composition.place",
        "composition.text",
    )
    for tool in ALL_COMPOSITION_TOOLS:
        assert tool.name in registry
        schema = registry.schemas()[tool.name]
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert "composition" in schema["required"]


def test_text_tool_schema_requires_text_and_composition() -> None:
    schema = composition_tool_registry().schemas()["composition.text"]

    assert schema["required"] == ["composition", "text"]
    assert schema["properties"]["text"]["type"] == "string"


def test_text_tool_adds_a_text_layer() -> None:
    result = _framework().execute(
        "composition.text",
        {
            "composition": _composition(),
            "text": "Hello world",
            "font_size": 40,
            "color": "#ff0000",
        },
    )

    output = result.output
    assert "patch" in output
    assert "composition" in output
    composition = output["composition"]
    assert len(composition["layers"]) == 1
    layer = composition["layers"][0]
    assert layer["type"] == "text"
    assert layer["text"] == "Hello world"
    assert layer["font_size"] == 40
    assert layer["color"] == "#ff0000"


def test_text_tool_patch_is_validated_and_reversible() -> None:
    result = _framework().execute(
        "composition.text",
        {"composition": _composition(), "text": "Hello world"},
    )

    patch = CompositionPatch.from_dict(result.output["patch"])
    original = Composition.model_validate(_composition())
    applied = apply_patch_reversible(original, patch)
    assert applied.patched_hash != applied.original_hash
    restored = applied.undo()
    assert composition_hash(restored) == applied.original_hash


def test_text_tool_edits_existing_layer() -> None:
    composition = _composition()
    composition["layers"] = [
        {"id": "title", "type": "text", "text": "Old", "font_size": 30}
    ]
    result = _framework().execute(
        "composition.text",
        {
            "composition": composition,
            "layer_id": "title",
            "text": "New",
            "operation": "set",
            "font_size": 50,
        },
    )

    layers = result.output["composition"]["layers"]
    assert len(layers) == 1
    assert layers[0]["id"] == "title"
    assert layers[0]["text"] == "New"
    assert layers[0]["font_size"] == 50


def test_placement_tool_adds_an_image_layer() -> None:
    result = _framework().execute(
        "composition.place",
        {
            "composition": _composition(),
            "source": "assets/photo.png",
            "type": "image",
            "x": 100,
            "y": 200,
            "scale": 1.5,
        },
    )

    layers = result.output["composition"]["layers"]
    assert len(layers) == 1
    layer = layers[0]
    assert layer["type"] == "image"
    assert layer["source"] == "assets/photo.png"
    assert layer["x"] == 100
    assert layer["y"] == 200
    assert layer["scale"] == 1.5


def test_placement_tool_rejects_unknown_layer_type() -> None:
    with pytest.raises(ToolValidationError):
        _framework().execute(
            "composition.place",
            {"composition": _composition(), "source": "x.png", "type": "notatype"},
        )


def test_overlay_tool_adds_an_overlay_effect() -> None:
    result = _framework().execute(
        "composition.overlay",
        {
            "composition": _composition(),
            "type": "confetti",
            "seed": 7,
            "opacity": 0.8,
        },
    )

    effects = result.output["composition"]["effects"]
    assert len(effects) == 1
    effect = effects[0]
    assert effect["type"] == "confetti"
    assert effect["seed"] == 7
    assert effect["opacity"] == 0.8
    assert effect["enabled"] is True


def test_animation_tool_adds_an_entrance_to_a_text_layer() -> None:
    composition = _composition()
    composition["layers"] = [{"id": "title", "type": "text", "text": "Hello"}]
    result = _framework().execute(
        "composition.animate",
        {
            "composition": composition,
            "layer_id": "title",
            "kind": "entrance",
            "preset": "fade",
            "duration": 0.5,
        },
    )

    layers = result.output["composition"]["layers"]
    assert layers[0]["entrance"]["type"] == "fade"
    assert layers[0]["entrance"]["duration"] == 0.5


def test_animation_tool_rejects_non_text_layer() -> None:
    composition = _composition()
    composition["layers"] = [{"id": "img", "type": "image", "source": "x.png"}]
    with pytest.raises(ToolValidationError):
        _framework().execute(
            "composition.animate",
            {"composition": composition, "layer_id": "img", "preset": "fade"},
        )


def test_animation_tool_rejects_unknown_layer() -> None:
    with pytest.raises(ToolValidationError):
        _framework().execute(
            "composition.animate",
            {"composition": _composition(), "layer_id": "missing", "preset": "fade"},
        )


def test_audio_mix_tool_adds_a_track() -> None:
    result = _framework().execute(
        "composition.audio",
        {
            "composition": _composition(),
            "source": "assets/voice.wav",
            "role": "voice",
            "volume_db": -6.0,
        },
    )

    tracks = result.output["composition"]["audio_tracks"]
    assert len(tracks) == 1
    track = tracks[0]
    assert track["source"] == "assets/voice.wav"
    assert track["role"] == "voice"
    assert track["volume_db"] == -6.0


def test_audio_mix_tool_edits_existing_track() -> None:
    composition = _composition()
    composition["audio_tracks"] = [
        {"id": "music", "source": "music.mp3", "role": "music", "volume_db": 0.0}
    ]
    result = _framework().execute(
        "composition.audio",
        {
            "composition": composition,
            "track_id": "music",
            "source": "music.mp3",
            "operation": "set",
            "volume_db": -12.0,
        },
    )

    tracks = result.output["composition"]["audio_tracks"]
    assert len(tracks) == 1
    assert tracks[0]["id"] == "music"
    assert tracks[0]["volume_db"] == -12.0


def test_audio_mix_tool_rejects_unknown_role() -> None:
    with pytest.raises(ToolValidationError):
        _framework().execute(
            "composition.audio",
            {"composition": _composition(), "source": "x.wav", "role": "notarole"},
        )


def test_invalid_patch_is_rejected() -> None:
    with pytest.raises(ToolValidationError):
        _framework().execute(
            "composition.text",
            {"composition": _composition()},
        )


def test_unknown_tool_is_rejected() -> None:
    framework = _framework()

    with pytest.raises(ToolUnknownError):
        framework.execute("composition.nonexistent", {"composition": _composition()})

    assert framework.tool_calls == 0


def test_allowlist_restricts_composition_tools() -> None:
    registry = composition_tool_registry()
    config = PlannerConfig(allowed_tools=["composition.text"])
    framework = ToolFramework(registry, planner_config=config)

    assert framework.permits("composition.text") is True
    assert framework.permits("composition.audio") is False

    result = framework.execute(
        "composition.text",
        {"composition": _composition(), "text": "Hi"},
    )
    assert len(result.output["composition"]["layers"]) == 1

    with pytest.raises(ToolNotAllowedError):
        framework.execute(
            "composition.audio",
            {"composition": _composition(), "source": "x.wav"},
        )
    assert framework.tool_calls == 1
