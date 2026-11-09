"""Assemble tool outputs and patches into a single composition rendered by the
shared renderer.

Generation tool outputs (``MediaResult``) become registered assets and
placement layers or audio tracks, composition tool patches are applied on top,
and the assembled composition is rendered with
:func:`aivia.compose.render_still`.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from PIL import Image

from aivia.assistant import Assembly
from aivia.assistant.tool_framework import ToolResult
from aivia.compose import render_still
from aivia.compose.io import composition_to_dict
from aivia.compose.patch import CompositionPatch, PatchOperation, apply_patch
from aivia.media import MediaResult


def _write_png(path: Path, size: tuple[int, int] = (64, 64)) -> Path:
    image = Image.new("RGBA", size, (255, 0, 0, 255))
    image.save(path, format="PNG")
    return path


def _media_result(kind: str, path: Path) -> MediaResult:
    outcome = SimpleNamespace(paths=[str(path)], assets=[])
    return MediaResult(kind=kind, outcome=cast(Any, outcome))


def _base_composition() -> dict[str, Any]:
    return {
        "canvas": {"width": 1280, "height": 720, "background": "#000000"},
        "duration_seconds": 5.0,
        "layers": [],
        "audio_tracks": [],
        "effects": [],
    }


def _text_patch() -> CompositionPatch:
    return CompositionPatch.from_operations(
        [
            PatchOperation(
                op="add",
                path="/layers/-",
                value={
                    "id": "title",
                    "type": "text",
                    "text": "Hello world",
                    "font_size": 40,
                },
            )
        ]
    )


def test_assembly_has_default_composition() -> None:
    assembly = Assembly()

    assert assembly.composition.width == 1280
    assert assembly.composition.height == 720
    assert assembly.composition.duration_seconds == 5.0
    assert assembly.composition.layers == []
    assert assembly.assets == {}
    assert assembly.outputs == []
    assert assembly.patches == []


def test_assembly_accepts_explicit_composition() -> None:
    assembly = Assembly(_base_composition())

    assert assembly.composition.width == 1280
    assert assembly.composition.height == 720


def test_assembly_records_outputs_and_patches() -> None:
    assembly = Assembly()
    output = ToolResult(output={"text": "done"}, cost_usd=0.0)
    patch = _text_patch()

    assembly.add_output("caption", output)
    assembly.add_patch(patch)

    assert assembly.outputs == [("caption", output)]
    assert len(assembly.patches) == 1


def test_assembly_image_output_adds_layer_and_registers_asset(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "photo.png"
    _write_png(image_path)

    assembly = Assembly()
    assembly.add_output("image.generate", _media_result("image", image_path))

    composition = assembly.assemble()

    assert len(composition.layers) == 1
    layers = composition_to_dict(composition)["layers"]
    assert layers[0]["type"] == "image"
    source = layers[0]["source"]
    assert source in assembly.assets
    assert assembly.assets[source] == str(image_path)


def test_assembly_audio_output_adds_track(tmp_path: Path) -> None:
    audio_path = tmp_path / "voice.wav"
    audio_path.write_bytes(b"RIFF....WAVE....")

    assembly = Assembly()
    assembly.add_output("voice.generate", _media_result("voice", audio_path))

    composition = assembly.assemble()

    assert len(composition.audio_tracks) == 1
    track = composition.audio_tracks[0]
    assert track.source == str(audio_path)
    assert track.role == "voice"


def test_assembly_patch_applies_to_composition() -> None:
    assembly = Assembly()
    assembly.add_patch(_text_patch())

    composition = assembly.assemble()

    assert len(composition.layers) == 1
    layers = composition_to_dict(composition)["layers"]
    assert layers[0]["type"] == "text"
    assert layers[0]["text"] == "Hello world"


def test_assembly_combines_output_and_patch(tmp_path: Path) -> None:
    image_path = tmp_path / "photo.png"
    _write_png(image_path)

    assembly = Assembly()
    assembly.add_output("image.generate", _media_result("image", image_path))
    assembly.add_patch(_text_patch())

    composition = assembly.assemble()

    layers = composition_to_dict(composition)["layers"]
    assert len(layers) == 2
    assert {layer["type"] for layer in layers} == {"image", "text"}
    assert layers[0]["source"] in assembly.assets


def test_assembly_render_returns_canvas_sized_image(tmp_path: Path) -> None:
    image_path = tmp_path / "photo.png"
    _write_png(image_path)

    assembly = Assembly()
    assembly.add_output("image.generate", _media_result("image", image_path))
    assembly.add_patch(_text_patch())

    image = assembly.render()

    assert isinstance(image, Image.Image)
    assert image.size == (1280, 720)


def test_assembly_render_uses_shared_renderer(tmp_path: Path) -> None:
    image_path = tmp_path / "photo.png"
    _write_png(image_path)

    assembly = Assembly()
    assembly.add_output("image.generate", _media_result("image", image_path))

    direct = render_still(assembly.assemble(), assets=assembly.assets)
    assembled = assembly.render()

    assert direct.size == assembled.size


def test_assembly_tool_result_patch_is_merged() -> None:
    patched = apply_patch(_base_composition(), _text_patch(), validate=True)
    result = ToolResult(
        output={"composition": patched.model_dump()},
        cost_usd=0.0,
    )

    assembly = Assembly()
    assembly.add_output("composition.text", result)

    composition = assembly.assemble()
    assert len(composition.layers) == 1
    assert composition_to_dict(composition)["layers"][0]["text"] == "Hello world"
