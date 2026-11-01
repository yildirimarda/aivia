"""The ffmpeg command builder is pure and never executes a binary."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from aivia.compose import (
    DEFAULT_AUDIO_CODEC,
    DEFAULT_CRF,
    DEFAULT_PIXEL_FORMAT,
    DEFAULT_PRESET,
    DEFAULT_VIDEO_CODEC,
    FFmpegCommand,
    FFmpegCommandError,
    FFmpegInput,
    build_ffmpeg_args,
    build_ffmpeg_command,
    build_ffmpeg_filtergraph,
    build_filtergraph,
    ffmpeg_args,
    ffmpeg_command,
    ffmpeg_filtergraph,
)


def _composition(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "canvas": {"width": 320, "height": 240, "background": "#000000"},
        "duration_seconds": 4.0,
        "fps": 24,
        "layers": [],
    }
    values.update(overrides)
    return values


def _image_layer(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "id": "image",
        "type": "image",
        "source": "missing.png",
    }
    values.update(overrides)
    return values


def _video_layer(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "id": "clip",
        "type": "video",
        "source": "missing.mp4",
    }
    values.update(overrides)
    return values


def _option(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def test_building_a_command_is_pure_and_returns_an_argument_list(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the command builder must not execute ffmpeg")

    monkeypatch.setattr(subprocess, "run", fail)
    monkeypatch.setattr(subprocess, "Popen", fail)

    command = build_ffmpeg_command(
        _composition(layers=[_image_layer(source="does-not-exist.png")]),
        tmp_path / "never-created.mp4",
    )

    assert isinstance(command, FFmpegCommand)
    assert isinstance(command, list)
    assert command.executable == "ffmpeg"
    assert command.argv == list(command)
    assert command.args == list(command)
    assert command.filter_complex == command.filtergraph
    assert command.output == str(tmp_path / "never-created.mp4")
    assert not (tmp_path / "never-created.mp4").exists()
    assert command[-1] == str(tmp_path / "never-created.mp4")
    assert command.inputs[0].source.startswith("color=c=")
    assert command.inputs[1].source == "does-not-exist.png"
    assert command.to_dict()["output"] == command.output


def test_the_background_and_output_options_are_deterministic() -> None:
    composition = _composition(
        canvas={"width": 640, "height": 360, "background": "#112233"},
        duration_seconds=2.5,
        fps=25,
    )

    command = build_ffmpeg_command(composition, "out.mp4")

    assert "color=c=0x112233:s=640x360:r=25:d=2.500000[base0]" in command.filtergraph
    assert command.filtergraph.endswith("[base0]null[vout]")
    assert _option(command, "-filter_complex") == command.filtergraph
    assert _option(command, "-map") == "[vout]"
    assert _option(command, "-t") == "2.500000"
    assert _option(command, "-r") == "25"
    assert _option(command, "-c:v") == DEFAULT_VIDEO_CODEC == "libx264"
    assert _option(command, "-preset") == DEFAULT_PRESET
    assert _option(command, "-crf") == str(DEFAULT_CRF)
    assert _option(command, "-pix_fmt") == DEFAULT_PIXEL_FORMAT
    assert _option(command, "-movflags") == "+faststart"
    assert "-an" in command
    assert _option(command, "-f") == "lavfi"


def test_image_layers_are_scaled_overlaid_and_limited_to_their_window() -> None:
    composition = _composition(
        layers=[
            _image_layer(id="back", source="back.png", z=0),
            _image_layer(
                id="front",
                source="front.png",
                z=1,
                x=10,
                y=20,
                width=100,
                height=50,
                fit="stretch",
                opacity=0.5,
                start=1.0,
                end=3.0,
            ),
        ]
    )

    command = build_ffmpeg_command(composition, "out.mp4")

    graph = command.filtergraph
    assert "[1:v]" in graph
    assert "[2:v]" in graph
    assert "scale=320:240:force_original_aspect_ratio=increase,crop=320:240" in graph
    assert "scale=100:50" in graph
    assert "colorchannelmixer=aa=0.500000" in graph
    assert "overlay=x=10:y=20" in graph
    assert "enable='between(t,1.000000,3.000000)'" in graph
    assert graph.index("[1:v]") < graph.index("[2:v]")
    assert graph.index("[base1]") < graph.index("[base2]")
    assert "trim=" not in graph

    first_input = command.inputs[1]
    assert isinstance(first_input, FFmpegInput)
    assert first_input.options == ("-loop", "1", "-t", "4.000000")
    assert command.inputs[2].options == ("-loop", "1", "-t", "2.000000")


def test_video_layers_are_trimmed_scaled_and_rate_adjusted() -> None:
    composition = _composition(
        layers=[
            _video_layer(
                trim_start=1.0,
                trim_end=4.0,
                playback_rate=2.0,
                loop=True,
                fit="cover",
                width=160,
                height=90,
            )
        ]
    )

    command = build_ffmpeg_command(composition, "out.mp4")
    graph = command.filtergraph

    assert command.inputs[1].options == ("-stream_loop", "-1")
    assert "trim=start=1.000000:end=4.000000" in graph
    assert "setpts=(PTS-STARTPTS)/2" in graph
    assert "fps=24" in graph
    assert "scale=160:90:force_original_aspect_ratio=increase,crop=160:90" in graph
    assert "overlay=x=0:y=0" in graph


def test_a_looped_video_is_trimmed_to_the_visible_layer_duration() -> None:
    composition = _composition(
        duration_seconds=6.0,
        layers=[
            _video_layer(
                loop=True,
                trim_start=2.0,
                playback_rate=0.5,
                start=1.0,
                end=5.0,
            )
        ],
    )

    graph = build_ffmpeg_filtergraph(composition)

    assert "trim=start=2.000000:end=4.000000" in graph
    assert "setpts=(PTS-STARTPTS)/0.5" in graph
    assert "enable='between(t,1.000000,5.000000)'" in graph


@pytest.mark.parametrize(
    ("anchor", "expected_x", "expected_y"),
    [
        ("top_left", "12", "8"),
        ("top_center", "(W-w)/2", "8"),
        ("top_right", "W-w-12", "8"),
        ("middle_left", "12", "(H-h)/2"),
        ("center", "(W-w)/2", "(H-h)/2"),
        ("bottom_right", "W-w-12", "H-h-8"),
    ],
)
def test_overlay_anchors_become_ffmpeg_position_expressions(
    anchor: str,
    expected_x: str,
    expected_y: str,
) -> None:
    composition = _composition(
        layers=[
            {
                "id": "sticker",
                "type": "sticker",
                "source": "sticker.png",
                "anchor": anchor,
                "margin": (12, 8),
                "width": 80,
                "scale": 2.0,
            }
        ]
    )

    graph = build_ffmpeg_filtergraph(composition)

    assert "scale=160:-1" in graph
    assert f"overlay=x={expected_x}:y={expected_y}" in graph


def test_stickers_and_logos_without_a_size_keep_their_natural_size() -> None:
    composition = _composition(
        layers=[
            {
                "id": "logo",
                "type": "logo",
                "source": "logo.svg",
                "anchor": "bottom_left",
                "scale": 0.5,
            }
        ]
    )

    graph = build_ffmpeg_filtergraph(composition)

    assert "scale=iw*0.5:ih*0.5" in graph
    assert "overlay=x=0:y=H-h" in graph


@pytest.mark.parametrize(
    ("fit", "expected"),
    [
        ("stretch", "scale=320:240"),
        (
            "contain",
            "pad=320:240:(ow-iw)/2:(oh-ih)/2:color=black@0",
        ),
        (
            "cover",
            "scale=320:240:force_original_aspect_ratio=increase,crop=320:240",
        ),
    ],
)
def test_image_fit_modes_map_to_filter_chains(fit: str, expected: str) -> None:
    composition = _composition(layers=[_image_layer(width=320, height=240, fit=fit)])

    assert expected in build_ffmpeg_filtergraph(composition)


def test_no_fit_leaves_the_source_geometry_untouched() -> None:
    composition = _composition(layers=[_image_layer(fit="none")])

    graph = build_ffmpeg_filtergraph(composition)

    assert "scale=320:240" not in graph
    assert "overlay=x=0:y=0" in graph


def test_text_layers_become_drawtext_filters() -> None:
    graph = build_ffmpeg_filtergraph(
        _composition(
            layers=[
                {
                    "id": "title",
                    "type": "text",
                    "text": "hi",
                    "x": 20,
                    "y": 30,
                }
            ]
        )
    )

    assert "drawtext=" in graph
    assert "text='hi'" in graph
    assert "x=20" in graph
    assert "y=30" in graph
    assert "overlay=x=0:y=0" in graph


def test_shape_layers_are_reported_instead_of_silently_dropped() -> None:
    with pytest.raises(FFmpegCommandError):
        build_ffmpeg_filtergraph(
            _composition(
                layers=[
                    {
                        "id": "box",
                        "type": "shape",
                        "shape": "rectangle",
                    }
                ]
            )
        )


def test_zero_length_and_invisible_layers_are_skipped() -> None:
    composition = _composition(
        layers=[
            _image_layer(id="hidden", visible=False),
            _image_layer(id="past", start=4.0),
            _image_layer(id="shown", start=0.0, end=1.0),
        ]
    )

    command = build_ffmpeg_command(composition, "out.mp4")

    assert [item.layer_id for item in command.inputs[1:]] == ["shown"]
    assert command.filtergraph.count("overlay=") == 1


def test_the_public_aliases_return_the_same_command() -> None:
    composition = _composition(layers=[_image_layer()])

    command = build_ffmpeg_command(composition, "out.mp4")
    assert ffmpeg_command(composition, "out.mp4") == command
    assert build_ffmpeg_args(composition, "out.mp4") == command.argv
    assert ffmpeg_args(composition, "out.mp4") == command.argv
    assert build_filtergraph(composition) == command.filtergraph
    assert ffmpeg_filtergraph(composition) == command.filtergraph
    assert build_ffmpeg_filtergraph(composition) == command.filtergraph


def test_command_options_are_validated() -> None:
    composition = _composition()

    with pytest.raises(FFmpegCommandError):
        build_ffmpeg_command(composition, executable="")
    with pytest.raises(FFmpegCommandError):
        build_ffmpeg_command(composition, video_codec="")
    with pytest.raises(FFmpegCommandError):
        build_ffmpeg_command(composition, preset="")
    with pytest.raises(FFmpegCommandError):
        build_ffmpeg_command(composition, pixel_format="")
    with pytest.raises(FFmpegCommandError):
        build_ffmpeg_command(composition, movflags=1)  # type: ignore[arg-type]
    with pytest.raises(FFmpegCommandError):
        build_ffmpeg_command(composition, crf=True)
    with pytest.raises(FFmpegCommandError):
        build_ffmpeg_command(composition, crf=-1)


def test_custom_output_options_replace_the_defaults() -> None:
    command = build_ffmpeg_command(
        _composition(),
        "out.webm",
        executable="/opt/ffmpeg",
        video_codec="libvpx-vp9",
        preset="slow",
        crf=30,
        pixel_format="yuv420p10le",
        movflags=None,
        overwrite=False,
    )

    assert command.executable == "/opt/ffmpeg"
    assert "-y" not in command
    assert _option(command, "-c:v") == "libvpx-vp9"
    assert _option(command, "-preset") == "slow"
    assert _option(command, "-crf") == "30"
    assert _option(command, "-pix_fmt") == "yuv420p10le"
    assert "-movflags" not in command
    assert DEFAULT_AUDIO_CODEC == "aac"


def test_a_mapping_is_accepted_like_a_composition() -> None:
    mapping = _composition(layers=[_image_layer()])

    from_mapping = build_ffmpeg_command(mapping, "out.mp4")
    from_model = build_ffmpeg_command(
        _composition(layers=[_image_layer()]),
        "out.mp4",
    )

    assert from_mapping == from_model
