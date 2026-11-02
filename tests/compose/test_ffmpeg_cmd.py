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
from aivia.core.config import RenderConfig


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
    assert first_input.options == (
        "-loop",
        "1",
        "-framerate",
        "24",
        "-t",
        "4.000000",
    )
    assert command.inputs[2].options == (
        "-loop",
        "1",
        "-framerate",
        "24",
        "-t",
        "2.000000",
    )


def test_image_inputs_use_the_composition_frame_rate() -> None:
    composition = _composition(
        fps=30,
        layers=[_image_layer(source="still.png")],
    )

    command = build_ffmpeg_command(composition, "out.mp4")

    assert command.inputs[1].options == (
        "-loop",
        "1",
        "-framerate",
        "30",
        "-t",
        "4.000000",
    )


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


def test_text_default_size_comes_from_render_config() -> None:
    config = RenderConfig.from_mapping({"font_size": 24})
    layer = {"id": "title", "type": "text", "text": "hi", "x": 0, "y": 0}

    configured = build_ffmpeg_filtergraph(
        _composition(layers=[layer]),
        render=config,
    )
    explicit = build_ffmpeg_filtergraph(
        _composition(layers=[{**layer, "font_size": 24}]),
    )

    assert configured == explicit
    assert "fontsize=24" in configured


def test_text_default_family_comes_from_render_config() -> None:
    config = RenderConfig.from_mapping({"default_font": "Amiri"})
    layer = {"id": "title", "type": "text", "text": "hi", "x": 0, "y": 0}

    configured = build_ffmpeg_filtergraph(
        _composition(layers=[layer]),
        render=config,
    )
    explicit = build_ffmpeg_filtergraph(
        _composition(layers=[{**layer, "font_family": "Amiri"}]),
    )

    assert configured == explicit
    assert "Amiri-Regular.ttf" in configured


def test_explicit_text_style_overrides_render_config() -> None:
    config = RenderConfig.from_mapping({"default_font": "Amiri", "font_size": 24})
    layer = {
        "id": "title",
        "type": "text",
        "text": "hi",
        "x": 0,
        "y": 0,
        "font_family": "Lato",
        "font_size": 32,
    }

    configured = build_ffmpeg_filtergraph(
        _composition(layers=[layer]),
        render=config,
    )
    explicit = build_ffmpeg_filtergraph(_composition(layers=[layer]))

    assert configured == explicit
    assert "fontsize=32" in configured
    assert "Lato-Regular.ttf" in configured


def test_text_contrast_adds_an_outline_on_a_low_contrast_background() -> None:
    graph = build_ffmpeg_filtergraph(
        _composition(
            canvas={"width": 320, "height": 240, "background": "#808080"},
            layers=[
                {
                    "id": "title",
                    "type": "text",
                    "text": "hi",
                    "x": 0,
                    "y": 0,
                    "color": "#808080",
                }
            ],
        )
    )

    assert "fontcolor=0x808080" in graph
    assert "borderw=2" in graph
    assert "bordercolor=0x000000" in graph


def test_text_contrast_leaves_high_contrast_text_untreated() -> None:
    graph = build_ffmpeg_filtergraph(
        _composition(
            canvas={"width": 320, "height": 240, "background": "#ffffff"},
            layers=[
                {
                    "id": "title",
                    "type": "text",
                    "text": "hi",
                    "x": 0,
                    "y": 0,
                    "color": "#000000",
                }
            ],
        )
    )

    assert "fontcolor=0x000000" in graph
    assert "borderw=" not in graph
    assert "shadowcolor=" not in graph
    assert "boxcolor=" not in graph


@pytest.mark.parametrize(
    ("background", "expected"),
    [("#ffffff", "0x000000"), ("#000000", "0xffffff")],
)
def test_text_without_an_explicit_colour_is_chosen_from_the_background(
    background: str,
    expected: str,
) -> None:
    graph = build_ffmpeg_filtergraph(
        _composition(
            canvas={"width": 320, "height": 240, "background": background},
            layers=[{"id": "title", "type": "text", "text": "hi", "x": 0, "y": 0}],
        )
    )

    assert f"fontcolor={expected}" in graph


def test_text_contrast_shadow_strategy_maps_to_drawtext_shadow() -> None:
    config = RenderConfig.from_mapping({"contrast_strategy": "shadow"})

    graph = build_ffmpeg_filtergraph(
        _composition(
            canvas={"width": 320, "height": 240, "background": "#808080"},
            layers=[
                {
                    "id": "title",
                    "type": "text",
                    "text": "hi",
                    "x": 0,
                    "y": 0,
                    "color": "#808080",
                }
            ],
        ),
        render=config,
    )

    assert "shadowcolor=0x000000" in graph
    assert "shadowx=2" in graph
    assert "shadowy=2" in graph
    assert "borderw=" not in graph


def test_text_contrast_box_strategy_maps_to_a_drawtext_box() -> None:
    config = RenderConfig.from_mapping({"contrast_strategy": "box"})

    graph = build_ffmpeg_filtergraph(
        _composition(
            canvas={"width": 320, "height": 240, "background": "#808080"},
            layers=[
                {
                    "id": "title",
                    "type": "text",
                    "text": "hi",
                    "x": 0,
                    "y": 0,
                    "color": "#808080",
                }
            ],
        ),
        render=config,
    )

    assert "box=1" in graph
    assert "boxcolor=0x000000" in graph
    assert "boxborderw=8" in graph


def test_explicit_text_stroke_wins_over_the_contrast_outline() -> None:
    graph = build_ffmpeg_filtergraph(
        _composition(
            canvas={"width": 320, "height": 240, "background": "#808080"},
            layers=[
                {
                    "id": "title",
                    "type": "text",
                    "text": "hi",
                    "x": 0,
                    "y": 0,
                    "color": "#808080",
                    "stroke": "#ff0000",
                    "stroke_width": 3,
                }
            ],
        )
    )

    assert "borderw=3" in graph
    assert "bordercolor=0xff0000" in graph
    assert "borderw=2" not in graph


def test_text_contrast_can_be_relaxed_through_render_config() -> None:
    config = RenderConfig.from_mapping({"min_contrast": 1.0})

    graph = build_ffmpeg_filtergraph(
        _composition(
            canvas={"width": 320, "height": 240, "background": "#808080"},
            layers=[
                {
                    "id": "title",
                    "type": "text",
                    "text": "hi",
                    "x": 0,
                    "y": 0,
                    "color": "#808080",
                }
            ],
        ),
        render=config,
    )

    assert "borderw=" not in graph


def test_build_command_forwards_render_config_to_the_filtergraph() -> None:
    config = RenderConfig.from_mapping({"font_size": 24})
    layer = {"id": "title", "type": "text", "text": "hi", "x": 0, "y": 0}

    command = build_ffmpeg_command(
        _composition(layers=[layer]),
        "out.mp4",
        render=config,
    )

    assert "fontsize=24" in command.filtergraph


def test_build_command_accepts_a_render_config_as_config() -> None:
    config = RenderConfig.from_mapping({"font_size": 24})
    layer = {"id": "title", "type": "text", "text": "hi", "x": 0, "y": 0}

    command = build_ffmpeg_command(
        _composition(layers=[layer]),
        "out.mp4",
        config=config,
    )

    assert "fontsize=24" in command.filtergraph
    assert "s=320x240" in command.filtergraph


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


def test_image_masks_add_an_input_and_composite_the_mask_channel() -> None:
    composition = _composition(
        layers=[
            _image_layer(
                id="masked",
                source="photo.png",
                width=100,
                height=50,
                mask={
                    "source": "mask.png",
                    "mode": "luminance",
                    "invert": True,
                    "feather": 1.5,
                    "fit": "cover",
                },
            )
        ]
    )

    command = build_ffmpeg_command(composition, "out.mp4")
    graph = command.filtergraph

    assert [item.source for item in command.inputs[1:]] == ["photo.png", "mask.png"]
    assert command.inputs[2].layer_id == "masked:mask"
    assert command.inputs[2].options == (
        "-loop",
        "1",
        "-framerate",
        "24",
        "-t",
        "4.000000",
    )
    assert "[1:v]" in graph
    assert "[2:v]" in graph
    assert "format=gray" in graph
    assert "negate" in graph
    assert "gblur=sigma=1.5" in graph
    assert "[p0][m0]alphamerge" in graph
    assert "overlay=" in graph


def test_an_alpha_mask_extracts_the_mask_alpha_channel() -> None:
    composition = _composition(
        layers=[
            _image_layer(
                source="photo.png",
                width=100,
                height=50,
                mask="mask.png",
            )
        ]
    )

    graph = build_ffmpeg_filtergraph(composition)

    assert "alphaextract" in graph
    assert "format=gray" not in graph
    assert "[p0][m0]alphamerge" in graph


def test_masks_are_supported_on_video_layers_too() -> None:
    composition = _composition(
        layers=[
            _video_layer(
                source="clip.mp4",
                width=160,
                height=90,
                mask={"source": "mask.png", "mode": "luminance"},
            )
        ]
    )

    command = build_ffmpeg_command(composition, "out.mp4")

    assert [item.source for item in command.inputs[1:]] == ["clip.mp4", "mask.png"]
    assert "alphamerge" in command.filtergraph


def test_each_mask_input_is_indexed_next_to_its_layer() -> None:
    composition = _composition(
        layers=[
            _image_layer(id="first", source="first.png", mask="first-mask.png"),
            _image_layer(id="second", source="second.png", mask="second-mask.png"),
        ]
    )

    command = build_ffmpeg_command(composition, "out.mp4")

    assert [item.source for item in command.inputs[1:]] == [
        "first.png",
        "first-mask.png",
        "second.png",
        "second-mask.png",
    ]
    graph = command.filtergraph
    assert "[1:v]" in graph
    assert "[2:v]" in graph
    assert "[3:v]" in graph
    assert "[4:v]" in graph
    assert "[p0][m0]alphamerge" in graph
    assert "[p1][m1]alphamerge" in graph


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("multiply", "multiply"),
        ("screen", "screen"),
        ("overlay", "overlay"),
        ("add", "addition"),
        ("subtract", "subtract"),
        ("darken", "darken"),
        ("lighten", "lighten"),
        ("difference", "difference"),
        ("hard_light", "hardlight"),
        ("soft_light", "softlight"),
    ],
)
def test_blend_modes_map_to_ffmpeg_blend_filters(mode: str, expected: str) -> None:
    composition = _composition(layers=[_image_layer(source="top.png", blend_mode=mode)])

    graph = build_ffmpeg_filtergraph(composition)

    assert f"blend=all_mode={expected}:shortest=1" in graph
    assert "overlay=" in graph


def test_the_normal_blend_mode_keeps_the_plain_overlay() -> None:
    graph = build_ffmpeg_filtergraph(
        _composition(layers=[_image_layer(source="top.png")])
    )

    assert "blend=all_mode" not in graph
    assert "overlay=" in graph


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ({"type": "blur", "radius": 2}, "gblur=sigma=2"),
        ({"type": "box_blur", "radius": 1}, "boxblur=1"),
        ({"type": "sharpen", "amount": 1}, "unsharp=5:5:1"),
        ({"type": "brightness", "amount": 0.5}, "eq=brightness=-0.500000"),
        ({"type": "contrast", "amount": 0.5}, "eq=contrast=0.500000"),
        ({"type": "saturation", "amount": 0.5}, "eq=saturation=0.500000"),
        ({"type": "grayscale"}, "hue=s=0"),
        ({"type": "invert"}, "negate"),
        ({"type": "autocontrast"}, "normalize"),
    ],
)
def test_basic_image_filters_map_to_ffmpeg_filters(
    spec: dict[str, Any],
    expected: str,
) -> None:
    composition = _composition(layers=[_image_layer(source="top.png", filters=[spec])])

    assert expected in build_ffmpeg_filtergraph(composition)


def test_image_filters_keep_their_declared_order() -> None:
    graph = build_ffmpeg_filtergraph(
        _composition(
            layers=[
                _image_layer(
                    source="top.png",
                    filters={"brightness": 1.5, "invert": True},
                )
            ]
        )
    )

    assert graph.index("eq=brightness=0.500000") < graph.index("negate")


def test_filters_and_masks_apply_to_video_image_layers_in_order() -> None:
    graph = build_ffmpeg_filtergraph(
        _composition(
            layers=[
                _video_layer(
                    source="clip.mp4",
                    width=160,
                    height=90,
                    filters=[{"type": "grayscale"}],
                    mask="mask.png",
                    blend="screen",
                )
            ]
        )
    )

    assert graph.index("hue=s=0") < graph.index("alphamerge")
    assert graph.index("alphamerge") < graph.index("blend=all_mode=screen")


def test_mask_inputs_shift_the_audio_input_indices() -> None:
    composition = _composition(
        layers=[_image_layer(source="photo.png", mask="mask.png")],
        audio_tracks=[{"id": "voice", "source": "voice.wav"}],
    )

    graph = build_ffmpeg_filtergraph(composition)

    assert "[3:a]" in graph
    assert "[2:a]" not in graph
