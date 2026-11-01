"""Cut, crossfade and slide transitions sampled from stills and real video."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from aivia.compose import (
    Composition,
    Effect,
    PillowStillRenderer,
    SlideDirection,
    Transition,
    TransitionError,
    TransitionType,
    build_ffmpeg_filtergraph,
    compose_transition,
    crossfade,
    cut,
    layer_transition_for,
    render_transition,
    render_video,
    slide_transition,
    transition_from_effect,
)

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

pytestmark = pytest.mark.slow

requires_ffmpeg = pytest.mark.skipif(
    FFMPEG is None or FFPROBE is None,
    reason="ffmpeg and ffprobe are required for a real video render",
)

RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)


def _solid(path: Path, color: tuple[int, int, int]) -> Path:
    Image.new("RGB", (40, 30), color).save(path)
    return path


def _composition(
    *layers: dict[str, Any],
    duration: float = 2.0,
    size: tuple[int, int] = (320, 240),
) -> Composition:
    return Composition.model_validate(
        {
            "canvas": {"width": size[0], "height": size[1], "background": "#000000"},
            "duration_seconds": duration,
            "fps": 24,
            "layers": list(layers),
        }
    )


def _assert_color(
    image: Image.Image,
    point: tuple[int, int],
    expected: tuple[int, int, int],
    *,
    tolerance: int = 24,
) -> None:
    actual = image.getpixel(point)
    assert isinstance(actual, tuple)
    for channel, target in zip(actual[:3], expected, strict=True):
        assert abs(channel - target) <= tolerance, (point, actual, expected)


def _run(argv: list[str]) -> None:
    subprocess.run(argv, check=True, capture_output=True, text=True)


def _video_frame(source: Path, time_seconds: float, destination: Path) -> Image.Image:
    assert FFMPEG is not None
    _run(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{time_seconds:.6f}",
            "-i",
            str(source),
            "-frames:v",
            "1",
            str(destination),
        ]
    )
    with Image.open(destination) as image:
        return image.convert("RGB")


def test_transition_shorthands_and_constructors() -> None:
    assert Transition.model_validate("cut").type is TransitionType.CUT
    assert Transition.model_validate("fade").type is TransitionType.CROSSFADE
    assert Transition.model_validate("slide_left").direction is SlideDirection.LEFT
    assert Transition.model_validate("right").direction is SlideDirection.RIGHT

    assert cut().is_cut is True
    assert crossfade(duration=0.75).duration == pytest.approx(0.75)
    assert slide_transition("up", duration=0.25).direction is SlideDirection.UP


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "wipe_left"},
        {"type": "crossfade", "duration": 0},
        {"type": "slide", "duration": -1},
        {"type": "crossfade", "duration": float("nan")},
        {"type": "slide", "direction": "sideways"},
    ],
)
def test_invalid_transitions_are_rejected(payload: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        Transition.model_validate(payload)


def test_crossfade_blends_the_two_complete_frames() -> None:
    before = Image.new("RGBA", (10, 10), (*RED, 255))
    after = Image.new("RGBA", (10, 10), (*BLUE, 255))
    transition = crossfade(duration=1.0)

    assert render_transition(before, after, transition, 0.0).getpixel((5, 5)) == (
        255,
        0,
        0,
        255,
    )
    middle = render_transition(before, after, transition, 0.5).getpixel((5, 5))
    assert isinstance(middle, tuple)
    assert middle[0] == pytest.approx(128, abs=2)
    assert middle[2] == pytest.approx(128, abs=2)
    assert render_transition(before, after, transition, 1.0).getpixel((5, 5)) == (
        0,
        0,
        255,
        255,
    )


def test_cut_selects_the_incoming_frame() -> None:
    before = Image.new("RGBA", (10, 10), (*RED, 255))
    after = Image.new("RGBA", (10, 10), (*BLUE, 255))

    assert render_transition(before, after, cut(), 0.0).getpixel((5, 5)) == (
        0,
        0,
        255,
        255,
    )


def test_slide_moves_the_incoming_frame_across() -> None:
    before = Image.new("RGBA", (100, 20), (*RED, 255))
    after = Image.new("RGBA", (100, 20), (*BLUE, 255))
    transition = slide_transition("left", duration=1.0)

    start = render_transition(before, after, transition, 0.0)
    assert start.getpixel((10, 10)) == (*RED, 255)

    middle = render_transition(before, after, transition, 0.5)
    assert middle.getpixel((10, 10)) == (*RED, 255)
    assert middle.getpixel((90, 10)) == (*BLUE, 255)

    end = render_transition(before, after, transition, 1.0)
    assert end.getpixel((10, 10)) == (*BLUE, 255)


def test_crossfade_layer_samples_previous_and_incoming_frames(tmp_path: Path) -> None:
    red = _solid(tmp_path / "red.png", RED)
    blue = _solid(tmp_path / "blue.png", BLUE)
    composition = _composition(
        {"id": "red", "type": "image", "source": str(red), "start": 0, "end": 1},
        {
            "id": "blue",
            "type": "image",
            "source": str(blue),
            "start": 1,
            "end": 2,
            "transition": {"type": "crossfade", "duration": 0.5},
        },
    )
    renderer = PillowStillRenderer()

    _assert_color(renderer.render_still(composition, 0.75), (160, 120), RED)
    _assert_color(renderer.render_still(composition, 1.0), (160, 120), RED)
    middle = renderer.render_still(composition, 1.25)
    _assert_color(middle, (160, 120), (128, 0, 128), tolerance=2)
    _assert_color(renderer.render_still(composition, 1.75), (160, 120), BLUE)


def test_slide_layer_reveals_the_previous_frame_on_the_far_side(
    tmp_path: Path,
) -> None:
    red = _solid(tmp_path / "red.png", RED)
    blue = _solid(tmp_path / "blue.png", BLUE)
    composition = _composition(
        {"id": "red", "type": "image", "source": str(red), "start": 0, "end": 1},
        {
            "id": "blue",
            "type": "image",
            "source": str(blue),
            "start": 1,
            "end": 2,
            "transition": {"type": "slide", "direction": "left", "duration": 0.5},
        },
    )
    renderer = PillowStillRenderer()

    middle = renderer.render_still(composition, 1.25)
    _assert_color(middle, (80, 120), RED)
    _assert_color(middle, (240, 120), BLUE)
    _assert_color(renderer.render_still(composition, 1.75), (80, 120), BLUE)


def test_cut_does_not_extend_the_outgoing_layer(tmp_path: Path) -> None:
    red = _solid(tmp_path / "red.png", RED)
    blue = _solid(tmp_path / "blue.png", BLUE)
    composition = _composition(
        {"id": "red", "type": "image", "source": str(red), "start": 0, "end": 1},
        {
            "id": "blue",
            "type": "image",
            "source": str(blue),
            "start": 1,
            "end": 2,
            "transition": "cut",
        },
    )
    renderer = PillowStillRenderer()

    _assert_color(renderer.render_still(composition, 0.75), (160, 120), RED)
    _assert_color(renderer.render_still(composition, 1.25), (160, 120), BLUE)


def test_layer_transition_is_read_from_the_schema() -> None:
    composition = _composition(
        {
            "id": "slide",
            "type": "image",
            "source": "still.png",
            "transition_in": "slide_down",
        }
    )

    transition = layer_transition_for(composition.layers[0])

    assert transition is not None
    assert transition.type is TransitionType.SLIDE
    assert transition.direction is SlideDirection.DOWN


def test_compose_transition_does_not_mutate_its_inputs() -> None:
    base = Image.new("RGBA", (10, 10), (*RED, 255))
    incoming = Image.new("RGBA", (10, 10), (*BLUE, 255))
    before = base.copy()

    result = compose_transition(base, incoming, crossfade(), 0.5)

    assert base.tobytes() == before.tobytes()
    assert result is not base


def test_ffmpeg_graph_fades_incoming_layers_and_extends_outgoing_ones(
    tmp_path: Path,
) -> None:
    red = _solid(tmp_path / "red.png", RED)
    blue = _solid(tmp_path / "blue.png", BLUE)
    composition = _composition(
        {"id": "red", "type": "image", "source": str(red), "start": 0, "end": 1},
        {
            "id": "blue",
            "type": "image",
            "source": str(blue),
            "start": 1,
            "end": 2,
            "transition": {"type": "crossfade", "duration": 0.5},
        },
    )

    graph = build_ffmpeg_filtergraph(composition)

    assert "clip(T/0.500000,0,1)" in graph
    assert "between(t,0.000000,1.500000)" in graph


def test_ffmpeg_graph_offsets_a_sliding_layer(tmp_path: Path) -> None:
    red = _solid(tmp_path / "red.png", RED)
    blue = _solid(tmp_path / "blue.png", BLUE)
    composition = _composition(
        {"id": "red", "type": "image", "source": str(red), "start": 0, "end": 1},
        {
            "id": "blue",
            "type": "image",
            "source": str(blue),
            "start": 1,
            "end": 2,
            "transition": {"type": "slide", "direction": "left", "duration": 0.5},
        },
    )

    graph = build_ffmpeg_filtergraph(composition)

    assert "clip((t-1.000000)/0.500000,0,1)" in graph
    assert "320" in graph


@requires_ffmpeg
def test_real_video_samples_cut_crossfade_and_slide(tmp_path: Path) -> None:
    red = _solid(tmp_path / "red.png", RED)
    blue = _solid(tmp_path / "blue.png", BLUE)
    green = _solid(tmp_path / "green.png", GREEN)
    yellow = _solid(tmp_path / "yellow.png", YELLOW)
    composition = _composition(
        {"id": "red", "type": "image", "source": str(red), "start": 0, "end": 1},
        {
            "id": "blue",
            "type": "image",
            "source": str(blue),
            "start": 1,
            "end": 2,
            "transition": {"type": "crossfade", "duration": 0.5},
        },
        {
            "id": "green",
            "type": "image",
            "source": str(green),
            "start": 2,
            "end": 3,
            "transition": {"type": "slide", "direction": "left", "duration": 0.5},
        },
        {
            "id": "yellow",
            "type": "image",
            "source": str(yellow),
            "start": 3,
            "end": 4,
            "transition": "cut",
        },
        duration=4.0,
    )
    output = tmp_path / "transitions.mp4"

    render_video(composition, output, preset="ultrafast", crf=28)

    _assert_color(_video_frame(output, 0.5, tmp_path / "a.png"), (160, 120), RED)
    _assert_color(
        _video_frame(output, 1.25, tmp_path / "b.png"),
        (160, 120),
        (128, 0, 128),
    )
    _assert_color(_video_frame(output, 1.75, tmp_path / "c.png"), (160, 120), BLUE)
    slide = _video_frame(output, 2.25, tmp_path / "d.png")
    _assert_color(slide, (80, 120), BLUE)
    _assert_color(slide, (240, 120), GREEN)
    _assert_color(_video_frame(output, 2.75, tmp_path / "e.png"), (160, 120), GREEN)
    _assert_color(_video_frame(output, 3.25, tmp_path / "f.png"), (160, 120), YELLOW)


def test_transition_effects_accept_generic_and_direct_types() -> None:
    generic = transition_from_effect(
        Effect.model_validate(
            {
                "type": "transition",
                "params": {"type": "crossfade", "duration": 0.4},
            }
        )
    )
    direct = transition_from_effect(Effect.model_validate("slide_up"))

    assert generic is not None
    assert generic.type is TransitionType.CROSSFADE
    assert generic.duration == pytest.approx(0.4)
    assert direct is not None
    assert direct.type is TransitionType.SLIDE
    assert direct.direction is SlideDirection.UP
    assert transition_from_effect(Effect.model_validate("blur")) is None


def test_mismatched_transition_frames_raise() -> None:
    with pytest.raises(TransitionError):
        render_transition(
            Image.new("RGBA", (10, 10), (*RED, 255)),
            Image.new("RGBA", (20, 10), (*BLUE, 255)),
            crossfade(),
            0.5,
        )
