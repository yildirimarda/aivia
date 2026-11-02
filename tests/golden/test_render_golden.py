"""Golden renders on the pinned Linux toolchain with the bundled fonts.

The comparison is authoritative only on the pinned Linux image: the same
bundled font files and the same ffmpeg build that produced the artifacts under
``tests/golden/render/``. A local run with another toolchain skips instead of
reporting a false regression. Set ``AIVA_COMPARE_GOLDENS=1`` to compare and
``AIVA_GOLDEN_FFMPEG`` / ``AIVA_GOLDEN_FFPROBE`` to point at the pinned
binaries; ``AIVA_GOLDEN_REQUIRE_FFMPEG=1`` turns a missing or mismatched pin
into a failure for CI.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageChops, ImageFont

from aivia.compose import (
    PILLOW_STILL_RENDERER_VERSION,
    VIDEO_RENDERER_VERSION,
    Composition,
    FontManager,
    PillowStillRenderer,
    render_video,
)
from aivia.core.toolchain import ffmpeg_version, font_set_identity

GOLDEN_DIR = Path(__file__).resolve().parent / "render"
MANIFEST_PATH = GOLDEN_DIR / "manifest.json"
COMPARE_ENV = "AIVA_COMPARE_GOLDENS"
FFMPEG_ENV = "AIVA_GOLDEN_FFMPEG"
FFPROBE_ENV = "AIVA_GOLDEN_FFPROBE"
REQUIRE_ENV = "AIVA_GOLDEN_REQUIRE_FFMPEG"
FONT_FAMILY = "Lato"

GLYPH_CASES: dict[str, str] = {
    "turkish": "İstanbul ışığı: çiçek, şüphe, öğüt, ğüzel — Iıİi",  # noqa: RUF001
    "german": "Grüße aus München: Äpfel, Öl, Übermut, Straße, ß",
}

pytestmark = [
    pytest.mark.golden,
    pytest.mark.skipif(
        not sys.platform.startswith("linux"),
        reason="golden renders are authoritative only on the pinned Linux image",
    ),
]


def _manifest() -> dict[str, Any]:
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _require_comparison() -> None:
    if os.environ.get(COMPARE_ENV) != "1":
        pytest.skip(f"set {COMPARE_ENV}=1 to compare golden renders")


def _still_composition(case_id: str) -> Composition:
    return Composition.model_validate(
        {
            "id": f"golden_still_{case_id}",
            "canvas": {"width": 640, "height": 160, "background": "#101820"},
            "duration_seconds": 1.0,
            "fps": 10,
            "layers": [
                {
                    "id": "glyphs",
                    "type": "text",
                    "text": GLYPH_CASES[case_id],
                    "font_family": FONT_FAMILY,
                    "font_size": 40,
                    "color": "#f2f2f2",
                    "x": 16,
                    "y": 16,
                    "width": 608,
                    "height": 128,
                    "vertical_align": "middle",
                    "wrap": False,
                }
            ],
        }
    )


def _video_composition(case_id: str) -> Composition:
    return Composition.model_validate(
        {
            "id": f"golden_video_{case_id}",
            "canvas": {"width": 320, "height": 120, "background": "#101820"},
            "duration_seconds": 0.5,
            "fps": 10,
            "layers": [
                {
                    "id": "glyphs",
                    "type": "text",
                    "text": GLYPH_CASES[case_id],
                    "font_family": FONT_FAMILY,
                    "font_size": 22,
                    "color": "#f2f2f2",
                    "x": 8,
                    "y": 8,
                    "width": 304,
                    "height": 104,
                    "vertical_align": "middle",
                    "wrap": False,
                }
            ],
        }
    )


def _render_still(case_id: str) -> Image.Image:
    return PillowStillRenderer(fonts=FontManager()).render_still(
        _still_composition(case_id),
        0.0,
    )


def _pinned_binaries() -> tuple[str, str]:
    manifest = _manifest()
    pinned = str(manifest["ffmpeg"]["version"])
    executable = os.environ.get(FFMPEG_ENV) or shutil.which("ffmpeg")
    probe = os.environ.get(FFPROBE_ENV) or shutil.which("ffprobe")
    if executable is None or probe is None:
        pytest.skip("the pinned ffmpeg and ffprobe are not installed")
    detected = ffmpeg_version(executable)
    if detected != pinned:
        pytest.skip(f"goldens require ffmpeg {pinned}, found {detected or 'unknown'}")
    return executable, probe


def _extract_frame(
    executable: str,
    source: Path,
    destination: Path,
    time_seconds: float,
) -> None:
    subprocess.run(
        [
            executable,
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
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _assert_matches(
    rendered: Image.Image,
    expected: Image.Image,
    tolerance: int,
) -> None:
    assert rendered.size == expected.size
    difference = ImageChops.difference(rendered, expected)
    maximum = max(
        int(band[1]) for band in difference.getextrema() if isinstance(band, tuple)
    )
    assert maximum <= tolerance, (
        f"golden differs by {maximum} channel levels (tolerance {tolerance})"
    )


def test_the_manifest_records_the_current_renderers_and_bundled_fonts() -> None:
    manifest = _manifest()

    assert manifest["renderer"] == PILLOW_STILL_RENDERER_VERSION
    assert manifest["video_renderer"] == VIDEO_RENDERER_VERSION
    assert manifest["font_family"] == FONT_FAMILY
    assert manifest["font_set"] == font_set_identity()
    assert manifest["ffmpeg"]["version"]
    assert manifest["ffmpeg"]["source"].startswith("https://")
    assert set(manifest["cases"]) == set(GLYPH_CASES)


@pytest.mark.parametrize("case_id", sorted(GLYPH_CASES))
def test_the_bundled_font_covers_every_glyph_in_the_case(case_id: str) -> None:
    font = ImageFont.truetype(str(FontManager().resolve(FONT_FAMILY)), 40)

    missing = [
        character
        for character in GLYPH_CASES[case_id]
        if not character.isspace() and font.getmask(character).getbbox() is None
    ]

    assert missing == []


def test_the_pinned_ffmpeg_is_installed_when_required() -> None:
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 to require the pinned toolchain")

    manifest = _manifest()
    executable = os.environ.get(FFMPEG_ENV) or shutil.which("ffmpeg")
    probe = os.environ.get(FFPROBE_ENV) or shutil.which("ffprobe")

    assert executable is not None, "the pinned ffmpeg is not on PATH"
    assert probe is not None, "the pinned ffprobe is not on PATH"
    assert ffmpeg_version(executable) == manifest["ffmpeg"]["version"]


@pytest.mark.parametrize("case_id", sorted(GLYPH_CASES))
def test_the_still_render_matches_the_golden_image(case_id: str) -> None:
    _require_comparison()
    manifest = _manifest()
    entry = manifest["cases"][case_id]["still"]

    rendered = _render_still(case_id)
    expected = Image.open(GOLDEN_DIR / entry["file"]).convert("RGBA")

    assert rendered.size == (entry["width"], entry["height"])
    _assert_matches(rendered, expected, int(entry["tolerance"]))


@pytest.mark.parametrize("case_id", sorted(GLYPH_CASES))
def test_the_video_frame_matches_the_golden_image(
    case_id: str,
    tmp_path: Path,
) -> None:
    _require_comparison()
    ffmpeg, ffprobe = _pinned_binaries()
    manifest = _manifest()
    entry = manifest["cases"][case_id]["video"]
    output = tmp_path / f"{case_id}.mp4"
    frame = tmp_path / f"{case_id}.png"

    render_video(
        _video_composition(case_id),
        output,
        ffmpeg_executable=ffmpeg,
        ffprobe_executable=ffprobe,
        preset="ultrafast",
        crf=28,
    )
    _extract_frame(ffmpeg, output, frame, float(entry["time_seconds"]))

    rendered = Image.open(frame).convert("RGBA")
    expected = Image.open(GOLDEN_DIR / entry["file"]).convert("RGBA")

    assert rendered.size == (entry["width"], entry["height"])
    _assert_matches(rendered, expected, int(entry["tolerance"]))
