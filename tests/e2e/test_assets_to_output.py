"""Mock generated assets become validated MP4, GIF and web card outputs.

The deterministic mock providers return real PNG and WAV bytes. The test feeds
those assets into the composition renderers, then checks every finished file
with the same platform validation the publishing path uses. Nothing here calls
the network: only the mock providers and the local ffmpeg toolchain run.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest
from PIL import Image

from aivia.compose import (
    Composition,
    OutputIssues,
    render_gif,
    render_video,
    render_web_card,
    validate_output,
)
from aivia.providers.mock import MockImageProvider, MockMusicProvider
from aivia.providers.protocols.audio import AudioFormat
from aivia.providers.protocols.image import ImageRequest
from aivia.providers.protocols.music import MusicRequest

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        FFMPEG is None or FFPROBE is None,
        reason="ffmpeg and ffprobe are required for a real render",
    ),
]


def _write_image(path: Path, data: bytes | None) -> Path:
    assert data is not None
    path.write_bytes(data)
    return path


@pytest.fixture
def assets(tmp_path: Path) -> dict[str, Path]:
    images = MockImageProvider(name="mock-image", seed=11)
    music = MockMusicProvider(name="mock-music", seed=11)

    vertical = asyncio.run(
        images.generate(
            ImageRequest(
                prompt="A vertical mock backdrop",
                width=1080,
                height=1920,
            )
        )
    )
    square = asyncio.run(
        images.generate(
            ImageRequest(
                prompt="A square mock backdrop",
                width=1080,
                height=1080,
            )
        )
    )
    badge = asyncio.run(
        images.generate(
            ImageRequest(
                prompt="A small mock badge",
                width=256,
                height=256,
            )
        )
    )
    score = asyncio.run(
        music.generate(
            MusicRequest(
                prompt="A warm mock score",
                duration_seconds=2.0,
                instrumental=True,
                format=AudioFormat.WAV,
            )
        )
    )

    track = score.tracks[0]
    return {
        "vertical": _write_image(tmp_path / "vertical.png", vertical.images[0].data),
        "square": _write_image(tmp_path / "square.png", square.images[0].data),
        "badge": _write_image(tmp_path / "badge.png", badge.images[0].data),
        "score": _write_image(tmp_path / "score.wav", track.data),
    }


def _video_composition(assets: dict[str, Path]) -> Composition:
    return Composition.model_validate(
        {
            "id": "mock-assets-video",
            "canvas": {"width": 1080, "height": 1920, "background": "#050816"},
            "duration_seconds": 2.0,
            "fps": 24,
            "loudness_lufs": -14.0,
            "layers": [
                {
                    "id": "background",
                    "type": "image",
                    "source": str(assets["vertical"]),
                    "x": 0,
                    "y": 0,
                    "width": 1080,
                    "height": 1920,
                    "fit": "stretch",
                    "z": 0,
                },
                {
                    "id": "headline",
                    "type": "text",
                    "text": "Mock assets, real outputs",
                    "x": 80,
                    "y": 1180,
                    "width": 920,
                    "height": 300,
                    "font_family": "Lato",
                    "font_size": 84,
                    "font_weight": "bold",
                    "color": "#ffffff",
                    "align": "center",
                    "vertical_align": "middle",
                    "z": 1,
                },
            ],
            "audio_tracks": [
                {
                    "id": "score",
                    "source": str(assets["score"]),
                    "role": "music",
                    "volume": -2.0,
                    "fade_out": 0.25,
                }
            ],
        }
    )


def _gif_composition(assets: dict[str, Path]) -> Composition:
    return Composition.model_validate(
        {
            "id": "mock-assets-gif",
            "canvas": {"width": 1080, "height": 1080, "background": "#050816"},
            "duration_seconds": 1.0,
            "fps": 12,
            "layers": [
                {
                    "id": "background",
                    "type": "image",
                    "source": str(assets["square"]),
                    "x": 0,
                    "y": 0,
                    "width": 1080,
                    "height": 1080,
                    "fit": "stretch",
                    "z": 0,
                },
                {
                    "id": "badge",
                    "type": "image",
                    "source": str(assets["badge"]),
                    "x": 0,
                    "y": 0,
                    "width": 220,
                    "height": 220,
                    "fit": "cover",
                    "keyframes": {
                        "position": [
                            {"time": 0, "value": [40, 40], "easing": "linear"},
                            {"time": 1.0, "value": [820, 820]},
                        ]
                    },
                    "z": 1,
                },
            ],
        }
    )


def _web_card_composition(assets: dict[str, Path]) -> Composition:
    return Composition.model_validate(
        {
            "id": "mock-assets-card",
            "canvas": {"width": 1080, "height": 1920, "background": "#050816"},
            "duration_seconds": 3.0,
            "fps": 30,
            "layers": [
                {
                    "id": "background",
                    "type": "image",
                    "source": str(assets["vertical"]),
                    "x": 0,
                    "y": 0,
                    "width": 1080,
                    "height": 1920,
                    "fit": "stretch",
                    "z": 0,
                },
                {
                    "id": "headline",
                    "type": "text",
                    "text": "Mock assets, real outputs",
                    "x": 80,
                    "y": 620,
                    "width": 920,
                    "height": 300,
                    "font_size": 84,
                    "color": "#ffffff",
                    "align": "center",
                    "entrance": {
                        "type": "slide_up",
                        "duration": 0.8,
                        "distance": 64,
                    },
                    "z": 1,
                },
                {
                    "id": "body",
                    "type": "text",
                    "text": "A self contained card built from a generated asset.",
                    "x": 80,
                    "y": 960,
                    "width": 920,
                    "height": 200,
                    "font_size": 40,
                    "color": "#e2e8f0",
                    "align": "center",
                    "start_seconds": 0.4,
                    "z": 2,
                },
            ],
        }
    )


def _messages(issues: OutputIssues) -> list[str]:
    return [str(issue) for issue in issues]


def test_mock_assets_render_to_a_validated_mp4_gif_and_web_card(
    assets: dict[str, Path],
    tmp_path: Path,
) -> None:
    assert FFMPEG is not None
    assert FFPROBE is not None
    output = tmp_path / "output"
    output.mkdir()

    mp4 = output / "story.mp4"
    video = render_video(
        _video_composition(assets),
        mp4,
        preset="ultrafast",
        crf=30,
    )
    assert mp4.is_file()
    assert mp4.stat().st_size > 0
    assert video.probe.primary_video is not None
    assert (video.probe.primary_video.width, video.probe.primary_video.height) == (
        1080,
        1920,
    )
    assert video.probe.audio_streams
    video_issues = validate_output(
        video,
        "youtube",
        measure=True,
        ffmpeg_executable=FFMPEG,
    )
    assert video_issues == [], _messages(video_issues)
    assert video_issues.ok is True
    assert video_issues.media_type == "video"
    assert video_issues.platform is not None
    assert video_issues.platform.id == "youtube"

    gif = output / "loop.gif"
    animated = render_gif(_gif_composition(assets), gif, fps=10)
    assert gif.is_file()
    assert gif.stat().st_size > 0
    assert animated.size_bytes == gif.stat().st_size
    with Image.open(gif) as image:
        assert int(getattr(image, "n_frames", 1)) > 1
    gif_issues = validate_output(animated, "instagram")
    assert gif_issues == [], _messages(gif_issues)
    assert gif_issues.ok is True
    assert gif_issues.media_type == "image"
    assert gif_issues.platform is not None
    assert gif_issues.platform.id == "instagram"

    card = output / "card.html"
    web_card = render_web_card(
        _web_card_composition(assets),
        card,
        title="Mock assets, real outputs",
        summary="Generated offline, validated for publishing.",
        animation="fade",
    )
    assert card.is_file()
    assert web_card.background_embedded is True
    document = card.read_text(encoding="utf-8")
    assert document.startswith("<!doctype html>")
    assert "data:image/png;base64," in document
    assert 'src="http' not in document
    assert 'href="http' not in document
    card_issues = validate_output(web_card, "telegram")
    assert card_issues == [], _messages(card_issues)
    assert card_issues.ok is True
    assert card_issues.media_type == "web_card"
    assert card_issues.platform is not None
    assert card_issues.platform.id == "telegram"
