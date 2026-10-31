"""Video request and result DTOs validate generation options and outputs."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from aivia.providers.media import VideoRequest, VideoResult
from aivia.providers.protocols.media import MediaInput, VideoFormat
from aivia.providers.protocols.video import VideoResponse

_FRAME_URL = "https://example.com/media/frame.png"
_VIDEO_URL = "https://example.com/media/clip.mp4"


def test_video_request_carries_duration_aspect_resolution_fps_seed_and_frames() -> None:
    first = MediaInput(data=b"first-frame", media_type="image/png")
    last = MediaInput(url=_FRAME_URL, media_type="image/png")

    request = VideoRequest(
        prompt="  a wave rolling in  ",
        negative_prompt="  text  ",
        model="  some-video-model  ",
        first_frame=first,
        last_frame=last,
        duration_seconds=5.5,
        width=1080,
        height=1920,
        aspect_ratio=" 9:16 ",
        fps=30,
        format=VideoFormat.WEBM,
        seed=11,
        count=2,
        generate_audio=True,
        audio_prompt="  soft ocean ambience  ",
    )

    assert request.prompt == "a wave rolling in"
    assert request.negative_prompt == "text"
    assert request.model == "some-video-model"
    assert request.first_frame == first
    assert request.last_frame == last
    assert request.duration_seconds == 5.5
    assert (request.width, request.height) == (1080, 1920)
    assert request.aspect_ratio == "9:16"
    assert request.fps == 30
    assert request.format is VideoFormat.WEBM
    assert request.seed == 11
    assert request.count == 2
    assert request.generate_audio is True
    assert request.audio_prompt == "soft ocean ambience"


def test_video_request_defaults_and_audio_flags() -> None:
    request = VideoRequest(prompt="a pan")

    assert request.duration_seconds is None
    assert request.aspect_ratio is None
    assert request.width is None
    assert request.height is None
    assert request.fps is None
    assert request.seed is None
    assert request.count == 1
    assert request.first_frame is None
    assert request.last_frame is None
    assert request.generate_audio is False
    assert request.audio_prompt is None
    assert request.format is VideoFormat.MP4


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prompt": "   "},
        {"prompt": "x", "duration_seconds": 0},
        {"prompt": "x", "duration_seconds": -1},
        {"prompt": "x", "width": 1080},
        {"prompt": "x", "height": 1920},
        {"prompt": "x", "width": 0, "height": 1920},
        {"prompt": "x", "height": 0, "width": 1080},
        {"prompt": "x", "aspect_ratio": "16x9"},
        {"prompt": "x", "aspect_ratio": "0:9"},
        {"prompt": "x", "aspect_ratio": "16:0"},
        {"prompt": "x", "fps": 0},
        {"prompt": "x", "fps": 121},
        {"prompt": "x", "count": 0},
        {"prompt": "x", "count": 5},
        {"prompt": "x", "timeout_seconds": 0},
    ],
)
def test_video_request_rejects_invalid_options(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        VideoRequest(**kwargs)


def test_video_request_rejects_audio_prompt_without_audio_and_bad_frames() -> None:
    with pytest.raises(ValidationError, match="audio_prompt"):
        VideoRequest(prompt="x", audio_prompt="ambience")

    with pytest.raises(ValidationError, match="exactly one"):
        MediaInput(data=b"frame", url=_FRAME_URL)

    with pytest.raises(ValidationError, match="non-empty"):
        MediaInput(data=b"")


def test_video_request_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        VideoRequest.model_validate({"prompt": "x", "unknown": True})


def test_video_result_reports_source_dimensions_seed_and_audio() -> None:
    result = VideoResult(
        data=b"video",
        format=VideoFormat.MP4,
        width=1920,
        height=1080,
        duration_seconds=6.0,
        fps=24,
        has_audio=True,
        seed=7,
        thumbnail_url=_FRAME_URL,
    )

    assert result.is_inline is True
    assert result.size_bytes == 5
    assert result.dimensions is not None
    assert (result.dimensions.width, result.dimensions.height) == (1920, 1080)
    assert result.duration_seconds == 6.0
    assert result.fps == 24
    assert result.has_audio is True
    assert result.seed == 7
    assert result.format is VideoFormat.MP4
    assert result.thumbnail_url == _FRAME_URL

    remote = VideoResult(url=_VIDEO_URL, has_audio=False, seed=8)

    assert remote.is_inline is False
    assert remote.size_bytes is None
    assert remote.dimensions is None
    assert remote.has_audio is False
    assert remote.seed == 8


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"data": b"video", "url": _VIDEO_URL},
        {"data": b""},
        {"data": b"video", "width": 1920},
        {"data": b"video", "height": 1080},
        {"data": b"video", "width": 0, "height": 1080},
        {"data": b"video", "duration_seconds": 0},
        {"data": b"video", "duration_seconds": -1},
        {"data": b"video", "fps": 0},
        {"data": b"video", "fps": 121},
        {"url": "ftp://example.com/clip.mp4"},
        {"data": b"video", "thumbnail_url": "ftp://example.com/frame.png"},
    ],
)
def test_video_result_rejects_invalid_sources_and_ranges(
    kwargs: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        VideoResult(**kwargs)


def test_video_result_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        VideoResult.model_validate({"data": b"video", "unknown": True})


def test_video_response_requires_clips_and_exposes_the_first() -> None:
    result = VideoResult(data=b"video", width=1080, height=1920)
    response = VideoResponse(
        videos=[result],
        model="some-video-model",
        provider="fake-video",
    )

    assert response.first is result
    assert response.count == 1

    with pytest.raises(ValidationError):
        VideoResponse(videos=[])


def test_video_dtos_are_immutable() -> None:
    request = VideoRequest(prompt="a pan")
    result = VideoResult(data=b"video")

    with pytest.raises(ValidationError):
        request.fps = 24
    with pytest.raises(ValidationError):
        result.seed = 1
