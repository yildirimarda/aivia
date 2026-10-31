"""Sample async declarative video spec: metadata and a mock queue run."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import httpx

from aivia.providers import (
    HttpProviderSpec,
    HttpSpecProvider,
    build_http_provider,
    load_http_spec,
)
from aivia.providers.protocols import (
    CAPABILITY_VIDEO_GEN,
    VideoRequest,
    VideoResponse,
)
from aivia.providers.protocols.media import VideoFormat

ROOT = Path(__file__).resolve().parents[3]
SAMPLE_SPEC = ROOT / "examples" / "provider_spec.example.yaml"
API_KEY = "test-key"
SUBMIT_URL = "https://video.example.test/v1/videos"
JOB_URL = f"{SUBMIT_URL}/video-job-1"


def _routes(
    routes: dict[tuple[str, str], list[tuple[int, Any]]],
) -> tuple[list[httpx.Request], Callable[[httpx.Request], httpx.Response]]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        key = (request.method, str(request.url))
        responses = routes[key]
        status, body = responses.pop(0) if len(responses) > 1 else responses[0]
        return httpx.Response(status, json=body)

    return requests, handler


def _successful_job_routes() -> dict[tuple[str, str], list[tuple[int, Any]]]:
    return {
        ("POST", SUBMIT_URL): [(202, {"job": {"id": "video-job-1"}})],
        ("GET", JOB_URL): [
            (200, {"job": {"state": "queued", "progress": 0}}),
            (200, {"job": {"state": "processing", "progress": 45}}),
            (
                200,
                {
                    "job": {"state": "completed", "progress": 100},
                    "data": {
                        "videos": [
                            {
                                "file": {
                                    "url": "/files/clip-1.mp4",
                                    "content_type": "video/mp4",
                                    "width": 1280,
                                    "height": 720,
                                    "duration_seconds": 6,
                                    "fps": 30,
                                    "has_audio": True,
                                    "seed": 17,
                                    "thumbnail_url": "/thumbs/clip-1.jpg",
                                }
                            },
                            {
                                "file": {
                                    "url": "/files/clip-2.webm",
                                    "content_type": "video/webm",
                                    "width": 720,
                                    "height": 1280,
                                    "duration_seconds": 6,
                                    "fps": 30,
                                    "has_audio": False,
                                    "seed": 18,
                                    "thumbnail_url": "/thumbs/clip-2.jpg",
                                }
                            },
                        ]
                    },
                },
            ),
        ],
    }


def test_sample_async_video_spec_describes_itself() -> None:
    spec = load_http_spec(SAMPLE_SPEC)

    assert isinstance(spec, HttpProviderSpec)
    assert spec.name == "sample-video"
    assert spec.mode == "async"
    assert spec.is_async is True
    assert spec.capabilities == [CAPABILITY_VIDEO_GEN]
    assert spec.target_hosts == ("video.example.test",)
    assert spec.secret_names == ("api_key",)
    assert spec.secret_references == ("SAMPLE_VIDEO_API_KEY",)
    assert [operation.method for operation in spec.operations] == [
        "POST",
        "GET",
        "POST",
    ]
    price = spec.pricing[CAPABILITY_VIDEO_GEN]
    assert price.amount_usd == 0.35
    assert price.unit == "clip"
    metadata = spec.to_metadata()
    assert metadata.capabilities == [CAPABILITY_VIDEO_GEN]
    assert metadata.pricing_for(CAPABILITY_VIDEO_GEN) is not None
    spec.validate_hosts(["video.example.test"])


def test_sample_async_video_spec_passes_a_mock_queue() -> None:
    requests, handler = _routes(_successful_job_routes())
    provider = build_http_provider(
        SAMPLE_SPEC,
        secret_values={"api_key": API_KEY},
        transport=httpx.MockTransport(handler),
        poll_interval_seconds=0.001,
    )
    assert isinstance(provider, HttpSpecProvider)
    request = VideoRequest(
        prompt="  a calm ocean at sunrise  ",
        negative_prompt="  text overlays  ",
        duration_seconds=6.0,
        aspect_ratio="9:16",
        fps=30,
        seed=17,
        count=2,
        generate_audio=True,
    )

    response = cast(VideoResponse, asyncio.run(provider.generate(request)))

    submitted = requests[0]
    assert submitted.method == "POST"
    assert str(submitted.url) == SUBMIT_URL
    assert submitted.headers["authorization"] == f"Bearer {API_KEY}"
    assert submitted.headers["accept"] == "application/json"
    assert json.loads(submitted.content) == {
        "prompt": "a calm ocean at sunrise",
        "negative_prompt": "text overlays",
        "duration_seconds": 6.0,
        "aspect_ratio": "9:16",
        "fps": 30,
        "seed": 17,
        "count": 2,
        "generate_audio": True,
    }
    assert [request.method for request in requests] == ["POST", "GET", "GET", "GET"]
    assert [str(request.url) for request in requests[1:]] == [JOB_URL] * 3

    assert response.provider == "sample-video"
    assert response.model == "sample-video-1"
    assert response.request_id == "video-job-1"
    assert response.count == 2
    first = response.videos[0]
    assert first.url == "https://video.example.test/files/clip-1.mp4"
    assert first.format is VideoFormat.MP4
    assert (first.width, first.height) == (1280, 720)
    assert first.duration_seconds == 6.0
    assert first.fps == 30
    assert first.has_audio is True
    assert first.seed == 17
    assert first.thumbnail_url == "https://video.example.test/thumbs/clip-1.jpg"
    second = response.videos[1]
    assert second.url == "https://video.example.test/files/clip-2.webm"
    assert second.format is VideoFormat.WEBM
    assert (second.width, second.height) == (720, 1280)
    assert second.has_audio is False
    assert second.thumbnail_url == "https://video.example.test/thumbs/clip-2.jpg"
