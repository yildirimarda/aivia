"""fal video provider: queue states, progress, timeout and result mapping."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from aivia.core.errors import (
    ConfigError,
    ContentPolicyError,
    InvalidRequestError,
)
from aivia.providers import FalVideo as TopLevelFalVideo
from aivia.providers import provider_registry
from aivia.providers.media import AsyncJobState, AsyncJobTimeoutError
from aivia.providers.protocols import CAPABILITY_VIDEO_GEN, VideoRequest
from aivia.providers.protocols.media import VideoFormat
from aivia.providers.video import (
    DEFAULT_BASE_URL,
    FAL_DEFAULT_BASE_URL,
    Fal,
    FalVideo,
    FalVideoGen,
    FalVideoGenProvider,
    FalVideoProvider,
)

API_KEY = "test-key"
MODEL = "fal-ai/kling-video/v1.6/pro/text-to-video"
ENDPOINT = f"{DEFAULT_BASE_URL}/{MODEL}"
STATUS_URL = "https://queue.fal.run/fal-ai/kling-video/requests/req-fal-video-1/status"
RESPONSE_URL = "https://queue.fal.run/fal-ai/kling-video/requests/req-fal-video-1"
FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "providers" / "fal"


def _fixture(name: str) -> dict[str, Any]:
    return cast(
        "dict[str, Any]",
        json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8")),
    )


def _provider(
    handler: Callable[[httpx.Request], httpx.Response],
    **kwargs: Any,
) -> FalVideoProvider:
    options: dict[str, Any] = {
        "api_key": API_KEY,
        "default_model": MODEL,
        "transport": httpx.MockTransport(handler),
        "poll_interval_seconds": 0.001,
    }
    options.update(kwargs)
    return FalVideoProvider(**options)


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


def _success_routes() -> dict[tuple[str, str], list[tuple[int, Any]]]:
    return {
        ("POST", ENDPOINT): [(200, _fixture("video_submit"))],
        ("GET", STATUS_URL): [
            (200, _fixture("video_in_queue")),
            (200, _fixture("video_in_progress")),
            (200, _fixture("video_completed")),
        ],
        ("GET", RESPONSE_URL): [(200, _fixture("video_success"))],
    }


def test_metadata_and_registration_describe_the_video_provider() -> None:
    provider = _provider(lambda request: httpx.Response(200, json={}))

    assert provider.name == "fal"
    assert provider.capabilities == (CAPABILITY_VIDEO_GEN,)
    assert provider.supports(CAPABILITY_VIDEO_GEN)
    assert provider.base_url == DEFAULT_BASE_URL
    assert provider.default_model == MODEL
    assert provider.endpoint == ENDPOINT
    assert FAL_DEFAULT_BASE_URL == DEFAULT_BASE_URL
    assert Fal is FalVideoProvider
    assert FalVideo is FalVideoProvider
    assert FalVideoGen is FalVideoProvider
    assert FalVideoGenProvider is FalVideoProvider
    assert TopLevelFalVideo is FalVideoProvider
    assert provider_registry.resolve("fal-video") is FalVideoProvider
    assert provider_registry.resolve("fal_video") is FalVideoProvider


def test_generate_submits_polls_and_maps_the_success_fixture() -> None:
    requests, handler = _routes(_success_routes())
    provider = _provider(handler)
    request = VideoRequest(
        prompt="  a calm ocean at sunrise  ",
        duration_seconds=5.0,
        aspect_ratio="16:9",
        seed=7,
        count=1,
        generate_audio=True,
        audio_prompt="  soft waves  ",
        format=VideoFormat.MP4,
    )

    response = asyncio.run(provider.generate(request))

    submitted = requests[0]
    assert submitted.method == "POST"
    assert str(submitted.url) == ENDPOINT
    assert submitted.headers["authorization"] == f"Key {API_KEY}"
    assert json.loads(submitted.content) == {
        "prompt": "a calm ocean at sunrise",
        "num_videos": 1,
        "seed": 7,
        "duration": 5.0,
        "aspect_ratio": "16:9",
        "output_format": "mp4",
        "generate_audio": True,
        "audio_prompt": "soft waves",
    }
    assert [str(sent.url) for sent in requests[1:]] == [
        STATUS_URL,
        STATUS_URL,
        STATUS_URL,
        RESPONSE_URL,
    ]
    assert response.provider == "fal"
    assert response.model == MODEL
    assert response.request_id == "req-fal-video-1"
    assert response.count == 1
    video = response.first
    assert video.url == "https://fal.media/files/clip-1.mp4"
    assert video.format is VideoFormat.MP4
    assert (video.width, video.height) == (1280, 720)
    assert video.duration_seconds == 5.0
    assert video.fps == 24
    assert video.has_audio is True
    assert video.seed == 7
    assert video.thumbnail_url == "https://fal.media/files/clip-1.jpg"
    assert response.metadata["job_id"]


def test_queue_states_and_poll_progress_are_reported() -> None:
    routes = _success_routes()
    _, handler = _routes(routes)
    provider = _provider(handler)

    job = asyncio.run(provider.submit(VideoRequest(prompt="a wave")))
    backend = provider.helper.backend
    assert backend is not None

    queued = asyncio.run(backend.poll(job.remote_id))
    assert queued.state is AsyncJobState.PENDING
    assert queued.progress is None

    running = asyncio.run(backend.poll(job.remote_id))
    assert running.state is AsyncJobState.RUNNING
    assert running.progress == pytest.approx(0.45)

    completed = asyncio.run(backend.poll(job.remote_id))
    assert completed.state is AsyncJobState.SUCCEEDED
    assert completed.progress == 1.0
    assert completed.result_urls == ("https://fal.media/files/clip-1.mp4",)


def test_generate_times_out_a_job_that_never_finishes() -> None:
    routes = {
        ("POST", ENDPOINT): [(200, _fixture("video_submit"))],
        ("GET", STATUS_URL): [(200, _fixture("video_in_progress"))],
    }
    _, handler = _routes(routes)
    now = [0.0]

    def clock() -> float:
        return now[0]

    async def sleep(seconds: float) -> None:
        now[0] += seconds

    provider = _provider(
        handler,
        clock=clock,
        sleep=sleep,
        poll_interval_seconds=1.0,
    )

    with pytest.raises(AsyncJobTimeoutError) as caught:
        asyncio.run(
            provider.generate(VideoRequest(prompt="a slow wave", timeout_seconds=3.0))
        )

    assert caught.value.job is not None
    assert caught.value.job.state is AsyncJobState.TIMED_OUT
    assert caught.value.job.progress == pytest.approx(0.45)
    assert caught.value.details["state"] == "timed_out"


def test_generate_derives_queue_urls_when_submit_omits_them() -> None:
    routes = _success_routes()
    routes[("POST", ENDPOINT)] = [(200, {"request_id": "req-fal-video-1"})]
    requests, handler = _routes(routes)
    provider = FalVideoProvider(
        api_key=API_KEY,
        transport=httpx.MockTransport(handler),
        poll_interval_seconds=0.001,
    )

    asyncio.run(provider.generate(VideoRequest(prompt="a wave", model=MODEL)))

    assert [str(sent.url) for sent in requests[1:]] == [
        STATUS_URL,
        STATUS_URL,
        STATUS_URL,
        RESPONSE_URL,
    ]


def test_generate_maps_a_failed_queue_status_to_content_policy() -> None:
    routes = _success_routes()
    routes[("GET", STATUS_URL)] = [(200, _fixture("video_failed"))]
    _, handler = _routes(routes)
    provider = _provider(handler)

    with pytest.raises(ContentPolicyError):
        asyncio.run(provider.generate(VideoRequest(prompt="blocked prompt")))


def test_generate_maps_a_filtered_result_to_content_policy() -> None:
    routes = _success_routes()
    routes[("GET", RESPONSE_URL)] = [(422, _fixture("filtered"))]
    _, handler = _routes(routes)
    provider = _provider(handler)

    with pytest.raises(ContentPolicyError) as caught:
        asyncio.run(provider.generate(VideoRequest(prompt="blocked prompt")))

    assert "filtered" in str(caught.value).lower()


def test_generate_requires_a_model_id() -> None:
    provider = FalVideoProvider(
        api_key=API_KEY,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})),
    )

    with pytest.raises(InvalidRequestError):
        asyncio.run(provider.generate(VideoRequest(prompt="a wave")))


def test_a_missing_api_key_is_a_config_error() -> None:
    with pytest.raises(ConfigError):
        FalVideoProvider()
