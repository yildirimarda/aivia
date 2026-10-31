"""Replicate and Veo video providers: predictions, operations and credentials."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any, cast

import httpx
import pytest

from aivia.core.errors import (
    ConfigError,
    ContentPolicyError,
    InvalidRequestError,
)
from aivia.core.secrets import SecretsLoader
from aivia.providers import ReplicateVideo as TopLevelReplicateVideo
from aivia.providers import VeoVideo as TopLevelVeoVideo
from aivia.providers import build_provider, provider_registry
from aivia.providers.protocols import CAPABILITY_VIDEO_GEN, VideoRequest
from aivia.providers.protocols.media import VideoFormat
from aivia.providers.video import (
    REPLICATE_DEFAULT_BASE_URL,
    ReplicateVideo,
    ReplicateVideoGen,
    ReplicateVideoGenProvider,
    ReplicateVideoProvider,
)
from aivia.providers.video.veo import (
    DEFAULT_API_VERSION,
    DEFAULT_BASE_URL,
    GOOGLE_API_URL,
    GOOGLE_DEFAULT_BASE_URL,
    GOOGLE_VEO_API_URL,
    GOOGLE_VEO_BASE_URL,
    VEO_API_URL,
    VEO_DEFAULT_BASE_URL,
    Veo,
    VeoProvider,
    VeoVideo,
    VeoVideoGen,
    VeoVideoGenProvider,
    VeoVideoProvider,
)

API_KEY = "test-key"
REPLICATE_MODEL_NAME = "owner/video-model"
REPLICATE_VERSION = "abc123def456"
REPLICATE_MODEL = f"{REPLICATE_MODEL_NAME}:{REPLICATE_VERSION}"
REPLICATE_ENDPOINT = f"{REPLICATE_DEFAULT_BASE_URL}/predictions"
PREDICTION_ID = "pred-video-1"
REPLICATE_GET_URL = f"{REPLICATE_DEFAULT_BASE_URL}/predictions/{PREDICTION_ID}"

VEO_MODEL = "veo-3.0-generate-001"
VEO_ENDPOINT = (
    f"{DEFAULT_BASE_URL}/{DEFAULT_API_VERSION}/models/{VEO_MODEL}"
    ":predictLongRunning"
)
OPERATION_NAME = f"models/{VEO_MODEL}/operations/op-1"
OPERATION_URL = f"{DEFAULT_BASE_URL}/{DEFAULT_API_VERSION}/{OPERATION_NAME}"
VIDEO_URL = "https://example.com/media/clip-1.mp4"
SECOND_VIDEO_URL = "https://example.com/media/clip-2.webm"


def _routes(
    routes: dict[tuple[str, str], list[tuple[int, Any]]],
) -> tuple[list[httpx.Request], Callable[[httpx.Request], httpx.Response]]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        key = (request.method, str(request.url))
        responses = routes[key]
        status, body = responses.pop(0) if len(responses) > 1 else responses[0]
        if isinstance(body, bytes):
            return httpx.Response(status, content=body)
        return httpx.Response(status, json=body)

    return requests, handler


def _replicate_routes() -> dict[tuple[str, str], list[tuple[int, Any]]]:
    return {
        ("POST", REPLICATE_ENDPOINT): [
            (
                200,
                {
                    "id": PREDICTION_ID,
                    "status": "starting",
                    "urls": {
                        "get": REPLICATE_GET_URL,
                        "cancel": f"{REPLICATE_GET_URL}/cancel",
                    },
                },
            )
        ],
        ("GET", REPLICATE_GET_URL): [
            (200, {"id": PREDICTION_ID, "status": "processing"}),
            (
                200,
                {
                    "id": PREDICTION_ID,
                    "status": "succeeded",
                    "input": {"prompt": "a calm ocean at sunrise", "seed": 7},
                    "output": [
                        {
                            "url": VIDEO_URL,
                            "content_type": "video/mp4",
                            "width": 1280,
                            "height": 720,
                            "duration": 5.0,
                            "fps": 24,
                            "has_audio": True,
                            "thumbnail_url": "https://example.com/media/clip-1.jpg",
                        },
                        {
                            "url": SECOND_VIDEO_URL,
                            "content_type": "video/webm",
                        },
                    ],
                },
            ),
        ],
    }


def _replicate_provider(
    handler: Callable[[httpx.Request], httpx.Response],
    **kwargs: Any,
) -> ReplicateVideoProvider:
    options: dict[str, Any] = {
        "api_key": API_KEY,
        "model": REPLICATE_MODEL,
        "transport": httpx.MockTransport(handler),
        "poll_interval_seconds": 0.001,
    }
    options.update(kwargs)
    return ReplicateVideoProvider(**options)


def _veo_provider(
    handler: Callable[[httpx.Request], httpx.Response],
    **kwargs: Any,
) -> VeoVideoProvider:
    options: dict[str, Any] = {
        "api_key": API_KEY,
        "default_model": VEO_MODEL,
        "transport": httpx.MockTransport(handler),
        "poll_interval_seconds": 0.001,
    }
    options.update(kwargs)
    return VeoVideoProvider(**options)


def _veo_routes() -> dict[tuple[str, str], list[tuple[int, Any]]]:
    return {
        ("POST", VEO_ENDPOINT): [(200, {"name": OPERATION_NAME, "done": False})],
        ("GET", OPERATION_URL): [
            (
                200,
                {
                    "name": OPERATION_NAME,
                    "done": False,
                    "metadata": {"progressPercent": 45},
                },
            ),
            (
                200,
                {
                    "name": OPERATION_NAME,
                    "done": True,
                    "response": {
                        "generateVideoResponse": {
                            "generatedSamples": [
                                {
                                    "video": {
                                        "uri": VIDEO_URL,
                                        "mimeType": "video/mp4",
                                    }
                                }
                            ]
                        }
                    },
                },
            ),
        ],
    }


def test_replicate_video_metadata_and_registration() -> None:
    provider = _replicate_provider(lambda request: httpx.Response(200, json={}))

    assert provider.name == "replicate"
    assert provider.capabilities == (CAPABILITY_VIDEO_GEN,)
    assert provider.supports(CAPABILITY_VIDEO_GEN)
    assert provider.base_url == REPLICATE_DEFAULT_BASE_URL
    assert provider.api_url == REPLICATE_DEFAULT_BASE_URL
    assert provider.default_model == REPLICATE_MODEL
    assert provider.model == REPLICATE_MODEL
    assert provider.version == REPLICATE_VERSION
    assert provider.endpoint == REPLICATE_ENDPOINT
    assert ReplicateVideo is ReplicateVideoProvider
    assert ReplicateVideoGen is ReplicateVideoProvider
    assert ReplicateVideoGenProvider is ReplicateVideoProvider
    assert TopLevelReplicateVideo is ReplicateVideoProvider
    assert provider_registry.resolve("replicate-video") is ReplicateVideoProvider
    assert provider_registry.resolve("replicate_video") is ReplicateVideoProvider


def test_replicate_video_submits_polls_and_maps_the_result() -> None:
    requests, handler = _routes(_replicate_routes())
    provider = _replicate_provider(handler)
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
    assert str(submitted.url) == REPLICATE_ENDPOINT
    assert submitted.headers["authorization"] == f"Bearer {API_KEY}"
    assert json.loads(submitted.content) == {
        "version": REPLICATE_VERSION,
        "input": {
            "prompt": "a calm ocean at sunrise",
            "num_outputs": 1,
            "duration": 5.0,
            "aspect_ratio": "16:9",
            "seed": 7,
            "output_format": "mp4",
            "generate_audio": True,
            "audio_prompt": "soft waves",
        },
    }
    assert [str(sent.url) for sent in requests[1:]] == [
        REPLICATE_GET_URL,
        REPLICATE_GET_URL,
    ]
    assert response.provider == "replicate"
    assert response.model == REPLICATE_MODEL
    assert response.request_id == PREDICTION_ID
    assert response.count == 2
    first = response.first
    assert first.url == VIDEO_URL
    assert first.format is VideoFormat.MP4
    assert (first.width, first.height) == (1280, 720)
    assert first.duration_seconds == 5.0
    assert first.fps == 24
    assert first.has_audio is True
    assert first.seed == 7
    assert first.thumbnail_url == "https://example.com/media/clip-1.jpg"
    assert response.videos[1].format is VideoFormat.WEBM
    assert response.metadata["job_id"]


def test_replicate_video_accepts_a_separate_version_option() -> None:
    requests, handler = _routes(_replicate_routes())
    provider = ReplicateVideoProvider(
        api_key=API_KEY,
        model=REPLICATE_MODEL_NAME,
        version=REPLICATE_VERSION,
        transport=httpx.MockTransport(handler),
        poll_interval_seconds=0.001,
    )

    asyncio.run(provider.generate(VideoRequest(prompt="a wave")))

    assert provider.default_model == REPLICATE_MODEL
    assert json.loads(requests[0].content)["version"] == REPLICATE_VERSION


def test_replicate_video_rejects_a_model_without_a_version() -> None:
    with pytest.raises(ConfigError):
        ReplicateVideoProvider(api_key=API_KEY, model=REPLICATE_MODEL_NAME)

    provider = ReplicateVideoProvider(
        api_key=API_KEY,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})),
    )
    with pytest.raises(InvalidRequestError):
        asyncio.run(
            provider.generate(
                VideoRequest(prompt="a wave", model=REPLICATE_MODEL_NAME)
            )
        )


def test_replicate_video_maps_a_failed_prediction_to_content_policy() -> None:
    routes = _replicate_routes()
    routes[("GET", REPLICATE_GET_URL)] = [
        (
            200,
            {
                "id": PREDICTION_ID,
                "status": "failed",
                "error": "The prompt was flagged as NSFW.",
            },
        )
    ]
    _, handler = _routes(routes)
    provider = _replicate_provider(handler)

    with pytest.raises(ContentPolicyError):
        asyncio.run(provider.generate(VideoRequest(prompt="blocked prompt")))


def test_replicate_video_treats_a_flagged_output_as_content_policy() -> None:
    routes = _replicate_routes()
    routes[("GET", REPLICATE_GET_URL)] = [
        (
            200,
            {
                "id": PREDICTION_ID,
                "status": "succeeded",
                "output": [],
                "has_nsfw_concepts": [True],
            },
        )
    ]
    _, handler = _routes(routes)
    provider = _replicate_provider(handler)

    with pytest.raises(ContentPolicyError):
        asyncio.run(provider.generate(VideoRequest(prompt="blocked prompt")))


def test_replicate_video_maps_an_error_response_to_content_policy() -> None:
    routes = _replicate_routes()
    routes[("POST", REPLICATE_ENDPOINT)] = [
        (422, {"detail": "The request was filtered by the content policy."})
    ]
    _, handler = _routes(routes)
    provider = _replicate_provider(handler)

    with pytest.raises(ContentPolicyError) as caught:
        asyncio.run(provider.generate(VideoRequest(prompt="blocked prompt")))

    assert caught.value.details["status"] == 422


def test_replicate_video_rejects_reference_frames() -> None:
    requests, handler = _routes(_replicate_routes())
    provider = _replicate_provider(handler)
    from aivia.providers.protocols.media import MediaInput

    with pytest.raises(InvalidRequestError) as caught:
        asyncio.run(
            provider.generate(
                VideoRequest(
                    prompt="animate this",
                    first_frame=MediaInput(url="https://example.com/frame.png"),
                )
            )
        )

    assert caught.value.details["field"] == "first_frame"
    assert requests == []


def test_replicate_video_downloads_results_when_configured() -> None:
    downloads = {VIDEO_URL: b"clip-one", SECOND_VIDEO_URL: b"clip-two"}
    _, base_handler = _routes(_replicate_routes())

    def handler(request: httpx.Request) -> httpx.Response:
        data = downloads.get(str(request.url))
        if data is not None:
            return httpx.Response(200, content=data)
        return base_handler(request)

    provider = _replicate_provider(handler, download_results=True)

    response = asyncio.run(provider.generate(VideoRequest(prompt="a wave")))

    assert [video.data for video in response.videos] == [b"clip-one", b"clip-two"]
    assert all(video.url is None for video in response.videos)


def test_replicate_video_requires_a_credential() -> None:
    with pytest.raises(ConfigError):
        ReplicateVideoProvider(model=REPLICATE_MODEL)


def test_veo_metadata_and_registration() -> None:
    provider = _veo_provider(lambda request: httpx.Response(200, json={}))

    assert provider.name == "veo"
    assert provider.capabilities == (CAPABILITY_VIDEO_GEN,)
    assert provider.supports(CAPABILITY_VIDEO_GEN)
    assert provider.base_url == DEFAULT_BASE_URL
    assert provider.api_url == DEFAULT_BASE_URL
    assert provider.api_version == DEFAULT_API_VERSION
    assert provider.default_model == VEO_MODEL
    assert provider.model == VEO_MODEL
    assert provider.endpoint == VEO_ENDPOINT
    assert GOOGLE_DEFAULT_BASE_URL == DEFAULT_BASE_URL
    assert GOOGLE_API_URL == DEFAULT_BASE_URL
    assert GOOGLE_VEO_BASE_URL == DEFAULT_BASE_URL
    assert GOOGLE_VEO_API_URL == DEFAULT_BASE_URL
    assert VEO_DEFAULT_BASE_URL == DEFAULT_BASE_URL
    assert VEO_API_URL == DEFAULT_BASE_URL
    assert Veo is VeoVideoProvider
    assert VeoProvider is VeoVideoProvider
    assert VeoVideo is VeoVideoProvider
    assert VeoVideoGen is VeoVideoProvider
    assert VeoVideoGenProvider is VeoVideoProvider
    assert TopLevelVeoVideo is VeoVideoProvider
    assert provider_registry.resolve("veo") is VeoVideoProvider
    assert provider_registry.resolve("veo-video") is VeoVideoProvider
    assert provider_registry.resolve("veo_video") is VeoVideoProvider
    assert provider_registry.resolve("google-veo") is VeoVideoProvider


def test_veo_generates_through_a_long_running_operation() -> None:
    requests, handler = _routes(_veo_routes())
    provider = _veo_provider(handler)
    request = VideoRequest(
        prompt="  a calm ocean at sunrise  ",
        duration_seconds=5.0,
        aspect_ratio="16:9",
        seed=7,
        count=1,
        generate_audio=True,
    )

    response = asyncio.run(provider.generate(request))

    submitted = requests[0]
    assert submitted.method == "POST"
    assert str(submitted.url) == VEO_ENDPOINT
    assert submitted.headers["x-goog-api-key"] == API_KEY
    assert json.loads(submitted.content) == {
        "instances": [{"prompt": "a calm ocean at sunrise"}],
        "parameters": {
            "sampleCount": 1,
            "aspectRatio": "16:9",
            "durationSeconds": 5.0,
            "seed": 7,
            "generateAudio": True,
        },
    }
    assert [str(sent.url) for sent in requests[1:]] == [OPERATION_URL, OPERATION_URL]
    assert response.provider == "veo"
    assert response.model == VEO_MODEL
    assert response.request_id == OPERATION_NAME
    assert response.count == 1
    video = response.first
    assert video.url == VIDEO_URL
    assert video.format is VideoFormat.MP4
    assert video.has_audio is True
    assert video.seed == 7
    assert response.metadata["job_id"]


def test_veo_authenticates_through_a_credential_reference() -> None:
    requests, handler = _routes(_veo_routes())
    loader = SecretsLoader(environ={"AIVA_GOOGLE_API_KEY": "referenced-key"})
    provider = _veo_provider(
        handler,
        api_key=None,
        api_key_ref="GOOGLE_API_KEY",
        secrets=loader,
    )

    asyncio.run(provider.generate(VideoRequest(prompt="a wave")))

    assert requests[0].headers["x-goog-api-key"] == "referenced-key"


def test_veo_builds_from_a_provider_config_secret_reference() -> None:
    requests, handler = _routes(_veo_routes())
    loader = SecretsLoader(environ={"AIVA_GOOGLE_API_KEY": "configured-key"})
    provider = cast(
        VeoVideoProvider,
        build_provider(
            {
                "type": "veo",
                "options": {
                    "default_model": VEO_MODEL,
                    "transport": httpx.MockTransport(handler),
                    "poll_interval_seconds": 0.001,
                },
                "secret_refs": {"api_key": "GOOGLE_API_KEY"},
            },
            "veo",
            secrets=loader,
        ),
    )

    asyncio.run(provider.generate(VideoRequest(prompt="a wave")))

    assert requests[0].headers["x-goog-api-key"] == "configured-key"


def test_veo_accepts_the_credential_ref_alias() -> None:
    requests, handler = _routes(_veo_routes())
    loader = SecretsLoader(environ={"AIVA_GOOGLE_API_KEY": "aliased-key"})
    provider = _veo_provider(
        handler,
        api_key=None,
        credential_ref="GOOGLE_API_KEY",
        secrets=loader,
    )

    asyncio.run(provider.generate(VideoRequest(prompt="a wave")))

    assert requests[0].headers["x-goog-api-key"] == "aliased-key"


def test_veo_reports_a_missing_credential_reference() -> None:
    with pytest.raises(ConfigError):
        VeoVideoProvider(
            api_key_ref="MISSING_KEY",
            secrets=SecretsLoader(environ={}),
            default_model=VEO_MODEL,
        )

    with pytest.raises(ConfigError):
        VeoVideoProvider(default_model=VEO_MODEL)


def test_veo_maps_operation_progress() -> None:
    routes = _veo_routes()
    _, handler = _routes(routes)
    provider = _veo_provider(handler)

    job = asyncio.run(provider.submit(VideoRequest(prompt="a wave")))
    backend = provider.helper.backend
    assert backend is not None

    running = asyncio.run(backend.poll(job.remote_id))
    assert running.state.value == "running"
    assert running.progress == pytest.approx(0.45)


def test_veo_maps_a_filtered_operation_to_content_policy() -> None:
    routes = _veo_routes()
    routes[("GET", OPERATION_URL)] = [
        (
            200,
            {
                "name": OPERATION_NAME,
                "done": True,
                "response": {
                    "raiMediaFilteredReasons": ["The prompt violates policy."]
                },
            },
        )
    ]
    _, handler = _routes(routes)
    provider = _veo_provider(handler)

    with pytest.raises(ContentPolicyError):
        asyncio.run(provider.generate(VideoRequest(prompt="blocked prompt")))


def test_veo_maps_a_failed_operation_to_content_policy() -> None:
    routes = _veo_routes()
    routes[("GET", OPERATION_URL)] = [
        (
            200,
            {
                "name": OPERATION_NAME,
                "done": True,
                "error": {
                    "code": 400,
                    "status": "FAILED_PRECONDITION",
                    "message": "The prompt was blocked by the safety system.",
                },
            },
        )
    ]
    _, handler = _routes(routes)
    provider = _veo_provider(handler)

    with pytest.raises(ContentPolicyError):
        asyncio.run(provider.generate(VideoRequest(prompt="blocked prompt")))


def test_veo_maps_a_failed_operation_to_invalid_request() -> None:
    routes = _veo_routes()
    routes[("GET", OPERATION_URL)] = [
        (
            200,
            {
                "name": OPERATION_NAME,
                "done": True,
                "error": {"code": 500, "message": "internal failure"},
            },
        )
    ]
    _, handler = _routes(routes)
    provider = _veo_provider(handler)

    with pytest.raises(InvalidRequestError):
        asyncio.run(provider.generate(VideoRequest(prompt="a wave")))


def test_veo_downloads_results_when_configured() -> None:
    downloads = {VIDEO_URL: b"clip-one"}
    _, base_handler = _routes(_veo_routes())

    def handler(request: httpx.Request) -> httpx.Response:
        data = downloads.get(str(request.url))
        if data is not None:
            return httpx.Response(200, content=data)
        return base_handler(request)

    provider = _veo_provider(handler, download_results=True)

    response = asyncio.run(provider.generate(VideoRequest(prompt="a wave")))

    assert response.first.data == b"clip-one"
    assert response.first.url is None


def test_veo_requires_a_model_id() -> None:
    provider = VeoVideoProvider(
        api_key=API_KEY,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})),
    )

    with pytest.raises(InvalidRequestError):
        asyncio.run(provider.generate(VideoRequest(prompt="a wave")))


def test_veo_rejects_reference_frames() -> None:
    requests, handler = _routes(_veo_routes())
    provider = _veo_provider(handler)
    from aivia.providers.protocols.media import MediaInput

    with pytest.raises(InvalidRequestError) as caught:
        asyncio.run(
            provider.generate(
                VideoRequest(
                    prompt="animate this",
                    first_frame=MediaInput(url="https://example.com/frame.png"),
                )
            )
        )

    assert caught.value.details["field"] == "first_frame"
    assert requests == []


def test_veo_resolves_operation_names_and_absolute_urls() -> None:
    provider = _veo_provider(lambda request: httpx.Response(200, json={}))

    assert provider.operation_url(OPERATION_NAME) == OPERATION_URL
    assert provider.operation_url(OPERATION_URL) == OPERATION_URL
    assert provider.operation_url(f"{DEFAULT_API_VERSION}/{OPERATION_NAME}") == (
        OPERATION_URL
    )
