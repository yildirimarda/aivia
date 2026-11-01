"""Replicate and Veo accept a first frame only behind the frame capability."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Callable
from typing import Any, cast

import httpx
import pytest

from aivia.core.errors import InvalidRequestError, ValidationError
from aivia.core.models.provider import ProviderMeta
from aivia.providers import (
    CAPABILITY_VIDEO_FIRST_FRAME as TOP_LEVEL_FIRST_FRAME,
)
from aivia.providers.protocols.media import MediaInput
from aivia.providers.protocols.video import (
    CAPABILITY_VIDEO_FIRST_FRAME,
    CAPABILITY_VIDEO_GEN,
    VideoRequest,
)
from aivia.providers.video import (
    CAPABILITY_VIDEO_FIRST_FRAME as VIDEO_FIRST_FRAME,
)
from aivia.providers.video import (
    REPLICATE_DEFAULT_BASE_URL,
    ReplicateVideoProvider,
    VeoVideoProvider,
)
from aivia.providers.video.veo import DEFAULT_API_VERSION, DEFAULT_BASE_URL

API_KEY = "test-key"
FRAME_URL = "https://example.com/frames/first.png"
FRAME_BYTES = b"veo-http-first-frame"
FRAME_CONTENT_TYPE = {"content-type": "image/png"}

REPLICATE_MODEL = "owner/video-model:abc123def456"
REPLICATE_ENDPOINT = f"{REPLICATE_DEFAULT_BASE_URL}/predictions"
PREDICTION_ID = "pred-frames-1"
REPLICATE_GET_URL = f"{REPLICATE_DEFAULT_BASE_URL}/predictions/{PREDICTION_ID}"
VIDEO_URL = "https://example.com/media/clip.mp4"

VEO_MODEL = "veo-3.0-generate-001"
VEO_ENDPOINT = (
    f"{DEFAULT_BASE_URL}/{DEFAULT_API_VERSION}/models/{VEO_MODEL}:predictLongRunning"
)
OPERATION_NAME = f"models/{VEO_MODEL}/operations/op-frames-1"
OPERATION_URL = f"{DEFAULT_BASE_URL}/{DEFAULT_API_VERSION}/{OPERATION_NAME}"


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
        if isinstance(body, tuple):
            content, headers = body
            return httpx.Response(status, content=content, headers=headers)
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
                    "urls": {"get": REPLICATE_GET_URL},
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
                    "output": [{"url": VIDEO_URL, "content_type": "video/mp4"}],
                },
            ),
        ],
    }


def _veo_routes() -> dict[tuple[str, str], list[tuple[int, Any]]]:
    return {
        ("GET", FRAME_URL): [(200, (FRAME_BYTES, FRAME_CONTENT_TYPE))],
        ("POST", VEO_ENDPOINT): [(200, {"name": OPERATION_NAME, "done": False})],
        ("GET", OPERATION_URL): [
            (200, {"name": OPERATION_NAME, "done": False}),
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


def _replicate(
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


def _veo(
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


def _replicate_payload(
    request: VideoRequest,
    **provider_options: Any,
) -> dict[str, Any]:
    requests, handler = _routes(_replicate_routes())
    provider = _replicate(handler, **provider_options)

    asyncio.run(provider.generate(request))

    return cast("dict[str, Any]", json.loads(requests[0].content))


def _veo_payload(
    request: VideoRequest,
    **provider_options: Any,
) -> dict[str, Any]:
    requests, handler = _routes(_veo_routes())
    provider = _veo(handler, **provider_options)

    asyncio.run(provider.generate(request))

    post = next(
        candidate
        for candidate in requests
        if candidate.method == "POST" and str(candidate.url) == VEO_ENDPOINT
    )
    return cast("dict[str, Any]", json.loads(post.content))


def test_the_first_frame_capability_is_exported_and_declared() -> None:
    replicate = _replicate(lambda request: httpx.Response(200, json={}))
    veo = _veo(lambda request: httpx.Response(200, json={}))

    assert CAPABILITY_VIDEO_FIRST_FRAME == "video_first_frame"
    assert TOP_LEVEL_FIRST_FRAME == CAPABILITY_VIDEO_FIRST_FRAME
    assert VIDEO_FIRST_FRAME == CAPABILITY_VIDEO_FIRST_FRAME
    for provider in (replicate, veo):
        assert provider.supports(CAPABILITY_VIDEO_FIRST_FRAME)
        assert provider.supports(CAPABILITY_VIDEO_GEN)
        assert CAPABILITY_VIDEO_FIRST_FRAME in provider.capabilities


def test_replicate_sends_a_first_frame_url_as_the_image_input() -> None:
    payload = _replicate_payload(
        VideoRequest(
            prompt="animate this still",
            first_frame=MediaInput(url=FRAME_URL, media_type="image/png"),
        )
    )

    assert payload["input"]["image"] == FRAME_URL


def test_replicate_encodes_an_inline_first_frame_as_a_data_url() -> None:
    frame = b"replicate-first-frame"
    encoded = base64.b64encode(frame).decode("ascii")

    payload = _replicate_payload(
        VideoRequest(
            prompt="animate this still",
            first_frame=MediaInput(data=frame, media_type="image/webp"),
        )
    )

    assert payload["input"]["image"] == f"data:image/webp;base64,{encoded}"


def test_replicate_inline_first_frame_defaults_to_png() -> None:
    frame = b"replicate-first-frame"
    encoded = base64.b64encode(frame).decode("ascii")

    payload = _replicate_payload(
        VideoRequest(
            prompt="animate this still",
            first_frame=MediaInput(data=frame),
        )
    )

    assert payload["input"]["image"] == f"data:image/png;base64,{encoded}"


def test_replicate_first_frame_field_is_configurable() -> None:
    payload = _replicate_payload(
        VideoRequest(
            prompt="animate this still",
            first_frame=MediaInput(url=FRAME_URL),
        ),
        first_frame_field="first_frame_image",
    )

    assert payload["input"]["first_frame_image"] == FRAME_URL
    assert "image" not in payload["input"]


def test_replicate_first_frame_field_can_come_from_request_metadata() -> None:
    payload = _replicate_payload(
        VideoRequest(
            prompt="animate this still",
            first_frame=MediaInput(url=FRAME_URL),
            metadata={"first_frame_field": "start_image"},
        )
    )

    assert payload["input"]["start_image"] == FRAME_URL


def test_replicate_omits_the_image_input_without_a_first_frame() -> None:
    payload = _replicate_payload(VideoRequest(prompt="a wave"))

    assert "image" not in payload["input"]


def test_replicate_rejects_a_first_frame_without_the_capability() -> None:
    class _TextOnlyReplicate(ReplicateVideoProvider):
        default_metadata = ProviderMeta(
            name="replicate-text-only",
            capabilities=[CAPABILITY_VIDEO_GEN],
        )

    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={})

    provider = _TextOnlyReplicate(
        api_key=API_KEY,
        model=REPLICATE_MODEL,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(InvalidRequestError) as caught:
        asyncio.run(
            provider.generate(
                VideoRequest(
                    prompt="animate this still",
                    first_frame=MediaInput(url=FRAME_URL),
                )
            )
        )

    assert caught.value.details["required_capability"] == CAPABILITY_VIDEO_FIRST_FRAME
    assert requests == []


def test_replicate_rejects_a_non_image_first_frame() -> None:
    requests, handler = _routes(_replicate_routes())
    provider = _replicate(handler)

    with pytest.raises(InvalidRequestError) as caught:
        asyncio.run(
            provider.generate(
                VideoRequest(
                    prompt="animate this still",
                    first_frame=MediaInput(
                        data=b"not-an-image", media_type="video/mp4"
                    ),
                )
            )
        )

    assert caught.value.details["field"] == "first_frame"
    assert requests == []


def test_replicate_rejects_a_last_frame_instead_of_ignoring_it() -> None:
    requests, handler = _routes(_replicate_routes())
    provider = _replicate(handler)

    with pytest.raises(InvalidRequestError) as caught:
        asyncio.run(
            provider.generate(
                VideoRequest(
                    prompt="animate this still",
                    last_frame=MediaInput(url=FRAME_URL),
                )
            )
        )

    assert caught.value.details["field"] == "last_frame"
    assert requests == []


def test_veo_sends_an_inline_first_frame_as_base64_image() -> None:
    frame = b"veo-first-frame"
    encoded = base64.b64encode(frame).decode("ascii")

    payload = _veo_payload(
        VideoRequest(
            prompt="animate this still",
            first_frame=MediaInput(data=frame, media_type="image/jpeg"),
        )
    )

    image = payload["instances"][0]["image"]
    assert image == {"bytesBase64Encoded": encoded, "mimeType": "image/jpeg"}


def test_veo_inline_first_frame_defaults_to_png() -> None:
    frame = b"veo-first-frame"
    encoded = base64.b64encode(frame).decode("ascii")

    payload = _veo_payload(
        VideoRequest(
            prompt="animate this still",
            first_frame=MediaInput(data=frame),
        )
    )

    image = payload["instances"][0]["image"]
    assert image == {"bytesBase64Encoded": encoded, "mimeType": "image/png"}


def test_veo_downloads_an_http_first_frame_and_sends_it_inline() -> None:
    encoded = base64.b64encode(FRAME_BYTES).decode("ascii")

    payload = _veo_payload(
        VideoRequest(
            prompt="animate this still",
            first_frame=MediaInput(url=FRAME_URL, media_type="image/png"),
        )
    )

    assert payload["instances"][0]["image"] == {
        "bytesBase64Encoded": encoded,
        "mimeType": "image/png",
    }
    assert "gcsUri" not in payload["instances"][0]["image"]


def test_veo_downloads_an_http_first_frame_before_submitting() -> None:
    requests, handler = _routes(_veo_routes())
    provider = _veo(handler)

    asyncio.run(
        provider.generate(
            VideoRequest(
                prompt="animate this still",
                first_frame=MediaInput(url=FRAME_URL, media_type="image/png"),
            )
        )
    )

    frame_index = next(
        index
        for index, request in enumerate(requests)
        if request.method == "GET" and str(request.url) == FRAME_URL
    )
    post_index = next(
        index
        for index, request in enumerate(requests)
        if request.method == "POST" and str(request.url) == VEO_ENDPOINT
    )
    assert frame_index < post_index


def test_veo_infers_a_downloaded_frame_type_from_the_response() -> None:
    requests, handler = _routes(_veo_routes())

    def _response(request: httpx.Request) -> httpx.Response:
        if str(request.url) == FRAME_URL:
            return httpx.Response(
                200,
                content=FRAME_BYTES,
                headers={"content-type": "image/webp; charset=binary"},
            )
        return handler(request)

    provider = _veo(_response)

    asyncio.run(
        provider.generate(
            VideoRequest(
                prompt="animate this still",
                first_frame=MediaInput(url=FRAME_URL),
            )
        )
    )

    post = next(
        candidate
        for candidate in requests
        if candidate.method == "POST" and str(candidate.url) == VEO_ENDPOINT
    )
    payload = cast("dict[str, Any]", json.loads(post.content))
    assert payload["instances"][0]["image"]["mimeType"] == "image/webp"


def test_veo_keeps_an_explicit_first_frame_media_type() -> None:
    encoded = base64.b64encode(FRAME_BYTES).decode("ascii")

    payload = _veo_payload(
        VideoRequest(
            prompt="animate this still",
            first_frame=MediaInput(url=FRAME_URL, media_type="image/jpeg"),
        )
    )

    assert payload["instances"][0]["image"] == {
        "bytesBase64Encoded": encoded,
        "mimeType": "image/jpeg",
    }


def test_veo_caps_a_downloaded_first_frame() -> None:
    requests, handler = _routes(_veo_routes())
    provider = _veo(handler, max_download_bytes=4)

    with pytest.raises(ValidationError) as caught:
        asyncio.run(
            provider.generate(
                VideoRequest(
                    prompt="animate this still",
                    first_frame=MediaInput(url=FRAME_URL, media_type="image/png"),
                )
            )
        )

    assert caught.value.details["limit"] == 4
    assert not any(
        request.method == "POST" and str(request.url) == VEO_ENDPOINT
        for request in requests
    )


def test_veo_reports_a_failed_first_frame_download() -> None:
    requests, handler = _routes(_veo_routes())

    def _response(request: httpx.Request) -> httpx.Response:
        if str(request.url) == FRAME_URL:
            return httpx.Response(404, json={"error": "missing"})
        return handler(request)

    provider = _veo(_response)

    with pytest.raises(InvalidRequestError) as caught:
        asyncio.run(
            provider.generate(
                VideoRequest(
                    prompt="animate this still",
                    first_frame=MediaInput(url=FRAME_URL, media_type="image/png"),
                )
            )
        )

    assert caught.value.details["status"] == 404
    assert not any(
        request.method == "POST" and str(request.url) == VEO_ENDPOINT
        for request in requests
    )


def test_veo_omits_the_image_without_a_first_frame() -> None:
    payload = _veo_payload(VideoRequest(prompt="a wave"))

    assert "image" not in payload["instances"][0]


def test_veo_rejects_a_first_frame_without_the_capability() -> None:
    class _TextOnlyVeo(VeoVideoProvider):
        default_metadata = ProviderMeta(
            name="veo-text-only",
            capabilities=[CAPABILITY_VIDEO_GEN],
        )

    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={})

    provider = _TextOnlyVeo(
        api_key=API_KEY,
        default_model=VEO_MODEL,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(InvalidRequestError) as caught:
        asyncio.run(
            provider.generate(
                VideoRequest(
                    prompt="animate this still",
                    first_frame=MediaInput(url=FRAME_URL),
                )
            )
        )

    assert caught.value.details["required_capability"] == CAPABILITY_VIDEO_FIRST_FRAME
    assert requests == []


def test_veo_rejects_a_non_image_first_frame() -> None:
    requests, handler = _routes(_veo_routes())
    provider = _veo(handler)

    with pytest.raises(InvalidRequestError) as caught:
        asyncio.run(
            provider.generate(
                VideoRequest(
                    prompt="animate this still",
                    first_frame=MediaInput(
                        data=b"not-an-image", media_type="video/mp4"
                    ),
                )
            )
        )

    assert caught.value.details["field"] == "first_frame"
    assert requests == []


def test_veo_rejects_a_last_frame_instead_of_ignoring_it() -> None:
    requests, handler = _routes(_veo_routes())
    provider = _veo(handler)

    with pytest.raises(InvalidRequestError) as caught:
        asyncio.run(
            provider.generate(
                VideoRequest(
                    prompt="animate this still",
                    last_frame=MediaInput(url=FRAME_URL),
                )
            )
        )

    assert caught.value.details["field"] == "last_frame"
    assert requests == []
