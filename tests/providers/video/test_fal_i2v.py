"""fal image-to-video sends a first frame only when the capability is declared."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from aivia.core.errors import InvalidRequestError
from aivia.core.models.provider import ProviderMeta
from aivia.providers import (
    CAPABILITY_VIDEO_FIRST_FRAME as TOP_LEVEL_FIRST_FRAME,
)
from aivia.providers import (
    CAPABILITY_VIDEO_IMAGE_INPUT as TOP_LEVEL_IMAGE_INPUT,
)
from aivia.providers.protocols.media import MediaInput
from aivia.providers.protocols.video import (
    CAPABILITY_VIDEO_FIRST_FRAME,
    CAPABILITY_VIDEO_GEN,
    CAPABILITY_VIDEO_IMAGE_INPUT,
    VideoRequest,
)
from aivia.providers.video import (
    CAPABILITY_VIDEO_FIRST_FRAME as VIDEO_FIRST_FRAME,
)
from aivia.providers.video import FalVideoProvider

API_KEY = "test-key"
MODEL = "fal-ai/kling-video/v1.6/pro/image-to-video"
BASE_URL = "https://queue.fal.run"
ENDPOINT = f"{BASE_URL}/{MODEL}"
STATUS_URL = "https://queue.fal.run/fal-ai/kling-video/requests/req-fal-video-1/status"
RESPONSE_URL = "https://queue.fal.run/fal-ai/kling-video/requests/req-fal-video-1"
FRAME_URL = "https://example.com/frames/first.png"
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


def _routes() -> tuple[list[httpx.Request], Callable[[httpx.Request], httpx.Response]]:
    routes: dict[tuple[str, str], list[tuple[int, Any]]] = {
        ("POST", ENDPOINT): [(200, _fixture("video_submit"))],
        ("GET", STATUS_URL): [
            (200, _fixture("video_in_queue")),
            (200, _fixture("video_completed")),
        ],
        ("GET", RESPONSE_URL): [(200, _fixture("video_success"))],
    }
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        key = (request.method, str(request.url))
        responses = routes[key]
        status, body = responses.pop(0) if len(responses) > 1 else responses[0]
        return httpx.Response(status, json=body)

    return requests, handler


def _submitted_payload(
    request: VideoRequest,
    **provider_options: Any,
) -> dict[str, Any]:
    requests, handler = _routes()
    provider = _provider(handler, **provider_options)

    asyncio.run(provider.generate(request))

    return cast("dict[str, Any]", json.loads(requests[0].content))


def test_the_first_frame_capability_is_exported_and_declared() -> None:
    provider = _provider(lambda request: httpx.Response(200, json={}))

    assert CAPABILITY_VIDEO_FIRST_FRAME == "video_first_frame"
    assert CAPABILITY_VIDEO_IMAGE_INPUT == CAPABILITY_VIDEO_FIRST_FRAME
    assert TOP_LEVEL_FIRST_FRAME == CAPABILITY_VIDEO_FIRST_FRAME
    assert TOP_LEVEL_IMAGE_INPUT == CAPABILITY_VIDEO_FIRST_FRAME
    assert VIDEO_FIRST_FRAME == CAPABILITY_VIDEO_FIRST_FRAME
    assert provider.supports(CAPABILITY_VIDEO_FIRST_FRAME)
    assert CAPABILITY_VIDEO_FIRST_FRAME in provider.capabilities
    assert CAPABILITY_VIDEO_GEN in provider.capabilities


def test_a_first_frame_url_is_sent_as_the_image_url() -> None:
    payload = _submitted_payload(
        VideoRequest(
            prompt="  animate this still  ",
            first_frame=MediaInput(url=FRAME_URL, media_type="image/png"),
            duration_seconds=5.0,
        )
    )

    assert payload["image_url"] == FRAME_URL
    assert payload["prompt"] == "animate this still"
    assert payload["duration"] == 5.0


def test_an_inline_first_frame_is_encoded_as_a_data_url() -> None:
    frame = b"first-frame-bytes"
    encoded = base64.b64encode(frame).decode("ascii")

    payload = _submitted_payload(
        VideoRequest(
            prompt="animate this still",
            first_frame=MediaInput(data=frame, media_type="image/webp"),
        )
    )

    assert payload["image_url"] == f"data:image/webp;base64,{encoded}"


def test_an_inline_first_frame_defaults_to_png() -> None:
    frame = b"first-frame-bytes"
    encoded = base64.b64encode(frame).decode("ascii")

    payload = _submitted_payload(
        VideoRequest(
            prompt="animate this still",
            first_frame=MediaInput(data=frame),
        )
    )

    assert payload["image_url"] == f"data:image/png;base64,{encoded}"


def test_a_request_without_a_first_frame_omits_the_image_url() -> None:
    payload = _submitted_payload(VideoRequest(prompt="a wave"))

    assert "image_url" not in payload


def test_a_provider_without_the_capability_rejects_a_first_frame() -> None:
    class _TextOnlyFal(FalVideoProvider):
        default_metadata = ProviderMeta(
            name="fal-text-only",
            capabilities=[CAPABILITY_VIDEO_GEN],
        )

    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_fixture("video_submit"))

    provider = _TextOnlyFal(
        api_key=API_KEY,
        default_model=MODEL,
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


def test_a_non_image_first_frame_is_rejected() -> None:
    requests, handler = _routes()
    provider = _provider(handler)

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


def test_a_last_frame_is_rejected_instead_of_being_ignored() -> None:
    requests, handler = _routes()
    provider = _provider(handler)

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
