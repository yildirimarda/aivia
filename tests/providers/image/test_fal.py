"""fal image provider: queue submission, polling, result mapping and filters."""

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
from aivia.providers import Fal as TopLevelFal
from aivia.providers import provider_registry
from aivia.providers.image import (
    DEFAULT_BASE_URL,
    FAL_DEFAULT_BASE_URL,
    Fal,
    FalImageProvider,
    FalProvider,
)
from aivia.providers.protocols import CAPABILITY_IMAGE_GEN, ImageFormat, ImageRequest

API_KEY = "test-key"
MODEL = "fal-ai/flux/dev"
ENDPOINT = f"{DEFAULT_BASE_URL}/{MODEL}"
STATUS_URL = "https://queue.fal.run/fal-ai/flux/requests/req-fal-1/status"
RESPONSE_URL = "https://queue.fal.run/fal-ai/flux/requests/req-fal-1"
FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "providers" / "fal"


def _fixture(name: str) -> dict[str, Any]:
    return cast(
        "dict[str, Any]",
        json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8")),
    )


def _provider(
    handler: Callable[[httpx.Request], httpx.Response],
    **kwargs: Any,
) -> FalImageProvider:
    return FalImageProvider(
        api_key=API_KEY,
        default_model=MODEL,
        transport=httpx.MockTransport(handler),
        poll_interval_seconds=0.001,
        **kwargs,
    )


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
        ("POST", ENDPOINT): [(200, _fixture("submit"))],
        ("GET", STATUS_URL): [
            (200, _fixture("in_progress")),
            (200, _fixture("completed")),
        ],
        ("GET", RESPONSE_URL): [(200, _fixture("success"))],
    }


def test_metadata_and_registration_describe_the_provider() -> None:
    provider = _provider(lambda request: httpx.Response(200, json={}))

    assert provider.name == "fal"
    assert provider.capabilities == (CAPABILITY_IMAGE_GEN,)
    assert provider.supports(CAPABILITY_IMAGE_GEN)
    assert provider.base_url == DEFAULT_BASE_URL
    assert provider.default_model == MODEL
    assert provider.endpoint == ENDPOINT
    assert FAL_DEFAULT_BASE_URL == DEFAULT_BASE_URL
    assert FalProvider is FalImageProvider
    assert Fal is FalImageProvider
    assert TopLevelFal is FalImageProvider
    assert provider_registry.resolve("fal") is FalImageProvider


def test_generate_submits_polls_and_maps_the_success_fixture() -> None:
    requests, handler = _routes(_success_routes())
    provider = _provider(handler)
    request = ImageRequest(
        prompt="  a calm ocean at sunrise  ",
        width=1024,
        height=768,
        count=2,
        seed=7,
        format=ImageFormat.PNG,
    )

    response = asyncio.run(provider.generate(request))

    submitted = requests[0]
    assert submitted.method == "POST"
    assert str(submitted.url) == ENDPOINT
    assert submitted.headers["authorization"] == f"Key {API_KEY}"
    assert json.loads(submitted.content) == {
        "prompt": "a calm ocean at sunrise",
        "num_images": 2,
        "seed": 7,
        "image_size": {"width": 1024, "height": 768},
        "output_format": "png",
    }
    assert [str(sent.url) for sent in requests[1:]] == [
        STATUS_URL,
        STATUS_URL,
        RESPONSE_URL,
    ]
    assert response.provider == "fal"
    assert response.model == MODEL
    assert response.request_id == "req-fal-1"
    assert response.count == 2
    assert response.images[0].url == "https://fal.media/files/success-1.png"
    assert response.images[0].format is ImageFormat.PNG
    assert response.images[0].width == 1024
    assert response.images[0].height == 768
    assert response.images[0].seed == 7
    assert response.images[1].url == "https://fal.media/files/success-2.png"


def test_generate_derives_queue_urls_when_submit_omits_them() -> None:
    routes = _success_routes()
    routes[("POST", ENDPOINT)] = [(200, {"request_id": "req-fal-1"})]
    requests, handler = _routes(routes)
    provider = FalImageProvider(
        api_key=API_KEY,
        transport=httpx.MockTransport(handler),
        poll_interval_seconds=0.001,
    )

    asyncio.run(provider.generate(ImageRequest(prompt="a wave", model=MODEL)))

    assert [str(sent.url) for sent in requests[1:]] == [
        STATUS_URL,
        STATUS_URL,
        RESPONSE_URL,
    ]


def test_generate_treats_a_flagged_result_as_content_policy() -> None:
    routes = _success_routes()
    routes[("GET", RESPONSE_URL)] = [
        (
            200,
            {
                "images": [],
                "has_nsfw_concepts": [True],
            },
        )
    ]
    _, handler = _routes(routes)
    provider = _provider(handler)

    with pytest.raises(ContentPolicyError):
        asyncio.run(provider.generate(ImageRequest(prompt="blocked prompt")))


def test_generate_maps_a_filtered_result_to_content_policy() -> None:
    routes = _success_routes()
    routes[("GET", RESPONSE_URL)] = [(422, _fixture("filtered"))]
    requests, handler = _routes(routes)
    provider = _provider(handler)

    with pytest.raises(ContentPolicyError) as caught:
        asyncio.run(provider.generate(ImageRequest(prompt="blocked prompt")))

    assert "filtered" in str(caught.value).lower()
    assert caught.value.details["status"] == 422
    assert str(requests[-1].url) == RESPONSE_URL


def test_generate_maps_a_failed_queue_status_to_content_policy() -> None:
    routes = _success_routes()
    routes[("GET", STATUS_URL)] = [
        (200, {"status": "FAILED", "error": "The prompt was flagged as NSFW."})
    ]
    requests, handler = _routes(routes)
    provider = _provider(handler)

    with pytest.raises(ContentPolicyError):
        asyncio.run(provider.generate(ImageRequest(prompt="blocked prompt")))

    assert str(requests[-1].url) == STATUS_URL


def test_generate_requires_a_model_id() -> None:
    provider = FalImageProvider(
        api_key=API_KEY,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})),
    )

    with pytest.raises(InvalidRequestError):
        asyncio.run(provider.generate(ImageRequest(prompt="a wave")))


def test_generate_accepts_the_model_on_the_request() -> None:
    requests, handler = _routes(_success_routes())
    provider = FalImageProvider(
        api_key=API_KEY,
        transport=httpx.MockTransport(handler),
        poll_interval_seconds=0.001,
    )

    asyncio.run(provider.generate(ImageRequest(prompt="a wave", model=MODEL)))

    assert str(requests[0].url) == ENDPOINT


def test_generate_maps_aspect_ratio_and_optional_options() -> None:
    requests, handler = _routes(_success_routes())
    provider = _provider(handler)

    asyncio.run(
        provider.generate(
            ImageRequest(
                prompt="a wave",
                aspect_ratio="16:9",
                negative_prompt="text",
                quality="high",
                style="photographic",
            )
        )
    )

    body = json.loads(requests[0].content)
    assert body["image_size"] == "landscape_16_9"
    assert body["negative_prompt"] == "text"
    assert body["quality"] == "high"
    assert body["style"] == "photographic"


def test_a_missing_api_key_is_a_config_error() -> None:
    with pytest.raises(ConfigError):
        FalImageProvider()
