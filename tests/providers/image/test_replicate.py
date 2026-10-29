"""Replicate image provider: versioned predictions, polling and filters."""

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
from aivia.providers import Replicate as TopLevelReplicate
from aivia.providers import provider_registry
from aivia.providers.image import (
    Replicate,
    ReplicateImageProvider,
    ReplicateProvider,
)
from aivia.providers.image.replicate import (
    DEFAULT_BASE_URL,
    REPLICATE_API_URL,
    REPLICATE_BASE_URL,
    REPLICATE_DEFAULT_BASE_URL,
)
from aivia.providers.protocols import CAPABILITY_IMAGE_GEN, ImageFormat, ImageRequest

API_KEY = "test-key"
MODEL_NAME = "black-forest-labs/flux-dev"
VERSION = "abc123def456"
MODEL = f"{MODEL_NAME}:{VERSION}"
ENDPOINT = f"{DEFAULT_BASE_URL}/predictions"
PREDICTION_ID = "pred-rep-1"
GET_URL = f"{DEFAULT_BASE_URL}/predictions/{PREDICTION_ID}"
CANCEL_URL = f"{GET_URL}/cancel"
FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "providers" / "replicate"


def _fixture(name: str) -> dict[str, Any]:
    return cast(
        "dict[str, Any]",
        json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8")),
    )


def _provider(
    handler: Callable[[httpx.Request], httpx.Response],
    **kwargs: Any,
) -> ReplicateImageProvider:
    return ReplicateImageProvider(
        api_key=API_KEY,
        model=MODEL,
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
        ("GET", GET_URL): [
            (200, _fixture("in_progress")),
            (200, _fixture("succeeded")),
        ],
    }


def test_metadata_and_registration_describe_the_provider() -> None:
    provider = _provider(lambda request: httpx.Response(200, json={}))

    assert provider.name == "replicate"
    assert provider.capabilities == (CAPABILITY_IMAGE_GEN,)
    assert provider.supports(CAPABILITY_IMAGE_GEN)
    assert provider.base_url == DEFAULT_BASE_URL
    assert provider.api_url == DEFAULT_BASE_URL
    assert provider.default_model == MODEL
    assert provider.model == MODEL
    assert provider.version == VERSION
    assert provider.endpoint == ENDPOINT
    assert REPLICATE_DEFAULT_BASE_URL == DEFAULT_BASE_URL
    assert REPLICATE_API_URL == DEFAULT_BASE_URL
    assert REPLICATE_BASE_URL == DEFAULT_BASE_URL
    assert ReplicateProvider is ReplicateImageProvider
    assert Replicate is ReplicateImageProvider
    assert TopLevelReplicate is ReplicateImageProvider
    assert provider_registry.resolve("replicate") is ReplicateImageProvider


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
    assert submitted.headers["authorization"] == f"Bearer {API_KEY}"
    assert json.loads(submitted.content) == {
        "version": VERSION,
        "input": {
            "prompt": "a calm ocean at sunrise",
            "num_outputs": 2,
            "width": 1024,
            "height": 768,
            "seed": 7,
            "output_format": "png",
        },
    }
    assert [str(sent.url) for sent in requests[1:]] == [GET_URL, GET_URL]
    assert response.provider == "replicate"
    assert response.model == MODEL
    assert response.request_id == PREDICTION_ID
    assert response.count == 2
    assert response.images[0].url == "https://replicate.delivery/pbxt/success-1.png"
    assert response.images[0].format is ImageFormat.PNG
    assert response.images[0].width == 1024
    assert response.images[0].height == 768
    assert response.images[0].seed == 7
    assert response.images[1].url == "https://replicate.delivery/pbxt/success-2.png"


def test_generate_accepts_the_version_as_a_separate_config_option() -> None:
    requests, handler = _routes(_success_routes())
    provider = ReplicateImageProvider(
        api_key=API_KEY,
        model=MODEL_NAME,
        version=VERSION,
        transport=httpx.MockTransport(handler),
        poll_interval_seconds=0.001,
    )

    asyncio.run(provider.generate(ImageRequest(prompt="a wave")))

    assert provider.default_model == MODEL
    assert provider.version == VERSION
    assert json.loads(requests[0].content)["version"] == VERSION


def test_generate_accepts_a_versioned_model_on_the_request() -> None:
    requests, handler = _routes(_success_routes())
    provider = ReplicateImageProvider(
        api_key=API_KEY,
        transport=httpx.MockTransport(handler),
        poll_interval_seconds=0.001,
    )

    asyncio.run(provider.generate(ImageRequest(prompt="a wave", model=MODEL)))

    assert str(requests[0].url) == ENDPOINT
    assert json.loads(requests[0].content)["version"] == VERSION


def test_a_model_without_a_version_is_a_config_error() -> None:
    with pytest.raises(ConfigError):
        ReplicateImageProvider(api_key=API_KEY, model=MODEL_NAME)


def test_an_unversioned_request_model_is_rejected() -> None:
    provider = ReplicateImageProvider(
        api_key=API_KEY,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})),
    )

    with pytest.raises(InvalidRequestError):
        asyncio.run(provider.generate(ImageRequest(prompt="a wave", model=MODEL_NAME)))


def test_generate_requires_a_model_id() -> None:
    provider = ReplicateImageProvider(
        api_key=API_KEY,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})),
    )

    with pytest.raises(InvalidRequestError):
        asyncio.run(provider.generate(ImageRequest(prompt="a wave")))


def test_generate_maps_a_failed_prediction_to_content_policy() -> None:
    routes = _success_routes()
    routes[("GET", GET_URL)] = [(200, _fixture("failed"))]
    requests, handler = _routes(routes)
    provider = _provider(handler)

    with pytest.raises(ContentPolicyError):
        asyncio.run(provider.generate(ImageRequest(prompt="blocked prompt")))

    assert str(requests[-1].url) == GET_URL


def test_generate_treats_a_flagged_output_as_content_policy() -> None:
    routes = _success_routes()
    routes[("GET", GET_URL)] = [
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
    provider = _provider(handler)

    with pytest.raises(ContentPolicyError):
        asyncio.run(provider.generate(ImageRequest(prompt="blocked prompt")))


def test_generate_maps_an_error_response_to_content_policy() -> None:
    routes = _success_routes()
    routes[("POST", ENDPOINT)] = [
        (
            422,
            {"detail": "The request was filtered by the content policy."},
        )
    ]
    _, handler = _routes(routes)
    provider = _provider(handler)

    with pytest.raises(ContentPolicyError) as caught:
        asyncio.run(provider.generate(ImageRequest(prompt="blocked prompt")))

    assert caught.value.details["status"] == 422


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
    assert body["input"]["aspect_ratio"] == "16:9"
    assert body["input"]["negative_prompt"] == "text"
    assert body["input"]["quality"] == "high"
    assert body["input"]["style"] == "photographic"


def test_generate_accepts_a_single_output_url() -> None:
    routes = _success_routes()
    routes[("GET", GET_URL)] = [
        (
            200,
            {
                "id": PREDICTION_ID,
                "status": "succeeded",
                "output": "https://replicate.delivery/pbxt/single.webp",
            },
        )
    ]
    _, handler = _routes(routes)
    provider = _provider(handler)

    response = asyncio.run(provider.generate(ImageRequest(prompt="a wave")))

    assert response.count == 1
    assert response.first.url == "https://replicate.delivery/pbxt/single.webp"
    assert response.first.format is ImageFormat.WEBP


def test_generate_downloads_results_when_configured() -> None:
    requests, base_handler = _routes(_success_routes())
    downloads = {
        "https://replicate.delivery/pbxt/success-1.png": b"image-one",
        "https://replicate.delivery/pbxt/success-2.png": b"image-two",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        data = downloads.get(str(request.url))
        if data is not None:
            requests.append(request)
            return httpx.Response(200, content=data)
        return base_handler(request)

    provider = _provider(handler, download_results=True)

    response = asyncio.run(provider.generate(ImageRequest(prompt="a wave")))

    assert [image.data for image in response.images] == [b"image-one", b"image-two"]
    assert all(image.url is None for image in response.images)


def test_a_missing_api_key_is_a_config_error() -> None:
    with pytest.raises(ConfigError):
        ReplicateImageProvider(model=MODEL)
