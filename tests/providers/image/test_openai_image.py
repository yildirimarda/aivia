"""OpenAI image provider: sizes, counts and base64 or URL results."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from aivia.core.errors import (
    ConfigError,
    ContentPolicyError,
    InvalidRequestError,
    RateLimitError,
    ValidationError,
)
from aivia.providers import OpenAIImage as TopLevelOpenAIImage
from aivia.providers import provider_registry
from aivia.providers.image import (
    OPENAI_API_URL,
    OPENAI_BASE_URL,
    OPENAI_DEFAULT_BASE_URL,
    OpenAIImage,
    OpenAIImageProvider,
    OpenAIImages,
    OpenAIImagesProvider,
)
from aivia.providers.image.openai import DEFAULT_BASE_URL
from aivia.providers.protocols import CAPABILITY_IMAGE_GEN, ImageFormat, ImageRequest

API_KEY = "test-key"
MODEL = "gpt-image-1"
ENDPOINT = f"{DEFAULT_BASE_URL}/v1/images/generations"
PNG = b"\x89PNG\r\n\x1a\n" + b"image-one"
JPEG = b"\xff\xd8\xff\xe0" + b"image-two"


def _provider(
    handler: Callable[[httpx.Request], httpx.Response],
    **kwargs: Any,
) -> OpenAIImageProvider:
    return OpenAIImageProvider(
        api_key=API_KEY,
        default_model=MODEL,
        transport=httpx.MockTransport(handler),
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


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _base64_routes() -> dict[tuple[str, str], list[tuple[int, Any]]]:
    return {
        ("POST", ENDPOINT): [
            (
                200,
                {
                    "created": 1713833628,
                    "data": [
                        {"b64_json": _b64(PNG), "revised_prompt": "  a calm ocean  "},
                        {"b64_json": _b64(PNG)},
                    ],
                },
            )
        ]
    }


def _url_routes() -> dict[tuple[str, str], list[tuple[int, Any]]]:
    return {
        ("POST", ENDPOINT): [
            (
                200,
                {
                    "created": 1713833628,
                    "data": [
                        {"url": "https://images.example.test/success-1.jpg"},
                        {"url": "https://images.example.test/success-2.jpg"},
                    ],
                },
            )
        ]
    }


def test_metadata_and_registration_describe_the_provider() -> None:
    provider = _provider(lambda request: httpx.Response(200, json={}))

    assert provider.name == "openai-image"
    assert provider.capabilities == (CAPABILITY_IMAGE_GEN,)
    assert provider.supports(CAPABILITY_IMAGE_GEN)
    assert provider.base_url == DEFAULT_BASE_URL
    assert provider.api_url == DEFAULT_BASE_URL
    assert provider.default_model == MODEL
    assert provider.model == MODEL
    assert provider.response_format is None
    assert provider.endpoint == ENDPOINT
    assert OPENAI_DEFAULT_BASE_URL == DEFAULT_BASE_URL
    assert OPENAI_API_URL == DEFAULT_BASE_URL
    assert OPENAI_BASE_URL == DEFAULT_BASE_URL
    assert OpenAIImage is OpenAIImageProvider
    assert OpenAIImages is OpenAIImageProvider
    assert OpenAIImagesProvider is OpenAIImageProvider
    assert TopLevelOpenAIImage is OpenAIImageProvider
    assert provider_registry.resolve("openai-image") is OpenAIImageProvider
    assert provider_registry.resolve("openai_image") is OpenAIImageProvider


def test_generate_submits_size_count_and_decodes_base64_results() -> None:
    requests, handler = _routes(_base64_routes())
    provider = _provider(handler)
    request = ImageRequest(
        prompt="  a calm ocean at sunrise  ",
        width=1024,
        height=1024,
        count=2,
        seed=7,
        format=ImageFormat.PNG,
        quality="high",
        style="natural",
    )

    response = asyncio.run(provider.generate(request))

    submitted = requests[0]
    assert submitted.method == "POST"
    assert str(submitted.url) == ENDPOINT
    assert submitted.headers["authorization"] == f"Bearer {API_KEY}"
    assert json.loads(submitted.content) == {
        "model": MODEL,
        "prompt": "a calm ocean at sunrise",
        "n": 2,
        "size": "1024x1024",
        "quality": "high",
        "style": "natural",
    }
    assert response.provider == "openai-image"
    assert response.model == MODEL
    assert response.count == 2
    assert response.images[0].data == PNG
    assert response.images[0].url is None
    assert response.images[0].format is ImageFormat.PNG
    assert response.images[0].width == 1024
    assert response.images[0].height == 1024
    assert response.images[0].seed == 7
    assert response.images[0].revised_prompt == "a calm ocean"
    assert response.images[1].data == PNG
    assert response.metadata["created"] == 1713833628


def test_generate_maps_url_results_and_keeps_remote_location() -> None:
    requests, handler = _routes(_url_routes())
    provider = _provider(handler)
    request = ImageRequest(
        prompt="a wave",
        width=1792,
        height=1024,
        count=2,
        model="dall-e-3",
        format=ImageFormat.JPEG,
    )

    response = asyncio.run(provider.generate(request))

    assert json.loads(requests[0].content)["size"] == "1792x1024"
    assert response.count == 2
    assert response.images[0].url == "https://images.example.test/success-1.jpg"
    assert response.images[0].data is None
    assert response.images[0].format is ImageFormat.JPEG
    assert response.images[0].width == 1792
    assert response.images[0].height == 1024


def test_generate_snaps_an_aspect_ratio_to_a_supported_size() -> None:
    requests, handler = _routes(_url_routes())
    provider = _provider(handler)

    response = asyncio.run(
        provider.generate(ImageRequest(prompt="a wave", aspect_ratio="16:9"))
    )

    body = json.loads(requests[0].content)
    assert body["size"] == "1536x1024"
    assert "n" in body
    assert response.images[0].width == 1536
    assert response.images[0].height == 1024


def test_generate_snaps_an_unsupported_size_to_the_closest_shape() -> None:
    requests, handler = _routes(_url_routes())
    provider = _provider(handler)

    asyncio.run(
        provider.generate(
            ImageRequest(prompt="a wave", width=1000, height=1000, count=1)
        )
    )

    assert json.loads(requests[0].content)["size"] == "1024x1024"


def test_generate_accepts_a_data_url_base64_result() -> None:
    routes = _base64_routes()
    routes[("POST", ENDPOINT)] = [
        (
            200,
            {"data": [{"b64_json": ("data:image/jpeg;base64," + _b64(JPEG))}]},
        )
    ]
    _, handler = _routes(routes)
    provider = _provider(handler)

    response = asyncio.run(provider.generate(ImageRequest(prompt="a wave")))

    assert response.first.data == JPEG
    assert response.first.format is ImageFormat.JPEG
    assert response.first.metadata["content_type"] == "image/jpeg"


def test_generate_sends_a_configured_response_format() -> None:
    requests, handler = _routes(_base64_routes())
    provider = _provider(handler, response_format="b64_json")

    asyncio.run(provider.generate(ImageRequest(prompt="a wave")))

    assert json.loads(requests[0].content)["response_format"] == "b64_json"


def test_a_request_metadata_response_format_overrides_the_config() -> None:
    requests, handler = _routes(_url_routes())
    provider = _provider(handler, response_format="b64_json")

    asyncio.run(
        provider.generate(
            ImageRequest(prompt="a wave", metadata={"response_format": "url"})
        )
    )

    assert json.loads(requests[0].content)["response_format"] == "url"


def test_an_invalid_response_format_is_a_config_error() -> None:
    with pytest.raises(ConfigError):
        OpenAIImageProvider(api_key=API_KEY, response_format="xml")


def test_generate_requires_a_model_id() -> None:
    provider = OpenAIImageProvider(
        api_key=API_KEY,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})),
    )

    with pytest.raises(InvalidRequestError):
        asyncio.run(provider.generate(ImageRequest(prompt="a wave")))


def test_generate_accepts_the_model_on_the_request() -> None:
    requests, handler = _routes(_url_routes())
    provider = OpenAIImageProvider(
        api_key=API_KEY,
        transport=httpx.MockTransport(handler),
    )

    asyncio.run(provider.generate(ImageRequest(prompt="a wave", model=MODEL)))

    assert str(requests[0].url) == ENDPOINT
    assert json.loads(requests[0].content)["model"] == MODEL


def test_a_missing_api_key_is_a_config_error() -> None:
    with pytest.raises(ConfigError):
        OpenAIImageProvider(model=MODEL)


def test_generate_maps_a_moderation_error_to_content_policy() -> None:
    _, handler = _routes(
        {
            ("POST", ENDPOINT): [
                (
                    400,
                    {
                        "error": {
                            "type": "invalid_request_error",
                            "code": "moderation_blocked",
                            "message": (
                                "Your request was rejected by the safety system."
                            ),
                        }
                    },
                )
            ]
        }
    )
    provider = _provider(handler)

    with pytest.raises(ContentPolicyError) as caught:
        asyncio.run(provider.generate(ImageRequest(prompt="blocked prompt")))

    assert caught.value.details["status"] == 400
    assert caught.value.details["code"] == "moderation_blocked"


def test_generate_maps_a_rate_limit_response() -> None:
    _, handler = _routes(
        {
            ("POST", ENDPOINT): [
                (
                    429,
                    {
                        "error": {
                            "type": "rate_limit_error",
                            "code": "rate_limit_exceeded",
                            "message": "Slow down.",
                        }
                    },
                )
            ]
        }
    )
    provider = _provider(handler)

    with pytest.raises(RateLimitError):
        asyncio.run(provider.generate(ImageRequest(prompt="a wave")))


def test_generate_rejects_a_response_without_images() -> None:
    _, handler = _routes({("POST", ENDPOINT): [(200, {"data": []})]})
    provider = _provider(handler)

    with pytest.raises(ValidationError):
        asyncio.run(provider.generate(ImageRequest(prompt="a wave")))


def test_generate_rejects_invalid_base64() -> None:
    _, handler = _routes(
        {("POST", ENDPOINT): [(200, {"data": [{"b64_json": "not base64!"}]})]}
    )
    provider = _provider(handler)

    with pytest.raises(ValidationError):
        asyncio.run(provider.generate(ImageRequest(prompt="a wave")))
