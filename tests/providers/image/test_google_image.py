"""Google image provider: Imagen predict, Gemini inline parts and safety blocks."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from aivia.core.errors import (
    AuthError,
    ConfigError,
    ContentPolicyError,
    InvalidRequestError,
    RateLimitError,
    ValidationError,
)
from aivia.providers import GoogleImage as TopLevelGoogleImage
from aivia.providers import provider_registry
from aivia.providers.image import (
    GOOGLE_API_URL,
    GOOGLE_BASE_URL,
    GOOGLE_DEFAULT_BASE_URL,
    GOOGLE_IMAGEN_API_URL,
    Google,
    GoogleImage,
    GoogleImagen,
    GoogleImagenProvider,
    GoogleImageProvider,
    GoogleImages,
    GoogleImagesProvider,
    GoogleProvider,
)
from aivia.providers.image.google import DEFAULT_API_VERSION, DEFAULT_BASE_URL
from aivia.providers.protocols import CAPABILITY_IMAGE_GEN, ImageFormat, ImageRequest

API_KEY = "test-key"
IMAGEN_MODEL = "imagen-4.0-generate-001"
GEMINI_MODEL = "gemini-2.5-flash-image"
IMAGEN_ENDPOINT = (
    f"{DEFAULT_BASE_URL}/{DEFAULT_API_VERSION}/models/{IMAGEN_MODEL}:predict"
)
GEMINI_ENDPOINT = (
    f"{DEFAULT_BASE_URL}/{DEFAULT_API_VERSION}/models/{GEMINI_MODEL}:generateContent"
)
PNG = b"\x89PNG\r\n\x1a\n" + b"image-one"
JPEG = b"\xff\xd8\xff\xe0" + b"image-two"


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _provider(
    handler: Callable[[httpx.Request], httpx.Response],
    **kwargs: Any,
) -> GoogleImageProvider:
    options: dict[str, Any] = {
        "api_key": API_KEY,
        "default_model": IMAGEN_MODEL,
        "transport": httpx.MockTransport(handler),
    }
    options.update(kwargs)
    return GoogleImageProvider(**options)


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


def _imagen_routes() -> dict[tuple[str, str], list[tuple[int, Any]]]:
    return {
        ("POST", IMAGEN_ENDPOINT): [
            (
                200,
                {
                    "predictions": [
                        {
                            "bytesBase64Encoded": _b64(PNG),
                            "mimeType": "image/png",
                        },
                        {
                            "bytesBase64Encoded": _b64(JPEG),
                            "mimeType": "image/jpeg",
                        },
                    ],
                    "modelVersion": IMAGEN_MODEL,
                    "responseId": "resp-google-1",
                },
            )
        ]
    }


def _gemini_routes() -> dict[tuple[str, str], list[tuple[int, Any]]]:
    return {
        ("POST", GEMINI_ENDPOINT): [
            (
                200,
                {
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [
                                    {
                                        "inlineData": {
                                            "mimeType": "image/png",
                                            "data": _b64(PNG),
                                        }
                                    }
                                ],
                            },
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 12,
                        "candidatesTokenCount": 34,
                        "totalTokenCount": 46,
                    },
                    "modelVersion": GEMINI_MODEL,
                    "responseId": "resp-google-2",
                },
            )
        ]
    }


def test_metadata_and_registration_describe_the_provider() -> None:
    provider = _provider(lambda request: httpx.Response(200, json={}))

    assert provider.name == "google-image"
    assert provider.capabilities == (CAPABILITY_IMAGE_GEN,)
    assert provider.supports(CAPABILITY_IMAGE_GEN)
    assert provider.base_url == DEFAULT_BASE_URL
    assert provider.api_url == DEFAULT_BASE_URL
    assert provider.api_version == DEFAULT_API_VERSION
    assert provider.default_model == IMAGEN_MODEL
    assert provider.model == IMAGEN_MODEL
    assert provider.endpoint == IMAGEN_ENDPOINT
    assert provider.endpoint_for(GEMINI_MODEL) == GEMINI_ENDPOINT
    assert GOOGLE_DEFAULT_BASE_URL == DEFAULT_BASE_URL
    assert GOOGLE_API_URL == DEFAULT_BASE_URL
    assert GOOGLE_BASE_URL == DEFAULT_BASE_URL
    assert GOOGLE_IMAGEN_API_URL == DEFAULT_BASE_URL
    assert Google is GoogleImageProvider
    assert GoogleImage is GoogleImageProvider
    assert GoogleImages is GoogleImageProvider
    assert GoogleImagesProvider is GoogleImageProvider
    assert GoogleImagen is GoogleImageProvider
    assert GoogleImagenProvider is GoogleImageProvider
    assert GoogleProvider is GoogleImageProvider
    assert TopLevelGoogleImage is GoogleImageProvider
    assert provider_registry.resolve("google-image") is GoogleImageProvider
    assert provider_registry.resolve("google_image") is GoogleImageProvider
    assert provider_registry.resolve("google") is GoogleImageProvider


def test_imagen_generates_count_size_and_decodes_base64_results() -> None:
    requests, handler = _routes(_imagen_routes())
    provider = _provider(handler)
    request = ImageRequest(
        prompt="  a calm ocean at sunrise  ",
        width=1024,
        height=768,
        count=2,
        seed=7,
        format=ImageFormat.PNG,
        negative_prompt="  text  ",
    )

    response = asyncio.run(provider.generate(request))

    submitted = requests[0]
    assert submitted.method == "POST"
    assert str(submitted.url) == IMAGEN_ENDPOINT
    assert submitted.headers["x-goog-api-key"] == API_KEY
    assert json.loads(submitted.content) == {
        "instances": [{"prompt": "a calm ocean at sunrise"}],
        "parameters": {
            "sampleCount": 2,
            "aspectRatio": "4:3",
            "negativePrompt": "text",
            "seed": 7,
            "outputOptions": {"mimeType": "image/png"},
        },
    }
    assert response.provider == "google-image"
    assert response.model == IMAGEN_MODEL
    assert response.request_id == "resp-google-1"
    assert response.count == 2
    assert response.images[0].data == PNG
    assert response.images[0].url is None
    assert response.images[0].format is ImageFormat.PNG
    assert response.images[0].width == 1024
    assert response.images[0].height == 768
    assert response.images[0].seed == 7
    assert response.images[1].data == JPEG
    assert response.images[1].format is ImageFormat.JPEG


def test_imagen_snaps_aspect_ratio_and_sends_no_output_options_for_webp() -> None:
    requests, handler = _routes(_imagen_routes())
    provider = _provider(handler)

    asyncio.run(
        provider.generate(
            ImageRequest(
                prompt="a wave",
                aspect_ratio="16:9",
                format=ImageFormat.WEBP,
            )
        )
    )

    body = json.loads(requests[0].content)
    assert body["parameters"]["aspectRatio"] == "16:9"
    assert "outputOptions" not in body["parameters"]


def test_imagen_snaps_an_unsupported_size_to_the_closest_shape() -> None:
    requests, handler = _routes(_imagen_routes())
    provider = _provider(handler)

    asyncio.run(
        provider.generate(
            ImageRequest(prompt="a wave", width=1000, height=1000, count=1)
        )
    )

    assert json.loads(requests[0].content)["parameters"]["aspectRatio"] == "1:1"


def test_gemini_image_generates_inline_parts_and_maps_usage() -> None:
    requests, handler = _routes(_gemini_routes())
    provider = _provider(handler, default_model=GEMINI_MODEL)
    request = ImageRequest(
        prompt="  a calm ocean at sunrise  ",
        width=1024,
        height=1024,
        count=1,
        seed=7,
        format=ImageFormat.PNG,
    )

    response = asyncio.run(provider.generate(request))

    submitted = requests[0]
    assert str(submitted.url) == GEMINI_ENDPOINT
    assert submitted.headers["x-goog-api-key"] == API_KEY
    assert json.loads(submitted.content) == {
        "contents": [{"role": "user", "parts": [{"text": "a calm ocean at sunrise"}]}],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
            "imageConfig": {"aspectRatio": "1:1"},
        },
    }
    assert response.provider == "google-image"
    assert response.model == GEMINI_MODEL
    assert response.request_id == "resp-google-2"
    assert response.count == 1
    assert response.images[0].data == PNG
    assert response.images[0].format is ImageFormat.PNG
    assert response.images[0].width == 1024
    assert response.images[0].height == 1024
    assert response.images[0].seed == 7
    assert response.metadata["usage"]["totalTokenCount"] == 46


def test_gemini_sends_a_candidate_count_for_variations() -> None:
    routes = _gemini_routes()
    body = routes[("POST", GEMINI_ENDPOINT)][0][1]
    body["candidates"] = body["candidates"] * 2
    requests, handler = _routes(routes)
    provider = _provider(handler, default_model=GEMINI_MODEL)

    response = asyncio.run(provider.generate(ImageRequest(prompt="a wave", count=2)))

    sent = json.loads(requests[0].content)
    assert sent["generationConfig"]["candidateCount"] == 2
    assert response.count == 2


def test_a_request_model_overrides_the_configured_imagen_default() -> None:
    requests, handler = _routes(_gemini_routes())
    provider = _provider(handler)

    asyncio.run(provider.generate(ImageRequest(prompt="a wave", model=GEMINI_MODEL)))

    assert str(requests[0].url) == GEMINI_ENDPOINT


def test_gemini_maps_a_blocked_prompt_to_content_policy() -> None:
    _, handler = _routes(
        {
            ("POST", GEMINI_ENDPOINT): [
                (
                    200,
                    {
                        "promptFeedback": {"blockReason": "SAFETY"},
                        "responseId": "resp-blocked",
                    },
                )
            ]
        }
    )
    provider = _provider(handler, default_model=GEMINI_MODEL)

    with pytest.raises(ContentPolicyError) as caught:
        asyncio.run(provider.generate(ImageRequest(prompt="blocked prompt")))

    assert caught.value.details["source"] == "prompt_feedback"
    assert caught.value.details["reason"] == "SAFETY"


def test_gemini_maps_a_safety_finish_reason_to_content_policy() -> None:
    _, handler = _routes(
        {
            ("POST", GEMINI_ENDPOINT): [
                (
                    200,
                    {
                        "candidates": [
                            {
                                "content": {"parts": []},
                                "finishReason": "IMAGE_SAFETY",
                            }
                        ]
                    },
                )
            ]
        }
    )
    provider = _provider(handler, default_model=GEMINI_MODEL)

    with pytest.raises(ContentPolicyError) as caught:
        asyncio.run(provider.generate(ImageRequest(prompt="blocked prompt")))

    assert caught.value.details["source"] == "candidate"
    assert caught.value.details["finish_reason"] == "IMAGE_SAFETY"


def test_imagen_maps_a_filtered_prediction_to_content_policy() -> None:
    _, handler = _routes(
        {
            ("POST", IMAGEN_ENDPOINT): [
                (
                    200,
                    {
                        "predictions": [
                            {
                                "raiFilteredReason": "The prompt was filtered.",
                            }
                        ]
                    },
                )
            ]
        }
    )
    provider = _provider(handler)

    with pytest.raises(ContentPolicyError) as caught:
        asyncio.run(provider.generate(ImageRequest(prompt="blocked prompt")))

    assert caught.value.details["source"] == "imagen"


def test_imagen_maps_a_filtered_response_to_content_policy() -> None:
    _, handler = _routes(
        {
            ("POST", IMAGEN_ENDPOINT): [
                (
                    200,
                    {
                        "predictions": [],
                        "raiFilteredReason": "The response was filtered.",
                    },
                )
            ]
        }
    )
    provider = _provider(handler)

    with pytest.raises(ContentPolicyError):
        asyncio.run(provider.generate(ImageRequest(prompt="blocked prompt")))


def test_generate_maps_a_safety_error_response_to_content_policy() -> None:
    _, handler = _routes(
        {
            ("POST", IMAGEN_ENDPOINT): [
                (
                    400,
                    {
                        "error": {
                            "status": "INVALID_ARGUMENT",
                            "message": (
                                "The prompt was filtered by the safety system."
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

    assert caught.value.details["status"] == "INVALID_ARGUMENT"


def test_generate_maps_authentication_and_rate_limit_errors() -> None:
    _, handler = _routes(
        {
            ("POST", IMAGEN_ENDPOINT): [
                (401, {"error": {"status": "UNAUTHENTICATED"}}),
            ]
        }
    )
    provider = _provider(handler)

    with pytest.raises(AuthError):
        asyncio.run(provider.generate(ImageRequest(prompt="a wave")))

    _, handler = _routes(
        {
            ("POST", IMAGEN_ENDPOINT): [
                (429, {"error": {"status": "RESOURCE_EXHAUSTED"}}),
            ]
        }
    )
    provider = _provider(handler)

    with pytest.raises(RateLimitError):
        asyncio.run(provider.generate(ImageRequest(prompt="a wave")))


def test_generate_requires_a_model_id() -> None:
    provider = GoogleImageProvider(
        api_key=API_KEY,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})),
    )

    with pytest.raises(InvalidRequestError):
        asyncio.run(provider.generate(ImageRequest(prompt="a wave")))


def test_a_missing_api_key_is_a_config_error() -> None:
    with pytest.raises(ConfigError):
        GoogleImageProvider(default_model=IMAGEN_MODEL)


def test_generate_rejects_a_response_without_images() -> None:
    _, handler = _routes({("POST", IMAGEN_ENDPOINT): [(200, {"predictions": []})]})
    provider = _provider(handler)

    with pytest.raises(ValidationError):
        asyncio.run(provider.generate(ImageRequest(prompt="a wave")))


def test_generate_rejects_invalid_base64() -> None:
    _, handler = _routes(
        {
            ("POST", IMAGEN_ENDPOINT): [
                (
                    200,
                    {
                        "predictions": [
                            {
                                "bytesBase64Encoded": "not base64!",
                                "mimeType": "image/png",
                            }
                        ]
                    },
                )
            ]
        }
    )
    provider = _provider(handler)

    with pytest.raises(ValidationError):
        asyncio.run(provider.generate(ImageRequest(prompt="a wave")))


def test_generate_rejects_a_non_json_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    provider = _provider(handler)

    with pytest.raises(ValidationError):
        asyncio.run(provider.generate(ImageRequest(prompt="a wave")))
