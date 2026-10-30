"""Remote background removal and upscaling: factors, payloads and dimensions."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from PIL import Image

from aivia.core.errors import ConfigError, InvalidRequestError, ValidationError
from aivia.providers import RemoteOps as TopLevelRemoteOps
from aivia.providers import provider_registry
from aivia.providers.image_edit import (
    DEFAULT_SCALES,
    DEFAULT_UPSCALE_FACTORS,
    REMOTE_OPS_DEFAULT_BASE_URL,
    SUPPORTED_UPSCALE_FACTORS,
    RemoteBackgroundRemovalAndUpscale,
    RemoteBackgroundRemovalAndUpscaleProvider,
    RemoteImageEdit,
    RemoteImageEditProvider,
    RemoteImageOps,
    RemoteImageOpsProvider,
    RemoteOps,
    RemoteOpsImageEditProvider,
    RemoteOpsProvider,
)
from aivia.providers.image_edit.remote_ops import SUPPORTED_OPERATIONS
from aivia.providers.protocols import (
    CAPABILITY_IMAGE_EDIT,
    ImageEditOperation,
    ImageEditRequest,
    ImageFormat,
    MediaInput,
)

API_KEY = "test-key"
REMOVE_MODEL = "vendor/remove-background"
UPSCALE_MODEL = "vendor/upscale"
MODELS: dict[ImageEditOperation | str, str] = {
    ImageEditOperation.REMOVE_BACKGROUND: REMOVE_MODEL,
    ImageEditOperation.UPSCALE: UPSCALE_MODEL,
}
SOURCE_URL = "https://media.example.com/source.png"
FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "providers" / "image_edit"
WIDTH = 64
HEIGHT = 48


def _fixture(name: str) -> dict[str, Any]:
    return cast(
        "dict[str, Any]",
        json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8")),
    )


def _provider(
    handler: Callable[[httpx.Request], httpx.Response],
    **kwargs: Any,
) -> RemoteOpsImageEditProvider:
    return RemoteOpsImageEditProvider(
        api_key=API_KEY,
        models=MODELS,
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


def _operation_routes(
    model: str,
    body: dict[str, Any],
) -> dict[tuple[str, str], list[tuple[int, Any]]]:
    base = REMOTE_OPS_DEFAULT_BASE_URL
    remote_id = f"req-{model.rsplit('/', 1)[-1]}"
    response_url = f"{base}/vendor/requests/{remote_id}"
    status_url = f"{response_url}/status"
    return {
        ("POST", f"{base}/{model}"): [
            (
                200,
                {
                    "request_id": remote_id,
                    "status_url": status_url,
                    "response_url": response_url,
                },
            )
        ],
        ("GET", status_url): [
            (200, {"status": "IN_QUEUE"}),
            (200, {"status": "IN_PROGRESS"}),
            (200, {"status": "COMPLETED"}),
        ],
        ("GET", response_url): [(200, body)],
    }


def _encode(image: Image.Image, image_format: str = "PNG") -> bytes:
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def _source(width: int = WIDTH, height: int = HEIGHT) -> bytes:
    return _encode(Image.new("RGB", (width, height), (20, 40, 60)))


def _inline_request(
    operation: ImageEditOperation,
    *,
    scale: float | None = None,
    width: int = WIDTH,
    height: int = HEIGHT,
    **kwargs: Any,
) -> ImageEditRequest:
    return ImageEditRequest(
        image=MediaInput(data=_source(width, height), media_type="image/png"),
        operation=operation,
        scale=scale,
        **kwargs,
    )


def test_metadata_registration_and_aliases_describe_the_provider() -> None:
    provider = _provider(lambda request: httpx.Response(200, json={}))

    assert provider.name == "remote-ops"
    assert provider.capabilities == (CAPABILITY_IMAGE_EDIT,)
    assert provider.supports(CAPABILITY_IMAGE_EDIT)
    assert provider.base_url == REMOTE_OPS_DEFAULT_BASE_URL
    assert provider.models == {
        "remove_background": REMOVE_MODEL,
        "upscale": UPSCALE_MODEL,
    }
    assert provider.model_for("remove_background") == REMOVE_MODEL
    assert provider.model_for(ImageEditOperation.UPSCALE) == UPSCALE_MODEL
    assert provider.factors == (2.0, 4.0)
    assert provider.supported_operations == (
        ImageEditOperation.REMOVE_BACKGROUND,
        ImageEditOperation.UPSCALE,
    )
    assert provider.supports_operation("remove_background")
    assert provider.supports_operation("upscale")
    assert not provider.supports_operation("edit")
    assert not provider.supports_operation("not-an-operation")
    assert DEFAULT_UPSCALE_FACTORS == (2.0, 4.0)
    assert SUPPORTED_UPSCALE_FACTORS == DEFAULT_UPSCALE_FACTORS
    assert DEFAULT_SCALES == DEFAULT_UPSCALE_FACTORS
    assert SUPPORTED_OPERATIONS == (
        ImageEditOperation.REMOVE_BACKGROUND,
        ImageEditOperation.UPSCALE,
    )
    assert RemoteOpsProvider is RemoteOpsImageEditProvider
    assert RemoteOps is RemoteOpsImageEditProvider
    assert RemoteImageOps is RemoteOpsImageEditProvider
    assert RemoteImageOpsProvider is RemoteOpsImageEditProvider
    assert RemoteImageEdit is RemoteOpsImageEditProvider
    assert RemoteImageEditProvider is RemoteOpsImageEditProvider
    assert RemoteBackgroundRemovalAndUpscale is RemoteOpsImageEditProvider
    assert (
        RemoteBackgroundRemovalAndUpscaleProvider is RemoteOpsImageEditProvider
    )
    assert TopLevelRemoteOps is RemoteOpsImageEditProvider
    assert provider_registry.resolve("remote-ops") is RemoteOpsImageEditProvider
    assert provider_registry.resolve("remote_ops") is RemoteOpsImageEditProvider
    assert (
        provider_registry.resolve("remote-image-ops") is RemoteOpsImageEditProvider
    )
    assert (
        provider_registry.resolve("remote-image-edit") is RemoteOpsImageEditProvider
    )


def test_remove_background_submits_an_image_and_keeps_the_source_size() -> None:
    requests, handler = _routes(
        _operation_routes(REMOVE_MODEL, _fixture("remove_background"))
    )
    provider = _provider(handler)
    source = _source()

    response = asyncio.run(
        provider.edit(_inline_request(ImageEditOperation.REMOVE_BACKGROUND))
    )

    submitted = requests[0]
    assert submitted.method == "POST"
    assert str(submitted.url) == f"{REMOTE_OPS_DEFAULT_BASE_URL}/{REMOVE_MODEL}"
    assert submitted.headers["authorization"] == f"Key {API_KEY}"
    encoded = base64.b64encode(source).decode("ascii")
    assert json.loads(submitted.content) == {
        "image_url": f"data:image/png;base64,{encoded}",
        "num_images": 1,
        "output_format": "png",
    }
    assert response.operation is ImageEditOperation.REMOVE_BACKGROUND
    assert response.provider == "remote-ops"
    assert response.model == REMOVE_MODEL
    assert response.count == 1
    assert response.first.dimensions is not None
    assert response.first.dimensions.width == WIDTH
    assert response.first.dimensions.height == HEIGHT
    assert response.first.format is ImageFormat.PNG


@pytest.mark.parametrize("factor", [2.0, 4.0])
def test_upscale_validates_the_factor_and_asserts_the_scaled_size(
    factor: float,
) -> None:
    width = round(WIDTH * factor)
    height = round(HEIGHT * factor)
    body = {
        "images": [
            {
                "url": f"https://media.example.com/upscaled-{int(factor)}x.png",
                "width": width,
                "height": height,
                "content_type": "image/webp",
            }
        ]
    }
    requests, handler = _routes(_operation_routes(UPSCALE_MODEL, body))
    provider = _provider(handler)

    response = asyncio.run(
        provider.edit(
            _inline_request(
                ImageEditOperation.UPSCALE,
                scale=factor,
                seed=7,
                format=ImageFormat.WEBP,
            )
        )
    )

    payload = json.loads(requests[0].content)
    assert payload["image_url"].startswith("data:image/png;base64,")
    assert {key: value for key, value in payload.items() if key != "image_url"} == {
        "num_images": 1,
        "output_format": "webp",
        "scale": factor,
        "seed": 7,
    }
    assert response.operation is ImageEditOperation.UPSCALE
    assert response.model == UPSCALE_MODEL
    assert response.first.dimensions is not None
    assert response.first.dimensions.width == width
    assert response.first.dimensions.height == height
    assert response.first.format is ImageFormat.WEBP


def test_a_url_source_uses_the_requested_target_size() -> None:
    body = {
        "images": [
            {
                "url": "https://media.example.com/upscaled-url.png",
                "width": 128,
                "height": 96,
                "content_type": "image/png",
            }
        ]
    }
    requests, handler = _routes(_operation_routes(UPSCALE_MODEL, body))
    provider = _provider(handler)
    request = ImageEditRequest(
        image=MediaInput(url=SOURCE_URL, media_type="image/png"),
        operation=ImageEditOperation.UPSCALE,
        scale=2.0,
        width=128,
        height=96,
    )

    response = asyncio.run(provider.edit(request))

    payload = json.loads(requests[0].content)
    assert payload["image_url"] == SOURCE_URL
    assert payload["scale"] == 2.0
    assert response.first.width == 128
    assert response.first.height == 96


def test_an_unsupported_factor_is_rejected_before_a_call() -> None:
    requests, handler = _routes({})
    provider = _provider(handler)
    request = _inline_request(ImageEditOperation.UPSCALE, scale=3.0)

    with pytest.raises(InvalidRequestError) as caught:
        asyncio.run(provider.edit(request))

    assert "unsupported upscale factor" in str(caught.value)
    assert caught.value.details["supported"] == [2.0, 4.0]
    assert requests == []


def test_an_unsupported_operation_is_rejected_before_a_call() -> None:
    requests, handler = _routes({})
    provider = _provider(handler)
    request = ImageEditRequest(
        image=MediaInput(url=SOURCE_URL),
        operation=ImageEditOperation.EDIT,
        prompt="add a hat",
    )

    with pytest.raises(InvalidRequestError) as caught:
        asyncio.run(provider.edit(request))

    assert "edit" in str(caught.value)
    assert caught.value.details["operation"] == "edit"
    assert requests == []


def test_background_removal_rejects_an_upscale_scale() -> None:
    requests, handler = _routes({})
    provider = _provider(handler)
    request = _inline_request(
        ImageEditOperation.REMOVE_BACKGROUND,
        scale=2.0,
    )

    with pytest.raises(InvalidRequestError) as caught:
        asyncio.run(provider.edit(request))

    assert caught.value.details["operation"] == "remove_background"
    assert requests == []


def test_wrong_or_missing_dimensions_are_rejected() -> None:
    _, wrong_handler = _routes(
        _operation_routes(UPSCALE_MODEL, _fixture("upscale_wrong_size"))
    )
    request = _inline_request(ImageEditOperation.UPSCALE, scale=2.0)

    with pytest.raises(ValidationError) as caught:
        asyncio.run(_provider(wrong_handler).edit(request))

    assert caught.value.details["expected"] == [128, 96]
    assert caught.value.details["actual"] == [64, 48]

    _, missing_handler = _routes(
        _operation_routes(UPSCALE_MODEL, _fixture("upscale_no_size"))
    )

    with pytest.raises(ValidationError) as caught:
        asyncio.run(_provider(missing_handler).edit(request))

    assert caught.value.details["expected"] == [128, 96]


def test_models_and_factors_are_validated_at_construction() -> None:
    with pytest.raises(ConfigError):
        RemoteOpsImageEditProvider(api_key=API_KEY, models={"blur": UPSCALE_MODEL})
    with pytest.raises(ConfigError):
        RemoteOpsImageEditProvider(
            api_key=API_KEY,
            models={ImageEditOperation.EDIT: UPSCALE_MODEL},
        )
    with pytest.raises(ConfigError):
        RemoteOpsImageEditProvider(
            api_key=API_KEY,
            models={"upscale": UPSCALE_MODEL, "upscaling": REMOVE_MODEL},
        )
    with pytest.raises(TypeError):
        RemoteOpsImageEditProvider(
            api_key=API_KEY,
            models={"upscale": 123},  # type: ignore[dict-item]
        )
    with pytest.raises(TypeError):
        RemoteOpsImageEditProvider(api_key=API_KEY, factors="2,4")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        RemoteOpsImageEditProvider(
            api_key=API_KEY,
            factors=(2.0, True),
        )
    with pytest.raises(ConfigError):
        RemoteOpsImageEditProvider(api_key=API_KEY, factors=(1.0,))
    with pytest.raises(ConfigError):
        RemoteOpsImageEditProvider(api_key=API_KEY, factors=(2.0, 9.0))
    with pytest.raises(ConfigError):
        RemoteOpsImageEditProvider(api_key=API_KEY, factors=(2.0, 2.0))
    with pytest.raises(ConfigError):
        RemoteOpsImageEditProvider(api_key=API_KEY, factors=())
    provider = RemoteOpsImageEditProvider(api_key=API_KEY, factors=(2, 4, 8))
    assert provider.factors == (2.0, 4.0, 8.0)


def test_a_fallback_model_is_used_when_an_operation_has_no_entry() -> None:
    body = {
        "images": [
            {
                "url": "https://media.example.com/fallback.png",
                "width": WIDTH,
                "height": HEIGHT,
                "content_type": "image/png",
            }
        ]
    }
    requests, handler = _routes(_operation_routes(UPSCALE_MODEL, body))
    provider = RemoteOpsImageEditProvider(
        api_key=API_KEY,
        model=UPSCALE_MODEL,
        transport=httpx.MockTransport(handler),
        poll_interval_seconds=0.001,
    )

    asyncio.run(
        provider.edit(_inline_request(ImageEditOperation.REMOVE_BACKGROUND))
    )

    assert str(requests[0].url) == f"{REMOTE_OPS_DEFAULT_BASE_URL}/{UPSCALE_MODEL}"
