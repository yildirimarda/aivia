"""Aggregator image edit provider: per operation models, payloads and fixtures."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from aivia.core.errors import ConfigError, InvalidRequestError
from aivia.providers import Aggregator as TopLevelAggregator
from aivia.providers import provider_registry
from aivia.providers.image_edit import (
    AGGREGATOR_DEFAULT_BASE_URL,
    SUPPORTED_OPERATIONS,
    Aggregator,
    AggregatorImageEdit,
    AggregatorImageEditProvider,
    AggregatorProvider,
)
from aivia.providers.protocols import (
    CAPABILITY_IMAGE_EDIT,
    ImageEditOperation,
    ImageEditRequest,
    ImageFormat,
    MediaInput,
)

API_KEY = "test-key"
EDIT_MODEL = "vendor/instruction-edit"
INPAINT_MODEL = "vendor/inpaint"
OUTPAINT_MODEL = "vendor/outpaint"
MODELS: dict[ImageEditOperation | str, str] = {
    ImageEditOperation.EDIT: EDIT_MODEL,
    ImageEditOperation.INPAINT: INPAINT_MODEL,
    ImageEditOperation.OUTPAINT: OUTPAINT_MODEL,
}
SOURCE_URL = "https://media.example.com/source.png"
MASK_URL = "https://media.example.com/mask.png"
FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "providers" / "image_edit"


def _fixture(name: str) -> dict[str, Any]:
    return cast(
        "dict[str, Any]",
        json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8")),
    )


def _provider(
    handler: Callable[[httpx.Request], httpx.Response],
    **kwargs: Any,
) -> AggregatorImageEditProvider:
    return AggregatorImageEditProvider(
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
    fixture: str,
) -> dict[tuple[str, str], list[tuple[int, Any]]]:
    base = AGGREGATOR_DEFAULT_BASE_URL
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
        ("GET", response_url): [(200, _fixture(fixture))],
    }


def _request(operation: ImageEditOperation) -> ImageEditRequest:
    image = MediaInput(url=SOURCE_URL, media_type="image/png")
    prompt = f"  {operation.value} the picture  "
    if operation is ImageEditOperation.INPAINT:
        return ImageEditRequest(
            image=image,
            operation=operation,
            prompt=prompt,
            seed=7,
            format=ImageFormat.PNG,
            mask=MediaInput(url=MASK_URL, media_type="image/png"),
        )
    if operation is ImageEditOperation.OUTPAINT:
        return ImageEditRequest(
            image=image,
            operation=operation,
            prompt=prompt,
            seed=7,
            format=ImageFormat.PNG,
            width=1536,
            height=1024,
        )
    return ImageEditRequest(
        image=image,
        operation=operation,
        prompt=prompt,
        seed=7,
        format=ImageFormat.PNG,
    )


def test_metadata_registration_and_aliases_describe_the_provider() -> None:
    provider = _provider(lambda request: httpx.Response(200, json={}))

    assert provider.name == "aggregator"
    assert provider.capabilities == (CAPABILITY_IMAGE_EDIT,)
    assert provider.supports(CAPABILITY_IMAGE_EDIT)
    assert provider.base_url == AGGREGATOR_DEFAULT_BASE_URL
    assert provider.models == {
        "edit": EDIT_MODEL,
        "inpaint": INPAINT_MODEL,
        "outpaint": OUTPAINT_MODEL,
    }
    assert provider.model_for("edit") == EDIT_MODEL
    assert provider.model_for(ImageEditOperation.INPAINT) == INPAINT_MODEL
    assert AggregatorProvider is AggregatorImageEditProvider
    assert Aggregator is AggregatorImageEditProvider
    assert AggregatorImageEdit is AggregatorImageEditProvider
    assert TopLevelAggregator is AggregatorImageEditProvider
    assert provider_registry.resolve("aggregator") is AggregatorImageEditProvider
    assert (
        provider_registry.resolve("aggregator-image-edit")
        is AggregatorImageEditProvider
    )
    assert SUPPORTED_OPERATIONS == (
        ImageEditOperation.EDIT,
        ImageEditOperation.INPAINT,
        ImageEditOperation.OUTPAINT,
    )


@pytest.mark.parametrize(
    ("operation", "model", "fixture", "extra_payload"),
    [
        (ImageEditOperation.EDIT, EDIT_MODEL, "edit", {}),
        (
            ImageEditOperation.INPAINT,
            INPAINT_MODEL,
            "inpaint",
            {"mask_url": MASK_URL},
        ),
        (
            ImageEditOperation.OUTPAINT,
            OUTPAINT_MODEL,
            "outpaint",
            {"image_size": {"width": 1536, "height": 1024}},
        ),
    ],
)
def test_each_operation_uses_its_model_payload_and_fixture(
    operation: ImageEditOperation,
    model: str,
    fixture: str,
    extra_payload: dict[str, Any],
) -> None:
    requests, handler = _routes(_operation_routes(model, fixture))
    provider = _provider(handler)

    response = asyncio.run(provider.edit(_request(operation)))

    submitted = requests[0]
    assert submitted.method == "POST"
    assert str(submitted.url) == f"{AGGREGATOR_DEFAULT_BASE_URL}/{model}"
    assert submitted.headers["authorization"] == f"Key {API_KEY}"
    assert json.loads(submitted.content) == {
        "prompt": f"{operation.value} the picture",
        "num_images": 1,
        "output_format": "png",
        "image_url": SOURCE_URL,
        "seed": 7,
        **extra_payload,
    }
    assert response.operation is operation
    assert response.provider == "aggregator"
    assert response.model == model
    assert response.count == 1
    assert response.first.url == _fixture(fixture)["images"][0]["url"]
    assert response.first.format is ImageFormat.PNG
    assert response.first.dimensions is not None
    assert response.first.dimensions.width == _fixture(fixture)["images"][0]["width"]
    assert response.first.dimensions.height == _fixture(fixture)["images"][0]["height"]
    assert response.first.seed == _fixture(fixture)["seed"]


def test_the_request_model_overrides_the_operation_model() -> None:
    requests, handler = _routes(_operation_routes("vendor/override", "edit"))
    provider = _provider(handler)
    request = ImageEditRequest(
        image=MediaInput(url=SOURCE_URL),
        prompt="add a red hat",
        model="vendor/override",
    )

    response = asyncio.run(provider.edit(request))

    assert str(requests[0].url) == (f"{AGGREGATOR_DEFAULT_BASE_URL}/vendor/override")
    assert response.model == "vendor/override"


def test_inline_source_and_mask_become_data_urls() -> None:
    requests, handler = _routes(_operation_routes(INPAINT_MODEL, "inpaint"))
    provider = _provider(handler)
    request = ImageEditRequest(
        image=MediaInput(data=b"source-bytes", media_type="image/png"),
        operation=ImageEditOperation.INPAINT,
        prompt="replace the sky",
        mask=MediaInput(data=b"mask-bytes", media_type="image/png"),
    )

    asyncio.run(provider.edit(request))

    payload = json.loads(requests[0].content)
    expected_source = base64.b64encode(b"source-bytes").decode("ascii")
    expected_mask = base64.b64encode(b"mask-bytes").decode("ascii")
    assert payload["image_url"] == f"data:image/png;base64,{expected_source}"
    assert payload["mask_url"] == f"data:image/png;base64,{expected_mask}"


def test_an_operation_without_a_model_is_rejected() -> None:
    provider = AggregatorImageEditProvider(
        api_key=API_KEY,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})),
    )

    with pytest.raises(InvalidRequestError) as caught:
        asyncio.run(provider.edit(_request(ImageEditOperation.EDIT)))

    assert caught.value.details["operation"] == "edit"


def test_unsupported_operations_are_rejected_before_a_call() -> None:
    requests, handler = _routes({})
    provider = _provider(handler)
    request = ImageEditRequest(
        image=MediaInput(url=SOURCE_URL),
        operation=ImageEditOperation.UPSCALE,
        scale=2.0,
    )

    with pytest.raises(InvalidRequestError) as caught:
        asyncio.run(provider.edit(request))

    assert "upscale" in str(caught.value)
    assert caught.value.details["operation"] == "upscale"
    assert requests == []


def test_the_models_mapping_rejects_unknown_or_unsupported_operations() -> None:
    with pytest.raises(ConfigError):
        AggregatorImageEditProvider(api_key=API_KEY, models={"blur": EDIT_MODEL})
    with pytest.raises(ConfigError):
        AggregatorImageEditProvider(
            api_key=API_KEY,
            models={ImageEditOperation.UPSCALE: EDIT_MODEL},
        )
    with pytest.raises(ConfigError):
        AggregatorImageEditProvider(
            api_key=API_KEY,
            models={"edit": EDIT_MODEL, "instruction_edit": INPAINT_MODEL},
        )
    with pytest.raises(TypeError):
        AggregatorImageEditProvider(
            api_key=API_KEY,
            models={"edit": 123},  # type: ignore[dict-item]
        )


def test_a_fallback_model_is_used_when_an_operation_has_no_entry() -> None:
    requests, handler = _routes(_operation_routes(EDIT_MODEL, "edit"))
    provider = AggregatorImageEditProvider(
        api_key=API_KEY,
        model=EDIT_MODEL,
        transport=httpx.MockTransport(handler),
        poll_interval_seconds=0.001,
    )

    asyncio.run(provider.edit(_request(ImageEditOperation.EDIT)))

    assert str(requests[0].url) == f"{AGGREGATOR_DEFAULT_BASE_URL}/{EDIT_MODEL}"
