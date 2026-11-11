"""Instagram publisher adapter tests.

Covers the Graph API media container flow, the business account check,
carousels, aspect validation, caption/hashtag limits, retries and error
classification.
"""

from __future__ import annotations

import asyncio
import io
from collections.abc import Callable
from typing import Any, cast
from urllib.parse import parse_qs

import httpx
import pytest
from PIL import Image

from aivia.core.errors import (
    AuthError,
    InvalidRequestError,
    RateLimitError,
    RetryableError,
)
from aivia.core.models.io import Attestation
from aivia.core.models.provider import ProviderPrice
from aivia.core.publish_errors import classify_error
from aivia.core.secrets import SecretsLoader
from aivia.providers import provider_registry
from aivia.providers.instagram_publish import (
    DEFAULT_BASE_URL,
    InstagramPublisher,
)
from aivia.providers.protocols.publish import (
    CAPABILITY_PUBLISH,
    PublishMedia,
    PublishRequest,
)

IG_USER_ID = "17841234567890123"
ACCESS_TOKEN = "EAA...instagram-token"
MEDIA_ID = "17987654321098765"


def _png_bytes(width: int, height: int) -> bytes:
    image = Image.new("RGB", (width, height))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _media(
    *,
    url: str | None = None,
    data: bytes | None = None,
    path: str | None = None,
    media_type: str = "image/jpeg",
    filename: str = "image.jpg",
) -> PublishMedia:
    return PublishMedia(
        data=data, path=path, url=url, media_type=media_type, filename=filename
    )


def _request(
    *,
    platform: str = "instagram",
    caption: str = "A run is waiting",
    hashtags: list[str] | None = None,
    dry_run: bool = False,
    media: list[PublishMedia] | None = None,
    format: str = "",
    attestation: Attestation | None = None,
) -> PublishRequest:
    return PublishRequest(
        platform=platform,
        caption=caption,
        hashtags=hashtags or [],
        dry_run=dry_run,
        media=media or [],
        format=format,
        attestation=attestation,
    )


def _capture(
    *,
    account_type: str = "BUSINESS",
    fail_containers: int = 0,
    container_id: str = "creation-1",
    media_id: str = MEDIA_ID,
) -> tuple[list[httpx.Request], list[int], Callable[[httpx.Request], httpx.Response]]:
    requests: list[httpx.Request] = []
    calls = [0]
    containers = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        calls[0] += 1
        url = str(request.url)
        if request.method == "GET" and IG_USER_ID in url and "/media" not in url:
            return httpx.Response(
                200,
                json={
                    "account_type": account_type,
                    "id": IG_USER_ID,
                    "username": "reviewer",
                },
            )
        if request.method == "POST" and "media_publish" in url:
            return httpx.Response(200, json={"id": media_id})
        if request.method == "POST" and f"{IG_USER_ID}/media" in url:
            containers[0] += 1
            if containers[0] <= fail_containers:
                return httpx.Response(500, json={"error": {"message": "transient"}})
            return httpx.Response(200, json={"id": f"{container_id}-{containers[0]}"})
        return httpx.Response(404, json={"error": {"message": "not found"}})

    return requests, calls, handler


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _provider(
    handler: Any,
    *,
    access_token: str | None = ACCESS_TOKEN,
    ig_user_id: str = IG_USER_ID,
    transport: httpx.AsyncBaseTransport | None = None,
    **kwargs: Any,
) -> InstagramPublisher:
    options: dict[str, Any] = {
        "access_token": access_token,
        "ig_user_id": ig_user_id,
        "transport": transport or httpx.MockTransport(handler),
        "retry_delay_seconds": 0.0,
    }
    options.update(kwargs)
    return InstagramPublisher(**options)


def test_metadata_and_registration_describe_the_provider() -> None:
    _requests, _, handler = _capture()
    provider = _provider(handler)

    assert provider.name == "instagram"
    assert provider.capabilities == (CAPABILITY_PUBLISH,)
    assert provider.ig_user_id == IG_USER_ID
    assert provider.base_url == DEFAULT_BASE_URL
    assert provider_registry.resolve("instagram") is InstagramPublisher
    assert provider_registry.resolve("instagram_publish") is InstagramPublisher


def test_dry_run_returns_a_not_live_result_without_posting() -> None:
    requests, calls, handler = _capture()
    provider = _provider(handler)
    request = _request(
        dry_run=True, media=[_media(url="https://example.com/image.jpg")]
    )

    result = _run(provider.publish(request))

    assert calls[0] == 0
    assert requests == []
    assert result.is_live is False
    assert result.remote_id == ""
    assert result.dry_run is True
    assert result.provider == "instagram"


def test_missing_access_token_is_rejected() -> None:
    with pytest.raises(InvalidRequestError):
        InstagramPublisher(
            ig_user_id=IG_USER_ID,
            transport=httpx.MockTransport(lambda request: httpx.Response(200)),
        )


def test_missing_ig_user_id_is_rejected() -> None:
    with pytest.raises(InvalidRequestError, match="ig_user_id"):
        InstagramPublisher(access_token=ACCESS_TOKEN)


def test_access_token_resolved_from_reference() -> None:
    requests, _, handler = _capture()
    secrets = SecretsLoader(environ={"AIVA_INSTAGRAM_ACCESS_TOKEN": ACCESS_TOKEN})
    provider = _provider(
        handler,
        access_token=None,
        access_token_ref="AIVA_INSTAGRAM_ACCESS_TOKEN",
        secrets=secrets,
    )

    _run(
        provider.publish(_request(media=[_media(url="https://example.com/image.jpg")]))
    )

    assert requests and ACCESS_TOKEN in str(requests[0].url)


def test_business_account_check_rejects_a_personal_account() -> None:
    provider = _provider(
        lambda request: httpx.Response(
            200,
            json={"account_type": "PERSONAL", "id": IG_USER_ID, "username": "reviewer"},
        )
    )

    with pytest.raises(InvalidRequestError, match="business or creator account"):
        _run(
            provider.publish(
                _request(media=[_media(url="https://example.com/image.jpg")])
            )
        )


def test_business_account_check_allows_a_business_account() -> None:
    requests, calls, handler = _capture(account_type="BUSINESS")
    provider = _provider(handler)

    result = _run(
        provider.publish(_request(media=[_media(url="https://example.com/image.jpg")]))
    )

    assert calls[0] == 3
    assert result.is_live is True
    assert result.remote_id == MEDIA_ID
    assert requests and requests[0].method == "GET"


def test_no_media_is_rejected() -> None:
    provider = _provider(lambda request: httpx.Response(200, json={}))

    with pytest.raises(InvalidRequestError, match="requires at least one"):
        _run(provider.publish(_request()))


def test_publish_requires_a_publish_request() -> None:
    provider = _provider(lambda request: httpx.Response(200, json={}))

    with pytest.raises(TypeError, match="PublishRequest"):
        _run(provider.publish(cast(PublishRequest, "not a request")))


def test_unsupported_media_type_is_rejected() -> None:
    provider = _provider(lambda request: httpx.Response(200, json={}))

    with pytest.raises(InvalidRequestError, match="image or video"):
        _run(
            provider.publish(
                _request(
                    media=[
                        _media(media_type="audio/mpeg", url="https://example.com/x.mp3")
                    ]
                )
            )
        )


def test_aspect_validation_rejects_an_unsupported_image() -> None:
    provider = _provider(lambda request: httpx.Response(200, json={}))
    request = _request(media=[_media(data=_png_bytes(1080, 608))])  # 16:9

    with pytest.raises(InvalidRequestError, match="is not supported"):
        _run(provider.publish(request))


def test_aspect_validation_accepts_square_and_portrait() -> None:
    provider = _provider(lambda request: httpx.Response(200, json={}))

    for width, height in ((1080, 1080), (1080, 1350)):
        request = _request(media=[_media(data=_png_bytes(width, height))])
        with pytest.raises(InvalidRequestError, match="requires a public url"):
            _run(provider.publish(request))


def test_aspect_validation_rejects_an_unsupported_video() -> None:
    provider = _provider(lambda request: httpx.Response(200, json={}))
    request = _request(
        media=[_media(data=_png_bytes(1080, 1080), media_type="video/mp4")]
    )  # 1:1, not 9:16

    with pytest.raises(InvalidRequestError, match="is not supported"):
        _run(provider.publish(request))


def test_caption_limit_is_enforced() -> None:
    provider = _provider(lambda request: httpx.Response(200, json={}))
    request = _request(
        caption="x" * 2201,
        media=[_media(url="https://example.com/image.jpg")],
    )

    with pytest.raises(InvalidRequestError) as excinfo:
        _run(provider.publish(request))
    assert excinfo.value.details["field"] == "caption"


def test_hashtag_limit_is_enforced() -> None:
    provider = _provider(lambda request: httpx.Response(200, json={}))
    request = _request(
        hashtags=[f"tag{i}" for i in range(31)],
        media=[_media(url="https://example.com/image.jpg")],
    )

    with pytest.raises(InvalidRequestError) as excinfo:
        _run(provider.publish(request))
    assert excinfo.value.details["field"] == "hashtags"


def test_successful_single_image_publish_returns_a_live_result() -> None:
    requests, calls, handler = _capture()
    provider = _provider(handler)
    request = _request(
        caption="A run is waiting",
        hashtags=["launch", "video"],
        media=[_media(url="https://example.com/image.jpg")],
    )

    result = _run(provider.publish(request))

    assert calls[0] == 3  # account GET, create container POST, media_publish POST
    assert len(requests) == 3  # account GET, create container POST, media_publish POST
    create_requests = [
        r
        for r in requests
        if r.method == "POST"
        and "/media" in str(r.url)
        and "media_publish" not in str(r.url)
    ]
    publish_requests = [r for r in requests if "media_publish" in str(r.url)]
    assert len(create_requests) == 1
    assert len(publish_requests) == 1
    assert (
        create_requests[0]
        .headers["content-type"]
        .startswith("application/x-www-form-urlencoded")
    )
    body = parse_qs(create_requests[0].content.decode())
    assert body["image_url"] == ["https://example.com/image.jpg"]
    caption = body["caption"][0]
    assert "A run is waiting" in caption
    assert "#launch" in caption
    assert "#video" in caption

    assert result.is_live is True
    assert result.remote_id == MEDIA_ID
    assert result.url == f"https://www.instagram.com/p/{MEDIA_ID}/"
    assert result.provider == "instagram"
    assert result.platform == "instagram"
    assert result.published_at is not None
    assert result.published_at.tzinfo is not None
    assert result.dry_run is False
    assert result.metadata["method"] == "container"
    assert result.metadata["container_type"] == "image"


def test_successful_carousel_publish_returns_a_live_result() -> None:
    requests, _, handler = _capture()
    provider = _provider(handler)
    request = _request(
        caption="A carousel post",
        format="carousel",
        media=[
            _media(url="https://example.com/child-1.jpg"),
            _media(url="https://example.com/child-2.jpg"),
        ],
    )

    result = _run(provider.publish(request))

    create_requests = [
        r
        for r in requests
        if r.method == "POST"
        and "/media" in str(r.url)
        and "media_publish" not in str(r.url)
    ]
    publish_requests = [r for r in requests if "media_publish" in str(r.url)]
    assert len(create_requests) == 3  # two children + one carousel container
    assert len(publish_requests) == 1

    carousel_request = create_requests[-1]
    body = parse_qs(carousel_request.content.decode())
    assert body["media_type"] == ["CAROUSEL"]
    children = body["children"][0]
    assert "creation-1-1" in children and "creation-1-2" in children

    assert result.is_live is True
    assert result.remote_id == MEDIA_ID
    assert result.metadata["container_type"] == "carousel"


def test_carousel_is_detected_when_multiple_images_are_sent() -> None:
    requests, _, handler = _capture()
    provider = _provider(handler)
    request = _request(
        media=[
            _media(url="https://example.com/child-1.jpg"),
            _media(url="https://example.com/child-2.jpg"),
        ],
    )

    _run(provider.publish(request))

    create_requests = [
        r
        for r in requests
        if r.method == "POST"
        and "/media" in str(r.url)
        and "media_publish" not in str(r.url)
    ]
    assert len(create_requests) == 3


def test_video_publish_uses_a_reels_container() -> None:
    requests, _, handler = _capture()
    provider = _provider(handler)
    request = _request(
        media=[_media(url="https://example.com/reel.mp4", media_type="video/mp4")]
    )

    _run(provider.publish(request))

    create_requests = [
        r
        for r in requests
        if r.method == "POST"
        and "/media" in str(r.url)
        and "media_publish" not in str(r.url)
    ]
    body = create_requests[0].content.decode()
    assert "media_type=REELS" in body
    assert "video_url=https%3A%2F%2Fexample.com%2Freel.mp4" in body


def test_retries_transient_container_creation_then_succeeds() -> None:
    requests, _, handler = _capture(fail_containers=2)
    provider = _provider(handler)
    request = _request(media=[_media(url="https://example.com/image.jpg")])

    result = _run(provider.publish(request))

    assert result.is_live is True
    create_requests = [
        r
        for r in requests
        if r.method == "POST"
        and "/media" in str(r.url)
        and "media_publish" not in str(r.url)
    ]
    assert len(create_requests) == 3


def test_permanent_errors_are_not_retried() -> None:
    provider = _provider(
        lambda request: httpx.Response(
            401,
            json={"error": {"message": "Invalid OAuth access token", "code": 190}},
        )
    )

    with pytest.raises(AuthError):
        _run(
            provider.publish(
                _request(media=[_media(url="https://example.com/image.jpg")])
            )
        )


def test_rate_limited_container_creation_is_retried() -> None:
    responses = [
        httpx.Response(429, json={"error": {"message": "rate limited", "code": 80002}}),
        httpx.Response(200, json={"id": "creation-1"}),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if request.method == "GET" and IG_USER_ID in url and "/media" not in url:
            return httpx.Response(
                200, json={"account_type": "BUSINESS", "id": IG_USER_ID}
            )
        if request.method == "POST" and "media_publish" in url:
            return httpx.Response(200, json={"id": MEDIA_ID})
        if request.method == "POST" and f"{IG_USER_ID}/media" in url:
            return responses.pop(0)
        return httpx.Response(404)

    provider = _provider(handler)
    result = _run(
        provider.publish(_request(media=[_media(url="https://example.com/image.jpg")]))
    )

    assert result.is_live is True


def test_classify_error_maps_instagram_statuses() -> None:
    assert isinstance(classify_error("instagram", 401), AuthError)
    assert isinstance(classify_error("instagram", 403), AuthError)
    assert isinstance(classify_error("instagram", 429), RateLimitError)
    assert isinstance(classify_error("instagram", 400), InvalidRequestError)
    assert isinstance(classify_error("instagram", 500), RetryableError)
    assert classify_error("instagram", 401).details["platform"] == "instagram"


def test_health_check_reports_healthy_for_a_business_account() -> None:
    provider = _provider(
        lambda request: httpx.Response(
            200,
            json={"account_type": "BUSINESS", "id": IG_USER_ID, "username": "reviewer"},
        )
    )

    health = _run(provider.health())

    assert health.healthy is True
    assert health.status.value == "healthy"


def test_health_check_reports_unhealthy_for_a_personal_account() -> None:
    provider = _provider(
        lambda request: httpx.Response(
            200,
            json={"account_type": "PERSONAL", "id": IG_USER_ID, "username": "reviewer"},
        )
    )

    health = _run(provider.health())

    assert health.healthy is False
    assert health.status.value == "unhealthy"


def test_health_check_reports_unhealthy_when_account_check_fails() -> None:
    provider = _provider(
        lambda request: httpx.Response(
            401, json={"error": {"message": "Invalid token", "code": 190}}
        )
    )

    health = _run(provider.health())

    assert health.healthy is False
    assert health.status.value == "unhealthy"


def test_pricing_is_zero_cost() -> None:
    provider = _provider(lambda request: httpx.Response(200, json={}))
    price = provider.metadata.pricing_for(CAPABILITY_PUBLISH)
    assert isinstance(price, ProviderPrice)
    assert price.amount_usd == 0.0
