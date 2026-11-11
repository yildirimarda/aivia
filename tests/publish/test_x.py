"""X publisher adapter tests.

Covers chunked media upload (init, append, finalize), post creation, metadata
limits, disclosure fields, retries, the tier limit hint and the clear message
emitted when an unaudited project forces private posts.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

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
from aivia.providers.protocols.publish import (
    CAPABILITY_PUBLISH,
    PostVisibility,
    PublishMedia,
    PublishRequest,
)
from aivia.providers.x_publish import (
    DEFAULT_BASE_URL,
    DEFAULT_CHUNK_SIZE_BYTES,
    DEFAULT_MAX_ATTEMPTS,
    XPublisher,
)

UPLOAD_URL = f"{DEFAULT_BASE_URL}/2/media/upload"
TWEET_URL = f"{DEFAULT_BASE_URL}/2/tweets"
MEDIA_ID = "17322345678901234567"
TWEET_ID = "17322345678901234568"
ACCESS_TOKEN = "session.x.token"

_INIT_CONTAINS = "command=INIT"
_APPEND_CONTAINS = "command=APPEND"
_FINALIZE_CONTAINS = "command=FINALIZE"
_TWEET_CONTAINS = "/2/tweets"


def _attestation(*, complete: bool = True) -> Attestation:
    if not complete:
        return Attestation()
    return Attestation(
        confirmed_by="reviewer",
        rights_confirmed_by="reviewer",
        disclosure_confirmed_by="reviewer",
        rights_confirmed=True,
        disclosure_confirmed=True,
        confirmed_at=datetime.now(UTC),
        disclosure_text="This content was generated with AI.",
        license_summary="CC0",
    )


def _media(
    *,
    data: bytes | None = None,
    path: str | None = None,
    url: str | None = None,
    media_type: str = "image/png",
    filename: str = "clip.png",
) -> PublishMedia:
    return PublishMedia(
        data=data, path=path, url=url, media_type=media_type, filename=filename
    )


def _request(
    *,
    platform: str = "x",
    title: str = "Review needed",
    caption: str = "A post is waiting",
    hashtags: list[str] | None = None,
    visibility: PostVisibility = PostVisibility.PUBLIC,
    dry_run: bool = False,
    attestation: Attestation | None = None,
    media: list[PublishMedia] | None = None,
) -> PublishRequest:
    return PublishRequest(
        platform=platform,
        title=title,
        caption=caption,
        hashtags=hashtags or [],
        visibility=visibility,
        dry_run=dry_run,
        attestation=attestation,
        media=media or [],
    )


def _capture(
    *,
    fail_inits: int = 0,
    fail_appends: int = 0,
    fail_finalizes: int = 0,
) -> tuple[list[httpx.Request], list[int], Callable[[httpx.Request], httpx.Response]]:
    requests: list[httpx.Request] = []
    calls = [0]
    inits = [0]
    appends = [0]
    finalizes = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        calls[0] += 1
        url = str(request.url)
        if request.method == "GET" and url == f"{DEFAULT_BASE_URL}/":
            return httpx.Response(200, json={})
        if request.method == "POST" and _INIT_CONTAINS in url:
            inits[0] += 1
            if inits[0] <= fail_inits:
                return httpx.Response(500, json={"error": {"message": "transient"}})
            return httpx.Response(200, json={"media_id": MEDIA_ID})
        if request.method == "POST" and _APPEND_CONTAINS in url:
            appends[0] += 1
            if appends[0] <= fail_appends:
                return httpx.Response(500, json={"error": {"message": "transient"}})
            return httpx.Response(204)
        if request.method == "POST" and _FINALIZE_CONTAINS in url:
            finalizes[0] += 1
            if finalizes[0] <= fail_finalizes:
                return httpx.Response(500, json={"error": {"message": "transient"}})
            return httpx.Response(200, json={"media_id": MEDIA_ID})
        if request.method == "POST" and _TWEET_CONTAINS in url:
            return httpx.Response(201, json={"data": {"id": TWEET_ID}})
        return httpx.Response(404, json={"error": {"message": "not found"}})

    return requests, calls, handler


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _provider(
    handler: Any,
    *,
    access_token: str | None = ACCESS_TOKEN,
    transport: httpx.AsyncBaseTransport | None = None,
    **kwargs: Any,
) -> XPublisher:
    options: dict[str, Any] = {
        "access_token": access_token,
        "transport": transport or httpx.MockTransport(handler),
        "max_attempts": DEFAULT_MAX_ATTEMPTS,
        "retry_delay_seconds": 0.0,
    }
    options.update(kwargs)
    return XPublisher(**options)


def test_metadata_and_registration_describe_the_provider() -> None:
    _requests, _calls, handler = _capture()
    provider = _provider(handler)

    assert provider.name == "x"
    assert provider.capabilities == (CAPABILITY_PUBLISH,)
    assert provider.base_url == DEFAULT_BASE_URL
    assert provider.tier == "essential"
    assert provider.chunk_size_bytes == DEFAULT_CHUNK_SIZE_BYTES
    assert provider_registry.resolve("x") is XPublisher
    assert provider_registry.resolve("x_publish") is XPublisher


def test_missing_access_token_is_rejected() -> None:
    with pytest.raises(InvalidRequestError):
        XPublisher(transport=httpx.MockTransport(lambda request: httpx.Response(200)))


def test_access_token_resolved_from_reference() -> None:
    requests, _calls, handler = _capture()
    secrets = SecretsLoader(environ={"AIVA_X_ACCESS_TOKEN": ACCESS_TOKEN})
    provider = _provider(
        handler,
        access_token=None,
        access_token_ref="AIVA_X_ACCESS_TOKEN",
        secrets=secrets,
    )

    _run(provider.publish(_request(media=[_media(data=b"media-bytes")])))

    assert requests and requests[0].headers["authorization"] == f"Bearer {ACCESS_TOKEN}"


def test_dry_run_returns_a_not_live_result_without_uploading() -> None:
    requests, calls, handler = _capture()
    provider = _provider(handler)
    request = _request(dry_run=True, media=[_media(data=b"media-bytes")])

    result = _run(provider.publish(request))

    assert calls[0] == 0
    assert requests == []
    assert result.is_live is False
    assert result.remote_id == ""
    assert result.dry_run is True
    assert result.provider == "x"


def test_successful_chunked_upload_and_post_returns_a_live_result() -> None:
    requests, calls, handler = _capture()
    provider = _provider(handler)
    request = _request(
        title="Review needed",
        caption="A post is waiting",
        hashtags=["launch", "video"],
        visibility=PostVisibility.PUBLIC,
        attestation=_attestation(complete=True),
        media=[_media(data=b"media-bytes")],
    )

    result = _run(provider.publish(request))

    assert calls[0] == 4
    assert requests[0].method == "POST"
    assert _INIT_CONTAINS in str(requests[0].url)
    assert requests[0].headers["authorization"] == f"Bearer {ACCESS_TOKEN}"

    assert requests[1].method == "POST"
    assert str(requests[1].url).startswith(f"{UPLOAD_URL}?{_APPEND_CONTAINS}")
    assert requests[1].content == b"media-bytes"

    assert requests[2].method == "POST"
    assert str(requests[2].url).startswith(f"{UPLOAD_URL}?{_FINALIZE_CONTAINS}")

    assert requests[3].method == "POST"
    assert str(requests[3].url) == TWEET_URL
    body = json.loads(requests[3].content)
    assert body["text"].startswith("A post is waiting")
    assert "This content was generated with AI." in body["text"]
    assert body["media"]["media_ids"] == [MEDIA_ID]

    assert result.is_live is True
    assert result.remote_id == TWEET_ID
    assert result.url == f"https://x.com/i/status/{TWEET_ID}"
    assert result.provider == "x"
    assert result.platform == "x"
    assert result.published_at is not None
    assert result.published_at.tzinfo is not None
    assert result.dry_run is False
    assert result.metadata["method"] == "chunked_upload_post"
    assert result.metadata["privacy"] == "public"
    assert result.metadata["size_bytes"] == len(b"media-bytes")
    assert result.metadata["media_id"] == MEDIA_ID
    assert result.metadata["upload_attempts"] == 1


def test_upload_from_a_local_file_path() -> None:
    requests, _calls, handler = _capture()
    provider = _provider(handler)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(b"file-media-bytes")
        path = tmp.name
    try:
        request = _request(media=[_media(path=path, filename="clip.png")])
        result = _run(provider.publish(request))

        assert result.is_live is True
        assert requests[1].content == b"file-media-bytes"
    finally:
        Path(path).unlink()


def test_media_must_be_image_or_video() -> None:
    provider = _provider(lambda request: httpx.Response(200, json={}))
    request = _request(media=[_media(media_type="audio/mp3", data=b"not-media")])

    with pytest.raises(InvalidRequestError, match="image or video"):
        _run(provider.publish(request))


def test_url_media_is_not_supported() -> None:
    provider = _provider(lambda request: httpx.Response(200, json={}))
    request = _request(media=[_media(url="https://example.com/clip.png")])

    with pytest.raises(InvalidRequestError, match="url media is not supported"):
        _run(provider.publish(request))


def test_no_media_is_rejected() -> None:
    provider = _provider(lambda request: httpx.Response(200, json={}))

    with pytest.raises(InvalidRequestError, match="requires exactly one media"):
        _run(provider.publish(_request()))


def test_publish_requires_a_publish_request() -> None:
    provider = _provider(lambda request: httpx.Response(200, json={}))

    with pytest.raises(TypeError, match="PublishRequest"):
        _run(provider.publish(cast(PublishRequest, "not a request")))


def test_metadata_limits_are_enforced() -> None:
    provider = _provider(lambda request: httpx.Response(200, json={}))

    long_caption = "x" * 281
    with pytest.raises(InvalidRequestError) as excinfo:
        _run(
            provider.publish(_request(caption=long_caption, media=[_media(data=b"m")]))
        )
    assert excinfo.value.details["platform"] == "x"
    assert any(p["field"] == "caption" for p in excinfo.value.details["problems"])

    too_many_hashtags = [f"tag{i}" for i in range(11)]
    with pytest.raises(InvalidRequestError) as excinfo:
        _run(
            provider.publish(
                _request(hashtags=too_many_hashtags, media=[_media(data=b"m")])
            )
        )
    assert any(p["field"] == "hashtags" for p in excinfo.value.details["problems"])


def test_metadata_limits_are_not_enforced_when_within_limits() -> None:
    _requests, _calls, handler = _capture()
    provider = _provider(handler)
    request = _request(
        caption="x" * 280,
        hashtags=[f"tag{i}" for i in range(10)],
        media=[_media(data=b"m")],
    )
    result = _run(provider.publish(request))
    assert result.is_live is True


def test_unaudited_project_forces_private_upload() -> None:
    requests, calls, handler = _capture()
    provider = _provider(handler)
    request = _request(
        visibility=PostVisibility.PUBLIC,
        media=[_media(data=b"media-bytes")],
        # no attestation -> unaudited
    )

    result = _run(provider.publish(request))

    assert calls[0] == 4
    assert requests[3].method == "POST"
    assert _TWEET_CONTAINS in str(requests[3].url)
    assert result.metadata["visibility_forced_private"] is True
    assert "disclosure_message" in result.metadata
    assert "unaudited" in result.metadata["disclosure_message"].lower()
    assert "private" in result.metadata["disclosure_message"].lower()
    assert result.metadata["privacy"] == "private"
    assert result.remote_id == TWEET_ID
    assert result.is_live is True


def test_unaudited_project_appends_disclosure_to_description() -> None:
    requests, _calls, handler = _capture()
    provider = _provider(handler)
    request = _request(caption="A post is waiting", media=[_media(data=b"m")])

    _run(provider.publish(request))

    body = json.loads(requests[3].content)
    assert "This content was generated with AI." in body["text"]
    assert body["text"].startswith("A post is waiting")


def test_audited_project_keeps_requested_visibility() -> None:
    _requests, _calls, handler = _capture()
    provider = _provider(handler)
    request = _request(
        visibility=PostVisibility.PUBLIC,
        attestation=_attestation(complete=True),
        media=[_media(data=b"m")],
    )

    result = _run(provider.publish(request))

    assert result.metadata.get("visibility_forced_private") is not True
    assert result.metadata["privacy"] == "public"


def test_disclosure_not_appended_when_already_present() -> None:
    requests, _calls, handler = _capture()
    provider = _provider(handler)
    disclosure = "This content was generated with AI."
    request = _request(
        caption=f"Intro {disclosure} outro",
        attestation=_attestation(complete=True),
        media=[_media(data=b"m")],
    )

    _run(provider.publish(request))

    body = json.loads(requests[3].content)
    assert body["text"].count(disclosure) == 1


def test_retries_transient_initiation_then_succeeds() -> None:
    requests, _calls, handler = _capture(fail_inits=2)
    provider = _provider(handler)
    request = _request(media=[_media(data=b"m")])

    result = _run(provider.publish(request))

    assert result.is_live is True
    assert result.metadata["upload_attempts"] == 3
    init_requests = [r for r in requests if _INIT_CONTAINS in str(r.url)]
    assert len(init_requests) == 3


def test_retries_transient_append_then_succeeds() -> None:
    requests, _calls, handler = _capture(fail_appends=2)
    provider = _provider(handler)
    request = _request(media=[_media(data=b"m")])

    result = _run(provider.publish(request))

    assert result.is_live is True
    assert result.metadata["upload_attempts"] == 3
    append_requests = [r for r in requests if _APPEND_CONTAINS in str(r.url)]
    assert len(append_requests) == 3


def test_permanent_errors_are_not_retried() -> None:
    requests, _calls, _handler = _capture()

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if _INIT_CONTAINS in str(request.url):
            return httpx.Response(401, json={"error": {"message": "unauthorized"}})
        return httpx.Response(404, json={"error": {"message": "not found"}})

    provider = _provider(handler)

    with pytest.raises(AuthError):
        _run(provider.publish(_request(media=[_media(data=b"m")])))

    assert len(requests) == 1


def test_tier_limit_hint_is_none_within_limit() -> None:
    provider = _provider(lambda request: httpx.Response(200, json={}))
    assert provider.tier_limit_hint(1) is None
    assert provider.tier_limit_hint(5 * 1024 * 1024) is None


def test_tier_limit_hint_returns_a_hint_over_limit() -> None:
    provider = _provider(lambda request: httpx.Response(200, json={}))
    hint = provider.tier_limit_hint(5 * 1024 * 1024 + 1)
    assert hint is not None
    assert "essential" in hint
    assert "tier" in hint.lower()
    assert "upgrade" in hint.lower()


def test_media_over_tier_limit_is_rejected_with_hint() -> None:
    provider = _provider(lambda request: httpx.Response(200, json={}))
    oversized = b"x" * (5 * 1024 * 1024 + 1)
    request = _request(media=[_media(data=oversized)])

    with pytest.raises(InvalidRequestError) as excinfo:
        _run(provider.publish(request))
    assert excinfo.value.details["platform"] == "x"
    assert "hint" in excinfo.value.details
    assert "essential" in excinfo.value.details["hint"]


def test_classify_error_maps_x_statuses() -> None:
    assert isinstance(classify_error("x", 401), AuthError)
    assert isinstance(classify_error("x", 403), AuthError)
    assert isinstance(classify_error("x", 429), RateLimitError)
    assert isinstance(classify_error("x", 400), InvalidRequestError)
    assert isinstance(classify_error("x", 500), RetryableError)
    assert classify_error("x", 401).details["platform"] == "x"


def test_health_check_reports_healthy_when_root_succeeds() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and str(request.url) == f"{DEFAULT_BASE_URL}/":
            return httpx.Response(200, json={})
        return httpx.Response(404)

    provider = _provider(handler)

    health = _run(provider.health())

    assert health.healthy is True
    assert health.status.value == "healthy"


def test_health_check_reports_unhealthy_when_root_fails() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and str(request.url) == f"{DEFAULT_BASE_URL}/":
            return httpx.Response(500, json={"error": {"message": "boom"}})
        return httpx.Response(404)

    provider = _provider(handler)

    health = _run(provider.health())

    assert health.healthy is False
    assert health.status.value == "unhealthy"


def test_pricing_is_zero_cost() -> None:
    provider = _provider(lambda request: httpx.Response(200, json={}))
    price = provider.metadata.pricing_for(CAPABILITY_PUBLISH)
    assert isinstance(price, ProviderPrice)
    assert price.amount_usd == 0.0
