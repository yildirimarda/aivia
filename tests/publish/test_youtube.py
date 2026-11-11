"""YouTube publisher adapter tests.

Covers resumable uploads, retries, quota hints, metadata limits, disclosure
fields and the clear message emitted when an unaudited project forces private
uploads.
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
    QuotaError,
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
from aivia.providers.youtube_publish import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_ATTEMPTS,
    YouTubePublisher,
)

SESSION_URL = "https://www.googleapis.com/upload/youtube/v3/resumable/sessions/12345"
VIDEO_ID = "dQw4w9WgXcQ"
ACCESS_TOKEN = "ya29.a0AfH6SMB"


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
    media_type: str = "video/mp4",
    filename: str = "clip.mp4",
) -> PublishMedia:
    return PublishMedia(
        data=data, path=path, url=url, media_type=media_type, filename=filename
    )


def _request(
    *,
    platform: str = "youtube",
    title: str = "Review needed",
    caption: str = "A run is waiting",
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
    fail_initiations: int = 0,
    fail_uploads: int = 0,
    quota_on_initiate: bool = False,
    status_on_initiate: int | None = None,
) -> tuple[list[httpx.Request], list[int], Callable[[httpx.Request], httpx.Response]]:
    requests: list[httpx.Request] = []
    calls = [0]
    initiations = [0]
    uploads = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        calls[0] += 1
        url = str(request.url)
        if "tokeninfo" in url:
            return httpx.Response(200, json={})
        if request.method == "POST" and "upload/youtube/v3/videos" in url:
            initiations[0] += 1
            if quota_on_initiate:
                body = {
                    "error": {
                        "errors": [
                            {
                                "domain": "youtube.quota",
                                "reason": "quotaExceeded",
                                "message": (
                                    "The request cannot be completed because you "
                                    "have exceeded your quota."
                                ),
                            }
                        ],
                        "code": 403,
                        "message": "quota exceeded",
                    }
                }
                return httpx.Response(403, json=body)
            if status_on_initiate is not None:
                return httpx.Response(
                    status_on_initiate, json={"error": {"message": "boom"}}
                )
            if initiations[0] <= fail_initiations:
                return httpx.Response(500, json={"error": {"message": "transient"}})
            return httpx.Response(200, headers={"Location": SESSION_URL})
        if request.method == "PUT":
            uploads[0] += 1
            if uploads[0] <= fail_uploads:
                return httpx.Response(500, json={"error": {"message": "transient"}})
            return httpx.Response(200, json={"id": VIDEO_ID})
        return httpx.Response(404, json={"error": {"message": "not found"}})

    return requests, calls, handler


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _request_body(request: httpx.Request) -> Any:
    return json.loads(request.content)


def _provider(
    handler: Any,
    *,
    access_token: str | None = ACCESS_TOKEN,
    transport: httpx.AsyncBaseTransport | None = None,
    **kwargs: Any,
) -> YouTubePublisher:
    options: dict[str, Any] = {
        "access_token": access_token,
        "transport": transport or httpx.MockTransport(handler),
        "max_attempts": DEFAULT_MAX_ATTEMPTS,
        "retry_delay_seconds": 0.0,
    }
    options.update(kwargs)
    return YouTubePublisher(**options)


def test_metadata_and_registration_describe_the_provider() -> None:
    _requests, _calls, handler = _capture()
    provider = _provider(handler)

    assert provider.name == "youtube"
    assert provider.capabilities == (CAPABILITY_PUBLISH,)
    assert provider.base_url == DEFAULT_BASE_URL
    assert provider_registry.resolve("youtube") is YouTubePublisher
    assert provider_registry.resolve("youtube_publish") is YouTubePublisher


def test_missing_access_token_is_rejected() -> None:
    with pytest.raises(InvalidRequestError):
        YouTubePublisher(
            transport=httpx.MockTransport(lambda request: httpx.Response(200))
        )


def test_access_token_resolved_from_reference() -> None:
    requests, _, handler = _capture()
    secrets = SecretsLoader(environ={"AIVA_YOUTUBE_ACCESS_TOKEN": ACCESS_TOKEN})
    provider = _provider(
        handler,
        access_token=None,
        access_token_ref="AIVA_YOUTUBE_ACCESS_TOKEN",
        secrets=secrets,
    )

    _run(provider.publish(_request(media=[_media(data=b"video-bytes")])))

    assert requests and requests[0].headers["authorization"] == f"Bearer {ACCESS_TOKEN}"


def test_dry_run_returns_a_not_live_result_without_uploading() -> None:
    requests, calls, handler = _capture()
    provider = _provider(handler)
    request = _request(dry_run=True, media=[_media(data=b"video-bytes")])

    result = _run(provider.publish(request))

    assert calls[0] == 0
    assert requests == []
    assert result.is_live is False
    assert result.remote_id == ""
    assert result.dry_run is True
    assert result.provider == "youtube"


def test_successful_resumable_upload_returns_a_live_result() -> None:
    requests, calls, handler = _capture()
    provider = _provider(handler)
    request = _request(
        title="Review needed",
        caption="A run is waiting",
        hashtags=["launch", "video"],
        visibility=PostVisibility.PUBLIC,
        attestation=_attestation(complete=True),
        media=[_media(data=b"video-bytes")],
    )

    result = _run(provider.publish(request))

    assert calls[0] == 2
    assert requests[0].method == "POST"
    assert str(requests[0].url) == (
        f"{DEFAULT_BASE_URL}/upload/youtube/v3/videos"
        "?uploadType=resumable&part=snippet,status"
    )
    assert requests[0].headers["authorization"] == f"Bearer {ACCESS_TOKEN}"
    assert requests[0].headers["x-upload-content-type"] == "video/mp4"
    assert requests[0].headers["x-upload-content-length"] == str(len(b"video-bytes"))
    assert requests[0].headers["content-type"].startswith("application/json")
    body = _request_body(requests[0])
    assert body["snippet"]["title"] == "Review needed"
    assert body["snippet"]["description"].startswith("A run is waiting")
    assert "This content was generated with AI." in body["snippet"]["description"]
    assert body["snippet"]["tags"] == ["launch", "video"]
    assert body["status"]["privacyStatus"] == "public"

    assert requests[1].method == "PUT"
    assert str(requests[1].url) == SESSION_URL
    assert requests[1].content == b"video-bytes"
    assert requests[1].headers["content-range"] == "bytes 0-10/11"

    assert result.is_live is True
    assert result.remote_id == VIDEO_ID
    assert result.url == f"https://www.youtube.com/watch?v={VIDEO_ID}"
    assert result.provider == "youtube"
    assert result.platform == "youtube"
    assert result.published_at is not None
    assert result.published_at.tzinfo is not None
    assert result.dry_run is False
    assert result.metadata["method"] == "resumable_upload"
    assert result.metadata["privacy"] == "public"
    assert result.metadata["size_bytes"] == len(b"video-bytes")
    assert result.metadata["attempts"] == 1


def test_upload_from_a_local_file_path() -> None:
    requests, _, handler = _capture()
    provider = _provider(handler)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(b"file-video-bytes")
        path = tmp.name
    try:
        request = _request(media=[_media(path=path, filename="clip.mp4")])
        result = _run(provider.publish(request))

        assert result.is_live is True
        assert requests[1].content == b"file-video-bytes"
    finally:
        Path(path).unlink()


def test_media_must_be_video() -> None:
    provider = _provider(lambda request: httpx.Response(200, json={}))
    request = _request(media=[_media(media_type="image/png", data=b"not-video")])

    with pytest.raises(InvalidRequestError, match="video"):
        _run(provider.publish(request))


def test_url_media_is_not_supported() -> None:
    provider = _provider(lambda request: httpx.Response(200, json={}))
    request = _request(media=[_media(url="https://example.com/clip.mp4")])

    with pytest.raises(InvalidRequestError, match="url media is not supported"):
        _run(provider.publish(request))


def test_no_media_is_rejected() -> None:
    provider = _provider(lambda request: httpx.Response(200, json={}))

    with pytest.raises(InvalidRequestError, match="requires exactly one video media"):
        _run(provider.publish(_request()))


def test_publish_requires_a_publish_request() -> None:
    provider = _provider(lambda request: httpx.Response(200, json={}))

    with pytest.raises(TypeError, match="PublishRequest"):
        _run(provider.publish(cast(PublishRequest, "not a request")))


def test_metadata_limits_are_enforced() -> None:
    provider = _provider(lambda request: httpx.Response(200, json={}))

    long_title = "x" * 101
    with pytest.raises(InvalidRequestError) as excinfo:
        _run(provider.publish(_request(title=long_title, media=[_media(data=b"v")])))
    assert excinfo.value.details["platform"] == "youtube"
    assert any(p["field"] == "title" for p in excinfo.value.details["problems"])

    long_caption = "x" * 5001
    with pytest.raises(InvalidRequestError) as excinfo:
        _run(
            provider.publish(_request(caption=long_caption, media=[_media(data=b"v")]))
        )
    assert any(p["field"] == "description" for p in excinfo.value.details["problems"])

    too_many_hashtags = [f"tag{i}" for i in range(16)]
    with pytest.raises(InvalidRequestError) as excinfo:
        _run(
            provider.publish(
                _request(hashtags=too_many_hashtags, media=[_media(data=b"v")])
            )
        )
    assert any(p["field"] == "hashtags" for p in excinfo.value.details["problems"])


def test_metadata_limits_are_not_enforced_when_within_limits() -> None:
    _requests, _, handler = _capture()
    provider = _provider(handler)
    request = _request(
        title="x" * 100,
        caption="x" * 5000,
        hashtags=[f"tag{i}" for i in range(15)],
        media=[_media(data=b"v")],
    )
    result = _run(provider.publish(request))
    assert result.is_live is True


def test_unaudited_project_forces_private_upload() -> None:
    requests, calls, handler = _capture()
    provider = _provider(handler)
    request = _request(
        visibility=PostVisibility.PUBLIC,
        media=[_media(data=b"video-bytes")],
        # no attestation -> unaudited
    )

    result = _run(provider.publish(request))

    assert calls[0] == 2
    body = _request_body(requests[0])
    assert body["status"]["privacyStatus"] == "private"
    assert result.metadata["visibility_forced_private"] is True
    assert "disclosure_message" in result.metadata
    assert "unaudited" in result.metadata["disclosure_message"].lower()
    assert "private" in result.metadata["disclosure_message"].lower()
    assert result.metadata["privacy"] == "private"
    assert result.remote_id == VIDEO_ID
    assert result.is_live is True


def test_unaudited_project_appends_disclosure_to_description() -> None:
    requests, _, handler = _capture()
    provider = _provider(handler)
    request = _request(caption="A run is waiting", media=[_media(data=b"v")])

    _run(provider.publish(request))

    body = _request_body(requests[0])
    assert "This content was generated with AI." in body["snippet"]["description"]
    assert body["snippet"]["description"].startswith("A run is waiting")
    assert body["status"]["privacyStatus"] == "private"


def test_metadata_field_records_synthetic_media() -> None:
    requests, _, handler = _capture()
    provider = _provider(handler)
    _run(provider.publish(_request(media=[_media(data=b"v")])))

    assert _request_body(requests[0])["snippet"]["description"].endswith(
        "This content was generated with AI."
    )


def test_audited_project_keeps_requested_visibility() -> None:
    requests, _, handler = _capture()
    provider = _provider(handler)
    request = _request(
        visibility=PostVisibility.PUBLIC,
        attestation=_attestation(complete=True),
        media=[_media(data=b"v")],
    )

    result = _run(provider.publish(request))

    body = _request_body(requests[0])
    assert body["status"]["privacyStatus"] == "public"
    assert result.metadata.get("visibility_forced_private") is not True
    assert result.metadata["privacy"] == "public"


def test_disclosure_not_appended_when_already_present() -> None:
    requests, _, handler = _capture()
    provider = _provider(handler)
    disclosure = "This content was generated with AI."
    request = _request(
        caption=f"Intro {disclosure} outro",
        attestation=_attestation(complete=True),
        media=[_media(data=b"v")],
    )

    _run(provider.publish(request))

    body = _request_body(requests[0])
    assert body["snippet"]["description"].count(disclosure) == 1
    assert body["status"]["privacyStatus"] == "public"


def test_retries_transient_initiation_then_succeeds() -> None:
    requests, _calls, handler = _capture(fail_initiations=2)
    provider = _provider(handler)
    request = _request(media=[_media(data=b"v")])

    result = _run(provider.publish(request))

    assert result.is_live is True
    assert result.metadata["attempts"] == 3
    post_requests = [r for r in requests if r.method == "POST"]
    put_requests = [r for r in requests if r.method == "PUT"]
    assert len(post_requests) == 3
    assert len(put_requests) == 1


def test_retries_transient_upload_then_succeeds() -> None:
    requests, _, handler = _capture(fail_uploads=2)
    provider = _provider(handler)
    request = _request(media=[_media(data=b"v")])

    result = _run(provider.publish(request))

    assert result.is_live is True
    assert result.metadata["attempts"] == 3
    put_requests = [r for r in requests if r.method == "PUT"]
    assert len(put_requests) == 3


def test_permanent_errors_are_not_retried() -> None:
    requests, _, handler = _capture(status_on_initiate=401)
    provider = _provider(handler)

    with pytest.raises(AuthError):
        _run(provider.publish(_request(media=[_media(data=b"v")])))

    assert len(requests) == 1


def test_quota_exceeded_raises_quota_error_with_a_hint() -> None:
    requests, _, handler = _capture(quota_on_initiate=True)
    provider = _provider(handler)

    with pytest.raises(QuotaError) as excinfo:
        _run(provider.publish(_request(media=[_media(data=b"v")])))

    assert excinfo.value.details["platform"] == "youtube"
    assert excinfo.value.details.get("quota_hint") is True
    hint = str(excinfo.value)
    assert "quota" in hint.lower()
    assert "increase" in hint.lower() or "off-peak" in hint.lower()
    assert not excinfo.value.retryable
    assert len(requests) == 1


def test_classify_error_maps_youtube_statuses() -> None:
    assert isinstance(classify_error("youtube", 401), AuthError)
    assert isinstance(classify_error("youtube", 403), AuthError)
    assert isinstance(classify_error("youtube", 429), RateLimitError)
    assert isinstance(classify_error("youtube", 400), InvalidRequestError)
    assert isinstance(classify_error("youtube", 500), RetryableError)
    assert isinstance(classify_error("youtube", 402), QuotaError)
    assert classify_error("youtube", 401).details["platform"] == "youtube"


def test_health_check_reports_healthy_when_tokeninfo_succeeds() -> None:
    _requests, _, handler = _capture()
    provider = _provider(handler)

    health = _run(provider.health())

    assert health.healthy is True
    assert health.status.value == "healthy"


def test_health_check_reports_unhealthy_when_tokeninfo_fails() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "tokeninfo" in str(request.url):
            return httpx.Response(401, json={"error": {"message": "invalid token"}})
        return httpx.Response(200)

    provider = _provider(handler)

    health = _run(provider.health())

    assert health.healthy is False
    assert health.status.value == "unhealthy"


def test_pricing_is_zero_cost() -> None:
    provider = _provider(lambda request: httpx.Response(200, json={}))
    price = provider.metadata.pricing_for(CAPABILITY_PUBLISH)
    assert isinstance(price, ProviderPrice)
    assert price.amount_usd == 0.0
