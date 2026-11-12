"""Publishing end to end against local mock servers.

Exercises idempotency (a retry after a timeout never double posts) and
backoff (retryable platform errors are retried until success or exhaustion)
using ``httpx.MockTransport`` as a local, offline mock server. No real network
call is ever made.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

import httpx
import pytest
from sqlalchemy import Engine

from aivia.core.db import PostRepository, create_database_engine, upgrade_database
from aivia.core.models import PostState
from aivia.core.models.io import Attestation
from aivia.core.publish_errors import PublishError
from aivia.providers.protocols.publish import (
    PostVisibility,
    PublishMedia,
    PublishRequest,
)
from aivia.providers.x_publish import DEFAULT_BASE_URL, XPublisher
from aivia.publish import IdempotentPublisher

UPLOAD_URL = f"{DEFAULT_BASE_URL}/2/media/upload"
TWEET_URL = f"{DEFAULT_BASE_URL}/2/tweets"
MEDIA_ID = "17322345678901234567"
TWEET_ID = "17322345678901234568"
ACCESS_TOKEN = "session.x.token"

_INIT = "command=INIT"
_APPEND = "command=APPEND"
_FINALIZE = "command=FINALIZE"
_TWEET = "/2/tweets"


def _attestation() -> Attestation:
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


def _request(
    *,
    key: str = "key-1",
    caption: str = "hello world",
) -> PublishRequest:
    return PublishRequest(
        platform="x",
        account="@account",
        caption=caption,
        media=[
            PublishMedia(
                data=b"media-bytes",
                media_type="image/png",
                filename="clip.png",
            )
        ],
        visibility=PostVisibility.PUBLIC,
        idempotency_key=key,
        attestation=_attestation(),
    )


def _capture(
    *,
    fail_inits: int = 0,
) -> tuple[list[httpx.Request], dict[str, int], httpx.AsyncBaseTransport]:
    requests: list[httpx.Request] = []
    counters = {"inits": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        url = str(request.url)
        if request.method == "GET" and url == DEFAULT_BASE_URL:
            return httpx.Response(200, json={})
        if request.method == "POST" and _INIT in url:
            counters["inits"] += 1
            if counters["inits"] <= fail_inits:
                return httpx.Response(429, json={"error": {"message": "rate limited"}})
            return httpx.Response(200, json={"media_id": MEDIA_ID})
        if request.method == "POST" and _APPEND in url:
            return httpx.Response(204)
        if request.method == "POST" and _FINALIZE in url:
            return httpx.Response(200, json={"media_id": MEDIA_ID})
        if request.method == "POST" and _TWEET in url:
            return httpx.Response(201, json={"data": {"id": TWEET_ID}})
        return httpx.Response(404, json={"error": {"message": "not found"}})

    return requests, counters, httpx.MockTransport(handler)


def _unauthorized() -> tuple[list[httpx.Request], httpx.AsyncBaseTransport]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if _INIT in str(request.url):
            return httpx.Response(401, json={"error": {"message": "unauthorized"}})
        return httpx.Response(404, json={"error": {"message": "not found"}})

    return requests, httpx.MockTransport(handler)


def _provider(
    transport: httpx.AsyncBaseTransport,
    *,
    max_attempts: int = 3,
    retry_delay_seconds: float = 0.0,
) -> XPublisher:
    return XPublisher(
        access_token=ACCESS_TOKEN,
        transport=transport,
        max_attempts=max_attempts,
        retry_delay_seconds=retry_delay_seconds,
    )


def _service(
    publisher: XPublisher,
    repository: PostRepository,
    *,
    max_attempts: int = 3,
) -> IdempotentPublisher:
    return IdempotentPublisher(publisher, repository, max_attempts=max_attempts)


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    created = create_database_engine(path=tmp_path / "data" / "aivia.db")
    upgrade_database(engine=created)
    try:
        yield created
    finally:
        created.dispose()


@pytest.fixture
def repository(engine: Engine) -> PostRepository:
    return PostRepository(engine)


T = TypeVar("T")


def _run[T](coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


# ---------------------------------------------------------------- success path


def test_publish_persists_remote_id_and_url(
    engine: Engine, repository: PostRepository
) -> None:
    requests, _counters, transport = _capture()
    provider = _provider(transport)
    service = _service(provider, repository)
    request = _request()

    result = _run(service.publish(request))

    assert result.is_live is True
    assert result.remote_id == TWEET_ID
    assert result.url == f"https://x.com/i/status/{TWEET_ID}"
    assert result.provider == "x"
    assert result.platform == "x"
    assert result.published_at is not None
    # one chunked upload: INIT, APPEND, FINALIZE, then the tweet
    assert len(requests) == 4
    assert any(_INIT in str(r.url) for r in requests)
    assert any(_APPEND in str(r.url) for r in requests)
    assert any(_FINALIZE in str(r.url) for r in requests)
    assert any(_TWEET in str(r.url) for r in requests)

    post = repository.find_by_idempotency_key("key-1")
    assert post is not None
    assert post.state is PostState.PUBLISHED
    assert post.remote_id == TWEET_ID
    assert post.url == result.url


def test_repeat_with_same_key_returns_stored_post_without_reposting(
    engine: Engine, repository: PostRepository
) -> None:
    requests, _counters, transport = _capture()
    provider = _provider(transport)
    service = _service(provider, repository)
    request = _request()

    first = _run(service.publish(request))
    second = _run(service.publish(request))

    # the provider is called exactly once: the repeat returns the stored post
    assert len(requests) == 4
    assert second.remote_id == first.remote_id
    assert second.url == first.url

    posts = repository.list(state="published")
    assert len(posts) == 1
    assert posts[0].remote_id == first.remote_id


# ---------------------------------------------------------------- backoff / retries


def test_rate_limit_is_retried_then_succeeds(
    engine: Engine, repository: PostRepository
) -> None:
    requests, counters, transport = _capture(fail_inits=1)
    provider = _provider(transport)
    service = _service(provider, repository)
    request = _request()

    result = _run(service.publish(request))

    assert result.is_live is True
    assert result.remote_id == TWEET_ID
    # the first INIT was rate limited, so the upload was retried (backed off)
    init_requests = [r for r in requests if _INIT in str(r.url)]
    assert len(init_requests) == 2
    assert counters["inits"] == 2

    post = repository.find_by_idempotency_key("key-1")
    assert post is not None
    assert post.state is PostState.PUBLISHED
    assert post.remote_id == TWEET_ID


def test_retryable_errors_exhaust_retries(
    engine: Engine, repository: PostRepository
) -> None:
    requests, _counters, transport = _capture(fail_inits=10)
    provider = _provider(transport, max_attempts=3)
    service = _service(provider, repository, max_attempts=2)
    request = _request()

    with pytest.raises(PublishError) as caught:
        _run(service.publish(request))

    assert caught.value.platform == "x"
    assert caught.value.details["error_code"] == "rate_limit"
    # every attempt exhausted its internal retries before the service gave up
    assert len(requests) == 3 * 2

    post = repository.find_by_idempotency_key("key-1")
    assert post is not None
    assert post.state is PostState.FAILED
    assert post.error_code == "rate_limit"


# ---------------------------------------------------------------- permanent errors


def test_auth_error_is_not_retried(engine: Engine, repository: PostRepository) -> None:
    requests, transport = _unauthorized()
    provider = _provider(transport, max_attempts=5)
    service = _service(provider, repository, max_attempts=5)
    request = _request()

    with pytest.raises(PublishError):
        _run(service.publish(request))

    # permanent errors are not retried: a single INIT attempt
    assert len(requests) == 1

    post = repository.find_by_idempotency_key("key-1")
    assert post is not None
    assert post.state is PostState.FAILED
    assert post.error_code == "auth"
    assert post.remote_id == ""
    assert post.url == ""


def test_repeat_after_permanent_failure_surfaces_stored_error_without_reposting(
    engine: Engine, repository: PostRepository
) -> None:
    requests, transport = _unauthorized()
    provider = _provider(transport, max_attempts=5)
    service = _service(provider, repository, max_attempts=5)
    request = _request()

    with pytest.raises(PublishError):
        _run(service.publish(request))

    # repeating the same key must not call the provider again
    with pytest.raises(PublishError):
        _run(service.publish(request))

    # only the first attempt reached the mock server
    assert len(requests) == 1

    post = repository.find_by_idempotency_key("key-1")
    assert post is not None
    assert post.state is PostState.FAILED
    assert post.error_code == "auth"
