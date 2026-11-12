"""Idempotent publishing: a retry after a timeout never double posts.

Covers the :class:`~aivia.publish.IdempotentPublisher` behaviour:

* a published post is returned from its stored ``remote_id`` and ``url`` on a
  repeated idempotency key, without calling the provider again;
* only retryable errors (timeouts, rate limits) are retried, while permanent
  errors (auth, invalid request) fail immediately;
* the platform post id and url are persisted on the post record.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Iterator

import pytest
from sqlalchemy import Engine

from aivia.core.db import (
    PostRepository,
    create_database_engine,
    upgrade_database,
)
from aivia.core.errors import (
    AuthError,
    InvalidRequestError,
    RateLimitError,
    TimeoutError,
)
from aivia.core.models import Post, PostState
from aivia.core.publish_errors import PublishError
from aivia.providers.mock import MockPublisher
from aivia.providers.protocols.publish import (
    PublishMedia,
    PublishRequest,
    PublishResult,
)
from aivia.publish import IdempotentPublisher


class _CountingPublisher(MockPublisher):
    """A mock publisher that records every publish invocation, including failures."""

    def __init__(
        self,
        *,
        failures: Iterable[BaseException | type[BaseException]] | None = None,
    ) -> None:
        super().__init__(failures=failures)
        self.invocations = 0

    async def publish(self, request: PublishRequest) -> PublishResult:
        self.invocations += 1
        return await super().publish(request)


@pytest.fixture
def engine() -> Iterator[Engine]:
    created = create_database_engine(url="sqlite+pysqlite:///:memory:")
    upgrade_database(engine=created)
    try:
        yield created
    finally:
        created.dispose()


@pytest.fixture
def repository(engine: Engine) -> PostRepository:
    return PostRepository(engine)


def _request(
    *,
    key: str = "key-1",
    platform: str = "telegram",
    caption: str = "hello world",
) -> PublishRequest:
    return PublishRequest(
        platform=platform,
        account="@channel",
        caption=caption,
        media=[
            PublishMedia(
                url="https://example.com/output.mp4",
                media_type="video/mp4",
            )
        ],
        idempotency_key=key,
    )


def _post(repository: PostRepository, key: str) -> Post | None:
    return repository.find_by_idempotency_key(key)


# ---------------------------------------------------------------- success path


def test_publish_persists_remote_id_and_url(
    engine: Engine, repository: PostRepository
) -> None:
    publisher = _CountingPublisher()
    service = IdempotentPublisher(publisher, repository)
    request = _request()

    result = asyncio.run(service.publish(request))

    assert publisher.invocations == 1
    assert result.remote_id
    assert result.url is not None
    assert result.url.startswith("https://example.com/posts/")

    post = _post(repository, "key-1")
    assert post is not None
    assert post.state is PostState.PUBLISHED
    assert post.remote_id == result.remote_id
    assert post.url == result.url
    assert post.platform == "telegram"
    assert post.caption == "hello world"


def test_repeat_with_same_key_returns_stored_post_without_reposting(
    engine: Engine, repository: PostRepository
) -> None:
    publisher = _CountingPublisher()
    service = IdempotentPublisher(publisher, repository)
    request = _request()

    first = asyncio.run(service.publish(request))
    asyncio.run(service.publish(request))

    # the provider is called exactly once: the repeat returns the stored post
    assert publisher.invocations == 1
    assert first.remote_id
    posts = repository.list(state="published")
    assert len(posts) == 1
    assert posts[0].remote_id == first.remote_id


# ---------------------------------------------------------------- retryable errors


def test_timeout_is_retried_and_never_double_posts(
    engine: Engine, repository: PostRepository
) -> None:
    publisher = _CountingPublisher(
        failures=[TimeoutError("the publish deadline was exceeded")]
    )
    service = IdempotentPublisher(publisher, repository, max_attempts=3)
    request = _request()

    result = asyncio.run(service.publish(request))

    assert publisher.invocations == 2  # one timeout, then success
    assert result.remote_id
    assert result.url

    post = _post(repository, "key-1")
    assert post is not None
    assert post.state is PostState.PUBLISHED
    assert post.remote_id == result.remote_id
    assert post.url == result.url


def test_rate_limit_is_retried(engine: Engine, repository: PostRepository) -> None:
    publisher = _CountingPublisher(
        failures=[RateLimitError("rate limited", retry_after=1.0)]
    )
    service = IdempotentPublisher(publisher, repository, max_attempts=3)
    request = _request()

    asyncio.run(service.publish(request))

    assert publisher.invocations == 2
    post = _post(repository, "key-1")
    assert post is not None
    assert post.state is PostState.PUBLISHED


def test_retryable_errors_exhaust_retries(
    engine: Engine, repository: PostRepository
) -> None:
    publisher = _CountingPublisher(
        failures=[
            RateLimitError("rate limited", retry_after=1.0),
            RateLimitError("rate limited", retry_after=1.0),
            RateLimitError("rate limited", retry_after=1.0),
        ]
    )
    service = IdempotentPublisher(publisher, repository, max_attempts=3)
    request = _request()

    with pytest.raises(PublishError) as caught:
        asyncio.run(service.publish(request))

    assert publisher.invocations == 3
    assert caught.value.platform == "telegram"
    assert caught.value.details["error_code"] == "rate_limit"

    post = _post(repository, "key-1")
    assert post is not None
    assert post.state is PostState.FAILED
    assert post.error_code == "rate_limit"


# ---------------------------------------------------------------- permanent errors


def test_permanent_error_is_not_retried(
    engine: Engine, repository: PostRepository
) -> None:
    publisher = _CountingPublisher(failures=[AuthError("invalid credentials")])
    service = IdempotentPublisher(publisher, repository, max_attempts=5)
    request = _request()

    with pytest.raises(PublishError):
        asyncio.run(service.publish(request))

    assert publisher.invocations == 1  # not retried
    post = _post(repository, "key-1")
    assert post is not None
    assert post.state is PostState.FAILED
    assert post.error_code == "auth"
    assert post.remote_id == ""
    assert post.url == ""


def test_invalid_request_is_not_retried(
    engine: Engine, repository: PostRepository
) -> None:
    publisher = _CountingPublisher(
        failures=[InvalidRequestError("unsupported media type")]
    )
    service = IdempotentPublisher(publisher, repository, max_attempts=5)
    request = _request()

    with pytest.raises(PublishError):
        asyncio.run(service.publish(request))

    assert publisher.invocations == 1  # not retried
    post = _post(repository, "key-1")
    assert post is not None
    assert post.state is PostState.FAILED
    assert post.error_code == "invalid_request"


def test_repeat_after_permanent_failure_surfaces_stored_error(
    engine: Engine, repository: PostRepository
) -> None:
    publisher = _CountingPublisher(failures=[AuthError("invalid credentials")])
    service = IdempotentPublisher(publisher, repository, max_attempts=5)
    request = _request()

    with pytest.raises(PublishError):
        asyncio.run(service.publish(request))

    # repeating the same key must not call the provider again
    with pytest.raises(PublishError):
        asyncio.run(service.publish(request))

    assert publisher.invocations == 1


# ---------------------------------------------------------------- no idempotency key


def test_publish_without_key_has_no_dedupe(
    engine: Engine, repository: PostRepository
) -> None:
    publisher = _CountingPublisher()
    service = IdempotentPublisher(publisher, repository)
    request = _request(key="")

    asyncio.run(service.publish(request))
    asyncio.run(service.publish(request))

    assert publisher.invocations == 2
    # no dedupe: two distinct post rows are created
    posts = repository.list(state="published")
    assert len(posts) == 2
    assert len({post.id for post in posts}) == 2
