"""Idempotent publishing to platform accounts.

A publish is keyed by ``PublishRequest.idempotency_key``. Before calling the
underlying publisher the service checks whether a post with that key has
already reached a terminal state:

* a :attr:`PostState.PUBLISHED` post is returned from its stored ``remote_id``
  and ``url`` without calling the provider again, so a retry after a timeout
  can never double post;
* a :attr:`PostState.FAILED` post surfaces its stored error instead of being
  retried.

Only :class:`~aivia.core.errors.RetryableError` failures (which includes
:class:`~aivia.core.errors.TimeoutError` and :class:`~aivia.core.errors.RateLimitError`)
are retried; permanent errors such as :class:`~aivia.core.errors.AuthError` or
:class:`~aivia.core.errors.InvalidRequestError` are raised immediately.

The platform ``remote_id`` and ``url`` are persisted on the post record when a
publish succeeds, so a later retry can return them without a second remote call.
"""

from __future__ import annotations

from aivia.core.db.repos import PostRepository
from aivia.core.errors import AiviaError, PermanentError, RetryableError
from aivia.core.models import Post, PostState
from aivia.core.publish_errors import PublishError
from aivia.providers.protocols.publish import (
    Publisher,
    PublishMedia,
    PublishRequest,
    PublishResult,
)

__all__ = ["IdempotentPublisher"]


def _media_location(media: PublishMedia) -> str:
    """Return the storable location of one piece of publish media."""
    return media.path or media.url or ""


def _result_from_post(post: Post, publisher: Publisher) -> PublishResult:
    """Rebuild a :class:`PublishResult` from a stored published post."""

    return PublishResult(
        platform=post.platform,
        account=post.account,
        provider=publisher.name,
        remote_id=post.remote_id,
        url=post.url or None,
        published_at=post.published_at,
        dry_run=post.dry_run,
        metadata=dict(post.metadata or {}),
    )


class IdempotentPublisher:
    """Publish once per idempotency key, retrying only retryable errors.

    ``publisher`` is the underlying platform adapter. ``repository`` persists
    post records so repeated attempts with the same idempotency key can return
    the stored result instead of reposting. ``max_attempts`` bounds the number
    of retries after a retryable failure.
    """

    def __init__(
        self,
        publisher: Publisher,
        repository: PostRepository,
        *,
        max_attempts: int = 3,
    ) -> None:
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1: {max_attempts}")
        self._publisher = publisher
        self._repository = repository
        self._max_attempts = max_attempts

    @property
    def publisher(self) -> Publisher:
        """The underlying platform publisher."""
        return self._publisher

    @property
    def repository(self) -> PostRepository:
        """The post repository used to persist and look up posts."""
        return self._repository

    @property
    def max_attempts(self) -> int:
        """The maximum number of publish attempts, including retries."""
        return self._max_attempts

    async def publish(self, request: PublishRequest) -> PublishResult:
        """Publish *request*, returning a stored post when already published.

        An idempotency key that names a :attr:`PostState.PUBLISHED` post causes
        the stored ``remote_id`` and ``url`` to be returned without calling the
        provider. A key naming a :attr:`PostState.FAILED` post re-raises its
        stored error. Otherwise the provider is called, retrying only
        :class:`~aivia.core.errors.RetryableError` failures, and the resulting
        ``remote_id`` and ``url`` are persisted on success.
        """

        key = request.idempotency_key.strip()
        existing: Post | None = None
        post_id: str | None = None
        if key:
            existing = self._repository.find_by_idempotency_key(key)
            if existing is not None:
                if existing.state is PostState.PUBLISHED:
                    return _result_from_post(existing, self._publisher)
                if existing.state is PostState.FAILED:
                    raise PublishError(
                        existing.error_message or "previous publish attempt failed",
                        platform=existing.platform,
                        details={
                            "post_id": existing.id,
                            "platform": existing.platform,
                            "error_code": existing.error_code,
                            "idempotency_key": key,
                        },
                    )
                post_id = existing.id

        last_error: AiviaError | None = None
        for _ in range(self._max_attempts):
            if post_id is None:
                stored = self._repository.create(
                    Post(
                        platform=request.platform,
                        account=request.account,
                        profile=request.profile,
                        language=request.language,
                        format=request.format,
                        title=request.title,
                        caption=request.caption,
                        hashtags=list(request.hashtags),
                        mentions=list(request.mentions),
                        media=[_media_location(m) for m in request.media],
                        media_type=(
                            request.media[0].media_type or "" if request.media else ""
                        ),
                        visibility=request.visibility,
                        state=PostState.PUBLISHING,
                        idempotency_key=key,
                        dry_run=request.dry_run,
                        disclosure_text=request.disclosure_text,
                        metadata={"provider": self._publisher.name},
                    )
                )
                post_id = stored.id
            try:
                result = await self._publisher.publish(request)
            except RetryableError as exc:
                last_error = exc
                self._repository.update(
                    post_id,
                    state=PostState.FAILED,
                    error_code=exc.code,
                    error_message=str(exc),
                )
                continue
            except PermanentError as exc:
                self._repository.update(
                    post_id,
                    state=PostState.FAILED,
                    error_code=exc.code,
                    error_message=str(exc),
                )
                raise PublishError(
                    str(exc),
                    platform=request.platform,
                    details={
                        "post_id": post_id,
                        "platform": request.platform,
                        "error_code": exc.code,
                        "idempotency_key": key,
                    },
                ) from exc
            except AiviaError as exc:
                self._repository.update(
                    post_id,
                    state=PostState.FAILED,
                    error_code=exc.code,
                    error_message=str(exc),
                )
                raise PublishError(
                    str(exc),
                    platform=request.platform,
                    details={
                        "post_id": post_id,
                        "platform": request.platform,
                        "error_code": exc.code,
                        "idempotency_key": key,
                    },
                ) from exc
            self._repository.update(
                post_id,
                state=PostState.PUBLISHED,
                remote_id=result.remote_id,
                url=result.url or "",
                published_at=result.published_at,
                error_code="",
                error_message="",
            )
            return result

        raise PublishError(
            f"publish to {request.platform} failed after "
            f"{self._max_attempts} attempt(s)",
            platform=request.platform,
            details={
                "post_id": post_id,
                "platform": request.platform,
                "error_code": last_error.code if last_error else "",
                "idempotency_key": key,
                "attempts": self._max_attempts,
            },
        ) from last_error
