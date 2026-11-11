"""X publisher using chunked media upload and post creation.

The adapter publishes a validated
:class:`~aivia.providers.protocols.publish.PublishRequest` to X in three steps:
it initiates a chunked media upload (``INIT``), sends the media bytes in
``APPEND`` segments, then finalizes the upload (``FINALIZE``), and finally
creates a post (``POST /2/tweets``) referencing the media id.

Transient failures (5xx, 429, timeouts and transport errors) are retried with
exponential backoff; permanent errors (400, 401, 403 auth) are raised
immediately.

A tier limit hint is returned when the media exceeds the configured X access
tier's maximum file size, so the caller knows which tier upgrade would allow
the post.

When the project is unaudited (the rights and AI disclosure attestation is not
complete) and X requires a disclosure, visibility is forced to ``private`` and
a clear message is recorded on the result so the caller knows the upload will
not be public.

Credentials are supplied through ``access_token`` or ``access_token_ref``
(resolved via :class:`~aivia.core.secrets.SecretsLoader`). ``transport``
injects an ``httpx`` mock transport for the offline suite.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx

from aivia.core.config.http import HttpClientConfig
from aivia.core.errors import (
    AiviaError,
    InvalidRequestError,
    RetryableError,
)
from aivia.core.models.base import utcnow
from aivia.core.models.provider import ProviderMeta, ProviderPrice
from aivia.core.platforms import PlatformCatalog
from aivia.core.publish_errors import error_for_result
from aivia.core.redact import redact
from aivia.core.secrets import SecretsLoader
from aivia.providers.base import HealthStatus, Provider, ProviderHealth
from aivia.providers.protocols.publish import (
    CAPABILITY_PUBLISH,
    PostVisibility,
    PublishMedia,
    PublishRequest,
    PublishResult,
)
from aivia.providers.registry import register_provider

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_CHUNK_SIZE_BYTES",
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_RETRY_DELAY_SECONDS",
    "DEFAULT_TIER",
    "XPublisher",
]

DEFAULT_BASE_URL = "https://api.x.com"
"""The X API v2 root, without a trailing slash."""

DEFAULT_CHUNK_SIZE_BYTES = 5 * 1024 * 1024
"""The default size of one media upload ``APPEND`` segment in bytes."""

DEFAULT_MAX_ATTEMPTS = 3
"""How many times a failed upload attempt is retried."""

DEFAULT_RETRY_DELAY_SECONDS = 1.0
"""Seconds to wait before the first retry; doubles on each retry."""

DEFAULT_TIER = "essential"
"""The default X access tier used when none is configured."""

_DEFAULT_CAPTION_LIMIT = 280
_DEFAULT_MAX_HASHTAGS = 10
_DEFAULT_DISCLOSURE_TEXT = "This content was generated with AI."
_DEFAULT_DISCLOSURE_REQUIRED = True
_DEFAULT_DISCLOSURE_METADATA_FIELD = "ai_content"

_TIER_MAX_MEDIA_BYTES: dict[str, int] = {
    "essential": 5 * 1024 * 1024,
    "elevated": 512 * 1024 * 1024,
    "basic": 512 * 1024 * 1024,
}

_INIT_PATH = "/2/media/upload"
_APPEND_PATH = "/2/media/upload"
_FINALIZE_PATH = "/2/media/upload"
_TWEET_PATH = "/2/tweets"


def _resolve_access_token(
    access_token: str | None,
    access_token_ref: str | None,
    secrets: SecretsLoader | None,
) -> str:
    """Resolve the access token from a literal value or a secret reference."""
    if access_token is not None:
        text = str(access_token).strip()
        if not text:
            raise InvalidRequestError("x access token must not be blank")
        return text
    if access_token_ref is None or not str(access_token_ref).strip():
        raise InvalidRequestError(
            "x publisher requires an access token or access_token_ref",
            details={"field": "access_token_ref"},
        )
    loader = secrets or SecretsLoader()
    return loader.get(str(access_token_ref).strip())


def _visibility_to_x(visibility: PostVisibility) -> str:
    """Map a :class:`PostVisibility` to an X tweet privacy setting."""
    if visibility is PostVisibility.PUBLIC:
        return "public"
    return "private"


def _tier_max_media_bytes(tier: str) -> int | None:
    """Return the maximum media size in bytes for *tier*, or ``None`` unbounded."""
    return _TIER_MAX_MEDIA_BYTES.get(tier.strip().lower())


def _tier_limit_hint(tier: str, size_bytes: int) -> str | None:
    """Return a hint when *size_bytes* exceeds the tier media cap, else ``None``."""
    limit = _tier_max_media_bytes(tier)
    if limit is None or size_bytes <= limit:
        return None
    return (
        f"X {tier} tier allows media up to {limit} bytes but {size_bytes} bytes "
        f"were provided. Upgrade your X access tier or reduce the media size "
        "so it fits the tier limit."
    )


class XPublisher(Provider):
    """A publisher that posts media and text to X via chunked upload."""

    default_metadata = ProviderMeta(
        name="x",
        capabilities=[CAPABILITY_PUBLISH],
        pricing={CAPABILITY_PUBLISH: ProviderPrice(amount_usd=0.0)},
    )

    def __init__(
        self,
        *,
        access_token: str | None = None,
        access_token_ref: str | None = None,
        secrets: SecretsLoader | None = None,
        base_url: str = DEFAULT_BASE_URL,
        transport: httpx.AsyncBaseTransport | None = None,
        name: str | None = None,
        health_timeout_seconds: float | None = None,
        close_timeout_seconds: float | None = None,
        http_config: HttpClientConfig | None = None,
        platforms: PlatformCatalog | None = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
        tier: str = DEFAULT_TIER,
        chunk_size_bytes: int = DEFAULT_CHUNK_SIZE_BYTES,
    ) -> None:
        if http_config is not None and not isinstance(http_config, HttpClientConfig):
            raise TypeError("http_config must be an HttpClientConfig")
        super().__init__(
            name=name,
            health_timeout_seconds=health_timeout_seconds,
            close_timeout_seconds=close_timeout_seconds,
            http_config=http_config,
        )
        if transport is not None and not isinstance(
            transport, httpx.AsyncBaseTransport
        ):
            raise TypeError("transport must be an httpx.AsyncBaseTransport")
        if max_attempts < 1:
            raise InvalidRequestError("max_attempts must be at least 1")
        if chunk_size_bytes < 1:
            raise InvalidRequestError("chunk_size_bytes must be at least 1")
        self._access_token = _resolve_access_token(
            access_token, access_token_ref, secrets
        )
        self._base_url = base_url.rstrip("/")
        self._platforms = platforms
        self._max_attempts = max_attempts
        self._retry_delay_seconds = float(retry_delay_seconds)
        self._tier = str(tier).strip() or DEFAULT_TIER
        self._chunk_size_bytes = int(chunk_size_bytes)
        if transport is not None:
            self._http = self.create_http_client(transport=transport)

    @property
    def base_url(self) -> str:
        """The configured X API v2 root, without a trailing slash."""
        return self._base_url

    @property
    def tier(self) -> str:
        """The X access tier whose media size cap is enforced."""
        return self._tier

    @property
    def chunk_size_bytes(self) -> int:
        """The size of one media upload ``APPEND`` segment in bytes."""
        return self._chunk_size_bytes

    def tier_limit_hint(self, size_bytes: int) -> str | None:
        """Return a hint when *size_bytes* exceeds the tier media cap, else ``None``."""
        return _tier_limit_hint(self._tier, size_bytes)

    def _x_limits(self) -> dict[str, Any]:
        """Return the x metadata and disclosure limits in force."""
        if self._platforms is not None:
            platform = self._platforms.get("x")
            if platform is not None:
                caption = platform.caption_limits
                disclosure = platform.disclosure_rules
                return {
                    "caption": caption.max_length,
                    "hashtags": caption.max_hashtags,
                    "disclosure_required": disclosure.required,
                    "disclosure_text": disclosure.text,
                    "metadata_field": disclosure.metadata_field,
                    "metadata_supported": disclosure.metadata,
                }
        return {
            "caption": _DEFAULT_CAPTION_LIMIT,
            "hashtags": _DEFAULT_MAX_HASHTAGS,
            "disclosure_required": _DEFAULT_DISCLOSURE_REQUIRED,
            "disclosure_text": _DEFAULT_DISCLOSURE_TEXT,
            "metadata_field": _DEFAULT_DISCLOSURE_METADATA_FIELD,
            "metadata_supported": True,
        }

    def _enforce_metadata_limits(
        self, request: PublishRequest, limits: dict[str, Any]
    ) -> None:
        """Raise when the request's caption or hashtags exceed limits."""
        problems: list[tuple[str, int, int]] = []
        caption_limit = limits["caption"]
        if caption_limit is not None and len(request.caption) > caption_limit:
            problems.append(("caption", caption_limit, len(request.caption)))
        hashtag_limit = limits["hashtags"]
        if hashtag_limit is not None and len(request.hashtags) > hashtag_limit:
            problems.append(("hashtags", hashtag_limit, len(request.hashtags)))
        if problems:
            detail = "; ".join(
                f"{field} exceeds limit {limit} (got {actual})"
                for field, limit, actual in problems
            )
            raise InvalidRequestError(
                "x metadata exceeds platform limits: " + detail,
                details={
                    "provider": self.name,
                    "platform": "x",
                    "problems": [
                        {"field": field, "limit": limit, "actual": actual}
                        for field, limit, actual in problems
                    ],
                },
            )

    def _resolve_visibility(
        self, request: PublishRequest, limits: dict[str, Any]
    ) -> tuple[PostVisibility, bool, str]:
        """Return the visibility to use, whether it was forced and a message."""
        visibility = request.visibility
        forced_private = False
        message = ""
        if limits["disclosure_required"]:
            audited = (
                request.attestation is not None and request.attestation.is_complete
            )
            if not audited:
                visibility = PostVisibility.PRIVATE
                forced_private = True
                message = (
                    "Project is unaudited: rights and AI disclosure not confirmed. "
                    "X upload forced to private. Complete the attestation "
                    "to publish publicly."
                )
        return visibility, forced_private, message

    def _build_description(
        self, request: PublishRequest, limits: dict[str, Any]
    ) -> str:
        """Return the caption with the disclosure text appended when required."""
        description = request.caption
        disclosure_text = limits["disclosure_text"]
        if (
            limits["disclosure_required"]
            and disclosure_text
            and disclosure_text not in description
        ):
            combined = (
                (description + "\n\n" + disclosure_text).strip()
                if description
                else disclosure_text
            )
            caption_limit = limits["caption"]
            if caption_limit is None or len(combined) <= caption_limit:
                description = combined
        return description

    def _media_bytes(self, media: PublishMedia) -> bytes:
        """Return the media bytes for a publish media, rejecting unsupported sources."""
        if media.data is not None:
            return media.data
        if media.path is not None:
            return Path(media.path).read_bytes()
        raise InvalidRequestError(
            "x upload requires inline data or a local path; url media is not supported",
            details={
                "provider": self.name,
                "platform": "x",
                "media_type": media.media_type,
            },
        )

    def _content_type(self, media: PublishMedia) -> str:
        """Return the upload content type for a publish media."""
        if media.media_type:
            return media.media_type
        if media.filename:
            suffix = Path(media.filename).suffix.lower()
            mapping = {
                ".mp4": "video/mp4",
                ".mov": "video/quicktime",
                ".mkv": "video/x-matroska",
                ".webm": "video/webm",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
            }
            if suffix in mapping:
                return mapping[suffix]
        return "application/octet-stream"

    def _media_category(self, media: PublishMedia) -> str:
        """Return the X media category for a publish media."""
        media_type = (media.media_type or "").lower()
        if media_type.startswith("video/"):
            return "tweet_video"
        if media_type.startswith("image/gif"):
            return "tweet_gif"
        return "tweet_image"

    def _parse_body(self, response: httpx.Response) -> Any:
        """Parse a response body as JSON, returning an empty mapping on failure."""
        try:
            return response.json()
        except (ValueError, TypeError):
            return {}

    def _body_message(self, body: Any) -> str:
        if isinstance(body, str):
            return body
        if isinstance(body, Mapping):
            for key in ("message", "error", "error_description", "reason"):
                value = body.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            error = body.get("error")
            if isinstance(error, Mapping):
                for key in ("message", "error", "error_description", "reason"):
                    value = error.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
        return ""

    def _classify(self, status: int, body: Any, operation: str) -> AiviaError:
        """Classify a non-successful X response into the shared taxonomy."""
        error = error_for_result("x", status, body=body)
        assert error is not None, "non-success status must produce an error"
        message = f"x {operation} failed: {redact(str(error))}"
        details = dict(error.details)
        details["operation"] = operation
        return type(error)(message, details=details)

    def _raise_on_error(self, response: httpx.Response, operation: str) -> None:
        """Raise a classified error when *response* is not successful."""
        if 200 <= response.status_code < 300:
            return
        body = self._parse_body(response)
        raise self._classify(response.status_code, body, operation)

    def _init_upload_response(self, body: Any) -> str:
        """Return the media id from an init response."""
        data: Any = None
        if isinstance(body, Mapping):
            data = body
        if not isinstance(data, Mapping):
            raise InvalidRequestError(
                "x did not return an upload init response body",
                details={
                    "provider": self.name,
                    "platform": "x",
                    "operation": "init media upload",
                },
            )
        media_id = data.get("media_id")
        if not isinstance(media_id, str) or not media_id.strip():
            raise InvalidRequestError(
                "x did not return a media id",
                details={
                    "provider": self.name,
                    "platform": "x",
                    "operation": "init media upload",
                },
            )
        return str(media_id).strip()

    def _finalize_response(self, body: Any) -> str:
        """Return the media id from a finalize response."""
        data: Any = None
        if isinstance(body, Mapping):
            data = body
        if not isinstance(data, Mapping):
            raise InvalidRequestError(
                "x did not return an upload finalize response body",
                details={
                    "provider": self.name,
                    "platform": "x",
                    "operation": "finalize media upload",
                },
            )
        media_id = data.get("media_id")
        if not isinstance(media_id, str) or not media_id.strip():
            raise InvalidRequestError(
                "x did not return a media id after finalize",
                details={
                    "provider": self.name,
                    "platform": "x",
                    "operation": "finalize media upload",
                },
            )
        return str(media_id).strip()

    def _tweet_response(self, body: Any) -> str:
        """Return the tweet id from a post creation response."""
        data: Any = None
        if isinstance(body, Mapping):
            data = body.get("data")
            if not isinstance(data, Mapping):
                data = body
        if not isinstance(data, Mapping):
            raise InvalidRequestError(
                "x did not return a post creation response body",
                details={
                    "provider": self.name,
                    "platform": "x",
                    "operation": "create post",
                },
            )
        tweet_id = data.get("id")
        if not isinstance(tweet_id, str) or not tweet_id.strip():
            raise InvalidRequestError(
                "x did not return a post id",
                details={
                    "provider": self.name,
                    "platform": "x",
                    "operation": "create post",
                },
            )
        return str(tweet_id).strip()

    async def _initiate_upload(
        self,
        access_token: str,
        size_bytes: int,
        media_type: str,
        media_category: str,
    ) -> str:
        """Initiate a chunked media upload and return the media id."""
        query = (
            f"?command=INIT&total_bytes={size_bytes}"
            f"&media_type={media_type}&media_category={media_category}"
        )
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json; charset=UTF-8",
            "Content-Length": "0",
        }
        url = f"{self._base_url}{_INIT_PATH}{query}"
        response = await self.http.post(
            url,
            content=b"",
            headers=headers,
            raise_for_status=False,
            retry=False,
        )
        self._raise_on_error(response, "init media upload")
        body = self._parse_body(response)
        return self._init_upload_response(body)

    async def _append_chunk(
        self,
        media_id: str,
        access_token: str,
        segment_index: int,
        chunk: bytes,
    ) -> None:
        """Send one media chunk to an upload session."""
        query = f"?command=APPEND&media_id={media_id}&segment_index={segment_index}"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/octet-stream",
            "Content-Length": str(len(chunk)),
        }
        url = f"{self._base_url}{_APPEND_PATH}{query}"
        response = await self.http.post(
            url,
            content=chunk,
            headers=headers,
            raise_for_status=False,
            retry=False,
        )
        self._raise_on_error(response, "append media chunk")

    async def _finalize_upload(self, access_token: str, media_id: str) -> str:
        """Finalize a chunked media upload and return the media id."""
        query = f"?command=FINALIZE&media_id={media_id}"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json; charset=UTF-8",
            "Content-Length": "0",
        }
        url = f"{self._base_url}{_FINALIZE_PATH}{query}"
        response = await self.http.post(
            url,
            content=b"",
            headers=headers,
            raise_for_status=False,
            retry=False,
        )
        self._raise_on_error(response, "finalize media upload")
        body = self._parse_body(response)
        return self._finalize_response(body)

    async def _create_tweet(
        self,
        access_token: str,
        text: str,
        media_id: str,
    ) -> str:
        """Create a post referencing the uploaded media and return the tweet id."""
        payload = json.dumps(
            {
                "text": text,
                "media": {"media_ids": [media_id]},
            }
        ).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json; charset=UTF-8",
            "Content-Length": str(len(payload)),
        }
        url = f"{self._base_url}{_TWEET_PATH}"
        response = await self.http.post(
            url,
            content=payload,
            headers=headers,
            raise_for_status=False,
            retry=False,
        )
        self._raise_on_error(response, "create post")
        body = self._parse_body(response)
        return self._tweet_response(body)

    async def _attempt_upload(
        self,
        access_token: str,
        data: bytes,
        content_type: str,
        media_category: str,
    ) -> str:
        """Attempt one full chunked upload cycle; returns the media id."""
        size_bytes = len(data)
        media_id = await self._initiate_upload(
            access_token,
            size_bytes,
            content_type,
            media_category,
        )
        segment_index = 0
        offset = 0
        while offset < size_bytes:
            chunk = data[offset : offset + self._chunk_size_bytes]
            await self._append_chunk(media_id, access_token, segment_index, chunk)
            offset += len(chunk)
            segment_index += 1
        return await self._finalize_upload(access_token, media_id)

    async def check_health(self) -> ProviderHealth:
        """Probe the X API v2 root and report whether it is reachable."""
        if self.closed:
            return ProviderHealth.closed()
        url = f"{self._base_url}/"
        try:
            response = await self.http.get(url, raise_for_status=False, retry=False)
        except Exception as exc:
            message = redact(str(exc)) or type(exc).__name__
            return ProviderHealth(
                status=HealthStatus.UNHEALTHY,
                message=message,
                details={"error": type(exc).__name__},
            )
        if 200 <= response.status_code < 300:
            return ProviderHealth(status=HealthStatus.HEALTHY, message="ok")
        body = self._parse_body(response)
        message = self._body_message(body) or f"root returned {response.status_code}"
        return ProviderHealth(
            status=HealthStatus.UNHEALTHY,
            message=message,
            details={"status": response.status_code},
        )

    async def publish(self, request: PublishRequest) -> PublishResult:
        """Publish *request* to X and return the post identifiers."""
        if not isinstance(request, PublishRequest):
            raise TypeError("request must be a PublishRequest")
        if request.dry_run:
            return PublishResult(
                platform=request.platform,
                account=request.account,
                provider=self.name,
                dry_run=True,
            )
        if not request.media:
            raise InvalidRequestError(
                "x publish requires exactly one media",
                details={"provider": self.name, "platform": "x"},
            )
        media = request.media[0]
        media_type = (media.media_type or "").lower()
        if media_type and not media_type.startswith(("image/", "video/")):
            raise InvalidRequestError(
                "x publisher only accepts image or video media",
                details={
                    "provider": self.name,
                    "platform": "x",
                    "media_type": media.media_type,
                },
            )
        limits = self._x_limits()
        self._enforce_metadata_limits(request, limits)
        visibility, forced_private, disclosure_message = self._resolve_visibility(
            request, limits
        )
        text = self._build_description(request, limits)
        access_token = self._access_token
        data = self._media_bytes(media)
        content_type = self._content_type(media)
        media_category = self._media_category(media)

        hint = self.tier_limit_hint(len(data))
        if hint is not None:
            raise InvalidRequestError(
                "x media exceeds tier limit: " + hint,
                details={
                    "provider": self.name,
                    "platform": "x",
                    "hint": hint,
                    "tier": self._tier,
                    "size_bytes": len(data),
                    "limit_bytes": _tier_max_media_bytes(self._tier),
                },
            )

        media_id: str | None = None
        last_error: RetryableError | Exception | None = None
        upload_attempts = 0
        for upload_attempts in range(1, self._max_attempts + 1):
            try:
                media_id = await self._attempt_upload(
                    access_token,
                    data,
                    content_type,
                    media_category,
                )
                break
            except RetryableError as exc:
                last_error = exc
                if upload_attempts == self._max_attempts:
                    raise
                delay = self._retry_delay_seconds * (2 ** (upload_attempts - 1))
                await asyncio.sleep(delay)
            except AiviaError:
                raise
            except Exception as exc:
                last_error = RetryableError(
                    str(exc),
                    details={
                        "provider": self.name,
                        "platform": "x",
                        "error": type(exc).__name__,
                    },
                )
                if upload_attempts == self._max_attempts:
                    raise last_error from exc
                delay = self._retry_delay_seconds * (2 ** (upload_attempts - 1))
                await asyncio.sleep(delay)
        if media_id is None:
            raise last_error or InvalidRequestError(
                "x upload failed without a usable response",
                details={"provider": self.name, "platform": "x"},
            )

        tweet_id: str | None = None
        last_error = None
        tweet_attempts = 0
        for tweet_attempts in range(1, self._max_attempts + 1):
            try:
                tweet_id = await self._create_tweet(
                    access_token,
                    text,
                    media_id,
                )
                break
            except RetryableError as exc:
                last_error = exc
                if tweet_attempts == self._max_attempts:
                    raise
                delay = self._retry_delay_seconds * (2 ** (tweet_attempts - 1))
                await asyncio.sleep(delay)
            except AiviaError:
                raise
            except Exception as exc:
                last_error = RetryableError(
                    str(exc),
                    details={
                        "provider": self.name,
                        "platform": "x",
                        "error": type(exc).__name__,
                    },
                )
                if tweet_attempts == self._max_attempts:
                    raise last_error from exc
                delay = self._retry_delay_seconds * (2 ** (tweet_attempts - 1))
                await asyncio.sleep(delay)
        if tweet_id is None:
            raise last_error or InvalidRequestError(
                "x post creation failed without a usable response",
                details={"provider": self.name, "platform": "x"},
            )

        metadata: dict[str, Any] = {
            "method": "chunked_upload_post",
            "privacy": _visibility_to_x(visibility),
            "size_bytes": len(data),
            "content_type": content_type,
            "media_category": media_category,
            "media_id": media_id,
            "upload_attempts": upload_attempts,
            "tweet_attempts": tweet_attempts,
        }
        if hint is not None:
            metadata["tier_limit_hint"] = hint
        if forced_private:
            metadata["visibility_forced_private"] = True
            metadata["disclosure_message"] = disclosure_message
        if limits["disclosure_required"] and limits["disclosure_text"]:
            metadata[limits["metadata_field"]] = True

        return PublishResult(
            platform=request.platform,
            account=request.account,
            provider=self.name,
            remote_id=tweet_id,
            url=f"https://x.com/i/status/{tweet_id}",
            published_at=utcnow(),
            metadata=metadata,
        )


register_provider("x", XPublisher, module=__name__)
register_provider("x_publish", XPublisher, module=__name__)
