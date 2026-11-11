"""TikTok publisher using init, upload and status.

The adapter uploads a validated
:class:`~aivia.providers.protocols.publish.PublishRequest` to TikTok in three
steps: it initiates an upload session (``init``), sends the video bytes to the
returned upload url (``upload``), then queries the video status endpoint
(``status``) until the post reaches a terminal state.

Transient failures (5xx, 429, timeouts and transport errors) are retried with
exponential backoff; permanent errors (400, 401, 403 auth) are raised
immediately.

When the project is unaudited (the rights and AI disclosure attestation is not
complete) and TikTok requires a disclosure, visibility is forced to ``private``
and a clear message is recorded on the result so the caller knows the upload
will not be public.

Credentials are supplied through ``access_token`` or ``access_token_ref``
(resolved via :class:`~aivia.core.secrets.SecretsLoader`). ``transport`` injects
an ``httpx`` mock transport for the offline suite.
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
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_RETRY_DELAY_SECONDS",
    "TikTokPublisher",
]

DEFAULT_BASE_URL = "https://open.tiktokapis.com"
"""The TikTok Open API root, without a trailing slash."""

DEFAULT_MAX_ATTEMPTS = 3
"""How many times a failed upload attempt is retried."""

DEFAULT_RETRY_DELAY_SECONDS = 1.0
"""Seconds to wait before the first retry; doubles on each retry."""

_DEFAULT_CAPTION_LIMIT = 2200
_DEFAULT_MAX_HASHTAGS = 20
_DEFAULT_DISCLOSURE_TEXT = "This content was generated with AI."
_DEFAULT_DISCLOSURE_REQUIRED = True
_DEFAULT_DISCLOSURE_METADATA_FIELD = "ai_content"

_TIKTOK_SUCCESS_STATUSES = ("PUBLISHED", "PROCESSING")

_INIT_PATH = "/v2/upload/video/init/"
_STATUS_PATH = "/v2/video/query/"


def _resolve_access_token(
    access_token: str | None,
    access_token_ref: str | None,
    secrets: SecretsLoader | None,
) -> str:
    """Resolve the access token from a literal value or a secret reference."""
    if access_token is not None:
        text = str(access_token).strip()
        if not text:
            raise InvalidRequestError("tiktok access token must not be blank")
        return text
    if access_token_ref is None or not str(access_token_ref).strip():
        raise InvalidRequestError(
            "tiktok publisher requires an access token or access_token_ref",
            details={"field": "access_token_ref"},
        )
    loader = secrets or SecretsLoader()
    return loader.get(str(access_token_ref).strip())


def _visibility_to_tiktok(visibility: PostVisibility) -> str:
    """Map a :class:`PostVisibility` to a TikTok privacy level."""
    if visibility is PostVisibility.PUBLIC:
        return "PUBLIC"
    if visibility is PostVisibility.FOLLOWERS:
        return "MUTUAL_FOLLOWERS"
    return "SELF_ONLY"


class TikTokPublisher(Provider):
    """A publisher that uploads videos to TikTok via init, upload and status."""

    default_metadata = ProviderMeta(
        name="tiktok",
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
        self._access_token = _resolve_access_token(
            access_token, access_token_ref, secrets
        )
        self._base_url = base_url.rstrip("/")
        self._platforms = platforms
        self._max_attempts = max_attempts
        self._retry_delay_seconds = float(retry_delay_seconds)
        if transport is not None:
            self._http = self.create_http_client(transport=transport)

    @property
    def base_url(self) -> str:
        """The configured TikTok Open API root, without a trailing slash."""
        return self._base_url

    def _tiktok_limits(self) -> dict[str, Any]:
        """Return the tiktok metadata and disclosure limits in force."""
        if self._platforms is not None:
            platform = self._platforms.get("tiktok")
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
                "tiktok metadata exceeds platform limits: " + detail,
                details={
                    "provider": self.name,
                    "platform": "tiktok",
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
                    "TikTok upload forced to private. Complete the attestation "
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

    def _video_bytes(self, media: PublishMedia) -> bytes:
        """Return the video bytes for a publish media, rejecting unsupported sources."""
        if media.data is not None:
            return media.data
        if media.path is not None:
            return Path(media.path).read_bytes()
        raise InvalidRequestError(
            "tiktok upload requires inline data or a local path; "
            "url media is not supported",
            details={
                "provider": self.name,
                "platform": "tiktok",
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
            }
            if suffix in mapping:
                return mapping[suffix]
        return "video/mp4"

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
        """Classify a non-successful TikTok response into the shared taxonomy."""
        error = error_for_result("tiktok", status, body=body)
        assert error is not None, "non-success status must produce an error"
        message = f"tiktok {operation} failed: {redact(str(error))}"
        details = dict(error.details)
        details["operation"] = operation
        return type(error)(message, details=details)

    def _raise_on_error(self, response: httpx.Response, operation: str) -> None:
        """Raise a classified error when *response* is not successful."""
        if 200 <= response.status_code < 300:
            return
        body = self._parse_body(response)
        raise self._classify(response.status_code, body, operation)

    def _init_upload_response(self, body: Any) -> tuple[str, str]:
        """Return the upload url and video id from an init response."""
        data: Any = None
        if isinstance(body, Mapping):
            data = body.get("data")
            if not isinstance(data, Mapping):
                data = body
        if not isinstance(data, Mapping):
            raise InvalidRequestError(
                "tiktok did not return an upload init response body",
                details={
                    "provider": self.name,
                    "platform": "tiktok",
                    "operation": "init upload session",
                },
            )
        upload_url = data.get("upload_url")
        video_id = data.get("video_id")
        if not isinstance(upload_url, str) or not upload_url.strip():
            raise InvalidRequestError(
                "tiktok did not return an upload url",
                details={
                    "provider": self.name,
                    "platform": "tiktok",
                    "operation": "init upload session",
                },
            )
        if not isinstance(video_id, str) or not video_id.strip():
            raise InvalidRequestError(
                "tiktok did not return a video id",
                details={
                    "provider": self.name,
                    "platform": "tiktok",
                    "operation": "init upload session",
                },
            )
        return str(upload_url).strip(), str(video_id).strip()

    def _status_from_body(self, body: Any) -> str:
        """Return the video status from a status response body."""
        data: Any = None
        if isinstance(body, Mapping):
            nested = body.get("data")
            data = nested if isinstance(nested, Mapping) else body
        if isinstance(data, Mapping):
            status = data.get("status")
            if isinstance(status, str) and status.strip():
                return status.strip()
        return ""

    async def _initiate_upload(
        self,
        access_token: str,
        caption: str,
        visibility: PostVisibility,
        size_bytes: int,
        content_type: str,
    ) -> tuple[str, str]:
        """Initiate an upload session and return the upload url and video id."""
        payload = json.dumps(
            {
                "source_info": {"source": "FILE_UPLOAD"},
                "privacy_level": _visibility_to_tiktok(visibility),
                "caption": caption,
                "size_bytes": size_bytes,
                "content_type": content_type,
            }
        ).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json; charset=UTF-8",
            "Content-Length": str(len(payload)),
        }
        url = f"{self._base_url}{_INIT_PATH}"
        response = await self.http.post(
            url, content=payload, headers=headers, raise_for_status=False, retry=False
        )
        self._raise_on_error(response, "init upload session")
        body = self._parse_body(response)
        return self._init_upload_response(body)

    async def _upload_bytes(
        self,
        upload_url: str,
        access_token: str,
        data: bytes,
        content_type: str,
    ) -> None:
        """Upload the video bytes to the upload url."""
        last_byte = len(data) - 1
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Length": str(len(data)),
            "Content-Range": f"bytes 0-{last_byte}/{len(data)}",
            "Content-Type": content_type,
        }
        response = await self.http.put(
            upload_url,
            content=data,
            headers=headers,
            raise_for_status=False,
            retry=False,
        )
        self._raise_on_error(response, "upload video bytes")

    async def _check_status(self, access_token: str, video_id: str) -> str:
        """Query the video status endpoint and return the reported status."""
        body = json.dumps({"video_id": video_id}).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json; charset=UTF-8",
        }
        url = f"{self._base_url}{_STATUS_PATH}"
        response = await self.http.post(
            url, content=body, headers=headers, raise_for_status=False, retry=False
        )
        self._raise_on_error(response, "check video status")
        parsed = self._parse_body(response)
        return self._status_from_body(parsed)

    async def _attempt_upload(
        self,
        access_token: str,
        caption: str,
        visibility: PostVisibility,
        data: bytes,
        content_type: str,
    ) -> str:
        """Attempt one full init/upload/status cycle; returns the video id."""
        size_bytes = len(data)
        upload_url, video_id = await self._initiate_upload(
            access_token,
            caption,
            visibility,
            size_bytes,
            content_type,
        )
        await self._upload_bytes(upload_url, access_token, data, content_type)
        status = await self._check_status(access_token, video_id)
        if status and status.upper() not in _TIKTOK_SUCCESS_STATUSES:
            raise InvalidRequestError(
                f"tiktok video {video_id} is not publishable (status {status})",
                details={
                    "provider": self.name,
                    "platform": "tiktok",
                    "video_id": video_id,
                    "status": status,
                },
            )
        return video_id

    async def check_health(self) -> ProviderHealth:
        """Probe the TikTok Open API root and report whether it is reachable."""
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
        """Publish *request* to TikTok and return the video identifiers."""
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
                "tiktok publish requires exactly one video media",
                details={"provider": self.name, "platform": "tiktok"},
            )
        media = request.media[0]
        if media.media_type is not None and not str(media.media_type).startswith(
            "video/"
        ):
            raise InvalidRequestError(
                "tiktok publisher only accepts video media",
                details={
                    "provider": self.name,
                    "platform": "tiktok",
                    "media_type": media.media_type,
                },
            )
        limits = self._tiktok_limits()
        self._enforce_metadata_limits(request, limits)
        visibility, forced_private, disclosure_message = self._resolve_visibility(
            request, limits
        )
        description = self._build_description(request, limits)
        access_token = self._access_token
        data = self._video_bytes(media)
        content_type = self._content_type(media)

        video_id: str | None = None
        last_error: RetryableError | Exception | None = None
        attempt = 0
        for attempt in range(1, self._max_attempts + 1):
            try:
                video_id = await self._attempt_upload(
                    access_token,
                    description,
                    visibility,
                    data,
                    content_type,
                )
                break
            except RetryableError as exc:
                last_error = exc
                if attempt == self._max_attempts:
                    raise
                await asyncio.sleep(self._retry_delay_seconds * (2 ** (attempt - 1)))
            except AiviaError:
                raise
            except Exception as exc:
                last_error = RetryableError(
                    str(exc),
                    details={
                        "provider": self.name,
                        "platform": "tiktok",
                        "error": type(exc).__name__,
                    },
                )
                if attempt == self._max_attempts:
                    raise last_error from exc
                await asyncio.sleep(self._retry_delay_seconds * (2 ** (attempt - 1)))
        if video_id is None:
            raise last_error or InvalidRequestError(
                "tiktok upload failed without a usable response",
                details={"provider": self.name, "platform": "tiktok"},
            )

        metadata: dict[str, Any] = {
            "method": "init_upload_status",
            "privacy": _visibility_to_tiktok(visibility),
            "size_bytes": len(data),
            "content_type": content_type,
            "attempts": attempt,
        }
        if forced_private:
            metadata["visibility_forced_private"] = True
            metadata["disclosure_message"] = disclosure_message
        if limits["disclosure_required"] and limits["disclosure_text"]:
            metadata[limits["metadata_field"]] = True

        return PublishResult(
            platform=request.platform,
            account=request.account,
            provider=self.name,
            remote_id=video_id,
            url=f"https://www.tiktok.com/video/{video_id}",
            published_at=utcnow(),
            metadata=metadata,
        )


register_provider("tiktok", TikTokPublisher, module=__name__)
register_provider("tiktok_publish", TikTokPublisher, module=__name__)
