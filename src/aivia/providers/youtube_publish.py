"""YouTube publisher using resumable uploads.

The adapter uploads a validated PublishRequest to YouTube using the resumable
upload protocol: it initiates an upload session,
sends the video bytes in a single ranged PUT, and returns the video id and watch
url. Transient failures (5xx, 429, timeouts and transport errors) are retried
with exponential backoff; permanent errors (400, 401, 403 auth) are raised
immediately. A YouTube quota exceeded response is surfaced as a
:class:`~aivia.core.errors.QuotaError` carrying a clear hint.

When the project is unaudited (the rights and AI disclosure attestation is not
complete) and YouTube requires a disclosure, visibility is forced to ``private``
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
    QuotaError,
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
    "DEFAULT_TOKENINFO_BASE_URL",
    "YouTubePublisher",
]

DEFAULT_BASE_URL = "https://www.googleapis.com"
"""The YouTube Data API root, without a trailing slash."""

DEFAULT_TOKENINFO_BASE_URL = "https://oauth2.googleapis.com"
"""The Google OAuth 2.0 tokeninfo endpoint root, without a trailing slash."""

DEFAULT_MAX_ATTEMPTS = 3
"""How many times a failed upload attempt is retried."""

DEFAULT_RETRY_DELAY_SECONDS = 1.0
"""Seconds to wait before the first retry; doubles on each retry."""

DEFAULT_CATEGORY_ID = "22"
"""YouTube video category id used when none is supplied (People & Blogs)."""

_DEFAULT_TITLE_LIMIT = 100
_DEFAULT_DESCRIPTION_LIMIT = 5000
_DEFAULT_HASHTAG_LIMIT = 15
_DEFAULT_DISCLOSURE_TEXT = "This content was generated with AI."
_DEFAULT_DISCLOSURE_REQUIRED = True
_DEFAULT_DISCLOSURE_METADATA_FIELD = "containsSyntheticMedia"

_YOUTUBE_ERROR_BODY_PATHS = ("error",)


def _resolve_access_token(
    access_token: str | None,
    access_token_ref: str | None,
    secrets: SecretsLoader | None,
) -> str:
    """Resolve the access token from a literal value or a secret reference."""
    if access_token is not None:
        text = str(access_token).strip()
        if not text:
            raise InvalidRequestError("youtube access token must not be blank")
        return text
    if access_token_ref is None or not str(access_token_ref).strip():
        raise InvalidRequestError(
            "youtube publisher requires an access token or access_token_ref",
            details={"field": "access_token_ref"},
        )
    loader = secrets or SecretsLoader()
    return loader.get(str(access_token_ref).strip())


def _visibility_to_youtube(visibility: PostVisibility) -> str:
    """Map a :class:`PostVisibility` to a YouTube privacy status."""
    if visibility is PostVisibility.PUBLIC:
        return "public"
    if visibility is PostVisibility.UNLISTED:
        return "unlisted"
    return "private"


class YouTubePublisher(Provider):
    """A publisher that uploads videos to YouTube via resumable uploads."""

    default_metadata = ProviderMeta(
        name="youtube",
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
        tokeninfo_base_url: str = DEFAULT_TOKENINFO_BASE_URL,
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
        self._tokeninfo_base_url = tokeninfo_base_url.rstrip("/")
        self._platforms = platforms
        self._max_attempts = max_attempts
        self._retry_delay_seconds = float(retry_delay_seconds)
        if transport is not None:
            self._http = self.create_http_client(transport=transport)

    @property
    def base_url(self) -> str:
        """The configured YouTube Data API root, without a trailing slash."""
        return self._base_url

    def _youtube_limits(self) -> dict[str, Any]:
        """Return the youtube metadata and disclosure limits in force."""
        if self._platforms is not None:
            platform = self._platforms.get("youtube")
            if platform is not None:
                caption = platform.caption_limits
                disclosure = platform.disclosure_rules
                return {
                    "title": caption.max_title_length,
                    "description": caption.max_description_length,
                    "hashtags": caption.max_hashtags,
                    "disclosure_required": disclosure.required,
                    "disclosure_text": disclosure.text,
                    "metadata_field": disclosure.metadata_field,
                    "metadata_supported": disclosure.metadata,
                }
        return {
            "title": _DEFAULT_TITLE_LIMIT,
            "description": _DEFAULT_DESCRIPTION_LIMIT,
            "hashtags": _DEFAULT_HASHTAG_LIMIT,
            "disclosure_required": _DEFAULT_DISCLOSURE_REQUIRED,
            "disclosure_text": _DEFAULT_DISCLOSURE_TEXT,
            "metadata_field": _DEFAULT_DISCLOSURE_METADATA_FIELD,
            "metadata_supported": True,
        }

    def _enforce_metadata_limits(
        self, request: PublishRequest, limits: dict[str, Any]
    ) -> None:
        """Raise when the request's title, description or hashtags exceed limits."""
        problems: list[tuple[str, int, int]] = []
        title_limit = limits["title"]
        if title_limit is not None and len(request.title) > title_limit:
            problems.append(("title", title_limit, len(request.title)))
        description_limit = limits["description"]
        if description_limit is not None and len(request.caption) > description_limit:
            problems.append(("description", description_limit, len(request.caption)))
        hashtag_limit = limits["hashtags"]
        if hashtag_limit is not None and len(request.hashtags) > hashtag_limit:
            problems.append(("hashtags", hashtag_limit, len(request.hashtags)))
        if problems:
            detail = "; ".join(
                f"{field} exceeds limit {limit} (got {actual})"
                for field, limit, actual in problems
            )
            raise InvalidRequestError(
                "youtube metadata exceeds platform limits: " + detail,
                details={
                    "provider": self.name,
                    "platform": "youtube",
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
                    "YouTube upload forced to private. Complete the attestation "
                    "to publish publicly."
                )
        return visibility, forced_private, message

    def _build_description(
        self, request: PublishRequest, limits: dict[str, Any]
    ) -> str:
        """Return the description with the disclosure text appended when required."""
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
            description_limit = limits["description"]
            if description_limit is None or len(combined) <= description_limit:
                description = combined
        return description

    def _video_bytes(self, media: PublishMedia) -> bytes:
        """Return the video bytes for a publish media, rejecting unsupported sources."""
        if media.data is not None:
            return media.data
        if media.path is not None:
            return Path(media.path).read_bytes()
        raise InvalidRequestError(
            "youtube resumable upload requires inline data or a local path; "
            "url media is not supported",
            details={
                "provider": self.name,
                "platform": "youtube",
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

    def _is_quota_error(self, status: int, body: Any) -> bool:
        """Whether a YouTube error response signals a quota exceeded condition."""
        if status not in (400, 403):
            return False
        candidates: list[str] = []
        if isinstance(body, Mapping):
            error = body.get("error")
            if isinstance(error, Mapping):
                candidates.append(str(error.get("message", "")))
                for entry in error.get("errors") or []:
                    if isinstance(entry, Mapping):
                        candidates.append(str(entry.get("reason", "")))
                        candidates.append(str(entry.get("message", "")))
        text = " | ".join(candidates).lower()
        return "quotaexceeded" in text or "quota exceeded" in text

    def _classify(self, status: int, body: Any, operation: str) -> AiviaError:
        """Classify a non-successful YouTube response into the shared taxonomy."""
        error = error_for_result("youtube", status, body=body)
        assert error is not None, "non-success status must produce an error"
        if self._is_quota_error(status, body):
            hint = (
                "YouTube quota exceeded. Reduce upload frequency, schedule uploads "
                "for off-peak hours, or request a quota increase in the "
                "Google Cloud Console."
            )
            error = QuotaError(
                hint,
                details={"platform": "youtube", "status": status, "quota_hint": True},
            )
        message = f"youtube {operation} failed: {redact(str(error))}"
        details = dict(error.details)
        details["operation"] = operation
        return type(error)(message, details=details)

    def _raise_on_error(self, response: httpx.Response, operation: str) -> None:
        """Raise a classified error when *response* is not successful."""
        if 200 <= response.status_code < 300:
            return
        body = self._parse_body(response)
        raise self._classify(response.status_code, body, operation)

    async def _initiate_session(
        self,
        access_token: str,
        title: str,
        description: str,
        hashtags: list[str],
        visibility: PostVisibility,
        size_bytes: int,
        content_type: str,
        language: str,
    ) -> str:
        """Initiate a resumable upload session and return its session url."""
        snippet: dict[str, Any] = {
            "title": title,
            "description": description,
            "tags": hashtags,
            "categoryId": DEFAULT_CATEGORY_ID,
        }
        if language:
            snippet["defaultLanguage"] = language
            snippet["defaultAudioLanguage"] = language
        status = {
            "privacyStatus": _visibility_to_youtube(visibility),
            "madeForKids": False,
            "selfDeclaredMadeForKids": False,
        }
        body = json.dumps({"snippet": snippet, "status": status}).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {access_token}",
            "X-Upload-Content-Type": content_type,
            "X-Upload-Content-Length": str(size_bytes),
            "Content-Type": "application/json; charset=UTF-8",
        }
        url = (
            f"{self._base_url}/upload/youtube/v3/videos"
            f"?uploadType=resumable&part=snippet,status"
        )
        response = await self.http.post(
            url, content=body, headers=headers, raise_for_status=False, retry=False
        )
        self._raise_on_error(response, "initiate resumable upload session")
        location = response.headers.get("Location") or response.headers.get("location")
        if not location:
            raise InvalidRequestError(
                "youtube did not return a resumable upload session url",
                details={
                    "provider": self.name,
                    "platform": "youtube",
                    "operation": "initiate resumable upload session",
                },
            )
        return str(location)

    async def _upload_bytes(
        self,
        session_url: str,
        access_token: str,
        data: bytes,
        content_type: str,
    ) -> str:
        """Upload the video bytes to a resumable session and return the video id."""
        last_byte = len(data) - 1
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Length": str(len(data)),
            "Content-Range": f"bytes 0-{last_byte}/{len(data)}",
            "Content-Type": content_type,
        }
        response = await self.http.put(
            session_url,
            content=data,
            headers=headers,
            raise_for_status=False,
            retry=False,
        )
        self._raise_on_error(response, "upload video bytes")
        body = self._parse_body(response)
        video_id = body.get("id") if isinstance(body, Mapping) else None
        if not video_id:
            raise InvalidRequestError(
                "youtube upload did not return a video id",
                details={
                    "provider": self.name,
                    "platform": "youtube",
                    "operation": "upload video bytes",
                },
            )
        return str(video_id)

    async def _attempt_upload(
        self,
        access_token: str,
        title: str,
        description: str,
        hashtags: list[str],
        visibility: PostVisibility,
        data: bytes,
        content_type: str,
        language: str,
    ) -> str:
        """Attempt one full resumable upload; returns the video id on success."""
        size_bytes = len(data)
        session_url = await self._initiate_session(
            access_token,
            title,
            description,
            hashtags,
            visibility,
            size_bytes,
            content_type,
            language,
        )
        return await self._upload_bytes(session_url, access_token, data, content_type)

    async def check_health(self) -> ProviderHealth:
        """Probe the access token with the Google tokeninfo endpoint."""
        if self.closed:
            return ProviderHealth.closed()
        url = f"{self._tokeninfo_base_url}/tokeninfo?access_token={self._access_token}"
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
        message = (
            self._body_message(body) or f"tokeninfo returned {response.status_code}"
        )
        return ProviderHealth(
            status=HealthStatus.UNHEALTHY,
            message=message,
            details={"status": response.status_code},
        )

    async def publish(self, request: PublishRequest) -> PublishResult:
        """Publish *request* to YouTube and return the video identifiers."""
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
                "youtube publish requires exactly one video media",
                details={"provider": self.name, "platform": "youtube"},
            )
        media = request.media[0]
        if media.media_type is not None and not str(media.media_type).startswith(
            "video/"
        ):
            raise InvalidRequestError(
                "youtube publisher only accepts video media",
                details={
                    "provider": self.name,
                    "platform": "youtube",
                    "media_type": media.media_type,
                },
            )
        limits = self._youtube_limits()
        self._enforce_metadata_limits(request, limits)
        visibility, forced_private, disclosure_message = self._resolve_visibility(
            request, limits
        )
        description = self._build_description(request, limits)
        access_token = self._access_token
        data = self._video_bytes(media)
        content_type = self._content_type(media)
        language = request.language or ""

        video_id: str | None = None
        last_error: RetryableError | Exception | None = None
        attempt = 0
        for attempt in range(1, self._max_attempts + 1):
            try:
                video_id = await self._attempt_upload(
                    access_token,
                    request.title,
                    description,
                    request.hashtags,
                    visibility,
                    data,
                    content_type,
                    language,
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
                        "platform": "youtube",
                        "error": type(exc).__name__,
                    },
                )
                if attempt == self._max_attempts:
                    raise last_error from exc
                await asyncio.sleep(self._retry_delay_seconds * (2 ** (attempt - 1)))
        if video_id is None:
            raise last_error or InvalidRequestError(
                "youtube upload failed without a usable response",
                details={"provider": self.name, "platform": "youtube"},
            )

        metadata: dict[str, Any] = {
            "method": "resumable_upload",
            "privacy": _visibility_to_youtube(visibility),
            "size_bytes": len(data),
            "content_type": content_type,
            "category_id": DEFAULT_CATEGORY_ID,
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
            url=f"https://www.youtube.com/watch?v={video_id}",
            published_at=utcnow(),
            metadata=metadata,
        )


register_provider("youtube", YouTubePublisher, module=__name__)
register_provider("youtube_publish", YouTubePublisher, module=__name__)
