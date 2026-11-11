"""Instagram publisher using the Graph API media container flow.

Publishing to Instagram Graph API is a two-step container flow:

1. ``POST /{ig-user-id}/media`` creates a media container and returns a
   ``creation_id``.
2. ``POST /{ig-user-id}/media_publish`` publishes that container and returns
   the media id.

A carousel repeats step 1 once per child image (each flagged
``is_carousel_item``) then creates one carousel container whose ``children``
are the child creation ids, before step 2 publishes the carousel.

Before publishing the adapter checks the account is a business or creator
account (personal accounts cannot use the publishing API) and validates each
image's aspect ratio against the limits in ``platforms.yaml``.

The access token is supplied through ``access_token`` or ``access_token_ref``
(resolved via :class:`~aivia.core.secrets.SecretsLoader`). ``transport``
injects an ``httpx`` mock transport for the offline suite.
"""

from __future__ import annotations

import asyncio
import io
from collections.abc import Mapping
from typing import Any, Callable, cast

import httpx
from PIL import Image, UnidentifiedImageError

from aivia.core.config.http import HttpClientConfig
from aivia.core.errors import AiviaError, InvalidRequestError, RetryableError
from aivia.core.models.base import utcnow
from aivia.core.models.provider import ProviderMeta, ProviderPrice
from aivia.core.platforms import PlatformCatalog
from aivia.core.publish_errors import error_for_result
from aivia.core.redact import redact
from aivia.core.secrets import SecretsLoader
from aivia.providers.base import HealthStatus, Provider, ProviderHealth
from aivia.providers.protocols.publish import (
    CAPABILITY_PUBLISH,
    PublishMedia,
    PublishRequest,
    PublishResult,
)
from aivia.providers.registry import register_provider

__all__ = [
    "DEFAULT_API_VERSION",
    "DEFAULT_BASE_URL",
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_RETRY_DELAY_SECONDS",
    "InstagramPublisher",
]

DEFAULT_BASE_URL = "https://graph.facebook.com"
"""The Instagram Graph API root, without a trailing slash or version."""

DEFAULT_API_VERSION = "v20.0"
"""The Instagram Graph API version used by the container endpoints."""

DEFAULT_MAX_ATTEMPTS = 3
"""How many times a failed container attempt is retried."""

DEFAULT_RETRY_DELAY_SECONDS = 1.0
"""Seconds to wait before the first retry; doubles on each retry."""

_DEFAULT_CAPTION_LIMIT = 2200
_DEFAULT_MAX_HASHTAGS = 30
_DEFAULT_IMAGE_ASPECT_RATIOS = ("1:1", "4:5")
_DEFAULT_VIDEO_ASPECT_RATIOS = ("9:16",)
_ACCOUNT_FIELDS = "account_type,profile_type,username"
_BUSINESS_ACCOUNT_TYPES = frozenset({"business", "creator"})
_RATIO_TOLERANCE = 1e-6


def _resolve_access_token(
    access_token: str | None,
    access_token_ref: str | None,
    secrets: SecretsLoader | None,
) -> str:
    """Resolve the access token from a literal value or a secret reference."""
    if access_token is not None:
        text = str(access_token).strip()
        if not text:
            raise InvalidRequestError("instagram access token must not be blank")
        return text
    if access_token_ref is None or not str(access_token_ref).strip():
        raise InvalidRequestError(
            "instagram publisher requires an access token or access_token_ref",
            details={"field": "access_token_ref"},
        )
    loader = secrets or SecretsLoader()
    return loader.get(str(access_token_ref).strip())


def _ratio_value(text: str) -> float:
    """Return the width/height value of a ``W:H`` aspect ratio."""
    width, _, height = str(text).partition(":")
    return float(width) / float(height)


def _aspect_ratio(width: int, height: int) -> float:
    """Return the width/height aspect ratio of an image."""
    return width / height


def _ratio_allowed(value: float, allowed: tuple[str, ...]) -> bool:
    """Whether *value* matches one of the ``W:H`` ratios in *allowed*."""
    return any(
        abs(value - _ratio_value(ratio)) <= _RATIO_TOLERANCE for ratio in allowed
    )


def _media_dimensions(media: PublishMedia) -> tuple[int, int] | None:
    """Read the pixel dimensions of an image from inline data or a local path."""
    if media.data is not None:
        source: Any = io.BytesIO(media.data)
    elif media.path is not None:
        source = media.path
    else:
        return None
    try:
        with Image.open(source) as image:
            return int(image.width), int(image.height)
    except (UnidentifiedImageError, OSError, ValueError):
        return None


class InstagramPublisher(Provider):
    """A publisher that posts images and carousels to Instagram via Graph API."""

    default_metadata = ProviderMeta(
        name="instagram",
        capabilities=[CAPABILITY_PUBLISH],
        pricing={CAPABILITY_PUBLISH: ProviderPrice(amount_usd=0.0)},
    )

    def __init__(
        self,
        *,
        access_token: str | None = None,
        access_token_ref: str | None = None,
        ig_user_id: str | None = None,
        secrets: SecretsLoader | None = None,
        base_url: str = DEFAULT_BASE_URL,
        api_version: str = DEFAULT_API_VERSION,
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
        if ig_user_id is None or not str(ig_user_id).strip():
            raise InvalidRequestError(
                "instagram publisher requires an ig_user_id",
                details={"field": "ig_user_id"},
            )
        if max_attempts < 1:
            raise InvalidRequestError("max_attempts must be at least 1")
        self._access_token = _resolve_access_token(
            access_token, access_token_ref, secrets
        )
        self._ig_user_id = str(ig_user_id).strip()
        self._base_url = base_url.rstrip("/")
        self._api_version = api_version.strip() or DEFAULT_API_VERSION
        self._platforms = platforms
        self._max_attempts = max_attempts
        self._retry_delay_seconds = float(retry_delay_seconds)
        if transport is not None:
            self._http = self.create_http_client(transport=transport)

    @property
    def base_url(self) -> str:
        """The configured Instagram Graph API root, without a trailing slash."""
        return self._base_url

    @property
    def ig_user_id(self) -> str:
        """The Instagram business or creator account id posts are published to."""
        return self._ig_user_id

    def _graph_url(self, path: str) -> str:
        """Return a Graph API URL for *path* under the configured version."""
        return f"{self._base_url}/{self._api_version}/{path.lstrip('/')}"

    def _account_url(self) -> str:
        """Return the account inspection URL used for the business account check."""
        return (
            f"{self._graph_url(self._ig_user_id)}"
            f"?fields={_ACCOUNT_FIELDS}&access_token={self._access_token}"
        )

    def _parse_body(self, response: httpx.Response) -> Any:
        """Parse a response body as JSON, returning an empty mapping on failure."""
        try:
            return response.json()
        except (ValueError, TypeError):
            return {}

    def _raise_on_error(self, response: httpx.Response, operation: str) -> None:
        """Raise a classified error when *response* is not successful."""
        if 200 <= response.status_code < 300:
            return
        body = self._parse_body(response)
        raise self._classify(response.status_code, body, operation)

    def _classify(self, status: int, body: Any, operation: str) -> AiviaError:
        """Classify a non-successful Instagram response into the shared taxonomy."""
        error = error_for_result("instagram", status, body=body)
        assert error is not None, "non-success status must produce an error"
        message = f"instagram {operation} failed: {redact(str(error))}"
        details = dict(error.details)
        details["operation"] = operation
        return type(error)(message, details=details)

    def _image_aspect_ratios(self) -> tuple[str, ...]:
        """Return the aspect ratios allowed for instagram images."""
        if self._platforms is not None:
            platform = self._platforms.get("instagram")
            if platform is not None:
                ratios = cast("tuple[str, ...]", platform.limit("image").aspect_ratios)
                if ratios:
                    return ratios
        return _DEFAULT_IMAGE_ASPECT_RATIOS

    def _video_aspect_ratios(self) -> tuple[str, ...]:
        """Return the aspect ratios allowed for instagram videos."""
        if self._platforms is not None:
            platform = self._platforms.get("instagram")
            if platform is not None:
                ratios = cast("tuple[str, ...]", platform.limit("video").aspect_ratios)
                if ratios:
                    return ratios
        return _DEFAULT_VIDEO_ASPECT_RATIOS

    def _caption_limit(self) -> int | None:
        """Return the caption character limit in force."""
        if self._platforms is not None:
            platform = self._platforms.get("instagram")
            if platform is not None:
                limit = cast("int | None", platform.caption_limits.max_length)
                if limit is not None:
                    return limit
        return _DEFAULT_CAPTION_LIMIT

    def _max_hashtags(self) -> int | None:
        """Return the maximum number of hashtags allowed."""
        if self._platforms is not None:
            platform = self._platforms.get("instagram")
            if platform is not None:
                limit = cast("int | None", platform.caption_limits.max_hashtags)
                if limit is not None:
                    return limit
        return _DEFAULT_MAX_HASHTAGS

    def _enforce_limits(self, request: PublishRequest) -> None:
        """Raise when the request's caption or hashtags exceed platform limits."""
        caption_limit = self._caption_limit()
        if caption_limit is not None and len(request.caption) > caption_limit:
            raise InvalidRequestError(
                f"instagram caption exceeds limit {caption_limit} "
                f"(got {len(request.caption)})",
                details={
                    "provider": self.name,
                    "platform": "instagram",
                    "field": "caption",
                    "limit": caption_limit,
                    "actual": len(request.caption),
                },
            )
        max_hashtags = self._max_hashtags()
        if max_hashtags is not None and len(request.hashtags) > max_hashtags:
            raise InvalidRequestError(
                f"instagram hashtags exceed limit {max_hashtags} "
                f"(got {len(request.hashtags)})",
                details={
                    "provider": self.name,
                    "platform": "instagram",
                    "field": "hashtags",
                    "limit": max_hashtags,
                    "actual": len(request.hashtags),
                },
            )

    def _validate_media(
        self,
        media: PublishMedia,
        image_ratios: tuple[str, ...],
        video_ratios: tuple[str, ...],
    ) -> None:
        """Validate one media item's type and aspect ratio."""
        media_type = (media.media_type or "image/jpeg").lower()
        if not media_type.startswith(("image/", "video/")):
            raise InvalidRequestError(
                "instagram publisher only accepts image or video media",
                details={
                    "provider": self.name,
                    "platform": "instagram",
                    "media_type": media.media_type,
                },
            )
        dimensions = _media_dimensions(media)
        if dimensions is None:
            return
        width, height = dimensions
        ratio = _aspect_ratio(width, height)
        allowed = video_ratios if media_type.startswith("video/") else image_ratios
        if not _ratio_allowed(ratio, allowed):
            raise InvalidRequestError(
                f"instagram {media_type} aspect {width}:{height} is not supported; "
                f"expected {', '.join(allowed)}",
                details={
                    "provider": self.name,
                    "platform": "instagram",
                    "media_type": media_type,
                    "width": width,
                    "height": height,
                },
            )

    def _media_url(self, media: PublishMedia) -> str:
        """Return the public url used as the container's image/video url."""
        if media.url:
            return str(media.url).strip()
        raise InvalidRequestError(
            "instagram media requires a public url",
            details={
                "provider": self.name,
                "platform": "instagram",
                "media_type": media.media_type,
            },
        )

    async def account_type(self) -> str | None:
        """Return the Instagram account type, or ``None`` when undetermined."""
        response = await self.http.get(
            self._account_url(), raise_for_status=False, retry=False
        )
        if 200 <= response.status_code < 300:
            body = self._parse_body(response)
            if isinstance(body, Mapping):
                value = body.get("account_type")
                return str(value).strip() if value is not None else ""
            return ""
        raise self._classify(
            response.status_code, self._parse_body(response), "account check"
        )

    async def check_business_account(self) -> None:
        """Raise unless the account is a business or creator account."""
        account_type = await self.account_type()
        if account_type and account_type.lower() not in _BUSINESS_ACCOUNT_TYPES:
            raise InvalidRequestError(
                "instagram publishing requires a business or creator account",
                details={
                    "provider": self.name,
                    "platform": "instagram",
                    "account_type": account_type,
                },
            )

    async def check_health(self) -> ProviderHealth:
        """Probe the account endpoint and report whether the token works."""
        if self.closed:
            return ProviderHealth.closed()
        try:
            account_type = await self.account_type()
        except Exception as exc:
            message = redact(str(exc)) or type(exc).__name__
            return ProviderHealth(
                status=HealthStatus.UNHEALTHY,
                message=message,
                details={"error": type(exc).__name__},
            )
        if account_type and account_type.lower() in _BUSINESS_ACCOUNT_TYPES:
            return ProviderHealth(status=HealthStatus.HEALTHY, message="ok")
        return ProviderHealth(
            status=HealthStatus.UNHEALTHY,
            message=(
                "instagram account is not a business or creator account "
                f"({account_type or 'unknown'})"
            ),
            details={"account_type": account_type},
        )

    async def _create_container(self, access_token: str, fields: dict[str, Any]) -> str:
        """Create a media container and return its creation id."""
        fields = dict(fields)
        fields["access_token"] = access_token
        response = await self.http.post(
            self._graph_url(f"{self._ig_user_id}/media"),
            data=fields,
            raise_for_status=False,
            retry=False,
        )
        self._raise_on_error(response, "create media container")
        body = self._parse_body(response)
        creation_id = body.get("id") if isinstance(body, Mapping) else None
        if not creation_id:
            raise InvalidRequestError(
                "instagram did not return a media container id",
                details={
                    "provider": self.name,
                    "platform": "instagram",
                    "operation": "create media container",
                },
            )
        return str(creation_id)

    async def _publish_container(self, access_token: str, creation_id: str) -> str:
        """Publish a media container and return the media id."""
        response = await self.http.post(
            self._graph_url(f"{self._ig_user_id}/media_publish"),
            data={"creation_id": creation_id, "access_token": access_token},
            raise_for_status=False,
            retry=False,
        )
        self._raise_on_error(response, "publish media container")
        body = self._parse_body(response)
        media_id = body.get("id") if isinstance(body, Mapping) else None
        if not media_id:
            raise InvalidRequestError(
                "instagram did not return a published media id",
                details={
                    "provider": self.name,
                    "platform": "instagram",
                    "operation": "publish media container",
                },
            )
        return str(media_id)

    async def _attempt(
        self, operation: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Run *operation*, retrying transient failures with exponential backoff."""
        last_error: RetryableError | Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                return await operation(*args, **kwargs)
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
                        "platform": "instagram",
                        "error": type(exc).__name__,
                    },
                )
                if attempt == self._max_attempts:
                    raise last_error from exc
                await asyncio.sleep(self._retry_delay_seconds * (2 ** (attempt - 1)))
        if last_error is not None:
            raise last_error
        raise InvalidRequestError(
            "instagram container flow failed without a usable response",
            details={"provider": self.name, "platform": "instagram"},
        )

    async def publish(self, request: PublishRequest) -> PublishResult:
        """Publish *request* to Instagram and return the media identifiers."""
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
                "instagram publish requires at least one image or video media",
                details={"provider": self.name, "platform": "instagram"},
            )
        await self.check_business_account()
        image_ratios = self._image_aspect_ratios()
        video_ratios = self._video_aspect_ratios()
        self._enforce_limits(request)
        for media in request.media:
            self._validate_media(media, image_ratios, video_ratios)

        is_carousel = request.format == "carousel" or len(request.media) > 1
        access_token = self._access_token
        caption = request.caption.strip()
        hashtags = " ".join(f"#{tag}" for tag in request.hashtags)
        full_caption = (caption + ("\n" + hashtags if hashtags else "")).strip()

        creation_id: str | None = None
        media_type = "image"
        if is_carousel:
            child_ids: list[str] = []
            for media in request.media:
                child_ids.append(
                    await self._attempt(
                        self._create_container,
                        access_token,
                        {
                            "image_url": self._media_url(media),
                            "is_carousel_item": "true",
                        },
                    )
                )
            creation_id = await self._attempt(
                self._create_container,
                access_token,
                {
                    "media_type": "CAROUSEL",
                    "children": ",".join(child_ids),
                    "caption": full_caption,
                },
            )
        else:
            media = request.media[0]
            media_type = (media.media_type or "image/jpeg").lower()
            fields: dict[str, Any] = {"caption": full_caption}
            if media_type.startswith("video/"):
                fields["media_type"] = "REELS"
                fields["video_url"] = self._media_url(media)
            else:
                fields["image_url"] = self._media_url(media)
            creation_id = await self._attempt(
                self._create_container, access_token, fields
            )

        media_id = await self._attempt(
            self._publish_container, access_token, creation_id
        )

        metadata: dict[str, Any] = {
            "method": "container",
            "container_type": (
                "carousel"
                if is_carousel
                else ("video" if media_type.startswith("video/") else "image")
            ),
            "size_bytes": sum((m.size_bytes or 0) for m in request.media),
        }
        return PublishResult(
            platform=request.platform,
            account=request.account,
            provider=self.name,
            remote_id=media_id,
            url=f"https://www.instagram.com/p/{media_id}/",
            published_at=utcnow(),
            metadata=metadata,
        )


register_provider("instagram", InstagramPublisher, module=__name__)
register_provider("instagram_publish", InstagramPublisher, module=__name__)
