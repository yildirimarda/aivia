"""Pixabay stock media adapter for images and video.

Pixabay serves images and videos from separate endpoints under one API root.
The adapter maps the vendor-neutral
:class:`~aivia.providers.protocols.stock.StockRequest` onto the matching
endpoint and records the contributor, license terms and a ready credit line on
every asset.

Pixabay accepts minimum dimensions but not a duration range or a square
orientation filter. Those constraints are applied to the returned assets so a
caller never receives a result outside the requested bounds. The adapter owns
no routing, retry or budget policy; it uses the shared HTTP client and maps
failures to the shared error taxonomy.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from aivia.core.config.http import HttpClientConfig
from aivia.core.errors import ValidationError
from aivia.core.models.asset import AssetLicense
from aivia.core.models.provider import ProviderMeta
from aivia.core.secrets import SecretsLoader
from aivia.providers.base import Provider
from aivia.providers.protocols.media import ImageFormat, VideoFormat
from aivia.providers.protocols.stock import (
    CAPABILITY_STOCK,
    StockAsset,
    StockMediaKind,
    StockOrientation,
    StockRequest,
    StockResponse,
)
from aivia.providers.registry import register_provider
from aivia.providers.stock.common import (
    raise_for_response,
    resolve_api_key,
    validate_base_url,
)

__all__ = [
    "DEFAULT_BASE_URL",
    "MAX_PER_PAGE",
    "MIN_PER_PAGE",
    "PIXABAY_DEFAULT_BASE_URL",
    "Pixabay",
    "PixabayProvider",
    "PixabayStockProvider",
]

DEFAULT_BASE_URL = "https://pixabay.com/api"
"""The Pixabay API root, without a trailing slash."""

PIXABAY_DEFAULT_BASE_URL = DEFAULT_BASE_URL
"""Alias for :data:`DEFAULT_BASE_URL`, distinct from other stock defaults."""

MIN_PER_PAGE = 3
"""The smallest page size Pixabay accepts for one search request."""

MAX_PER_PAGE = 200
"""The largest page size Pixabay accepts for one search request."""

LICENSE_NAME = "Pixabay Content License"
LICENSE_URL = "https://pixabay.com/service/license-summary/"

_ORIENTATION = {
    StockOrientation.LANDSCAPE: "horizontal",
    StockOrientation.PORTRAIT: "vertical",
}

_DEFAULT_METADATA = ProviderMeta(
    name="pixabay",
    capabilities=[CAPABILITY_STOCK],
    license=AssetLicense(
        name=LICENSE_NAME,
        url=LICENSE_URL,
        commercial_use=True,
        attribution_required=False,
    ),
    commercial_use_notes=(
        "Free to use under the Pixabay Content License; a credit line is "
        "recorded but not required."
    ),
)


class _PixabayImageHit(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: int | str
    page_url: str | None = Field(default=None, alias="pageURL")
    preview_url: str | None = Field(default=None, alias="previewURL")
    webformat_url: str | None = Field(default=None, alias="webformatURL")
    large_image_url: str | None = Field(default=None, alias="largeImageURL")
    image_width: int | None = Field(default=None, alias="imageWidth")
    image_height: int | None = Field(default=None, alias="imageHeight")
    preview_width: int | None = Field(default=None, alias="previewWidth")
    preview_height: int | None = Field(default=None, alias="previewHeight")
    user: str | None = None
    user_id: int | str | None = Field(default=None, alias="user_id")
    tags: str | None = None
    type: str | None = None


class _PixabayImageResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    total: int | None = None
    total_hits: int | None = Field(default=None, alias="totalHits")
    hits: list[_PixabayImageHit] = Field(default_factory=list)


class _PixabayRendition(BaseModel):
    model_config = ConfigDict(extra="ignore")

    url: str | None = None
    width: int | None = None
    height: int | None = None
    size: int | None = None


class _PixabayVideos(BaseModel):
    model_config = ConfigDict(extra="ignore")

    large: _PixabayRendition | None = None
    medium: _PixabayRendition | None = None
    small: _PixabayRendition | None = None
    tiny: _PixabayRendition | None = None


class _PixabayVideoHit(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: int | str
    page_url: str | None = Field(default=None, alias="pageURL")
    duration: float | None = None
    videos: _PixabayVideos = Field(default_factory=_PixabayVideos)
    user: str | None = None
    user_id: int | str | None = Field(default=None, alias="user_id")
    tags: str | None = None
    type: str | None = None


class _PixabayVideoResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    total: int | None = None
    total_hits: int | None = Field(default=None, alias="totalHits")
    hits: list[_PixabayVideoHit] = Field(default_factory=list)


def _dimensions(width: int | None, height: int | None) -> tuple[int | None, int | None]:
    if width is None or height is None:
        return None, None
    return width, height


def _creator_url(user: str | None, user_id: int | str | None) -> str | None:
    if not user or user_id is None:
        return None
    return f"https://pixabay.com/users/{quote(user)}-{user_id}/"


def _attribution(kind: StockMediaKind, creator: str | None) -> str | None:
    if not creator:
        return None
    label = "Image" if kind is StockMediaKind.IMAGE else "Video"
    return f"{label} by {creator} on Pixabay"


def _best_rendition(videos: _PixabayVideos) -> _PixabayRendition | None:
    for rendition in (videos.large, videos.medium, videos.small, videos.tiny):
        if rendition is not None and rendition.url and rendition.url.strip():
            return rendition
    return None


def _orientation_matches(
    request: StockRequest,
    asset: StockAsset,
) -> bool:
    if request.orientation is None:
        return True
    if asset.width is None or asset.height is None:
        return False
    if request.orientation is StockOrientation.LANDSCAPE:
        return asset.width > asset.height
    if request.orientation is StockOrientation.PORTRAIT:
        return asset.width < asset.height
    return asset.width == asset.height


def _passes_filters(request: StockRequest, asset: StockAsset) -> bool:
    if request.min_width is not None and (
        asset.width is None or asset.width < request.min_width
    ):
        return False
    if request.min_height is not None and (
        asset.height is None or asset.height < request.min_height
    ):
        return False
    if request.orientation is StockOrientation.SQUARE and not _orientation_matches(
        request, asset
    ):
        return False
    if asset.is_video:
        duration = asset.duration_seconds
        if request.min_duration_seconds is not None and (
            duration is None or duration < request.min_duration_seconds
        ):
            return False
        if request.max_duration_seconds is not None and (
            duration is None or duration > request.max_duration_seconds
        ):
            return False
    return True


class PixabayStockProvider(Provider):
    """A stock provider backed by the Pixabay image and video search APIs.

    ``api_key`` accepts an already resolved credential while ``api_key_ref``
    names a secret resolved through ``secrets``. ``base_url`` may point at a
    local test server and ``transport`` injects an ``httpx`` mock transport so
    the offline suite never touches the network.
    """

    default_metadata = _DEFAULT_METADATA

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_key_ref: str | None = None,
        secrets: SecretsLoader | None = None,
        base_url: str = DEFAULT_BASE_URL,
        transport: httpx.AsyncBaseTransport | None = None,
        name: str | None = None,
        health_timeout_seconds: float | None = None,
        close_timeout_seconds: float | None = None,
        http_config: HttpClientConfig | None = None,
    ) -> None:
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

        self._api_key = resolve_api_key("pixabay", api_key, api_key_ref, secrets)
        self._base_url = validate_base_url(base_url, label="pixabay")
        if transport is not None:
            self._http = self.create_http_client(transport=transport)

    @property
    def base_url(self) -> str:
        """The configured API root, without a trailing slash."""
        return self._base_url

    @property
    def image_endpoint(self) -> str:
        """The image search endpoint."""
        return f"{self._base_url}/"

    @property
    def video_endpoint(self) -> str:
        """The video search endpoint."""
        return f"{self._base_url}/videos/"

    @property
    def endpoint(self) -> str:
        """The default image search endpoint."""
        return self.image_endpoint

    def endpoint_for(self, kind: StockMediaKind) -> str:
        """Return the search endpoint for *kind*."""
        if kind is StockMediaKind.VIDEO:
            return self.video_endpoint
        return self.image_endpoint

    def _build_params(self, request: StockRequest) -> dict[str, Any]:
        params: dict[str, Any] = {
            "key": self._api_key,
            "q": request.query,
            "per_page": min(max(request.per_page, MIN_PER_PAGE), MAX_PER_PAGE),
            "page": request.page,
            "safesearch": "true" if request.safe_search else "false",
        }
        if request.kind is StockMediaKind.VIDEO:
            params["video_type"] = "all"
        else:
            params["image_type"] = "photo"
        if request.language is not None:
            params["lang"] = request.language.lower()
        if request.color is not None and request.kind is StockMediaKind.IMAGE:
            params["colors"] = request.color
        if request.min_width is not None:
            params["min_width"] = request.min_width
        if request.min_height is not None:
            params["min_height"] = request.min_height
        orientation = (
            _ORIENTATION.get(request.orientation)
            if request.orientation is not None
            else None
        )
        if orientation is not None:
            params["orientation"] = orientation
        return params

    def _to_image(self, hit: _PixabayImageHit) -> StockAsset | None:
        url = (
            hit.large_image_url or hit.webformat_url or hit.preview_url or ""
        ).strip()
        if not url:
            return None
        width, height = _dimensions(hit.image_width, hit.image_height)
        metadata: dict[str, Any] = {}
        if hit.tags:
            metadata["tags"] = hit.tags
        if hit.type:
            metadata["type"] = hit.type
        if hit.user_id is not None:
            metadata["user_id"] = str(hit.user_id)
        return StockAsset(
            id=str(hit.id),
            provider=self.name,
            kind=StockMediaKind.IMAGE,
            url=url,
            preview_url=hit.preview_url or hit.webformat_url,
            page_url=hit.page_url,
            width=width,
            height=height,
            format=ImageFormat.JPEG,
            creator=hit.user,
            creator_url=_creator_url(hit.user, hit.user_id),
            license=LICENSE_NAME,
            license_url=LICENSE_URL,
            attribution=_attribution(StockMediaKind.IMAGE, hit.user),
            metadata=metadata,
        )

    def _to_video(self, hit: _PixabayVideoHit) -> StockAsset | None:
        rendition = _best_rendition(hit.videos)
        if rendition is None or not rendition.url:
            return None
        width, height = _dimensions(rendition.width, rendition.height)
        metadata: dict[str, Any] = {}
        if hit.tags:
            metadata["tags"] = hit.tags
        if hit.type:
            metadata["type"] = hit.type
        if hit.user_id is not None:
            metadata["user_id"] = str(hit.user_id)
        if rendition.size is not None:
            metadata["size_bytes"] = rendition.size
        return StockAsset(
            id=str(hit.id),
            provider=self.name,
            kind=StockMediaKind.VIDEO,
            url=rendition.url,
            page_url=hit.page_url,
            width=width,
            height=height,
            duration_seconds=hit.duration,
            format=VideoFormat.MP4,
            creator=hit.user,
            creator_url=_creator_url(hit.user, hit.user_id),
            license=LICENSE_NAME,
            license_url=LICENSE_URL,
            attribution=_attribution(StockMediaKind.VIDEO, hit.user),
            metadata=metadata,
        )

    def _to_response(
        self,
        body: Any,
        request: StockRequest,
    ) -> StockResponse:
        try:
            if request.kind is StockMediaKind.VIDEO:
                parsed_video = _PixabayVideoResponse.model_validate(body)
                assets = [
                    asset
                    for item in parsed_video.hits
                    if (asset := self._to_video(item)) is not None
                ]
                total = (
                    parsed_video.total_hits
                    if parsed_video.total_hits is not None
                    else parsed_video.total
                )
            else:
                parsed_image = _PixabayImageResponse.model_validate(body)
                assets = [
                    asset
                    for item in parsed_image.hits
                    if (asset := self._to_image(item)) is not None
                ]
                total = (
                    parsed_image.total_hits
                    if parsed_image.total_hits is not None
                    else parsed_image.total
                )
        except PydanticValidationError as exc:
            raise ValidationError(
                "pixabay returned a malformed response",
                details={"provider": self.name},
            ) from exc

        assets = [asset for asset in assets if _passes_filters(request, asset)]
        per_page = min(max(request.per_page, MIN_PER_PAGE), MAX_PER_PAGE)
        computed_next: int | None = None
        if total is not None and request.page * per_page < total:
            computed_next = request.page + 1
        return StockResponse(
            query=request.query,
            assets=assets,
            provider=self.name,
            total_results=total,
            page=request.page,
            next_page=computed_next,
        )

    async def search(self, request: StockRequest) -> StockResponse:
        """Search Pixabay for images or video matching *request*."""
        if not isinstance(request, StockRequest):
            raise TypeError("request must be a StockRequest")

        response = await self.http.get(
            self.endpoint_for(request.kind),
            params=self._build_params(request),
            timeout=request.timeout_seconds,
        )
        raise_for_response(response, provider=self.name)

        try:
            body = response.json()
        except ValueError as exc:
            raise ValidationError(
                "pixabay returned a non-JSON response",
                details={"provider": self.name},
            ) from exc
        if not isinstance(body, dict):
            raise ValidationError(
                "pixabay returned a malformed response",
                details={"provider": self.name},
            )
        return self._to_response(body, request)


PixabayProvider = PixabayStockProvider
"""Alias for :class:`PixabayStockProvider`."""

Pixabay = PixabayStockProvider
"""Alias for :class:`PixabayStockProvider`."""

register_provider("pixabay", PixabayStockProvider, module=__name__)
