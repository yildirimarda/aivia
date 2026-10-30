"""Pexels stock media adapter for images and video.

Pexels exposes separate photo and video search endpoints. The adapter maps the
vendor-neutral :class:`~aivia.providers.protocols.stock.StockRequest` onto the
right endpoint, keeps the photographer or videographer and the license terms on
every result, and records the exact credit line a publisher may show.

Filters Pexels does not accept, such as a minimum pixel size or a duration
range, are applied to the returned assets so a caller never receives a result
outside the requested bounds. The adapter owns no routing, retry or budget
policy; it uses the shared HTTP client and maps failures to the shared error
taxonomy.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

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
    StockRequest,
    StockResponse,
)
from aivia.providers.registry import register_provider
from aivia.providers.stock.common import (
    next_page,
    raise_for_response,
    resolve_api_key,
    validate_base_url,
)

__all__ = [
    "DEFAULT_BASE_URL",
    "MAX_PER_PAGE",
    "PEXELS_DEFAULT_BASE_URL",
    "Pexels",
    "PexelsProvider",
    "PexelsStockProvider",
]

DEFAULT_BASE_URL = "https://api.pexels.com"
"""The Pexels API root, without a trailing slash."""

PEXELS_DEFAULT_BASE_URL = DEFAULT_BASE_URL
"""Alias for :data:`DEFAULT_BASE_URL`, distinct from other stock defaults."""

MAX_PER_PAGE = 80
"""The largest page size Pexels accepts for one search request."""

LICENSE_NAME = "Pexels License"
LICENSE_URL = "https://www.pexels.com/license/"

_DEFAULT_METADATA = ProviderMeta(
    name="pexels",
    capabilities=[CAPABILITY_STOCK],
    license=AssetLicense(
        name=LICENSE_NAME,
        url=LICENSE_URL,
        commercial_use=True,
        attribution_required=False,
    ),
    commercial_use_notes=(
        "Free to use under the Pexels License; a credit line is recorded but "
        "not required."
    ),
)


class _PexelsPhotoSource(BaseModel):
    model_config = ConfigDict(extra="ignore")

    original: str | None = None
    large2x: str | None = None
    large: str | None = None
    medium: str | None = None
    small: str | None = None
    portrait: str | None = None
    landscape: str | None = None
    tiny: str | None = None


class _PexelsPhoto(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int | str
    width: int | None = None
    height: int | None = None
    url: str | None = None
    photographer: str | None = None
    photographer_url: str | None = None
    photographer_id: int | str | None = None
    avg_color: str | None = None
    alt: str | None = None
    src: _PexelsPhotoSource = Field(default_factory=_PexelsPhotoSource)


class _PexelsPhotoResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    page: int | None = None
    per_page: int | None = None
    total_results: int | None = None
    next_page: Any = None
    photos: list[_PexelsPhoto] = Field(default_factory=list)


class _PexelsVideoFile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int | str | None = None
    quality: str | None = None
    file_type: str | None = None
    width: int | None = None
    height: int | None = None
    link: str | None = None
    fps: float | None = None


class _PexelsVideoUser(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int | str | None = None
    name: str | None = None
    url: str | None = None


class _PexelsVideo(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int | str
    width: int | None = None
    height: int | None = None
    url: str | None = None
    image: str | None = None
    duration: float | None = None
    user: _PexelsVideoUser | None = None
    video_files: list[_PexelsVideoFile] = Field(default_factory=list)


class _PexelsVideoResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    page: int | None = None
    per_page: int | None = None
    total_results: int | None = None
    next_page: Any = None
    videos: list[_PexelsVideo] = Field(default_factory=list)


def _dimensions(width: int | None, height: int | None) -> tuple[int | None, int | None]:
    if width is None or height is None:
        return None, None
    return width, height


def _attribution(kind: StockMediaKind, creator: str | None) -> str | None:
    if not creator:
        return None
    label = "Photo" if kind is StockMediaKind.IMAGE else "Video"
    return f"{label} by {creator} on Pexels"


def _best_video_file(files: list[_PexelsVideoFile]) -> _PexelsVideoFile | None:
    usable = [
        item
        for item in files
        if item.link
        and item.link.strip()
        and (item.file_type is None or "mp4" in item.file_type.lower())
    ]
    if not usable:
        usable = [item for item in files if item.link and item.link.strip()]
    if not usable:
        return None

    def rank(item: _PexelsVideoFile) -> tuple[int, int, int]:
        width = item.width or 0
        height = item.height or 0
        is_mp4 = int(
            "mp4" in (item.file_type or "").lower()
            or urlsplit(item.link or "").path.lower().endswith(".mp4")
        )
        return (is_mp4, width * height, width)

    return max(usable, key=rank)


def _passes_filters(request: StockRequest, asset: StockAsset) -> bool:
    if request.min_width is not None and (
        asset.width is None or asset.width < request.min_width
    ):
        return False
    if request.min_height is not None and (
        asset.height is None or asset.height < request.min_height
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


class PexelsStockProvider(Provider):
    """A stock provider backed by the Pexels photo and video search APIs.

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

        self._api_key = resolve_api_key("pexels", api_key, api_key_ref, secrets)
        self._base_url = validate_base_url(base_url, label="pexels")
        if transport is not None:
            self._http = self.create_http_client(transport=transport)

    @property
    def base_url(self) -> str:
        """The configured API root, without a trailing slash."""
        return self._base_url

    @property
    def photo_endpoint(self) -> str:
        """The photo search endpoint."""
        return f"{self._base_url}/v1/search"

    @property
    def video_endpoint(self) -> str:
        """The video search endpoint."""
        return f"{self._base_url}/videos/search"

    @property
    def endpoint(self) -> str:
        """The default photo search endpoint."""
        return self.photo_endpoint

    def endpoint_for(self, kind: StockMediaKind) -> str:
        """Return the search endpoint for *kind*."""
        if kind is StockMediaKind.VIDEO:
            return self.video_endpoint
        return self.photo_endpoint

    def _headers(self) -> dict[str, str]:
        return {
            "authorization": self._api_key,
            "accept": "application/json",
        }

    def _build_params(self, request: StockRequest) -> dict[str, Any]:
        params: dict[str, Any] = {
            "query": request.query,
            "per_page": min(request.per_page, MAX_PER_PAGE),
            "page": request.page,
        }
        if request.orientation is not None:
            params["orientation"] = request.orientation.value
        if request.color is not None:
            params["color"] = request.color
        if request.language is not None:
            params["locale"] = request.language.lower()
        return params

    def _to_photo(self, photo: _PexelsPhoto) -> StockAsset | None:
        url = (
            photo.src.original
            or photo.src.large2x
            or photo.src.large
            or photo.src.medium
            or ""
        ).strip()
        if not url:
            return None
        width, height = _dimensions(photo.width, photo.height)
        metadata: dict[str, Any] = {}
        if photo.alt:
            metadata["alt"] = photo.alt
        if photo.avg_color:
            metadata["avg_color"] = photo.avg_color
        if photo.photographer_id is not None:
            metadata["photographer_id"] = str(photo.photographer_id)
        return StockAsset(
            id=str(photo.id),
            provider=self.name,
            kind=StockMediaKind.IMAGE,
            url=url,
            preview_url=photo.src.large or photo.src.medium,
            page_url=photo.url,
            width=width,
            height=height,
            format=ImageFormat.JPEG,
            creator=photo.photographer,
            creator_url=photo.photographer_url,
            license=LICENSE_NAME,
            license_url=LICENSE_URL,
            attribution=_attribution(StockMediaKind.IMAGE, photo.photographer),
            metadata=metadata,
        )

    def _to_video(self, video: _PexelsVideo) -> StockAsset | None:
        file = _best_video_file(video.video_files)
        if file is None or not file.link:
            return None
        width, height = _dimensions(file.width, file.height)
        if width is None or height is None:
            width, height = _dimensions(video.width, video.height)
        creator = video.user.name if video.user is not None else None
        creator_url = video.user.url if video.user is not None else None
        metadata: dict[str, Any] = {}
        if file.quality:
            metadata["quality"] = file.quality
        if file.file_type:
            metadata["file_type"] = file.file_type
        if file.fps is not None:
            metadata["fps"] = file.fps
        if file.id is not None:
            metadata["video_file_id"] = str(file.id)
        return StockAsset(
            id=str(video.id),
            provider=self.name,
            kind=StockMediaKind.VIDEO,
            url=file.link,
            preview_url=video.image,
            page_url=video.url,
            width=width,
            height=height,
            duration_seconds=video.duration,
            format=VideoFormat.MP4,
            creator=creator,
            creator_url=creator_url,
            license=LICENSE_NAME,
            license_url=LICENSE_URL,
            attribution=_attribution(StockMediaKind.VIDEO, creator),
            metadata=metadata,
        )

    def _to_response(
        self,
        body: Any,
        request: StockRequest,
    ) -> StockResponse:
        try:
            if request.kind is StockMediaKind.VIDEO:
                parsed_video = _PexelsVideoResponse.model_validate(body)
                assets = [
                    asset
                    for item in parsed_video.videos
                    if (asset := self._to_video(item)) is not None
                ]
                parsed_page = parsed_video.page
                total = parsed_video.total_results
                raw_next = parsed_video.next_page
            else:
                parsed_photo = _PexelsPhotoResponse.model_validate(body)
                assets = [
                    asset
                    for item in parsed_photo.photos
                    if (asset := self._to_photo(item)) is not None
                ]
                parsed_page = parsed_photo.page
                total = parsed_photo.total_results
                raw_next = parsed_photo.next_page
        except PydanticValidationError as exc:
            raise ValidationError(
                "pexels returned a malformed response",
                details={"provider": self.name},
            ) from exc

        assets = [asset for asset in assets if _passes_filters(request, asset)]
        return StockResponse(
            query=request.query,
            assets=assets,
            provider=self.name,
            total_results=total,
            page=parsed_page or request.page,
            next_page=next_page(raw_next, parsed_page or request.page),
        )

    async def search(self, request: StockRequest) -> StockResponse:
        """Search Pexels for images or video matching *request*."""
        if not isinstance(request, StockRequest):
            raise TypeError("request must be a StockRequest")

        response = await self.http.get(
            self.endpoint_for(request.kind),
            headers=self._headers(),
            params=self._build_params(request),
            timeout=request.timeout_seconds,
        )
        raise_for_response(response, provider=self.name)

        try:
            body = response.json()
        except ValueError as exc:
            raise ValidationError(
                "pexels returned a non-JSON response",
                details={"provider": self.name},
            ) from exc
        if not isinstance(body, dict):
            raise ValidationError(
                "pexels returned a malformed response",
                details={"provider": self.name},
            )
        return self._to_response(body, request)


PexelsProvider = PexelsStockProvider
"""Alias for :class:`PexelsStockProvider`."""

Pexels = PexelsStockProvider
"""Alias for :class:`PexelsStockProvider`."""

register_provider("pexels", PexelsStockProvider, module=__name__)
