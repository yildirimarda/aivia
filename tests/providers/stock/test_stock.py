"""Pexels and Pixabay stock providers: mapping, filtering and attribution."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from aivia.core.errors import (
    AuthError,
    ConfigError,
    InvalidRequestError,
    RateLimitError,
    RetryableError,
    ValidationError,
)
from aivia.core.secrets import SecretsLoader
from aivia.providers import Pexels as TopLevelPexels
from aivia.providers import Pixabay as TopLevelPixabay
from aivia.providers import provider_registry
from aivia.providers.protocols import (
    CAPABILITY_STOCK,
    ImageFormat,
    StockMediaKind,
    StockOrientation,
    StockRequest,
    VideoFormat,
)
from aivia.providers.stock import (
    PEXELS_DEFAULT_BASE_URL,
    PIXABAY_DEFAULT_BASE_URL,
    Pexels,
    PexelsProvider,
    PexelsStockProvider,
    Pixabay,
    PixabayProvider,
    PixabayStockProvider,
)

API_KEY = "test-key"
QUERY = "ocean waves"


def _pexels_photos(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "page": 1,
        "per_page": 2,
        "total_results": 2,
        "next_page": 2,
        "photos": [
            {
                "id": 101,
                "width": 4000,
                "height": 3000,
                "url": "https://www.pexels.com/photo/101/",
                "photographer": "Ada Lovelace",
                "photographer_url": "https://www.pexels.com/@ada",
                "photographer_id": 7,
                "avg_color": "#123456",
                "alt": "Waves breaking on a beach",
                "src": {
                    "original": "https://images.pexels.com/photos/101/original.jpg",
                    "large": "https://images.pexels.com/photos/101/large.jpg",
                    "medium": "https://images.pexels.com/photos/101/medium.jpg",
                },
            },
            {
                "id": 102,
                "width": 800,
                "height": 600,
                "url": "https://www.pexels.com/photo/102/",
                "photographer": "Grace Hopper",
                "photographer_url": "https://www.pexels.com/@grace",
                "src": {
                    "original": "https://images.pexels.com/photos/102/original.jpg",
                },
            },
        ],
    }
    body.update(overrides)
    return body


def _pexels_videos(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "page": 1,
        "per_page": 2,
        "total_results": 2,
        "next_page": "https://api.pexels.com/videos/search?query=ocean&page=2",
        "videos": [
            {
                "id": 201,
                "width": 1920,
                "height": 1080,
                "url": "https://www.pexels.com/video/201/",
                "image": "https://images.pexels.com/videos/201/preview.jpg",
                "duration": 12.5,
                "user": {
                    "id": 9,
                    "name": "Grace Hopper",
                    "url": "https://www.pexels.com/@grace",
                },
                "video_files": [
                    {
                        "id": 1,
                        "quality": "sd",
                        "file_type": "video/mp4",
                        "width": 640,
                        "height": 360,
                        "link": "https://videos.pexels.com/201/sd.mp4",
                        "fps": 25,
                    },
                    {
                        "id": 2,
                        "quality": "hd",
                        "file_type": "video/mp4",
                        "width": 1920,
                        "height": 1080,
                        "link": "https://videos.pexels.com/201/hd.mp4",
                        "fps": 30,
                    },
                    {
                        "id": 3,
                        "quality": "hls",
                        "file_type": "application/x-mpegURL",
                        "width": 1920,
                        "height": 1080,
                        "link": "https://videos.pexels.com/201/hls.m3u8",
                    },
                ],
            },
            {
                "id": 202,
                "width": 1280,
                "height": 720,
                "url": "https://www.pexels.com/video/202/",
                "image": "https://images.pexels.com/videos/202/preview.jpg",
                "duration": 40.0,
                "user": {
                    "id": 10,
                    "name": "Alan Turing",
                    "url": "https://www.pexels.com/@alan",
                },
                "video_files": [
                    {
                        "id": 4,
                        "quality": "hd",
                        "file_type": "video/mp4",
                        "width": 1280,
                        "height": 720,
                        "link": "https://videos.pexels.com/202/hd.mp4",
                        "fps": 30,
                    }
                ],
            },
        ],
    }
    body.update(overrides)
    return body


def _pixabay_images(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "total": 100,
        "totalHits": 10,
        "hits": [
            {
                "id": 301,
                "pageURL": "https://pixabay.com/photos/301/",
                "previewURL": "https://cdn.pixabay.com/301_preview.jpg",
                "webformatURL": "https://cdn.pixabay.com/301_640.jpg",
                "largeImageURL": "https://cdn.pixabay.com/301_1280.jpg",
                "imageWidth": 1280,
                "imageHeight": 853,
                "previewWidth": 150,
                "previewHeight": 100,
                "user": "Linus Torvalds",
                "user_id": 55,
                "tags": "ocean, waves",
                "type": "photo",
            },
            {
                "id": 302,
                "pageURL": "https://pixabay.com/photos/302/",
                "previewURL": "https://cdn.pixabay.com/302_preview.jpg",
                "webformatURL": "https://cdn.pixabay.com/302_640.jpg",
                "largeImageURL": "https://cdn.pixabay.com/302_1280.jpg",
                "imageWidth": 400,
                "imageHeight": 300,
                "user": "Margaret Hamilton",
                "user_id": 56,
                "tags": "sea",
                "type": "photo",
            },
        ],
    }
    body.update(overrides)
    return body


def _pixabay_videos(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "total": 50,
        "totalHits": 2,
        "hits": [
            {
                "id": 401,
                "pageURL": "https://pixabay.com/videos/401/",
                "duration": 15.0,
                "user": "Mia Wong",
                "user_id": 77,
                "tags": "ocean, waves",
                "type": "film",
                "videos": {
                    "large": {
                        "url": "https://cdn.pixabay.com/video/401_large.mp4",
                        "width": 1920,
                        "height": 1080,
                        "size": 1_000_000,
                    },
                    "medium": {
                        "url": "https://cdn.pixabay.com/video/401_medium.mp4",
                        "width": 1280,
                        "height": 720,
                        "size": 500_000,
                    },
                },
            },
            {
                "id": 402,
                "pageURL": "https://pixabay.com/videos/402/",
                "duration": 60.0,
                "user": "Mia Wong",
                "user_id": 77,
                "tags": "sea",
                "type": "film",
                "videos": {
                    "large": {
                        "url": "https://cdn.pixabay.com/video/402_large.mp4",
                        "width": 640,
                        "height": 360,
                        "size": 100_000,
                    }
                },
            },
        ],
    }
    body.update(overrides)
    return body


def _provider(
    provider_type: type[PexelsStockProvider] | type[PixabayStockProvider],
    handler: Callable[[httpx.Request], httpx.Response],
    **kwargs: Any,
) -> Any:
    return provider_type(
        api_key=API_KEY,
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def _capture(
    body: dict[str, Any],
    *,
    status: int = 200,
    headers: dict[str, str] | None = None,
) -> tuple[list[httpx.Request], Callable[[httpx.Request], httpx.Response]]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status, json=body, headers=headers)

    return requests, handler


def test_pexels_metadata_registration_and_aliases() -> None:
    provider = _provider(
        PexelsStockProvider,
        lambda request: httpx.Response(200, json=_pexels_photos()),
    )

    assert provider.name == "pexels"
    assert provider.capabilities == (CAPABILITY_STOCK,)
    assert provider.supports(CAPABILITY_STOCK)
    assert provider.base_url == PEXELS_DEFAULT_BASE_URL
    assert provider.endpoint == "https://api.pexels.com/v1/search"
    assert provider.photo_endpoint == "https://api.pexels.com/v1/search"
    assert provider.video_endpoint == "https://api.pexels.com/videos/search"
    assert provider.endpoint_for(StockMediaKind.IMAGE) == provider.photo_endpoint
    assert provider.endpoint_for(StockMediaKind.VIDEO) == provider.video_endpoint
    assert PexelsProvider is PexelsStockProvider
    assert Pexels is PexelsStockProvider
    assert TopLevelPexels is PexelsStockProvider
    assert provider_registry.resolve("pexels") is PexelsStockProvider


def test_pixabay_metadata_registration_and_aliases() -> None:
    provider = _provider(
        PixabayStockProvider,
        lambda request: httpx.Response(200, json=_pixabay_images()),
    )

    assert provider.name == "pixabay"
    assert provider.capabilities == (CAPABILITY_STOCK,)
    assert provider.supports(CAPABILITY_STOCK)
    assert provider.base_url == PIXABAY_DEFAULT_BASE_URL
    assert provider.endpoint == "https://pixabay.com/api/"
    assert provider.image_endpoint == "https://pixabay.com/api/"
    assert provider.video_endpoint == "https://pixabay.com/api/videos/"
    assert provider.endpoint_for(StockMediaKind.IMAGE) == provider.image_endpoint
    assert provider.endpoint_for(StockMediaKind.VIDEO) == provider.video_endpoint
    assert PixabayProvider is PixabayStockProvider
    assert Pixabay is PixabayStockProvider
    assert TopLevelPixabay is PixabayStockProvider
    assert provider_registry.resolve("pixabay") is PixabayStockProvider


def test_pexels_maps_an_image_search_and_records_attribution() -> None:
    requests, handler = _capture(_pexels_photos())
    provider = _provider(PexelsStockProvider, handler)
    request = StockRequest(
        query=f"  {QUERY}  ",
        orientation=StockOrientation.PORTRAIT,
        color="blue",
        language="EN",
        per_page=200,
        page=2,
    )

    response = asyncio.run(provider.search(request))

    sent = requests[0]
    assert sent.method == "GET"
    assert str(sent.url).startswith("https://api.pexels.com/v1/search?")
    assert sent.headers["authorization"] == API_KEY
    assert dict(sent.url.params) == {
        "query": QUERY,
        "per_page": "80",
        "page": "2",
        "orientation": "portrait",
        "color": "blue",
        "locale": "en",
    }

    assert response.query == QUERY
    assert response.provider == "pexels"
    assert response.total_results == 2
    assert response.page == 1
    assert response.next_page == 2
    first = response.assets[0]
    assert first.id == "101"
    assert first.kind is StockMediaKind.IMAGE
    assert first.url == "https://images.pexels.com/photos/101/original.jpg"
    assert first.preview_url == "https://images.pexels.com/photos/101/large.jpg"
    assert first.page_url == "https://www.pexels.com/photo/101/"
    assert first.dimensions is not None
    assert (first.dimensions.width, first.dimensions.height) == (4000, 3000)
    assert first.format is ImageFormat.JPEG
    assert first.creator == "Ada Lovelace"
    assert first.creator_url == "https://www.pexels.com/@ada"
    assert first.license == "Pexels License"
    assert first.license_url == "https://www.pexels.com/license/"
    assert first.attribution == "Photo by Ada Lovelace on Pexels"
    assert first.has_attribution
    assert first.metadata["alt"] == "Waves breaking on a beach"
    assert first.metadata["avg_color"] == "#123456"


def test_pexels_maps_a_video_search_and_selects_the_best_mp4() -> None:
    requests, handler = _capture(_pexels_videos())
    provider = _provider(PexelsStockProvider, handler)
    request = StockRequest(
        query=QUERY,
        kind=StockMediaKind.VIDEO,
        orientation=StockOrientation.LANDSCAPE,
        min_width=1000,
        min_height=700,
        min_duration_seconds=5.0,
        max_duration_seconds=20.0,
    )

    response = asyncio.run(provider.search(request))

    sent = requests[0]
    assert str(sent.url).startswith("https://api.pexels.com/videos/search?")
    assert dict(sent.url.params) == {
        "query": QUERY,
        "per_page": "20",
        "page": "1",
        "orientation": "landscape",
    }
    assert response.count == 1
    asset = response.assets[0]
    assert asset.id == "201"
    assert asset.kind is StockMediaKind.VIDEO
    assert asset.is_video
    assert asset.url == "https://videos.pexels.com/201/hd.mp4"
    assert asset.preview_url == "https://images.pexels.com/videos/201/preview.jpg"
    assert asset.page_url == "https://www.pexels.com/video/201/"
    assert asset.duration_seconds == 12.5
    assert asset.format is VideoFormat.MP4
    assert asset.creator == "Grace Hopper"
    assert asset.creator_url == "https://www.pexels.com/@grace"
    assert asset.attribution == "Video by Grace Hopper on Pexels"
    assert asset.metadata["quality"] == "hd"
    assert asset.metadata["fps"] == 30.0
    assert response.next_page == 2


def test_pexels_filters_dimensions_the_api_cannot_express() -> None:
    _requests, handler = _capture(_pexels_photos())
    provider = _provider(PexelsStockProvider, handler)

    response = asyncio.run(
        provider.search(StockRequest(query=QUERY, min_width=1000, min_height=1000))
    )

    assert [asset.id for asset in response.assets] == ["101"]


def test_pixabay_maps_an_image_search_and_records_attribution() -> None:
    requests, handler = _capture(_pixabay_images())
    provider = _provider(PixabayStockProvider, handler)
    request = StockRequest(
        query=f"  {QUERY}  ",
        orientation=StockOrientation.LANDSCAPE,
        color="blue",
        language="EN",
        min_width=1000,
        min_height=600,
        per_page=1,
        page=1,
    )

    response = asyncio.run(provider.search(request))

    sent = requests[0]
    assert sent.method == "GET"
    assert str(sent.url).startswith("https://pixabay.com/api/?")
    assert dict(sent.url.params) == {
        "key": API_KEY,
        "q": QUERY,
        "per_page": "3",
        "page": "1",
        "safesearch": "true",
        "image_type": "photo",
        "lang": "en",
        "colors": "blue",
        "min_width": "1000",
        "min_height": "600",
        "orientation": "horizontal",
    }

    assert response.query == QUERY
    assert response.provider == "pixabay"
    assert response.total_results == 10
    assert response.page == 1
    assert response.next_page == 2
    assert response.count == 1
    first = response.assets[0]
    assert first.id == "301"
    assert first.kind is StockMediaKind.IMAGE
    assert first.url == "https://cdn.pixabay.com/301_1280.jpg"
    assert first.preview_url == "https://cdn.pixabay.com/301_preview.jpg"
    assert first.page_url == "https://pixabay.com/photos/301/"
    assert first.dimensions is not None
    assert (first.dimensions.width, first.dimensions.height) == (1280, 853)
    assert first.format is ImageFormat.JPEG
    assert first.creator == "Linus Torvalds"
    assert first.creator_url == "https://pixabay.com/users/Linus%20Torvalds-55/"
    assert first.license == "Pixabay Content License"
    assert first.license_url == "https://pixabay.com/service/license-summary/"
    assert first.attribution == "Image by Linus Torvalds on Pixabay"
    assert first.has_attribution
    assert first.metadata["tags"] == "ocean, waves"
    assert first.metadata["type"] == "photo"
    assert first.metadata["user_id"] == "55"


def test_pixabay_maps_a_video_search_and_selects_the_large_rendition() -> None:
    requests, handler = _capture(_pixabay_videos())
    provider = _provider(PixabayStockProvider, handler)
    request = StockRequest(
        query=QUERY,
        kind=StockMediaKind.VIDEO,
        orientation=StockOrientation.PORTRAIT,
        min_width=1000,
        min_height=700,
        min_duration_seconds=5.0,
        max_duration_seconds=30.0,
        safe_search=False,
        per_page=200,
    )

    response = asyncio.run(provider.search(request))

    sent = requests[0]
    assert str(sent.url).startswith("https://pixabay.com/api/videos/?")
    assert dict(sent.url.params) == {
        "key": API_KEY,
        "q": QUERY,
        "per_page": "200",
        "page": "1",
        "safesearch": "false",
        "video_type": "all",
        "min_width": "1000",
        "min_height": "700",
        "orientation": "vertical",
    }
    assert response.count == 1
    asset = response.assets[0]
    assert asset.id == "401"
    assert asset.kind is StockMediaKind.VIDEO
    assert asset.url == "https://cdn.pixabay.com/video/401_large.mp4"
    assert asset.page_url == "https://pixabay.com/videos/401/"
    assert asset.duration_seconds == 15.0
    assert asset.format is VideoFormat.MP4
    assert asset.creator == "Mia Wong"
    assert asset.creator_url == "https://pixabay.com/users/Mia%20Wong-77/"
    assert asset.attribution == "Video by Mia Wong on Pixabay"
    assert asset.metadata["tags"] == "ocean, waves"
    assert asset.metadata["type"] == "film"
    assert asset.metadata["size_bytes"] == 1_000_000


def test_pixabay_filters_square_videos_the_api_cannot_express() -> None:
    square = _pixabay_videos()
    square["hits"][0]["videos"]["large"]["width"] = 1080
    square["hits"][0]["videos"]["large"]["height"] = 1080
    _requests, handler = _capture(square)
    provider = _provider(PixabayStockProvider, handler)

    response = asyncio.run(
        provider.search(
            StockRequest(
                query=QUERY,
                kind=StockMediaKind.VIDEO,
                orientation=StockOrientation.SQUARE,
            )
        )
    )

    assert [asset.id for asset in response.assets] == ["401"]


def test_api_keys_can_be_resolved_from_secrets() -> None:
    pexels_requests, pexels_handler = _capture(_pexels_photos())
    pexels = PexelsStockProvider(
        api_key_ref="PEXELS_API_KEY",
        secrets=SecretsLoader(environ={"AIVA_PEXELS_API_KEY": "pexels-secret"}),
        transport=httpx.MockTransport(pexels_handler),
    )
    asyncio.run(pexels.search(StockRequest(query=QUERY)))
    assert pexels_requests[0].headers["authorization"] == "pexels-secret"

    pixabay_requests, pixabay_handler = _capture(_pixabay_images())
    pixabay = PixabayStockProvider(
        api_key_ref="PIXABAY_API_KEY",
        secrets=SecretsLoader(environ={"AIVA_PIXABAY_API_KEY": "pixabay-secret"}),
        transport=httpx.MockTransport(pixabay_handler),
    )
    asyncio.run(pixabay.search(StockRequest(query=QUERY)))
    assert pixabay_requests[0].url.params["key"] == "pixabay-secret"


def test_a_missing_api_key_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match="API key"):
        PexelsStockProvider()
    with pytest.raises(ConfigError, match="API key"):
        PixabayStockProvider()


@pytest.mark.parametrize(
    "provider_type",
    [PexelsStockProvider, PixabayStockProvider],
)
@pytest.mark.parametrize(
    "base_url",
    ["ftp://example.com", "https://user" + ":pass" + "@example.com"],
)
def test_base_url_must_be_credential_free_http(
    provider_type: type[PexelsStockProvider] | type[PixabayStockProvider],
    base_url: str,
) -> None:
    with pytest.raises(ConfigError, match="base_url"):
        provider_type(api_key=API_KEY, base_url=base_url)


@pytest.mark.parametrize(
    ("provider_type", "body"),
    [
        (PexelsStockProvider, _pexels_photos()),
        (PixabayStockProvider, _pixabay_images()),
    ],
)
@pytest.mark.parametrize(
    ("status", "error_type"),
    [
        (401, AuthError),
        (400, InvalidRequestError),
        (429, RateLimitError),
        (500, RetryableError),
    ],
)
def test_http_errors_map_to_the_shared_taxonomy(
    provider_type: type[PexelsStockProvider] | type[PixabayStockProvider],
    body: dict[str, Any],
    status: int,
    error_type: type[Exception],
) -> None:
    _requests, handler = _capture(
        {"error": "provider rejected the request"},
        status=status,
    )
    provider = _provider(provider_type, handler)

    with pytest.raises(error_type, match="provider rejected the request"):
        asyncio.run(provider.search(StockRequest(query=QUERY)))


@pytest.mark.parametrize(
    ("provider_type", "body"),
    [
        (PexelsStockProvider, {"photos": [{"width": 10}]}),
        (PixabayStockProvider, {"hits": [{"pageURL": "not-a-url"}]}),
    ],
)
def test_malformed_results_are_rejected(
    provider_type: type[PexelsStockProvider] | type[PixabayStockProvider],
    body: dict[str, Any],
) -> None:
    _requests, handler = _capture(body)
    provider = _provider(provider_type, handler)

    with pytest.raises(ValidationError, match="malformed"):
        asyncio.run(provider.search(StockRequest(query=QUERY)))
