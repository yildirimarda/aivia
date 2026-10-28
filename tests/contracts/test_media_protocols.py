"""Image, image edit, video, music, sfx and stock protocols and fakes."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from aivia.core.errors import RateLimitError
from aivia.core.models import ProviderMeta
from aivia.providers import Provider
from aivia.providers.protocols import (
    CAPABILITY_IMAGE_EDIT,
    CAPABILITY_IMAGE_GEN,
    CAPABILITY_MUSIC_GEN,
    CAPABILITY_SFX,
    CAPABILITY_STOCK,
    CAPABILITY_VIDEO_GEN,
    AudioFormat,
    CapabilityProvider,
    ImageEditOperation,
    ImageEditProvider,
    ImageEditRequest,
    ImageEditResponse,
    ImageFormat,
    ImageProvider,
    ImageRequest,
    ImageResponse,
    ImageResult,
    MediaDimensions,
    MediaInput,
    MusicProvider,
    MusicRequest,
    MusicResponse,
    MusicTrack,
    SFXClip,
    SFXProvider,
    SFXRequest,
    SFXResponse,
    StockAsset,
    StockMediaKind,
    StockOrientation,
    StockProvider,
    StockRequest,
    StockResponse,
    VideoFormat,
    VideoProvider,
    VideoRequest,
    VideoResponse,
    VideoResult,
)

from .provider_contract import ProviderContract

_IMAGE_URL = "https://example.com/media/image.png"
_VIDEO_URL = "https://example.com/media/clip.mp4"
_AUDIO_URL = "https://example.com/media/track.mp3"


class FakeImageProvider(Provider):
    """A deterministic image fake that records requests and echoes options."""

    default_metadata = ProviderMeta(
        name="fake-image",
        capabilities=[CAPABILITY_IMAGE_GEN],
    )

    def __init__(self, error: Exception | None = None) -> None:
        super().__init__()
        self._error = error
        self.requests: list[ImageRequest] = []

    async def generate(self, request: ImageRequest) -> ImageResponse:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        images = [
            ImageResult(
                data=b"FAKEIMG:" + request.prompt.encode("utf-8"),
                format=request.format,
                width=request.width or 512,
                height=request.height or 512,
                seed=request.seed,
            )
            for _ in range(request.count)
        ]
        return ImageResponse(
            images=images,
            model=request.model or "fake-image-1",
            provider=self.name,
        )


class FakeImageEditProvider(Provider):
    """A deterministic image edit fake that records the operation."""

    default_metadata = ProviderMeta(
        name="fake-image-edit",
        capabilities=[CAPABILITY_IMAGE_EDIT],
    )

    def __init__(self, error: Exception | None = None) -> None:
        super().__init__()
        self._error = error
        self.requests: list[ImageEditRequest] = []

    async def edit(self, request: ImageEditRequest) -> ImageEditResponse:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        label = request.prompt or request.operation.value
        image = ImageResult(
            data=b"FAKEEDIT:" + label.encode("utf-8"),
            format=request.format or ImageFormat.PNG,
            width=request.width or 512,
            height=request.height or 512,
            seed=request.seed,
        )
        return ImageEditResponse(
            images=[image],
            operation=request.operation,
            model=request.model or "fake-edit-1",
            provider=self.name,
        )


class FakeVideoProvider(Provider):
    """A deterministic video fake that records requests and audio intent."""

    default_metadata = ProviderMeta(
        name="fake-video",
        capabilities=[CAPABILITY_VIDEO_GEN],
    )

    def __init__(self, error: Exception | None = None) -> None:
        super().__init__()
        self._error = error
        self.requests: list[VideoRequest] = []

    async def generate(self, request: VideoRequest) -> VideoResponse:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        videos = [
            VideoResult(
                data=b"FAKEVIDEO:" + request.prompt.encode("utf-8"),
                format=request.format,
                width=request.width or 1080,
                height=request.height or 1920,
                duration_seconds=request.duration_seconds or 5.0,
                fps=request.fps or 30,
                has_audio=request.generate_audio,
                seed=request.seed,
            )
            for _ in range(request.count)
        ]
        return VideoResponse(
            videos=videos,
            model=request.model or "fake-video-1",
            provider=self.name,
        )


class FakeMusicProvider(Provider):
    """A deterministic music fake that records requests and lyrics."""

    default_metadata = ProviderMeta(
        name="fake-music",
        capabilities=[CAPABILITY_MUSIC_GEN],
    )

    def __init__(self, error: Exception | None = None) -> None:
        super().__init__()
        self._error = error
        self.requests: list[MusicRequest] = []

    async def generate(self, request: MusicRequest) -> MusicResponse:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        tracks = [
            MusicTrack(
                data=b"FAKEMUSIC:" + request.prompt.encode("utf-8"),
                format=request.format,
                duration_seconds=request.duration_seconds or 30.0,
                sample_rate=request.sample_rate or 44_100,
                seed=request.seed,
                title=request.prompt,
                lyrics=None if request.instrumental else request.lyrics,
            )
            for _ in range(request.count)
        ]
        return MusicResponse(
            tracks=tracks,
            model=request.model or "fake-music-1",
            provider=self.name,
        )


class FakeSFXProvider(Provider):
    """A deterministic sound effect fake that records requests and looping."""

    default_metadata = ProviderMeta(
        name="fake-sfx",
        capabilities=[CAPABILITY_SFX],
    )

    def __init__(self, error: Exception | None = None) -> None:
        super().__init__()
        self._error = error
        self.requests: list[SFXRequest] = []

    async def generate(self, request: SFXRequest) -> SFXResponse:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        clips = [
            SFXClip(
                data=b"FAKESFX:" + request.prompt.encode("utf-8"),
                format=request.format,
                duration_seconds=request.duration_seconds or 1.0,
                sample_rate=request.sample_rate or 44_100,
                loop=request.loop,
                seed=request.seed,
            )
            for _ in range(request.count)
        ]
        return SFXResponse(
            clips=clips,
            model=request.model or "fake-sfx-1",
            provider=self.name,
        )


class FakeStockProvider(Provider):
    """A deterministic stock fake that returns synthetic licensed assets."""

    default_metadata = ProviderMeta(
        name="fake-stock",
        capabilities=[CAPABILITY_STOCK],
    )

    def __init__(
        self,
        assets: Sequence[StockAsset] | None = None,
        error: Exception | None = None,
    ) -> None:
        super().__init__()
        self._assets = list(assets) if assets is not None else None
        self._error = error
        self.requests: list[StockRequest] = []

    async def search(self, request: StockRequest) -> StockResponse:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        if self._assets is not None:
            assets = self._assets
        else:
            assets = [
                StockAsset(
                    id=f"{request.kind.value}-{index}",
                    provider=self.name,
                    kind=request.kind,
                    url=f"https://example.com/stock/{index}",
                    width=1920,
                    height=1080,
                    format=(
                        VideoFormat.MP4
                        if request.kind is StockMediaKind.VIDEO
                        else ImageFormat.JPEG
                    ),
                    creator="Synthetic Creator",
                    license="CC0",
                    attribution="Synthetic Creator / example.com",
                )
                for index in range(1, 4)
            ]
        return StockResponse(
            query=request.query,
            assets=assets[: request.per_page],
            provider=self.name,
            total_results=len(assets),
            page=request.page,
        )


class TestFakeImageProviderContract(ProviderContract):
    @pytest.fixture
    def provider(self) -> Provider:
        return FakeImageProvider()


class TestFakeImageEditProviderContract(ProviderContract):
    @pytest.fixture
    def provider(self) -> Provider:
        return FakeImageEditProvider()


class TestFakeVideoProviderContract(ProviderContract):
    @pytest.fixture
    def provider(self) -> Provider:
        return FakeVideoProvider()


class TestFakeMusicProviderContract(ProviderContract):
    @pytest.fixture
    def provider(self) -> Provider:
        return FakeMusicProvider()


class TestFakeSFXProviderContract(ProviderContract):
    @pytest.fixture
    def provider(self) -> Provider:
        return FakeSFXProvider()


class TestFakeStockProviderContract(ProviderContract):
    @pytest.fixture
    def provider(self) -> Provider:
        return FakeStockProvider()


def test_fakes_satisfy_the_capability_protocols() -> None:
    image = FakeImageProvider()
    image_edit = FakeImageEditProvider()
    video = FakeVideoProvider()
    music = FakeMusicProvider()
    sfx = FakeSFXProvider()
    stock = FakeStockProvider()

    assert isinstance(image, ImageProvider)
    assert isinstance(image_edit, ImageEditProvider)
    assert isinstance(video, VideoProvider)
    assert isinstance(music, MusicProvider)
    assert isinstance(sfx, SFXProvider)
    assert isinstance(stock, StockProvider)

    for provider in (image, image_edit, video, music, sfx, stock):
        assert isinstance(provider, CapabilityProvider)

    assert not isinstance(image, ImageEditProvider)
    assert not isinstance(image_edit, ImageProvider)
    assert not isinstance(video, ImageEditProvider)
    assert not isinstance(music, ImageEditProvider)
    assert not isinstance(sfx, ImageEditProvider)
    assert not isinstance(stock, ImageProvider)
    assert not isinstance(image, StockProvider)
    assert not isinstance(Provider(), ImageProvider)


def test_capability_names_are_stable() -> None:
    assert CAPABILITY_IMAGE_GEN == "image_gen"
    assert CAPABILITY_IMAGE_EDIT == "image_edit"
    assert CAPABILITY_VIDEO_GEN == "video_gen"
    assert CAPABILITY_MUSIC_GEN == "music_gen"
    assert CAPABILITY_SFX == "sfx"
    assert CAPABILITY_STOCK == "stock"


def test_media_input_requires_exactly_one_source_and_a_media_type() -> None:
    inline = MediaInput(
        data=b"fake-image",
        media_type=" Image/PNG ",
        filename="  source.png  ",
    )

    assert inline.is_inline is True
    assert inline.media_type == "image/png"
    assert inline.filename == "source.png"
    assert inline.size_bytes == 10

    remote = MediaInput(url=f"  {_IMAGE_URL}  ")
    assert remote.is_inline is False
    assert remote.size_bytes is None
    assert remote.url == _IMAGE_URL

    with pytest.raises(ValidationError, match="exactly one"):
        MediaInput()
    with pytest.raises(ValidationError, match="exactly one"):
        MediaInput(data=b"fake", url=_IMAGE_URL)
    with pytest.raises(ValidationError, match="non-empty"):
        MediaInput(data=b"")
    with pytest.raises(ValidationError, match="media_type"):
        MediaInput(data=b"fake", media_type="text/plain")
    with pytest.raises(ValidationError):
        MediaInput(url="ftp://example.com/image.png")


def test_media_dimensions_report_orientation() -> None:
    landscape = MediaDimensions(width=1920, height=1080)
    portrait = MediaDimensions(width=1080, height=1920)
    square = MediaDimensions(width=512, height=512)

    assert landscape.is_landscape is True
    assert landscape.is_portrait is False
    assert landscape.aspect_ratio == pytest.approx(16 / 9)
    assert landscape.pixels == 1920 * 1080
    assert portrait.is_portrait is True
    assert square.is_square is True
    assert square.aspect_ratio == 1.0

    with pytest.raises(ValidationError):
        MediaDimensions(width=0, height=10)


def test_image_request_normalizes_and_validates_options() -> None:
    reference = MediaInput(url=_IMAGE_URL, media_type="image/png")
    request = ImageRequest(
        prompt="  a lighthouse at dusk  ",
        negative_prompt="  text, watermark  ",
        model="  some-image-model  ",
        width=1024,
        height=1536,
        aspect_ratio=" 16 : 9 ",
        format=ImageFormat.WEBP,
        quality="  high  ",
        style="  cinematic  ",
        seed=7,
        count=2,
        reference_images=[reference],
    )

    assert request.prompt == "a lighthouse at dusk"
    assert request.negative_prompt == "text, watermark"
    assert request.model == "some-image-model"
    assert request.aspect_ratio == "16:9"
    assert request.format is ImageFormat.WEBP
    assert request.quality == "high"
    assert request.style == "cinematic"
    assert request.seed == 7
    assert request.count == 2
    assert request.reference_images == [reference]

    assert ImageRequest(prompt="x", aspect_ratio="  ").aspect_ratio is None

    with pytest.raises(ValidationError):
        ImageRequest(prompt="   ")
    with pytest.raises(ValidationError, match="together"):
        ImageRequest(prompt="x", width=1024)
    with pytest.raises(ValidationError):
        ImageRequest(prompt="x", count=0)
    with pytest.raises(ValidationError, match="aspect_ratio"):
        ImageRequest(prompt="x", aspect_ratio="wide")
    with pytest.raises(ValidationError, match="positive"):
        ImageRequest(prompt="x", aspect_ratio="0:9")


def test_image_result_and_response_helpers() -> None:
    result = ImageResult(
        data=b"fake-image",
        format=ImageFormat.PNG,
        width=1024,
        height=768,
        seed=3,
        revised_prompt="  a clearer prompt  ",
    )
    response = ImageResponse(
        images=[result],
        model="some-image-model",
        provider="fake-image",
    )

    assert result.is_inline is True
    assert result.size_bytes == 10
    assert result.revised_prompt == "a clearer prompt"
    assert result.dimensions == MediaDimensions(width=1024, height=768)
    assert response.first is result
    assert response.count == 1

    remote = ImageResult(url=_IMAGE_URL)
    assert remote.is_inline is False
    assert remote.size_bytes is None
    assert remote.dimensions is None

    with pytest.raises(ValidationError, match="exactly one"):
        ImageResult()
    with pytest.raises(ValidationError, match="exactly one"):
        ImageResult(data=b"fake", url=_IMAGE_URL)
    with pytest.raises(ValidationError, match="non-empty"):
        ImageResult(data=b"")
    with pytest.raises(ValidationError, match="together"):
        ImageResult(data=b"fake", width=10)
    with pytest.raises(ValidationError):
        ImageResponse(images=[])


def test_image_edit_request_requires_operation_inputs() -> None:
    source = MediaInput(data=b"fake-image", media_type="image/png")
    mask = MediaInput(data=b"fake-mask", media_type="image/png")

    edit = ImageEditRequest(image=source, prompt="  add a hat  ")
    assert edit.operation is ImageEditOperation.EDIT
    assert edit.prompt == "add a hat"

    inpaint = ImageEditRequest(
        image=source,
        operation=ImageEditOperation.INPAINT,
        prompt="replace the sky",
        mask=mask,
    )
    assert inpaint.mask == mask

    upscale = ImageEditRequest(
        image=source,
        operation=ImageEditOperation.UPSCALE,
        scale=2.0,
        width=2048,
        height=2048,
    )
    assert upscale.scale == 2.0

    background = ImageEditRequest(
        image=source,
        operation=ImageEditOperation.REMOVE_BACKGROUND,
    )
    assert background.prompt is None

    with pytest.raises(ValidationError, match="requires a prompt"):
        ImageEditRequest(image=source)
    with pytest.raises(ValidationError, match="requires a mask"):
        ImageEditRequest(
            image=source,
            operation=ImageEditOperation.INPAINT,
            prompt="replace the sky",
        )
    with pytest.raises(ValidationError, match="requires a scale"):
        ImageEditRequest(
            image=source,
            operation=ImageEditOperation.UPSCALE,
        )
    with pytest.raises(ValidationError, match="together"):
        ImageEditRequest(image=source, prompt="x", width=10)
    with pytest.raises(ValidationError):
        ImageEditRequest(image=source, prompt="x", scale=1.0)


def test_image_edit_response_helpers() -> None:
    result = ImageResult(data=b"fake-edited")
    response = ImageEditResponse(
        images=[result],
        operation=ImageEditOperation.UPSCALE,
        model="some-edit-model",
        provider="fake-image-edit",
    )

    assert response.first is result
    assert response.count == 1
    assert response.operation is ImageEditOperation.UPSCALE

    with pytest.raises(ValidationError):
        ImageEditResponse(images=[])


def test_video_request_validates_size_audio_and_aspect() -> None:
    first = MediaInput(url=_IMAGE_URL, media_type="image/png")
    last = MediaInput(url=_IMAGE_URL, media_type="image/png")
    request = VideoRequest(
        prompt="  a wave rolling in  ",
        negative_prompt="  text  ",
        model="  some-video-model  ",
        first_frame=first,
        last_frame=last,
        duration_seconds=5.5,
        width=1080,
        height=1920,
        aspect_ratio=" 9:16 ",
        fps=30,
        format=VideoFormat.WEBM,
        seed=11,
        count=2,
        generate_audio=True,
        audio_prompt="  soft ocean ambience  ",
    )

    assert request.prompt == "a wave rolling in"
    assert request.negative_prompt == "text"
    assert request.model == "some-video-model"
    assert request.first_frame == first
    assert request.last_frame == last
    assert request.duration_seconds == 5.5
    assert request.aspect_ratio == "9:16"
    assert request.fps == 30
    assert request.format is VideoFormat.WEBM
    assert request.seed == 11
    assert request.count == 2
    assert request.generate_audio is True
    assert request.audio_prompt == "soft ocean ambience"

    with pytest.raises(ValidationError):
        VideoRequest(prompt="   ")
    with pytest.raises(ValidationError, match="together"):
        VideoRequest(prompt="x", width=1080)
    with pytest.raises(ValidationError, match="audio_prompt"):
        VideoRequest(prompt="x", audio_prompt="ambience")
    with pytest.raises(ValidationError):
        VideoRequest(prompt="x", fps=0)
    with pytest.raises(ValidationError):
        VideoRequest(prompt="x", count=0)


def test_video_result_and_response_helpers() -> None:
    result = VideoResult(
        data=b"fake-video",
        format=VideoFormat.MP4,
        width=1080,
        height=1920,
        duration_seconds=5.0,
        fps=30,
        has_audio=True,
        seed=2,
        thumbnail_url=_IMAGE_URL,
    )
    response = VideoResponse(
        videos=[result],
        model="some-video-model",
        provider="fake-video",
    )

    assert result.is_inline is True
    assert result.size_bytes == 10
    assert result.dimensions == MediaDimensions(width=1080, height=1920)
    assert result.thumbnail_url == _IMAGE_URL
    assert response.first is result
    assert response.count == 1

    remote = VideoResult(url=_VIDEO_URL, has_audio=False)
    assert remote.is_inline is False
    assert remote.size_bytes is None
    assert remote.dimensions is None

    with pytest.raises(ValidationError, match="exactly one"):
        VideoResult()
    with pytest.raises(ValidationError, match="exactly one"):
        VideoResult(data=b"fake", url=_VIDEO_URL)
    with pytest.raises(ValidationError, match="non-empty"):
        VideoResult(data=b"")
    with pytest.raises(ValidationError, match="together"):
        VideoResult(data=b"fake", width=1080)
    with pytest.raises(ValidationError):
        VideoResponse(videos=[])


def test_music_request_validates_vocals_and_options() -> None:
    request = MusicRequest(
        prompt="  a warm synth pop song  ",
        lyrics="  verse one  ",
        model="  some-music-model  ",
        duration_seconds=45.0,
        format=AudioFormat.WAV,
        sample_rate=48_000,
        tempo_bpm=120.0,
        style="  synth pop  ",
        seed=5,
        count=2,
    )

    assert request.prompt == "a warm synth pop song"
    assert request.lyrics == "verse one"
    assert request.model == "some-music-model"
    assert request.duration_seconds == 45.0
    assert request.format is AudioFormat.WAV
    assert request.sample_rate == 48_000
    assert request.tempo_bpm == 120.0
    assert request.style == "synth pop"
    assert request.seed == 5
    assert request.count == 2

    instrumental = MusicRequest(prompt="calm piano", instrumental=True)
    assert instrumental.lyrics is None

    with pytest.raises(ValidationError):
        MusicRequest(prompt="   ")
    with pytest.raises(ValidationError, match="instrumental"):
        MusicRequest(prompt="x", instrumental=True, lyrics="words")
    with pytest.raises(ValidationError):
        MusicRequest(prompt="x", duration_seconds=0)
    with pytest.raises(ValidationError):
        MusicRequest(prompt="x", count=0)


def test_music_track_and_response_helpers() -> None:
    track = MusicTrack(
        data=b"fake-music",
        format=AudioFormat.MP3,
        duration_seconds=30.0,
        sample_rate=44_100,
        channels=2,
        seed=1,
        title="  Synthetic Song  ",
        lyrics="  la la  ",
    )
    response = MusicResponse(
        tracks=[track],
        model="some-music-model",
        provider="fake-music",
    )

    assert track.is_inline is True
    assert track.size_bytes == 10
    assert track.title == "Synthetic Song"
    assert track.lyrics == "la la"
    assert track.has_lyrics is True
    assert response.first is track
    assert response.count == 1
    assert response.has_lyrics is True

    remote = MusicTrack(url=_AUDIO_URL)
    assert remote.is_inline is False
    assert remote.size_bytes is None
    assert remote.has_lyrics is False

    with pytest.raises(ValidationError, match="exactly one"):
        MusicTrack()
    with pytest.raises(ValidationError, match="exactly one"):
        MusicTrack(data=b"fake", url=_AUDIO_URL)
    with pytest.raises(ValidationError, match="non-empty"):
        MusicTrack(data=b"")
    with pytest.raises(ValidationError):
        MusicResponse(tracks=[])


def test_sfx_request_and_response_helpers() -> None:
    request = SFXRequest(
        prompt="  a door creaking open  ",
        model="  some-sfx-model  ",
        duration_seconds=1.5,
        format=AudioFormat.OGG,
        sample_rate=48_000,
        loop=True,
        seed=9,
        count=2,
    )

    assert request.prompt == "a door creaking open"
    assert request.model == "some-sfx-model"
    assert request.duration_seconds == 1.5
    assert request.format is AudioFormat.OGG
    assert request.sample_rate == 48_000
    assert request.loop is True
    assert request.seed == 9
    assert request.count == 2

    clip = SFXClip(
        data=b"fake-sfx",
        format=AudioFormat.OGG,
        duration_seconds=1.5,
        loop=True,
        seed=9,
    )
    response = SFXResponse(
        clips=[clip],
        model="some-sfx-model",
        provider="fake-sfx",
    )

    assert clip.is_inline is True
    assert clip.size_bytes == 8
    assert response.first is clip
    assert response.count == 1

    remote = SFXClip(url=_AUDIO_URL)
    assert remote.is_inline is False
    assert remote.size_bytes is None

    with pytest.raises(ValidationError):
        SFXRequest(prompt="   ")
    with pytest.raises(ValidationError):
        SFXRequest(prompt="x", duration_seconds=0)
    with pytest.raises(ValidationError, match="exactly one"):
        SFXClip()
    with pytest.raises(ValidationError, match="exactly one"):
        SFXClip(data=b"fake", url=_AUDIO_URL)
    with pytest.raises(ValidationError):
        SFXResponse(clips=[])


def test_stock_request_validates_filters_and_duration_range() -> None:
    request = StockRequest(
        query="  ocean waves  ",
        kind=StockMediaKind.VIDEO,
        orientation=StockOrientation.LANDSCAPE,
        min_width=1920,
        min_height=1080,
        min_duration_seconds=5.0,
        max_duration_seconds=30.0,
        color="  blue  ",
        language=" EN ",
        safe_search=False,
        per_page=50,
        page=2,
    )

    assert request.query == "ocean waves"
    assert request.kind is StockMediaKind.VIDEO
    assert request.orientation is StockOrientation.LANDSCAPE
    assert request.min_width == 1920
    assert request.min_height == 1080
    assert request.min_duration_seconds == 5.0
    assert request.max_duration_seconds == 30.0
    assert request.color == "blue"
    assert request.language == "EN"
    assert request.safe_search is False
    assert request.per_page == 50
    assert request.page == 2

    with pytest.raises(ValidationError):
        StockRequest(query="   ")
    with pytest.raises(ValidationError, match="min_duration_seconds"):
        StockRequest(
            query="x",
            min_duration_seconds=30.0,
            max_duration_seconds=5.0,
        )
    with pytest.raises(ValidationError):
        StockRequest(query="x", per_page=0)
    with pytest.raises(ValidationError):
        StockRequest(query="x", page=0)


def test_stock_asset_and_response_helpers() -> None:
    asset = StockAsset(
        id="  asset-1  ",
        provider="  fake-stock  ",
        kind=StockMediaKind.IMAGE,
        url=f"  {_IMAGE_URL}  ",
        preview_url="https://example.com/media/preview.png",
        page_url="https://example.com/stock/asset-1",
        width=1920,
        height=1080,
        format=ImageFormat.JPEG,
        creator="  Synthetic Creator  ",
        creator_url="https://example.com/creator",
        license="  CC0  ",
        license_url="https://example.com/license",
        attribution="  Synthetic Creator / example.com  ",
    )
    response = StockResponse(
        query="  ocean waves  ",
        assets=[asset],
        provider="fake-stock",
        total_results=1,
        page=1,
        next_page=2,
    )

    assert asset.id == "asset-1"
    assert asset.provider == "fake-stock"
    assert asset.url == _IMAGE_URL
    assert asset.creator == "Synthetic Creator"
    assert asset.license == "CC0"
    assert asset.attribution == "Synthetic Creator / example.com"
    assert asset.dimensions == MediaDimensions(width=1920, height=1080)
    assert asset.is_video is False
    assert asset.has_attribution is True
    assert response.query == "ocean waves"
    assert response.urls == [_IMAGE_URL]
    assert response.count == 1
    assert response.first is asset

    video = StockAsset(
        id="clip-1",
        kind=StockMediaKind.VIDEO,
        url=_VIDEO_URL,
        width=1080,
        height=1920,
        duration_seconds=12.0,
        format=VideoFormat.MP4,
    )
    assert video.is_video is True
    assert video.has_attribution is False

    empty = StockResponse(query="nothing")
    assert empty.count == 0
    assert empty.first is None
    assert empty.urls == []

    with pytest.raises(ValidationError, match="together"):
        StockAsset(id="x", url=_IMAGE_URL, width=10)
    with pytest.raises(ValidationError):
        StockAsset(id="x", url="ftp://example.com/image.png")
    with pytest.raises(ValidationError):
        StockResponse(query="   ")


def test_media_protocol_models_are_immutable_and_reject_unknown_fields() -> None:
    image = ImageRequest(prompt="hello")
    video = VideoRequest(prompt="hello")
    music = MusicRequest(prompt="hello")
    sfx = SFXRequest(prompt="hello")
    stock = StockRequest(query="hello")
    source = MediaInput(data=b"fake")

    with pytest.raises(ValidationError):
        image.count = 2
    with pytest.raises(ValidationError):
        video.fps = 24
    with pytest.raises(ValidationError):
        music.instrumental = True
    with pytest.raises(ValidationError):
        sfx.loop = True
    with pytest.raises(ValidationError):
        stock.page = 2
    with pytest.raises(ValidationError):
        source.url = _IMAGE_URL

    with pytest.raises(ValidationError):
        ImageRequest.model_validate({"prompt": "hello", "unknown": True})
    with pytest.raises(ValidationError):
        ImageEditRequest.model_validate(
            {"image": {"data": b"fake"}, "prompt": "x", "unknown": True}
        )
    with pytest.raises(ValidationError):
        VideoRequest.model_validate({"prompt": "hello", "unknown": True})
    with pytest.raises(ValidationError):
        MusicRequest.model_validate({"prompt": "hello", "unknown": True})
    with pytest.raises(ValidationError):
        SFXRequest.model_validate({"prompt": "hello", "unknown": True})
    with pytest.raises(ValidationError):
        StockRequest.model_validate({"query": "hello", "unknown": True})


def test_fake_image_provider_returns_images_and_records_requests() -> None:
    fake = FakeImageProvider()
    request = ImageRequest(prompt="hello", width=256, height=256, count=2, seed=4)

    response = asyncio.run(fake.generate(request))

    assert isinstance(response, ImageResponse)
    assert response.provider == "fake-image"
    assert response.model == "fake-image-1"
    assert response.count == 2
    assert all(image.data == b"FAKEIMG:hello" for image in response.images)
    assert all(
        image.dimensions == MediaDimensions(width=256, height=256)
        for image in response.images
    )
    assert all(image.seed == 4 for image in response.images)
    assert fake.requests == [request]


def test_fake_image_edit_provider_returns_edited_image() -> None:
    fake = FakeImageEditProvider()
    request = ImageEditRequest(
        image=MediaInput(data=b"fake"),
        operation=ImageEditOperation.REMOVE_BACKGROUND,
    )

    response = asyncio.run(fake.edit(request))

    assert isinstance(response, ImageEditResponse)
    assert response.operation is ImageEditOperation.REMOVE_BACKGROUND
    assert response.first.data == b"FAKEEDIT:remove_background"
    assert response.provider == "fake-image-edit"
    assert fake.requests == [request]


def test_fake_video_provider_returns_clips_with_audio_state() -> None:
    fake = FakeVideoProvider()
    request = VideoRequest(
        prompt="hello",
        duration_seconds=6.0,
        fps=24,
        count=2,
        generate_audio=True,
    )

    response = asyncio.run(fake.generate(request))

    assert isinstance(response, VideoResponse)
    assert response.count == 2
    assert all(video.has_audio is True for video in response.videos)
    assert all(video.duration_seconds == 6.0 for video in response.videos)
    assert all(video.fps == 24 for video in response.videos)
    assert fake.requests == [request]


def test_fake_music_provider_returns_tracks_and_lyrics() -> None:
    fake = FakeMusicProvider()
    request = MusicRequest(prompt="hello", lyrics="la la", count=2)

    response = asyncio.run(fake.generate(request))

    assert isinstance(response, MusicResponse)
    assert response.count == 2
    assert response.has_lyrics is True
    assert all(track.lyrics == "la la" for track in response.tracks)
    assert all(track.title == "hello" for track in response.tracks)
    assert fake.requests == [request]


def test_fake_sfx_provider_returns_clips_and_looping() -> None:
    fake = FakeSFXProvider()
    request = SFXRequest(prompt="hello", loop=True, count=2)

    response = asyncio.run(fake.generate(request))

    assert isinstance(response, SFXResponse)
    assert response.count == 2
    assert all(clip.loop is True for clip in response.clips)
    assert all(clip.data == b"FAKESFX:hello" for clip in response.clips)
    assert fake.requests == [request]


def test_fake_stock_provider_returns_licensed_assets() -> None:
    fake = FakeStockProvider()
    request = StockRequest(query="ocean waves", kind=StockMediaKind.VIDEO, per_page=2)

    response = asyncio.run(fake.search(request))

    assert isinstance(response, StockResponse)
    assert response.provider == "fake-stock"
    assert response.total_results == 3
    assert response.count == 2
    assert all(asset.kind is StockMediaKind.VIDEO for asset in response.assets)
    assert all(asset.has_attribution for asset in response.assets)
    assert all(asset.is_video for asset in response.assets)
    assert fake.requests == [request]


def test_fake_media_providers_propagate_typed_errors() -> None:
    error = RateLimitError("slow down", retry_after=1.0)

    with pytest.raises(RateLimitError):
        asyncio.run(
            FakeImageProvider(error=error).generate(ImageRequest(prompt="hello"))
        )
    with pytest.raises(RateLimitError):
        asyncio.run(
            FakeImageEditProvider(error=error).edit(
                ImageEditRequest(image=MediaInput(data=b"fake"), prompt="hello")
            )
        )
    with pytest.raises(RateLimitError):
        asyncio.run(
            FakeVideoProvider(error=error).generate(VideoRequest(prompt="hello"))
        )
    with pytest.raises(RateLimitError):
        asyncio.run(
            FakeMusicProvider(error=error).generate(MusicRequest(prompt="hello"))
        )
    with pytest.raises(RateLimitError):
        asyncio.run(FakeSFXProvider(error=error).generate(SFXRequest(prompt="hello")))
    with pytest.raises(RateLimitError):
        asyncio.run(FakeStockProvider(error=error).search(StockRequest(query="hello")))
