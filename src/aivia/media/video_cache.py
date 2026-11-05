"""Download, validate, store and reuse generated video clips.

A video provider result is untrusted: it may be inline bytes, a remote URL that
serves an error page, or a container different from the one the request asked
for. This module keeps the video-specific half of the media path in one place:

* a URL is downloaded through the shared HTTP client and the bytes that arrive
  are inspected with ffprobe, never trusted from a filename or Content-Type;
* only a validated clip is written to storage and recorded as a video
  :class:`~aivia.core.models.asset.Asset`;
* an identical :class:`~aivia.providers.protocols.video.VideoRequest` is
  answered from the asset cache before any provider or download is touched.

The cache key deliberately leaves the provider empty, so a repeated request can
be answered before the router has selected a provider. The stored asset still
records the provider and model that actually served the call. A cache hit
records a zero-amount cost event, so it is visible in the ledger without
counting as spend.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from sqlalchemy import Engine

from aivia.core.asset_cache import AssetCache, AssetCacheKey
from aivia.core.config.media import (
    DEFAULT_MEDIA_FFPROBE_TIMEOUT_SECONDS as DEFAULT_FFPROBE_TIMEOUT_SECONDS,
)
from aivia.core.errors import ValidationError
from aivia.core.media_probe import (
    FFprobeRunner,
    MediaCategory,
    MediaDownloader,
    MediaFormat,
    MediaLimits,
    MediaProbe,
    sniff_media,
    validate_media,
)
from aivia.core.models.asset import (
    Asset,
    AssetAnalysis,
    AssetCostLink,
    AssetKind,
)
from aivia.core.models.cost import CostEvent
from aivia.media.provenance import alt_text_from, license_from
from aivia.providers.protocols.media import MediaInput, VideoFormat
from aivia.providers.protocols.storage import (
    StorageObject,
    StorageProvider,
    StoragePutRequest,
)
from aivia.providers.protocols.video import VideoRequest, VideoResult
from aivia.providers.video.flags import (
    FFmpegRunner,
    should_strip_audio,
    strip_audio,
)

__all__ = [
    "VIDEO_CACHE_OPERATION",
    "VIDEO_OPERATION",
    "MaterialisedVideo",
    "VideoAssetCache",
    "VideoCache",
    "VideoCacheOutcome",
    "VideoCacheResult",
    "cache_video",
    "cached_video",
    "load_cached_video",
    "store_video",
    "validated_video_format",
    "validated_video_fps",
    "validated_video_has_audio",
    "video_cache_key",
]

VIDEO_OPERATION = "video_gen"
"""The default operation recorded for a video cache hit."""

VIDEO_CACHE_OPERATION = VIDEO_OPERATION
"""Alias for :data:`VIDEO_OPERATION`."""

_VIDEO_FORMATS: dict[MediaFormat, VideoFormat] = {
    MediaFormat.MP4: VideoFormat.MP4,
    MediaFormat.WEBM: VideoFormat.WEBM,
    MediaFormat.MOV: VideoFormat.MOV,
    MediaFormat.MKV: VideoFormat.MKV,
    MediaFormat.GIF: VideoFormat.GIF,
}


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _file_path(url: str) -> Path | None:
    parsed = urlsplit(url)
    if parsed.scheme.lower() != "file":
        return None
    return Path(unquote(parsed.path))


def _frame_hash(frame: MediaInput | None, role: str) -> str | None:
    if frame is None:
        return None
    if frame.data is not None:
        digest = hashlib.sha256(frame.data).hexdigest()
    elif frame.url is not None:
        digest = hashlib.sha256(frame.url.encode("utf-8")).hexdigest()
    else:
        return None
    return f"{role}:{digest}"


def _reference_hashes(request: VideoRequest) -> tuple[str, ...]:
    hashes = [
        value
        for value in (
            _frame_hash(request.first_frame, "first"),
            _frame_hash(request.last_frame, "last"),
        )
        if value is not None
    ]
    return tuple(hashes)


def parameters_for_request(
    request: VideoRequest,
    *,
    index: int = 0,
) -> dict[str, Any]:
    """Return the request values that affect one generated clip.

    Prompt, model and frame inputs are keyed separately, so they are excluded
    here. ``metadata`` and ``timeout_seconds`` do not change the produced video
    and are deliberately omitted.
    """
    if not isinstance(request, VideoRequest):
        raise TypeError("request must be a VideoRequest")
    return {
        "negative_prompt": request.negative_prompt,
        "model": request.model,
        "duration_seconds": request.duration_seconds,
        "width": request.width,
        "height": request.height,
        "aspect_ratio": request.aspect_ratio,
        "fps": request.fps,
        "format": request.format.value,
        "seed": request.seed,
        "count": request.count,
        "index": index,
        "generate_audio": request.generate_audio,
        "audio_prompt": request.audio_prompt,
    }


def video_cache_key(
    request: VideoRequest,
    *,
    index: int = 0,
    cache: AssetCache | None = None,
) -> AssetCacheKey:
    """Build the cache key for one output of *request*.

    The provider is left empty on purpose: a cache hit is answered before the
    router has selected one.
    """
    if not isinstance(request, VideoRequest):
        raise TypeError("request must be a VideoRequest")
    parameters = parameters_for_request(request, index=index)
    reference_hashes = _reference_hashes(request)
    model = request.model or ""
    if cache is None:
        return AssetCacheKey(
            provider="",
            model=model,
            prompt=request.prompt,
            parameters=parameters,
            reference_hashes=reference_hashes,
        )
    return cache.key(
        "",
        model,
        request.prompt,
        parameters=parameters,
        reference_hashes=reference_hashes,
    )


def _video_format(probe: MediaProbe, fallback: VideoFormat) -> VideoFormat:
    return _VIDEO_FORMATS.get(probe.format, fallback)


def validated_video_format(
    probe: MediaProbe,
    fallback: VideoFormat,
) -> VideoFormat:
    """Return the container ffprobe validated, falling back to *fallback*."""
    return _video_format(probe, fallback)


def _frame_rate_fps(value: str | None) -> int | None:
    """Parse an ffprobe frame-rate fraction such as ``30000/1001``."""
    if not value:
        return None
    numerator_text, separator, denominator_text = value.strip().partition("/")
    try:
        numerator = float(numerator_text)
        denominator = float(denominator_text) if separator else 1.0
    except ValueError:
        return None
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return None
    if numerator <= 0.0 or denominator <= 0.0:
        return None
    rate = numerator / denominator
    if not math.isfinite(rate) or rate <= 0.0:
        return None
    return round(rate)


def validated_video_fps(
    probe: MediaProbe,
    fallback: int | None = None,
) -> int | None:
    """Return the frame rate ffprobe measured, or *fallback* when unknown."""
    info = probe.ffprobe
    if info is not None:
        stream = info.primary_video
        if stream is not None:
            measured = _frame_rate_fps(stream.frame_rate)
            if measured is not None:
                return measured
    return fallback


def validated_video_has_audio(probe: MediaProbe, fallback: bool) -> bool:
    """Return whether ffprobe found an audio stream, or *fallback*."""
    if probe.ffprobe is not None:
        return bool(probe.ffprobe.audio_streams)
    return bool(fallback)


def _stored_video_format(
    analysis: AssetAnalysis,
    fallback: VideoFormat,
) -> VideoFormat:
    value = _text(analysis.format)
    if value:
        try:
            return VideoFormat(value)
        except ValueError:
            pass
    return fallback


@dataclass(frozen=True, slots=True)
class MaterialisedVideo:
    """One validated clip and whether its unwanted audio was removed."""

    probe: MediaProbe
    storage_object: StorageObject | None = None
    audio_stripped: bool = False

    @property
    def stored(self) -> StorageObject | None:
        """Alias for :attr:`storage_object`."""
        return self.storage_object


@dataclass(frozen=True, slots=True)
class VideoCacheOutcome:
    """One stored or reused video together with its asset and cache metadata.

    ``probe`` and ``storage_object`` are present on a fresh store. A cache hit
    only carries the reconstructed :class:`VideoResult` and the existing asset;
    ``event`` is then the zero-amount hit record. ``audio_stripped`` records
    that audio the request did not want was removed before storage.
    """

    request: VideoRequest
    result: VideoResult
    asset: Asset
    probe: MediaProbe | None = None
    storage_object: StorageObject | None = None
    event: CostEvent | None = None
    key: AssetCacheKey | None = None
    cache_hit: bool = False
    audio_stripped: bool = False

    @property
    def stripped_audio(self) -> bool:
        """Alias for :attr:`audio_stripped`."""
        return self.audio_stripped

    @property
    def cached(self) -> bool:
        """Whether this outcome came from the asset cache."""
        return self.cache_hit

    @property
    def asset_id(self) -> str:
        """The stored asset identifier."""
        return self.asset.id

    @property
    def path(self) -> str:
        """The stored asset path or URL."""
        return self.asset.path

    @property
    def data(self) -> bytes | None:
        """The validated clip bytes."""
        return self.result.data

    @property
    def assets(self) -> tuple[Asset, ...]:
        """The one stored asset, in a tuple for media-service symmetry."""
        return (self.asset,)

    @property
    def videos(self) -> list[VideoResult]:
        """The one video, in a list for media-service symmetry."""
        return [self.result]

    @property
    def format(self) -> VideoFormat:
        """The validated container format."""
        return self.result.format

    @property
    def width(self) -> int | None:
        """The validated pixel width, when ffprobe reported it."""
        return self.result.width

    @property
    def height(self) -> int | None:
        """The validated pixel height, when ffprobe reported it."""
        return self.result.height

    @property
    def duration_seconds(self) -> float | None:
        """The validated duration in seconds, when known."""
        return self.result.duration_seconds

    @property
    def fps(self) -> int | None:
        """The requested frame rate, when known."""
        return self.result.fps

    @property
    def has_audio(self) -> bool:
        """Whether the clip carries an audio stream."""
        return self.result.has_audio

    def __getattr__(self, name: str) -> Any:
        """Delegate asset fields such as ``id``, ``kind`` and ``analysis``."""
        return getattr(self.asset, name)


VideoCacheResult = VideoCacheOutcome
"""Alias for :class:`VideoCacheOutcome`."""


class VideoCache:
    """Validate, store and reuse generated video clips by their request.

    Pass an SQLAlchemy :class:`~sqlalchemy.Engine` or a ready
    :class:`~aivia.core.asset_cache.AssetCache`. A storage provider is needed
    to persist a clip so a later identical request can be served from disk.
    ``downloader`` may be supplied in tests to avoid the network; when it is
    omitted the shared :class:`~aivia.core.media_probe.MediaDownloader` is
    built lazily.

    A cached clip is verified against the asset's recorded SHA-256 digest
    before it is returned. Because those exact bytes were already probed when
    they were stored, the hit does not download or run ffprobe again.
    """

    def __init__(
        self,
        engine: Engine | None = None,
        *,
        cache: AssetCache | None = None,
        storage: StorageProvider | None = None,
        downloader: MediaDownloader | None = None,
        limits: MediaLimits | None = None,
        no_cache: bool = False,
        ffprobe_executable: str = "ffprobe",
        ffprobe_timeout_seconds: float = DEFAULT_FFPROBE_TIMEOUT_SECONDS,
        ffprobe_runner: FFprobeRunner | None = None,
        ffmpeg_executable: str = "ffmpeg",
        ffmpeg_runner: FFmpegRunner | None = None,
    ) -> None:
        if engine is not None and cache is not None:
            raise TypeError("pass either an engine or a cache, not both")
        if cache is not None and not isinstance(cache, AssetCache):
            raise TypeError("cache must be an AssetCache")
        if downloader is not None and not isinstance(downloader, MediaDownloader):
            raise TypeError("downloader must be a MediaDownloader")
        if cache is None and engine is not None:
            cache = AssetCache(engine, no_cache=no_cache)
        if cache is not None and no_cache:
            cache.no_cache = True
        self._cache = cache
        self._storage = storage
        self._downloader = downloader
        self._limits = limits or MediaLimits()
        self._ffprobe_executable = ffprobe_executable
        self._ffprobe_timeout_seconds = ffprobe_timeout_seconds
        self._ffprobe_runner = ffprobe_runner
        self._ffmpeg_executable = ffmpeg_executable
        self._ffmpeg_runner = ffmpeg_runner
        if downloader is not None:
            if ffprobe_runner is None:
                self._ffprobe_runner = downloader.ffprobe_runner
            if downloader.ffprobe_executable:
                self._ffprobe_executable = downloader.ffprobe_executable
            self._ffprobe_timeout_seconds = downloader.ffprobe_timeout_seconds

    @property
    def cache(self) -> AssetCache | None:
        """The asset cache, when one was configured."""
        return self._cache

    @property
    def storage(self) -> StorageProvider | None:
        """The storage provider, when one was configured."""
        return self._storage

    @property
    def limits(self) -> MediaLimits:
        """The caps applied to every downloaded or inline video."""
        return self._limits

    def key(self, request: VideoRequest, *, index: int = 0) -> AssetCacheKey:
        """Return the cache key for one output of *request*."""
        return video_cache_key(request, index=index, cache=self._cache)

    def cache_key(self, request: VideoRequest, *, index: int = 0) -> AssetCacheKey:
        """Alias for :meth:`key`."""
        return self.key(request, index=index)

    def key_for_request(
        self,
        request: VideoRequest,
        *,
        index: int = 0,
    ) -> AssetCacheKey:
        """Alias for :meth:`key`."""
        return self.key(request, index=index)

    async def lookup(
        self,
        request: VideoRequest,
        *,
        index: int = 0,
        run_id: str = "",
        operation: str = VIDEO_OPERATION,
        record_hit: bool = True,
        storage: StorageProvider | None = None,
    ) -> VideoCacheOutcome | None:
        """Return an identical cached clip, or ``None`` on a miss.

        The stored payload's digest must match the asset's recorded hash before
        it is trusted. A mismatch raises :class:`ValidationError` rather than
        silently serving different bytes.
        """
        if not isinstance(request, VideoRequest):
            raise TypeError("request must be a VideoRequest")
        cache = self._cache
        if cache is None:
            return None
        key = self.key(request, index=index)
        entry = cache.entry(key)
        if entry is None:
            return None
        asset = cache.assets.get(entry.asset_id)
        if asset is None or asset.kind is not AssetKind.VIDEO:
            return None
        payload = await self._cached_payload(asset, storage or self._storage)
        if payload is None:
            return None
        digest = hashlib.sha256(payload).hexdigest()
        if asset.content_hash and digest != asset.content_hash:
            raise ValidationError(
                "cached video content does not match its recorded hash",
                details={
                    "asset_id": asset.id,
                    "expected_sha256": asset.content_hash,
                    "actual_sha256": digest,
                },
            )
        signature = sniff_media(payload)
        if signature is None:
            raise ValidationError(
                "cached video format could not be recognised",
                details={"asset_id": asset.id},
            )
        result = VideoResult(
            data=payload,
            format=_stored_video_format(
                asset.analysis,
                _VIDEO_FORMATS.get(signature.format, request.format),
            ),
            width=asset.analysis.width,
            height=asset.analysis.height,
            duration_seconds=asset.analysis.duration_seconds,
            fps=(asset.analysis.fps if asset.analysis.fps is not None else request.fps),
            has_audio=(
                asset.analysis.has_audio
                if asset.analysis.has_audio is not None
                else bool(request.generate_audio)
            ),
            seed=asset.seed,
        )
        event = (
            cache.record_hit(key, asset, run_id=run_id, operation=operation)
            if record_hit and not cache.no_cache
            else None
        )
        return VideoCacheOutcome(
            request=request,
            result=result,
            asset=asset,
            event=event,
            key=key,
            cache_hit=True,
            audio_stripped=bool(asset.analysis.audio_stripped),
        )

    async def get(
        self,
        request: VideoRequest,
        **values: Any,
    ) -> VideoCacheOutcome | None:
        """Alias for :meth:`lookup`."""
        return await self.lookup(request, **values)

    async def store(
        self,
        request: VideoRequest,
        result: VideoResult,
        *,
        provider: str = "",
        model: str | None = None,
        index: int = 0,
        run_id: str = "",
        seed: int | None = None,
        pinned: bool = True,
        storage: StorageProvider | None = None,
        key: str | None = None,
        key_prefix: str | None = None,
        cost_links: Iterable[AssetCostLink] | None = None,
        license: Any = None,
        alt_text: str | None = None,
    ) -> VideoCacheOutcome:
        """Validate *result*, store it as a video asset and cache the request.

        Inline bytes and remote URLs follow the same path after the bytes have
        arrived: sniff the container, run ffprobe, enforce the media limits,
        write the validated payload to storage, then record the asset under the
        request key. Nothing is stored or cached when validation fails.

        ``cost_links`` attaches the cost events that produced the clip to the
        stored asset, so a caller that records a video event does not have to
        update the asset in a second write.
        """
        if not isinstance(request, VideoRequest):
            raise TypeError("request must be a VideoRequest")
        if not isinstance(result, VideoResult):
            raise TypeError("result must be a VideoResult")
        active_storage = storage if storage is not None else self._storage
        materialised = await self.materialise_with_policy(
            result,
            request,
            index=index,
            run_id=run_id,
            storage=active_storage,
            key=key,
            key_prefix=key_prefix,
        )
        probe = materialised.probe
        stored = materialised.storage_object
        audio_stripped = materialised.audio_stripped
        provider_name = _text(provider)
        model_name = _text(model) if model is not None else (request.model or "")
        selected_seed = (
            seed
            if seed is not None
            else (result.seed if result.seed is not None else request.seed)
        )
        video_format = validated_video_format(probe, result.format)
        video_fps = validated_video_fps(
            probe,
            result.fps if result.fps is not None else request.fps,
        )
        video_has_audio = validated_video_has_audio(probe, result.has_audio)
        asset = Asset(
            kind=AssetKind.VIDEO,
            provider=provider_name,
            model=model_name,
            prompt=request.prompt,
            path=(
                (stored.url or stored.key) if stored is not None else (result.url or "")
            ),
            content_hash=probe.sha256,
            run_id=run_id,
            pinned=pinned,
            seed=selected_seed,
            license=license_from(
                license,
                request.metadata,
            ),
            analysis=AssetAnalysis(
                width=probe.width,
                height=probe.height,
                duration_seconds=probe.duration_seconds,
                fps=video_fps,
                format=video_format.value,
                has_audio=video_has_audio,
                audio_stripped=audio_stripped,
            ),
            alt_text=(
                alt_text.strip()
                if isinstance(alt_text, str) and alt_text.strip()
                else alt_text_from(request.metadata)
            ),
            cost_links=list(cost_links or ()),
        )
        if self._cache is not None and not self._cache.no_cache:
            asset = self._cache.store(asset, key=self.key(request, index=index))
        video = VideoResult(
            data=probe.data,
            format=video_format,
            width=probe.width,
            height=probe.height,
            duration_seconds=probe.duration_seconds,
            fps=video_fps,
            has_audio=video_has_audio,
            seed=asset.seed,
            metadata=dict(result.metadata),
        )
        return VideoCacheOutcome(
            request=request,
            result=video,
            asset=asset,
            probe=probe,
            storage_object=stored,
            key=self.key(request, index=index),
            cache_hit=False,
            audio_stripped=audio_stripped,
        )

    async def put(
        self,
        request: VideoRequest,
        result: VideoResult,
        **values: Any,
    ) -> VideoCacheOutcome:
        """Alias for :meth:`store`."""
        return await self.store(request, result, **values)

    async def materialise(
        self,
        result: VideoResult,
        request: VideoRequest,
        *,
        index: int = 0,
        run_id: str = "",
        storage: StorageProvider | None = None,
        key: str | None = None,
        key_prefix: str | None = None,
    ) -> tuple[MediaProbe, StorageObject | None]:
        """Validate *result* and optionally write its bytes to *storage*.

        A URL result is downloaded and probed by the shared downloader; inline
        bytes are probed directly. The returned probe is the only evidence a
        caller should use for dimensions and duration.
        """
        materialised = await self.materialise_with_policy(
            result,
            request,
            index=index,
            run_id=run_id,
            storage=storage,
            key=key,
            key_prefix=key_prefix,
        )
        return materialised.probe, materialised.storage_object

    async def materialise_with_policy(
        self,
        result: VideoResult,
        request: VideoRequest,
        *,
        index: int = 0,
        run_id: str = "",
        storage: StorageProvider | None = None,
        key: str | None = None,
        key_prefix: str | None = None,
    ) -> MaterialisedVideo:
        """Validate *result*, apply the audio policy and optionally store it.

        Audio the request did not ask for is removed before the bytes reach
        storage, so an asset never records a soundtrack the caller rejected.
        The stripped payload is probed again before it is written, and
        ``audio_stripped`` records that removal happened.
        """
        if result.data is not None:
            payload = result.data
            probe = validate_media(
                payload,
                limits=self._limits,
                expected_format=result.format.value,
                ffprobe_executable=self._ffprobe_executable,
                ffprobe_timeout_seconds=self._ffprobe_timeout_seconds,
                ffprobe_runner=self._ffprobe_runner,
            )
        else:
            if result.url is None:
                raise ValueError("video result has neither data nor url")
            probe = await self._downloader_for().download(
                result.url,
                limits=self._limits,
                expected_format=result.format.value,
            )
            payload = probe.data

        if (
            probe.category is not MediaCategory.VIDEO
            and probe.format is not MediaFormat.GIF
        ):
            raise ValidationError(
                "provider result is not a video",
                details={"format": probe.format.value},
            )

        audio_stripped = False
        if should_strip_audio(result, request):
            payload = strip_audio(
                payload,
                ffmpeg_executable=self._ffmpeg_executable,
                ffprobe_executable=self._ffprobe_executable,
                timeout_seconds=self._ffprobe_timeout_seconds,
                ffmpeg_runner=self._ffmpeg_runner,
                ffprobe_runner=self._ffprobe_runner,
            )
            probe = validate_media(
                payload,
                limits=self._limits,
                expected_format=result.format.value,
                ffprobe_executable=self._ffprobe_executable,
                ffprobe_timeout_seconds=self._ffprobe_timeout_seconds,
                ffprobe_runner=self._ffprobe_runner,
            )
            audio_stripped = True

        stored: StorageObject | None = None
        if storage is not None:
            storage_key = self._storage_key(
                probe,
                index=index,
                key=key,
                key_prefix=key_prefix,
                count=request.count,
            )
            metadata = {
                "kind": "video",
                "format": probe.format.value,
                "run_id": run_id,
            }
            if audio_stripped:
                metadata["audio_stripped"] = "true"
            stored = await storage.put(
                StoragePutRequest(
                    data=payload,
                    key=storage_key,
                    content_type=probe.mime_type,
                    metadata=metadata,
                )
            )
        return MaterialisedVideo(
            probe=probe,
            storage_object=stored,
            audio_stripped=audio_stripped,
        )

    def _downloader_for(self) -> MediaDownloader:
        if self._downloader is None:
            self._downloader = MediaDownloader(
                limits=self._limits,
                ffprobe_executable=self._ffprobe_executable,
                ffprobe_timeout_seconds=self._ffprobe_timeout_seconds,
                ffprobe_runner=self._ffprobe_runner,
            )
        return self._downloader

    async def _cached_payload(
        self,
        asset: Asset,
        storage: StorageProvider | None,
    ) -> bytes | None:
        if asset.path:
            path = _file_path(asset.path)
            if path is not None and path.is_file():
                return path.read_bytes()
        if storage is not None:
            if asset.path:
                try:
                    return await storage.get(asset.path)
                except (KeyError, ValueError):
                    pass
            if asset.content_hash:
                digest = asset.content_hash
                key = f"objects/{digest[:2]}/{digest}"
                try:
                    return await storage.get(key)
                except KeyError:
                    return None
        return None

    @staticmethod
    def _storage_key(
        probe: MediaProbe,
        *,
        index: int,
        key: str | None,
        key_prefix: str | None,
        count: int,
    ) -> str | None:
        if key:
            base = key.strip()
            if not base:
                raise ValueError("storage key must be non-empty")
            if count > 1:
                return f"{base}-{index}.{probe.extension}"
            return base
        if key_prefix:
            digest = probe.sha256
            return f"{key_prefix}/{digest[:2]}/{digest}.{probe.extension}"
        return None

    async def aclose(self) -> None:
        """Close the downloader when this cache owns it."""
        if self._downloader is not None:
            await self._downloader.aclose()

    async def close(self) -> None:
        """Alias for :meth:`aclose`."""
        await self.aclose()

    async def __aenter__(self) -> VideoCache:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> None:
        await self.aclose()


VideoAssetCache = VideoCache
"""Alias for :class:`VideoCache`."""


async def cache_video(
    request: VideoRequest,
    result: VideoResult,
    *,
    engine: Engine | None = None,
    cache: AssetCache | None = None,
    storage: StorageProvider | None = None,
    downloader: MediaDownloader | None = None,
    limits: MediaLimits | None = None,
    **values: Any,
) -> VideoCacheOutcome:
    """Store one video without constructing a long-lived cache object."""
    video_cache = VideoCache(
        engine,
        cache=cache,
        storage=storage,
        downloader=downloader,
        limits=limits,
    )
    return await video_cache.store(request, result, **values)


store_video = cache_video
"""Alias for :func:`cache_video`."""


async def cached_video(
    request: VideoRequest,
    *,
    engine: Engine | None = None,
    cache: AssetCache | None = None,
    storage: StorageProvider | None = None,
    downloader: MediaDownloader | None = None,
    limits: MediaLimits | None = None,
    **values: Any,
) -> VideoCacheOutcome | None:
    """Look up one cached video without constructing a long-lived cache."""
    video_cache = VideoCache(
        engine,
        cache=cache,
        storage=storage,
        downloader=downloader,
        limits=limits,
    )
    return await video_cache.lookup(request, **values)


load_cached_video = cached_video
"""Alias for :func:`cached_video`."""
