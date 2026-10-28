"""Deterministic mock providers for every capability.

The mocks exist for offline tests, dry runs and development: they never touch
the network and return the same result for the same request. Media is real
enough for downstream stages to consume it: images are valid PNG files, audio
is a valid WAV payload sized to the requested duration, and video is rendered
with ffmpeg's ``lavfi`` sources when the tool is available.

Every call records a simulated cost derived from the provider metadata, so a
caller can exercise budget and cost reporting without spending money. Failures
are injected explicitly, either as a permanent ``error`` or as a queue of
``failures`` consumed one call at a time; the router can then be tested against
retryable and permanent errors without a fake HTTP server.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import math
import shutil
import struct
import subprocess
import tempfile
import wave
import zlib
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar

from aivia.core.errors import InvalidRequestError, PermanentError
from aivia.core.models.provider import ProviderMeta, ProviderPrice, RateLimit
from aivia.providers.base import Provider
from aivia.providers.protocols import (
    CAPABILITY_CHARACTER_TRAINER,
    CAPABILITY_DELIVERY,
    CAPABILITY_EMBEDDINGS,
    CAPABILITY_IMAGE_EDIT,
    CAPABILITY_IMAGE_GEN,
    CAPABILITY_LIPSYNC,
    CAPABILITY_LLM,
    CAPABILITY_MUSIC_GEN,
    CAPABILITY_NOTIFY,
    CAPABILITY_PAGE_FETCH,
    CAPABILITY_PUBLISH,
    CAPABILITY_SEARCH,
    CAPABILITY_SFX,
    CAPABILITY_STOCK,
    CAPABILITY_STORAGE,
    CAPABILITY_STT,
    CAPABILITY_TTS,
    CAPABILITY_VIDEO_GEN,
    AudioFormat,
    CharacterTrainerRequest,
    CharacterTrainingJob,
    DeliveryMethod,
    DeliveryReceipt,
    DeliveryRequest,
    Embedding,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
    FinishReason,
    ImageEditRequest,
    ImageEditResponse,
    ImageFormat,
    ImageRequest,
    ImageResponse,
    ImageResult,
    LipSyncRequest,
    LipSyncResponse,
    LipSyncResult,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    MusicRequest,
    MusicResponse,
    MusicTrack,
    NotificationMessage,
    NotificationReceipt,
    PageFetchRequest,
    PageFetchResponse,
    PageKind,
    PublishRequest,
    PublishResult,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SFXClip,
    SFXRequest,
    SFXResponse,
    StockAsset,
    StockMediaKind,
    StockRequest,
    StockResponse,
    StorageObject,
    StoragePutRequest,
    STTRequest,
    STTResponse,
    ToolCall,
    ToolChoice,
    TrainingState,
    TranscriptSegment,
    TTSRequest,
    TTSResponse,
    VideoFormat,
    VideoRequest,
    VideoResponse,
    VideoResult,
    WordTimestamp,
)
from aivia.providers.registry import ProviderRegistry

__all__ = [
    "MOCK_EPOCH",
    "MockAll",
    "MockCall",
    "MockCharacterTrainer",
    "MockDelivery",
    "MockDeliveryProvider",
    "MockEmbedding",
    "MockEmbeddingProvider",
    "MockEmbeddings",
    "MockImage",
    "MockImageEdit",
    "MockImageEditProvider",
    "MockImageProvider",
    "MockLLM",
    "MockLLMProvider",
    "MockLipSync",
    "MockLipSyncProvider",
    "MockMusic",
    "MockMusicProvider",
    "MockNotifier",
    "MockNotify",
    "MockPageFetch",
    "MockPageFetchProvider",
    "MockProvider",
    "MockPublish",
    "MockPublisher",
    "MockSFX",
    "MockSFXProvider",
    "MockSTT",
    "MockSTTProvider",
    "MockSearch",
    "MockSearchProvider",
    "MockStock",
    "MockStockProvider",
    "MockStorage",
    "MockStorageProvider",
    "MockTTS",
    "MockTTSProvider",
    "MockVideo",
    "MockVideoProvider",
    "build_mock_providers",
    "mock_provider_for",
    "register_mock_providers",
]

MOCK_EPOCH = datetime(2024, 1, 1, tzinfo=UTC)
"""Fixed instant used for every timestamp a mock returns."""

_DEFAULT_IMAGE_SIZE = 512
_DEFAULT_VIDEO_SIZE = (512, 512)
_DEFAULT_VIDEO_DURATION = 5.0
_DEFAULT_VIDEO_FPS = 30
_DEFAULT_MUSIC_DURATION = 30.0
_DEFAULT_SFX_DURATION = 1.0
_DEFAULT_TTS_SAMPLE_RATE = 24_000
_DEFAULT_AUDIO_SAMPLE_RATE = 44_100
_DEFAULT_EMBEDDING_DIMENSIONS = 8
_WORDS_PER_SECOND = 2.5
_MP3_SILENT_FRAME = b"\xff\xfb\x90\x64" + b"\x00" * 413


@dataclass(frozen=True, slots=True)
class MockCall:
    """One simulated provider call and its deterministic cost."""

    capability: str
    operation: str
    units: float
    cost_usd: float
    request: Any = field(default=None, repr=False, compare=False)

    @property
    def amount_usd(self) -> float:
        """Alias for :attr:`cost_usd`."""
        return self.cost_usd

    @property
    def estimated(self) -> bool:
        """Mock costs are always estimates, never provider reported amounts."""
        return True


def _digest(*parts: object) -> str:
    text = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_seed(value: object, seed: int) -> int:
    return int(_digest(seed, value)[:16], 16)


def _mock_metadata(
    name: str,
    capabilities: Sequence[str],
    prices: Mapping[str, float],
) -> ProviderMeta:
    return ProviderMeta(
        name=name,
        capabilities=list(capabilities),
        pricing={
            capability: ProviderPrice(amount_usd=prices.get(capability, 0.0))
            for capability in capabilities
        },
        rate_limit=RateLimit(requests_per_minute=0),
    )


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return (
        struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)
    )


def _png_bytes(width: int, height: int, seed: int) -> bytes:
    """Return a deterministic, valid RGB PNG of the requested size."""
    width = max(1, width)
    height = max(1, height)
    red = (seed >> 8) & 0xFF
    green = (seed >> 16) & 0xFF
    blue = (seed >> 24) & 0xFF
    rows = bytearray()
    for y in range(height):
        rows.append(0)
        for x in range(width):
            rows.extend(
                (
                    (red + x * 7 + y * 3) & 0xFF,
                    (green + x * 3 + y * 11) & 0xFF,
                    (blue + x * 5 + y * 13) & 0xFF,
                )
            )
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(bytes(rows), level=9))
        + _png_chunk(b"IEND", b"")
    )


def _png_dimensions(data: bytes | None) -> tuple[int, int] | None:
    if data is None or len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    if data[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", data[16:24])
    return (width, height) if width and height else None


def _wav_duration(data: bytes | None) -> float | None:
    if data is None:
        return None
    try:
        with wave.open(io.BytesIO(data), "rb") as reader:
            frame_rate = reader.getframerate()
            if frame_rate <= 0:
                return None
            return reader.getnframes() / frame_rate
    except (wave.Error, EOFError):
        return None


def _wav_bytes(
    duration_seconds: float,
    *,
    sample_rate: int,
    channels: int = 1,
    seed: int = 0,
) -> bytes:
    """Return a valid 16-bit PCM WAV whose duration matches the request."""
    duration = max(duration_seconds, 1.0 / sample_rate)
    sample_rate = max(1, sample_rate)
    channels = max(1, channels)
    total_frames = max(1, round(duration * sample_rate))
    frequency = 220 + seed % 220
    one_second = bytearray()
    for index in range(sample_rate):
        value = int(12_000 * math.sin(2.0 * math.pi * frequency * index / sample_rate))
        one_second.extend(struct.pack("<h", value) * channels)
    repeats, remainder = divmod(total_frames, sample_rate)
    frames = bytes(one_second) * repeats + bytes(one_second[: remainder * 2 * channels])
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(frames)
    return buffer.getvalue()


def _mp3_bytes(duration_seconds: float) -> bytes:
    """Return deterministic silent MPEG frames covering *duration_seconds*."""
    frames = max(1, math.ceil(max(duration_seconds, 0.0) * 44_100 / 1152))
    return _MP3_SILENT_FRAME * frames


def _aspect_dimensions(
    width: int | None,
    height: int | None,
    aspect_ratio: str | None,
    *,
    default: int = _DEFAULT_IMAGE_SIZE,
) -> tuple[int, int]:
    if width is not None and height is not None:
        return width, height
    if aspect_ratio:
        left, _, right = aspect_ratio.partition(":")
        try:
            ratio_width = int(left)
            ratio_height = int(right)
        except ValueError:
            ratio_width = ratio_height = 0
        if ratio_width > 0 and ratio_height > 0:
            if ratio_width >= ratio_height:
                return default, max(1, round(default * ratio_height / ratio_width))
            return max(1, round(default * ratio_width / ratio_height)), default
    return default, default


def _video_dimensions(request: VideoRequest) -> tuple[int, int]:
    if request.width is not None and request.height is not None:
        return request.width, request.height
    if request.aspect_ratio:
        return _aspect_dimensions(None, None, request.aspect_ratio, default=512)
    return _DEFAULT_VIDEO_SIZE


class _MockProviderBase(Provider):
    """Shared deterministic behaviour for the concrete mock providers."""

    capabilities: ClassVar[tuple[str, ...]] = ()
    mock_name: ClassVar[str] = "mock"
    prices: ClassVar[Mapping[str, float]] = {}

    def __init__(
        self,
        metadata: ProviderMeta | Mapping[str, Any] | None = None,
        *,
        name: str | None = None,
        error: BaseException | type[BaseException] | None = None,
        failures: Iterable[BaseException | type[BaseException] | None] | None = None,
        cost_usd: float | None = None,
        seed: int = 0,
        ffmpeg: str = "ffmpeg",
        ffmpeg_runner: Callable[[Sequence[str]], bytes] | None = None,
    ) -> None:
        if metadata is None:
            prices = dict(self.prices)
            if cost_usd is not None:
                prices = dict.fromkeys(self.capabilities, cost_usd)
            metadata = _mock_metadata(
                name or self.mock_name,
                self.capabilities,
                prices,
            )
        super().__init__(metadata, name=name)
        self._error = error
        self._failures: deque[BaseException | type[BaseException] | None] = deque(
            failures or ()
        )
        self._seed = seed
        self._ffmpeg = ffmpeg
        self._ffmpeg_runner = ffmpeg_runner
        self._calls: list[MockCall] = []
        self._total_cost_usd = 0.0
        self._objects: dict[str, tuple[bytes, StorageObject]] = {}
        self._jobs: dict[str, CharacterTrainingJob] = {}

    @property
    def calls(self) -> tuple[MockCall, ...]:
        """Every successful call in order, with its simulated cost."""
        return tuple(self._calls)

    @property
    def costs(self) -> tuple[float, ...]:
        """The simulated cost of every successful call."""
        return tuple(call.cost_usd for call in self._calls)

    @property
    def total_cost_usd(self) -> float:
        """The sum of every simulated call cost."""
        return self._total_cost_usd

    @property
    def total_cost(self) -> float:
        """Alias for :attr:`total_cost_usd`."""
        return self._total_cost_usd

    @property
    def last_cost(self) -> float:
        """Alias for :attr:`last_cost_usd`."""
        return self.last_cost_usd

    @property
    def last_cost_usd(self) -> float:
        """The cost of the most recent call, or ``0.0`` before any call."""
        return self._calls[-1].cost_usd if self._calls else 0.0

    @property
    def last_call(self) -> MockCall | None:
        """The most recent call record, or ``None`` before any call."""
        return self._calls[-1] if self._calls else None

    def estimate_cost(self, capability: str | None = None, units: float = 1.0) -> float:
        """Return the simulated price for *capability* and *units*."""
        target = capability or (self.capabilities[0] if self.capabilities else "")
        price = self.metadata.pricing_for(target)
        return 0.0 if price is None else price.cost_for(units)

    def simulated_cost(
        self,
        capability: str | None = None,
        units: float = 1.0,
    ) -> float:
        """Alias for :meth:`estimate_cost`."""
        return self.estimate_cost(capability, units)

    def fail_next(self, error: BaseException | type[BaseException]) -> None:
        """Raise *error* on the next call, then continue normally."""
        self._failures.append(error)

    def reset(self) -> None:
        """Forget recorded calls, costs and queued failures."""
        self._calls.clear()
        self._total_cost_usd = 0.0
        self._failures.clear()

    @staticmethod
    def _raise_failure(failure: BaseException | type[BaseException]) -> None:
        if isinstance(failure, type):
            try:
                raise failure("injected mock failure")
            except TypeError:
                raise failure() from None
        raise failure

    def _maybe_fail(self, operation: str) -> None:
        if self._error is not None:
            self._raise_failure(self._error)
        if self._failures:
            failure = self._failures.popleft()
            if failure is not None:
                self._raise_failure(failure)

    def _record(
        self,
        capability: str,
        operation: str,
        units: float,
        request: object,
    ) -> float:
        cost = self.estimate_cost(capability, units)
        self._calls.append(
            MockCall(
                capability=capability,
                operation=operation,
                units=units,
                cost_usd=cost,
                request=request,
            )
        )
        self._total_cost_usd += cost
        return cost

    @staticmethod
    def _response_metadata(cost: float, **extra: object) -> dict[str, Any]:
        return {"mock": True, "cost_usd": cost, **extra}

    def _encode_image(self, png: bytes, image_format: ImageFormat) -> bytes:
        if image_format is ImageFormat.PNG:
            return png
        if self._ffmpeg_runner is None and not shutil.which(self._ffmpeg):
            return png
        suffix = f".{image_format.value}"
        with tempfile.TemporaryDirectory(prefix="aivia-mock-image-") as directory:
            source = Path(directory) / "source.png"
            target = Path(directory) / f"output{suffix}"
            source.write_bytes(png)
            args = [
                self._ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-frames:v",
                "1",
                "-fflags",
                "+bitexact",
                "-flags",
                "+bitexact",
                "-map_metadata",
                "-1",
                "-threads",
                "1",
                str(target),
            ]
            try:
                return self._run_ffmpeg(args, target)
            except subprocess.CalledProcessError:
                return png

    def _encode_audio(
        self,
        wav: bytes,
        audio_format: AudioFormat,
        *,
        sample_rate: int,
    ) -> bytes:
        if audio_format in {AudioFormat.WAV, AudioFormat.PCM}:
            return wav
        if self._ffmpeg_runner is None and not shutil.which(self._ffmpeg):
            if audio_format is AudioFormat.MP3:
                return _mp3_bytes(len(wav) / max(1, sample_rate * 2))
            return wav
        suffix = f".{audio_format.value}"
        with tempfile.TemporaryDirectory(prefix="aivia-mock-audio-") as directory:
            source = Path(directory) / "source.wav"
            target = Path(directory) / f"output{suffix}"
            source.write_bytes(wav)
            args = [
                self._ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-fflags",
                "+bitexact",
                "-flags",
                "+bitexact",
                "-map_metadata",
                "-1",
                "-threads",
                "1",
                str(target),
            ]
            return self._run_ffmpeg(args, target)

    def _run_ffmpeg(self, args: Sequence[str], output: Path) -> bytes:
        if self._ffmpeg_runner is not None:
            return self._ffmpeg_runner(args)
        if shutil.which(self._ffmpeg) is None:
            raise PermanentError("ffmpeg is required for mock media generation")
        subprocess.run(
            list(args),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        return output.read_bytes()

    def _video_bytes(
        self,
        *,
        width: int,
        height: int,
        duration: float,
        fps: int,
        video_format: VideoFormat,
        generate_audio: bool,
        seed: int,
    ) -> bytes:
        width = max(2, width - width % 2)
        height = max(2, height - height % 2)
        if self._ffmpeg_runner is None and not shutil.which(self._ffmpeg):
            header = b"MOCKVIDEO" + struct.pack(
                ">HHdfQ", width, height, duration, fps, seed
            )
            return header + hashlib.sha256(header).digest()
        suffix = f".{video_format.value}"
        with tempfile.TemporaryDirectory(prefix="aivia-mock-video-") as directory:
            target = Path(directory) / f"output{suffix}"
            source = (
                f"testsrc2=size={width}x{height}:rate={fps}:duration={duration:.3f}"
            )
            args = [
                self._ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                source,
            ]
            if generate_audio and video_format is not VideoFormat.GIF:
                args += [
                    "-f",
                    "lavfi",
                    "-i",
                    f"sine=frequency=440:sample_rate=44100:duration={duration:.3f}",
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-shortest",
                ]
            else:
                args += ["-map", "0:v:0"]
            args += [
                "-t",
                f"{duration:.3f}",
                "-r",
                str(fps),
                "-fflags",
                "+bitexact",
                "-flags",
                "+bitexact",
                "-map_metadata",
                "-1",
                "-threads",
                "1",
            ]
            if video_format is VideoFormat.GIF:
                args += ["-c:v", "gif"]
            elif video_format is VideoFormat.WEBM:
                args += [
                    "-c:v",
                    "libvpx-vp9",
                    "-deadline",
                    "realtime",
                    "-cpu-used",
                    "8",
                ]
            else:
                args += ["-c:v", "mpeg4", "-q:v", "5"]
            if generate_audio and video_format is not VideoFormat.GIF:
                args += (
                    ["-c:a", "libopus"]
                    if video_format is VideoFormat.WEBM
                    else ["-c:a", "aac", "-b:a", "96k"]
                )
            args.append(str(target))
            return self._run_ffmpeg(args, target)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self._maybe_fail("complete")
        cost = self._record(CAPABILITY_LLM, "complete", 1.0, request)
        text = request.last_user_text
        content = f"mock: {text}" if text else "mock response"
        tool_calls: list[ToolCall] = []
        finish_reason = FinishReason.STOP
        if request.tool_choice is ToolChoice.REQUIRED and request.tools:
            tool = request.tools[0]
            tool_calls = [
                ToolCall(
                    id=f"call-{_digest(self._seed, text, tool.name)[:12]}",
                    name=tool.name,
                    arguments={},
                )
            ]
            finish_reason = FinishReason.TOOL_CALLS
        input_tokens = sum(len(message.text.split()) for message in request.messages)
        return LLMResponse(
            content=content,
            model=request.model or "mock-llm",
            provider=self.name,
            usage=LLMUsage(
                input_tokens=input_tokens,
                output_tokens=len(content.split()),
            ),
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            request_id=f"mock-{_digest(self._seed, text)[:12]}",
            metadata=self._response_metadata(cost),
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self._maybe_fail("embed")
        cost = self._record(
            CAPABILITY_EMBEDDINGS,
            "embed",
            float(len(request.inputs)),
            request,
        )
        dimensions = request.dimensions or _DEFAULT_EMBEDDING_DIMENSIONS
        embeddings: list[Embedding] = []
        for index, text in enumerate(request.inputs):
            raw = hashlib.sha256(f"{self._seed}:{index}:{text}".encode()).digest()
            values = [
                round((raw[position % len(raw)] / 127.5) - 1.0, 6)
                for position in range(dimensions)
            ]
            if request.normalize:
                norm = math.sqrt(sum(value * value for value in values)) or 1.0
                values = [round(value / norm, 6) for value in values]
            embeddings.append(Embedding(index=index, vector=values))
        return EmbeddingResponse(
            embeddings=embeddings,
            model=request.model or "mock-embedding",
            provider=self.name,
            usage=EmbeddingUsage(
                input_tokens=sum(len(text.split()) for text in request.inputs)
            ),
            request_id=f"mock-{_digest(self._seed, *request.inputs)[:12]}",
            metadata=self._response_metadata(cost),
        )

    async def _generate_image(self, request: ImageRequest) -> ImageResponse:
        self._maybe_fail("generate")
        cost = self._record(
            CAPABILITY_IMAGE_GEN,
            "generate",
            float(request.count),
            request,
        )
        width, height = _aspect_dimensions(
            request.width,
            request.height,
            request.aspect_ratio,
        )
        seed = request.seed if request.seed is not None else self._seed
        images: list[ImageResult] = []
        for index in range(request.count):
            png = _png_bytes(
                width,
                height,
                _stable_seed((request.prompt, index), seed),
            )
            data = await asyncio.to_thread(self._encode_image, png, request.format)
            images.append(
                ImageResult(
                    data=data,
                    format=request.format,
                    width=width,
                    height=height,
                    seed=None if request.seed is None else request.seed + index,
                    revised_prompt=request.prompt,
                )
            )
        return ImageResponse(
            images=images,
            model=request.model or "mock-image",
            provider=self.name,
            request_id=f"mock-{_digest(self._seed, request.prompt)[:12]}",
            metadata=self._response_metadata(cost),
        )

    async def edit(self, request: ImageEditRequest) -> ImageEditResponse:
        self._maybe_fail("edit")
        cost = self._record(
            CAPABILITY_IMAGE_EDIT,
            "edit",
            float(request.count),
            request,
        )
        source_size = _png_dimensions(request.image.data)
        width = request.width
        height = request.height
        if width is None or height is None:
            if source_size is not None:
                width, height = source_size
            else:
                width = height = _DEFAULT_IMAGE_SIZE
        if request.scale is not None:
            width = max(1, round(width * request.scale))
            height = max(1, round(height * request.scale))
        seed = request.seed if request.seed is not None else self._seed
        label = request.prompt or request.operation.value
        image_format = request.format or ImageFormat.PNG
        images: list[ImageResult] = []
        for index in range(request.count):
            png = _png_bytes(
                width,
                height,
                _stable_seed((label, index), seed),
            )
            data = await asyncio.to_thread(self._encode_image, png, image_format)
            images.append(
                ImageResult(
                    data=data,
                    format=image_format,
                    width=width,
                    height=height,
                    seed=None if request.seed is None else request.seed + index,
                )
            )
        return ImageEditResponse(
            images=images,
            operation=request.operation,
            model=request.model or "mock-image-edit",
            provider=self.name,
            request_id=f"mock-{_digest(self._seed, label)[:12]}",
            metadata=self._response_metadata(cost),
        )

    async def _generate_video(self, request: VideoRequest) -> VideoResponse:
        self._maybe_fail("generate")
        cost = self._record(
            CAPABILITY_VIDEO_GEN,
            "generate",
            float(request.count),
            request,
        )
        width, height = _video_dimensions(request)
        duration = request.duration_seconds or _DEFAULT_VIDEO_DURATION
        fps = request.fps or _DEFAULT_VIDEO_FPS
        seed = request.seed if request.seed is not None else self._seed
        videos: list[VideoResult] = []
        for index in range(request.count):
            data = await asyncio.to_thread(
                self._video_bytes,
                width=width,
                height=height,
                duration=duration,
                fps=fps,
                video_format=request.format,
                generate_audio=request.generate_audio,
                seed=_stable_seed((request.prompt, index), seed),
            )
            videos.append(
                VideoResult(
                    data=data,
                    format=request.format,
                    width=width,
                    height=height,
                    duration_seconds=duration,
                    fps=fps,
                    has_audio=request.generate_audio,
                    seed=None if request.seed is None else request.seed + index,
                )
            )
        return VideoResponse(
            videos=videos,
            model=request.model or "mock-video",
            provider=self.name,
            request_id=f"mock-{_digest(self._seed, request.prompt)[:12]}",
            metadata=self._response_metadata(cost),
        )

    async def _generate_music(self, request: MusicRequest) -> MusicResponse:
        self._maybe_fail("generate")
        cost = self._record(
            CAPABILITY_MUSIC_GEN,
            "generate",
            float(request.count),
            request,
        )
        duration = request.duration_seconds or _DEFAULT_MUSIC_DURATION
        sample_rate = request.sample_rate or _DEFAULT_AUDIO_SAMPLE_RATE
        seed = request.seed if request.seed is not None else self._seed
        tracks: list[MusicTrack] = []
        for index in range(request.count):
            track_seed = _stable_seed((request.prompt, index), seed)
            wav = _wav_bytes(duration, sample_rate=sample_rate, seed=track_seed)
            audio = await asyncio.to_thread(
                self._encode_audio,
                wav,
                request.format,
                sample_rate=sample_rate,
            )
            tracks.append(
                MusicTrack(
                    data=audio,
                    format=request.format,
                    duration_seconds=duration,
                    sample_rate=sample_rate,
                    channels=1,
                    seed=None if request.seed is None else request.seed + index,
                    title=request.prompt,
                    lyrics=None if request.instrumental else request.lyrics,
                )
            )
        return MusicResponse(
            tracks=tracks,
            model=request.model or "mock-music",
            provider=self.name,
            request_id=f"mock-{_digest(self._seed, request.prompt)[:12]}",
            metadata=self._response_metadata(cost),
        )

    async def _generate_sfx(self, request: SFXRequest) -> SFXResponse:
        self._maybe_fail("generate")
        cost = self._record(
            CAPABILITY_SFX,
            "generate",
            float(request.count),
            request,
        )
        duration = request.duration_seconds or _DEFAULT_SFX_DURATION
        sample_rate = request.sample_rate or _DEFAULT_AUDIO_SAMPLE_RATE
        seed = request.seed if request.seed is not None else self._seed
        clips: list[SFXClip] = []
        for index in range(request.count):
            clip_seed = _stable_seed((request.prompt, index), seed)
            wav = _wav_bytes(duration, sample_rate=sample_rate, seed=clip_seed)
            audio = await asyncio.to_thread(
                self._encode_audio,
                wav,
                request.format,
                sample_rate=sample_rate,
            )
            clips.append(
                SFXClip(
                    data=audio,
                    format=request.format,
                    duration_seconds=duration,
                    sample_rate=sample_rate,
                    channels=1,
                    loop=request.loop,
                    seed=None if request.seed is None else request.seed + index,
                )
            )
        return SFXResponse(
            clips=clips,
            model=request.model or "mock-sfx",
            provider=self.name,
            request_id=f"mock-{_digest(self._seed, request.prompt)[:12]}",
            metadata=self._response_metadata(cost),
        )

    async def _search_stock(self, request: StockRequest) -> StockResponse:
        self._maybe_fail("search")
        cost = self._record(CAPABILITY_STOCK, "search", 1.0, request)
        assets = [
            StockAsset(
                id=f"mock-{request.kind.value}-{index}",
                provider=self.name,
                kind=request.kind,
                url=f"https://example.com/stock/{request.kind.value}/{index}",
                width=1920,
                height=1080,
                duration_seconds=(
                    5.0 if request.kind is StockMediaKind.VIDEO else None
                ),
                format=(
                    VideoFormat.MP4
                    if request.kind is StockMediaKind.VIDEO
                    else ImageFormat.JPEG
                ),
                creator="Mock Creator",
                license="CC0",
                license_url="https://creativecommons.org/publicdomain/zero/1.0/",
                attribution="Mock Creator / example.com",
            )
            for index in range(1, min(request.per_page, 3) + 1)
        ]
        return StockResponse(
            query=request.query,
            assets=assets,
            provider=self.name,
            total_results=len(assets),
            page=request.page,
            request_id=f"mock-{_digest(self._seed, request.query)[:12]}",
            metadata=self._response_metadata(cost),
        )

    async def synthesize(self, request: TTSRequest) -> TTSResponse:
        self._maybe_fail("synthesize")
        cost = self._record(
            CAPABILITY_TTS,
            "synthesize",
            float(max(1, len(request.text))),
            request,
        )
        words = request.text.split()
        duration = max(len(words), 1) / _WORDS_PER_SECOND
        sample_rate = request.sample_rate or _DEFAULT_TTS_SAMPLE_RATE
        seed = request.seed if request.seed is not None else self._seed
        wav = _wav_bytes(
            duration,
            sample_rate=sample_rate,
            seed=_stable_seed(request.text, seed),
        )
        audio = await asyncio.to_thread(
            self._encode_audio,
            wav,
            request.format,
            sample_rate=sample_rate,
        )
        timestamps: list[WordTimestamp] = []
        cursor = 0.0
        for word in words:
            end = cursor + 1.0 / _WORDS_PER_SECOND
            timestamps.append(
                WordTimestamp(word=word, start_seconds=cursor, end_seconds=end)
            )
            cursor = end
        return TTSResponse(
            audio=audio,
            format=request.format,
            sample_rate=sample_rate,
            duration_seconds=duration,
            voice=request.voice,
            model=request.model or "mock-tts",
            provider=self.name,
            word_timestamps=timestamps if request.word_timestamps else [],
            request_id=f"mock-{_digest(self._seed, request.text)[:12]}",
            metadata=self._response_metadata(cost),
        )

    async def transcribe(self, request: STTRequest) -> STTResponse:
        self._maybe_fail("transcribe")
        cost = self._record(CAPABILITY_STT, "transcribe", 1.0, request)
        if request.audio is not None:
            text = request.prompt or "mock transcript"
            duration = _wav_duration(request.audio) or len(text.split()) / (
                _WORDS_PER_SECOND
            )
        else:
            text = request.prompt or f"mock transcript for {request.audio_url}"
            duration = len(text.split()) / _WORDS_PER_SECOND
        words = text.split()
        timestamps: list[WordTimestamp] = []
        cursor = 0.0
        step = 1.0 / _WORDS_PER_SECOND
        for word in words:
            timestamps.append(
                WordTimestamp(
                    word=word, start_seconds=cursor, end_seconds=cursor + step
                )
            )
            cursor += step
        segments = (
            [
                TranscriptSegment(
                    text=text,
                    start_seconds=0.0,
                    end_seconds=max(cursor, duration),
                    words=timestamps,
                )
            ]
            if text
            else []
        )
        return STTResponse(
            text=text,
            language=request.language or "en",
            duration_seconds=max(cursor, duration),
            segments=segments,
            words=timestamps if request.word_timestamps else [],
            model=request.model or "mock-stt",
            provider=self.name,
            request_id=f"mock-{_digest(self._seed, text)[:12]}",
            metadata=self._response_metadata(cost),
        )

    async def _search_web(self, request: SearchRequest) -> SearchResponse:
        self._maybe_fail("search")
        cost = self._record(CAPABILITY_SEARCH, "search", 1.0, request)
        count = min(request.max_results, 3)
        results = [
            SearchResult(
                title=f"Mock result {index} for {request.query}",
                url=f"https://example.com/search/{index}",
                snippet=f"Deterministic snippet {index} for {request.query}",
                content=f"Deterministic content {index} for {request.query}",
                source="example.com",
                score=round(1.0 - index / 10.0, 2),
            )
            for index in range(1, count + 1)
        ]
        return SearchResponse(
            query=request.query,
            results=results,
            provider=self.name,
            answer=f"Mock answer for {request.query}",
            total_results=len(results),
            took_ms=1.0,
            request_id=f"mock-{_digest(self._seed, request.query)[:12]}",
            metadata=self._response_metadata(cost),
        )

    async def fetch(self, request: PageFetchRequest) -> PageFetchResponse:
        self._maybe_fail("fetch")
        cost = self._record(CAPABILITY_PAGE_FETCH, "fetch", 1.0, request)
        text = f"Mock page content for {request.url}"
        truncated = False
        if request.max_characters is not None and len(text) > request.max_characters:
            text = text[: request.max_characters]
            truncated = True
        return PageFetchResponse(
            url=request.url,
            requested_url=request.url,
            status_code=200,
            content_type="text/html",
            kind=PageKind.HTML,
            title="Mock page",
            text=text,
            links=[request.url],
            fetched_at=MOCK_EPOCH,
            elapsed_ms=1.0,
            size_bytes=len(text.encode("utf-8")),
            truncated=truncated,
            metadata=self._response_metadata(cost),
        )

    async def send(self, message: NotificationMessage) -> NotificationReceipt:
        self._maybe_fail("send")
        cost = self._record(CAPABILITY_NOTIFY, "send", 1.0, message)
        identifier = _digest(self._seed, message.title, message.text)[:12]
        return NotificationReceipt(
            message_id=f"mock-message-{identifier}",
            channel="mock",
            provider=self.name,
            delivered_at=MOCK_EPOCH,
            metadata=self._response_metadata(cost, level=message.level.value),
        )

    async def publish(self, request: PublishRequest) -> PublishResult:
        self._maybe_fail("publish")
        cost = self._record(CAPABILITY_PUBLISH, "publish", 1.0, request)
        identifier = _digest(
            self._seed,
            request.platform,
            request.account,
            request.idempotency_key,
            request.caption,
        )[:12]
        if request.dry_run:
            return PublishResult(
                platform=request.platform,
                account=request.account,
                provider=self.name,
                dry_run=True,
                metadata=self._response_metadata(cost),
            )
        return PublishResult(
            platform=request.platform,
            account=request.account,
            provider=self.name,
            remote_id=f"mock-post-{identifier}",
            url=f"https://example.com/posts/{identifier}",
            published_at=MOCK_EPOCH,
            metadata=self._response_metadata(cost),
        )

    async def deliver(self, request: DeliveryRequest) -> DeliveryReceipt:
        self._maybe_fail("deliver")
        cost = self._record(CAPABILITY_DELIVERY, "deliver", 1.0, request)
        identifier = _digest(
            self._seed,
            request.method.value,
            request.target,
            request.idempotency_key,
        )[:12]
        expires_at = (
            MOCK_EPOCH + timedelta(seconds=request.expires_seconds)
            if request.expires_seconds is not None
            else None
        )
        url = (
            f"https://example.com/deliveries/{identifier}"
            if request.method in {DeliveryMethod.LINK, DeliveryMethod.WEB_CARD}
            else None
        )
        return DeliveryReceipt(
            method=request.method,
            reference=f"mock-delivery-{identifier}",
            target=request.target,
            provider=self.name,
            delivered_at=MOCK_EPOCH,
            url=url,
            expires_at=expires_at,
            metadata=self._response_metadata(cost),
        )

    async def put(self, request: StoragePutRequest) -> StorageObject:
        self._maybe_fail("put")
        cost = self._record(
            CAPABILITY_STORAGE,
            "put",
            float(len(request.data)),
            request,
        )
        digest = hashlib.sha256(request.data).hexdigest()
        key = request.key or f"objects/{digest[:2]}/{digest}"
        existing = self._objects.get(key)
        if existing is not None:
            if existing[0] == request.data:
                return existing[1]
            if not request.overwrite:
                raise InvalidRequestError(f"storage key already exists: {key}")
        stored = StorageObject(
            key=key,
            size_bytes=len(request.data),
            content_type=request.content_type,
            sha256=digest,
            url=self.url(key),
            created_at=MOCK_EPOCH,
            metadata=self._response_metadata(cost),
        )
        self._objects[key] = (request.data, stored)
        return stored

    async def get(self, key: str) -> bytes:
        if key not in self._objects:
            raise KeyError(key)
        return self._objects[key][0]

    async def stat(self, key: str) -> StorageObject | None:
        entry = self._objects.get(key)
        return None if entry is None else entry[1]

    async def delete(self, key: str) -> bool:
        return self._objects.pop(key, None) is not None

    def url(self, key: str, *, expires_seconds: int | None = None) -> str:
        base = f"https://storage.example.com/{key}"
        if expires_seconds is not None:
            return f"{base}?expires={expires_seconds}"
        return base

    async def animate(self, request: LipSyncRequest) -> LipSyncResponse:
        self._maybe_fail("animate")
        cost = self._record(CAPABILITY_LIPSYNC, "animate", 1.0, request)
        duration = (
            request.duration_seconds
            or _wav_duration(request.audio.data)
            or _DEFAULT_SFX_DURATION
        )
        source_size = _png_dimensions(
            request.source.data if request.source is not None else None
        )
        width, height = source_size or _DEFAULT_VIDEO_SIZE
        data = await asyncio.to_thread(
            self._video_bytes,
            width=width,
            height=height,
            duration=duration,
            fps=request.fps,
            video_format=request.format,
            generate_audio=False,
            seed=_stable_seed((request.audio.data, duration), self._seed),
        )
        return LipSyncResponse(
            video=LipSyncResult(
                data=data,
                format=request.format,
                width=width,
                height=height,
                duration_seconds=duration,
                fps=request.fps,
                used_fallback=request.sprite_sheet is not None,
            ),
            model="mock-lipsync",
            provider=self.name,
            request_id=f"mock-{_digest(self._seed, duration)[:12]}",
            metadata=self._response_metadata(cost),
        )

    async def train(
        self,
        request: CharacterTrainerRequest,
    ) -> CharacterTrainingJob:
        self._maybe_fail("train")
        cost = self._record(CAPABILITY_CHARACTER_TRAINER, "train", 1.0, request)
        identifier = _digest(
            self._seed,
            request.character_id,
            *request.trigger_words,
        )[:12]
        job = CharacterTrainingJob(
            job_id=f"mock-train-{identifier}",
            character_id=request.character_id,
            state=TrainingState.READY,
            model_id=f"mock-adapter-{identifier}",
            trigger_words=request.trigger_words,
            cost_usd=cost,
            created_at=MOCK_EPOCH,
            completed_at=MOCK_EPOCH,
            metadata=self._response_metadata(cost),
        )
        self._jobs[job.job_id] = job
        return job

    async def status(self, job_id: str) -> CharacterTrainingJob:
        try:
            return self._jobs[job_id]
        except KeyError:
            raise KeyError(f"unknown mock training job: {job_id}") from None


class MockProvider(_MockProviderBase):
    """One mock provider that declares and serves every capability."""

    mock_name = "mock"
    capabilities = (
        CAPABILITY_LLM,
        CAPABILITY_EMBEDDINGS,
        CAPABILITY_IMAGE_GEN,
        CAPABILITY_IMAGE_EDIT,
        CAPABILITY_VIDEO_GEN,
        CAPABILITY_MUSIC_GEN,
        CAPABILITY_SFX,
        CAPABILITY_STOCK,
        CAPABILITY_TTS,
        CAPABILITY_STT,
        CAPABILITY_SEARCH,
        CAPABILITY_PAGE_FETCH,
        CAPABILITY_NOTIFY,
        CAPABILITY_PUBLISH,
        CAPABILITY_DELIVERY,
        CAPABILITY_STORAGE,
        CAPABILITY_LIPSYNC,
        CAPABILITY_CHARACTER_TRAINER,
    )
    prices: ClassVar[Mapping[str, float]] = {
        CAPABILITY_LLM: 0.0005,
        CAPABILITY_EMBEDDINGS: 0.00002,
        CAPABILITY_IMAGE_GEN: 0.02,
        CAPABILITY_IMAGE_EDIT: 0.03,
        CAPABILITY_VIDEO_GEN: 0.05,
        CAPABILITY_MUSIC_GEN: 0.01,
        CAPABILITY_SFX: 0.002,
        CAPABILITY_STOCK: 0.0,
        CAPABILITY_TTS: 0.00001,
        CAPABILITY_STT: 0.0001,
        CAPABILITY_SEARCH: 0.001,
        CAPABILITY_PAGE_FETCH: 0.0,
        CAPABILITY_NOTIFY: 0.0,
        CAPABILITY_PUBLISH: 0.0,
        CAPABILITY_DELIVERY: 0.0,
        CAPABILITY_STORAGE: 0.0,
        CAPABILITY_LIPSYNC: 0.01,
        CAPABILITY_CHARACTER_TRAINER: 0.5,
    }

    async def generate(
        self,
        request: ImageRequest | VideoRequest | MusicRequest | SFXRequest,
    ) -> ImageResponse | VideoResponse | MusicResponse | SFXResponse:
        if isinstance(request, ImageRequest):
            return await self._generate_image(request)
        if isinstance(request, VideoRequest):
            return await self._generate_video(request)
        if isinstance(request, MusicRequest):
            return await self._generate_music(request)
        if isinstance(request, SFXRequest):
            return await self._generate_sfx(request)
        raise TypeError(f"unsupported generate request: {type(request).__name__}")

    async def search(
        self,
        request: SearchRequest | StockRequest,
    ) -> SearchResponse | StockResponse:
        if isinstance(request, SearchRequest):
            return await self._search_web(request)
        if isinstance(request, StockRequest):
            return await self._search_stock(request)
        raise TypeError(f"unsupported search request: {type(request).__name__}")


class MockLLMProvider(_MockProviderBase):
    """A deterministic LLM mock."""

    mock_name = "mock-llm"
    capabilities = (CAPABILITY_LLM,)
    prices: ClassVar[Mapping[str, float]] = {CAPABILITY_LLM: 0.0005}


class MockEmbeddingProvider(_MockProviderBase):
    """A deterministic embeddings mock."""

    mock_name = "mock-embeddings"
    capabilities = (CAPABILITY_EMBEDDINGS,)
    prices: ClassVar[Mapping[str, float]] = {CAPABILITY_EMBEDDINGS: 0.00002}


class MockImageProvider(_MockProviderBase):
    """A deterministic image generation mock returning valid PNG files."""

    mock_name = "mock-image"
    capabilities = (CAPABILITY_IMAGE_GEN,)
    prices: ClassVar[Mapping[str, float]] = {CAPABILITY_IMAGE_GEN: 0.02}

    async def generate(self, request: ImageRequest) -> ImageResponse:
        return await self._generate_image(request)


class MockImageEditProvider(_MockProviderBase):
    """A deterministic image edit mock returning valid PNG files."""

    mock_name = "mock-image-edit"
    capabilities = (CAPABILITY_IMAGE_EDIT,)
    prices: ClassVar[Mapping[str, float]] = {CAPABILITY_IMAGE_EDIT: 0.03}


class MockVideoProvider(_MockProviderBase):
    """A deterministic video mock rendered with ffmpeg lavfi sources."""

    mock_name = "mock-video"
    capabilities = (CAPABILITY_VIDEO_GEN,)
    prices: ClassVar[Mapping[str, float]] = {CAPABILITY_VIDEO_GEN: 0.05}

    async def generate(self, request: VideoRequest) -> VideoResponse:
        return await self._generate_video(request)


class MockMusicProvider(_MockProviderBase):
    """A deterministic music mock returning sized audio."""

    mock_name = "mock-music"
    capabilities = (CAPABILITY_MUSIC_GEN,)
    prices: ClassVar[Mapping[str, float]] = {CAPABILITY_MUSIC_GEN: 0.01}

    async def generate(self, request: MusicRequest) -> MusicResponse:
        return await self._generate_music(request)


class MockSFXProvider(_MockProviderBase):
    """A deterministic sound effect mock returning sized audio."""

    mock_name = "mock-sfx"
    capabilities = (CAPABILITY_SFX,)
    prices: ClassVar[Mapping[str, float]] = {CAPABILITY_SFX: 0.002}

    async def generate(self, request: SFXRequest) -> SFXResponse:
        return await self._generate_sfx(request)


class MockStockProvider(_MockProviderBase):
    """A deterministic stock search mock returning licensed assets."""

    mock_name = "mock-stock"
    capabilities = (CAPABILITY_STOCK,)
    prices: ClassVar[Mapping[str, float]] = {CAPABILITY_STOCK: 0.0}

    async def search(self, request: StockRequest) -> StockResponse:
        return await self._search_stock(request)


class MockTTSProvider(_MockProviderBase):
    """A deterministic text to speech mock returning sized audio."""

    mock_name = "mock-tts"
    capabilities = (CAPABILITY_TTS,)
    prices: ClassVar[Mapping[str, float]] = {CAPABILITY_TTS: 0.00001}


class MockSTTProvider(_MockProviderBase):
    """A deterministic speech to text mock with word timestamps."""

    mock_name = "mock-stt"
    capabilities = (CAPABILITY_STT,)
    prices: ClassVar[Mapping[str, float]] = {CAPABILITY_STT: 0.0001}


class MockSearchProvider(_MockProviderBase):
    """A deterministic search mock returning synthetic results."""

    mock_name = "mock-search"
    capabilities = (CAPABILITY_SEARCH,)
    prices: ClassVar[Mapping[str, float]] = {CAPABILITY_SEARCH: 0.001}

    async def search(self, request: SearchRequest) -> SearchResponse:
        return await self._search_web(request)


class MockPageFetchProvider(_MockProviderBase):
    """A deterministic page fetch mock returning synthetic pages."""

    mock_name = "mock-page-fetch"
    capabilities = (CAPABILITY_PAGE_FETCH,)
    prices: ClassVar[Mapping[str, float]] = {CAPABILITY_PAGE_FETCH: 0.0}


class MockNotifier(_MockProviderBase):
    """A deterministic notifier mock acknowledging every message."""

    mock_name = "mock-notify"
    capabilities = (CAPABILITY_NOTIFY,)
    prices: ClassVar[Mapping[str, float]] = {CAPABILITY_NOTIFY: 0.0}


class MockPublisher(_MockProviderBase):
    """A deterministic publisher mock honouring dry run."""

    mock_name = "mock-publish"
    capabilities = (CAPABILITY_PUBLISH,)
    prices: ClassVar[Mapping[str, float]] = {CAPABILITY_PUBLISH: 0.0}


class MockDeliveryProvider(_MockProviderBase):
    """A deterministic delivery mock acknowledging every target."""

    mock_name = "mock-delivery"
    capabilities = (CAPABILITY_DELIVERY,)
    prices: ClassVar[Mapping[str, float]] = {CAPABILITY_DELIVERY: 0.0}


class MockStorageProvider(_MockProviderBase):
    """An in-memory, content addressed storage mock."""

    mock_name = "mock-storage"
    capabilities = (CAPABILITY_STORAGE,)
    prices: ClassVar[Mapping[str, float]] = {CAPABILITY_STORAGE: 0.0}


class MockLipSyncProvider(_MockProviderBase):
    """A deterministic lip sync mock using the same video renderer."""

    mock_name = "mock-lipsync"
    capabilities = (CAPABILITY_LIPSYNC,)
    prices: ClassVar[Mapping[str, float]] = {CAPABILITY_LIPSYNC: 0.01}


class MockCharacterTrainer(_MockProviderBase):
    """A deterministic character trainer mock producing a ready adapter."""

    mock_name = "mock-character-trainer"
    capabilities = (CAPABILITY_CHARACTER_TRAINER,)
    prices: ClassVar[Mapping[str, float]] = {CAPABILITY_CHARACTER_TRAINER: 0.5}


MockAll = MockProvider
MockLLM = MockLLMProvider
MockEmbedding = MockEmbeddingProvider
MockEmbeddings = MockEmbeddingProvider
MockImage = MockImageProvider
MockImageEdit = MockImageEditProvider
MockVideo = MockVideoProvider
MockMusic = MockMusicProvider
MockSFX = MockSFXProvider
MockStock = MockStockProvider
MockTTS = MockTTSProvider
MockSTT = MockSTTProvider
MockSearch = MockSearchProvider
MockPageFetch = MockPageFetchProvider
MockNotify = MockNotifier
MockPublish = MockPublisher
MockDelivery = MockDeliveryProvider
MockStorage = MockStorageProvider
MockLipSync = MockLipSyncProvider


_MOCK_PROVIDERS: dict[str, type[_MockProviderBase]] = {
    CAPABILITY_CHARACTER_TRAINER: MockCharacterTrainer,
    CAPABILITY_DELIVERY: MockDeliveryProvider,
    CAPABILITY_EMBEDDINGS: MockEmbeddingProvider,
    CAPABILITY_IMAGE_EDIT: MockImageEditProvider,
    CAPABILITY_IMAGE_GEN: MockImageProvider,
    CAPABILITY_LIPSYNC: MockLipSyncProvider,
    CAPABILITY_LLM: MockLLMProvider,
    CAPABILITY_MUSIC_GEN: MockMusicProvider,
    CAPABILITY_NOTIFY: MockNotifier,
    CAPABILITY_PAGE_FETCH: MockPageFetchProvider,
    CAPABILITY_PUBLISH: MockPublisher,
    CAPABILITY_SEARCH: MockSearchProvider,
    CAPABILITY_SFX: MockSFXProvider,
    CAPABILITY_STOCK: MockStockProvider,
    CAPABILITY_STORAGE: MockStorageProvider,
    CAPABILITY_STT: MockSTTProvider,
    CAPABILITY_TTS: MockTTSProvider,
    CAPABILITY_VIDEO_GEN: MockVideoProvider,
}


def build_mock_providers() -> dict[str, _MockProviderBase]:
    """Return one fresh mock provider per capability."""
    return {capability: provider() for capability, provider in _MOCK_PROVIDERS.items()}


def mock_provider_for(capability: str) -> _MockProviderBase:
    """Return a new mock provider that serves *capability*."""
    try:
        provider = _MOCK_PROVIDERS[capability]
    except KeyError:
        raise KeyError(f"no mock provider for capability: {capability}") from None
    return provider()


def register_mock_providers(target: ProviderRegistry | None = None) -> None:
    """Register the mock types on *target* (the shared registry by default)."""
    from aivia.providers.registry import registry

    destination: ProviderRegistry = registry if target is None else target
    registrations: dict[str, type[Provider]] = {"mock": MockProvider}
    registrations.update(
        {provider.mock_name: provider for provider in _MOCK_PROVIDERS.values()}
    )
    for name, provider in registrations.items():
        if name not in destination:
            destination.register(name, provider, module=__name__)


register_mock_providers()
