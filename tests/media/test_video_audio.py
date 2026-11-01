"""Unwanted returned audio is stripped before a clip is stored and recorded."""

from __future__ import annotations

import asyncio
import hashlib
import json
import struct
import subprocess
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import pytest
from sqlalchemy import Engine

from aivia.core.config.capability import CapabilityChainConfig
from aivia.core.cost import CostTracker
from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.errors import ValidationError
from aivia.core.models import ProviderMeta, ProviderPrice, RateLimit
from aivia.core.router import Router
from aivia.media.screening import screen_prompt
from aivia.media.video_cache import VideoCache
from aivia.media.video_service import VideoService
from aivia.providers import Provider
from aivia.providers.protocols import (
    CAPABILITY_VIDEO_GEN,
    VideoFormat,
    VideoRequest,
    VideoResponse,
    VideoResult,
)
from aivia.providers.storage_local import LocalStorageProvider

PROMPT = "A lighthouse in a storm"
PRICE = 0.05


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


@pytest.fixture
def engine() -> Iterator[Engine]:
    created = create_database_engine(url="sqlite+pysqlite:///:memory:")
    upgrade_database(engine=created)
    try:
        yield created
    finally:
        created.dispose()


@pytest.fixture
def tracker(engine: Engine) -> CostTracker:
    return CostTracker(engine)


@pytest.fixture
def storage(tmp_path: Path) -> LocalStorageProvider:
    return LocalStorageProvider(tmp_path / "media")


def mp4(padding: int = 12) -> bytes:
    return struct.pack(">I", 24) + b"ftypisom" + b"\x00" * padding


def probe_json(*, audio: bool) -> str:
    streams: list[dict[str, Any]] = [
        {
            "index": 0,
            "codec_type": "video",
            "codec_name": "h264",
            "width": 1280,
            "height": 720,
            "duration": "5.0",
            "avg_frame_rate": "30/1",
        }
    ]
    if audio:
        streams.append(
            {
                "index": 1,
                "codec_type": "audio",
                "codec_name": "aac",
                "duration": "5.0",
                "sample_rate": "48000",
                "channels": 2,
            }
        )
    return json.dumps(
        {
            "streams": streams,
            "format": {
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                "format_long_name": "QuickTime / MOV",
                "duration": "5.0",
                "size": "12345",
            },
        }
    )


def _probe_runner(
    *,
    audio_sequence: Sequence[bool],
    calls: list[list[str]] | None = None,
) -> Any:
    remaining = list(audio_sequence)

    def runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        if calls is not None:
            calls.append(list(argv))
        audio = remaining.pop(0) if remaining else False
        return subprocess.CompletedProcess(
            list(argv),
            0,
            stdout=probe_json(audio=audio),
            stderr="",
        )

    return runner


def _ffmpeg_writer(
    *,
    payload: bytes,
    calls: list[list[str]] | None = None,
) -> Any:
    def runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        if calls is not None:
            calls.append(list(argv))
        Path(argv[-1]).write_bytes(payload)
        return subprocess.CompletedProcess(list(argv), 0, stdout="", stderr="")

    return runner


def _failing_ffmpeg(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(list(argv), 1, stdout="", stderr="boom")


def _path(asset_path: str) -> Path:
    parsed = urlsplit(asset_path)
    assert parsed.scheme == "file"
    return Path(unquote(parsed.path))


class AudioVideoProvider(Provider):
    """A provider returning one inline clip that reports an audio track."""

    def __init__(self, *, has_audio: bool = True) -> None:
        super().__init__(
            ProviderMeta(
                name="acme",
                capabilities=[CAPABILITY_VIDEO_GEN],
                pricing={
                    CAPABILITY_VIDEO_GEN: ProviderPrice(
                        amount_usd=PRICE,
                        unit="clip",
                    )
                },
                rate_limit=RateLimit(requests_per_minute=0),
            )
        )
        self.has_audio = has_audio
        self.calls: list[VideoRequest] = []

    async def generate(self, request: VideoRequest) -> VideoResponse:
        self.calls.append(request)
        return VideoResponse(
            videos=[
                VideoResult(
                    data=mp4(),
                    format=VideoFormat.MP4,
                    fps=request.fps,
                    has_audio=self.has_audio,
                )
                for _ in range(request.count)
            ],
            model="video-1",
            provider=self.name,
        )


def _router(*providers: Provider) -> Router:
    return Router(
        {provider.name: provider for provider in providers},
        config=CapabilityChainConfig(chain=[provider.name for provider in providers]),
    )


def _token(prompt: str = PROMPT) -> str:
    return str(screen_prompt(prompt))


def _service(
    provider: Provider,
    tracker: CostTracker,
    storage: LocalStorageProvider,
    engine: Engine,
    *,
    ffprobe_runner: Any,
    ffmpeg_runner: Any,
) -> VideoService:
    cache = VideoCache(
        engine=engine,
        storage=storage,
        ffprobe_runner=ffprobe_runner,
        ffmpeg_runner=ffmpeg_runner,
    )
    return VideoService(
        _router(provider),
        tracker=tracker,
        cache=cache,
        storage=storage,
    )


def test_unwanted_audio_is_stripped_before_storage_and_recorded(
    tracker: CostTracker,
    storage: LocalStorageProvider,
    engine: Engine,
) -> None:
    provider = AudioVideoProvider(has_audio=True)
    silent = mp4(padding=20)
    ffmpeg_calls: list[list[str]] = []
    probe_calls: list[list[str]] = []
    service = _service(
        provider,
        tracker,
        storage,
        engine,
        ffprobe_runner=_probe_runner(
            audio_sequence=(True, False, False),
            calls=probe_calls,
        ),
        ffmpeg_runner=_ffmpeg_writer(payload=silent, calls=ffmpeg_calls),
    )
    request = VideoRequest(
        prompt=PROMPT,
        duration_seconds=5.0,
        width=1280,
        height=720,
        fps=30,
        generate_audio=False,
    )

    outcome = _run(service.generate(request, token=_token(), run_id="run_1"))

    assert len(provider.calls) == 1
    assert len(ffmpeg_calls) == 1
    assert "-an" in ffmpeg_calls[0]
    assert outcome.audio_stripped is True
    assert outcome.stripped_audio is True
    assert outcome.stored[0].audio_stripped is True
    assert outcome.videos[0].has_audio is False
    assert outcome.videos[0].data == silent
    assert outcome.asset is not None
    assert outcome.asset.analysis.has_audio is False
    assert outcome.asset.analysis.audio_stripped is True
    assert outcome.asset.content_hash == hashlib.sha256(silent).hexdigest()
    assert _path(outcome.asset.path).read_bytes() == silent
    assert outcome.event is not None
    assert outcome.event.metadata["audio_stripped"] is True
    assert tracker.total() == pytest.approx(PRICE)

    cached = _run(service.generate(request, token=_token(), run_id="run_2"))

    assert cached.cache_hit is True
    assert cached.audio_stripped is True
    assert cached.stored[0].audio_stripped is True
    assert cached.videos[0].has_audio is False
    assert cached.videos[0].data == silent
    assert len(ffmpeg_calls) == 1
    assert len(provider.calls) == 1


def test_requested_audio_is_preserved_without_running_ffmpeg(
    tracker: CostTracker,
    storage: LocalStorageProvider,
    engine: Engine,
) -> None:
    provider = AudioVideoProvider(has_audio=True)
    ffmpeg_calls: list[list[str]] = []
    service = _service(
        provider,
        tracker,
        storage,
        engine,
        ffprobe_runner=_probe_runner(audio_sequence=(True,)),
        ffmpeg_runner=_ffmpeg_writer(payload=mp4(padding=20), calls=ffmpeg_calls),
    )
    request = VideoRequest(
        prompt=PROMPT,
        duration_seconds=5.0,
        fps=30,
        generate_audio=True,
    )

    outcome = _run(service.generate(request, token=_token()))

    assert ffmpeg_calls == []
    assert outcome.audio_stripped is False
    assert outcome.videos[0].has_audio is True
    assert outcome.videos[0].data == mp4()
    assert outcome.asset is not None
    assert outcome.asset.analysis.has_audio is True
    assert outcome.asset.analysis.audio_stripped is False
    assert _path(outcome.asset.path).read_bytes() == mp4()


def test_a_clip_that_already_has_no_audio_is_left_untouched(
    tracker: CostTracker,
    storage: LocalStorageProvider,
    engine: Engine,
) -> None:
    provider = AudioVideoProvider(has_audio=False)
    ffmpeg_calls: list[list[str]] = []
    service = _service(
        provider,
        tracker,
        storage,
        engine,
        ffprobe_runner=_probe_runner(audio_sequence=(False,)),
        ffmpeg_runner=_ffmpeg_writer(payload=mp4(padding=20), calls=ffmpeg_calls),
    )

    outcome = _run(
        service.generate(
            VideoRequest(prompt=PROMPT, duration_seconds=5.0),
            token=_token(),
        )
    )

    assert ffmpeg_calls == []
    assert outcome.audio_stripped is False
    assert outcome.videos[0].has_audio is False
    assert outcome.videos[0].data == mp4()
    assert _path(outcome.asset.path).read_bytes() == mp4()


def test_the_cache_applies_the_audio_policy_before_storing(
    tracker: CostTracker,
    storage: LocalStorageProvider,
    engine: Engine,
) -> None:
    silent = mp4(padding=20)
    ffmpeg_calls: list[list[str]] = []
    cache = VideoCache(
        engine=engine,
        storage=storage,
        ffprobe_runner=_probe_runner(audio_sequence=(True, False, False)),
        ffmpeg_runner=_ffmpeg_writer(payload=silent, calls=ffmpeg_calls),
    )
    request = VideoRequest(prompt=PROMPT, duration_seconds=5.0)

    outcome = _run(
        cache.store(
            request,
            VideoResult(data=mp4(), format=VideoFormat.MP4, has_audio=True),
            provider="acme",
            model="video-1",
        )
    )

    assert outcome.audio_stripped is True
    assert outcome.videos[0].has_audio is False
    assert outcome.videos[0].data == silent
    assert outcome.asset.analysis.audio_stripped is True
    assert _path(outcome.asset.path).read_bytes() == silent
    assert len(ffmpeg_calls) == 1

    hit = _run(cache.lookup(request))

    assert hit is not None
    assert hit.audio_stripped is True
    assert hit.result.data == silent


def test_a_failed_strip_is_rejected_before_storage_cache_or_cost(
    tracker: CostTracker,
    storage: LocalStorageProvider,
    engine: Engine,
) -> None:
    provider = AudioVideoProvider(has_audio=True)
    service = _service(
        provider,
        tracker,
        storage,
        engine,
        ffprobe_runner=_probe_runner(audio_sequence=(True, False)),
        ffmpeg_runner=_failing_ffmpeg,
    )

    with pytest.raises(ValidationError, match="ffmpeg rejected"):
        _run(
            service.generate(
                VideoRequest(prompt=PROMPT, duration_seconds=5.0),
                token=_token(),
            )
        )

    assert tracker.count() == 0
    assert service.cache.cache is not None
    assert service.cache.cache.repository.list() == []
    assert list(storage.root.rglob("*")) == []
