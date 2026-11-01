"""The media cache is keyed on the request the provider actually receives."""

from __future__ import annotations

import asyncio
import json
import struct
import subprocess
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine

from aivia.core.config.capability import CapabilityChainConfig
from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.models import ProviderMeta, ProviderPrice, RateLimit
from aivia.core.router import Router
from aivia.media.adaptation import adapt_for_provider
from aivia.media.image_service import ImageService
from aivia.media.music_service import MusicService
from aivia.media.tts_service import SpeechService
from aivia.media.video_service import VideoService
from aivia.providers.media.adapt import AdaptationAction
from aivia.providers.media.dto import MediaKind
from aivia.providers.mock import MockProvider
from aivia.providers.protocols import (
    CAPABILITY_IMAGE_GEN,
    CAPABILITY_MUSIC_GEN,
    CAPABILITY_TTS,
    CAPABILITY_VIDEO_GEN,
    AudioFormat,
    ImageRequest,
    MusicRequest,
    TTSRequest,
    VideoRequest,
)
from aivia.providers.storage_local import LocalStorageProvider

PROMPT = "a red fox in a very long meadow"


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
def storage(tmp_path: Path) -> LocalStorageProvider:
    return LocalStorageProvider(tmp_path / "media")


def _mp4() -> bytes:
    return struct.pack(">I", 24) + b"ftypisom" + b"\x00" * 12


def _probe_json(*, duration: float = 2.0, video: bool = False) -> str:
    if video:
        streams: list[dict[str, Any]] = [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264",
                "width": 8,
                "height": 8,
                "duration": f"{duration:.3f}",
                "avg_frame_rate": "30/1",
            }
        ]
        format_name = "mov,mp4,m4a,3gp,3g2,mj2"
    else:
        streams = [
            {
                "index": 0,
                "codec_type": "audio",
                "codec_name": "pcm_s16le",
                "duration": f"{duration:.3f}",
                "sample_rate": "44100",
                "channels": 1,
            }
        ]
        format_name = "wav"
    return json.dumps(
        {
            "streams": streams,
            "format": {
                "format_name": format_name,
                "format_long_name": format_name,
                "duration": f"{duration:.3f}",
                "size": "12345",
            },
        }
    )


def _probe_runner(*, duration: float = 2.0, video: bool = False) -> Any:
    def runner(argv: Sequence[str]) -> Any:
        return subprocess.CompletedProcess(
            list(argv),
            0,
            stdout=_probe_json(duration=duration, video=video),
            stderr="",
        )

    return runner


def _provider(
    capability: str,
    options: Sequence[str],
    *,
    name: str = "acme",
) -> MockProvider:
    return MockProvider(
        ProviderMeta(
            name=name,
            capabilities=[capability],
            supported_options={capability: list(options)},
            pricing={capability: ProviderPrice(amount_usd=0.0)},
            rate_limit=RateLimit(requests_per_minute=0),
        ),
        ffmpeg_runner=lambda _argv: _mp4(),
    )


def _router(provider: MockProvider) -> Router:
    return Router(
        {provider.name: provider},
        config=CapabilityChainConfig(chain=[provider.name]),
    )


def test_image_repeat_reuses_the_snapped_size_and_clamped_count(
    engine: Engine,
    storage: LocalStorageProvider,
) -> None:
    provider = _provider(CAPABILITY_IMAGE_GEN, ["8x8", "count:2"])
    service = ImageService(
        _router(provider),
        engine=engine,
        storage=storage,
        record_sidecar=False,
    )
    request = ImageRequest(prompt=PROMPT, width=7, height=7, count=3)

    first = _run(service.generate(request, token=service.screen(PROMPT)))
    second = _run(
        service.generate(request, token=service.screen(PROMPT), run_id="run_2")
    )

    assert len(provider.calls) == 1
    sent = provider.calls[0].request
    assert isinstance(sent, ImageRequest)
    assert (sent.width, sent.height, sent.count) == (8, 8, 2)
    assert [warning.action for warning in first.adaptation_warnings] == [
        AdaptationAction.SNAPPED_SIZE,
        AdaptationAction.CLAMPED_COUNT,
    ]
    assert second.cache_hit is True
    assert second.response.count == 2
    assert len(second.assets) == 2
    assert second.event is not None
    assert second.event.amount_usd == 0.0


def test_a_different_size_that_snaps_to_the_same_size_hits_the_cache(
    engine: Engine,
    storage: LocalStorageProvider,
) -> None:
    provider = _provider(CAPABILITY_IMAGE_GEN, ["8x8", "count:2"])
    service = ImageService(
        _router(provider),
        engine=engine,
        storage=storage,
        record_sidecar=False,
    )
    first_request = ImageRequest(prompt=PROMPT, width=7, height=7, count=3)
    second_request = ImageRequest(prompt=PROMPT, width=6, height=6, count=3)

    first = _run(service.generate(first_request, token=service.screen(PROMPT)))
    second = _run(service.generate(second_request, token=service.screen(PROMPT)))

    assert len(provider.calls) == 1
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.response.count == 2
    assert [asset.id for asset in second.assets] == [asset.id for asset in first.assets]

    adapted = adapt_for_provider(first_request, provider, MediaKind.IMAGE).request
    assert service.cache_key(first_request) == service.cache_key(adapted)


def test_video_repeat_reuses_the_clamped_duration_and_count(
    engine: Engine,
    storage: LocalStorageProvider,
) -> None:
    provider = _provider(CAPABILITY_VIDEO_GEN, ["duration:1-2", "count:2"])
    service = VideoService(
        _router(provider),
        engine=engine,
        storage=storage,
        record_sidecar=False,
        ffprobe_runner=_probe_runner(duration=2.0, video=True),
    )
    request = VideoRequest(prompt="a storm", duration_seconds=9.0, count=3)

    first = _run(service.generate(request, token=service.screen(request.prompt)))
    second = _run(
        service.generate(request, token=service.screen(request.prompt), run_id="run_2")
    )

    assert len(provider.calls) == 1
    sent = provider.calls[0].request
    assert isinstance(sent, VideoRequest)
    assert (sent.duration_seconds, sent.count) == (2.0, 2)
    assert [warning.action for warning in first.adaptation_warnings] == [
        AdaptationAction.CLAMPED_DURATION,
        AdaptationAction.CLAMPED_COUNT,
    ]
    assert second.cache_hit is True
    assert second.response.count == 2
    assert second.event is not None
    assert second.event.amount_usd == 0.0


def test_music_repeat_reuses_the_clamped_duration_and_count(
    engine: Engine,
    storage: LocalStorageProvider,
) -> None:
    provider = _provider(CAPABILITY_MUSIC_GEN, ["duration:1-2", "count:2"])
    service = MusicService(
        _router(provider),
        engine=engine,
        storage=storage,
        record_sidecar=False,
        ffprobe_runner=_probe_runner(duration=2.0),
    )
    request = MusicRequest(
        prompt="warm piano",
        duration_seconds=30.0,
        count=3,
        format=AudioFormat.WAV,
    )

    first = _run(service.generate(request, token=service.screen(request.prompt)))
    second = _run(
        service.generate(request, token=service.screen(request.prompt), run_id="run_2")
    )

    assert len(provider.calls) == 1
    sent = provider.calls[0].request
    assert isinstance(sent, MusicRequest)
    assert (sent.duration_seconds, sent.count) == (2.0, 2)
    assert [warning.action for warning in first.adaptation_warnings] == [
        AdaptationAction.CLAMPED_DURATION,
        AdaptationAction.CLAMPED_COUNT,
    ]
    assert second.cache_hit is True
    assert second.response.count == 2
    assert second.event is not None
    assert second.event.amount_usd == 0.0


def test_speech_repeat_reuses_the_truncated_prompt(
    engine: Engine,
    storage: LocalStorageProvider,
) -> None:
    provider = _provider(CAPABILITY_TTS, ["prompt:10"])
    service = SpeechService(
        _router(provider),
        engine=engine,
        storage=storage,
        record_sidecar=False,
        process=False,
        ffprobe_runner=_probe_runner(duration=1.0),
    )
    request = TTSRequest(
        text="one two three four five",
        voice="narrator",
        format=AudioFormat.WAV,
    )

    first = _run(service.synthesize(request, token=service.screen(request.text)))
    second = _run(
        service.synthesize(
            request,
            token=service.screen(request.text),
            run_id="run_2",
        )
    )

    assert len(provider.calls) == 1
    sent = provider.calls[0].request
    assert isinstance(sent, TTSRequest)
    assert sent.text == "one two"
    assert [warning.action for warning in first.adaptation_warnings] == [
        AdaptationAction.TRUNCATED_PROMPT
    ]
    assert second.cache_hit is True
    assert second.event is not None
    assert second.event.amount_usd == 0.0
