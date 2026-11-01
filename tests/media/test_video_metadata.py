"""Validated video metadata survives storage and a cache hit."""

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

from aivia.core.db import create_database_engine, upgrade_database
from aivia.media.video_cache import VideoCache
from aivia.providers.protocols.media import VideoFormat
from aivia.providers.protocols.video import VideoRequest, VideoResult
from aivia.providers.storage_local import LocalStorageProvider


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


def mp4() -> bytes:
    return struct.pack(">I", 24) + b"ftypisom" + b"\x00" * 12


def probe_json(
    *,
    width: int = 640,
    height: int = 360,
    duration: str = "2.5",
    frame_rate: str = "24/1",
    audio: bool = False,
) -> str:
    streams: list[dict[str, Any]] = [
        {
            "index": 0,
            "codec_type": "video",
            "codec_name": "h264",
            "width": width,
            "height": height,
            "duration": duration,
            "avg_frame_rate": frame_rate,
        }
    ]
    if audio:
        streams.append(
            {
                "index": 1,
                "codec_type": "audio",
                "codec_name": "aac",
                "duration": duration,
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
                "duration": duration,
                "size": "12345",
            },
        }
    )


def runner_for(
    stdout: str,
    *,
    calls: list[list[str]] | None = None,
) -> Any:
    def runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        if calls is not None:
            calls.append(list(argv))
        return subprocess.CompletedProcess(
            list(argv),
            0,
            stdout=stdout,
            stderr="",
        )

    return runner


def _request(
    *,
    fps: int | None = 30,
    generate_audio: bool = True,
) -> VideoRequest:
    return VideoRequest(
        prompt="A lighthouse in a storm",
        duration_seconds=2.5,
        width=640,
        height=360,
        fps=fps,
        generate_audio=generate_audio,
    )


def _result(
    *,
    fps: int | None = 30,
    has_audio: bool = True,
) -> VideoResult:
    return VideoResult(
        data=mp4(),
        format=VideoFormat.MP4,
        fps=fps,
        has_audio=has_audio,
    )


def test_cache_hit_rebuilds_the_validated_fps_format_and_audio_presence(
    engine: Engine,
    storage: LocalStorageProvider,
) -> None:
    probe_calls: list[list[str]] = []
    cache = VideoCache(
        engine=engine,
        storage=storage,
        ffprobe_runner=runner_for(
            probe_json(frame_rate="24/1", audio=False),
            calls=probe_calls,
        ),
    )
    request = _request(fps=30, generate_audio=True)

    first = _run(
        cache.store(
            request,
            _result(fps=30, has_audio=True),
            provider="acme",
            model="video-1",
            run_id="run_1",
        )
    )

    assert first.result.fps == 24
    assert first.result.format is VideoFormat.MP4
    assert first.result.has_audio is False
    assert first.asset.analysis.fps == 24
    assert first.asset.analysis.format == "mp4"
    assert first.asset.analysis.has_audio is False
    assert first.asset.fps == 24
    assert first.asset.format == "mp4"
    assert first.asset.has_audio is False

    second = _run(cache.lookup(request, run_id="run_2"))

    assert second is not None
    assert second.cache_hit is True
    assert second.result.fps == first.result.fps == 24
    assert second.result.format is first.result.format
    assert second.result.has_audio is first.result.has_audio is False
    assert second.result.width == first.result.width == 640
    assert second.result.height == first.result.height == 360
    assert second.result.duration_seconds == first.result.duration_seconds == 2.5
    assert len(probe_calls) == 1


def test_cache_hit_uses_measured_audio_presence_not_the_request(
    engine: Engine,
    storage: LocalStorageProvider,
) -> None:
    cache = VideoCache(
        engine=engine,
        storage=storage,
        ffprobe_runner=runner_for(probe_json(audio=True)),
    )
    request = _request(fps=30, generate_audio=False)

    first = _run(cache.store(request, _result(fps=30, has_audio=False)))
    second = _run(cache.lookup(request))

    assert first.asset.analysis.has_audio is True
    assert first.result.has_audio is True
    assert second is not None
    assert second.result.has_audio is True


def test_fractional_ffprobe_frame_rate_is_persisted_as_an_integer(
    engine: Engine,
    storage: LocalStorageProvider,
) -> None:
    cache = VideoCache(
        engine=engine,
        storage=storage,
        ffprobe_runner=runner_for(probe_json(frame_rate="30000/1001")),
    )
    request = _request(fps=None)

    first = _run(cache.store(request, _result(fps=None)))
    second = _run(cache.lookup(request))

    assert first.asset.analysis.fps == 30
    assert first.result.fps == 30
    assert second is not None
    assert second.result.fps == 30
