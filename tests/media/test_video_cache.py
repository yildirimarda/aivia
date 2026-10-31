"""Downloaded video is probed, stored as an asset and reused from cache."""

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

import httpx
import pytest
from sqlalchemy import Engine

from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.errors import ValidationError
from aivia.core.media_probe import MediaDownloader
from aivia.core.models import AssetKind
from aivia.media.video_cache import (
    VIDEO_OPERATION,
    VideoCache,
    video_cache_key,
)
from aivia.providers.protocols.media import MediaInput, VideoFormat
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
    width: int = 1280,
    height: int = 720,
    duration: str = "5.0",
    audio: bool = True,
) -> str:
    streams: list[dict[str, Any]] = [
        {
            "index": 0,
            "codec_type": "video",
            "codec_name": "h264",
            "width": width,
            "height": height,
            "duration": duration,
            "avg_frame_rate": "30/1",
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


def _path(asset_path: str) -> Path:
    parsed = urlsplit(asset_path)
    assert parsed.scheme == "file"
    return Path(unquote(parsed.path))


def test_a_downloaded_video_is_probed_stored_and_served_from_cache(
    engine: Engine,
    storage: LocalStorageProvider,
) -> None:
    payload = mp4()
    downloads: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        downloads.append(str(request.url))
        return httpx.Response(
            200,
            headers={"content-type": "video/mp4"},
            content=payload,
        )

    probe_calls: list[list[str]] = []
    downloader = MediaDownloader(
        transport=httpx.MockTransport(handler),
        ffprobe_runner=runner_for(probe_json(), calls=probe_calls),
    )
    cache = VideoCache(engine=engine, storage=storage, downloader=downloader)
    request = VideoRequest(
        prompt="A lighthouse in a storm",
        duration_seconds=5.0,
        width=1280,
        height=720,
        fps=30,
        generate_audio=True,
    )
    result = VideoResult(
        url="https://cdn.example.com/clip.mp4",
        format=VideoFormat.MP4,
        fps=30,
    )

    first = _run(
        cache.store(
            request,
            result,
            provider="acme",
            model="video-1",
            run_id="run_1",
        )
    )

    assert len(downloads) == 1
    assert len(probe_calls) == 1
    assert first.cache_hit is False
    assert first.asset.kind is AssetKind.VIDEO
    assert first.asset.provider == "acme"
    assert first.asset.model == "video-1"
    assert first.asset.prompt == request.prompt
    assert first.asset.run_id == "run_1"
    assert first.asset.pinned is True
    assert first.asset.content_hash == hashlib.sha256(payload).hexdigest()
    assert first.asset.analysis.width == 1280
    assert first.asset.analysis.height == 720
    assert first.asset.analysis.duration_seconds == 5.0
    assert first.result.data == payload
    assert first.result.width == 1280
    assert first.result.height == 720
    assert first.result.duration_seconds == 5.0
    assert first.result.has_audio is True
    assert first.result.format is VideoFormat.MP4
    assert first.probe is not None
    assert first.storage_object is not None

    stored_path = _path(first.asset.path)
    assert stored_path.is_file()
    assert stored_path.read_bytes() == payload
    assert stored_path == _path(first.storage_object.url or "")

    assert cache.cache is not None
    entries = cache.cache.repository.list()
    assert len(entries) == 1
    assert entries[0].provider == ""
    assert entries[0].model == ""
    assert entries[0].prompt_hash == first.asset.prompt_hash

    second = _run(cache.lookup(request, run_id="run_2"))

    assert second is not None
    assert second.cache_hit is True
    assert second.asset.id == first.asset.id
    assert second.result.data == payload
    assert second.result.width == 1280
    assert second.result.height == 720
    assert second.result.duration_seconds == 5.0
    assert second.result.has_audio is True
    assert len(downloads) == 1
    assert len(probe_calls) == 1
    assert second.event is not None
    assert second.event.amount_usd == 0.0
    assert second.event.operation == VIDEO_OPERATION
    assert second.event.run_id == "run_2"
    assert second.event.metadata["cache"] == "hit"
    entry = cache.cache.entry(cache.key(request))
    assert entry is not None
    assert entry.hit_count == 1


def test_inline_video_bytes_are_probed_and_stored(
    engine: Engine,
    storage: LocalStorageProvider,
) -> None:
    probe_calls: list[list[str]] = []
    cache = VideoCache(
        engine=engine,
        storage=storage,
        ffprobe_runner=runner_for(
            probe_json(width=640, height=360, duration="2.5"),
            calls=probe_calls,
        ),
    )
    request = VideoRequest(
        prompt="A paper boat",
        duration_seconds=2.5,
        width=640,
        height=360,
    )
    result = VideoResult(
        data=mp4(),
        format=VideoFormat.MP4,
        width=640,
        height=360,
        duration_seconds=2.5,
    )

    outcome = _run(cache.store(request, result, provider="acme", model="video-1"))

    assert probe_calls
    assert probe_calls[0][-1].endswith(".mp4")
    assert outcome.asset.kind is AssetKind.VIDEO
    assert outcome.asset.analysis.width == 640
    assert outcome.asset.analysis.height == 360
    assert outcome.asset.analysis.duration_seconds == 2.5
    assert _path(outcome.asset.path).read_bytes() == mp4()

    hit = _run(cache.lookup(request))

    assert hit is not None
    assert hit.cache_hit is True
    assert hit.asset.id == outcome.asset.id
    assert hit.result.data == mp4()


def test_an_invalid_download_is_rejected_before_storage_or_cache(
    engine: Engine,
    storage: LocalStorageProvider,
) -> None:
    audio_only = json.dumps(
        {
            "streams": [{"index": 0, "codec_type": "audio", "codec_name": "aac"}],
            "format": {"format_name": "mp4", "duration": "1.0"},
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=mp4())

    downloader = MediaDownloader(
        transport=httpx.MockTransport(handler),
        ffprobe_runner=runner_for(audio_only),
    )
    cache = VideoCache(engine=engine, storage=storage, downloader=downloader)
    request = VideoRequest(prompt="A broken clip", duration_seconds=1.0)
    result = VideoResult(
        url="https://cdn.example.com/broken.mp4",
        format=VideoFormat.MP4,
    )

    with pytest.raises(ValidationError, match="no video stream"):
        _run(cache.store(request, result, provider="acme", model="video-1"))

    assert cache.cache is not None
    assert cache.cache.repository.list() == []
    assert list(storage.root.rglob("*")) == []


def test_no_cache_never_serves_or_records_an_entry(
    engine: Engine,
    storage: LocalStorageProvider,
) -> None:
    cache = VideoCache(
        engine=engine,
        storage=storage,
        no_cache=True,
        ffprobe_runner=runner_for(probe_json()),
    )
    request = VideoRequest(prompt="A fresh clip", duration_seconds=5.0)
    result = VideoResult(data=mp4(), format=VideoFormat.MP4)

    first = _run(cache.store(request, result, provider="acme", model="video-1"))

    assert first.asset.kind is AssetKind.VIDEO
    assert cache.cache is not None
    assert cache.cache.repository.list() == []
    assert _run(cache.lookup(request)) is None


def test_a_cached_payload_with_a_different_digest_is_rejected(
    engine: Engine,
    storage: LocalStorageProvider,
) -> None:
    cache = VideoCache(
        engine=engine,
        storage=storage,
        ffprobe_runner=runner_for(probe_json()),
    )
    request = VideoRequest(prompt="A tampered clip", duration_seconds=5.0)
    result = VideoResult(data=mp4(), format=VideoFormat.MP4)

    outcome = _run(cache.store(request, result, provider="acme", model="video-1"))
    path = _path(outcome.asset.path)
    path.write_bytes(mp4() + b"tampered")

    with pytest.raises(ValidationError, match="does not match its recorded hash"):
        _run(cache.lookup(request))


def test_the_cache_key_separates_every_generating_option() -> None:
    request = VideoRequest(prompt="A city at night", duration_seconds=4.0)
    base = video_cache_key(request)

    assert video_cache_key(request.model_copy(update={"duration_seconds": 5.0})) != base
    assert video_cache_key(request.model_copy(update={"fps": 24})) != base
    assert video_cache_key(request.model_copy(update={"generate_audio": True})) != base
    assert video_cache_key(request.model_copy(update={"seed": 7})) != base
    assert video_cache_key(request, index=1) != base

    first_frame = request.model_copy(
        update={"first_frame": MediaInput(data=b"first frame")}
    )
    last_frame = request.model_copy(
        update={"last_frame": MediaInput(data=b"first frame")}
    )
    assert video_cache_key(first_frame) != base
    assert video_cache_key(last_frame) != base
    assert video_cache_key(first_frame) != video_cache_key(last_frame)

    assert (
        video_cache_key(request.model_copy(update={"metadata": {"note": "x"}})) == base
    )
