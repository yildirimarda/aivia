"""The toolkit exposes every media capability with a uniform cost result."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import struct
import subprocess
import wave
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine

from aivia.core.config.capability import CapabilityChainConfig
from aivia.core.cost import CostTracker
from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.router import Router
from aivia.media import MediaResult, MediaToolkit
from aivia.providers.mock import MockProvider
from aivia.providers.protocols.image import ImageRequest
from aivia.providers.protocols.image_edit import ImageEditOperation
from aivia.providers.storage_local import LocalStorageProvider

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
    "z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
)


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


def mp4() -> bytes:
    return struct.pack(">I", 24) + b"ftypisom" + b"\x00" * 12


def wav_bytes(duration: float = 1.0) -> bytes:
    sample_rate = 8000
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(b"\x00\x00" * int(sample_rate * duration))
    return buffer.getvalue()


def probe_json(*, video: bool, duration: float = 1.0) -> str:
    if video:
        streams: list[dict[str, Any]] = [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264",
                "width": 512,
                "height": 512,
                "duration": f"{duration:.3f}",
                "avg_frame_rate": "24/1",
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
                "sample_rate": "8000",
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


def probe_runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    target = Path(argv[-1])
    video = target.suffix.lower() in {".mp4", ".mov", ".webm", ".mkv", ".gif"}
    return subprocess.CompletedProcess(
        list(argv),
        0,
        stdout=probe_json(video=video),
        stderr="",
    )


@pytest.fixture
def provider() -> MockProvider:
    return MockProvider(name="mock", ffmpeg_runner=lambda _argv: mp4())


@pytest.fixture
def toolkit(
    provider: MockProvider,
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> MediaToolkit:
    router = Router(
        {provider.name: provider},
        config=CapabilityChainConfig(chain=[provider.name]),
    )
    return MediaToolkit(
        router,
        tracker=tracker,
        storage=storage,
        ffprobe_runner=probe_runner,
    )


def test_image_returns_paths_and_cost(
    toolkit: MediaToolkit,
    provider: MockProvider,
    tracker: CostTracker,
) -> None:
    result = _run(toolkit.image("A lighthouse at sunset"))

    assert isinstance(result, MediaResult)
    assert result.kind == "image"
    assert result.provider == "mock"
    assert result.cached is False
    assert result.cost_usd == pytest.approx(0.02)
    assert result.amount_usd == result.cost_usd
    assert result.event is not None
    assert len(result.paths) == 1
    assert result.local_paths[0].is_file()
    assert result.to_dict()["cost_usd"] == pytest.approx(0.02)
    assert provider.total_cost_usd == pytest.approx(0.02)
    assert tracker.count() == 1


def test_video_returns_a_clip_and_cost(toolkit: MediaToolkit) -> None:
    result = _run(toolkit.video("A storm over the sea", duration_seconds=1.0))

    assert result.kind == "video"
    assert result.cost_usd == pytest.approx(0.05)
    assert result.local_paths[0].suffix == ".mp4"
    assert result.local_paths[0].is_file()
    assert result.cached is False


def test_speech_returns_audio_and_cost(toolkit: MediaToolkit) -> None:
    result = _run(
        toolkit.speech("A lighthouse in a storm", voice="narrator", format="wav")
    )

    assert result.kind == "speech"
    assert result.audio is not None
    assert result.audio.startswith(b"RIFF")
    assert result.cost_usd > 0
    assert result.local_paths[0].is_file()
    assert result.event is not None


def test_transcription_returns_text_and_cost(toolkit: MediaToolkit) -> None:
    result = _run(toolkit.transcribe(wav_bytes(), language="en", word_timestamps=True))

    assert result.kind == "transcription"
    assert result.text
    assert result.cost_usd > 0
    assert result.event is not None
    assert result.local_paths[0].is_file()


def test_music_returns_tracks_and_cost(toolkit: MediaToolkit) -> None:
    result = _run(
        toolkit.music(
            "Warm ambient piano",
            instrumental=True,
            duration_seconds=1.0,
            format="wav",
        )
    )

    assert result.kind == "music"
    assert len(result.tracks) == 1
    assert result.cost_usd > 0
    assert result.local_paths[0].is_file()
    assert result.license is not None


def test_edit_returns_an_image_and_cost(toolkit: MediaToolkit) -> None:
    result = _run(
        toolkit.edit(
            PNG_1X1,
            operation=ImageEditOperation.UPSCALE,
            scale=2.0,
        )
    )

    assert result.kind == "edit"
    assert result.cost_usd == pytest.approx(0.03)
    assert result.local_paths[0].is_file()
    assert result.images


def test_a_second_identical_image_call_is_cached_and_free(
    toolkit: MediaToolkit,
    provider: MockProvider,
    tracker: CostTracker,
) -> None:
    first = _run(toolkit.image("A lighthouse at sunset"))
    second = _run(toolkit.image("A lighthouse at sunset"))

    assert first.cached is False
    assert second.cached is True
    assert second.cost_usd == pytest.approx(0.0)
    assert second.paths == first.paths
    assert len(provider.calls) == 1
    assert tracker.count() == 2


def test_typed_requests_and_explicit_tokens_are_accepted(
    toolkit: MediaToolkit,
) -> None:
    request = ImageRequest(prompt="A quiet harbour", count=2)
    token = toolkit.image_service.screen(request.prompt)

    result = _run(toolkit.image(request, token=token))

    assert result.kind == "image"
    assert len(result.paths) == 2
    assert result.cost_usd == pytest.approx(0.04)


def test_sync_alias_returns_the_same_result_shape(toolkit: MediaToolkit) -> None:
    result = toolkit.image_sync("A quiet harbour")

    assert result.kind == "image"
    assert result.cost_usd > 0


def test_from_config_builds_a_usable_toolkit(
    engine: Engine,
    tmp_path: Path,
) -> None:
    config = {
        "providers": {"mock": {"type": "mock"}},
        "storage": {
            "backend": "local",
            "local_dir": str(tmp_path / "configured-media"),
        },
    }

    with MediaToolkit.from_config(
        config,
        data_dir=tmp_path / "data",
        engine=engine,
        ffprobe_runner=probe_runner,
    ) as toolkit:
        result = _run(toolkit.image("A lighthouse at sunset"))

    assert result.kind == "image"
    assert result.cost_usd > 0
    assert result.local_paths[0].is_file()
