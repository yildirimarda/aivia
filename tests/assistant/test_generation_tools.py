"""Generation tools for image, image edit, background removal, upscale, video,
voice and music.

Each tool is exercised through the
:class:`~aivia.assistant.tool_framework.ToolFramework` with a real
:class:`~aivia.media.MediaToolkit` backed by the deterministic mock provider, so
every call runs the screened, cached and cost tracked path without spending
money or touching the network.
"""

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
from urllib.parse import unquote, urlsplit

import pytest
from sqlalchemy import Engine

from aivia.assistant import (
    ToolFramework,
    ToolNotAllowedError,
    ToolUnknownError,
    ToolValidationError,
)
from aivia.assistant.generation_tools import (
    ALL_GENERATION_TOOLS,
    ImageGenerationTool,
    generation_tool_registry,
)
from aivia.core.config.capability import CapabilityChainConfig
from aivia.core.config.planner import PlannerConfig
from aivia.core.cost import CostTracker
from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.router import Router
from aivia.media import MediaToolkit
from aivia.providers.mock import MockProvider
from aivia.providers.protocols.image_edit import ImageEditOperation
from aivia.providers.storage_local import LocalStorageProvider

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
    "z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
)


def _local_path(value: str) -> Path:
    parsed = urlsplit(value)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    return Path(value)


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


@pytest.fixture(autouse=True)
def bind_toolkit(toolkit: MediaToolkit) -> None:
    for tool in ALL_GENERATION_TOOLS:
        tool.bind_toolkit(toolkit)


def _framework() -> ToolFramework:
    return ToolFramework(generation_tool_registry())


def test_all_generation_tools_are_registered() -> None:
    registry = generation_tool_registry()

    assert registry.names() == (
        "image.background_removal",
        "image.edit",
        "image.generate",
        "image.upscale",
        "music.generate",
        "video.generate",
        "voice.generate",
    )
    for tool in ALL_GENERATION_TOOLS:
        assert tool.name in registry
        schema = registry.schemas()[tool.name]
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False


def test_image_generation_schema_requires_prompt() -> None:
    schema = ImageGenerationTool.schema()

    assert schema["required"] == ["prompt"]
    assert schema["properties"]["prompt"]["type"] == "string"


def test_image_generation_returns_paths_and_cost(
    toolkit: MediaToolkit,
    tracker: CostTracker,
) -> None:
    result = _framework().execute(
        "image.generate", {"prompt": "A lighthouse at sunset"}
    )

    output = result.output
    assert output["kind"] == "image"
    assert output["provider"] == "mock"
    assert output["cached"] is False
    assert output["cost_usd"] == pytest.approx(0.02)
    assert result.cost_usd == pytest.approx(0.02)
    assert len(output["paths"]) == 1
    assert _local_path(output["paths"][0]).is_file()
    assert tracker.count() == 1


def test_image_edit_returns_an_image_and_cost(
    toolkit: MediaToolkit,
) -> None:
    result = _framework().execute(
        "image.edit",
        {
            "image": PNG_1X1,
            "operation": ImageEditOperation.UPSCALE.value,
            "scale": 2.0,
        },
    )

    output = result.output
    assert output["kind"] == "edit"
    assert output["provider"] == "mock"
    assert output["cost_usd"] == pytest.approx(0.03)
    assert len(output["paths"]) == 1
    assert _local_path(output["paths"][0]).is_file()


def test_background_removal_returns_an_image(
    toolkit: MediaToolkit,
) -> None:
    result = _framework().execute(
        "image.background_removal",
        {"image": PNG_1X1},
    )

    output = result.output
    assert output["kind"] == "edit"
    assert output["provider"] == "mock"
    assert output["cost_usd"] == pytest.approx(0.03)
    assert len(output["paths"]) == 1
    assert _local_path(output["paths"][0]).is_file()


def test_upscale_returns_a_scaled_image(
    toolkit: MediaToolkit,
) -> None:
    result = _framework().execute(
        "image.upscale",
        {"image": PNG_1X1, "scale": 2.0},
    )

    output = result.output
    assert output["kind"] == "edit"
    assert output["provider"] == "mock"
    assert output["cost_usd"] == pytest.approx(0.03)
    assert len(output["paths"]) == 1
    assert _local_path(output["paths"][0]).is_file()


def test_video_generation_returns_a_clip(
    toolkit: MediaToolkit,
) -> None:
    result = _framework().execute(
        "video.generate",
        {"prompt": "A storm over the sea", "duration_seconds": 1.0},
    )

    output = result.output
    assert output["kind"] == "video"
    assert output["provider"] == "mock"
    assert output["cost_usd"] == pytest.approx(0.05)
    assert output["cached"] is False
    assert len(output["paths"]) == 1
    assert _local_path(output["paths"][0]).suffix == ".mp4"
    assert _local_path(output["paths"][0]).is_file()


def test_voice_generation_returns_audio(
    toolkit: MediaToolkit,
) -> None:
    result = _framework().execute(
        "voice.generate",
        {"text": "A lighthouse in a storm", "voice": "narrator"},
    )

    output = result.output
    assert output["kind"] == "speech"
    assert output["provider"] == "mock"
    assert output["cost_usd"] > 0
    assert len(output["paths"]) == 1
    assert _local_path(output["paths"][0]).is_file()


def test_music_generation_returns_a_track(
    toolkit: MediaToolkit,
) -> None:
    result = _framework().execute(
        "music.generate",
        {
            "prompt": "Warm ambient piano",
            "instrumental": True,
            "duration_seconds": 1.0,
            "format": "wav",
        },
    )

    output = result.output
    assert output["kind"] == "music"
    assert output["provider"] == "mock"
    assert output["cost_usd"] > 0
    assert len(output["paths"]) == 1
    assert _local_path(output["paths"][0]).is_file()


def test_missing_required_argument_raises_before_execution() -> None:
    framework = _framework()

    with pytest.raises(ToolValidationError):
        framework.execute("image.generate", {})

    assert framework.tool_calls == 0


def test_unknown_tool_is_rejected() -> None:
    framework = _framework()

    with pytest.raises(ToolUnknownError):
        framework.execute("image.nonexistent", {})

    assert framework.tool_calls == 0


def test_allowlist_restricts_generation_tools() -> None:
    registry = generation_tool_registry()
    config = PlannerConfig(allowed_tools=["image.generate"])
    framework = ToolFramework(registry, planner_config=config)

    assert framework.permits("image.generate") is True
    assert framework.permits("music.generate") is False

    result = framework.execute("image.generate", {"prompt": "A lighthouse at sunset"})
    assert result.output["kind"] == "image"

    with pytest.raises(ToolNotAllowedError):
        framework.execute("music.generate", {"prompt": "Warm ambient piano"})

    assert framework.tool_calls == 1
