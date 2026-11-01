"""Every media service adapts its request to the selected provider's limits."""

from __future__ import annotations

import asyncio
import json
import struct
import subprocess
from collections.abc import Sequence
from typing import Any

import pytest

from aivia.core.config.capability import CapabilityChainConfig
from aivia.core.models import ProviderMeta, ProviderPrice, RateLimit
from aivia.core.router import Router
from aivia.media.image_service import ImageService
from aivia.media.music_service import MusicService
from aivia.media.tts_service import SpeechService
from aivia.media.video_service import VideoService
from aivia.providers.media.adapt import (
    AdaptationAction,
    UnsupportedOptionError,
)
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


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


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


def test_image_request_is_adapted_before_the_provider_call() -> None:
    provider = _provider(
        CAPABILITY_IMAGE_GEN,
        ["8x8", "4x4", "count:2", "prompt:20"],
    )
    service = ImageService(_router(provider), record_sidecar=False)
    request = ImageRequest(
        prompt="a red fox in a very long meadow",
        width=7,
        height=7,
        count=3,
    )

    outcome = _run(service.generate(request, token=service.screen(request.prompt)))

    sent = provider.calls[-1].request
    assert isinstance(sent, ImageRequest)
    assert (sent.width, sent.height) == (8, 8)
    assert sent.count == 2
    assert sent.prompt == "a red fox in a very"
    assert [warning.action for warning in outcome.adaptation_warnings] == [
        AdaptationAction.SNAPPED_SIZE,
        AdaptationAction.TRUNCATED_PROMPT,
        AdaptationAction.CLAMPED_COUNT,
    ]
    assert outcome.warnings == outcome.adaptation_warnings
    assert outcome.adaptation_messages


def test_video_request_is_adapted_before_the_provider_call() -> None:
    provider = _provider(
        CAPABILITY_VIDEO_GEN,
        ["duration:1-2", "count:2"],
    )
    service = VideoService(
        _router(provider),
        record_sidecar=False,
        ffprobe_runner=_probe_runner(duration=2.0, video=True),
    )
    request = VideoRequest(prompt="a storm", duration_seconds=9.0, count=3)

    outcome = _run(service.generate(request, token=service.screen(request.prompt)))

    sent = provider.calls[-1].request
    assert isinstance(sent, VideoRequest)
    assert sent.duration_seconds == 2.0
    assert sent.count == 2
    assert [warning.action for warning in outcome.adaptation_warnings] == [
        AdaptationAction.CLAMPED_DURATION,
        AdaptationAction.CLAMPED_COUNT,
    ]


def test_music_request_is_adapted_before_the_provider_call() -> None:
    provider = _provider(
        CAPABILITY_MUSIC_GEN,
        ["duration:1-2", "count:2"],
    )
    service = MusicService(
        _router(provider),
        record_sidecar=False,
        ffprobe_runner=_probe_runner(duration=2.0),
    )
    request = MusicRequest(
        prompt="warm piano",
        duration_seconds=30.0,
        count=3,
        format=AudioFormat.WAV,
    )

    outcome = _run(service.generate(request, token=service.screen(request.prompt)))

    sent = provider.calls[-1].request
    assert isinstance(sent, MusicRequest)
    assert sent.duration_seconds == 2.0
    assert sent.count == 2
    assert [warning.action for warning in outcome.adaptation_warnings] == [
        AdaptationAction.CLAMPED_DURATION,
        AdaptationAction.CLAMPED_COUNT,
    ]


def test_speech_request_is_adapted_before_the_provider_call() -> None:
    provider = _provider(CAPABILITY_TTS, ["prompt:10"])
    service = SpeechService(
        _router(provider),
        record_sidecar=False,
        process=False,
        ffprobe_runner=_probe_runner(duration=1.0),
    )
    request = TTSRequest(
        text="one two three four five",
        voice="narrator",
        format=AudioFormat.WAV,
    )

    outcome = _run(service.synthesize(request, token=service.screen(request.text)))

    sent = provider.calls[-1].request
    assert isinstance(sent, TTSRequest)
    assert sent.text == "one two"
    assert [warning.action for warning in outcome.adaptation_warnings] == [
        AdaptationAction.TRUNCATED_PROMPT
    ]


@pytest.mark.parametrize(
    ("capability", "options", "media_request", "option"),
    [
        (
            CAPABILITY_IMAGE_GEN,
            ["8x8"],
            ImageRequest(prompt="x", style="cinematic"),
            "style",
        ),
        (
            CAPABILITY_VIDEO_GEN,
            ["duration:1-2"],
            VideoRequest(prompt="x", generate_audio=True),
            "generate_audio",
        ),
        (
            CAPABILITY_MUSIC_GEN,
            ["duration:1-2"],
            MusicRequest(prompt="x", tempo_bpm=120.0),
            "tempo_bpm",
        ),
        (
            CAPABILITY_TTS,
            ["pronunciation"],
            TTSRequest(text="hello", voice="narrator", speed=1.5),
            "speed",
        ),
    ],
)
def test_unsupported_options_are_rejected_before_the_provider_call(
    capability: str,
    options: Sequence[str],
    media_request: ImageRequest | VideoRequest | MusicRequest | TTSRequest,
    option: str,
) -> None:
    provider = _provider(capability, options)
    if isinstance(media_request, ImageRequest):
        service: Any = ImageService(_router(provider), record_sidecar=False)
        call = service.generate(
            media_request,
            token=service.screen(media_request.prompt),
        )
    elif isinstance(media_request, VideoRequest):
        service = VideoService(
            _router(provider),
            record_sidecar=False,
            ffprobe_runner=_probe_runner(duration=2.0, video=True),
        )
        call = service.generate(
            media_request,
            token=service.screen(media_request.prompt),
        )
    elif isinstance(media_request, MusicRequest):
        service = MusicService(
            _router(provider),
            record_sidecar=False,
            ffprobe_runner=_probe_runner(duration=2.0),
        )
        call = service.generate(
            media_request,
            token=service.screen(media_request.prompt),
        )
    else:
        service = SpeechService(
            _router(provider),
            record_sidecar=False,
            process=False,
            ffprobe_runner=_probe_runner(duration=1.0),
        )
        call = service.synthesize(
            media_request,
            token=service.screen(media_request.text),
        )

    with pytest.raises(UnsupportedOptionError) as caught:
        _run(call)

    assert caught.value.option == option
    assert provider.calls == ()
