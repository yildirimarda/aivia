"""Video flags filter providers and remove audio the request did not ask for."""

from __future__ import annotations

import json
import struct
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from aivia.core.config import CapabilityChainConfig, CapabilityStrategy
from aivia.core.errors import (
    InvalidRequestError,
    ProviderExhaustedError,
    ValidationError,
)
from aivia.core.models import ProviderMeta
from aivia.core.router import Router
from aivia.providers import Provider
from aivia.providers.protocols.media import MediaInput
from aivia.providers.protocols.video import (
    CAPABILITY_VIDEO_AUDIO,
    CAPABILITY_VIDEO_AUDIO_NATIVE,
    CAPABILITY_VIDEO_FIRST_FRAME,
    CAPABILITY_VIDEO_GEN,
    CAPABILITY_VIDEO_LAST_FRAME,
    CAPABILITY_VIDEO_NATIVE_AUDIO,
    VideoRequest,
    VideoResult,
)
from aivia.providers.video import (
    VideoCapabilityNotSupportedError,
    VideoFlags,
    declared_video_flags,
    ensure_video_support,
    filter_providers_for_video_flags,
    prepare_video_request,
    required_video_capabilities,
    strip_audio,
    strip_audio_file,
    strip_unwanted_audio,
    supports_audio,
    supports_first_frame,
    supports_last_frame,
    supports_native_audio,
    wants_native_audio,
)

_FRAME_URL = "https://example.com/media/frame.png"


class FakeVideoProvider(Provider):
    def __init__(self, name: str, capabilities: list[str]) -> None:
        super().__init__(ProviderMeta(name=name, capabilities=capabilities))


def _router(*providers: Provider) -> Router:
    return Router(
        providers={provider.name: provider for provider in providers},
        config=CapabilityChainConfig(
            strategy=CapabilityStrategy.FIRST_AVAILABLE,
            chain=[provider.name for provider in providers],
        ),
    )


def mp4() -> bytes:
    return struct.pack(">I", 24) + b"ftypisom" + b"\x00" * 12


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


def probe_runner(
    payload: str,
    *,
    calls: list[list[str]] | None = None,
) -> Any:
    def runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        if calls is not None:
            calls.append(list(argv))
        return subprocess.CompletedProcess(list(argv), 0, stdout=payload, stderr="")

    return runner


def ffmpeg_writer(
    *,
    calls: list[list[str]] | None = None,
    payload: bytes | None = None,
) -> Any:
    output = payload or mp4()

    def runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        if calls is not None:
            calls.append(list(argv))
        Path(argv[-1]).write_bytes(output)
        return subprocess.CompletedProcess(list(argv), 0, stdout="", stderr="")

    return runner


def test_capability_constants_are_declared_and_exported() -> None:
    assert CAPABILITY_VIDEO_LAST_FRAME == "video_last_frame"
    assert CAPABILITY_VIDEO_NATIVE_AUDIO == "video_native_audio"
    assert CAPABILITY_VIDEO_AUDIO == CAPABILITY_VIDEO_NATIVE_AUDIO
    assert CAPABILITY_VIDEO_AUDIO_NATIVE == CAPABILITY_VIDEO_NATIVE_AUDIO
    assert CAPABILITY_VIDEO_AUDIO == "video_audio"


def test_supports_reports_each_optional_video_input() -> None:
    provider = FakeVideoProvider(
        "full",
        [
            CAPABILITY_VIDEO_GEN,
            CAPABILITY_VIDEO_FIRST_FRAME,
            CAPABILITY_VIDEO_LAST_FRAME,
            CAPABILITY_VIDEO_NATIVE_AUDIO,
        ],
    )

    assert supports_first_frame(provider)
    assert supports_last_frame(provider)
    assert supports_native_audio(provider)
    assert supports_audio(provider)
    assert declared_video_flags(provider) == VideoFlags(
        first_frame=True,
        last_frame=True,
        native_audio=True,
    )
    assert declared_video_flags(provider).capabilities == (
        CAPABILITY_VIDEO_FIRST_FRAME,
        CAPABILITY_VIDEO_LAST_FRAME,
        CAPABILITY_VIDEO_NATIVE_AUDIO,
    )

    text_only = FakeVideoProvider("text-only", [CAPABILITY_VIDEO_GEN])

    assert not supports_first_frame(text_only)
    assert not supports_last_frame(text_only)
    assert not supports_native_audio(text_only)
    assert declared_video_flags(text_only) == VideoFlags()


def test_supported_options_also_declare_a_flag() -> None:
    provider = FakeVideoProvider("declarative", [CAPABILITY_VIDEO_GEN])
    provider.metadata.supported_options[CAPABILITY_VIDEO_GEN] = [
        "first_frame",
        "last_frame",
        "generate_audio",
    ]

    assert supports_first_frame(provider)
    assert supports_last_frame(provider)
    assert supports_native_audio(provider)


def test_required_capabilities_follow_the_request_flags() -> None:
    plain = VideoRequest(prompt="a wave")
    assert required_video_capabilities(plain) == ()
    assert wants_native_audio(plain) is False

    request = VideoRequest(
        prompt="animate",
        first_frame=MediaInput(url=_FRAME_URL),
        last_frame=MediaInput(url=_FRAME_URL),
        generate_audio=True,
    )

    assert required_video_capabilities(request) == (
        CAPABILITY_VIDEO_FIRST_FRAME,
        CAPABILITY_VIDEO_LAST_FRAME,
        CAPABILITY_VIDEO_NATIVE_AUDIO,
    )
    assert wants_native_audio(request) is True


def test_filtering_keeps_only_providers_with_every_required_capability() -> None:
    text_only = FakeVideoProvider("text-only", [CAPABILITY_VIDEO_GEN])
    first_only = FakeVideoProvider(
        "first-only",
        [CAPABILITY_VIDEO_GEN, CAPABILITY_VIDEO_FIRST_FRAME],
    )
    full = FakeVideoProvider(
        "full",
        [
            CAPABILITY_VIDEO_GEN,
            CAPABILITY_VIDEO_FIRST_FRAME,
            CAPABILITY_VIDEO_LAST_FRAME,
            CAPABILITY_VIDEO_NATIVE_AUDIO,
        ],
    )
    request = VideoRequest(
        prompt="animate",
        first_frame=MediaInput(url=_FRAME_URL),
        last_frame=MediaInput(url=_FRAME_URL),
        generate_audio=True,
    )

    assert filter_providers_for_video_flags([text_only, first_only, full], request) == (
        full,
    )
    assert filter_providers_for_video_flags(
        [text_only, first_only], VideoRequest(prompt="a wave")
    ) == (text_only, first_only)


def test_the_router_skips_providers_missing_a_flag() -> None:
    text_only = FakeVideoProvider("text-only", [CAPABILITY_VIDEO_GEN])
    first_only = FakeVideoProvider(
        "first-only",
        [CAPABILITY_VIDEO_GEN, CAPABILITY_VIDEO_FIRST_FRAME],
    )
    request = VideoRequest(
        prompt="animate",
        first_frame=MediaInput(url=_FRAME_URL),
    )

    selected = _router(text_only, first_only).select(
        CAPABILITY_VIDEO_GEN,
        required=required_video_capabilities(request),
    )

    assert selected is first_only


def test_the_router_reports_exhaustion_when_no_provider_accepts_the_flags() -> None:
    text_only = FakeVideoProvider("text-only", [CAPABILITY_VIDEO_GEN])
    request = VideoRequest(prompt="animate", generate_audio=True)

    with pytest.raises(ProviderExhaustedError):
        _router(text_only).select(
            CAPABILITY_VIDEO_GEN,
            required=required_video_capabilities(request),
        )


def test_ensure_raises_for_the_missing_capability_and_field() -> None:
    provider = FakeVideoProvider("text-only", [CAPABILITY_VIDEO_GEN])
    request = VideoRequest(
        prompt="animate",
        last_frame=MediaInput(url=_FRAME_URL),
    )

    with pytest.raises(VideoCapabilityNotSupportedError) as caught:
        ensure_video_support(provider, request)

    assert isinstance(caught.value, InvalidRequestError)
    assert caught.value.provider == "text-only"
    assert caught.value.capability == CAPABILITY_VIDEO_LAST_FRAME
    assert caught.value.details["required_capability"] == CAPABILITY_VIDEO_LAST_FRAME
    assert caught.value.details["field"] == "last_frame"
    assert CAPABILITY_VIDEO_LAST_FRAME in str(caught.value)


def test_prepare_returns_a_request_only_when_the_provider_accepts_it() -> None:
    full = FakeVideoProvider(
        "full",
        [
            CAPABILITY_VIDEO_GEN,
            CAPABILITY_VIDEO_FIRST_FRAME,
            CAPABILITY_VIDEO_LAST_FRAME,
            CAPABILITY_VIDEO_NATIVE_AUDIO,
        ],
    )
    request = VideoRequest(
        prompt="animate",
        first_frame=MediaInput(url=_FRAME_URL),
        last_frame=MediaInput(url=_FRAME_URL),
        generate_audio=True,
    )

    assert prepare_video_request(full, request) is request

    with pytest.raises(VideoCapabilityNotSupportedError):
        prepare_video_request(
            FakeVideoProvider("text-only", [CAPABILITY_VIDEO_GEN]),
            request,
        )


def test_unwanted_audio_is_stripped_and_verified_by_ffprobe() -> None:
    ffmpeg_calls: list[list[str]] = []
    probe_calls: list[list[str]] = []
    result = VideoResult(data=mp4(), has_audio=True)
    request = VideoRequest(prompt="a silent wave")

    stripped = strip_unwanted_audio(
        result,
        request,
        ffmpeg_runner=ffmpeg_writer(calls=ffmpeg_calls),
        ffprobe_runner=probe_runner(probe_json(audio=False), calls=probe_calls),
    )

    assert isinstance(stripped, VideoResult)
    assert stripped.has_audio is False
    assert stripped.data == mp4()
    assert len(ffmpeg_calls) == 1
    assert "-an" in ffmpeg_calls[0]
    assert len(probe_calls) == 1


def test_requested_audio_and_already_silent_results_are_left_alone() -> None:
    result = VideoResult(data=mp4(), has_audio=True)

    assert (
        strip_unwanted_audio(
            result,
            VideoRequest(prompt="with sound", generate_audio=True),
        )
        is result
    )

    silent = VideoResult(data=mp4(), has_audio=False)
    assert strip_unwanted_audio(silent, VideoRequest(prompt="silent")) is silent


def test_bytes_and_files_can_be_stripped(tmp_path: Path) -> None:
    stripped_bytes = strip_audio(
        mp4(),
        ffmpeg_runner=ffmpeg_writer(),
        ffprobe_runner=probe_runner(probe_json(audio=False)),
    )
    assert stripped_bytes == mp4()

    source = tmp_path / "source.mp4"
    source.write_bytes(mp4())
    output = tmp_path / "output.mp4"
    stripped_path = strip_audio_file(
        source,
        output,
        ffmpeg_runner=ffmpeg_writer(),
        ffprobe_runner=probe_runner(probe_json(audio=False)),
    )
    assert stripped_path == output
    assert output.read_bytes() == mp4()


def test_a_failed_ffmpeg_run_is_rejected() -> None:
    def failing(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(list(argv), 1, stdout="", stderr="boom")

    with pytest.raises(ValidationError, match="ffmpeg rejected"):
        strip_audio(
            mp4(),
            ffmpeg_runner=failing,
            ffprobe_runner=probe_runner(probe_json(audio=False)),
        )


def test_audio_that_survives_stripping_is_rejected() -> None:
    with pytest.raises(ValidationError, match="audio stream remains"):
        strip_audio(
            mp4(),
            ffmpeg_runner=ffmpeg_writer(),
            ffprobe_runner=probe_runner(probe_json(audio=True)),
        )


def test_remote_video_is_not_silently_rewritten() -> None:
    result = VideoResult(url="https://example.com/clip.mp4", has_audio=True)

    with pytest.raises(ValidationError, match="remote video"):
        strip_unwanted_audio(result, VideoRequest(prompt="silent"))
