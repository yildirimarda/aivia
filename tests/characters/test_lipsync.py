"""Lip sync provider adapter and the no-AI sprite sheet fallback.

The adapter delegates to a lip sync provider when no sprite sheet is supplied
and otherwise animates a sprite sheet locally: mouth frames are swapped in
time with the audio, blinking is inserted and a gentle idle drift is applied.
No model call is made in the fallback path.
"""

from __future__ import annotations

import asyncio
import io
from typing import Any

import pytest
from PIL import Image

from aivia.characters.lipsync import LipSyncAdapter, LipSyncError
from aivia.core.models import ProviderMeta
from aivia.providers import Provider
from aivia.providers.protocols import CAPABILITY_LIPSYNC
from aivia.providers.protocols.lipsync import (
    LipSyncRequest,
    LipSyncResponse,
    LipSyncResult,
    LipSyncSpriteSheet,
)
from aivia.providers.protocols.media import MediaInput


class _FakeProvider(Provider):
    """A stand-in lip sync provider that echoes the request back as a clip."""

    default_metadata = ProviderMeta(
        name="fake-lipsync", capabilities=[CAPABILITY_LIPSYNC]
    )

    def __init__(self, used_fallback: bool = False) -> None:
        super().__init__()
        self.used_fallback = used_fallback
        self.calls: list[LipSyncRequest] = []

    async def animate(self, request: LipSyncRequest) -> LipSyncResponse:
        self.calls.append(request)
        return LipSyncResponse(
            video=LipSyncResult(
                data=b"fake-clip",
                format=request.format,
                width=64,
                height=64,
                duration_seconds=request.duration_seconds,
                fps=request.fps,
                used_fallback=self.used_fallback,
            ),
            model="fake-lipsync-1",
            provider="fake-lipsync",
        )


def _png_bytes(size: tuple[int, int] = (256, 64)) -> bytes:
    image = Image.new("RGBA", size, (255, 0, 0, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _runner(payload: bytes = b"FAKE-MP4") -> tuple[list[list[str]], Any]:
    calls: list[list[str]] = []

    def run(args: Any) -> bytes:
        calls.append(list(args))
        return payload

    return calls, run


def _request(
    *,
    sprite_sheet: LipSyncSpriteSheet | None = None,
    duration_seconds: float | None = 1.0,
    fps: int = 10,
) -> LipSyncRequest:
    return LipSyncRequest(
        audio=MediaInput(data=b"audio", media_type="audio/wav"),
        source=MediaInput(data=_png_bytes((64, 64)), media_type="image/png"),
        sprite_sheet=sprite_sheet,
        duration_seconds=duration_seconds,
        fps=fps,
    )


def test_adapter_delegates_to_provider_without_sprite_sheet() -> None:
    provider = _FakeProvider(used_fallback=False)
    adapter = LipSyncAdapter(provider=provider)
    response = asyncio.run(adapter.animate(_request(sprite_sheet=None)))

    assert provider.calls == [_request(sprite_sheet=None)]
    assert response.video.used_fallback is False
    assert response.video.data == b"fake-clip"
    assert response.provider == "fake-lipsync"


def test_adapter_uses_sprite_sheet_fallback_and_marks_it() -> None:
    calls, run = _runner()
    adapter = LipSyncAdapter(ffmpeg_runner=run)
    sheet = LipSyncSpriteSheet(
        image=MediaInput(data=_png_bytes((256, 64)), media_type="image/png"),
        frame_width=64,
        frame_height=64,
        frame_count=4,
        mouth_frames={"a": 0, "b": 1},
        blink_frame=2,
        idle_frames=[3],
    )
    response = asyncio.run(adapter.animate(_request(sprite_sheet=sheet)))

    assert response.video.used_fallback is True
    assert response.video.data == b"FAKE-MP4"
    assert response.video.duration_seconds == pytest.approx(1.0)
    assert response.video.fps == 10
    assert response.video.width == 64
    assert response.video.height == 64
    assert response.model == "aivia-lipsync-fallback"
    assert response.provider == "aivia"
    assert len(calls) == 1


def test_fallback_ffmpeg_command_uses_both_inputs_and_the_crop_filter() -> None:
    calls, run = _runner()
    adapter = LipSyncAdapter(ffmpeg_runner=run)
    sheet = LipSyncSpriteSheet(
        image=MediaInput(data=_png_bytes((256, 64)), media_type="image/png"),
        frame_width=64,
        frame_height=64,
        frame_count=4,
        mouth_frames={"a": 0, "b": 1},
    )
    asyncio.run(adapter.animate(_request(sprite_sheet=sheet)))

    args = calls[0]
    assert "-loop" in args and "1" in args
    assert "-i" in args
    assert "-filter_complex" in args
    filter_index = args.index("-filter_complex")
    graph = args[filter_index + 1]
    assert "crop=64:64" in graph
    assert "fps=10" in graph
    assert "-map" in args and "[v]" in args
    assert "-c:v" in args and "libx264" in args
    assert "-c:a" in args and "aac" in args
    assert "-t" in args


def test_adapter_raises_without_provider_or_sprite_sheet() -> None:
    adapter = LipSyncAdapter()
    with pytest.raises(LipSyncError) as caught:
        asyncio.run(adapter.animate(_request(sprite_sheet=None)))
    assert caught.value.code == "lipsync"


def test_fallback_requires_a_duration() -> None:
    calls, run = _runner()
    adapter = LipSyncAdapter(ffmpeg_runner=run)
    sheet = LipSyncSpriteSheet(
        image=MediaInput(data=_png_bytes((256, 64)), media_type="image/png"),
        frame_width=64,
        frame_height=64,
        frame_count=4,
    )
    request = _request(sprite_sheet=sheet, duration_seconds=None)
    with pytest.raises(LipSyncError):
        asyncio.run(adapter.animate(request))
    assert calls == []


def test_fallback_requires_inline_sprite_sheet_bytes() -> None:
    calls, run = _runner()
    adapter = LipSyncAdapter(ffmpeg_runner=run)
    sheet = LipSyncSpriteSheet(
        image=MediaInput(url="https://example.com/sheet.png", media_type="image/png"),
        frame_width=64,
        frame_height=64,
        frame_count=4,
    )
    with pytest.raises(LipSyncError):
        asyncio.run(adapter.animate(_request(sprite_sheet=sheet)))
    assert calls == []


def test_fallback_blink_and_idle_frames_appear_in_the_filter() -> None:
    calls, run = _runner()
    adapter = LipSyncAdapter(ffmpeg_runner=run)
    sheet = LipSyncSpriteSheet(
        image=MediaInput(data=_png_bytes((256, 64)), media_type="image/png"),
        frame_width=64,
        frame_height=64,
        frame_count=4,
        mouth_frames={"a": 0, "b": 1},
        blink_frame=2,
        idle_frames=[3],
    )
    asyncio.run(adapter.animate(_request(sprite_sheet=sheet)))

    graph = calls[0][calls[0].index("-filter_complex") + 1]
    assert "crop=64:64" in graph
    # The blink frame is referenced as a possible key frame.
    assert "2" in graph


def test_fallback_ffmpeg_is_required() -> None:
    adapter = LipSyncAdapter(ffmpeg_executable="definitely-not-ffmpeg")
    sheet = LipSyncSpriteSheet(
        image=MediaInput(data=_png_bytes((256, 64)), media_type="image/png"),
        frame_width=64,
        frame_height=64,
        frame_count=4,
    )
    with pytest.raises(LipSyncError):
        asyncio.run(adapter.animate(_request(sprite_sheet=sheet)))
