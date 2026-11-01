"""Accessibility sidecars for every audio and video output."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from aivia.core.config.sections import AccessibilityConfig
from aivia.media.accessibility import (
    ALT_TEXT_SUFFIX,
    SRT_SUFFIX,
    VTT_SUFFIX,
    AccessibilityEmitter,
    AccessibilitySidecars,
    AccessibilityStoredSidecars,
    accessibility_paths,
    accessibility_storage_key,
    alt_text_path,
    emit_accessibility_sidecars,
    srt_path,
    store_accessibility_sidecars,
    subtitle_paths,
    vtt_path,
    write_accessibility_sidecars,
    write_alt_text,
    write_subtitle_sidecars,
)
from aivia.media.subtitles import Subtitle, SubtitleWord
from aivia.providers.protocols import STTResponse, WordTimestamp


def _words() -> list[SubtitleWord]:
    return [
        SubtitleWord(word="Hello", start_seconds=0.0, end_seconds=0.5),
        SubtitleWord(word="brave", start_seconds=0.5, end_seconds=1.0),
        SubtitleWord(word="world", start_seconds=1.25, end_seconds=1.9),
    ]


def _subtitle() -> Subtitle:
    return Subtitle.from_words(_words())


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_sidecar_paths_append_to_the_full_output_name(tmp_path: Path) -> None:
    audio = tmp_path / "voice.mp3"
    video = tmp_path / "clip.mp4"

    assert alt_text_path(audio) == tmp_path / f"voice.mp3{ALT_TEXT_SUFFIX}"
    assert srt_path(video) == tmp_path / f"clip.mp4{SRT_SUFFIX}"
    assert vtt_path(video) == tmp_path / f"clip.mp4{VTT_SUFFIX}"
    assert subtitle_paths(video) == (
        tmp_path / "clip.mp4.srt",
        tmp_path / "clip.mp4.vtt",
    )
    assert accessibility_paths(audio) == (
        tmp_path / "voice.mp3.alt.txt",
        tmp_path / "voice.mp3.srt",
        tmp_path / "voice.mp3.vtt",
    )
    assert accessibility_paths(str(audio)) == accessibility_paths(audio)


@pytest.mark.parametrize("name", ["voice.mp3", "clip.mp4", "song.wav"])
def test_alt_srt_and_vtt_are_written_next_to_audio_and_video(
    tmp_path: Path,
    name: str,
) -> None:
    output = tmp_path / name

    written = write_accessibility_sidecars(
        output,
        config=AccessibilityConfig(),
        alt_text="A lighthouse in a storm.",
        subtitle=_subtitle(),
    )

    assert isinstance(written, AccessibilitySidecars)
    assert written.output == output
    assert written.alt_text == tmp_path / f"{name}.alt.txt"
    assert written.srt == tmp_path / f"{name}.srt"
    assert written.vtt == tmp_path / f"{name}.vtt"
    assert written.paths == (written.alt_text, written.srt, written.vtt)
    assert written.alt_text.read_text("utf-8") == "A lighthouse in a storm.\n"
    assert written.srt.read_text("utf-8") == _subtitle().to_srt()
    assert written.vtt.read_text("utf-8") == _subtitle().to_vtt()
    assert "WEBVTT" in written.vtt.read_text("utf-8")


def test_disabled_accessibility_writes_nothing(tmp_path: Path) -> None:
    output = tmp_path / "clip.mp4"

    written = write_accessibility_sidecars(
        output,
        config=AccessibilityConfig(enabled=False),
        alt_text="A lighthouse.",
        subtitle=_subtitle(),
    )

    assert written.paths == ()
    assert not written
    assert list(tmp_path.iterdir()) == []


def test_alt_text_and_subtitle_flags_are_independent(tmp_path: Path) -> None:
    only_alt = write_accessibility_sidecars(
        tmp_path / "voice.mp3",
        config=AccessibilityConfig(alt_text=True, subtitles=False),
        alt_text="A narrator.",
        subtitle=_subtitle(),
    )
    only_subtitles = write_accessibility_sidecars(
        tmp_path / "clip.mp4",
        config=AccessibilityConfig(alt_text=False, subtitles=True),
        alt_text="A lighthouse.",
        subtitle=_subtitle(),
    )

    assert only_alt.alt_text is not None
    assert only_alt.srt is None and only_alt.vtt is None
    assert only_subtitles.alt_text is None
    assert only_subtitles.srt is not None and only_subtitles.vtt is not None
    assert not (tmp_path / "clip.mp4.alt.txt").exists()


def test_explicit_write_flags_override_the_config(tmp_path: Path) -> None:
    written = write_accessibility_sidecars(
        tmp_path / "voice.mp3",
        config=AccessibilityConfig(enabled=True, alt_text=False, subtitles=False),
        alt_text="A narrator.",
        subtitle=_subtitle(),
        write_alt_text=True,
        write_subtitles=True,
    )

    assert written.alt_text is not None
    assert written.srt is not None and written.vtt is not None


def test_subtitles_accept_words_a_transcript_and_plain_text(tmp_path: Path) -> None:
    from_words = write_accessibility_sidecars(
        tmp_path / "words.mp3",
        words=_words(),
        alt_text="Words.",
    )
    from_transcript = write_accessibility_sidecars(
        tmp_path / "transcript.mp3",
        transcript=STTResponse(
            text="Hello brave world",
            words=[
                WordTimestamp(word="Hello", start_seconds=0.0, end_seconds=0.5),
                WordTimestamp(word="brave", start_seconds=0.5, end_seconds=1.0),
                WordTimestamp(word="world", start_seconds=1.25, end_seconds=1.9),
            ],
        ),
        alt_text="Transcript.",
    )
    from_text = write_accessibility_sidecars(
        tmp_path / "text.mp4",
        text="Hello brave world",
        duration_seconds=2.0,
        alt_text="Text.",
    )

    assert from_words.srt is not None
    assert from_transcript.vtt is not None
    assert from_text.srt is not None
    assert "Hello brave world" in from_words.srt.read_text("utf-8")
    assert "Hello brave world" in from_transcript.vtt.read_text("utf-8")
    text_srt = from_text.srt.read_text("utf-8")
    assert "00:00:00,000 --> 00:00:02,000" in text_srt
    assert "Hello brave world" in text_srt


def test_alt_text_falls_back_to_the_description_or_subtitle_text(
    tmp_path: Path,
) -> None:
    described = write_accessibility_sidecars(
        tmp_path / "clip.mp4",
        description="A cliff above the sea.",
        subtitle=_subtitle(),
    )
    from_subtitle = write_accessibility_sidecars(
        tmp_path / "voice.mp3",
        subtitle=_subtitle(),
    )

    assert described.alt_text is not None
    assert from_subtitle.alt_text is not None
    assert described.alt_text.read_text("utf-8") == "A cliff above the sea.\n"
    assert from_subtitle.alt_text.read_text("utf-8") == "Hello brave world\n"


def test_config_may_be_a_root_mapping_or_a_bool(tmp_path: Path) -> None:
    root = write_accessibility_sidecars(
        tmp_path / "root.mp3",
        config={"accessibility": {"enabled": True, "subtitles": False}},
        alt_text="Root.",
        subtitle=_subtitle(),
    )
    disabled = write_accessibility_sidecars(
        tmp_path / "off.mp3",
        config=False,
        alt_text="Off.",
        subtitle=_subtitle(),
    )

    assert root.alt_text is not None and root.srt is None
    assert disabled.paths == ()


def test_the_emitter_reuses_its_config_and_supports_aliases(tmp_path: Path) -> None:
    emitter = AccessibilityEmitter(AccessibilityConfig(alt_text=True, subtitles=True))

    written = emitter.emit(
        tmp_path / "voice.mp3",
        alt_text="A narrator.",
        subtitle=_subtitle(),
    )
    alias = emit_accessibility_sidecars(
        tmp_path / "clip.mp4",
        alt_text="A lighthouse.",
        subtitle=_subtitle(),
    )

    assert emitter.enabled is True
    assert emitter.config.subtitles is True
    assert written.paths == alias.paths or len(written.paths) == 3
    assert len(alias.paths) == 3


def test_individual_writers_return_paths_and_reject_empty_content(
    tmp_path: Path,
) -> None:
    alt = write_alt_text(tmp_path / "voice.mp3", "A narrator.")
    srt, vtt = write_subtitle_sidecars(tmp_path / "voice.mp3", _subtitle())

    assert alt.read_text("utf-8") == "A narrator.\n"
    assert srt == tmp_path / "voice.mp3.srt"
    assert vtt == tmp_path / "voice.mp3.vtt"

    with pytest.raises(ValueError):
        write_alt_text(tmp_path / "voice.mp3", "   ")
    with pytest.raises(ValueError):
        write_subtitle_sidecars(tmp_path / "voice.mp3", [])


def test_no_content_writes_nothing_but_still_returns_the_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "empty.mp3"

    written = write_accessibility_sidecars(output, config=AccessibilityConfig())

    assert written.output == output
    assert written.paths == ()
    assert list(tmp_path.iterdir()) == []


def test_summary_is_json_serializable(tmp_path: Path) -> None:
    written = write_accessibility_sidecars(
        tmp_path / "clip.mp4",
        alt_text="A lighthouse.",
        subtitle=_subtitle(),
    )

    payload = json.loads(json.dumps(written.to_dict()))

    assert payload["output"] == str(tmp_path / "clip.mp4")
    assert payload["alt_text"] == str(tmp_path / "clip.mp4.alt.txt")
    assert payload["srt"] == str(tmp_path / "clip.mp4.srt")
    assert payload["vtt"] == str(tmp_path / "clip.mp4.vtt")
    assert len(payload["paths"]) == 3


def test_the_media_toolkit_writes_sidecars_for_audio_and_video_outputs(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from aivia.core.router import Router
    from aivia.media.toolkit import MediaToolkit
    from aivia.providers.mock import MockProvider

    toolkit = MediaToolkit(
        Router({"mock": MockProvider()}),
        accessibility=AccessibilityConfig(),
    )
    speech_asset = SimpleNamespace(
        path=str(tmp_path / "voice.mp3"),
        analysis=SimpleNamespace(duration_seconds=1.5),
    )
    video_asset = SimpleNamespace(
        path=str(tmp_path / "clip.mp4"),
        analysis=SimpleNamespace(duration_seconds=2.0),
    )
    speech_request = SimpleNamespace(
        text="Hello brave world",
        metadata={},
        language=None,
    )
    video_request = SimpleNamespace(
        prompt="A lighthouse in a storm",
        metadata={},
        language=None,
    )
    speech_outcome = SimpleNamespace(
        assets=(speech_asset,),
        response=SimpleNamespace(word_timestamps=[]),
    )
    video_outcome = SimpleNamespace(assets=(video_asset,))

    _run(toolkit._write_accessibility("speech", speech_request, speech_outcome))
    _run(toolkit._write_accessibility("video", video_request, video_outcome))

    assert (tmp_path / "voice.mp3.alt.txt").read_text("utf-8") == (
        "Hello brave world\n"
    )
    assert "Hello brave world" in (tmp_path / "voice.mp3.srt").read_text("utf-8")
    assert "WEBVTT" in (tmp_path / "voice.mp3.vtt").read_text("utf-8")
    assert (tmp_path / "clip.mp4.alt.txt").read_text("utf-8") == (
        "A lighthouse in a storm\n"
    )
    assert "A lighthouse in a storm" in (tmp_path / "clip.mp4.srt").read_text("utf-8")


def test_the_speech_service_writes_accessibility_sidecars_for_its_asset(
    tmp_path: Path,
) -> None:
    from aivia.core.config.capability import CapabilityChainConfig
    from aivia.core.models.asset import Asset, AssetAnalysis, AssetKind
    from aivia.core.router import Router
    from aivia.media.tts_service import SpeechService
    from aivia.providers.protocols import TTSRequest, TTSResponse, WordTimestamp

    service = SpeechService(
        Router({}, config=CapabilityChainConfig(chain=[])),
        accessibility=AccessibilityConfig(),
    )
    asset = Asset(
        kind=AssetKind.VOICE,
        path=(tmp_path / "voice.mp3").as_uri(),
        analysis=AssetAnalysis(duration_seconds=2.0),
    )
    request = TTSRequest(text="Hello brave world", voice="narrator")
    response = TTSResponse(
        audio=b"ID3",
        word_timestamps=[
            WordTimestamp(word="Hello", start_seconds=0.0, end_seconds=0.5),
            WordTimestamp(word="brave", start_seconds=0.5, end_seconds=1.0),
            WordTimestamp(word="world", start_seconds=1.25, end_seconds=1.9),
        ],
    )

    written = _run(
        service._write_accessibility_sidecars(
            asset,
            request,
            response,
            storage=None,
            stored=None,
        )
    )

    assert len(written) == 1
    assert (tmp_path / "voice.mp3.alt.txt").read_text("utf-8") == (
        "Hello brave world\n"
    )
    assert "Hello brave world" in (tmp_path / "voice.mp3.srt").read_text("utf-8")
    assert "WEBVTT" in (tmp_path / "voice.mp3.vtt").read_text("utf-8")


def test_remote_sidecars_are_stored_under_the_output_key() -> None:
    from aivia.providers.mock import MockStorageProvider

    storage = MockStorageProvider()
    output = "https://storage.example.com/speech/voice.mp3"

    written = _run(
        store_accessibility_sidecars(
            output,
            storage,
            AccessibilityConfig(),
            alt_text="A narrator.",
            subtitle=_subtitle(),
        )
    )

    assert isinstance(written, AccessibilityStoredSidecars)
    assert written.output == output
    assert written.keys == (
        "speech/voice.mp3.alt.txt",
        "speech/voice.mp3.srt",
        "speech/voice.mp3.vtt",
    )
    assert written.alt_text is not None
    assert written.alt_text.content_type == "text/plain; charset=utf-8"
    assert written.srt is not None
    assert written.vtt is not None
    assert _run(storage.get("speech/voice.mp3.alt.txt")) == b"A narrator.\n"
    assert _run(storage.get("speech/voice.mp3.srt")) == _subtitle().to_srt().encode()
    assert _run(storage.get("speech/voice.mp3.vtt")) == _subtitle().to_vtt().encode()


def test_remote_sidecars_use_the_supplied_key_and_overwrite_on_repeat() -> None:
    from aivia.providers.mock import MockStorageProvider

    storage = MockStorageProvider()

    first = _run(
        store_accessibility_sidecars(
            "https://cdn.example.com/video/clip.mp4",
            storage,
            AccessibilityConfig(),
            key="video/clip.mp4",
            alt_text="First description.",
        )
    )
    second = _run(
        store_accessibility_sidecars(
            "https://cdn.example.com/video/clip.mp4",
            storage,
            AccessibilityConfig(),
            key="video/clip.mp4",
            alt_text="Second description.",
        )
    )

    assert first.keys == ("video/clip.mp4.alt.txt",)
    assert second.keys == first.keys
    assert _run(storage.get("video/clip.mp4.alt.txt")) == b"Second description.\n"


def test_accessibility_storage_key_derives_a_clean_key() -> None:
    assert accessibility_storage_key("clip.mp4") == "clip.mp4"
    assert (
        accessibility_storage_key("https://cdn.example.com/a/clip.mp4?token=secret")
        == "a/clip.mp4"
    )
    assert accessibility_storage_key("file:///tmp/voice.mp3") == "voice.mp3"
    assert accessibility_storage_key(Path("/tmp/voice.mp3")) == "voice.mp3"
    assert accessibility_storage_key("clip.mp4", key=" pinned/key ") == "pinned/key"
    with pytest.raises(ValueError):
        accessibility_storage_key("clip.mp4", key="   ")


def test_disabled_remote_accessibility_stores_nothing() -> None:
    from aivia.providers.mock import MockStorageProvider

    storage = MockStorageProvider()

    written = _run(
        store_accessibility_sidecars(
            "https://storage.example.com/speech/off.mp3",
            storage,
            AccessibilityConfig(enabled=False),
            alt_text="Off.",
            subtitle=_subtitle(),
        )
    )

    assert not written
    assert written.keys == ()
    assert _run(storage.stat("speech/off.mp3.alt.txt")) is None


def test_a_remote_speech_output_stores_accessibility_sidecars() -> None:
    import subprocess

    from aivia.core.config.capability import CapabilityChainConfig
    from aivia.core.models import ProviderMeta, ProviderPrice, RateLimit
    from aivia.core.router import Router
    from aivia.media.tts_service import SpeechService
    from aivia.providers import Provider
    from aivia.providers.mock import MockStorageProvider
    from aivia.providers.protocols import (
        CAPABILITY_TTS,
        AudioFormat,
        TTSRequest,
        TTSResponse,
        WordTimestamp,
    )

    class RemoteSpeech(Provider):
        def __init__(self) -> None:
            super().__init__(
                ProviderMeta(
                    name="remote-speech",
                    capabilities=[CAPABILITY_TTS],
                    pricing={
                        CAPABILITY_TTS: ProviderPrice(
                            amount_usd=0.001,
                            unit="character",
                        )
                    },
                    rate_limit=RateLimit(requests_per_minute=0),
                )
            )

        async def synthesize(self, request: TTSRequest) -> TTSResponse:
            return TTSResponse(
                audio=b"ID3\x04\x00\x00\x00\x00\x00\x00remote",
                format=AudioFormat.MP3,
                word_timestamps=[
                    WordTimestamp(word="Hello", start_seconds=0.0, end_seconds=0.5),
                    WordTimestamp(word="world", start_seconds=0.5, end_seconds=1.0),
                ],
            )

    def probe_runner(argv: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            list(argv),
            0,
            stdout=json.dumps(
                {
                    "streams": [
                        {
                            "index": 0,
                            "codec_type": "audio",
                            "codec_name": "mp3",
                            "duration": "1.000",
                            "sample_rate": "44100",
                            "channels": 1,
                        }
                    ],
                    "format": {
                        "format_name": "mp3",
                        "format_long_name": "MP2/3 (MPEG audio layer 2/3)",
                        "duration": "1.000",
                        "size": "12345",
                    },
                }
            ),
            stderr="",
        )

    provider = RemoteSpeech()
    storage = MockStorageProvider()
    service = SpeechService(
        Router(
            {provider.name: provider},
            config=CapabilityChainConfig(chain=[provider.name]),
        ),
        storage=storage,
        accessibility=AccessibilityConfig(),
        record_cost=False,
        record_sidecar=False,
        ffprobe_runner=probe_runner,
    )

    outcome = _run(service.synthesize(TTSRequest(text="Hello world", voice="narrator")))

    assert outcome.asset is not None
    assert outcome.asset.path.startswith("https://storage.example.com/")
    stored = outcome.storage_objects[0]
    assert outcome.accessibility_sidecars
    written = outcome.accessibility_sidecars[0]
    assert isinstance(written, AccessibilityStoredSidecars)
    assert written.keys == (
        f"{stored.key}.alt.txt",
        f"{stored.key}.srt",
        f"{stored.key}.vtt",
    )
    assert _run(storage.get(f"{stored.key}.alt.txt")) == b"Hello world\n"


def test_a_remote_video_output_stores_accessibility_sidecars() -> None:
    import struct
    import subprocess

    from aivia.core.config.capability import CapabilityChainConfig
    from aivia.core.db import create_database_engine, upgrade_database
    from aivia.core.models import ProviderMeta, ProviderPrice, RateLimit
    from aivia.core.router import Router
    from aivia.media.screening import screen_prompt
    from aivia.media.video_cache import VideoCache
    from aivia.media.video_service import VideoService
    from aivia.providers import Provider
    from aivia.providers.mock import MockStorageProvider
    from aivia.providers.protocols import (
        CAPABILITY_VIDEO_GEN,
        VideoFormat,
        VideoRequest,
        VideoResponse,
        VideoResult,
    )

    class RemoteVideo(Provider):
        def __init__(self) -> None:
            super().__init__(
                ProviderMeta(
                    name="remote-video",
                    capabilities=[CAPABILITY_VIDEO_GEN],
                    pricing={
                        CAPABILITY_VIDEO_GEN: ProviderPrice(
                            amount_usd=0.05,
                            unit="clip",
                        )
                    },
                    rate_limit=RateLimit(requests_per_minute=0),
                )
            )

        async def generate(self, request: VideoRequest) -> VideoResponse:
            return VideoResponse(
                videos=[
                    VideoResult(
                        data=struct.pack(">I", 24) + b"ftypisom" + b"\x00" * 12,
                        format=VideoFormat.MP4,
                        fps=request.fps,
                        seed=request.seed,
                    )
                ],
                model="video-1",
                provider=self.name,
            )

    def probe_runner(argv: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            list(argv),
            0,
            stdout=json.dumps(
                {
                    "streams": [
                        {
                            "index": 0,
                            "codec_type": "video",
                            "codec_name": "h264",
                            "width": 1280,
                            "height": 720,
                            "duration": "5.0",
                            "avg_frame_rate": "30/1",
                        }
                    ],
                    "format": {
                        "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                        "format_long_name": "QuickTime / MOV",
                        "duration": "5.0",
                        "size": "12345",
                    },
                }
            ),
            stderr="",
        )

    provider = RemoteVideo()
    storage = MockStorageProvider()
    engine = create_database_engine(url="sqlite+pysqlite:///:memory:")
    upgrade_database(engine=engine)
    try:
        cache = VideoCache(
            engine=engine,
            storage=storage,
            ffprobe_runner=probe_runner,
        )
        service = VideoService(
            Router(
                {provider.name: provider},
                config=CapabilityChainConfig(chain=[provider.name]),
            ),
            cache=cache,
            storage=storage,
            accessibility=AccessibilityConfig(),
            record_cost=False,
            record_sidecar=False,
        )
        prompt = "A lighthouse in a storm"
        outcome = _run(
            service.generate(
                VideoRequest(prompt=prompt, duration_seconds=5.0),
                token=str(screen_prompt(prompt)),
            )
        )
    finally:
        engine.dispose()

    assert outcome.asset is not None
    assert outcome.asset.path.startswith("https://storage.example.com/")
    stored = outcome.storage_objects[0]
    assert outcome.accessibility_sidecars
    written = outcome.accessibility_sidecars[0]
    assert isinstance(written, AccessibilityStoredSidecars)
    assert written.keys == (
        f"{stored.key}.alt.txt",
        f"{stored.key}.srt",
        f"{stored.key}.vtt",
    )
    assert _run(storage.get(f"{stored.key}.alt.txt")) == (prompt + "\n").encode()
