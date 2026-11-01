"""Accessibility sidecars for every audio and video output."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aivia.core.config.sections import AccessibilityConfig
from aivia.media.accessibility import (
    ALT_TEXT_SUFFIX,
    SRT_SUFFIX,
    VTT_SUFFIX,
    AccessibilityEmitter,
    AccessibilitySidecars,
    accessibility_paths,
    alt_text_path,
    emit_accessibility_sidecars,
    srt_path,
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
    emitter = AccessibilityEmitter(
        AccessibilityConfig(alt_text=True, subtitles=True)
    )

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

    toolkit._write_accessibility("speech", speech_request, speech_outcome)
    toolkit._write_accessibility("video", video_request, video_outcome)

    assert (tmp_path / "voice.mp3.alt.txt").read_text("utf-8") == (
        "Hello brave world\n"
    )
    assert "Hello brave world" in (tmp_path / "voice.mp3.srt").read_text("utf-8")
    assert "WEBVTT" in (tmp_path / "voice.mp3.vtt").read_text("utf-8")
    assert (tmp_path / "clip.mp4.alt.txt").read_text("utf-8") == (
        "A lighthouse in a storm\n"
    )
    assert "A lighthouse in a storm" in (tmp_path / "clip.mp4.srt").read_text(
        "utf-8"
    )


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

    written = service._write_accessibility_sidecars(asset, request, response)

    assert len(written) == 1
    assert (tmp_path / "voice.mp3.alt.txt").read_text("utf-8") == (
        "Hello brave world\n"
    )
    assert "Hello brave world" in (tmp_path / "voice.mp3.srt").read_text("utf-8")
    assert "WEBVTT" in (tmp_path / "voice.mp3.vtt").read_text("utf-8")
