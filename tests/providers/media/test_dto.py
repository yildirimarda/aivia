"""The media DTOs validate kinds, payloads and numeric ranges."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aivia.providers.media import (
    ImageRequest,
    ImageResult,
    MediaKind,
    MediaRequest,
    MediaResult,
    MusicRequest,
    MusicResult,
    SpeechRequest,
    SpeechResult,
    VideoRequest,
    VideoResult,
)
from aivia.providers.protocols.audio import WordTimestamp
from aivia.providers.protocols.music import MusicTrack
from aivia.providers.protocols.tts import TTSRequest, TTSResponse


def test_speech_and_music_aliases_point_at_the_protocol_models() -> None:
    assert MusicResult is MusicTrack
    assert SpeechRequest is TTSRequest
    assert SpeechResult is TTSResponse


def test_image_request_validates_ranges_and_size_pair() -> None:
    request = ImageRequest(
        prompt="  a red fox  ",
        width=512,
        height=768,
        aspect_ratio=" 3:4 ",
        count=4,
    )

    assert request.prompt == "a red fox"
    assert request.aspect_ratio == "3:4"
    assert request.count == 4

    with pytest.raises(ValidationError):
        ImageRequest(prompt="   ")
    with pytest.raises(ValidationError, match="together"):
        ImageRequest(prompt="x", width=512)
    with pytest.raises(ValidationError):
        ImageRequest(prompt="x", width=0, height=512)
    with pytest.raises(ValidationError):
        ImageRequest(prompt="x", aspect_ratio="16x9")
    with pytest.raises(ValidationError):
        ImageRequest(prompt="x", aspect_ratio="0:9")
    with pytest.raises(ValidationError):
        ImageRequest(prompt="x", count=0)
    with pytest.raises(ValidationError):
        ImageRequest(prompt="x", count=9)


def test_image_result_requires_one_source_and_a_size_pair() -> None:
    result = ImageResult(data=b"image", width=512, height=768)

    assert result.is_inline is True
    assert result.size_bytes == 5
    assert result.dimensions is not None

    with pytest.raises(ValidationError, match="exactly one"):
        ImageResult()
    with pytest.raises(ValidationError, match="exactly one"):
        ImageResult(data=b"image", url="https://example.com/image.png")
    with pytest.raises(ValidationError, match="non-empty"):
        ImageResult(data=b"")
    with pytest.raises(ValidationError, match="together"):
        ImageResult(data=b"image", width=512)
    with pytest.raises(ValidationError):
        ImageResult(data=b"image", height=0)


def test_video_request_validates_duration_fps_count_and_audio() -> None:
    request = VideoRequest(
        prompt="  a slow pan  ",
        duration_seconds=6.0,
        width=1920,
        height=1080,
        fps=30,
        count=2,
        generate_audio=True,
        audio_prompt="  wind  ",
    )

    assert request.prompt == "a slow pan"
    assert request.duration_seconds == 6.0
    assert request.fps == 30
    assert request.count == 2
    assert request.generate_audio is True

    with pytest.raises(ValidationError):
        VideoRequest(prompt="   ")
    with pytest.raises(ValidationError):
        VideoRequest(prompt="x", duration_seconds=0)
    with pytest.raises(ValidationError, match="together"):
        VideoRequest(prompt="x", width=1920)
    with pytest.raises(ValidationError):
        VideoRequest(prompt="x", fps=0)
    with pytest.raises(ValidationError):
        VideoRequest(prompt="x", fps=121)
    with pytest.raises(ValidationError):
        VideoRequest(prompt="x", count=0)
    with pytest.raises(ValidationError):
        VideoRequest(prompt="x", count=5)
    with pytest.raises(ValidationError, match="audio_prompt"):
        VideoRequest(prompt="x", audio_prompt="wind")


def test_video_result_validates_ranges_and_audio_state() -> None:
    result = VideoResult(
        data=b"video",
        width=1920,
        height=1080,
        duration_seconds=6.0,
        fps=30,
        has_audio=True,
    )

    assert result.is_inline is True
    assert result.size_bytes == 5
    assert result.duration_seconds == 6.0
    assert result.has_audio is True

    with pytest.raises(ValidationError, match="exactly one"):
        VideoResult()
    with pytest.raises(ValidationError, match="exactly one"):
        VideoResult(data=b"video", url="https://example.com/video.mp4")
    with pytest.raises(ValidationError, match="non-empty"):
        VideoResult(data=b"")
    with pytest.raises(ValidationError, match="together"):
        VideoResult(data=b"video", width=1920)
    with pytest.raises(ValidationError):
        VideoResult(data=b"video", duration_seconds=0)
    with pytest.raises(ValidationError):
        VideoResult(data=b"video", fps=121)


def test_music_request_validates_duration_count_and_vocals() -> None:
    request = MusicRequest(
        prompt="  a warm synth song  ",
        lyrics="  verse  ",
        duration_seconds=45.0,
        sample_rate=48_000,
        tempo_bpm=120.0,
        count=2,
    )

    assert request.prompt == "a warm synth song"
    assert request.lyrics == "verse"
    assert request.duration_seconds == 45.0
    assert request.sample_rate == 48_000
    assert request.tempo_bpm == 120.0
    assert request.count == 2

    with pytest.raises(ValidationError):
        MusicRequest(prompt="   ")
    with pytest.raises(ValidationError):
        MusicRequest(prompt="x", duration_seconds=0)
    with pytest.raises(ValidationError):
        MusicRequest(prompt="x", sample_rate=0)
    with pytest.raises(ValidationError):
        MusicRequest(prompt="x", tempo_bpm=0)
    with pytest.raises(ValidationError):
        MusicRequest(prompt="x", count=0)
    with pytest.raises(ValidationError):
        MusicRequest(prompt="x", count=9)
    with pytest.raises(ValidationError, match="instrumental"):
        MusicRequest(prompt="x", instrumental=True, lyrics="words")


def test_music_result_requires_one_source_and_valid_ranges() -> None:
    result = MusicResult(data=b"music", duration_seconds=30.0, sample_rate=44_100)

    assert result.is_inline is True
    assert result.size_bytes == 5
    assert result.duration_seconds == 30.0

    with pytest.raises(ValidationError, match="exactly one"):
        MusicResult()
    with pytest.raises(ValidationError, match="exactly one"):
        MusicResult(data=b"music", url="https://example.com/music.mp3")
    with pytest.raises(ValidationError, match="non-empty"):
        MusicResult(data=b"")
    with pytest.raises(ValidationError):
        MusicResult(data=b"music", duration_seconds=-1)
    with pytest.raises(ValidationError):
        MusicResult(data=b"music", sample_rate=0)


def test_speech_request_validates_text_speed_and_pronunciation() -> None:
    request = SpeechRequest(
        text="  hello world  ",
        voice="  narrator  ",
        speed=1.5,
        sample_rate=24_000,
        pronunciation={"aivia": "ah-vee-ah"},
    )

    assert request.text == "hello world"
    assert request.voice == "narrator"
    assert request.speed == 1.5
    assert request.pronunciation == {"aivia": "ah-vee-ah"}

    with pytest.raises(ValidationError):
        SpeechRequest(text="   ", voice="narrator")
    with pytest.raises(ValidationError):
        SpeechRequest(text="hello", voice="   ")
    with pytest.raises(ValidationError):
        SpeechRequest(text="hello", voice="narrator", speed=0.24)
    with pytest.raises(ValidationError):
        SpeechRequest(text="hello", voice="narrator", speed=4.01)
    with pytest.raises(ValidationError):
        SpeechRequest(text="hello", voice="narrator", sample_rate=0)
    with pytest.raises(ValidationError):
        SpeechRequest(text="hello", voice="narrator", pronunciation={" ": "x"})
    with pytest.raises(ValidationError):
        SpeechRequest(text="hello", voice="narrator", pronunciation={"x": " "})


def test_speech_result_validates_audio_and_word_order() -> None:
    result = SpeechResult(
        audio=b"speech",
        duration_seconds=1.0,
        word_timestamps=[
            WordTimestamp(word="hello", start_seconds=0.0, end_seconds=0.5),
            WordTimestamp(word="world", start_seconds=0.5, end_seconds=1.0),
        ],
    )

    assert result.size_bytes == 6
    assert result.has_word_timestamps is True

    with pytest.raises(ValidationError):
        SpeechResult(audio=b"")
    with pytest.raises(ValidationError):
        SpeechResult(audio=b"speech", duration_seconds=-1)
    with pytest.raises(ValidationError, match="ordered"):
        SpeechResult(
            audio=b"speech",
            word_timestamps=[
                WordTimestamp(word="later", start_seconds=1.0, end_seconds=1.5),
                WordTimestamp(word="earlier", start_seconds=0.0, end_seconds=0.5),
            ],
        )


def test_media_request_wraps_exactly_one_matching_payload() -> None:
    image = ImageRequest(prompt="a fox", count=2)
    video = VideoRequest(prompt="a pan", duration_seconds=5.0)
    music = MusicRequest(prompt="a song", count=2)
    speech = SpeechRequest(text="hello", voice="narrator")

    requests = [
        (MediaRequest.for_image(image), MediaKind.IMAGE, image),
        (MediaRequest.for_video(video), MediaKind.VIDEO, video),
        (MediaRequest.for_music(music), MediaKind.MUSIC, music),
        (MediaRequest.for_speech(speech), MediaKind.SPEECH, speech),
    ]
    for request, kind, payload in requests:
        assert request.kind is kind
        assert request.payload is payload

    with pytest.raises(ValidationError, match="exactly one"):
        MediaRequest(kind=MediaKind.IMAGE)
    with pytest.raises(ValidationError, match="exactly one"):
        MediaRequest(kind=MediaKind.IMAGE, image=image, video=video)
    with pytest.raises(ValidationError, match="does not match"):
        MediaRequest(kind=MediaKind.VIDEO, image=image)
    with pytest.raises(ValidationError):
        MediaRequest.model_validate({"kind": "image", "unknown": True})


def test_media_result_wraps_exactly_one_matching_payload() -> None:
    image = ImageResult(data=b"image")
    video = VideoResult(data=b"video")
    music = MusicResult(data=b"music")
    speech = SpeechResult(audio=b"speech")

    results = [
        (MediaResult.for_image(image), MediaKind.IMAGE, image),
        (MediaResult.for_video(video), MediaKind.VIDEO, video),
        (MediaResult.for_music(music), MediaKind.MUSIC, music),
        (MediaResult.for_speech(speech), MediaKind.SPEECH, speech),
    ]
    for result, kind, payload in results:
        assert result.kind is kind
        assert result.payload is payload

    with pytest.raises(ValidationError, match="exactly one"):
        MediaResult(kind=MediaKind.IMAGE)
    with pytest.raises(ValidationError, match="exactly one"):
        MediaResult(kind=MediaKind.IMAGE, image=image, music=music)
    with pytest.raises(ValidationError, match="does not match"):
        MediaResult(kind=MediaKind.MUSIC, image=image)


def test_media_dtos_are_immutable() -> None:
    request = MediaRequest.for_image(ImageRequest(prompt="a fox"))

    with pytest.raises(ValidationError):
        request.kind = MediaKind.VIDEO
    with pytest.raises(ValidationError):
        request.image = None
