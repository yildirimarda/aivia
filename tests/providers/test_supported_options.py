"""Every media adapter declares the request options it can consume."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from aivia.core.models.provider import ProviderMeta
from aivia.providers import Provider
from aivia.providers.image import (
    FalImageProvider,
    GoogleImageProvider,
    OpenAIImageProvider,
    ReplicateImageProvider,
)
from aivia.providers.media import (
    MediaConstraints,
    UnsupportedOptionError,
    adapt_request,
)
from aivia.providers.music import (
    ElevenLabsMusicProvider,
    LocalMusicStubProvider,
    StableAudioMusicProvider,
)
from aivia.providers.protocols import (
    CAPABILITY_IMAGE_GEN,
    CAPABILITY_MUSIC_GEN,
    CAPABILITY_TTS,
    CAPABILITY_VIDEO_GEN,
    AudioFormat,
    ImageFormat,
    ImageRequest,
    MediaInput,
    MusicRequest,
    TTSRequest,
    VideoFormat,
    VideoRequest,
)
from aivia.providers.tts import (
    AzureTTSProvider,
    ElevenLabsTTSProvider,
    GoogleTTSProvider,
    KokoroTTSProvider,
    LocalTTSProvider,
    OpenAITTSProvider,
    PiperTTSProvider,
)
from aivia.providers.video import (
    FalVideoProvider,
    ReplicateVideoProvider,
    VeoVideoProvider,
)

_FRAME_URL = "https://example.com/media/frame.png"


@dataclass(frozen=True)
class AdapterOptions:
    """One media adapter and the option names it promises to handle."""

    name: str
    provider: type[Provider]
    capability: str
    options: tuple[str, ...]


_ADAPTERS: tuple[AdapterOptions, ...] = (
    AdapterOptions(
        "image/fal",
        FalImageProvider,
        CAPABILITY_IMAGE_GEN,
        (
            "prompt",
            "negative_prompt",
            "size",
            "aspect_ratio",
            "format",
            "quality",
            "style",
            "seed",
            "count",
            "model",
        ),
    ),
    AdapterOptions(
        "image/google",
        GoogleImageProvider,
        CAPABILITY_IMAGE_GEN,
        (
            "prompt",
            "negative_prompt",
            "size",
            "aspect_ratio",
            "format",
            "seed",
            "count",
            "model",
        ),
    ),
    AdapterOptions(
        "image/openai",
        OpenAIImageProvider,
        CAPABILITY_IMAGE_GEN,
        (
            "prompt",
            "size",
            "aspect_ratio",
            "format",
            "quality",
            "style",
            "seed",
            "count",
            "model",
        ),
    ),
    AdapterOptions(
        "image/replicate",
        ReplicateImageProvider,
        CAPABILITY_IMAGE_GEN,
        (
            "prompt",
            "negative_prompt",
            "size",
            "aspect_ratio",
            "format",
            "quality",
            "style",
            "seed",
            "count",
            "model",
        ),
    ),
    AdapterOptions(
        "video/fal",
        FalVideoProvider,
        CAPABILITY_VIDEO_GEN,
        (
            "prompt",
            "negative_prompt",
            "first_frame",
            "size",
            "aspect_ratio",
            "format",
            "fps",
            "seed",
            "duration",
            "count",
            "model",
            "generate_audio",
            "audio_prompt",
        ),
    ),
    AdapterOptions(
        "video/replicate",
        ReplicateVideoProvider,
        CAPABILITY_VIDEO_GEN,
        (
            "prompt",
            "negative_prompt",
            "first_frame",
            "size",
            "aspect_ratio",
            "format",
            "fps",
            "seed",
            "duration",
            "count",
            "model",
            "generate_audio",
            "audio_prompt",
        ),
    ),
    AdapterOptions(
        "video/veo",
        VeoVideoProvider,
        CAPABILITY_VIDEO_GEN,
        (
            "prompt",
            "negative_prompt",
            "first_frame",
            "size",
            "aspect_ratio",
            "format",
            "seed",
            "duration",
            "count",
            "model",
            "generate_audio",
        ),
    ),
    AdapterOptions(
        "music/elevenlabs",
        ElevenLabsMusicProvider,
        CAPABILITY_MUSIC_GEN,
        (
            "prompt",
            "lyrics",
            "instrumental",
            "duration",
            "format",
            "sample_rate",
            "seed",
            "count",
            "model",
        ),
    ),
    AdapterOptions(
        "music/local",
        LocalMusicStubProvider,
        CAPABILITY_MUSIC_GEN,
        (
            "prompt",
            "lyrics",
            "instrumental",
            "duration",
            "format",
            "sample_rate",
            "seed",
            "count",
        ),
    ),
    AdapterOptions(
        "music/stable-audio",
        StableAudioMusicProvider,
        CAPABILITY_MUSIC_GEN,
        (
            "prompt",
            "instrumental",
            "duration",
            "format",
            "sample_rate",
            "seed",
            "count",
            "model",
        ),
    ),
    AdapterOptions(
        "tts/azure",
        AzureTTSProvider,
        CAPABILITY_TTS,
        (
            "text",
            "model",
            "language",
            "format",
            "speed",
            "pronunciation",
        ),
    ),
    AdapterOptions(
        "tts/elevenlabs",
        ElevenLabsTTSProvider,
        CAPABILITY_TTS,
        (
            "text",
            "model",
            "language",
            "format",
            "sample_rate",
            "speed",
            "word_timestamps",
        ),
    ),
    AdapterOptions(
        "tts/google",
        GoogleTTSProvider,
        CAPABILITY_TTS,
        (
            "text",
            "model",
            "language",
            "format",
            "sample_rate",
            "speed",
            "pronunciation",
        ),
    ),
    AdapterOptions(
        "tts/kokoro",
        KokoroTTSProvider,
        CAPABILITY_TTS,
        ("text", "model", "format", "sample_rate", "speed"),
    ),
    AdapterOptions(
        "tts/local",
        LocalTTSProvider,
        CAPABILITY_TTS,
        ("text", "model", "format", "sample_rate", "speed"),
    ),
    AdapterOptions(
        "tts/openai",
        OpenAITTSProvider,
        CAPABILITY_TTS,
        ("text", "model", "format", "sample_rate", "speed", "instructions"),
    ),
    AdapterOptions(
        "tts/piper",
        PiperTTSProvider,
        CAPABILITY_TTS,
        ("text", "model", "format", "sample_rate", "speed"),
    ),
)

_OPTION_VALUES: dict[str, dict[str, Any]] = {
    "prompt": {"prompt": "a test prompt"},
    "text": {"text": "a test sentence"},
    "negative_prompt": {"negative_prompt": "blurry"},
    "size": {"width": 1024, "height": 1024},
    "aspect_ratio": {"aspect_ratio": "16:9"},
    "format": {},
    "quality": {"quality": "high"},
    "style": {"style": "cinematic"},
    "seed": {"seed": 7},
    "count": {"count": 2},
    "model": {"model": "test-model"},
    "duration": {"duration_seconds": 5.0},
    "fps": {"fps": 24},
    "first_frame": {"first_frame": MediaInput(url=_FRAME_URL)},
    "last_frame": {"last_frame": MediaInput(url=_FRAME_URL)},
    "reference_images": {"reference_images": [MediaInput(url=_FRAME_URL)]},
    "generate_audio": {"generate_audio": True},
    "audio_prompt": {"generate_audio": True, "audio_prompt": "wind"},
    "lyrics": {"lyrics": "la la la"},
    "instrumental": {"instrumental": True},
    "tempo_bpm": {"tempo_bpm": 120.0},
    "sample_rate": {"sample_rate": 24_000},
    "language": {"language": "en-US"},
    "speed": {"speed": 1.5},
    "instructions": {"instructions": "read calmly"},
    "pronunciation": {"pronunciation": {"aivia": "ah-vee-ah"}},
    "word_timestamps": {"word_timestamps": True},
}

_FORMATS: dict[str, Any] = {
    CAPABILITY_IMAGE_GEN: ImageFormat.PNG,
    CAPABILITY_VIDEO_GEN: VideoFormat.MP4,
    CAPABILITY_MUSIC_GEN: AudioFormat.MP3,
    CAPABILITY_TTS: AudioFormat.MP3,
}


def _metadata(case: AdapterOptions) -> ProviderMeta:
    metadata = case.provider.default_metadata
    assert isinstance(metadata, ProviderMeta)
    return metadata


def _request(case: AdapterOptions, option: str) -> Any:
    values = dict(_OPTION_VALUES[option])
    if option == "format":
        values["format"] = _FORMATS[case.capability]
    if case.capability == CAPABILITY_IMAGE_GEN:
        values.setdefault("prompt", "a test prompt")
        return ImageRequest(**values)
    if case.capability == CAPABILITY_VIDEO_GEN:
        values.setdefault("prompt", "a test prompt")
        return VideoRequest(**values)
    if case.capability == CAPABILITY_MUSIC_GEN:
        values.setdefault("prompt", "a test prompt")
        return MusicRequest(**values)
    values.setdefault("text", "a test sentence")
    values.setdefault("voice", "narrator")
    return TTSRequest(**values)


def _case_id(case: AdapterOptions) -> str:
    return case.name


@pytest.mark.parametrize("case", _ADAPTERS, ids=_case_id)
def test_every_declared_option_is_accepted_by_adaptation(case: AdapterOptions) -> None:
    constraints = MediaConstraints.from_metadata(_metadata(case), case.capability)

    assert constraints.supports is not None

    for option in case.options:
        assert option in constraints.supports

        request = _request(case, option)
        adapted = adapt_request(request, constraints)

        assert adapted.request == request


@pytest.mark.parametrize("case", _ADAPTERS, ids=_case_id)
def test_options_an_adapter_cannot_handle_are_still_rejected(
    case: AdapterOptions,
) -> None:
    constraints = MediaConstraints.from_metadata(_metadata(case), case.capability)
    unsupported = _unsupported_option(case)

    with pytest.raises(UnsupportedOptionError) as caught:
        adapt_request(_request(case, unsupported), constraints)

    assert caught.value.option == unsupported


def _unsupported_option(case: AdapterOptions) -> str:
    if case.capability == CAPABILITY_IMAGE_GEN:
        return "reference_images" if "reference_images" not in case.options else "style"
    if case.capability == CAPABILITY_VIDEO_GEN:
        return "last_frame"
    if case.capability == CAPABILITY_MUSIC_GEN:
        return "tempo_bpm"
    return "instructions" if "instructions" not in case.options else "pronunciation"
