"""Local music stub: deterministic audio and public-domain licensing metadata."""

from __future__ import annotations

import asyncio
import io
import wave

import pytest

from aivia.core.errors import ValidationError
from aivia.providers import provider_registry
from aivia.providers.music import (
    DEFAULT_DURATION_SECONDS,
    DEFAULT_SAMPLE_RATE,
    LocalMusic,
    LocalMusicProvider,
    LocalMusicStub,
    LocalMusicStubProvider,
)
from aivia.providers.protocols import CAPABILITY_MUSIC_GEN, AudioFormat, MusicRequest


def _wav_duration(data: bytes) -> float:
    with wave.open(io.BytesIO(data), "rb") as reader:
        return reader.getnframes() / reader.getframerate()


def test_metadata_and_registration_record_commercial_use() -> None:
    provider = LocalMusicStubProvider()

    assert provider.name == "local-music"
    assert provider.capabilities == (CAPABILITY_MUSIC_GEN,)
    assert provider.default_duration_seconds == DEFAULT_DURATION_SECONDS
    assert provider.sample_rate == DEFAULT_SAMPLE_RATE
    assert provider.metadata.is_commercial is True
    assert provider.metadata.license.name == "CC0-1.0"
    assert provider.metadata.pricing[CAPABILITY_MUSIC_GEN].amount_usd == 0.0
    assert LocalMusic is LocalMusicStubProvider
    assert LocalMusicProvider is LocalMusicStubProvider
    assert LocalMusicStub is LocalMusicStubProvider
    assert provider_registry.resolve("local-music") is LocalMusicStubProvider
    assert provider_registry.resolve("local_music") is LocalMusicStubProvider


def test_it_generates_a_valid_wav_with_the_requested_duration() -> None:
    provider = LocalMusicStubProvider()
    request = MusicRequest(
        prompt="calm loop",
        duration_seconds=2.0,
        format=AudioFormat.WAV,
        sample_rate=8_000,
        lyrics="la la",
        seed=3,
        count=2,
    )

    response = asyncio.run(provider.generate(request))

    assert response.count == 2
    assert response.model == "local-stub"
    assert response.provider == "local-music"
    assert response.metadata["commercial_use"] is True
    assert response.has_lyrics is True
    for track in response.tracks:
        assert track.format is AudioFormat.WAV
        assert track.duration_seconds == pytest.approx(2.0)
        assert track.sample_rate == 8_000
        assert track.channels == 1
        assert track.lyrics == "la la"
        assert _wav_duration(track.data or b"") == pytest.approx(2.0, abs=1e-3)
    assert [track.seed for track in response.tracks] == [3, 4]


def test_the_same_prompt_and_seed_are_deterministic() -> None:
    provider = LocalMusicStubProvider()
    request = MusicRequest(
        prompt="same prompt",
        duration_seconds=1.0,
        format=AudioFormat.WAV,
        sample_rate=8_000,
        instrumental=True,
        seed=11,
    )

    first = asyncio.run(provider.generate(request))
    second = asyncio.run(provider.generate(request))

    assert first.first.data == second.first.data
    assert first.first.lyrics is None


def test_an_injected_encoder_handles_a_non_wav_format() -> None:
    calls: list[tuple[AudioFormat, int]] = []

    def encoder(wav: bytes, audio_format: AudioFormat, sample_rate: int) -> bytes:
        calls.append((audio_format, sample_rate))
        assert wav.startswith(b"RIFF")
        return b"ID3\x04\x00\x00\x00\x00\x00\x00encoded"

    provider = LocalMusicStubProvider(encoder=encoder)
    request = MusicRequest(
        prompt="encoded",
        duration_seconds=1.0,
        format=AudioFormat.MP3,
        sample_rate=8_000,
        instrumental=True,
    )

    response = asyncio.run(provider.generate(request))

    assert calls == [(AudioFormat.MP3, 8_000)]
    assert response.first.format is AudioFormat.MP3
    assert response.first.data == b"ID3\x04\x00\x00\x00\x00\x00\x00encoded"


def test_mp3_falls_back_to_silent_frames_without_ffmpeg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("aivia.providers.music.local.shutil.which", lambda _: None)
    provider = LocalMusicStubProvider()
    request = MusicRequest(prompt="encoded", duration_seconds=1.0, sample_rate=8_000)

    response = asyncio.run(provider.generate(request))

    assert response.first.format is AudioFormat.MP3
    assert (response.first.data or b"").startswith(b"\xff\xfb")


def test_other_non_wav_output_without_ffmpeg_is_a_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("aivia.providers.music.local.shutil.which", lambda _: None)
    provider = LocalMusicStubProvider()
    request = MusicRequest(
        prompt="encoded",
        duration_seconds=1.0,
        format=AudioFormat.OGG,
        sample_rate=8_000,
        instrumental=True,
    )

    with pytest.raises(ValidationError, match="ffmpeg"):
        asyncio.run(provider.generate(request))


def test_wrong_request_type_and_invalid_settings_are_rejected() -> None:
    provider = LocalMusicStubProvider()

    with pytest.raises(TypeError, match="MusicRequest"):
        asyncio.run(provider.generate("not a request"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive"):
        LocalMusicStubProvider(default_duration_seconds=0)
    with pytest.raises(ValueError, match="positive"):
        LocalMusicStubProvider(sample_rate=0)
    with pytest.raises(TypeError, match="callable"):
        LocalMusicStubProvider(encoder=object())  # type: ignore[arg-type]


def test_the_default_duration_is_used_when_the_request_omits_one() -> None:
    provider = LocalMusicStubProvider(default_duration_seconds=1.5, sample_rate=8_000)
    request = MusicRequest(
        prompt="default",
        format=AudioFormat.WAV,
        instrumental=True,
    )

    response = asyncio.run(provider.generate(request))

    assert response.first.duration_seconds == pytest.approx(1.5)
    assert _wav_duration(response.first.data or b"") == pytest.approx(1.5, abs=1e-3)
