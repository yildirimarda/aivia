"""Local Piper, Kokoro and command text to speech adapters."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

from aivia.core.errors import ConfigError, InvalidRequestError, ValidationError
from aivia.providers import HealthStatus, provider_registry
from aivia.providers.protocols import CAPABILITY_TTS, AudioFormat, TTSRequest
from aivia.providers.tts import (
    Kokoro,
    KokoroTTS,
    KokoroTTSProvider,
    Local,
    LocalTTS,
    LocalTTSProvider,
    Piper,
    PiperTTS,
    PiperTTSProvider,
    TTSCostCalculator,
)

WAV = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 20
MP3 = b"ID3\x04\x00\x00\x00\x00\x00\x00converted"


def test_metadata_and_registration_describe_the_providers() -> None:
    provider = LocalTTSProvider(synthesizer=lambda *args: WAV)

    assert provider.name == "local-tts"
    assert provider.capabilities == (CAPABILITY_TTS,)
    assert provider.local_backend == "piper"
    assert provider.format is AudioFormat.WAV
    assert provider.sample_rate == 22050
    assert provider.voice_map.resolve("narrator") == "narrator"
    assert Local is LocalTTSProvider
    assert LocalTTS is LocalTTSProvider
    assert Piper is PiperTTSProvider
    assert PiperTTS is PiperTTSProvider
    assert Kokoro is KokoroTTSProvider
    assert KokoroTTS is KokoroTTSProvider
    assert provider_registry.resolve("local-tts") is LocalTTSProvider
    assert provider_registry.resolve("piper-tts") is PiperTTSProvider
    assert provider_registry.resolve("kokoro-tts") is KokoroTTSProvider
    assert PiperTTSProvider().name == "piper-tts"
    assert KokoroTTSProvider().name == "kokoro-tts"


def test_an_injected_synthesizer_receives_the_resolved_voice_and_settings() -> None:
    calls: list[tuple[str, str, str, AudioFormat, int]] = []

    def synthesizer(
        text: str,
        voice: str,
        model: str,
        audio_format: AudioFormat,
        sample_rate: int,
    ) -> bytes:
        calls.append((text, voice, model, audio_format, sample_rate))
        return WAV

    provider = LocalTTSProvider(
        voice_map={"narrator": "en_US-amy-medium"},
        default_model="model.onnx",
        synthesizer=synthesizer,
    )

    response = asyncio.run(
        provider.synthesize(
            TTSRequest(
                text="Hello local voice",
                voice="narrator",
                format=AudioFormat.WAV,
                sample_rate=16000,
            )
        )
    )

    assert calls == [
        ("Hello local voice", "en_US-amy-medium", "model.onnx", AudioFormat.WAV, 16000)
    ]
    assert response.audio == WAV
    assert response.format is AudioFormat.WAV
    assert response.voice == "en_US-amy-medium"
    assert response.model == "model.onnx"
    assert response.provider == "local-tts"
    assert response.metadata["characters"] == len("Hello local voice")
    assert response.metadata["backend"] == "piper"
    assert TTSCostCalculator().cost_for(provider, 17).amount_usd == 0.0


def test_a_custom_command_reads_stdin_and_writes_an_output_file() -> None:
    script = (
        "import pathlib, sys\n"
        "data = sys.stdin.buffer.read()\n"
        "pathlib.Path(sys.argv[1]).write_bytes("
        "b'RIFF\\x24\\x00\\x00\\x00WAVEfmt ' + data)\n"
    )
    provider = LocalTTSProvider(
        backend="command",
        command=[sys.executable, "-c", script, "{output}"],
    )

    response = asyncio.run(
        provider.synthesize(
            TTSRequest(text="spoken text", voice="default", format=AudioFormat.WAV)
        )
    )

    assert response.audio.startswith(b"RIFF")
    assert response.audio[8:12] == b"WAVE"
    assert response.format is AudioFormat.WAV
    assert response.metadata["characters"] == len("spoken text")


def test_an_injected_converter_transcodes_to_the_requested_container() -> None:
    calls: list[tuple[AudioFormat, AudioFormat, int]] = []

    def converter(
        audio: bytes,
        source: AudioFormat,
        target: AudioFormat,
        sample_rate: int,
    ) -> bytes:
        assert audio == WAV
        calls.append((source, target, sample_rate))
        return MP3

    provider = LocalTTSProvider(
        synthesizer=lambda *args: WAV,
        converter=converter,
    )

    response = asyncio.run(
        provider.synthesize(
            TTSRequest(text="spoken", voice="default", format=AudioFormat.MP3)
        )
    )

    assert calls == [(AudioFormat.WAV, AudioFormat.MP3, 22050)]
    assert response.audio == MP3
    assert response.format is AudioFormat.MP3


def test_an_injected_encoder_transcodes_to_the_requested_container() -> None:
    calls: list[tuple[AudioFormat, int]] = []

    def encoder(audio: bytes, target: AudioFormat, sample_rate: int) -> bytes:
        assert audio == WAV
        calls.append((target, sample_rate))
        return b"OggS\x00\x00converted"

    provider = LocalTTSProvider(
        synthesizer=lambda *args: WAV,
        encoder=encoder,
    )

    response = asyncio.run(
        provider.synthesize(
            TTSRequest(
                text="spoken",
                voice="default",
                format=AudioFormat.OGG,
                sample_rate=16000,
            )
        )
    )

    assert calls == [(AudioFormat.OGG, 16000)]
    assert response.format is AudioFormat.OGG


def test_a_matching_container_is_not_converted() -> None:
    def converter(*args: object) -> bytes:
        raise AssertionError("a matching container must not be converted")

    provider = LocalTTSProvider(
        synthesizer=lambda *args: WAV,
        converter=converter,
    )

    response = asyncio.run(
        provider.synthesize(
            TTSRequest(text="spoken", voice="default", format=AudioFormat.WAV)
        )
    )

    assert response.audio == WAV
    assert response.format is AudioFormat.WAV


def test_raw_pcm_output_matching_the_request_is_not_converted() -> None:
    raw = b"\x00\x01" * 16
    provider = LocalTTSProvider(synthesizer=lambda *args: raw)

    response = asyncio.run(
        provider.synthesize(
            TTSRequest(text="spoken", voice="default", format=AudioFormat.PCM)
        )
    )

    assert response.audio == raw
    assert response.format is AudioFormat.PCM


def test_a_conversion_without_ffmpeg_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("aivia.providers.tts.local.shutil.which", lambda _: None)
    provider = LocalTTSProvider(synthesizer=lambda *args: WAV)

    with pytest.raises(ValidationError, match="ffmpeg"):
        asyncio.run(
            provider.synthesize(
                TTSRequest(text="spoken", voice="default", format=AudioFormat.MP3)
            )
        )


def test_a_conversion_runs_ffmpeg_with_the_target_codec() -> None:
    calls: list[list[str]] = []

    def runner(argv: list[str]) -> subprocess.CompletedProcess[bytes]:
        calls.append(argv)
        Path(argv[-1]).write_bytes(MP3)
        return subprocess.CompletedProcess(argv, 0, stdout=b"", stderr=b"")

    provider = LocalTTSProvider(
        synthesizer=lambda *args: WAV,
        ffmpeg_runner=runner,
    )

    response = asyncio.run(
        provider.synthesize(
            TTSRequest(
                text="spoken",
                voice="default",
                format=AudioFormat.MP3,
                sample_rate=16000,
            )
        )
    )

    assert response.audio == MP3
    assert response.format is AudioFormat.MP3
    argv = calls[0]
    assert argv[argv.index("-c:a") + 1] == "libmp3lame"
    assert argv[argv.index("-ar") + 1] == "16000"
    assert any(part.endswith("source.wav") for part in argv)


def test_a_failing_ffmpeg_conversion_is_a_validation_error() -> None:
    def runner(argv: list[str]) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(argv, 1, stdout=b"", stderr=b"boom")

    provider = LocalTTSProvider(
        synthesizer=lambda *args: WAV,
        ffmpeg_runner=runner,
    )

    with pytest.raises(ValidationError, match="conversion failed"):
        asyncio.run(
            provider.synthesize(
                TTSRequest(text="spoken", voice="default", format=AudioFormat.MP3)
            )
        )


def test_an_unrecognised_source_is_rejected_as_an_unsupported_conversion() -> None:
    provider = LocalTTSProvider(synthesizer=lambda *args: b"not audio")

    with pytest.raises(InvalidRequestError, match="does not support converting"):
        asyncio.run(
            provider.synthesize(
                TTSRequest(text="spoken", voice="default", format=AudioFormat.MP3)
            )
        )


def test_piper_without_a_model_is_rejected() -> None:
    provider = LocalTTSProvider()

    with pytest.raises(InvalidRequestError, match="model"):
        asyncio.run(
            provider.synthesize(
                TTSRequest(text="hi", voice="default", format=AudioFormat.WAV)
            )
        )


def test_a_failing_command_is_reported_as_a_validation_error() -> None:
    provider = LocalTTSProvider(
        backend="command",
        command=[sys.executable, "-c", "import sys; sys.exit(3)"],
    )

    with pytest.raises(ValidationError, match="command failed"):
        asyncio.run(
            provider.synthesize(
                TTSRequest(text="hi", voice="default", format=AudioFormat.WAV)
            )
        )


def test_a_missing_executable_is_reported_as_a_validation_error() -> None:
    provider = LocalTTSProvider(
        backend="command",
        command=["aivia-no-such-tts-binary"],
    )

    with pytest.raises(ValidationError, match="not found"):
        asyncio.run(
            provider.synthesize(
                TTSRequest(text="hi", voice="default", format=AudioFormat.WAV)
            )
        )


def test_an_unknown_backend_and_bad_command_are_rejected() -> None:
    with pytest.raises(ConfigError, match="unknown local"):
        LocalTTSProvider(backend="unknown")
    with pytest.raises(ValueError, match="must not be empty"):
        LocalTTSProvider(command=[])
    with pytest.raises(TypeError, match="command"):
        LocalTTSProvider(command=123)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="encoder"):
        LocalTTSProvider(encoder=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="converter"):
        LocalTTSProvider(converter=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ffmpeg_runner"):
        LocalTTSProvider(ffmpeg_runner=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ffmpeg_executable"):
        LocalTTSProvider(ffmpeg_executable=" ")


def test_command_placeholders_are_expanded_without_a_shell() -> None:
    provider = LocalTTSProvider(
        backend="command",
        command=["echo", "{text}", "{voice}", "{model}", "{format}", "{sample_rate}"],
    )

    assert provider.command == (
        "echo",
        "{text}",
        "{voice}",
        "{model}",
        "{format}",
        "{sample_rate}",
    )


def test_health_probes_the_executable_and_model(tmp_path: Path) -> None:
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"model")
    provider = LocalTTSProvider(
        executable=sys.executable,
        default_model=str(model),
    )

    report = asyncio.run(provider.health())

    assert report.status is HealthStatus.HEALTHY
    assert report.details["executable"] == sys.executable
    assert report.details["model"] == str(model)


def test_health_reports_a_missing_executable() -> None:
    provider = LocalTTSProvider(executable="aivia-no-such-tts-binary")

    report = asyncio.run(provider.health())

    assert report.status is HealthStatus.UNHEALTHY
    assert "not found" in report.message
    assert report.details["executable"] == "aivia-no-such-tts-binary"


def test_health_reports_a_missing_model(tmp_path: Path) -> None:
    missing = tmp_path / "missing.onnx"
    provider = LocalTTSProvider(
        executable=sys.executable,
        default_model=str(missing),
    )

    report = asyncio.run(provider.health())

    assert report.status is HealthStatus.UNHEALTHY
    assert "model" in report.message
    assert report.details["model"] == str(missing)


def test_health_uses_the_command_executable() -> None:
    provider = LocalTTSProvider(
        backend="command",
        command=["aivia-no-such-tts-binary", "{output}"],
    )

    report = asyncio.run(provider.health())

    assert report.status is HealthStatus.UNHEALTHY
    assert report.details["executable"] == "aivia-no-such-tts-binary"


def test_health_reports_a_command_backend_without_a_command() -> None:
    provider = LocalTTSProvider(backend="command")

    report = asyncio.run(provider.health())

    assert report.status is HealthStatus.UNHEALTHY
    assert "command" in report.message


def test_health_skips_the_probe_for_an_injected_synthesizer() -> None:
    provider = LocalTTSProvider(
        executable="aivia-no-such-tts-binary",
        synthesizer=lambda *args: WAV,
    )

    report = asyncio.run(provider.health())

    assert report.status is HealthStatus.HEALTHY
    assert report.details["synthesizer"] is True


def test_health_accepts_an_opaque_model_id_for_kokoro() -> None:
    provider = KokoroTTSProvider(
        executable=sys.executable,
        default_model="kokoro-v0_19",
    )

    report = asyncio.run(provider.health())

    assert report.status is HealthStatus.HEALTHY


def test_health_reports_a_missing_kokoro_model_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing.onnx"
    provider = KokoroTTSProvider(
        executable=sys.executable,
        default_model=str(missing),
    )

    report = asyncio.run(provider.health())

    assert report.status is HealthStatus.UNHEALTHY
    assert report.details["model"] == str(missing)


@pytest.mark.parametrize(
    "audio_format",
    [AudioFormat.MP3, AudioFormat.OGG, AudioFormat.PCM],
)
def test_health_reports_a_missing_ffmpeg_for_a_converted_default_output(
    monkeypatch: pytest.MonkeyPatch,
    audio_format: AudioFormat,
) -> None:
    def which(name: str) -> str | None:
        return None if name == "ffmpeg" else name

    monkeypatch.setattr("aivia.providers.tts.local.shutil.which", which)
    provider = LocalTTSProvider(
        executable=sys.executable,
        format=audio_format,
    )

    report = asyncio.run(provider.health())

    assert report.status is HealthStatus.UNHEALTHY
    assert "ffmpeg" in report.message
    assert "not found" in report.message
    assert report.details["ffmpeg"] == "ffmpeg"
    assert report.details["ffmpeg_executable"] == "ffmpeg"
    assert report.details["format"] == audio_format.value


def test_health_accepts_a_present_ffmpeg_for_a_converted_default_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "aivia.providers.tts.local.shutil.which",
        lambda name: name,
    )
    provider = LocalTTSProvider(
        executable=sys.executable,
        format=AudioFormat.MP3,
    )

    report = asyncio.run(provider.health())

    assert report.status is HealthStatus.HEALTHY
    assert report.details["ffmpeg"] == "ffmpeg"
    assert report.details["ffmpeg_executable"] == "ffmpeg"
    assert report.details["format"] == "mp3"


def test_health_does_not_probe_ffmpeg_for_the_default_wav_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probed: list[str] = []

    def which(name: str) -> str | None:
        probed.append(name)
        return name

    monkeypatch.setattr("aivia.providers.tts.local.shutil.which", which)
    provider = LocalTTSProvider(executable=sys.executable)

    report = asyncio.run(provider.health())

    assert report.status is HealthStatus.HEALTHY
    assert probed == [sys.executable]
    assert "ffmpeg" not in report.details


def test_health_skips_the_ffmpeg_probe_when_conversion_is_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def which(name: str) -> str | None:
        return None if name == "ffmpeg" else name

    monkeypatch.setattr("aivia.providers.tts.local.shutil.which", which)

    def converter(
        audio: bytes,
        source: AudioFormat,
        target: AudioFormat,
        sample_rate: int,
    ) -> bytes:
        return MP3

    def encoder(audio: bytes, target: AudioFormat, sample_rate: int) -> bytes:
        return MP3

    def runner(argv: list[str]) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(argv, 0, stdout=MP3, stderr=b"")

    providers = [
        LocalTTSProvider(
            executable=sys.executable,
            format=AudioFormat.MP3,
            converter=converter,
        ),
        LocalTTSProvider(
            executable=sys.executable,
            format=AudioFormat.MP3,
            encoder=encoder,
        ),
        LocalTTSProvider(
            executable=sys.executable,
            format=AudioFormat.MP3,
            ffmpeg_runner=runner,
        ),
    ]

    for provider in providers:
        report = asyncio.run(provider.health())

        assert report.status is HealthStatus.HEALTHY
        assert "ffmpeg" not in report.details
