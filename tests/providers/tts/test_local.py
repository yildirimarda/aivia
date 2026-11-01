"""Local Piper, Kokoro and command text to speech adapters."""

from __future__ import annotations

import asyncio
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
