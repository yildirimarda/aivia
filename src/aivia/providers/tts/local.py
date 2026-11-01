"""Local text to speech adapters for Piper, Kokoro and custom commands.

A local synthesizer runs on the machine and costs nothing, so it is the natural
fallback when no hosted voice is configured. The adapter never uses a shell: a
configured command is split into an argv list, text is written to stdin unless
the command names a ``{text}`` placeholder, and the produced audio is read from
the ``{output}`` file or stdout. When the produced container differs from the
requested one, it is converted with ffmpeg; an unrecognised source or a missing
encoder is rejected rather than returned under the wrong format. Tests inject a
``synthesizer`` or ``encoder`` callable so the offline suite never needs Piper,
Kokoro or ffmpeg installed.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from aivia.core.config.http import HttpClientConfig
from aivia.core.errors import ConfigError, InvalidRequestError, ValidationError
from aivia.core.models.provider import ProviderMeta, ProviderPrice
from aivia.providers.base import HealthStatus, Provider, ProviderHealth
from aivia.providers.protocols.audio import AudioFormat
from aivia.providers.protocols.tts import CAPABILITY_TTS, TTSRequest, TTSResponse
from aivia.providers.registry import register_provider
from aivia.providers.tts.common import detect_audio_format
from aivia.providers.tts.voice import VoiceMap

__all__ = [
    "DEFAULT_SAMPLE_RATE",
    "Converter",
    "Encoder",
    "FFmpegRunner",
    "Kokoro",
    "KokoroTTS",
    "KokoroTTSProvider",
    "Local",
    "LocalTTS",
    "LocalTTSProvider",
    "Piper",
    "PiperTTS",
    "PiperTTSProvider",
    "Synthesizer",
]

DEFAULT_SAMPLE_RATE = 22050
"""The default sample rate requested from a local synthesizer."""

Synthesizer = Callable[[str, str, str, AudioFormat, int], bytes]
"""A callable taking text, voice, model, format and sample rate."""

Encoder = Callable[[bytes, AudioFormat, int], bytes]
"""A callable taking source audio, the target format and the sample rate."""

Converter = Callable[[bytes, AudioFormat, AudioFormat, int], bytes]
"""A callable taking source audio, source format, target format and sample rate."""

FFmpegRunner = Callable[[list[str]], subprocess.CompletedProcess[bytes]]
"""A callable running an ffmpeg argv and returning its completed process."""

_FFMPEG_CODECS: dict[AudioFormat, str] = {
    AudioFormat.MP3: "libmp3lame",
    AudioFormat.WAV: "pcm_s16le",
    AudioFormat.OGG: "libvorbis",
    AudioFormat.OPUS: "libopus",
    AudioFormat.AAC: "aac",
    AudioFormat.FLAC: "flac",
    AudioFormat.PCM: "pcm_s16le",
    AudioFormat.M4A: "aac",
}
"""The ffmpeg audio encoder used for each requested output container."""


_DEFAULT_METADATA = ProviderMeta(
    name="local-tts",
    capabilities=[CAPABILITY_TTS],
    supported_options={
        CAPABILITY_TTS: ["text", "model", "format", "sample_rate", "speed"]
    },
    pricing={CAPABILITY_TTS: ProviderPrice(amount_usd=0.0, unit="character")},
)

_PIPER_METADATA = ProviderMeta(
    name="piper-tts",
    capabilities=[CAPABILITY_TTS],
    supported_options={
        CAPABILITY_TTS: ["text", "model", "format", "sample_rate", "speed"]
    },
    pricing={CAPABILITY_TTS: ProviderPrice(amount_usd=0.0, unit="character")},
)

_KOKORO_METADATA = ProviderMeta(
    name="kokoro-tts",
    capabilities=[CAPABILITY_TTS],
    supported_options={
        CAPABILITY_TTS: ["text", "model", "format", "sample_rate", "speed"]
    },
    pricing={CAPABILITY_TTS: ProviderPrice(amount_usd=0.0, unit="character")},
)


def _command(value: str | Sequence[str] | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        parts = shlex.split(value)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        parts = list(value)
    else:
        raise TypeError("command must be a string, a sequence of strings or None")
    if not parts:
        raise ValueError("command must not be empty")
    if not all(isinstance(part, str) and part for part in parts):
        raise ValueError("command entries must be non-empty strings")
    return parts


def _looks_like_model_path(value: str) -> bool:
    """Return whether *value* names a file rather than an opaque model id."""
    path = Path(value).expanduser()
    return (
        path.is_absolute()
        or value.startswith(".")
        or os.sep in value
        or (os.altsep is not None and os.altsep in value)
        or bool(path.suffix)
    )


class LocalTTSProvider(Provider):
    """A text to speech provider that runs a local command or callable.

    ``backend`` selects the default command shape: ``piper``, ``kokoro`` or
    ``command``. ``command`` overrides the default argv and may use the
    ``{text}``, ``{voice}``, ``{model}``, ``{format}``, ``{sample_rate}`` and
    ``{output}`` placeholders. ``synthesizer`` replaces command execution
    entirely, which is how tests stay offline. ``model`` is the Piper voice
    model path or the Kokoro model id.
    """

    default_metadata = _DEFAULT_METADATA
    backend = "piper"
    default_format = AudioFormat.WAV
    default_sample_rate = DEFAULT_SAMPLE_RATE

    def __init__(
        self,
        *,
        backend: str | None = None,
        command: str | Sequence[str] | None = None,
        executable: str | None = None,
        default_model: str | None = None,
        model: str | None = None,
        voice_map: Mapping[str, str] | VoiceMap | None = None,
        default_voice: str | None = None,
        format: AudioFormat | None = None,
        sample_rate: int | None = None,
        synthesizer: Synthesizer | None = None,
        encoder: Encoder | None = None,
        converter: Converter | None = None,
        ffmpeg_executable: str = "ffmpeg",
        ffmpeg_runner: FFmpegRunner | None = None,
        timeout_seconds: float = 60.0,
        name: str | None = None,
        health_timeout_seconds: float | None = None,
        close_timeout_seconds: float | None = None,
        http_config: HttpClientConfig | None = None,
    ) -> None:
        super().__init__(
            name=name,
            health_timeout_seconds=health_timeout_seconds,
            close_timeout_seconds=close_timeout_seconds,
            http_config=http_config,
        )
        selected_backend = (backend or self.backend).strip().lower()
        if selected_backend not in {"piper", "kokoro", "command"}:
            raise ConfigError(
                f"unknown local text to speech backend: {selected_backend!r}",
                details={"field": "backend"},
            )
        self._backend = selected_backend
        self._command = _command(command)
        if executable is not None and not isinstance(executable, str):
            raise TypeError("executable must be a string or None")
        self._executable = executable.strip() if executable else None
        selected_model = model if model is not None else default_model
        if selected_model is not None:
            if not isinstance(selected_model, str):
                raise TypeError("model must be a string or None")
            selected_model = selected_model.strip() or None
        self._default_model = selected_model
        self._voices = VoiceMap.from_value(voice_map, default=default_voice)
        if format is not None and not isinstance(format, AudioFormat):
            raise TypeError("format must be an AudioFormat or None")
        self._format = format or self.default_format
        if sample_rate is None:
            sample_rate = self.default_sample_rate
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
            raise TypeError("sample_rate must be an integer")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self._sample_rate = sample_rate
        if synthesizer is not None and not callable(synthesizer):
            raise TypeError("synthesizer must be callable or None")
        self._synthesizer = synthesizer
        if encoder is not None and not callable(encoder):
            raise TypeError("encoder must be callable or None")
        self._encoder = encoder
        if converter is not None and not callable(converter):
            raise TypeError("converter must be callable or None")
        self._converter = converter
        if not isinstance(ffmpeg_executable, str) or not ffmpeg_executable.strip():
            raise ValueError("ffmpeg_executable must be a non-empty string")
        self._ffmpeg = ffmpeg_executable.strip()
        if ffmpeg_runner is not None and not callable(ffmpeg_runner):
            raise TypeError("ffmpeg_runner must be callable or None")
        self._ffmpeg_runner = ffmpeg_runner
        if isinstance(timeout_seconds, bool) or not isinstance(
            timeout_seconds, (int, float)
        ):
            raise TypeError("timeout_seconds must be a number")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._timeout_seconds = float(timeout_seconds)

    @property
    def local_backend(self) -> str:
        """The selected local backend name."""
        return self._backend

    @property
    def command(self) -> tuple[str, ...] | None:
        """The configured argv template, if any."""
        return tuple(self._command) if self._command is not None else None

    @property
    def default_model(self) -> str | None:
        """The model path or id used when a request does not name one."""
        return self._default_model

    @property
    def model(self) -> str | None:
        """Alias for :attr:`default_model`."""
        return self._default_model

    @property
    def voice_map(self) -> VoiceMap:
        """The alias map used to resolve request voices."""
        return self._voices

    @property
    def format(self) -> AudioFormat:
        """The default output container."""
        return self._format

    @property
    def sample_rate(self) -> int:
        """The default sample rate."""
        return self._sample_rate

    @property
    def encoder(self) -> Encoder | None:
        """The injected container encoder, or ``None`` for ffmpeg."""
        return self._encoder

    @property
    def converter(self) -> Converter | None:
        """The injected converter, or ``None`` for ffmpeg."""
        return self._converter

    @property
    def ffmpeg_executable(self) -> str:
        """The ffmpeg binary used for container conversion."""
        return self._ffmpeg

    @property
    def ffmpeg_runner(self) -> FFmpegRunner | None:
        """The injected ffmpeg runner, or ``None`` for subprocess."""
        return self._ffmpeg_runner

    def _missing_model(self) -> str | None:
        model = self._default_model
        if not model:
            return None
        if self._backend != "piper" and not _looks_like_model_path(model):
            return None
        if Path(model).expanduser().exists():
            return None
        return model

    def _needs_ffmpeg(self) -> bool:
        """Whether the default output format needs ffmpeg conversion.

        Local synthesizers produce WAV, so a non-WAV default is converted with
        ffmpeg unless the caller injected a converter, encoder or runner.
        """
        if self._format is AudioFormat.WAV:
            return False
        return (
            self._converter is None
            and self._encoder is None
            and self._ffmpeg_runner is None
        )

    async def check_health(self) -> ProviderHealth:
        """Probe the configured executable and model without synthesizing."""
        details: dict[str, object] = {
            "provider": self.name,
            "backend": self._backend,
        }
        if self._synthesizer is not None:
            return ProviderHealth(
                status=HealthStatus.HEALTHY,
                message="ok",
                details={**details, "synthesizer": True},
            )
        try:
            template = self._template()
        except InvalidRequestError as exc:
            return ProviderHealth(
                status=HealthStatus.UNHEALTHY,
                message=str(exc),
                details={**details, "error": "InvalidRequestError"},
            )
        executable = self._executable or template[0]
        if shutil.which(executable) is None:
            return ProviderHealth(
                status=HealthStatus.UNHEALTHY,
                message=f"local text to speech executable not found: {executable}",
                details={**details, "executable": executable},
            )
        missing_model = self._missing_model()
        if missing_model is not None:
            return ProviderHealth(
                status=HealthStatus.UNHEALTHY,
                message=f"local text to speech model not found: {missing_model}",
                details={
                    **details,
                    "executable": executable,
                    "model": missing_model,
                },
            )
        if self._default_model:
            details["model"] = self._default_model
        if self._needs_ffmpeg():
            if shutil.which(self._ffmpeg) is None:
                return ProviderHealth(
                    status=HealthStatus.UNHEALTHY,
                    message=(
                        "local text to speech needs ffmpeg to convert to "
                        f"{self._format.value}, but ffmpeg was not found: "
                        f"{self._ffmpeg}"
                    ),
                    details={
                        **details,
                        "executable": executable,
                        "ffmpeg": self._ffmpeg,
                        "ffmpeg_executable": self._ffmpeg,
                        "format": self._format.value,
                    },
                )
            details["ffmpeg"] = self._ffmpeg
            details["ffmpeg_executable"] = self._ffmpeg
            details["format"] = self._format.value
        return ProviderHealth(
            status=HealthStatus.HEALTHY,
            message="ok",
            details={**details, "executable": executable},
        )

    def _template(self) -> list[str]:
        if self._command is not None:
            return list(self._command)
        if self._backend == "piper":
            return ["piper", "--output_file", "{output}"]
        if self._backend == "kokoro":
            return ["kokoro", "--output-file", "{output}"]
        raise InvalidRequestError(
            "local text to speech backend needs a command",
            details={"provider": self.name, "backend": self._backend},
        )

    def _argv(
        self,
        template: list[str],
        *,
        text: str,
        voice: str,
        model: str,
        audio_format: AudioFormat,
        sample_rate: int,
        output: str,
        speed: float,
    ) -> list[str]:
        argv = [self._executable or template[0], *template[1:]]
        values = {
            "{text}": text,
            "{voice}": voice,
            "{model}": model,
            "{format}": audio_format.value,
            "{sample_rate}": str(sample_rate),
            "{output}": output,
        }
        expanded = [_replace_all(argument, values) for argument in argv]
        if self._backend == "piper":
            if model and "--model" not in expanded:
                expanded[1:1] = ["--model", model]
            if voice and voice != "default" and "--speaker" not in expanded:
                expanded[1:1] = ["--speaker", voice]
            if speed != 1.0 and "--length-scale" not in expanded:
                expanded[1:1] = ["--length-scale", f"{1.0 / speed:.4f}"]
        return expanded

    def _run_command(
        self,
        text: str,
        voice: str,
        model: str,
        audio_format: AudioFormat,
        sample_rate: int,
        *,
        speed: float,
        timeout_seconds: float,
    ) -> bytes:
        template = self._template()
        if self._backend == "piper" and not model:
            raise InvalidRequestError(
                "Piper text to speech needs a model path",
                details={"provider": self.name},
            )
        with tempfile.TemporaryDirectory(prefix="aivia-local-tts-") as directory:
            output = str(Path(directory) / f"speech.{audio_format.value}")
            argv = self._argv(
                template,
                text=text,
                voice=voice,
                model=model,
                audio_format=audio_format,
                sample_rate=sample_rate,
                output=output,
                speed=speed,
            )
            uses_text_argument = any("{text}" in part for part in template)
            uses_output_file = any("{output}" in part for part in template)
            try:
                completed = subprocess.run(
                    argv,
                    input=None if uses_text_argument else text.encode("utf-8"),
                    capture_output=True,
                    check=False,
                    timeout=timeout_seconds,
                )
            except FileNotFoundError as exc:
                raise ValidationError(
                    f"local text to speech executable not found: {argv[0]}",
                    details={"provider": self.name, "executable": argv[0]},
                ) from exc
            except subprocess.TimeoutExpired as exc:
                raise ValidationError(
                    "local text to speech command timed out",
                    details={"provider": self.name, "timeout": timeout_seconds},
                ) from exc
            if completed.returncode != 0:
                message = completed.stderr.decode("utf-8", "replace").strip()
                raise ValidationError(
                    "local text to speech command failed",
                    details={
                        "provider": self.name,
                        "returncode": completed.returncode,
                        "stderr": message[:500],
                    },
                )
            audio = b""
            if uses_output_file and os.path.exists(output):
                audio = Path(output).read_bytes()
            if not audio:
                audio = completed.stdout
            if not audio:
                raise ValidationError(
                    "local text to speech command produced no audio",
                    details={"provider": self.name},
                )
            return audio

    def _source_format(self, audio: bytes, target: AudioFormat) -> AudioFormat:
        detected = detect_audio_format(audio)
        if detected is not None:
            return detected
        if target is AudioFormat.PCM:
            return AudioFormat.PCM
        if (
            self._converter is not None
            or self._encoder is not None
            or self._ffmpeg_runner is not None
        ):
            return AudioFormat.PCM
        raise InvalidRequestError(
            "local text to speech does not support converting unrecognised "
            f"audio to {target.value}",
            details={"provider": self.name, "format": target.value},
        )

    def _convert(
        self,
        audio: bytes,
        source: AudioFormat,
        target: AudioFormat,
        sample_rate: int,
        *,
        timeout_seconds: float,
    ) -> bytes:
        if source is target:
            return audio
        if target not in _FFMPEG_CODECS:
            raise InvalidRequestError(
                f"local text to speech does not support converting to {target.value}",
                details={"provider": self.name, "format": target.value},
            )
        if self._converter is not None:
            converted = self._converter(audio, source, target, sample_rate)
        elif self._encoder is not None:
            converted = self._encoder(audio, target, sample_rate)
        else:
            converted = self._run_ffmpeg(
                audio,
                source,
                target,
                sample_rate,
                timeout_seconds=timeout_seconds,
            )
        if not converted:
            raise ValidationError(
                f"local text to speech conversion to {target.value} produced no audio",
                details={"provider": self.name, "format": target.value},
            )
        return converted

    def _run_ffmpeg(
        self,
        audio: bytes,
        source: AudioFormat,
        target: AudioFormat,
        sample_rate: int,
        *,
        timeout_seconds: float,
    ) -> bytes:
        codec = _FFMPEG_CODECS[target]
        if self._ffmpeg_runner is None and shutil.which(self._ffmpeg) is None:
            raise ValidationError(
                f"local text to speech needs ffmpeg to convert to {target.value}",
                details={
                    "provider": self.name,
                    "executable": self._ffmpeg,
                    "format": target.value,
                },
            )
        with tempfile.TemporaryDirectory(prefix="aivia-local-tts-") as directory:
            source_path = Path(directory) / f"source.{source.value}"
            target_path = Path(directory) / f"output.{target.value}"
            source_path.write_bytes(audio)
            argv = [
                self._ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
            ]
            if source is AudioFormat.PCM:
                argv.extend(["-f", "s16le", "-ar", str(sample_rate), "-ac", "1"])
            argv.extend(["-i", str(source_path), "-vn", "-c:a", codec])
            argv.extend(["-ar", str(sample_rate), "-ac", "1"])
            if target is AudioFormat.PCM:
                argv.extend(["-f", "s16le"])
            argv.extend(
                [
                    "-map_metadata",
                    "-1",
                    "-fflags",
                    "+bitexact",
                    "-flags",
                    "+bitexact",
                    "-threads",
                    "1",
                    str(target_path),
                ]
            )
            try:
                if self._ffmpeg_runner is not None:
                    completed = self._ffmpeg_runner(argv)
                else:
                    completed = subprocess.run(
                        argv,
                        capture_output=True,
                        check=False,
                        timeout=timeout_seconds,
                    )
            except FileNotFoundError as exc:
                raise ValidationError(
                    f"local text to speech executable not found: {argv[0]}",
                    details={"provider": self.name, "executable": argv[0]},
                ) from exc
            except subprocess.TimeoutExpired as exc:
                raise ValidationError(
                    "local text to speech conversion timed out",
                    details={"provider": self.name, "timeout": timeout_seconds},
                ) from exc
            if completed.returncode != 0:
                message = (completed.stderr or b"").decode("utf-8", "replace").strip()
                raise ValidationError(
                    "local text to speech conversion failed",
                    details={
                        "provider": self.name,
                        "returncode": completed.returncode,
                        "stderr": message[:500],
                    },
                )
            converted = b""
            if target_path.is_file():
                converted = target_path.read_bytes()
            if not converted and completed.stdout:
                converted = completed.stdout
            if not converted:
                raise ValidationError(
                    f"local text to speech conversion to {target.value} "
                    "produced no audio",
                    details={"provider": self.name, "format": target.value},
                )
            return converted

    async def synthesize(self, request: TTSRequest) -> TTSResponse:
        """Synthesize *request* with the configured local backend."""
        if not isinstance(request, TTSRequest):
            raise TypeError("request must be a TTSRequest")
        voice = self._voices.resolve(request.voice)
        model = request.model or self._default_model or ""
        audio_format = request.format
        sample_rate = request.sample_rate or self._sample_rate
        timeout_seconds = request.timeout_seconds or self._timeout_seconds
        if self._synthesizer is not None:
            audio = await asyncio.to_thread(
                self._synthesizer,
                request.text,
                voice,
                model,
                audio_format,
                sample_rate,
            )
        else:
            audio = await asyncio.to_thread(
                self._run_command,
                request.text,
                voice,
                model,
                audio_format,
                sample_rate,
                speed=request.speed,
                timeout_seconds=timeout_seconds,
            )
        if not audio:
            raise ValidationError(
                "local text to speech produced empty audio",
                details={"provider": self.name},
            )
        source_format = self._source_format(audio, audio_format)
        audio = await asyncio.to_thread(
            self._convert,
            audio,
            source_format,
            audio_format,
            sample_rate,
            timeout_seconds=timeout_seconds,
        )
        return TTSResponse(
            audio=audio,
            format=audio_format,
            sample_rate=sample_rate,
            voice=voice,
            model=model or self._backend,
            provider=self.name,
            metadata={
                "characters": len(request.text),
                "backend": self._backend,
            },
        )


def _replace_all(value: str, replacements: Mapping[str, str]) -> str:
    result = value
    for placeholder, replacement in replacements.items():
        result = result.replace(placeholder, replacement)
    return result


class PiperTTSProvider(LocalTTSProvider):
    """A local text to speech provider using the Piper CLI."""

    default_metadata = _PIPER_METADATA
    backend = "piper"


class KokoroTTSProvider(LocalTTSProvider):
    """A local text to speech provider using the Kokoro CLI."""

    default_metadata = _KOKORO_METADATA
    backend = "kokoro"


Local = LocalTTSProvider
"""Alias for :class:`LocalTTSProvider`."""

LocalTTS = LocalTTSProvider
"""Alias for :class:`LocalTTSProvider`."""

Piper = PiperTTSProvider
"""Alias for :class:`PiperTTSProvider`."""

PiperTTS = PiperTTSProvider
"""Alias for :class:`PiperTTSProvider`."""

Kokoro = KokoroTTSProvider
"""Alias for :class:`KokoroTTSProvider`."""

KokoroTTS = KokoroTTSProvider
"""Alias for :class:`KokoroTTSProvider`."""

register_provider("local-tts", LocalTTSProvider, module=__name__)
register_provider("piper-tts", PiperTTSProvider, module=__name__)
register_provider("kokoro-tts", KokoroTTSProvider, module=__name__)
register_provider("piper", PiperTTSProvider, module=__name__)
register_provider("kokoro", KokoroTTSProvider, module=__name__)
