"""A deterministic local music stub.

The local stub runs on the machine, calls no service and costs nothing, which
makes it the natural offline fallback for music generation. It synthesizes a
valid, deterministic PCM WAV track from the prompt and seed; when another
container is requested it transcodes through an injected encoder or a local
ffmpeg binary. Tests inject an encoder so the offline suite needs no external
binary.

The stub records a public-domain license and commercial-use metadata so a later
license gate never has to guess the rights of a locally generated track.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import math
import shutil
import struct
import subprocess
import tempfile
import wave
from collections.abc import Callable
from pathlib import Path

from aivia.core.config.http import HttpClientConfig
from aivia.core.errors import ValidationError
from aivia.core.models.asset import AssetLicense
from aivia.core.models.provider import ProviderMeta, ProviderPrice
from aivia.providers.base import Provider
from aivia.providers.music.common import sniff_audio_format, validate_audio
from aivia.providers.protocols.audio import AudioFormat
from aivia.providers.protocols.music import (
    CAPABILITY_MUSIC_GEN,
    MusicRequest,
    MusicResponse,
    MusicTrack,
)
from aivia.providers.registry import register_provider

__all__ = [
    "DEFAULT_DURATION_SECONDS",
    "DEFAULT_SAMPLE_RATE",
    "Encoder",
    "LocalMusic",
    "LocalMusicProvider",
    "LocalMusicStub",
    "LocalMusicStubProvider",
]

DEFAULT_DURATION_SECONDS = 30.0
"""The duration used when a request does not name one."""

DEFAULT_SAMPLE_RATE = 44_100
"""The sample rate used when a request does not name one."""

Encoder = Callable[[bytes, AudioFormat, int], bytes]
"""A callable taking WAV bytes, the target format and the sample rate."""

LICENSE_NAME = "CC0-1.0"
LICENSE_URL = "https://creativecommons.org/publicdomain/zero/1.0/"

_MP3_SILENT_FRAME = b"\xff\xfb\x90\x64" + b"\x00" * 413

_DEFAULT_METADATA = ProviderMeta(
    name="local-music",
    capabilities=[CAPABILITY_MUSIC_GEN],
    supported_options={
        CAPABILITY_MUSIC_GEN: [
            "vocals",
            "instrumental",
            "wav",
            "pcm",
            "mp3",
            "ogg",
            "opus",
            "flac",
            "prompt",
            "lyrics",
            "duration",
            "sample_rate",
            "seed",
            "count",
            "format",
        ]
    },
    pricing={CAPABILITY_MUSIC_GEN: ProviderPrice(amount_usd=0.0, unit="second")},
    license=AssetLicense(
        name=LICENSE_NAME,
        url=LICENSE_URL,
        commercial_use=True,
        attribution_required=False,
    ),
    commercial_use_notes=(
        "Synthesized locally by the stub; no third-party rights attach to the "
        "generated audio."
    ),
)


def _digest(*parts: object) -> str:
    text = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _wav_bytes(duration_seconds: float, *, sample_rate: int, seed: int) -> bytes:
    """Return a valid 16-bit PCM WAV whose duration matches the request."""
    duration = max(duration_seconds, 1.0 / sample_rate)
    sample_rate = max(1, sample_rate)
    total_frames = max(1, round(duration * sample_rate))
    frequency = 110 + seed % 220
    one_second = bytearray()
    for index in range(sample_rate):
        value = int(9_000 * math.sin(2.0 * math.pi * frequency * index / sample_rate))
        one_second.extend(struct.pack("<h", value))
    repeats, remainder = divmod(total_frames, sample_rate)
    frames = bytes(one_second) * repeats + bytes(one_second[: remainder * 2])
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(frames)
    return buffer.getvalue()


def _mp3_bytes(duration_seconds: float) -> bytes:
    """Return deterministic silent MPEG frames covering *duration_seconds*."""
    frames = max(1, math.ceil(max(duration_seconds, 0.0) * 44_100 / 1152))
    return _MP3_SILENT_FRAME * frames


class LocalMusicStubProvider(Provider):
    """A local, deterministic music stub that never calls a service.

    ``encoder`` replaces transcoding entirely, which is how tests stay offline.
    Without one, a non-WAV request is transcoded by ``ffmpeg_executable``; if
    ffmpeg is missing the adapter raises a validation error rather than
    returning a container the caller did not ask for.
    """

    default_metadata = _DEFAULT_METADATA

    def __init__(
        self,
        *,
        encoder: Encoder | None = None,
        default_duration_seconds: float = DEFAULT_DURATION_SECONDS,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        ffmpeg_executable: str = "ffmpeg",
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
        if encoder is not None and not callable(encoder):
            raise TypeError("encoder must be callable or None")
        if (
            isinstance(default_duration_seconds, bool)
            or not isinstance(default_duration_seconds, (int, float))
            or not math.isfinite(float(default_duration_seconds))
            or default_duration_seconds <= 0
        ):
            raise ValueError("default_duration_seconds must be positive and finite")
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
            raise TypeError("sample_rate must be an int")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not isinstance(ffmpeg_executable, str) or not ffmpeg_executable.strip():
            raise ValueError("ffmpeg_executable must be a non-empty string")
        self._encoder = encoder
        self._default_duration = float(default_duration_seconds)
        self._sample_rate = sample_rate
        self._ffmpeg = ffmpeg_executable.strip()

    @property
    def encoder(self) -> Encoder | None:
        """The injected encoder, or ``None`` for ffmpeg."""
        return self._encoder

    @property
    def default_duration_seconds(self) -> float:
        """The duration used when a request does not name one."""
        return self._default_duration

    @property
    def sample_rate(self) -> int:
        """The sample rate used when a request does not name one."""
        return self._sample_rate

    def _encode(
        self,
        wav: bytes,
        audio_format: AudioFormat,
        sample_rate: int,
        duration_seconds: float,
    ) -> bytes:
        if audio_format in {AudioFormat.WAV, AudioFormat.PCM}:
            return wav
        if self._encoder is not None:
            return self._encoder(wav, audio_format, sample_rate)
        if shutil.which(self._ffmpeg) is None:
            if audio_format is AudioFormat.MP3:
                return _mp3_bytes(duration_seconds)
            raise ValidationError(
                "local music stub needs ffmpeg for non-WAV output",
                details={"provider": self.name, "executable": self._ffmpeg},
            )
        with tempfile.TemporaryDirectory(prefix="aivia-local-music-") as directory:
            source = Path(directory) / "source.wav"
            target = Path(directory) / f"output.{audio_format.value}"
            source.write_bytes(wav)
            args = [
                self._ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-fflags",
                "+bitexact",
                "-flags",
                "+bitexact",
                "-map_metadata",
                "-1",
                "-threads",
                "1",
                str(target),
            ]
            try:
                completed = subprocess.run(
                    args,
                    capture_output=True,
                    check=False,
                    timeout=60,
                )
            except FileNotFoundError as exc:
                raise ValidationError(
                    "local music stub executable not found",
                    details={"provider": self.name, "executable": self._ffmpeg},
                ) from exc
            except subprocess.TimeoutExpired as exc:
                raise ValidationError(
                    "local music stub encoder timed out",
                    details={"provider": self.name},
                ) from exc
            if completed.returncode != 0:
                message = completed.stderr.decode("utf-8", "replace").strip()
                raise ValidationError(
                    "local music stub encoder failed",
                    details={
                        "provider": self.name,
                        "returncode": completed.returncode,
                        "stderr": message[:500],
                    },
                )
            return target.read_bytes()

    async def generate(self, request: MusicRequest) -> MusicResponse:
        """Generate deterministic local music for *request*."""
        if not isinstance(request, MusicRequest):
            raise TypeError("request must be a MusicRequest")
        duration = request.duration_seconds or self._default_duration
        sample_rate = request.sample_rate or self._sample_rate
        tracks: list[MusicTrack] = []
        for index in range(request.count):
            seed = (
                request.seed + index
                if request.seed is not None
                else int(_digest(request.prompt, index)[:16], 16)
            )
            wav = await asyncio.to_thread(
                _wav_bytes,
                duration,
                sample_rate=sample_rate,
                seed=seed,
            )
            audio = await asyncio.to_thread(
                self._encode,
                wav,
                request.format,
                sample_rate,
                duration,
            )
            audio = validate_audio(audio, provider=self.name)
            tracks.append(
                MusicTrack(
                    data=audio,
                    format=sniff_audio_format(audio, request.format),
                    duration_seconds=duration,
                    sample_rate=sample_rate,
                    channels=1,
                    seed=seed,
                    title=request.prompt,
                    lyrics=None if request.instrumental else request.lyrics,
                    metadata={
                        "license": LICENSE_NAME,
                        "license_url": LICENSE_URL,
                        "commercial_use": True,
                        "local_stub": True,
                    },
                )
            )
        return MusicResponse(
            tracks=tracks,
            model="local-stub",
            provider=self.name,
            metadata={
                "license": LICENSE_NAME,
                "license_url": LICENSE_URL,
                "commercial_use": True,
                "local_stub": True,
                "instrumental": request.instrumental,
            },
        )


LocalMusic = LocalMusicStubProvider
"""Alias for :class:`LocalMusicStubProvider`."""

LocalMusicProvider = LocalMusicStubProvider
"""Alias for :class:`LocalMusicStubProvider`."""

LocalMusicStub = LocalMusicStubProvider
"""Alias for :class:`LocalMusicStubProvider`."""

register_provider("local-music", LocalMusicStubProvider, module=__name__)
register_provider("local_music", LocalMusicStubProvider, module=__name__)
register_provider("local-music-stub", LocalMusicStubProvider, module=__name__)
