"""Music generation provider adapters.

Importing this package makes every built-in music adapter available to the
provider registry. A hosted adapter generates vocal songs or instrumentals
through its vendor API, while the local stub synthesizes deterministic audio
without a network call. Every adapter records licensing and commercial-use
metadata so a later gate can decide whether a track may be published.
"""

from aivia.providers.music.elevenlabs import (
    DEFAULT_BASE_URL,
    ELEVENLABS_MUSIC_BASE_URL,
    ELEVENLABS_MUSIC_DEFAULT_BASE_URL,
    ElevenLabsMusic,
    ElevenLabsMusicGen,
    ElevenLabsMusicGenProvider,
    ElevenLabsMusicProvider,
)
from aivia.providers.music.local import (
    DEFAULT_DURATION_SECONDS,
    DEFAULT_SAMPLE_RATE,
    Encoder,
    LocalMusic,
    LocalMusicProvider,
    LocalMusicStub,
    LocalMusicStubProvider,
)
from aivia.providers.music.stable_audio import (
    STABLE_AUDIO_BASE_URL,
    STABLE_AUDIO_DEFAULT_BASE_URL,
    StableAudio,
    StableAudioMusic,
    StableAudioMusicProvider,
    StableAudioProvider,
)

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_DURATION_SECONDS",
    "DEFAULT_SAMPLE_RATE",
    "ELEVENLABS_MUSIC_BASE_URL",
    "ELEVENLABS_MUSIC_DEFAULT_BASE_URL",
    "STABLE_AUDIO_BASE_URL",
    "STABLE_AUDIO_DEFAULT_BASE_URL",
    "ElevenLabsMusic",
    "ElevenLabsMusicGen",
    "ElevenLabsMusicGenProvider",
    "ElevenLabsMusicProvider",
    "Encoder",
    "LocalMusic",
    "LocalMusicProvider",
    "LocalMusicStub",
    "LocalMusicStubProvider",
    "StableAudio",
    "StableAudioMusic",
    "StableAudioMusicProvider",
    "StableAudioProvider",
]
