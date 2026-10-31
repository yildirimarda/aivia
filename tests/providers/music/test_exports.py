"""Top-level exports for the music adapters."""

from __future__ import annotations


def test_top_level_package_exposes_unambiguous_music_names() -> None:
    from aivia import providers
    from aivia.providers import music

    assert providers.ElevenLabsMusicProvider is music.ElevenLabsMusicProvider
    assert providers.StableAudioMusicProvider is music.StableAudioMusicProvider
    assert providers.LocalMusicStubProvider is music.LocalMusicStubProvider
    assert providers.LocalMusicProvider is music.LocalMusicProvider
    assert providers.StableAudioProvider is music.StableAudioProvider
    assert providers.ElevenLabsMusicGenProvider is music.ElevenLabsMusicGenProvider
    assert providers.Encoder is music.Encoder
