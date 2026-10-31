"""Top-level exports for the text to speech adapters."""

from __future__ import annotations


def test_top_level_package_exposes_unambiguous_tts_names() -> None:
    from aivia import providers
    from aivia.providers import tts

    assert providers.ElevenLabsTTSProvider is tts.ElevenLabsTTSProvider
    assert providers.OpenAITTSProvider is tts.OpenAITTSProvider
    assert providers.GoogleTTSProvider is tts.GoogleTTSProvider
    assert providers.AzureTTSProvider is tts.AzureTTSProvider
    assert providers.LocalTTSProvider is tts.LocalTTSProvider
    assert providers.PiperTTSProvider is tts.PiperTTSProvider
    assert providers.KokoroTTSProvider is tts.KokoroTTSProvider
    assert providers.VoiceMap is tts.VoiceMap
    assert providers.TTSCostCalculator is tts.TTSCostCalculator
    assert providers.tts_cost_event is tts.tts_cost_event
