"""Voice alias mapping shared by every text to speech adapter."""

from __future__ import annotations

import pytest

from aivia.core.errors import InvalidRequestError
from aivia.providers.tts import VoiceMap, VoiceMapping


def test_aliases_are_case_insensitive_and_preserve_provider_ids() -> None:
    voices = VoiceMap({"Narrator": "voice-1", "host_de": "voice-2"})

    assert voices.resolve("narrator") == "voice-1"
    assert voices.resolve("  NARRATOR  ") == "voice-1"
    assert voices.resolve("host_de") == "voice-2"
    assert voices.resolve("provider-voice-3") == "provider-voice-3"
    assert voices.voices == {"narrator": "voice-1", "host_de": "voice-2"}
    assert "NARRATOR" in voices
    assert len(voices) == 2
    assert VoiceMapping is VoiceMap


def test_a_default_is_used_only_for_an_empty_voice() -> None:
    voices = VoiceMap({"narrator": "voice-1"}, default="fallback")

    assert voices.resolve("") == "fallback"
    assert voices.resolve("narrator") == "voice-1"
    assert voices.resolve("other") == "other"


def test_an_empty_voice_without_a_default_is_rejected() -> None:
    with pytest.raises(InvalidRequestError, match="needs a voice"):
        VoiceMap().resolve("   ")


def test_an_invalid_mapping_is_rejected() -> None:
    with pytest.raises(TypeError, match="mapping"):
        VoiceMap([])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        VoiceMap({"": "voice-1"})
    with pytest.raises(ValueError, match="non-empty"):
        VoiceMap({"narrator": " "})
    with pytest.raises(ValueError, match="duplicate"):
        VoiceMap({"narrator": "voice-1", "Narrator": "voice-2"})
