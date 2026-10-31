"""Per-language voice selection, listing and pronunciation hints."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Callable

import httpx
import pytest

from aivia.core.errors import InvalidRequestError
from aivia.core.models.provider import ProviderMeta
from aivia.media.voices import (
    DEFAULT_ALPHABET,
    LanguageVoices,
    PronunciationHint,
    Voice,
    VoiceCatalog,
    VoiceResolver,
    apply_pronunciation,
    list_voices,
    normalize_pronunciation,
    prepare_speech_request,
    resolve_voice,
    supports_pronunciation,
)
from aivia.providers.protocols import CAPABILITY_TTS, AudioFormat, TTSRequest
from aivia.providers.tts.azure import AzureTTSProvider
from aivia.providers.tts.google import GoogleTTSProvider
from aivia.providers.tts.openai import OpenAITTSProvider

MP3 = b"ID3\x04\x00\x00\x00\x00\x00\x00audio"


def _metadata(*options: str) -> ProviderMeta:
    return ProviderMeta(
        name="fake-tts",
        capabilities=[CAPABILITY_TTS],
        supported_options={CAPABILITY_TTS: list(options)},
    )


def _azure(
    handler: Callable[[httpx.Request], httpx.Response],
) -> AzureTTSProvider:
    return AzureTTSProvider(
        api_key="test-key",
        region="westeurope",
        transport=httpx.MockTransport(handler),
    )


def _google(
    handler: Callable[[httpx.Request], httpx.Response],
) -> GoogleTTSProvider:
    return GoogleTTSProvider(
        api_key="test-key",
        transport=httpx.MockTransport(handler),
    )


def test_a_language_falls_back_to_its_base_then_the_default_voice() -> None:
    catalog = VoiceCatalog({"en": "voice-en", "de": "voice-de"}, default="voice-all")

    assert catalog.resolve("en") == "voice-en"
    assert catalog.resolve("DE") == "voice-de"
    assert catalog.resolve("de-AT") == "voice-de"
    assert catalog.resolve("fr") == "voice-all"
    assert catalog.resolve("fr", "explicit-voice") == "explicit-voice"
    assert catalog.resolve("") == "voice-all"
    assert VoiceResolver is VoiceCatalog
    assert LanguageVoices is VoiceCatalog


def test_configured_fallbacks_are_followed_and_may_name_a_voice_or_language() -> None:
    catalog = VoiceCatalog(
        {
            "en": "voice-en",
            "de": "voice-de",
            "default": "voice-all",
            "fallbacks": {"fr": "de", "it": "voice-it"},
        }
    )

    assert catalog.resolve("fr") == "voice-de"
    assert catalog.resolve("fr-CA") == "voice-de"
    assert catalog.resolve("it") == "voice-it"
    assert catalog.resolve("ja") == "voice-all"
    assert catalog.fallbacks == {"fr": "de", "it": "voice-it"}


def test_a_missing_language_without_a_fallback_is_rejected() -> None:
    catalog = VoiceCatalog({"en": "voice-en"})

    with pytest.raises(InvalidRequestError, match="no voice"):
        catalog.resolve("de")
    with pytest.raises(InvalidRequestError, match="no voice"):
        catalog.select("de")


def test_voices_can_be_listed_filtered_and_compared_to_strings() -> None:
    catalog = VoiceCatalog(
        [
            Voice(voice="voice-de", language="de"),
            Voice(voice="voice-en", language="en"),
        ],
        default="voice-all",
    )

    assert [voice.voice for voice in list_voices(catalog)] == [
        "voice-all",
        "voice-de",
        "voice-en",
    ]
    assert list_voices(catalog, "de") == ["voice-de"]
    assert list_voices(catalog, "de-AT") == ["voice-de"]
    assert catalog.list_voices("en") == ["voice-en"]
    assert catalog.entries[0] == "voice-all"
    assert catalog.voices["en"].label == "en:voice-en"
    assert catalog.languages["de"] == "voice-de"
    assert len(catalog) == 2
    assert "EN" in catalog
    assert [voice.voice for voice in catalog] == [
        "voice-all",
        "voice-de",
        "voice-en",
    ]


def test_module_helpers_accept_both_argument_orders() -> None:
    catalog = VoiceCatalog({"en": "voice-en", "de": "voice-de"})

    assert resolve_voice(catalog, "de") == "voice-de"
    assert resolve_voice("de", catalog) == "voice-de"
    assert resolve_voice(catalog, "de", "override") == "override"
    assert resolve_voice(catalog, "fr", fallback="voice-fallback") == "voice-fallback"


def test_a_voice_entry_coerces_a_plain_string_and_exposes_aliases() -> None:
    voice = Voice.model_validate("narrator")

    assert voice.voice == voice.id == voice.voice_id == voice.name == "narrator"
    assert voice["voice"] == "narrator"
    assert voice.get("language") == ""
    assert str(voice) == "narrator"
    assert voice.to_dict()["voice"] == "narrator"


def test_pronunciation_accepts_a_mapping_or_hint_objects() -> None:
    assert normalize_pronunciation({"Ada": "AY-duh"}) == {"Ada": "AY-duh"}
    assert normalize_pronunciation(
        [PronunciationHint(word="Ada", phonetic="AY-duh")]
    ) == {"Ada": "AY-duh"}
    assert normalize_pronunciation(None) == {}
    assert DEFAULT_ALPHABET == "ipa"

    with pytest.raises(ValueError, match="non-empty"):
        normalize_pronunciation({"Ada": "  "})
    with pytest.raises(ValueError, match="non-empty"):
        normalize_pronunciation({"  ": "AY-duh"})


def test_hints_are_passed_only_to_a_provider_that_declares_support() -> None:
    request = TTSRequest(
        text="Ada spoke",
        voice="voice-en",
        pronunciation={"Ada": "AY-duh"},
    )

    supported = apply_pronunciation(request, _metadata("pronunciation"))
    unsupported = apply_pronunciation(request, _metadata())

    assert supported.pronunciation == {"Ada": "AY-duh"}
    assert unsupported.pronunciation == {}
    assert apply_pronunciation(request, supported=True).pronunciation == {
        "Ada": "AY-duh"
    }
    assert apply_pronunciation(request, supported=False).pronunciation == {}

    with pytest.raises(InvalidRequestError, match="does not support"):
        apply_pronunciation(request, _metadata(), strict=True)


def test_support_is_read_from_provider_metadata_not_a_vendor_name() -> None:
    azure = _azure(lambda request: httpx.Response(200, content=MP3))
    google = _google(lambda request: httpx.Response(200, json={"audioContent": ""}))
    openai = OpenAITTSProvider(api_key="test-key")

    assert supports_pronunciation(azure) is True
    assert supports_pronunciation(google) is True
    assert supports_pronunciation(openai) is False
    assert supports_pronunciation(_metadata("phonemes")) is True
    assert supports_pronunciation(_metadata("speed")) is False
    assert supports_pronunciation(None) is False


def test_prepare_speech_request_resolves_the_voice_and_passes_hints() -> None:
    request = TTSRequest(
        text="Ada spoke",
        voice="auto",
        language="de-AT",
        pronunciation={"Ada": "AY-duh"},
    )

    prepared = prepare_speech_request(
        request,
        VoiceCatalog({"de": "voice-de"}),
        voice="",
        provider=_metadata("pronunciation"),
    )

    assert prepared.voice == "voice-de"
    assert prepared.language == request.language
    assert prepared.pronunciation == {"Ada": "AY-duh"}


def test_azure_renders_pronunciation_hints_as_phoneme_markup() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=MP3)

    provider = _azure(handler)
    asyncio.run(
        provider.synthesize(
            TTSRequest(
                text="Ada & Bob",
                voice="en-US-JennyNeural",
                pronunciation={"Ada": "AY-duh"},
            )
        )
    )

    ssml = requests[0].content.decode("utf-8")
    assert '<phoneme alphabet="ipa" ph="AY-duh">Ada</phoneme>' in ssml
    assert "Ada &amp; Bob" not in ssml
    assert "&amp; Bob" in ssml


def test_google_switches_to_ssml_only_when_hints_are_present() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = {"audioContent": base64.b64encode(MP3).decode("ascii")}
        return httpx.Response(200, json=body)

    provider = _google(handler)
    asyncio.run(
        provider.synthesize(
            TTSRequest(
                text="Ada spoke",
                voice="en-US-Neural2-C",
                format=AudioFormat.MP3,
                pronunciation={"Ada": "AY-duh"},
            )
        )
    )
    asyncio.run(
        provider.synthesize(
            TTSRequest(
                text="Ada spoke",
                voice="en-US-Neural2-C",
                format=AudioFormat.MP3,
            )
        )
    )

    hinted = json.loads(requests[0].content)
    plain = json.loads(requests[1].content)
    assert "ssml" in hinted["input"]
    assert (
        '<phoneme alphabet="ipa" ph="AY-duh">Ada</phoneme>' in (hinted["input"]["ssml"])
    )
    assert plain["input"] == {"text": "Ada spoke"}
