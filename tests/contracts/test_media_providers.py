"""Shared provider contract for every real media adapter, replayed offline.

Each adapter is built with an ``httpx.MockTransport`` whose handler fails on
any request. The inherited :class:`ProviderContract` suite only exercises
metadata, health and lifecycle, so a passing run proves every real media
adapter can be constructed and closed without touching the network.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest

from aivia.core.models import ProviderMeta
from aivia.providers import Provider, provider_registry
from aivia.providers.image import (
    FalImageProvider,
    GoogleImageProvider,
    OpenAIImageProvider,
    ReplicateImageProvider,
)
from aivia.providers.image_edit import (
    AggregatorImageEditProvider,
    RembgImageEditProvider,
    RemoteOpsImageEditProvider,
)
from aivia.providers.music import (
    ElevenLabsMusicProvider,
    LocalMusicStubProvider,
    StableAudioMusicProvider,
)
from aivia.providers.protocols import (
    CAPABILITY_IMAGE_EDIT,
    CAPABILITY_IMAGE_GEN,
    CAPABILITY_MUSIC_GEN,
    CAPABILITY_STOCK,
    CAPABILITY_STT,
    CAPABILITY_TTS,
    CAPABILITY_VIDEO_GEN,
)
from aivia.providers.stock import PexelsStockProvider, PixabayStockProvider
from aivia.providers.stt import LocalSTTProvider, OpenAISTTProvider
from aivia.providers.tts import (
    AzureTTSProvider,
    ElevenLabsTTSProvider,
    GoogleTTSProvider,
    KokoroTTSProvider,
    LocalTTSProvider,
    OpenAITTSProvider,
    PiperTTSProvider,
)
from aivia.providers.video import (
    FalVideoProvider,
    ReplicateVideoProvider,
    VeoVideoProvider,
)

from .provider_contract import ProviderContract

API_KEY = "test-key"

_MEDIA_CAPABILITIES = frozenset(
    {
        CAPABILITY_IMAGE_GEN,
        CAPABILITY_IMAGE_EDIT,
        CAPABILITY_VIDEO_GEN,
        CAPABILITY_MUSIC_GEN,
        CAPABILITY_STOCK,
        CAPABILITY_TTS,
        CAPABILITY_STT,
    }
)


def _offline_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"contract test attempted a network call: {request.url}")

    return httpx.MockTransport(handler)


@dataclass(frozen=True)
class AdapterSpec:
    """How to build one real media adapter without network access."""

    name: str
    provider: type[Provider]
    capabilities: frozenset[str]
    options: Mapping[str, Any] = field(default_factory=dict)
    uses_http: bool = True

    def build(self) -> Provider:
        options = dict(self.options)
        if self.uses_http:
            options["transport"] = _offline_transport()
        return self.provider(**options)


_SPECS: tuple[AdapterSpec, ...] = (
    AdapterSpec(
        "image/fal",
        FalImageProvider,
        frozenset({CAPABILITY_IMAGE_GEN}),
        {"api_key": API_KEY, "default_model": "fal-ai/flux/dev"},
    ),
    AdapterSpec(
        "image/google",
        GoogleImageProvider,
        frozenset({CAPABILITY_IMAGE_GEN}),
        {"api_key": API_KEY, "default_model": "imagen-4.0-generate-001"},
    ),
    AdapterSpec(
        "image/openai",
        OpenAIImageProvider,
        frozenset({CAPABILITY_IMAGE_GEN}),
        {"api_key": API_KEY, "default_model": "gpt-image-1"},
    ),
    AdapterSpec(
        "image/replicate",
        ReplicateImageProvider,
        frozenset({CAPABILITY_IMAGE_GEN}),
        {"api_key": API_KEY, "model": "owner/name:version"},
    ),
    AdapterSpec(
        "image_edit/aggregator",
        AggregatorImageEditProvider,
        frozenset({CAPABILITY_IMAGE_EDIT}),
        {"api_key": API_KEY, "models": {"edit": "fal-ai/model"}},
    ),
    AdapterSpec(
        "image_edit/rembg",
        RembgImageEditProvider,
        frozenset({CAPABILITY_IMAGE_EDIT}),
        uses_http=False,
    ),
    AdapterSpec(
        "image_edit/remote_ops",
        RemoteOpsImageEditProvider,
        frozenset({CAPABILITY_IMAGE_EDIT}),
        {"api_key": API_KEY, "models": {"upscale": "fal-ai/model"}},
    ),
    AdapterSpec(
        "video/fal",
        FalVideoProvider,
        frozenset({CAPABILITY_VIDEO_GEN}),
        {"api_key": API_KEY, "default_model": "fal-ai/model"},
    ),
    AdapterSpec(
        "video/replicate",
        ReplicateVideoProvider,
        frozenset({CAPABILITY_VIDEO_GEN}),
        {"api_key": API_KEY, "model": "owner/name:version"},
    ),
    AdapterSpec(
        "video/veo",
        VeoVideoProvider,
        frozenset({CAPABILITY_VIDEO_GEN}),
        {"api_key": API_KEY, "default_model": "veo-3"},
    ),
    AdapterSpec(
        "music/elevenlabs",
        ElevenLabsMusicProvider,
        frozenset({CAPABILITY_MUSIC_GEN}),
        {"api_key": API_KEY},
    ),
    AdapterSpec(
        "music/local",
        LocalMusicStubProvider,
        frozenset({CAPABILITY_MUSIC_GEN}),
        uses_http=False,
    ),
    AdapterSpec(
        "music/stable_audio",
        StableAudioMusicProvider,
        frozenset({CAPABILITY_MUSIC_GEN}),
        {"api_key": API_KEY},
    ),
    AdapterSpec(
        "stock/pexels",
        PexelsStockProvider,
        frozenset({CAPABILITY_STOCK}),
        {"api_key": API_KEY},
    ),
    AdapterSpec(
        "stock/pixabay",
        PixabayStockProvider,
        frozenset({CAPABILITY_STOCK}),
        {"api_key": API_KEY},
    ),
    AdapterSpec(
        "tts/azure",
        AzureTTSProvider,
        frozenset({CAPABILITY_TTS}),
        {"api_key": API_KEY, "region": "eastus"},
    ),
    AdapterSpec(
        "tts/elevenlabs",
        ElevenLabsTTSProvider,
        frozenset({CAPABILITY_TTS}),
        {"api_key": API_KEY},
    ),
    AdapterSpec(
        "tts/google",
        GoogleTTSProvider,
        frozenset({CAPABILITY_TTS}),
        {"api_key": API_KEY},
    ),
    AdapterSpec(
        "tts/local",
        LocalTTSProvider,
        frozenset({CAPABILITY_TTS}),
        uses_http=False,
    ),
    AdapterSpec(
        "tts/piper",
        PiperTTSProvider,
        frozenset({CAPABILITY_TTS}),
        uses_http=False,
    ),
    AdapterSpec(
        "tts/kokoro",
        KokoroTTSProvider,
        frozenset({CAPABILITY_TTS}),
        uses_http=False,
    ),
    AdapterSpec(
        "tts/openai",
        OpenAITTSProvider,
        frozenset({CAPABILITY_TTS}),
        {"api_key": API_KEY},
    ),
    AdapterSpec(
        "stt/local",
        LocalSTTProvider,
        frozenset({CAPABILITY_STT}),
        uses_http=False,
    ),
    AdapterSpec(
        "stt/openai",
        OpenAISTTProvider,
        frozenset({CAPABILITY_STT}),
        {"api_key": API_KEY},
    ),
)

_SPEC_BY_NAME = {spec.name: spec for spec in _SPECS}
_SPEC_NAMES = [spec.name for spec in _SPECS]


@pytest.fixture(params=_SPEC_NAMES, ids=_SPEC_NAMES)
def media_provider(request: pytest.FixtureRequest) -> Provider:
    return _SPEC_BY_NAME[request.param].build()


class TestMediaProviderContract(ProviderContract):
    @pytest.fixture
    def provider(self, media_provider: Provider) -> Provider:
        return media_provider


@pytest.mark.parametrize("spec", _SPECS, ids=lambda spec: spec.name)
def test_adapter_declares_its_media_capability(spec: AdapterSpec) -> None:
    provider = spec.build()

    assert spec.capabilities <= set(provider.capabilities)


def test_every_registered_real_media_adapter_has_a_contract() -> None:
    registered: set[type[Provider]] = set()
    for registration in provider_registry.registrations():
        provider = registration.provider
        if provider.__module__.startswith("aivia.providers.mock"):
            continue
        metadata = provider.default_metadata
        if metadata is None:
            continue
        capabilities = (
            metadata.capabilities
            if isinstance(metadata, ProviderMeta)
            else metadata.get("capabilities", ())
        )
        if _MEDIA_CAPABILITIES.intersection(capabilities):
            registered.add(provider)

    assert registered == {spec.provider for spec in _SPECS}
