"""Video adapters that forward ``generate_audio`` declare native audio support."""

from __future__ import annotations

import pytest

from aivia.core.models.provider import ProviderMeta
from aivia.providers import Provider
from aivia.providers.docs import build_provider_docs
from aivia.providers.protocols import CAPABILITY_VIDEO_NATIVE_AUDIO
from aivia.providers.video import (
    FalVideoProvider,
    ReplicateVideoProvider,
    VeoVideoProvider,
)
from aivia.providers.video.flags import supports_native_audio

_ADAPTERS: tuple[tuple[str, type[Provider], tuple[str, ...]], ...] = (
    ("fal", FalVideoProvider, ("fal-video", "fal_video")),
    (
        "replicate",
        ReplicateVideoProvider,
        ("replicate-video", "replicate_video"),
    ),
    (
        "veo",
        VeoVideoProvider,
        ("veo", "veo-video", "veo_video", "google-veo", "google_veo"),
    ),
)


def _declared_metadata(provider: type[Provider]) -> ProviderMeta:
    metadata = provider.default_metadata
    assert isinstance(metadata, ProviderMeta)
    return metadata


@pytest.mark.parametrize(("name", "provider", "aliases"), _ADAPTERS)
def test_video_adapter_declares_native_audio(
    name: str,
    provider: type[Provider],
    aliases: tuple[str, ...],
) -> None:
    metadata = _declared_metadata(provider)

    assert metadata.name == name
    assert CAPABILITY_VIDEO_NATIVE_AUDIO in metadata.capabilities
    assert metadata.supports(CAPABILITY_VIDEO_NATIVE_AUDIO)
    assert supports_native_audio(metadata)


@pytest.mark.parametrize(("name", "provider", "aliases"), _ADAPTERS)
def test_capability_matrix_lists_native_audio_for_every_alias(
    name: str,
    provider: type[Provider],
    aliases: tuple[str, ...],
) -> None:
    text = build_provider_docs()

    for alias in aliases:
        row = next(
            line for line in text.splitlines() if line.startswith(f"| `{alias}` |")
        )
        assert "`video_native_audio`" in row
