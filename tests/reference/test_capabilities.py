"""Providers are filtered by reference capability with a prompt-only fallback."""

from __future__ import annotations

import pytest

from aivia.core.models import ProviderMeta
from aivia.providers import Provider
from aivia.providers.image import CAPABILITY_IMAGE_INPUT
from aivia.providers.protocols.image import (
    CAPABILITY_IMAGE_GEN,
    ImageRequest,
)
from aivia.providers.protocols.media import MediaInput
from aivia.reference import reference_providers_for_request

_URL = "https://example.com/reference.png"


class FakeImageProvider(Provider):
    def __init__(
        self,
        name: str,
        capabilities: list[str],
    ) -> None:
        super().__init__(
            ProviderMeta(
                name=name,
                capabilities=capabilities,
            )
        )


def _request(*, references: bool = True) -> ImageRequest:
    images = [MediaInput(url=_URL)] if references else []
    return ImageRequest(prompt="a red fox", reference_images=images)


def test_providers_declaring_image_input_are_returned_with_references() -> None:
    text_only = FakeImageProvider("text-only", [CAPABILITY_IMAGE_GEN])
    with_references = FakeImageProvider(
        "with-references",
        [CAPABILITY_IMAGE_GEN, CAPABILITY_IMAGE_INPUT],
    )

    providers, warning = reference_providers_for_request(
        [text_only, with_references], _request()
    )

    assert warning is None
    assert providers == (with_references,)


def test_no_references_returns_all_providers_without_warning() -> None:
    text_only = FakeImageProvider("text-only", [CAPABILITY_IMAGE_GEN])
    with_references = FakeImageProvider(
        "with-references",
        [CAPABILITY_IMAGE_GEN, CAPABILITY_IMAGE_INPUT],
    )

    providers, warning = reference_providers_for_request(
        [text_only, with_references], _request(references=False)
    )

    assert warning is None
    assert providers == (text_only, with_references)


def test_no_reference_capable_provider_falls_back_to_prompt_only_with_warning() -> None:
    text_only = FakeImageProvider("text-only", [CAPABILITY_IMAGE_GEN])

    providers, warning = reference_providers_for_request([text_only], _request())

    assert warning is not None
    assert "prompt-only style transfer" in warning
    assert CAPABILITY_IMAGE_INPUT in warning
    assert providers == (text_only,)


def test_fallback_emits_a_user_warning() -> None:
    text_only = FakeImageProvider("text-only", [CAPABILITY_IMAGE_GEN])

    with pytest.warns(UserWarning, match="prompt-only style transfer"):
        reference_providers_for_request([text_only], _request())


def test_empty_provider_list_with_references_falls_back() -> None:
    providers, warning = reference_providers_for_request([], _request())

    assert warning is not None
    assert "prompt-only style transfer" in warning
    assert providers == ()
