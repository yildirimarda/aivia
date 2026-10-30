"""Reference images are passed only to providers declaring image input support."""

from __future__ import annotations

import pytest

from aivia.core.config import CapabilityChainConfig, CapabilityStrategy
from aivia.core.errors import InvalidRequestError, ProviderExhaustedError
from aivia.core.models import ProviderMeta
from aivia.core.router import Router
from aivia.providers import Provider
from aivia.providers.image import (
    CAPABILITY_IMAGE_INPUT,
    ReferenceInputNotSupportedError,
    ensure_reference_support,
    filter_providers_for_reference_input,
    prepare_image_request,
    required_image_capabilities,
    supports_reference_images,
)
from aivia.providers.media import MediaConstraints, adapt_image_request
from aivia.providers.protocols import CAPABILITY_IMAGE_GEN, ImageRequest
from aivia.providers.protocols.media import MediaInput

_URL = "https://example.com/reference.png"


class FakeImageProvider(Provider):
    def __init__(
        self,
        name: str,
        capabilities: list[str],
        supported_options: dict[str, list[str]] | None = None,
    ) -> None:
        super().__init__(
            ProviderMeta(
                name=name,
                capabilities=capabilities,
                supported_options=supported_options or {},
            )
        )


def _request(*, references: bool = True) -> ImageRequest:
    images = [MediaInput(url=_URL)] if references else []
    return ImageRequest(prompt="a red fox", reference_images=images)


def _router(*providers: Provider) -> Router:
    return Router(
        providers={provider.name: provider for provider in providers},
        config=CapabilityChainConfig(
            strategy=CapabilityStrategy.FIRST_AVAILABLE,
            chain=[provider.name for provider in providers],
        ),
    )


def test_a_provider_declaring_image_input_receives_the_references() -> None:
    provider = FakeImageProvider(
        "with-references",
        [CAPABILITY_IMAGE_GEN, CAPABILITY_IMAGE_INPUT],
    )
    request = _request()

    assert supports_reference_images(provider)
    assert supports_reference_images(provider.metadata)
    assert required_image_capabilities(request) == (CAPABILITY_IMAGE_INPUT,)
    assert prepare_image_request(provider, request) is request
    ensure_reference_support(provider, request)


def test_a_provider_without_support_is_rejected_instead_of_losing_references() -> None:
    provider = FakeImageProvider("text-to-image", [CAPABILITY_IMAGE_GEN])
    request = _request()

    assert not supports_reference_images(provider)
    with pytest.raises(ReferenceInputNotSupportedError) as caught:
        prepare_image_request(provider, request)

    assert isinstance(caught.value, InvalidRequestError)
    assert caught.value.provider == "text-to-image"
    assert caught.value.count == 1
    assert caught.value.details["required_capability"] == CAPABILITY_IMAGE_INPUT
    assert CAPABILITY_IMAGE_INPUT in str(caught.value)


def test_a_request_without_references_needs_no_extra_capability() -> None:
    provider = FakeImageProvider("text-to-image", [CAPABILITY_IMAGE_GEN])
    request = _request(references=False)

    assert required_image_capabilities(request) == ()
    assert filter_providers_for_reference_input([provider], request) == (provider,)
    assert prepare_image_request(provider, request) is request


def test_the_legacy_option_token_also_declares_support() -> None:
    provider = FakeImageProvider(
        "declarative",
        [CAPABILITY_IMAGE_GEN],
        supported_options={CAPABILITY_IMAGE_GEN: ["reference_images"]},
    )

    assert supports_reference_images(provider)


def test_filtering_keeps_only_providers_that_declare_support() -> None:
    text_only = FakeImageProvider("text-only", [CAPABILITY_IMAGE_GEN])
    with_references = FakeImageProvider(
        "with-references",
        [CAPABILITY_IMAGE_GEN, CAPABILITY_IMAGE_INPUT],
    )

    assert filter_providers_for_reference_input(
        [text_only, with_references],
        _request(),
    ) == (with_references,)


def test_the_router_falls_back_to_a_provider_that_accepts_references() -> None:
    text_only = FakeImageProvider("text-only", [CAPABILITY_IMAGE_GEN])
    with_references = FakeImageProvider(
        "with-references",
        [CAPABILITY_IMAGE_GEN, CAPABILITY_IMAGE_INPUT],
    )
    request = _request()

    selected = _router(text_only, with_references).select(
        CAPABILITY_IMAGE_GEN,
        required=required_image_capabilities(request),
    )

    assert selected is with_references


def test_the_router_reports_exhaustion_when_no_provider_accepts_references() -> None:
    text_only = FakeImageProvider("text-only", [CAPABILITY_IMAGE_GEN])
    request = _request()

    with pytest.raises(ProviderExhaustedError):
        _router(text_only).select(
            CAPABILITY_IMAGE_GEN,
            required=required_image_capabilities(request),
        )


def test_adaptation_honours_the_declared_image_input_capability() -> None:
    metadata = ProviderMeta(
        name="with-references",
        capabilities=[CAPABILITY_IMAGE_GEN, CAPABILITY_IMAGE_INPUT],
    )
    constraints = MediaConstraints.from_metadata(metadata, CAPABILITY_IMAGE_GEN)

    result = adapt_image_request(_request(), constraints)

    assert result.request.reference_images == [MediaInput(url=_URL)]
