"""Video-to-video is routed only to providers that declare video input."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from aivia.core.config import CapabilityChainConfig, CapabilityStrategy
from aivia.core.errors import InvalidRequestError, ProviderExhaustedError
from aivia.core.models import ProviderMeta
from aivia.core.router import Router
from aivia.providers import (
    CAPABILITY_VIDEO_INPUT as TOP_LEVEL_VIDEO_INPUT,
)
from aivia.providers import Provider
from aivia.providers.protocols import (
    CAPABILITY_VIDEO_GEN,
    CAPABILITY_VIDEO_INPUT,
)
from aivia.providers.protocols.media import MediaInput
from aivia.providers.protocols.video import (
    CAPABILITY_VIDEO_TO_VIDEO,
    CAPABILITY_VIDEO_V2V,
    VideoRequest,
)
from aivia.providers.video import (
    VIDEO_TO_VIDEO_SKIP_REASON,
    FalVideoProvider,
    ReplicateVideoProvider,
    VeoVideoProvider,
    VideoCapabilityNotSupportedError,
    VideoFlags,
    VideoToVideoNotSupportedError,
    declared_video_flags,
    ensure_video_support,
    ensure_video_to_video_support,
    filter_providers_for_video_to_video,
    is_video_to_video,
    prepare_video_to_video_request,
    required_video_capabilities,
    required_video_to_video_capabilities,
    should_skip_video_to_video,
    supports_video_input,
    supports_video_to_video,
    video_to_video_skip_reason,
)

_SOURCE_URL = "https://example.com/media/source.mp4"


class FakeVideoProvider(Provider):
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


def _request(
    *,
    source: bool = True,
    media_type: str | None = "video/mp4",
) -> VideoRequest:
    source_video = (
        MediaInput(url=_SOURCE_URL, media_type=media_type) if source else None
    )
    return VideoRequest(prompt="restyle the clip", source_video=source_video)


def _router(*providers: Provider) -> Router:
    return Router(
        providers={provider.name: provider for provider in providers},
        config=CapabilityChainConfig(
            strategy=CapabilityStrategy.FIRST_AVAILABLE,
            chain=[provider.name for provider in providers],
        ),
    )


def test_the_video_input_capability_is_declared_and_exported() -> None:
    assert CAPABILITY_VIDEO_INPUT == "video_input"
    assert CAPABILITY_VIDEO_TO_VIDEO == CAPABILITY_VIDEO_INPUT
    assert CAPABILITY_VIDEO_V2V == CAPABILITY_VIDEO_INPUT
    assert TOP_LEVEL_VIDEO_INPUT == CAPABILITY_VIDEO_INPUT
    assert VideoFlags(video_input=True).video is True
    assert VideoFlags(video_input=True).capabilities == (CAPABILITY_VIDEO_INPUT,)


def test_a_source_video_is_carried_on_the_video_request() -> None:
    request = _request()

    assert is_video_to_video(request)
    assert request.source_video == MediaInput(
        url=_SOURCE_URL,
        media_type="video/mp4",
    )
    assert _request(source=False).source_video is None
    assert not is_video_to_video(_request(source=False))


def test_supports_reports_a_declared_capability_or_option_token() -> None:
    capable = FakeVideoProvider(
        "with-video-input",
        [CAPABILITY_VIDEO_GEN, CAPABILITY_VIDEO_INPUT],
    )
    declarative = FakeVideoProvider(
        "declarative",
        [CAPABILITY_VIDEO_GEN],
        supported_options={CAPABILITY_VIDEO_GEN: ["source_video"]},
    )
    text_only = FakeVideoProvider("text-only", [CAPABILITY_VIDEO_GEN])

    assert supports_video_input(capable)
    assert supports_video_to_video(capable)
    assert supports_video_input(declarative)
    assert not supports_video_input(text_only)
    assert declared_video_flags(capable) == VideoFlags(video_input=True)


def test_built_in_video_providers_document_the_skip() -> None:
    built_ins = (
        FalVideoProvider.default_metadata,
        ReplicateVideoProvider.default_metadata,
        VeoVideoProvider.default_metadata,
    )

    for metadata in built_ins:
        assert metadata is not None
        assert not supports_video_to_video(metadata)

    assert (
        video_to_video_skip_reason(built_ins, _request()) == VIDEO_TO_VIDEO_SKIP_REASON
    )
    assert "video_input" in VIDEO_TO_VIDEO_SKIP_REASON
    assert should_skip_video_to_video(built_ins, _request())


def test_required_capabilities_follow_the_source_clip() -> None:
    assert required_video_to_video_capabilities(_request(source=False)) == ()
    assert required_video_to_video_capabilities(_request()) == (CAPABILITY_VIDEO_INPUT,)
    assert required_video_capabilities(_request()) == (CAPABILITY_VIDEO_INPUT,)


def test_filtering_keeps_only_providers_that_declare_video_input() -> None:
    text_only = FakeVideoProvider("text-only", [CAPABILITY_VIDEO_GEN])
    capable = FakeVideoProvider(
        "with-video-input",
        [CAPABILITY_VIDEO_GEN, CAPABILITY_VIDEO_INPUT],
    )

    assert filter_providers_for_video_to_video(
        [text_only, capable],
        _request(),
    ) == (capable,)
    assert filter_providers_for_video_to_video(
        [text_only, capable],
        _request(source=False),
    ) == (text_only, capable)


def test_the_router_skips_providers_without_video_input() -> None:
    text_only = FakeVideoProvider("text-only", [CAPABILITY_VIDEO_GEN])
    capable = FakeVideoProvider(
        "with-video-input",
        [CAPABILITY_VIDEO_GEN, CAPABILITY_VIDEO_INPUT],
    )
    request = _request()

    selected = _router(text_only, capable).select(
        CAPABILITY_VIDEO_GEN,
        required=required_video_to_video_capabilities(request),
    )

    assert selected is capable
    assert video_to_video_skip_reason([text_only, capable], request) is None


def test_the_router_reports_exhaustion_when_no_provider_accepts_the_clip() -> None:
    text_only = FakeVideoProvider("text-only", [CAPABILITY_VIDEO_GEN])

    with pytest.raises(ProviderExhaustedError):
        _router(text_only).select(
            CAPABILITY_VIDEO_GEN,
            required=required_video_to_video_capabilities(_request()),
        )


def test_ensure_raises_a_dedicated_error_for_a_missing_capability() -> None:
    text_only = FakeVideoProvider("text-only", [CAPABILITY_VIDEO_GEN])

    with pytest.raises(VideoToVideoNotSupportedError) as caught:
        ensure_video_to_video_support(text_only, _request())

    assert isinstance(caught.value, InvalidRequestError)
    assert caught.value.provider == "text-only"
    assert caught.value.details["field"] == "source_video"
    assert caught.value.details["required_capability"] == CAPABILITY_VIDEO_INPUT
    assert CAPABILITY_VIDEO_INPUT in str(caught.value)


def test_ensure_and_prepare_are_no_ops_without_a_source_or_with_support() -> None:
    text_only = FakeVideoProvider("text-only", [CAPABILITY_VIDEO_GEN])
    capable = FakeVideoProvider(
        "with-video-input",
        [CAPABILITY_VIDEO_GEN, CAPABILITY_VIDEO_INPUT],
    )

    ensure_video_to_video_support(text_only, _request(source=False))
    request = _request()
    assert prepare_video_to_video_request(capable, request) is request
    ensure_video_to_video_support(capable, request)


def test_the_flags_guard_also_rejects_a_source_video() -> None:
    text_only = FakeVideoProvider("text-only", [CAPABILITY_VIDEO_GEN])

    with pytest.raises(VideoCapabilityNotSupportedError) as caught:
        ensure_video_support(text_only, _request())

    assert caught.value.details["field"] == "source_video"
    assert caught.value.details["required_capability"] == CAPABILITY_VIDEO_INPUT


def test_skip_reason_is_none_for_a_plain_text_to_video_request() -> None:
    text_only = FakeVideoProvider("text-only", [CAPABILITY_VIDEO_GEN])

    assert video_to_video_skip_reason([text_only], _request(source=False)) is None
    assert not should_skip_video_to_video([text_only], _request(source=False))


def test_fal_rejects_a_source_video_instead_of_dropping_it() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("the source video must be rejected before any call")

    provider = FalVideoProvider(
        api_key="test-key",
        default_model="fal-ai/kling-video/v1.6/pro/text-to-video",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(InvalidRequestError) as caught:
        asyncio.run(provider.generate(_request()))

    assert caught.value.details["field"] == "source_video"
    assert caught.value.details["required_capability"] == CAPABILITY_VIDEO_INPUT


def test_a_source_video_with_a_non_video_media_type_is_rejected() -> None:
    capable = FakeVideoProvider(
        "with-video-input",
        [CAPABILITY_VIDEO_GEN, CAPABILITY_VIDEO_INPUT],
    )

    with pytest.raises(InvalidRequestError) as caught:
        prepare_video_to_video_request(
            capable,
            _request(media_type="image/png"),
        )

    assert caught.value.details["field"] == "source_video"
    assert caught.value.details["media_type"] == "image/png"


def test_a_source_video_without_a_media_type_is_still_a_clip() -> None:
    capable = FakeVideoProvider(
        "with-video-input",
        [CAPABILITY_VIDEO_GEN, CAPABILITY_VIDEO_INPUT],
    )
    request = _request(media_type=None)

    assert prepare_video_to_video_request(capable, request) is request


def test_fal_rejects_a_non_video_source_before_any_call() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("the source video must be rejected before any call")

    provider = FalVideoProvider(
        api_key="test-key",
        default_model="fal-ai/kling-video/v1.6/pro/text-to-video",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(InvalidRequestError) as caught:
        asyncio.run(provider.generate(_request(media_type="image/png")))

    assert caught.value.details["field"] == "source_video"
    assert caught.value.details["media_type"] == "image/png"
