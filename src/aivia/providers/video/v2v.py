"""Video-to-video support policy and the documented skip.

A video-to-video request hands an existing clip to a provider and asks for a
transformed result. No built-in adapter declares the ``video_input`` capability
yet: fal, Replicate, Veo and the mock render text-to-video with optional still
frames only. A source clip therefore has to be routed only to a provider that
declares the capability, and when the configured chain has none the flow must
record an explicit skip instead of silently dropping the clip.

This module owns that decision. It re-exports the capability check and filter
from :mod:`aivia.providers.video.flags`, adds a dedicated error and a
human-readable skip reason so callers can log why a restyle did not run.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from aivia.core.errors import InvalidRequestError
from aivia.core.models.provider import ProviderMeta
from aivia.providers.base import Provider
from aivia.providers.protocols.video import (
    CAPABILITY_VIDEO_INPUT,
    VideoRequest,
)
from aivia.providers.video.flags import supports_video_input

__all__ = [
    "CAPABILITY_VIDEO_INPUT",
    "SOURCE_VIDEO_MEDIA_PREFIX",
    "VIDEO_TO_VIDEO_SKIP_REASON",
    "VideoToVideoNotSupportedError",
    "ensure_source_video_media_type",
    "ensure_video_to_video_support",
    "filter_providers_for_video_to_video",
    "is_video_to_video",
    "prepare_video_to_video_request",
    "required_video_to_video_capabilities",
    "should_skip_video_to_video",
    "supports_video_to_video",
    "video_to_video_skip_reason",
]

SOURCE_VIDEO_MEDIA_PREFIX = "video/"
"""The media type prefix a source clip must declare."""

VIDEO_TO_VIDEO_SKIP_REASON = (
    "no configured provider declares video_input support; video-to-video is skipped"
)
"""Why a video-to-video request is skipped when no provider can accept it."""


def _metadata(
    source: Provider | ProviderMeta | Mapping[str, Any],
) -> ProviderMeta:
    if isinstance(source, ProviderMeta):
        return source
    if isinstance(source, Provider):
        return source.metadata
    if isinstance(source, Mapping):
        return ProviderMeta.model_validate(source)
    raise TypeError("source must be a Provider, ProviderMeta or mapping")


class VideoToVideoNotSupportedError(InvalidRequestError):
    """A source clip was sent to a provider without ``video_input`` support."""

    code = "video_to_video_not_supported"

    def __init__(
        self,
        provider: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.provider = provider
        merged: dict[str, Any] = {
            "provider": provider,
            "field": "source_video",
            "required_capability": CAPABILITY_VIDEO_INPUT,
        }
        if details:
            merged.update(details)
        super().__init__(
            f"provider {provider!r} does not declare {CAPABILITY_VIDEO_INPUT} "
            "support for video-to-video",
            details=merged,
        )


def is_video_to_video(request: VideoRequest) -> bool:
    """Whether *request* carries a source clip for video-to-video work."""
    if not isinstance(request, VideoRequest):
        raise TypeError("request must be a VideoRequest")
    return request.source_video is not None


def ensure_source_video_media_type(request: VideoRequest) -> None:
    """Raise when *request* carries a source clip that is not a video.

    A missing media type is accepted because it may still be a clip; an
    explicit non-video type is rejected before any provider is called so a
    still image or an audio track cannot be submitted as a source video.
    """
    if not isinstance(request, VideoRequest):
        raise TypeError("request must be a VideoRequest")
    source = request.source_video
    if source is None or source.media_type is None:
        return
    if source.media_type.startswith(SOURCE_VIDEO_MEDIA_PREFIX):
        return
    raise InvalidRequestError(
        "source video must be a video media type",
        details={
            "field": "source_video",
            "media_type": source.media_type,
        },
    )


def required_video_to_video_capabilities(
    request: VideoRequest,
) -> tuple[str, ...]:
    """Return the capabilities *request* needs to transform its source clip."""
    if not is_video_to_video(request):
        return ()
    ensure_source_video_media_type(request)
    return (CAPABILITY_VIDEO_INPUT,)


def supports_video_to_video(
    source: Provider | ProviderMeta | Mapping[str, Any],
) -> bool:
    """Whether *source* declares that it accepts a source video."""
    return supports_video_input(source)


def filter_providers_for_video_to_video(
    providers: Iterable[Provider],
    request: VideoRequest,
) -> tuple[Provider, ...]:
    """Return only providers that can receive *request* with its source clip."""
    candidates = tuple(providers)
    if not is_video_to_video(request):
        return candidates
    ensure_source_video_media_type(request)
    return tuple(
        provider for provider in candidates if supports_video_to_video(provider)
    )


def video_to_video_skip_reason(
    providers: Iterable[Provider | ProviderMeta | Mapping[str, Any]],
    request: VideoRequest | None = None,
) -> str | None:
    """Return why video-to-video is skipped, or ``None`` when it can run.

    Without *request* the answer only asks whether any candidate declares the
    capability. With a request that carries no source clip nothing is skipped,
    because it is an ordinary text-to-video call.
    """
    if request is not None and not isinstance(request, VideoRequest):
        raise TypeError("request must be a VideoRequest or None")
    if request is not None and not is_video_to_video(request):
        return None
    if request is not None:
        ensure_source_video_media_type(request)
    if any(supports_video_to_video(provider) for provider in providers):
        return None
    return VIDEO_TO_VIDEO_SKIP_REASON


def should_skip_video_to_video(
    providers: Iterable[Provider | ProviderMeta | Mapping[str, Any]],
    request: VideoRequest | None = None,
) -> bool:
    """Whether *providers* cannot serve the video-to-video request."""
    return video_to_video_skip_reason(providers, request) is not None


def ensure_video_to_video_support(
    source: Provider | ProviderMeta | Mapping[str, Any],
    request: VideoRequest,
) -> None:
    """Raise when *request* carries a source clip *source* cannot accept."""
    if not is_video_to_video(request):
        return
    ensure_source_video_media_type(request)
    if supports_video_to_video(source):
        return
    raise VideoToVideoNotSupportedError(_metadata(source).name or "provider")


def prepare_video_to_video_request(
    source: Provider | ProviderMeta | Mapping[str, Any],
    request: VideoRequest,
) -> VideoRequest:
    """Return *request* after proving *source* can receive its source clip."""
    ensure_video_to_video_support(source, request)
    return request
