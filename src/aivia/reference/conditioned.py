"""Reference conditioned generation paths.

A conditioned path pairs a generation request with a visual reference and
forwards the reference only to providers that declare the matching input
capability. When no configured provider can consume references, the output is
rebuilt scene by scene: every scene in the reference is rendered individually
with a prompt-only request and the per-scene results are returned together.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from aivia.media.image_service import ImageOutcome, ImageService
from aivia.media.video_service import VideoOutcome, VideoService
from aivia.providers.base import Provider
from aivia.providers.image import filter_providers_for_reference_input
from aivia.providers.protocols import (
    CAPABILITY_IMAGE_GEN,
    CAPABILITY_IMAGE_INPUT,
    CAPABILITY_VIDEO_FIRST_FRAME,
    CAPABILITY_VIDEO_GEN,
    CAPABILITY_VIDEO_INPUT,
    ImageRequest,
    MediaInput,
    VideoRequest,
)
from aivia.reference.review import SpecReview, review_style_spec
from aivia.reference.schema import (
    ReferenceAnalysis,
    ReferenceSpec,
    SceneSpec,
    StyleSpec,
)

__all__ = [
    "CAPABILITY_IMAGE_GEN",
    "CAPABILITY_IMAGE_INPUT",
    "CAPABILITY_VIDEO_FIRST_FRAME",
    "CAPABILITY_VIDEO_GEN",
    "CAPABILITY_VIDEO_INPUT",
    "ConditionedPathResult",
    "SpecReview",
    "generate_first_frame_to_video_conditioned",
    "generate_image_to_image_conditioned",
    "generate_text_to_image_conditioned",
    "generate_video_to_video_conditioned",
    "rebuild_scene_by_scene",
    "review_style_spec",
]


@dataclass(frozen=True, slots=True)
class ConditionedPathResult:
    """Outcome of one reference conditioned generation path.

    ``outcomes`` holds every produced asset or clip, in order. ``scene_count``
    is the number of scenes rendered. ``rebuilt`` is true when the output was
    rebuilt scene by scene because no provider could consume the reference.
    ``warning`` is the fallback message, when any, and ``used_references`` is
    true when the reference was forwarded to a provider instead of rebuilt.
    """

    outcomes: tuple[Any, ...]
    scene_count: int
    rebuilt: bool
    warning: str | None
    used_references: bool
    review_declined: bool = False
    """True when spec review was enabled and declined, so no generation ran."""


def _scenes_of(reference: ReferenceAnalysis | ReferenceSpec | None) -> list[SceneSpec]:
    """Return the scenes carried by *reference*, if any."""
    if reference is None:
        return []
    structure = getattr(reference, "structure", None)
    if structure is None:
        analysis = getattr(reference, "analysis", None)
        if analysis is not None:
            structure = getattr(analysis, "structure", None)
    if structure is None:
        return []
    return list(getattr(structure, "scenes", []) or [])


def _media_from_reference(
    reference: ReferenceAnalysis | ReferenceSpec | None,
) -> list[MediaInput]:
    """Turn a :class:`ReferenceSpec` into the media input a provider receives.

    A :class:`ReferenceAnalysis` carries no transportable media of its own, so
    it yields nothing. A :class:`ReferenceSpec` yields one input from its URL or
    local file when one is available.
    """
    if reference is None or not isinstance(reference, ReferenceSpec):
        return []
    media_type = reference.mime_type or None
    if reference.url:
        return [MediaInput(url=reference.url, media_type=media_type)]
    if reference.path:
        path = Path(reference.path)
        if path.is_file():
            return [MediaInput(data=path.read_bytes(), media_type=media_type)]
    return []


def _spec_from_reference(
    reference: ReferenceAnalysis | ReferenceSpec | None,
) -> StyleSpec | None:
    """Return the learned style spec carried by *reference*, if any.

    A :class:`ReferenceAnalysis` is itself a style spec; a :class:`ReferenceSpec`
    carries one in ``analysis``. A spec with no learned sections is not worth
    reviewing.
    """
    if reference is None:
        return None
    analysis = getattr(reference, "analysis", None)
    if analysis is None:
        analysis = getattr(reference, "style_spec", None)
    if analysis is None:
        return None
    if getattr(analysis, "is_empty", True):
        return None
    return cast(StyleSpec, analysis)


def _providers_support(
    providers: Sequence[Provider],
    capability: str,
) -> tuple[Provider, ...]:
    """Return the providers that declare *capability*."""
    return tuple(
        provider for provider in providers if provider.metadata.supports(capability)
    )


def _scene_prompt(base_prompt: str, scene: SceneSpec, index: int) -> str:
    """Build the prompt used to render one scene during a scene-by-scene rebuild."""
    parts = [base_prompt.strip()]
    if scene.description:
        parts.append(scene.description.strip())
    parts.append(f"scene {index + 1}")
    return " ".join(part for part in parts if part)


async def _conditioned_path(
    *,
    generate_one: Callable[[Any, Any], Awaitable[Any]],
    request: Any,
    reference: ReferenceAnalysis | ReferenceSpec | None,
    media: list[MediaInput],
    supported: tuple[Provider, ...],
    attach_reference: Callable[[Any, list[MediaInput]], Any],
    detach_reference: Callable[[Any], Any],
    build_scene_requests: Callable[[Any, list[SceneSpec]], list[Any]],
    screen_prompt: Callable[[str], Any] | None,
    default_token: Any,
    run_id: str,
    generate_values: Mapping[str, Any],
    review_enabled: bool = False,
    review_confirm: bool | None = None,
) -> ConditionedPathResult:
    """Run one reference conditioned path end to end.

    When references can be forwarded to at least one capable provider they are
    sent with a single request. Otherwise the output is rebuilt scene by scene,
    or rendered once prompt-only with the reference stripped when the reference
    holds a single scene.

    When *review_enabled* is true the extracted style spec is shown for edit or
    confirmation before any paid generation. A declined review returns without
    calling the provider.
    """
    scenes = _scenes_of(reference)
    scene_count = len(scenes) or 1

    if review_enabled:
        spec = _spec_from_reference(reference)
        if spec is not None:
            review = review_style_spec(spec, enabled=True, confirm=review_confirm)
            if not review.confirmed:
                return ConditionedPathResult(
                    outcomes=(),
                    scene_count=scene_count,
                    rebuilt=False,
                    warning="style spec review declined; no paid generation",
                    used_references=False,
                    review_declined=True,
                )

    forwardable = bool(media and supported)

    if forwardable:
        attached = attach_reference(request, media)
        outcome = await generate_one(attached, default_token)
        return ConditionedPathResult(
            outcomes=(outcome,),
            scene_count=scene_count,
            rebuilt=False,
            warning=None,
            used_references=True,
        )

    if media:
        warning = (
            "no provider declares the reference input capability; "
            "rebuilding the reference scene by scene without forwarding it"
        )
    else:
        warning = None

    if len(scenes) > 1:
        outcomes = await rebuild_scene_by_scene(
            generate_one,
            request,
            scenes,
            screen_prompt=screen_prompt,
            default_token=default_token,
            run_id=run_id,
            generate_values=generate_values,
            build_scene_requests=build_scene_requests,
        )
        return ConditionedPathResult(
            outcomes=tuple(outcomes),
            scene_count=scene_count,
            rebuilt=True,
            warning=warning,
            used_references=False,
        )

    outcome = await generate_one(detach_reference(request), default_token)
    return ConditionedPathResult(
        outcomes=(outcome,),
        scene_count=scene_count,
        rebuilt=False,
        warning=warning,
        used_references=False,
    )


async def rebuild_scene_by_scene(
    generate_one: Callable[[Any, Any], Awaitable[Any]],
    request: Any,
    scenes: Sequence[SceneSpec],
    *,
    screen_prompt: Callable[[str], Any] | None,
    default_token: Any,
    run_id: str,
    generate_values: Mapping[str, Any],
    build_scene_requests: Callable[[Any, list[SceneSpec]], list[Any]],
) -> list[Any]:
    """Render every scene of *scenes* individually and return the outcomes.

    Each scene prompt is screened before submission so the per-scene token
    matches the text being sent.
    """
    scene_requests = build_scene_requests(request, list(scenes))
    outcomes: list[Any] = []
    for scene_request in scene_requests:
        token = default_token
        if screen_prompt is not None:
            token = screen_prompt(scene_request.prompt)
        outcomes.append(await generate_one(scene_request, token))
    return outcomes


def _image_scene_requests(
    request: ImageRequest, scenes: list[SceneSpec]
) -> list[ImageRequest]:
    """Build one prompt-only image request per scene."""
    return [
        request.model_copy(
            update={
                "reference_images": [],
                "count": 1,
                "prompt": _scene_prompt(request.prompt, scene, index),
            }
        )
        for index, scene in enumerate(scenes)
    ]


def _video_scene_requests(
    request: VideoRequest, scenes: list[SceneSpec]
) -> list[VideoRequest]:
    """Build one prompt-only video request per scene."""
    scene_requests: list[VideoRequest] = []
    for index, scene in enumerate(scenes):
        update: dict[str, Any] = {
            "first_frame": None,
            "source_video": None,
            "count": 1,
            "prompt": _scene_prompt(request.prompt, scene, index),
        }
        if scene.duration_seconds is not None:
            update["duration_seconds"] = scene.duration_seconds
        scene_requests.append(request.model_copy(update=update))
    return scene_requests


def _make_image_generate_one(
    service: ImageService,
    active_token: Any,
    values: Mapping[str, Any],
    run_id: str,
) -> Callable[[ImageRequest, Any], Awaitable[ImageOutcome]]:
    async def generate_one(req: ImageRequest, token: Any = None) -> ImageOutcome:
        return await service.generate(
            req,
            token=token if token is not None else active_token,
            run_id=run_id,
            **values,
        )

    return generate_one


def _make_video_generate_one(
    service: VideoService,
    active_token: Any,
    values: Mapping[str, Any],
    run_id: str,
) -> Callable[[VideoRequest, Any], Awaitable[VideoOutcome]]:
    async def generate_one(req: VideoRequest, token: Any = None) -> VideoOutcome:
        return await service.generate(
            req,
            token=token if token is not None else active_token,
            run_id=run_id,
            **values,
        )

    return generate_one


async def generate_text_to_image_conditioned(
    service: ImageService,
    request: ImageRequest,
    *,
    reference: ReferenceAnalysis | ReferenceSpec | None = None,
    reference_media: Sequence[MediaInput] | None = None,
    token: Any = None,
    screening_token: Any = None,
    screen_token: Any = None,
    run_id: str = "",
    review_enabled: bool = False,
    review_confirm: bool | None = None,
    **values: Any,
) -> ConditionedPathResult:
    """Generate an image from a text prompt conditioned on a reference.

    The reference images are forwarded to a provider that declares
    :data:`~aivia.providers.protocols.CAPABILITY_IMAGE_INPUT`. When no
    configured provider can consume them the reference is rebuilt scene by
    scene: one image per scene, each rendered prompt-only.

    When *review_enabled* is true the extracted style spec is shown for edit or
    confirmation before any paid generation. A declined review returns without
    calling the provider.
    """
    active_token = token if token is not None else screening_token or screen_token
    if active_token is None:
        active_token = service.screen(request.prompt)

    media = list(reference_media or _media_from_reference(reference))
    providers = tuple(service.router.providers.values())
    supported = (
        filter_providers_for_reference_input(
            providers, request.model_copy(update={"reference_images": media})
        )
        if media
        else ()
    )

    return await _conditioned_path(
        generate_one=_make_image_generate_one(service, active_token, values, run_id),
        request=request,
        reference=reference,
        media=media,
        supported=supported,
        attach_reference=lambda req, m: req.model_copy(
            update={"reference_images": list(m)}
        ),
        detach_reference=lambda req: req.model_copy(update={"reference_images": []}),
        build_scene_requests=_image_scene_requests,
        screen_prompt=service.screen,
        default_token=active_token,
        run_id=run_id,
        generate_values=values,
        review_enabled=review_enabled,
        review_confirm=review_confirm,
    )


async def generate_image_to_image_conditioned(
    service: ImageService,
    request: ImageRequest,
    *,
    reference: ReferenceAnalysis | ReferenceSpec | None = None,
    reference_media: Sequence[MediaInput] | None = None,
    token: Any = None,
    screening_token: Any = None,
    screen_token: Any = None,
    run_id: str = "",
    review_enabled: bool = False,
    review_confirm: bool | None = None,
    **values: Any,
) -> ConditionedPathResult:
    """Generate an image from an input image conditioned on a reference.

    The source image travels with *request* and the reference conditions the
    result. References are forwarded to a provider that declares
    :data:`CAPABILITY_IMAGE_INPUT`; otherwise the output is rebuilt scene by
    scene.

    When *review_enabled* is true the extracted style spec is shown for edit or
    confirmation before any paid generation. A declined review returns without
    calling the provider.
    """
    active_token = token if token is not None else screening_token or screen_token
    if active_token is None:
        active_token = service.screen(request.prompt)

    media = list(reference_media or _media_from_reference(reference))
    providers = tuple(service.router.providers.values())
    supported = (
        filter_providers_for_reference_input(
            providers, request.model_copy(update={"reference_images": media})
        )
        if media
        else ()
    )

    return await _conditioned_path(
        generate_one=_make_image_generate_one(service, active_token, values, run_id),
        request=request,
        reference=reference,
        media=media,
        supported=supported,
        attach_reference=lambda req, m: req.model_copy(
            update={"reference_images": list(m)}
        ),
        detach_reference=lambda req: req.model_copy(update={"reference_images": []}),
        build_scene_requests=_image_scene_requests,
        screen_prompt=service.screen,
        default_token=active_token,
        run_id=run_id,
        generate_values=values,
        review_enabled=review_enabled,
        review_confirm=review_confirm,
    )


async def generate_first_frame_to_video_conditioned(
    service: VideoService,
    request: VideoRequest,
    *,
    reference: ReferenceAnalysis | ReferenceSpec | None = None,
    reference_media: Sequence[MediaInput] | None = None,
    token: Any = None,
    screening_token: Any = None,
    screen_token: Any = None,
    run_id: str = "",
    review_enabled: bool = False,
    review_confirm: bool | None = None,
    **values: Any,
) -> ConditionedPathResult:
    """Generate a video from a first frame conditioned on a reference.

    The first reference image becomes the request's first frame and is sent to
    a provider that declares
    :data:`~aivia.providers.protocols.CAPABILITY_VIDEO_FIRST_FRAME`. When no
    configured provider can consume it the reference is rebuilt scene by scene:
    one clip per scene, each rendered prompt-only.

    When *review_enabled* is true the extracted style spec is shown for edit or
    confirmation before any paid generation. A declined review returns without
    calling the provider.
    """
    active_token = token if token is not None else screening_token or screen_token
    if active_token is None:
        active_token = service.screen(request.prompt)

    media = list(reference_media or _media_from_reference(reference))
    providers = tuple(service.router.providers.values())
    supported = (
        _providers_support(providers, CAPABILITY_VIDEO_FIRST_FRAME) if media else ()
    )

    return await _conditioned_path(
        generate_one=_make_video_generate_one(service, active_token, values, run_id),
        request=request,
        reference=reference,
        media=media,
        supported=supported,
        attach_reference=lambda req, m: req.model_copy(update={"first_frame": m[0]}),
        detach_reference=lambda req: req.model_copy(
            update={"first_frame": None, "source_video": None}
        ),
        build_scene_requests=_video_scene_requests,
        screen_prompt=service.screen,
        default_token=active_token,
        run_id=run_id,
        generate_values=values,
        review_enabled=review_enabled,
        review_confirm=review_confirm,
    )


async def generate_video_to_video_conditioned(
    service: VideoService,
    request: VideoRequest,
    *,
    reference: ReferenceAnalysis | ReferenceSpec | None = None,
    reference_media: Sequence[MediaInput] | None = None,
    token: Any = None,
    screening_token: Any = None,
    screen_token: Any = None,
    run_id: str = "",
    review_enabled: bool = False,
    review_confirm: bool | None = None,
    **values: Any,
) -> ConditionedPathResult:
    """Restyle a source video conditioned on a reference.

    The request must already carry its :attr:`VideoRequest.source_video`. The
    source clip is forwarded to a provider that declares
    :data:`~aivia.providers.protocols.CAPABILITY_VIDEO_INPUT`. When no
    configured provider can consume it the reference is rebuilt scene by scene:
    one clip per scene, each rendered prompt-only.

    When *review_enabled* is true the extracted style spec is shown for edit or
    confirmation before any paid generation. A declined review returns without
    calling the provider.
    """
    active_token = token if token is not None else screening_token or screen_token
    if active_token is None:
        active_token = service.screen(request.prompt)

    media = list(reference_media or _media_from_reference(reference)) or (
        [request.source_video] if request.source_video is not None else []
    )
    providers = tuple(service.router.providers.values())
    supported = _providers_support(providers, CAPABILITY_VIDEO_INPUT) if media else ()

    return await _conditioned_path(
        generate_one=_make_video_generate_one(service, active_token, values, run_id),
        request=request,
        reference=reference,
        media=media,
        supported=supported,
        attach_reference=lambda req, m: req,
        detach_reference=lambda req: req.model_copy(
            update={"first_frame": None, "source_video": None}
        ),
        build_scene_requests=_video_scene_requests,
        screen_prompt=service.screen,
        default_token=active_token,
        run_id=run_id,
        generate_values=values,
        review_enabled=review_enabled,
        review_confirm=review_confirm,
    )
