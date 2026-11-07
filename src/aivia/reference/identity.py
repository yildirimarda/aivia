"""Detect and handle identifiable faces, logos and trademarks in references.

An image reference may show identifiable people (faces) or brands (logos and
trademarks). Before a reference guides generation, each detected element is
either stripped (anonymized in place) or the reference is blocked with a clear
message. The detection itself is best-effort: face rectangles come from a vision
analyser, trademarks are read from the visible text and logos are read from the
vision description. No vendor name, model id or brand list is hard-coded here.

The decision to strip or block is explicit so no silent copying of an
identifiable reference ever happens.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from PIL import Image, ImageFilter

from aivia.core.errors import ContentPolicyError
from aivia.core.models.asset import FaceRegion
from aivia.media.image_analysis import ImageAnalysis, analyze_image
from aivia.media.image_utils import open_image

__all__ = [
    "Identity",
    "IdentityAction",
    "IdentityElement",
    "IdentityError",
    "block_identity",
    "detect_identity",
    "detect_identity_sync",
    "enforce_identity",
    "process_identity",
    "redact_text",
    "strip_faces",
    "strip_identity",
]

class IdentityAction(StrEnum):
    """What to do with an identified element in a reference."""

    STRIP = "strip"
    BLOCK = "block"


class IdentityError(ContentPolicyError):
    """A reference contains identifiable people or brands and is blocked.

    The message lists the detected elements so the caller can show a clear,
    actionable message to the person who supplied the reference.
    """

    code = "identity"


@dataclass(frozen=True, slots=True)
class IdentityElement:
    """One identifiable element found in a reference.

    ``kind`` is ``face``, ``logo`` or ``trademark``. ``box`` holds the pixel
    rectangle ``(x, y, width, height)`` when the analyser located the element,
    otherwise it is ``None``. ``label`` is the human readable name, such as the
    detected text for a trademark or a description for a logo.
    """

    kind: str
    label: str = ""
    box: tuple[float, float, float, float] | None = None
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class Identity:
    """The identifiable elements found in one reference.

    ``faces`` carry pixel rectangles. ``logos`` and ``trademarks`` carry labels
    and may carry boxes when the analyser located them.
    """

    faces: tuple[IdentityElement, ...] = field(default_factory=tuple)
    logos: tuple[IdentityElement, ...] = field(default_factory=tuple)
    trademarks: tuple[IdentityElement, ...] = field(default_factory=tuple)

    @property
    def elements(self) -> tuple[IdentityElement, ...]:
        """Every detected element, faces first then logos then trademarks."""
        return (*self.faces, *self.logos, *self.trademarks)

    @property
    def has_identity(self) -> bool:
        """Whether at least one identifiable element was found."""
        return bool(self.elements)


_TRADEMARK_SYMBOL = re.compile(r"[®™©]")
_PROPER_NOUN = re.compile(r"\b[A-Z][a-z0-9]{2,}(?:['-][A-Z][a-z0-9]{2,})*\b")
_ALL_CAPS = re.compile(r"\b[A-Z0-9][A-Z0-9]{2,}\b")


def _trademarks(text: str) -> list[str]:
    """Return best-effort trademark candidates from visible text.

    A token counts when it carries a trademark symbol (®™©), is all-caps, or is
    a title-cased proper noun of at least three characters. Results are
    de-duplicated while preserving first-seen order.
    """
    found: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", text):
        if token in seen:
            continue
        if (
            _TRADEMARK_SYMBOL.search(token)
            or _ALL_CAPS.fullmatch(token)
            or _PROPER_NOUN.fullmatch(token)
        ):
            seen.add(token)
            found.append(token)
    return found


def _logos(text: str) -> list[str]:
    """Return logo candidates from a vision description.

    Any description that mentions a logo is treated as containing one brand
    mark. Results are de-duplicated while preserving first-seen order.
    """
    found: list[str] = []
    if "logo" in text.lower() and "logo" not in found:
        found.append("logo")
    return found


def _element_from_face(face: FaceRegion) -> IdentityElement:
    return IdentityElement(
        kind="face",
        box=(face.x, face.y, face.width, face.height),
        confidence=face.confidence,
    )


async def _analysis_for(
    source: Any,
    *,
    analyzer: Any = None,
    provider: Any = None,
    helper: Any = None,
    model: str | None = None,
) -> ImageAnalysis:
    return await analyze_image(
        source,
        analyzer=analyzer,
        provider=provider,
        helper=helper,
        model=model,
    )


async def detect_identity(
    source: Any,
    *,
    analyzer: Any = None,
    provider: Any = None,
    helper: Any = None,
    model: str | None = None,
) -> Identity:
    """Detect the identifiable elements in *source*.

    *source* may be an image path, raw bytes or an open
    :class:`~PIL.Image.Image`. When an analyser, provider or helper is supplied
    the image is sent through vision analysis and faces, visible text and the
    description are inspected. Without one this is an offline pass and returns
    an empty identity.
    """
    analysis = await _analysis_for(
        source,
        analyzer=analyzer,
        provider=provider,
        helper=helper,
        model=model,
    )
    faces = tuple(_element_from_face(face) for face in analysis.faces)
    logos = tuple(
        IdentityElement(kind="logo", label=label)
        for label in _logos(analysis.description)
    )
    trademarks = tuple(
        IdentityElement(kind="trademark", label=label)
        for label in _trademarks(analysis.detected_text)
    )
    return Identity(faces=faces, logos=logos, trademarks=trademarks)


def detect_identity_sync(
    source: Any,
    *,
    analyzer: Any = None,
    provider: Any = None,
    helper: Any = None,
    model: str | None = None,
) -> Identity:
    """Run :func:`detect_identity` from synchronous code."""
    return asyncio.run(
        detect_identity(
            source,
            analyzer=analyzer,
            provider=provider,
            helper=helper,
            model=model,
        )
    )


def strip_faces(image: Image.Image, faces: tuple[IdentityElement, ...]) -> Image.Image:
    """Return a copy of *image* with every boxed face region blurred.

    The input image is never modified. Faces without a box are ignored. The
    blur radius scales with the smaller face side so small marks stay legible
    while large faces are anonymized.
    """
    work = image.copy()
    if work.mode not in ("L", "RGB", "RGBA"):
        work = work.convert("RGBA")
    for face in faces:
        if face.kind != "face" or face.box is None:
            continue
        x, y, width, height = (float(value) for value in face.box)
        left = max(0, round(x))
        top = max(0, round(y))
        right = min(work.width, round(x + width))
        bottom = min(work.height, round(y + height))
        if right <= left or bottom <= top:
            continue
        region = work.crop((left, top, right, bottom))
        radius = max(1, min(right - left, bottom - top) // 4)
        region = region.filter(ImageFilter.GaussianBlur(radius=float(radius)))
        work.paste(region, (left, top))
    return work


def strip_identity(image: Image.Image, identity: Identity) -> Image.Image:
    """Return a copy of *image* with every located identifiable element blurred.

    Only elements that carry a pixel box can be removed from the image itself
    (faces and any located logos or trademarks). Elements reported only as
    labels survive as text; callers should redact those with
    :func:`redact_text` before publishing derived text.
    """
    boxed = [element for element in identity.elements if element.box is not None]
    return strip_faces(image, tuple(boxed))


def redact_text(text: str, trademarks: tuple[IdentityElement, ...]) -> str:
    """Return *text* with every trademark label replaced by ``[REDACTED]``.

    Matching is case-insensitive and word-bounded so a trademark never leaves a
    partial word behind. Trademarks without a label are ignored.
    """
    result = text
    for trademark in trademarks:
        label = trademark.label.strip()
        if not label:
            continue
        result = re.sub(
            rf"\b{re.escape(label)}\b",
            "[REDACTED]",
            result,
            flags=re.IGNORECASE,
        )
    return result


def enforce_identity(
    identity: Identity,
    *,
    action: IdentityAction = IdentityAction.BLOCK,
    message: str | None = None,
) -> Identity:
    """Return *identity* when safe, otherwise raise :class:`IdentityError`.

    With :attr:`IdentityAction.BLOCK` any detectable identity raises. With
    :attr:`IdentityAction.STRIP` the identity is returned unchanged so the
    caller can strip it with :func:`strip_identity`. The raised message lists
    every detected element so it can be shown to the person who supplied the
    reference.
    """
    if action is IdentityAction.BLOCK and identity.has_identity:
        parts = [
            f"{element.kind}: {element.label or element.box}"
            for element in identity.elements
        ]
        detail = "; ".join(parts)
        raise IdentityError(
            message
            or (
                "reference contains identifiable "
                f"{detail} and cannot be used as a reference; strip them or "
                "supply a different reference"
            ),
            details={
                "identity": [
                    element.label or str(element.box)
                    for element in identity.elements
                ]
            },
        )
    return identity


async def block_identity(
    source: Any,
    *,
    message: str | None = None,
    analyzer: Any = None,
    provider: Any = None,
    helper: Any = None,
    model: str | None = None,
) -> None:
    """Detect identity in *source* and raise :class:`IdentityError` if found.

    Use this when a reference must never contain identifiable people or brands.
    """
    identity = await detect_identity(
        source,
        analyzer=analyzer,
        provider=provider,
        helper=helper,
        model=model,
    )
    enforce_identity(identity, action=IdentityAction.BLOCK, message=message)


async def process_identity(
    source: Any,
    *,
    action: IdentityAction = IdentityAction.BLOCK,
    message: str | None = None,
    analyzer: Any = None,
    provider: Any = None,
    helper: Any = None,
    model: str | None = None,
) -> tuple[Image.Image, Identity]:
    """Detect identity in *source* and either strip or block it.

    Returns the (possibly stripped) image and the detected identity. With
    :attr:`IdentityAction.BLOCK` the call raises :class:`IdentityError` before
    returning an image when anything is detected. With
    :attr:`IdentityAction.STRIP` the image is returned with every located
    element blurred.
    """
    image = open_image(source)
    identity = await detect_identity(
        source,
        analyzer=analyzer,
        provider=provider,
        helper=helper,
        model=model,
    )
    enforce_identity(identity, action=action, message=message)
    if action is IdentityAction.STRIP:
        image = strip_identity(image, identity)
    return image, identity
