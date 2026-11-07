"""Identifiable faces, logos and trademarks are detected, stripped or blocked."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine, Mapping
from dataclasses import FrozenInstanceError
from io import BytesIO
from typing import Any

import pytest
from PIL import Image

from aivia.core.errors import ContentPolicyError
from aivia.media.image_analysis import analyze_image
from aivia.providers.protocols.llm import LLMRequest, LLMResponse, LLMUsage
from aivia.reference import (
    Identity,
    IdentityAction,
    IdentityElement,
    IdentityError,
    block_identity,
    detect_identity,
    detect_identity_sync,
    enforce_identity,
    process_identity,
    redact_text,
    strip_identity,
)


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine)


def _image_bytes(size: tuple[int, int] = (100, 50)) -> bytes:
    image = Image.new("RGB", size, (255, 255, 255))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class FakeVisionProvider:
    """An async LLM provider that returns one canned vision payload."""

    def __init__(self, payload: Mapping[str, Any], *, name: str = "acme") -> None:
        self.payload = payload
        self.name = name
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(
            content=json.dumps(self.payload),
            provider=self.name,
            model="vision-model",
            usage=LLMUsage(input_tokens=10, output_tokens=5),
        )


def _payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "description": "a plain scene",
        "detected_text": "",
        "faces": [],
        "alt_text": "",
    }
    payload.update(overrides)
    return payload


def test_detect_identity_finds_faces() -> None:
    provider = FakeVisionProvider(
        _payload(faces=[{"x": 10.0, "y": 10.0, "width": 30.0, "height": 20.0}])
    )

    identity = _run(detect_identity(_image_bytes(), provider=provider))

    assert len(identity.faces) == 1
    face = identity.faces[0]
    assert face.kind == "face"
    assert face.box == (10.0, 10.0, 30.0, 20.0)
    assert identity.has_identity is True


def test_detect_identity_finds_trademarks_from_visible_text() -> None:
    provider = FakeVisionProvider(_payload(detected_text="Coca-Cola® and Nike"))

    identity = _run(detect_identity(_image_bytes(), provider=provider))

    labels = {t.label for t in identity.trademarks}
    assert "Coca-Cola" in labels
    assert "Nike" in labels
    assert identity.has_identity is True


def test_detect_identity_finds_logos_from_description() -> None:
    provider = FakeVisionProvider(_payload(description="a bag with the Acme logo"))

    identity = _run(detect_identity(_image_bytes(), provider=provider))

    assert len(identity.logos) == 1
    assert identity.logos[0].kind == "logo"
    assert identity.logos[0].label == "logo"
    assert identity.has_identity is True


def test_detect_identity_offline_is_empty() -> None:
    identity = _run(detect_identity(_image_bytes()))

    assert identity.faces == ()
    assert identity.logos == ()
    assert identity.trademarks == ()
    assert identity.has_identity is False


def test_detect_identity_ignores_plain_text_without_brands() -> None:
    provider = FakeVisionProvider(
        _payload(description="open field", detected_text="open today")
    )

    identity = _run(detect_identity(_image_bytes(), provider=provider))

    assert identity.trademarks == ()
    assert identity.logos == ()
    assert identity.faces == ()


def test_detect_identity_sync_matches_async() -> None:
    provider = FakeVisionProvider(
        _payload(faces=[{"x": 1.0, "y": 2.0, "width": 3.0, "height": 4.0}])
    )

    async_result = _run(detect_identity(_image_bytes(), provider=provider))
    sync_result = detect_identity_sync(_image_bytes(), provider=provider)

    assert sync_result.faces == async_result.faces


def test_strip_identity_blurs_located_faces() -> None:
    image = Image.new("RGB", (100, 100), (255, 255, 255))
    image.paste((255, 0, 0), (45, 45, 55, 55))
    before = image.getpixel((50, 50))

    identity = Identity(
        faces=(IdentityElement(kind="face", box=(40.0, 40.0, 20.0, 20.0)),)
    )
    stripped = strip_identity(image, identity)

    assert stripped is not image
    assert stripped.size == image.size
    assert stripped.getpixel((50, 50)) != before
    # the input image is never modified
    assert image.getpixel((50, 50)) == before


def test_strip_identity_ignores_elements_without_a_box() -> None:
    image = Image.new("RGB", (20, 20), (255, 255, 255))
    identity = Identity(trademarks=(IdentityElement(kind="trademark", label="Nike"),))
    stripped = strip_identity(image, identity)

    assert stripped.size == image.size


def test_redact_text_replaces_trademark_labels() -> None:
    trademarks = (
        IdentityElement(kind="trademark", label="Nike"),
        IdentityElement(kind="trademark", label="Coca-Cola"),
    )
    text = "I like Nike and Coca-Cola shoes."

    result = redact_text(text, trademarks)

    assert "Nike" not in result
    assert "Coca-Cola" not in result
    assert result.count("[REDACTED]") == 2


def test_redact_text_is_case_insensitive_and_word_bounded() -> None:
    trademarks = (IdentityElement(kind="trademark", label="Acme"),)
    text = "acme ACME Acme notacme"

    result = redact_text(text, trademarks)

    assert result == "[REDACTED] [REDACTED] [REDACTED] notacme"


def test_enforce_identity_blocks_with_a_clear_message() -> None:
    identity = Identity(
        faces=(IdentityElement(kind="face", box=(10.0, 10.0, 30.0, 20.0)),),
        trademarks=(IdentityElement(kind="trademark", label="Nike"),),
    )

    with pytest.raises(IdentityError) as exc:
        enforce_identity(identity, action=IdentityAction.BLOCK)

    error = exc.value
    assert isinstance(error, ContentPolicyError)
    assert error.code == "identity"
    assert error.retryable is False
    message = str(error)
    assert "face" in message
    assert "Nike" in message
    assert "cannot be used" in message


def test_enforce_identity_uses_the_supplied_message() -> None:
    identity = Identity(faces=(IdentityElement(kind="face", box=(1.0, 2.0, 3.0, 4.0)),))

    with pytest.raises(IdentityError, match="please supply a different photo"):
        enforce_identity(
            identity,
            action=IdentityAction.BLOCK,
            message="please supply a different photo",
        )


def test_enforce_identity_passes_when_clean() -> None:
    identity = Identity()

    result = enforce_identity(identity, action=IdentityAction.BLOCK)

    assert result is identity


def test_enforce_identity_returns_identity_when_stripping() -> None:
    identity = Identity(faces=(IdentityElement(kind="face", box=(1.0, 2.0, 3.0, 4.0)),))

    result = enforce_identity(identity, action=IdentityAction.STRIP)

    assert result is identity


def test_process_identity_strip_returns_blurred_image() -> None:
    image = Image.new("RGB", (100, 100), (255, 255, 255))
    image.paste((255, 0, 0), (45, 45, 55, 55))
    buffer = BytesIO()
    image.save(buffer, format="PNG")

    provider = FakeVisionProvider(
        _payload(faces=[{"x": 40.0, "y": 40.0, "width": 20.0, "height": 20.0}])
    )
    stripped, identity = _run(
        process_identity(
            buffer.getvalue(),
            action=IdentityAction.STRIP,
            provider=provider,
        )
    )

    assert identity.has_identity is True
    assert stripped.size == (100, 100)
    assert stripped.getpixel((50, 50)) != (255, 0, 0)


def test_process_identity_block_raises_before_returning_an_image() -> None:
    provider = FakeVisionProvider(
        _payload(faces=[{"x": 10.0, "y": 10.0, "width": 30.0, "height": 20.0}])
    )

    with pytest.raises(IdentityError):
        _run(
            process_identity(
                _image_bytes(), action=IdentityAction.BLOCK, provider=provider
            )
        )


def test_process_identity_block_passes_when_clean() -> None:
    image, identity = _run(
        process_identity(_image_bytes(), action=IdentityAction.BLOCK)
    )

    assert identity.has_identity is False
    assert image.size == (100, 50)


def test_block_identity_raises_on_detected_identity() -> None:
    provider = FakeVisionProvider(_payload(detected_text="Nike shoes"))

    with pytest.raises(IdentityError, match="Nike"):
        _run(block_identity(_image_bytes(), provider=provider))


def test_identity_is_immutable() -> None:
    identity = Identity(faces=(IdentityElement(kind="face", box=(1.0, 2.0, 3.0, 4.0)),))

    with pytest.raises((AttributeError, FrozenInstanceError)):
        identity.faces = ()  # type: ignore[misc]


def test_identity_elements_order_is_faces_logos_trademarks() -> None:
    identity = Identity(
        trademarks=(IdentityElement(kind="trademark", label="Nike"),),
        faces=(IdentityElement(kind="face", box=(1.0, 2.0, 3.0, 4.0)),),
        logos=(IdentityElement(kind="logo", label="logo"),),
    )

    kinds = [element.kind for element in identity.elements]
    assert kinds == ["face", "logo", "trademark"]


def test_analyze_image_still_returns_faces_for_reference() -> None:
    """The reference analyser keeps exposing face regions for identity detection."""
    provider = FakeVisionProvider(
        _payload(faces=[{"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4}])
    )
    analysis = _run(analyze_image(_image_bytes((100, 50)), provider=provider))

    assert len(analysis.faces) == 1
    assert analysis.faces[0].box == (10.0, 10.0, 30.0, 20.0)
