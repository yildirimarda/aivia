"""The visual similarity gate regenerates or refuses near copies of a reference."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from io import BytesIO
from typing import Any

import pytest
from PIL import Image

from aivia.core.errors import ValidationError
from aivia.reference import (
    DEFAULT_HASH_SIZE,
    VisualGateError,
    VisualGateResult,
    average_hash,
    embedding_similarity,
    gate_visual_similarity,
    hamming_similarity,
    perceptual_similarity,
    visual_similarity,
)


def _run(coro: Any) -> Any:
    import asyncio

    return asyncio.run(coro)


def _solid_png(color: tuple[int, int, int], size: int = 16) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (size, size), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _gradient_png(size: int = 32) -> bytes:
    buffer = BytesIO()
    image = Image.new("RGB", (size, size))
    for y in range(size):
        for x in range(size):
            image.putpixel((x, y), (x * 8, y * 8, 128))
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _transposed_gradient_png(size: int = 32) -> bytes:
    buffer = BytesIO()
    image = Image.new("RGB", (size, size))
    for y in range(size):
        for x in range(size):
            image.putpixel((x, y), (y * 8, x * 8, 128))
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _brightness_vector(image: Image.Image) -> list[float]:
    """Map an image to [1, 0, 0] when bright, [-1, 0, 0] when dark."""
    grayscale = image.convert("L")
    pixels = list(grayscale.getdata())
    mean = sum(pixels) / len(pixels)
    return [1.0, 0.0, 0.0] if mean > 128 else [-1.0, 0.0, 0.0]


async def _embed_by_brightness() -> Callable[[Image.Image], Awaitable[list[float]]]:
    async def embed(image: Image.Image) -> list[float]:
        return _brightness_vector(image)

    return embed


def test_average_hash_is_deterministic() -> None:
    image = Image.new("RGB", (16, 16), (200, 100, 50))
    first = average_hash(image)
    second = average_hash(image)
    assert first == second
    assert len(first) == DEFAULT_HASH_SIZE * DEFAULT_HASH_SIZE // 4


def test_average_hash_differs_for_different_images() -> None:
    left = average_hash(_gradient_png())
    right = average_hash(_transposed_gradient_png())
    assert left != right


def test_average_hash_rejects_bad_size() -> None:
    image = Image.new("RGB", (8, 8), (10, 10, 10))
    with pytest.raises(ValueError, match="hash size must be a positive integer"):
        average_hash(image, size=0)
    with pytest.raises(ValueError, match="hash size must be a positive integer"):
        average_hash(image, size=True)  # type: ignore[arg-type]


def test_hamming_similarity_identical_is_one() -> None:
    digest = average_hash(_gradient_png())
    assert hamming_similarity(digest, digest) == 1.0


def test_hamming_similarity_different_is_below_one() -> None:
    left = average_hash(_gradient_png())
    right = average_hash(_transposed_gradient_png())
    score = hamming_similarity(left, right)
    assert 0.0 <= score < 1.0


def test_hamming_similarity_pads_different_lengths() -> None:
    assert hamming_similarity("ff", "00") == 0.0
    assert hamming_similarity("ff", "f") == hamming_similarity("ff", "0f")


def test_perceptual_similarity_identical_is_one() -> None:
    image = _gradient_png()
    assert perceptual_similarity(image, image) == 1.0


def test_perceptual_similarity_different_is_lower() -> None:
    score = perceptual_similarity(_gradient_png(), _transposed_gradient_png())
    assert 0.0 <= score < 1.0


def test_perceptual_similarity_accepts_bytes_and_path(tmp_path) -> None:
    red_bytes = _gradient_png()
    path = tmp_path / "red.png"
    path.write_bytes(red_bytes)
    score = perceptual_similarity(red_bytes, path)
    assert score == 1.0


def test_embedding_similarity_identical_vectors_is_one() -> None:
    score = _run(
        embedding_similarity(
            _run(_embed_by_brightness()),
            _solid_png((200, 200, 200)),
            _solid_png((240, 240, 240)),
        )
    )
    assert score == 1.0


def test_embedding_similarity_opposite_is_zero() -> None:
    score = _run(
        embedding_similarity(
            _run(_embed_by_brightness()),
            _solid_png((200, 200, 200)),
            _solid_png((10, 10, 10)),
        )
    )
    assert score == 0.0


def test_visual_similarity_perceptual_only() -> None:
    score, method = _run(visual_similarity(_gradient_png(), _transposed_gradient_png()))
    assert method == "perceptual_hash"
    assert 0.0 <= score <= 1.0


def test_visual_similarity_combined_with_embedding() -> None:
    score, method = _run(
        visual_similarity(
            _solid_png((200, 200, 200)),
            _solid_png((10, 10, 10)),
            embed=_run(_embed_by_brightness()),
        )
    )
    assert method == "combined"
    # perceptual hash of two solid colours collides (all zeros) -> 1.0,
    # embedding is opposite -> 0.0, so the combined score stays 1.0.
    assert score == 1.0


def test_gate_passes_below_threshold() -> None:
    result = _run(
        gate_visual_similarity(
            reference=_gradient_png(),
            candidate=_transposed_gradient_png(),
            threshold=0.9,
        )
    )
    assert isinstance(result, VisualGateResult)
    assert result.passed is True
    assert result.refused is False
    assert result.regenerated == 0
    assert result.reason is None
    assert result.score < 0.9


def test_gate_refuses_at_threshold_without_regenerator() -> None:
    identical = _gradient_png()
    result = _run(
        gate_visual_similarity(
            reference=identical,
            candidate=identical,
            threshold=0.9,
        )
    )
    assert result.passed is False
    assert result.refused is True
    assert result.regenerated == 0
    assert result.score >= 0.9
    assert result.reason is not None


def test_gate_regenerates_then_passes() -> None:
    reference = _gradient_png()
    copies = [reference, _solid_png((30, 30, 30))]
    index = 0

    async def regenerate() -> Any:
        nonlocal index
        image = copies[index]
        index += 1
        return image

    result = _run(
        gate_visual_similarity(
            reference=reference,
            candidate=reference,
            threshold=0.9,
            max_regenerations=2,
            regenerate=regenerate,
        )
    )
    assert result.passed is True
    assert result.refused is False
    assert result.regenerated == 2
    assert result.score < 0.9


def test_gate_refuses_after_exhausting_regenerations() -> None:
    reference = _gradient_png()

    async def regenerate() -> Any:
        return reference

    result = _run(
        gate_visual_similarity(
            reference=reference,
            candidate=reference,
            threshold=0.9,
            max_regenerations=2,
            regenerate=regenerate,
        )
    )
    assert result.passed is False
    assert result.refused is True
    assert result.regenerated == 2
    assert result.score >= 0.9
    assert "after 2 regeneration(s)" in result.reason


def test_gate_refuses_immediately_when_no_regenerator() -> None:
    reference = _gradient_png()
    result = _run(
        gate_visual_similarity(
            reference=reference,
            candidate=reference,
            threshold=0.9,
            max_regenerations=3,
        )
    )
    assert result.passed is False
    assert result.refused is True
    assert result.regenerated == 0
    assert "no regenerator is configured" in result.reason


def test_gate_zero_regenerations_refuses_above_limit() -> None:
    reference = _gradient_png()
    result = _run(
        gate_visual_similarity(
            reference=reference,
            candidate=reference,
            threshold=0.9,
            max_regenerations=0,
            regenerate=lambda: _transposed_gradient_png(),
        )
    )
    assert result.passed is False
    assert result.refused is True
    assert result.regenerated == 0


def test_gate_rejects_bad_threshold() -> None:
    with pytest.raises((TypeError, ValueError)):
        _run(
            gate_visual_similarity(
                reference=_solid_png((1, 1, 1)),
                candidate=_solid_png((2, 2, 2)),
                threshold=1.5,
            )
        )
    with pytest.raises((TypeError, ValueError)):
        _run(
            gate_visual_similarity(
                reference=_solid_png((1, 1, 1)),
                candidate=_solid_png((2, 2, 2)),
                threshold=-0.1,
            )
        )


def test_gate_rejects_bad_max_regenerations() -> None:
    with pytest.raises((TypeError, ValueError)):
        _run(
            gate_visual_similarity(
                reference=_solid_png((1, 1, 1)),
                candidate=_solid_png((2, 2, 2)),
                threshold=0.5,
                max_regenerations=-1,
            )
        )


def test_visual_gate_error_is_a_validation_error() -> None:
    error = VisualGateError("too similar")
    assert isinstance(error, ValidationError)
    assert error.code == "visual_similarity"
    assert error.retryable is False
    assert str(error) == "too similar"


def test_visual_gate_result_is_immutable() -> None:
    result = VisualGateResult(
        passed=True,
        score=0.1,
        method="perceptual_hash",
        regenerated=0,
        refused=False,
        reason=None,
    )
    with pytest.raises(AttributeError):
        result.score = 0.9  # type: ignore[misc]
