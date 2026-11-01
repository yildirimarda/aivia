"""Image analysis reports pixels offline and vision fields through an LLM."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Coroutine, Iterator
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from sqlalchemy import Engine

from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.errors import ValidationError
from aivia.core.models.asset import AssetAnalysis, FaceRegion
from aivia.llm.structured import StructuredOutputHelper
from aivia.media.image_analysis import (
    ImageAnalysis,
    ImageAnalysisCache,
    ImageAnalyzer,
    analyze_image,
    analyze_image_offline,
    analyze_image_sync,
    build_analysis_request,
    describe_image,
    detect_faces,
    detect_text,
    dominant_colors,
    generate_alt_text,
    image_content_hash,
    image_dimensions,
)
from aivia.providers.protocols.llm import (
    ChatMessage,
    ImagePart,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    MessageRole,
    TextPart,
)


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine)


@pytest.fixture
def engine() -> Iterator[Engine]:
    created = create_database_engine(url="sqlite+pysqlite:///:memory:")
    upgrade_database(engine=created)
    try:
        yield created
    finally:
        created.dispose()


def _image_bytes(
    size: tuple[int, int] = (100, 50),
    *,
    left: tuple[int, int, int, int] = (255, 0, 0, 255),
    right: tuple[int, int, int, int] = (0, 0, 255, 255),
) -> bytes:
    image = Image.new("RGBA", size, left)
    image.paste(right, (size[0] // 2, 0, size[0], size[1]))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _solid_bytes(
    color: tuple[int, int, int, int],
    size: tuple[int, int] = (20, 10),
) -> bytes:
    image = Image.new("RGBA", size, color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class FakeVisionProvider:
    """An async LLM provider that returns one canned analysis payload."""

    def __init__(self, payload: dict[str, Any] | str) -> None:
        self.payload = payload
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        content = (
            self.payload if isinstance(self.payload, str) else json.dumps(self.payload)
        )
        return LLMResponse(
            content=content,
            provider="acme",
            model="vision-model",
            usage=LLMUsage(input_tokens=10, output_tokens=5),
        )


def test_offline_analysis_reports_dimensions_and_dominant_colors() -> None:
    analysis = analyze_image_offline(_image_bytes())

    assert analysis.width == 100
    assert analysis.height == 50
    assert analysis.dimensions == (100, 50)
    assert analysis.colors[:2] == ["#ff0000", "#0000ff"]
    assert analysis.description == ""
    assert analysis.detected_text == ""
    assert analysis.faces == []
    assert analysis.alt_text == ""


def test_offline_analysis_accepts_a_path_and_an_open_image(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    path.write_bytes(_image_bytes())

    assert analyze_image_offline(path).dimensions == (100, 50)
    assert analyze_image_offline(Image.open(path)).dimensions == (100, 50)
    assert image_dimensions(path) == (100, 50)


def test_a_solid_image_has_one_dominant_color() -> None:
    assert dominant_colors(_solid_bytes((12, 34, 56, 255))) == ["#0c2238"]


def test_transparent_pixels_are_ignored() -> None:
    image = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    image.paste((10, 20, 30, 255), (0, 0, 5, 10))
    buffer = BytesIO()
    image.save(buffer, format="PNG")

    assert dominant_colors(buffer.getvalue()) == ["#0a141e"]


def test_an_entirely_transparent_image_has_no_colors() -> None:
    assert dominant_colors(_solid_bytes((1, 2, 3, 0))) == []


def test_color_count_is_validated_and_max_colors_is_an_alias() -> None:
    data = _image_bytes()

    with pytest.raises(ValidationError, match="positive integer"):
        dominant_colors(data, 0)
    with pytest.raises(ValidationError, match="integer"):
        dominant_colors(data, 2.5)  # type: ignore[arg-type]

    assert dominant_colors(data, max_colors=1) == ["#ff0000"]


def test_generate_alt_text_prefers_description_and_falls_back() -> None:
    assert generate_alt_text("  A red car  ") == "A red car"
    assert generate_alt_text("", detected_text="HELLO") == "HELLO"
    assert generate_alt_text("", fallback="An image") == "An image"
    assert generate_alt_text("") == ""


def test_generate_alt_text_truncates_on_a_word_boundary() -> None:
    text = generate_alt_text("one two three four", max_length=10)

    assert text == "one two…"
    assert len(text) <= 10
    with pytest.raises(ValidationError, match="positive"):
        generate_alt_text("text", max_length=0)


def test_a_vision_provider_enriches_the_offline_analysis() -> None:
    provider = FakeVisionProvider(
        {
            "description": "A red square beside a blue square.",
            "detected_text": "SALE",
            "faces": [{"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4}],
            "alt_text": "",
        }
    )

    analysis = _run(analyze_image(_image_bytes(), provider=provider))

    assert analysis.width == 100
    assert analysis.height == 50
    assert analysis.colors[:2] == ["#ff0000", "#0000ff"]
    assert analysis.description == "A red square beside a blue square."
    assert analysis.detected_text == "SALE"
    assert analysis.alt_text == "A red square beside a blue square."
    assert len(analysis.faces) == 1
    assert analysis.faces[0].box == (10.0, 10.0, 30.0, 20.0)


def test_a_vision_result_records_the_provider_and_model() -> None:
    provider = FakeVisionProvider(
        {"description": "A red square.", "alt_text": "A red square."}
    )

    analysis = _run(analyze_image(_image_bytes(), provider=provider))

    assert analysis.provider == "acme"
    assert analysis.model == "vision-model"


def test_the_vision_request_carries_the_image_inline() -> None:
    provider = FakeVisionProvider({"description": "A picture", "alt_text": "A picture"})
    data = _image_bytes()

    _run(analyze_image(data, provider=provider, model="vision-model"))

    request = provider.requests[0]
    assert request.model == "vision-model"
    assert request.role == "vision_qa"
    assert request.requires_vision is True
    message = request.messages[0]
    assert isinstance(message, ChatMessage)
    assert message.role is MessageRole.USER
    assert isinstance(message.content, list)
    text = next(part for part in message.content if isinstance(part, TextPart))
    image = next(part for part in message.content if isinstance(part, ImagePart))
    assert "alt text" in text.text
    assert image.media_type == "image/png"
    assert base64.b64decode(image.data or "") == data


def test_vision_accepts_an_open_image() -> None:
    provider = FakeVisionProvider({"description": "open", "alt_text": "open"})
    image = Image.open(BytesIO(_image_bytes()))

    analysis = _run(analyze_image(image, provider=provider))

    assert analysis.description == "open"
    image_part = next(
        part
        for part in provider.requests[0].messages[0].content
        if isinstance(part, ImagePart)
    )
    assert image_part.media_type == "image/png"


def test_analyze_without_a_provider_is_the_offline_result() -> None:
    offline = analyze_image_offline(_image_bytes())
    analysis = _run(analyze_image(_image_bytes()))

    assert analysis == offline


def test_a_custom_analyzer_may_be_sync_or_async() -> None:
    def sync_analyzer(source: str | Path | bytes) -> dict[str, Any]:
        assert source
        return {"description": "sync", "alt_text": "sync"}

    async def async_analyzer(source: str | Path | bytes) -> dict[str, Any]:
        assert source
        return {"description": "async", "alt_text": "async"}

    assert _run(analyze_image(_image_bytes(), analyzer=sync_analyzer)).description == (
        "sync"
    )
    assert _run(analyze_image(_image_bytes(), analyzer=async_analyzer)).description == (
        "async"
    )


def test_a_helper_is_accepted_for_cost_aware_calls() -> None:
    provider = FakeVisionProvider({"description": "helped", "alt_text": "helped"})
    helper = StructuredOutputHelper(provider)

    analysis = _run(analyze_image(_image_bytes(), helper=helper))

    assert analysis.description == "helped"
    assert analysis.alt_text == "helped"


def test_convenience_helpers_return_one_field() -> None:
    provider = FakeVisionProvider(
        {
            "description": "a cat",
            "detected_text": "MEOW",
            "faces": [[1, 2, 3, 4]],
        }
    )

    assert _run(describe_image(_image_bytes(), provider=provider)) == "a cat"
    assert _run(detect_text(_image_bytes(), provider=provider)) == "MEOW"
    faces = _run(detect_faces(_image_bytes(), provider=provider))
    assert [face.box for face in faces] == [(1.0, 2.0, 3.0, 4.0)]


def test_the_analyzer_class_reuses_settings_offline() -> None:
    analyzer = ImageAnalyzer(color_count=1)

    analysis = _run(analyzer.analyze(_image_bytes()))

    assert analyzer.has_vision is False
    assert analysis.colors == ["#ff0000"]
    assert analyzer.offline(_image_bytes()).colors == ["#ff0000"]


def test_the_analyzer_class_uses_a_vision_provider() -> None:
    provider = FakeVisionProvider({"description": "seen", "alt_text": "seen"})
    analyzer = ImageAnalyzer(provider, model="vision-model", color_count=2)

    analysis = _run(analyzer(_image_bytes()))

    assert analyzer.has_vision is True
    assert analysis.description == "seen"
    assert provider.requests[0].model == "vision-model"


def test_the_result_converts_to_and_from_an_asset_analysis() -> None:
    analysis = ImageAnalysis(
        width=10,
        height=5,
        colors=["#ffffff"],
        description="white",
        detected_text="TEXT",
        faces=[FaceRegion(x=1, y=2, width=3, height=4)],
        alt_text="white rectangle",
        provider="acme",
        model="vision-model",
    )

    asset_analysis = analysis.to_asset_analysis()
    restored = ImageAnalysis.from_asset_analysis(
        asset_analysis,
        alt_text=analysis.alt_text,
    )

    assert isinstance(asset_analysis, AssetAnalysis)
    assert asset_analysis.description == "white"
    assert asset_analysis.provider == "acme"
    assert asset_analysis.model == "vision-model"
    assert restored == analysis


def test_analysis_provenance_can_override_the_stored_analysis() -> None:
    analysis = ImageAnalysis(description="white", provider="stored", model="stored-1")

    restored = ImageAnalysis.from_asset_analysis(
        analysis.to_asset_analysis(),
        provider="fallback",
        model="fallback-1",
    )

    assert restored.provider == "fallback"
    assert restored.model == "fallback-1"


def test_build_analysis_request_rejects_an_empty_prompt() -> None:
    with pytest.raises(ValidationError, match="prompt"):
        build_analysis_request(_image_bytes(), prompt="  ")


def test_analyze_image_sync_wraps_the_async_function() -> None:
    assert analyze_image_sync(_image_bytes()).width == 100


def test_the_content_hash_is_the_same_for_bytes_a_path_and_an_open_image(
    tmp_path: Path,
) -> None:
    data = _image_bytes()
    path = tmp_path / "image.png"
    path.write_bytes(data)
    opened = Image.open(path)

    expected = image_content_hash(data)
    assert image_content_hash(path) == expected
    assert image_content_hash(opened) == expected
    assert image_content_hash(_solid_bytes((0, 0, 0, 255))) != expected


def test_vision_analysis_is_cached_by_content_hash(engine: Engine) -> None:
    provider = FakeVisionProvider({"description": "cached", "alt_text": "cached"})
    cache = ImageAnalysisCache(engine)
    data = _image_bytes()

    first = _run(analyze_image(data, provider=provider, cache=cache))
    second = _run(analyze_image(data, provider=provider, cache=cache))

    assert first == second
    assert second.description == "cached"
    assert len(provider.requests) == 1


def test_a_cached_analysis_survives_a_new_cache_instance(engine: Engine) -> None:
    data = _image_bytes()
    first_provider = FakeVisionProvider(
        {"description": "from the first call", "alt_text": "first"},
    )
    _run(
        analyze_image(
            data,
            provider=first_provider,
            cache=ImageAnalysisCache(engine),
        )
    )

    second_provider = FakeVisionProvider(
        {"description": "from the second call", "alt_text": "second"},
    )
    analysis = _run(
        analyze_image(
            data,
            provider=second_provider,
            cache=ImageAnalysisCache(engine),
        )
    )

    assert second_provider.requests == []
    assert analysis.description == "from the first call"
    assert analysis.alt_text == "first"


def test_analysis_provenance_survives_a_new_cache_instance(engine: Engine) -> None:
    data = _image_bytes()
    first_provider = FakeVisionProvider(
        {"description": "from the first call", "alt_text": "first"},
    )
    first = _run(
        analyze_image(
            data,
            provider=first_provider,
            cache=ImageAnalysisCache(engine),
        )
    )

    second_provider = FakeVisionProvider(
        {"description": "from the second call", "alt_text": "second"},
    )
    second = _run(
        analyze_image(
            data,
            provider=second_provider,
            cache=ImageAnalysisCache(engine),
        )
    )

    assert second_provider.requests == []
    assert first.provider == second.provider == "acme"
    assert first.model == second.model == "vision-model"


def test_changed_image_content_is_not_served_from_the_cache(engine: Engine) -> None:
    provider = FakeVisionProvider({"description": "seen", "alt_text": "seen"})
    cache = ImageAnalysisCache(engine)

    _run(analyze_image(_image_bytes(), provider=provider, cache=cache))
    _run(
        analyze_image(
            _solid_bytes((1, 2, 3, 255)),
            provider=provider,
            cache=cache,
        )
    )

    assert len(provider.requests) == 2


def test_no_cache_bypasses_lookup_and_store(engine: Engine) -> None:
    provider = FakeVisionProvider({"description": "fresh", "alt_text": "fresh"})
    cache = ImageAnalysisCache(engine)
    data = _image_bytes()

    _run(analyze_image(data, provider=provider, cache=cache))
    _run(analyze_image(data, provider=provider, cache=cache, no_cache=True))
    _run(analyze_image(data, provider=provider, cache=cache, no_cache=True))

    assert len(provider.requests) == 3


def test_a_cache_hit_still_recomputes_offline_fields(engine: Engine) -> None:
    provider = FakeVisionProvider({"description": "cached", "alt_text": "cached"})
    cache = ImageAnalysisCache(engine)
    data = _image_bytes()

    _run(analyze_image(data, provider=provider, cache=cache, color_count=1))
    analysis = _run(analyze_image(data, provider=provider, cache=cache, color_count=2))

    assert len(provider.requests) == 1
    assert analysis.colors == ["#ff0000", "#0000ff"]


def test_the_analyzer_class_reuses_the_analysis_cache(engine: Engine) -> None:
    provider = FakeVisionProvider({"description": "reused", "alt_text": "reused"})
    analyzer = ImageAnalyzer(provider, cache=ImageAnalysisCache(engine))
    data = _image_bytes()

    first = _run(analyzer(data))
    second = _run(analyzer(data))

    assert first == second
    assert analyzer.cache is not None
    assert len(provider.requests) == 1


def test_a_different_prompt_does_not_reuse_the_cached_analysis(
    engine: Engine,
) -> None:
    provider = FakeVisionProvider({"description": "seen", "alt_text": "seen"})
    cache = ImageAnalysisCache(engine)
    data = _image_bytes()

    _run(analyze_image(data, provider=provider, cache=cache, prompt="first"))
    _run(analyze_image(data, provider=provider, cache=cache, prompt="second"))

    assert len(provider.requests) == 2


def test_a_cache_hit_records_a_zero_cost_event(engine: Engine) -> None:
    provider = FakeVisionProvider({"description": "seen", "alt_text": "seen"})
    cache = ImageAnalysisCache(engine)
    data = _image_bytes()

    _run(analyze_image(data, provider=provider, cache=cache))
    _run(analyze_image(data, provider=provider, cache=cache))

    assert cache.cache is not None
    events = cache.cache.cost.events()
    assert len(events) == 1
    assert events[0].amount_usd == 0.0
    assert events[0].metadata["cache"] == "hit"
