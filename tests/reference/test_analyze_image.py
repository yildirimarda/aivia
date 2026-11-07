"""An image reference is analyzed into style, palette and a generic prompt."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Coroutine, Mapping
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from aivia.core.errors import ValidationError
from aivia.providers.protocols.llm import (
    ImagePart,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    TextPart,
)
from aivia.reference import (
    DEFAULT_REFERENCE_IMAGE_PROMPT,
    ImageReferenceAnalyzer,
    ReferenceAnalysis,
    ReferenceKind,
    ReferenceSpec,
    ShotType,
    analyze_image_reference,
    analyze_image_reference_offline,
    analyze_image_reference_sync,
    build_image_reference_request,
    parse_image_reference_analysis,
)


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine)


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


class FakeProvider:
    """An async LLM provider that returns one canned reference payload."""

    def __init__(self, payload: Mapping[str, Any] | str, *, name: str = "acme") -> None:
        self.payload = payload
        self.name = name
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        content = (
            self.payload if isinstance(self.payload, str) else json.dumps(self.payload)
        )
        return LLMResponse(
            content=content,
            provider=self.name,
            model="vision-model",
            usage=LLMUsage(input_tokens=10, output_tokens=5),
        )


def _payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "composition": {
            "scene_count": 1,
            "shot_types": ["wide"],
            "narrative_arc": "single focus",
        },
        "palette": {
            "colors": ["#123456", "#abcdef"],
            "primary": "#123456",
            "temperature": "cool",
        },
        "style": "editorial minimal",
        "typography": {
            "font_family": "Inter",
            "font_weight": "bold",
            "align": "left",
        },
        "subject": "a golden retriever",
        "reproducing_prompt": (
            "A golden retriever sits in a wide editorial frame with cool "
            "blue tones and bold sans-serif typography."
        ),
        "detected_text": "HELLO",
        "confidence": 0.8,
    }
    payload.update(overrides)
    return payload


def test_offline_analysis_reads_palette_aspect_and_still_composition() -> None:
    analysis = analyze_image_reference_offline(_image_bytes())

    assert isinstance(analysis, ReferenceAnalysis)
    assert analysis.kind is ReferenceKind.IMAGE
    assert analysis.palette.colors[:2] == ["#ff0000", "#0000ff"]
    assert analysis.palette.primary == "#ff0000"
    assert analysis.aspect.width == 100
    assert analysis.aspect.height == 50
    assert analysis.aspect.ratio == "2:1"
    assert analysis.structure.scene_count == 1
    assert analysis.description == ""
    assert analysis.typography.font_family == ""
    assert analysis.reproducing_prompt == ""


def test_offline_analysis_accepts_paths_specs_bytes_and_open_images(
    tmp_path: Path,
) -> None:
    path = tmp_path / "look.png"
    path.write_bytes(_image_bytes())
    spec = ReferenceSpec(path=str(path), label="look")

    for source in (path, str(path), spec, _image_bytes(), Image.open(path)):
        analysis = analyze_image_reference_offline(source)
        assert analysis.aspect.dimensions == (100, 50)

    assert analyze_image_reference_offline(spec).reference_id == "look"


def test_vision_analysis_maps_composition_palette_style_and_typography() -> None:
    provider = FakeProvider(_payload())

    analysis = _run(analyze_image_reference(_image_bytes(), provider=provider))

    assert analysis.structure.scene_count == 1
    assert analysis.structure.shot_types == [ShotType.WIDE]
    assert analysis.structure.narrative_arc == "single focus"
    assert analysis.composition is analysis.structure
    assert analysis.palette.colors == ["#123456", "#abcdef"]
    assert analysis.palette.primary == "#123456"
    assert analysis.style == "editorial minimal"
    assert analysis.typography.font_family == "Inter"
    assert analysis.typography.font_weight == 700
    assert analysis.detected_text == "HELLO"
    assert analysis.subject == "a golden retriever"
    assert analysis.confidence == 0.8
    assert analysis.provider == "acme"
    assert analysis.model == "vision-model"
    assert analysis.kind is ReferenceKind.IMAGE


def test_reproducing_prompt_never_names_the_original_subject() -> None:
    provider = FakeProvider(_payload())

    analysis = _run(analyze_image_reference(_image_bytes(), provider=provider))

    assert "golden retriever" not in analysis.reproducing_prompt.lower()
    assert "wide editorial frame" in analysis.reproducing_prompt.lower()
    assert analysis.prompt_suffix == "editorial minimal"


def test_prompt_asks_for_generic_style_and_carries_the_image_inline() -> None:
    provider = FakeProvider(_payload())
    data = _image_bytes()

    _run(analyze_image_reference(data, provider=provider, model="vision-model"))

    request = provider.requests[0]
    assert request.model == "vision-model"
    assert request.role == "reference_analysis"
    assert request.requires_vision is True
    assert request.response_schema is not None
    schema = request.response_schema
    assert {"structure", "palette", "typography"} <= set(schema["properties"])
    assert "reproducing_prompt" in schema["properties"]
    message = request.messages[0]
    text = next(part for part in message.content if isinstance(part, TextPart))
    image = next(part for part in message.content if isinstance(part, ImagePart))
    assert "composition" in text.text
    assert "palette" in text.text
    assert "typography" in text.text
    assert "subject" in text.text
    assert "never" in text.text or "without naming" in text.text
    assert image.media_type == "image/png"
    assert base64.b64decode(image.data or "") == data


def test_the_default_prompt_names_every_required_section() -> None:
    prompt = DEFAULT_REFERENCE_IMAGE_PROMPT.lower()

    assert "composition" in prompt
    assert "palette" in prompt
    assert "style" in prompt
    assert "typography" in prompt
    assert "reproducing_prompt" in prompt
    assert "subject" in prompt


def test_analysis_rejects_a_reference_that_is_not_an_image(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00\x01\x02\x03")

    with pytest.raises(ValidationError, match="not an image"):
        analyze_image_reference_offline(
            ReferenceSpec(kind=ReferenceKind.VIDEO, path=str(video))
        )


def test_analysis_rejects_a_remote_reference_before_any_download() -> None:
    with pytest.raises(ValidationError, match="downloaded"):
        analyze_image_reference_offline(
            ReferenceSpec(url="https://example.test/look.png")
        )


def test_parse_normalizes_aliases_and_merges_measured_fields() -> None:
    analysis = parse_image_reference_analysis(
        {
            "layout": {"scene_count": 2},
            "palette": {"colors": ["#112233"]},
            "style": "moody",
            "original_subject": "the old lighthouse",
            "generic_prompt": "The old lighthouse at dusk with moody tones.",
            "ocr_text": "KEEP OUT",
            "lang": "en",
        },
        source=_image_bytes(),
        reference_id="ref-1",
        provider="acme",
        model="vision-model",
    )

    assert analysis.structure.scene_count == 2
    assert analysis.style == "moody"
    assert analysis.palette.colors == ["#112233"]
    assert analysis.aspect.dimensions == (100, 50)
    assert analysis.detected_text == "KEEP OUT"
    assert analysis.language == "en"
    assert analysis.reference_id == "ref-1"
    assert analysis.provider == "acme"
    assert analysis.model == "vision-model"
    assert "lighthouse" not in analysis.reproducing_prompt.lower()
    assert "moody" in analysis.reproducing_prompt.lower()


def test_a_missing_prompt_is_synthesized_without_the_subject() -> None:
    provider = FakeProvider(_payload(reproducing_prompt=""))

    analysis = _run(analyze_image_reference(_image_bytes(), provider=provider))

    assert analysis.reproducing_prompt
    assert "golden retriever" not in analysis.reproducing_prompt.lower()
    assert "editorial minimal" in analysis.reproducing_prompt.lower()


def test_sync_wrapper_and_reusable_analyzer_share_the_result() -> None:
    provider = FakeProvider(_payload())

    direct = analyze_image_reference_sync(_image_bytes(), provider=provider)
    reusable = _run(ImageReferenceAnalyzer(provider).analyze(_image_bytes()))

    assert direct.reproducing_prompt == reusable.reproducing_prompt
    assert reusable.provider == "acme"


def test_build_request_can_be_inspected_without_a_provider() -> None:
    request = build_image_reference_request(_image_bytes(), model="vision-model")

    assert isinstance(request, LLMRequest)
    assert request.role == "reference_analysis"
    assert request.model == "vision-model"
    assert request.response_schema is not None
