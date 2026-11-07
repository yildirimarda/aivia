"""A text reference is analyzed for tone, structure, hook pattern and language."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine, Mapping
from pathlib import Path
from typing import Any

import pytest

from aivia.core.errors import ValidationError
from aivia.providers.protocols.llm import (
    LLMRequest,
    LLMResponse,
    LLMUsage,
    TextPart,
)
from aivia.reference import (
    DEFAULT_REFERENCE_TEXT_PROMPT,
    ReferenceAnalysis,
    ReferenceKind,
    ReferenceSpec,
    TextReferenceAnalyzer,
    analyze_text_reference,
    analyze_text_reference_offline,
    analyze_text_reference_sync,
    build_text_reference_request,
    parse_text_reference_analysis,
    sample_text_reference,
)

TEXT = """# Why Most People Get This Wrong

Most people get this wrong. The usual advice sounds reasonable, but it fails.

## The common mistake

You were told to work harder. Actually, the evidence points the other way.

## What to do instead

Try a smaller experiment first. Then measure what happened.
"""


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine)


class FakeProvider:
    """An async LLM provider that returns one canned text payload."""

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
            model="text-model",
            usage=LLMUsage(input_tokens=10, output_tokens=5),
        )


def _payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "tone": "dry and urgent",
        "structure": {
            "sections": [
                {"name": "hook", "description": "opens with a contrarian claim"},
                {"name": "evidence", "description": "supports the claim"},
                {"name": "payoff", "description": "closes the loop"},
            ],
            "hook": "Most people get this wrong.",
            "narrative_arc": "claim to evidence to payoff",
            "beats": ["hook", "evidence", "payoff"],
        },
        "hook_pattern": "contrarian",
        "language": "en",
        "subject": "The Old Advice",
        "reproducing_prompt": (
            "Use The Old Advice with a dry tone and a contrarian hook."
        ),
        "confidence": 0.82,
    }
    payload.update(overrides)
    return payload


def test_offline_analysis_reads_structure_hook_pattern_and_language() -> None:
    analysis = analyze_text_reference_offline(TEXT)

    assert isinstance(analysis, ReferenceAnalysis)
    assert analysis.kind is ReferenceKind.TEXT
    assert analysis.language == "en"
    assert analysis.tone == ""
    assert analysis.hook_pattern == "contrarian"
    assert analysis.structure.hook == "Most people get this wrong."
    assert analysis.structure.beats == [
        "Why Most People Get This Wrong",
        "The common mistake",
        "What to do instead",
    ]
    assert analysis.structure.scene_count == 3
    assert [scene.description for scene in analysis.structure.scenes] == [
        "Why Most People Get This Wrong",
        "The common mistake",
        "What to do instead",
    ]
    assert analysis.structure.hook_seconds == pytest.approx(2.0)
    assert analysis.duration_seconds == pytest.approx(19.2)
    assert analysis.pacing.words_per_second == pytest.approx(2.5)
    assert analysis.pacing.style == "measured"
    assert analysis.metadata["word_count"] == 48
    assert analysis.metadata["paragraph_count"] == 3
    assert analysis.metadata["section_count"] == 3
    assert analysis.detected_text == TEXT.strip()
    assert analysis.reproducing_prompt == ""


def test_offline_analysis_accepts_paths_specs_bytes_and_raw_text(
    tmp_path: Path,
) -> None:
    path = tmp_path / "article.md"
    path.write_text(TEXT, encoding="utf-8")
    spec = ReferenceSpec(path=str(path), label="article")

    for source in (path, str(path), spec, TEXT.encode("utf-8"), TEXT):
        analysis = analyze_text_reference_offline(source)
        assert analysis.kind is ReferenceKind.TEXT
        assert analysis.language == "en"
        assert analysis.hook_pattern == "contrarian"

    assert analyze_text_reference_offline(spec).reference_id == "article"


def test_offline_analysis_uses_paragraphs_when_there_are_no_headings() -> None:
    analysis = analyze_text_reference_offline(
        "First paragraph opens the idea. It adds one sentence.\n\n"
        "Second paragraph explains why. It ends with a question?"
    )

    assert analysis.structure.scene_count == 2
    assert analysis.structure.beats == [
        "First paragraph opens the idea.",
        "Second paragraph explains why.",
    ]
    assert analysis.structure.hook == "First paragraph opens the idea."
    assert analysis.metadata["paragraph_count"] == 2


def test_analysis_maps_tone_structure_hook_pattern_and_language() -> None:
    provider = FakeProvider(_payload())

    analysis = _run(analyze_text_reference(TEXT, provider=provider))

    assert analysis.kind is ReferenceKind.TEXT
    assert analysis.tone == "dry and urgent"
    assert analysis.metadata["tone"] == "dry and urgent"
    assert analysis.hook_pattern == "contrarian"
    assert analysis.metadata["hook_pattern"] == "contrarian"
    assert analysis.language == "en"
    assert analysis.structure.hook == "Most people get this wrong."
    assert analysis.structure.narrative_arc == "claim to evidence to payoff"
    assert analysis.structure.beats == ["hook", "evidence", "payoff"]
    assert analysis.structure.scene_count == 3
    assert [scene.description for scene in analysis.structure.scenes] == [
        "opens with a contrarian claim",
        "supports the claim",
        "closes the loop",
    ]
    assert analysis.subject == "The Old Advice"
    assert "the old advice" not in analysis.reproducing_prompt.lower()
    assert "dry tone" in analysis.reproducing_prompt.lower()
    assert analysis.confidence == 0.82
    assert analysis.provider == "acme"
    assert analysis.model == "text-model"


def test_analysis_fills_missing_traits_from_the_measured_text() -> None:
    provider = FakeProvider({"tone": "warm and direct"})

    analysis = _run(analyze_text_reference(TEXT, provider=provider))

    assert analysis.tone == "warm and direct"
    assert analysis.language == "en"
    assert analysis.hook_pattern == "contrarian"
    assert analysis.structure.hook == "Most people get this wrong."
    assert analysis.structure.scene_count == 3
    assert analysis.duration_seconds == pytest.approx(19.2)


def test_request_carries_reference_text_measured_summary_and_schema() -> None:
    provider = FakeProvider(_payload())

    _run(analyze_text_reference(TEXT, provider=provider, model="text-model"))

    request = provider.requests[0]
    assert request.model == "text-model"
    assert request.role == "reference_analysis"
    assert request.response_schema is not None
    assert {"tone", "structure", "hook_pattern", "language"} <= set(
        request.response_schema["properties"]
    )
    assert request.metadata["reference_kind"] == "text"
    assert request.metadata["word_count"] == 48
    assert request.metadata["language"] == "en"
    assert request.metadata["hook_pattern"] == "contrarian"
    assert request.metadata["reading_time_seconds"] == pytest.approx(19.2)
    content = request.messages[0].content
    assert isinstance(content, list)
    text = next(part for part in content if isinstance(part, TextPart)).text
    assert "Measured:" in text
    assert "48 words" in text
    assert "language en" in text
    assert "3 sections" in text
    assert "reading time 19.20s" in text
    assert "hook pattern contrarian" in text
    assert "Most people get this wrong." in text


def test_build_request_can_be_inspected_without_a_provider() -> None:
    sample = sample_text_reference(TEXT)

    request = build_text_reference_request(sample, model="text-model")

    assert isinstance(request, LLMRequest)
    assert request.role == "reference_analysis"
    assert request.model == "text-model"
    assert request.response_schema is not None
    assert request.metadata["word_count"] == 48
    content = request.messages[0].content
    assert isinstance(content, list)
    text = next(part for part in content if isinstance(part, TextPart)).text
    assert "Reference text" in text
    assert "Why Most People Get This Wrong" in text


def test_parse_merges_measured_traits_and_strips_the_subject(
    tmp_path: Path,
) -> None:
    path = tmp_path / "article.md"
    path.write_text(TEXT, encoding="utf-8")

    analysis = parse_text_reference_analysis(
        _payload(),
        source=path,
        reference_id="article-1",
        provider="acme",
        model="text-model",
    )

    assert analysis.reference_id == "article-1"
    assert analysis.kind is ReferenceKind.TEXT
    assert analysis.tone == "dry and urgent"
    assert analysis.hook_pattern == "contrarian"
    assert analysis.language == "en"
    assert analysis.duration_seconds == pytest.approx(19.2)
    assert analysis.structure.hook == "Most people get this wrong."
    assert analysis.structure.beats == ["hook", "evidence", "payoff"]
    assert "the old advice" not in analysis.reproducing_prompt.lower()
    assert analysis.reproducing_prompt


def test_top_level_text_aliases_are_folded_into_the_analysis() -> None:
    analysis = parse_text_reference_analysis(
        {
            "tone": "plain and warm",
            "hook_pattern": "question",
            "lang": "en",
            "hook": "Why does this keep happening?",
            "beats": ["question", "answer"],
            "sections": [
                {"name": "question", "description": "opens the loop"},
                {"name": "answer", "description": "closes it"},
            ],
        }
    )

    assert analysis.tone == "plain and warm"
    assert analysis.hook_pattern == "question"
    assert analysis.language == "en"
    assert analysis.structure.hook == "Why does this keep happening?"
    assert analysis.structure.beats == ["question", "answer"]
    assert analysis.structure.scene_count == 2


def test_analysis_rejects_a_reference_that_is_not_text(tmp_path: Path) -> None:
    path = tmp_path / "look.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n")

    with pytest.raises(ValidationError, match="not text"):
        analyze_text_reference_offline(
            ReferenceSpec(kind=ReferenceKind.IMAGE, path=str(path))
        )


def test_analysis_rejects_a_remote_reference_before_any_download() -> None:
    with pytest.raises(ValidationError, match="downloaded"):
        analyze_text_reference_offline(
            ReferenceSpec(kind=ReferenceKind.TEXT, url="https://example.test/a.txt")
        )


def test_analysis_rejects_bytes_that_are_not_utf8() -> None:
    with pytest.raises(ValidationError, match="UTF-8"):
        analyze_text_reference_offline(b"\xff\xfe\x00\x00")


def test_sync_wrapper_and_reusable_analyzer_share_the_result() -> None:
    provider = FakeProvider(_payload())

    direct = analyze_text_reference_sync(TEXT, provider=provider)
    reusable = _run(TextReferenceAnalyzer(provider).analyze(TEXT))

    assert direct.reproducing_prompt == reusable.reproducing_prompt
    assert reusable.provider == "acme"
    assert reusable.tone == "dry and urgent"
    assert reusable.hook_pattern == "contrarian"


def test_the_default_prompt_names_every_required_section_and_forbids_copying() -> None:
    prompt = DEFAULT_REFERENCE_TEXT_PROMPT.lower()

    for word in (
        "tone",
        "structure",
        "hook_pattern",
        "language",
        "reproducing_prompt",
        "subject",
        "never",
        "copy",
    ):
        assert word in prompt
