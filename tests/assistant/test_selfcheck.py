"""Visual self check: verify a preview against the instruction and auto-fix.

The check renders a composition to a still, asks a (fake) vision provider whether
it matches the instruction, applies the provider's patch on failure and re-renders,
bounded by ``max_iterations`` from the planner configuration.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine
from typing import Any

import pytest
from PIL import Image

from aivia.assistant.selfcheck import SelfCheck, SelfCheckError, SelfCheckResult
from aivia.compose.schema import Composition
from aivia.core.config.planner import PlannerConfig
from aivia.providers.protocols.llm import (
    LLMRequest,
    LLMResponse,
    LLMUsage,
)


def _composition() -> dict[str, Any]:
    return {
        "canvas": {"width": 1280, "height": 720, "background": "#ffffff"},
        "duration_seconds": 5.0,
        "layers": [],
        "audio_tracks": [],
        "effects": [],
    }


def _verdict(
    passed: bool,
    reason: str = "",
    patch: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"passed": passed, "reason": reason}
    if patch is not None:
        result["patch"] = patch
    return result


def _response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        model="vision-model",
        provider="test",
        usage=LLMUsage(input_tokens=100, output_tokens=50),
    )


class _SequenceProvider:
    """An async provider returning one canned response per call."""

    def __init__(self, responses: list[str] | str | None = None) -> None:
        if responses is None:
            self.responses = [json.dumps(_verdict(True))]
        elif isinstance(responses, str):
            self.responses = [responses]
        else:
            self.responses = list(responses)
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        index = min(len(self.requests) - 1, len(self.responses) - 1)
        return _response(self.responses[index])


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine)


def _selfcheck(
    *,
    max_iterations: int = 3,
    provider: Any = None,
) -> SelfCheck:
    config = PlannerConfig(max_iterations=max_iterations)
    return SelfCheck(planner_config=config, provider=provider)


def _base_composition() -> Composition:
    return Composition.model_validate(_composition())


def _text_layer_composition() -> Composition:
    data = _composition()
    data["layers"] = [
        {
            "id": "title",
            "type": "text",
            "text": "hello",
            "font_size": 24,
        }
    ]
    return Composition.model_validate(data)


def _patch_replace_font_size() -> list[dict[str, Any]]:
    return [
        {
            "op": "replace",
            "path": "/layers/0/font_size",
            "value": 48,
        }
    ]


def test_selfcheck_passes_on_first_preview() -> None:
    provider = _SequenceProvider(json.dumps(_verdict(True, "matches")))
    selfcheck = _selfcheck(provider=provider)

    result = _run(selfcheck.check("a white title card", _base_composition()))

    assert isinstance(result, SelfCheckResult)
    assert result.passed is True
    assert result.iterations == 0
    assert result.reason == "matches"
    assert isinstance(result.preview, Image.Image)
    assert len(provider.requests) == 1


def test_selfcheck_sends_preview_image_to_provider() -> None:
    provider = _SequenceProvider(json.dumps(_verdict(True)))
    selfcheck = _selfcheck(provider=provider)

    _run(selfcheck.check("show a red band", _base_composition()))

    request = provider.requests[0]
    assert request.has_images is True
    assert len(request.images) == 1
    assert request.images[0].media_type == "image/png"
    assert request.images[0].data is not None
    prompt = request.messages[0].content
    assert isinstance(prompt, list)
    texts = [part.text for part in prompt if part.type == "text"]
    assert "show a red band" in texts[0]


def test_selfcheck_applies_patch_then_passes() -> None:
    fix_patch = _patch_replace_font_size()
    provider = _SequenceProvider(
        [
            json.dumps(_verdict(False, "text too small", fix_patch)),
            json.dumps(_verdict(True, "now readable")),
        ]
    )
    selfcheck = _selfcheck(provider=provider)

    result = _run(selfcheck.check("a large title", _text_layer_composition()))

    assert result.passed is True
    assert result.iterations == 1
    assert result.reason == "now readable"
    assert len(provider.requests) == 2


def test_selfcheck_re_render_after_patch() -> None:
    """A patch that adds a layer changes the rendered preview on the next pass."""
    provider = _SequenceProvider(
        [
            json.dumps(
                _verdict(
                    False,
                    "missing layer",
                    [
                        {
                            "op": "add",
                            "path": "/layers/-",
                            "value": {
                                "id": "title",
                                "type": "text",
                                "text": "hello",
                                "font_size": 40,
                            },
                        }
                    ],
                )
            ),
            json.dumps(_verdict(True, "layer present")),
        ]
    )
    selfcheck = _selfcheck(provider=provider)

    result = _run(selfcheck.check("say hello", _base_composition()))

    assert result.passed is True
    assert result.iterations == 1
    layer_ids = [layer.id for layer in result.composition.layers]
    assert "title" in layer_ids


def test_selfcheck_raises_after_exhausting_iterations() -> None:
    provider = _SequenceProvider(
        [
            json.dumps(_verdict(False, "still wrong")),
            json.dumps(_verdict(False, "still wrong")),
            json.dumps(_verdict(False, "still wrong")),
            json.dumps(_verdict(False, "still wrong")),
        ]
    )
    selfcheck = _selfcheck(max_iterations=3, provider=provider)

    with pytest.raises(SelfCheckError) as raised:
        _run(selfcheck.check("never satisfied", _base_composition()))

    error = raised.value
    assert error.code == "self_check_error"
    assert error.iterations == 3
    assert error.max_iterations == 3
    assert error.reason == "still wrong"
    assert len(provider.requests) == 4


def test_selfcheck_zero_iterations_means_first_failure_is_final() -> None:
    provider = _SequenceProvider(json.dumps(_verdict(False, "nope")))
    selfcheck = _selfcheck(max_iterations=0, provider=provider)

    with pytest.raises(SelfCheckError) as raised:
        _run(selfcheck.check("impossible", _base_composition()))

    error = raised.value
    assert error.iterations == 0
    assert error.max_iterations == 0
    assert error.reason == "nope"
    assert len(provider.requests) == 1


def test_selfcheck_without_provider_raises() -> None:
    selfcheck = _selfcheck(provider=None)

    with pytest.raises(SelfCheckError) as raised:
        _run(selfcheck.check("nothing", _base_composition()))

    error = raised.value
    assert error.code == "self_check_error"
    assert error.iterations == 0


def test_selfcheck_iteration_limit_comes_from_config() -> None:
    provider = _SequenceProvider(json.dumps(_verdict(True)))
    config = PlannerConfig(max_iterations=5)
    selfcheck = SelfCheck(planner_config=config, provider=provider)

    assert selfcheck.max_iterations == 5
    result = _run(selfcheck.check("ok", _base_composition()))
    assert result.passed is True


def test_selfcheck_result_preview_is_deterministic() -> None:
    provider = _SequenceProvider(json.dumps(_verdict(True)))
    selfcheck = _selfcheck(provider=provider)

    first = _run(selfcheck.check("same", _base_composition()))
    second = _run(selfcheck.check("same", _base_composition()))

    assert first.preview.size == second.preview.size
    assert first.preview.tobytes() == second.preview.tobytes()


def test_selfcheck_exposes_planner_config() -> None:
    config = PlannerConfig(max_iterations=2)
    selfcheck = SelfCheck(planner_config=config)

    assert selfcheck.planner_config is config
    assert selfcheck.max_iterations == 2
