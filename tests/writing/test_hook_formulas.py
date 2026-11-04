"""The hook formula library ships as data and candidates use distinct formulas."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine
from typing import Any

import pytest
from pydantic import ValidationError as PydanticValidationError

from aivia.core import paths
from aivia.core.errors import ConfigError
from aivia.providers.protocols.llm import LLMRequest, LLMResponse, LLMUsage
from aivia.writing import (
    DEFAULT_HOOK_FORMULAS,
    DEFAULT_HOOK_FORMULAS_FILENAME,
    HOOK_FORMULAS_FILENAME,
    HookCandidate,
    HookCandidates,
    HookFormula,
    HookFormulaError,
    HookFormulaLibrary,
    HookSelection,
    build_hook_prompt,
    distinct_formulas,
    generate_hooks,
    load_hook_formulas,
    require_distinct_formulas,
)


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine)


class FakeHookProvider:
    """An async LLM provider that returns one canned hook payload."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(
            content=json.dumps(self.payload),
            provider="acme",
            model="hook-model",
            usage=LLMUsage(input_tokens=20, output_tokens=10),
        )


def _formula(
    formula_id: str,
    *,
    shape: str = "Ask a question the script answers.",
    failure_mode: str = "A question the facts cannot answer.",
    patterns: tuple[str, ...] = (r"\?\s*$",),
) -> dict[str, Any]:
    return {
        "id": formula_id,
        "shape": shape,
        "failure_mode": failure_mode,
        "match_patterns": list(patterns),
    }


def test_the_packaged_library_has_shape_failure_mode_and_match_patterns() -> None:
    library = load_hook_formulas()

    assert library.source == paths.package_defaults_dir() / HOOK_FORMULAS_FILENAME
    assert DEFAULT_HOOK_FORMULAS_FILENAME == HOOK_FORMULAS_FILENAME
    assert len(library) >= 8
    assert len(set(library.keys)) == len(library)
    assert len(DEFAULT_HOOK_FORMULAS) == len(library)

    for formula in library:
        assert formula.id
        assert formula.name
        assert formula.shape
        assert formula.failure_mode
        assert formula.match_patterns
        assert formula.example
        assert formula.matches(formula.example)
        assert formula.match(formula.example) in formula.match_patterns


def test_the_library_loads_from_data_and_resolves_ids_names_and_aliases() -> None:
    library = load_hook_formulas(
        {
            "schema_version": 1,
            "formulas": [
                {
                    **_formula("question"),
                    "name": "Direct question",
                    "aliases": ["ask"],
                },
                _formula("statistic", patterns=(r"\b\d+ percent\b",)),
            ],
        }
    )

    assert library.ids == ("question", "statistic")
    assert library.names == ("Direct question", "statistic")
    assert library.get("Question") is library.get("ask")
    assert library.get("direct-question") is library.get("question")
    assert library.require("statistic").id == "statistic"
    assert "ask" in library
    assert library.match("Why does this happen?") == library.get("question")
    assert library.match("Only 40 percent noticed.") == library.get("statistic")
    assert library.match("No formula here.") is None
    assert [formula.id for formula in library.matching("Why?")] == ["question"]


def test_a_formula_manifest_with_missing_or_invalid_fields_is_rejected() -> None:
    with pytest.raises(PydanticValidationError, match="match pattern"):
        HookFormula.model_validate(_formula("question", patterns=()))
    with pytest.raises(PydanticValidationError, match="invalid hook formula"):
        HookFormula.model_validate(_formula("question", patterns=("[",)))
    with pytest.raises(PydanticValidationError, match="failure_mode"):
        HookFormula.model_validate(
            {"id": "question", "shape": "Ask a question.", "match_patterns": [r"\?"]}
        )
    with pytest.raises(ConfigError, match="at least one formula"):
        HookFormulaLibrary.from_mapping({"formulas": []})
    with pytest.raises(ConfigError, match="duplicate hook formula id"):
        HookFormulaLibrary.from_mapping(
            {"formulas": [_formula("question"), _formula("question")]}
        )
    with pytest.raises(HookFormulaError, match="unknown hook formula"):
        load_hook_formulas({"formulas": [_formula("question")]}).require("missing")


def test_a_missing_manifest_fails_clearly(tmp_path: Any) -> None:
    missing = tmp_path / "missing.yaml"

    with pytest.raises(ConfigError, match="not found"):
        load_hook_formulas(missing)


def test_candidates_must_come_from_distinct_formulas() -> None:
    duplicate = {
        "candidates": [
            {"text": "Why does this happen?", "formula": "question"},
            {"text": "What changed?", "formula": "Question"},
        ],
        "selected": 0,
        "reason": "The first question is clearer.",
    }
    with pytest.raises(PydanticValidationError, match="distinct formulas"):
        HookCandidates.model_validate(duplicate)

    distinct = {
        "candidates": [
            {"text": "Why does this happen?", "formula": "question"},
            {"text": "Nine in ten miss it.", "formula": "statistic"},
        ],
        "selected": 1,
        "reason": "The number is more concrete.",
    }
    output = HookCandidates.model_validate(distinct)
    assert output.candidate_formulas == ("question", "statistic")
    assert output.formulas == output.candidate_formulas

    with pytest.raises(PydanticValidationError, match="distinct formulas"):
        HookSelection.model_validate(distinct | {"candidates": duplicate["candidates"]})


def test_the_distinct_formula_helper_ignores_blank_formulas_unless_required() -> None:
    candidates = [
        HookCandidate(text="One.", formula="question"),
        HookCandidate(text="Two."),
        HookCandidate(text="Three.", formula="statistic"),
    ]

    assert distinct_formulas(candidates) == ("question", "statistic")
    assert require_distinct_formulas(candidates) == ("question", "statistic")
    with pytest.raises(HookFormulaError, match="must name a formula"):
        distinct_formulas(candidates, require_all=True)


def test_candidates_with_formulas_absent_from_the_library_are_rejected() -> None:
    library = load_hook_formulas(
        {
            "formulas": [
                _formula("question"),
                _formula("statistic", patterns=(r"\b\d+ percent\b",)),
            ]
        }
    )
    candidates = [
        HookCandidate(text="Why does this happen?", formula="question"),
        HookCandidate(text="A made up opening.", formula="Invented Formula"),
        HookCandidate(text="Another made up opening.", formula="mystery"),
    ]

    with pytest.raises(HookFormulaError, match="unknown hook formulas") as refused:
        library.validate_candidates(candidates, require_known=True)

    assert refused.value.details["unknown"] == ["invented-formula", "mystery"]
    assert refused.value.details["known"] == ["question", "statistic"]
    assert [formula.id for formula in library.validate_candidates(candidates)] == [
        "question"
    ]


def test_generation_rejects_an_unknown_formula_and_records_the_id() -> None:
    library = load_hook_formulas(
        {
            "formulas": [
                _formula("question"),
                _formula("statistic", patterns=(r"\b\d+ percent\b",)),
            ]
        }
    )
    provider = FakeHookProvider(
        {
            "candidates": [
                {"text": "Why does this happen?", "formula": "question"},
                {"text": "Nine in ten people miss it.", "formula": "invented"},
            ],
            "selected": 0,
            "reason": "The question opens the loop.",
        }
    )

    with pytest.raises(HookFormulaError, match="unknown hook formula") as refused:
        _run(generate_hooks("Topic", provider=provider, formulas=library, count=2))

    assert refused.value.details["unknown"] == ["invented"]
    assert refused.value.details["known"] == ["question", "statistic"]


def test_generation_accepts_aliases_and_blank_formulas_from_the_library() -> None:
    library = load_hook_formulas(
        {
            "formulas": [
                {**_formula("question"), "aliases": ["ask"]},
                _formula("statistic", patterns=(r"\b\d+ percent\b",)),
            ]
        }
    )
    provider = FakeHookProvider(
        {
            "candidates": [
                {"text": "Why does this happen?", "formula": "ask"},
                {"text": "A hook without a formula."},
            ],
            "selected": 0,
            "reason": "The question opens the loop.",
        }
    )

    selection = _run(
        generate_hooks("Topic", provider=provider, formulas=library, count=2)
    )

    assert selection.candidate_formulas == ("ask",)


def test_two_aliases_of_one_formula_are_not_distinct() -> None:
    library = load_hook_formulas(
        {
            "formulas": [
                {**_formula("question"), "aliases": ["ask"]},
                _formula("statistic", patterns=(r"\b\d+ percent\b",)),
            ]
        }
    )
    candidates = [
        HookCandidate(text="Why does this happen?", formula="question"),
        HookCandidate(text="What changed?", formula="Ask"),
    ]

    with pytest.raises(HookFormulaError, match="distinct formulas") as refused:
        library.validate_candidates(candidates, require_known=True)

    assert "question" in str(refused.value)
    assert [formula.id for formula in library.validate_candidates(candidates[1:])] == [
        "question"
    ]


def test_generation_rejects_two_aliases_of_the_same_formula() -> None:
    library = load_hook_formulas(
        {
            "formulas": [
                {**_formula("question"), "aliases": ["ask"]},
                _formula("statistic", patterns=(r"\b\d+ percent\b",)),
            ]
        }
    )
    provider = FakeHookProvider(
        {
            "candidates": [
                {"text": "Why does this happen?", "formula": "question"},
                {"text": "What changed?", "formula": "ask"},
            ],
            "selected": 0,
            "reason": "The question opens the loop.",
        }
    )

    with pytest.raises(HookFormulaError, match="distinct formulas"):
        _run(generate_hooks("Topic", provider=provider, formulas=library, count=2))


def test_the_prompt_carries_the_formula_library_and_distinct_requirement() -> None:
    library = load_hook_formulas(
        {
            "formulas": [
                _formula("question"),
                _formula("statistic", patterns=(r"\b\d+ percent\b",)),
            ]
        }
    )

    prompt = build_hook_prompt("Roman engineering", count=2, formulas=library)

    assert '"formula_requirement"' in prompt
    assert "Every candidate must use a different formula" in prompt
    for formula in library:
        assert formula.id in prompt
        assert formula.shape in prompt
        assert formula.failure_mode in prompt


def test_generation_records_the_distinct_formulas_used() -> None:
    provider = FakeHookProvider(
        {
            "candidates": [
                {"text": "Why does this keep happening?", "formula": "question"},
                {"text": "Nine in ten people miss it.", "formula": "statistic"},
            ],
            "selected": 0,
            "reason": "The question opens the loop.",
        }
    )

    selection = _run(generate_hooks("Topic", provider=provider, count=2))

    assert selection.hook == "Why does this keep happening?"
    assert selection.metadata["candidate_formulas"] == ["question", "statistic"]
    assert "formula_requirement" in provider.requests[0].last_user_text
