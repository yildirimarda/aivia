"""The facts mode builds a hook, beats and outro from a structural fact list."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

import pytest

import aivia.writing.facts as facts_module
from aivia.core.errors import ConfigError
from aivia.writing import (
    BuiltinWritingMode,
    FactsMode,
    FactsPayload,
    FactsWritingMode,
    FactWritingMode,
    Script,
    WritingMode,
    build_facts_script,
    builtin_writing_mode,
    coerce_facts,
    extract_facts,
    facts_script,
    normalise_facts,
    normalize_facts,
    parse_facts,
    resolve_mode,
)


def _payload(script: Script) -> FactsPayload:
    assert isinstance(script.payload, FactsPayload)
    return script.payload


@dataclass
class ResearchFact:
    """A research-shaped claim without importing the research module."""

    text: str
    source_urls: list[str]
    confidence: float = 0.9
    verified: bool = True


@dataclass
class ResearchFactList:
    """A research-shaped fact list: anything exposing ``claims`` qualifies."""

    claims: list[ResearchFact] = field(default_factory=list)
    title: str = ""
    source_urls: list[str] = field(default_factory=list)


def test_facts_mode_replaces_the_builtin_placeholder() -> None:
    assert resolve_mode("facts") is FactsMode
    assert builtin_writing_mode(WritingMode.FACTS) is FactsMode
    assert issubclass(FactsMode, BuiltinWritingMode)

    mode = FactsMode()

    assert mode.name == "facts"
    assert mode.mode == "facts"
    assert mode.payload_model is FactsPayload
    assert FactsWritingMode is FactsMode
    assert FactWritingMode is FactsMode


def test_facts_mode_does_not_import_research() -> None:
    source = Path(facts_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)

    assert not [name for name in imported if name.startswith("aivia.research")]


def test_user_text_builds_hook_beats_and_outro() -> None:
    script = FactsMode().build_script("First fact.\nSecond fact.\nThird fact.")

    assert isinstance(script, Script)
    assert script.mode is WritingMode.FACTS
    assert script.hook == "First fact."
    assert script.outro == "That is the full picture."
    assert [beat.text for beat in script.beats] == [
        "First fact.",
        "Second fact.",
        "Third fact.",
    ]
    assert [beat.index for beat in script.beats] == [0, 1, 2]
    assert [line.speaker for line in script.dialogue] == ["narrator"] * 3
    assert [fact.text for fact in _payload(script).facts] == [
        "First fact.",
        "Second fact.",
        "Third fact.",
    ]
    assert _payload(script).source_urls == []
    assert Script.model_validate_json(script.to_json()) == script


def test_user_text_splits_sentences_and_bullets() -> None:
    sentences = FactsMode().build_script("One fact. Two facts! Three facts?")
    bullets = FactsMode().build_script("- First fact.\n* Second fact.\n1. Third fact.")

    assert [beat.text for beat in sentences.beats] == [
        "One fact.",
        "Two facts!",
        "Three facts?",
    ]
    assert [beat.text for beat in bullets.beats] == [
        "First fact.",
        "Second fact.",
        "Third fact.",
    ]


def test_explicit_options_override_the_built_defaults() -> None:
    script = FactsMode().build_script(
        "One fact.\nTwo facts.",
        title="A short history",
        language="de",
        angle="warm and personal",
        notes="Keep it friendly.",
        hook="Open with a question.",
        outro="Close with a smile.",
        speaker="guide",
    )

    assert script.title == "A short history"
    assert script.language == "de"
    assert script.hook == "Open with a question."
    assert script.outro == "Close with a smile."
    assert _payload(script).angle == "warm and personal"
    assert _payload(script).notes == "Keep it friendly."
    assert [line.speaker for line in script.dialogue] == ["guide", "guide"]


def test_research_output_is_read_structurally_without_importing_it() -> None:
    fact_list = ResearchFactList(
        claims=[
            ResearchFact("The first claim.", ["https://one.test/a"]),
            ResearchFact("The second claim.", ["https://two.test/b"]),
        ],
        title="A researched topic",
    )

    script = FactsMode().build_script(fact_list)

    assert script.title == "A researched topic"
    assert script.hook == "A researched topic: The first claim."
    assert [beat.text for beat in script.beats] == [
        "The first claim.",
        "The second claim.",
    ]
    assert _payload(script).source_urls == [
        "https://one.test/a",
        "https://two.test/b",
    ]
    assert [fact.verified for fact in _payload(script).facts] == [True, True]
    assert [fact.confidence for fact in _payload(script).facts] == [0.9, 0.9]


def test_mapping_supplies_metadata_facts_and_sources() -> None:
    script = FactsMode().build_script(
        {
            "title": "Mapped",
            "language": "fr",
            "angle": "curious",
            "hook": "A supplied hook.",
            "outro": "A supplied outro.",
            "facts": [
                {"statement": "A sourced fact.", "sources": "https://source.test/a"},
                "A plain fact.",
            ],
        }
    )

    assert script.title == "Mapped"
    assert script.language == "fr"
    assert script.hook == "A supplied hook."
    assert script.outro == "A supplied outro."
    assert _payload(script).angle == "curious"
    assert [beat.text for beat in script.beats] == [
        "A sourced fact.",
        "A plain fact.",
    ]
    assert _payload(script).source_urls == ["https://source.test/a"]


def test_facts_are_deduplicated_and_their_sources_are_merged() -> None:
    script = FactsMode().build_script(
        [
            "A repeated fact.",
            {"text": "  a   REPEATED fact.  ", "sources": ["https://one.test/a"]},
            {"text": "A repeated fact.", "sources": ["https://two.test/b"]},
            {"text": "A different fact.", "sources": ["https://one.test/a"]},
        ]
    )

    assert [beat.text for beat in script.beats] == [
        "A repeated fact.",
        "A different fact.",
    ]
    assert _payload(script).source_urls == [
        "https://one.test/a",
        "https://two.test/b",
    ]


def test_build_payload_keeps_the_base_contract_for_research_output() -> None:
    fact_list = ResearchFactList(
        claims=[ResearchFact("A claim.", ["https://source.test/a"])]
    )
    mode = FactsMode()

    payload = mode.build_payload(fact_list)

    assert isinstance(payload, FactsPayload)
    assert [fact.text for fact in payload.facts] == ["A claim."]
    assert payload.source_urls == ["https://source.test/a"]
    assert mode.build(fact_list) == payload


def test_limits_cap_the_payload_and_the_beats() -> None:
    script = FactsMode().build_script(
        "One.\nTwo.\nThree.",
        max_facts=2,
        max_beats=1,
    )

    assert [fact.text for fact in _payload(script).facts] == ["One.", "Two."]
    assert [beat.text for beat in script.beats] == ["One."]


def test_empty_input_fails_clearly() -> None:
    with pytest.raises(ConfigError, match="at least one fact"):
        FactsMode().build_script("")
    with pytest.raises(ConfigError, match="at least one fact"):
        FactsMode().build_script([])
    with pytest.raises(ConfigError, match="at least one fact"):
        FactsMode().build_script(None)


def test_building_is_deterministic() -> None:
    mode = FactsMode()
    fact_list = ResearchFactList(
        claims=[
            ResearchFact("One claim.", ["https://one.test/a"]),
            ResearchFact("Two claims.", ["https://two.test/b"]),
        ],
        title="Stable",
    )

    first = mode.build_script(fact_list)
    second = mode.build_script(fact_list)

    def structure(script: Script) -> tuple[object, ...]:
        return (
            script.mode,
            script.title,
            script.language,
            script.hook,
            script.outro,
            _payload(script),
            tuple(
                (
                    beat.index,
                    beat.text,
                    tuple((line.speaker, line.text) for line in beat.dialogue),
                )
                for beat in script.beats
            ),
        )

    assert structure(first) == structure(second)


def test_individual_builders_can_be_used_on_their_own() -> None:
    mode = FactsMode()
    facts = ["One.", "Two."]

    assert mode.build_hook(facts) == "One."
    assert mode.build_hook(facts, title="Topic") == "Topic: One."
    assert [beat.text for beat in mode.build_beats(facts)] == ["One.", "Two."]
    assert mode.build_outro(facts) == "That is the full picture."
    assert mode.build_outro(facts, title="Topic") == "That is the story of Topic."


def test_extract_facts_aliases_accept_every_input_shape() -> None:
    assert coerce_facts is extract_facts
    assert parse_facts is extract_facts
    assert normalize_facts is extract_facts
    assert normalise_facts is extract_facts

    facts = extract_facts(
        {"claims": [{"text": "A claim.", "sources": ["https://source.test/a"]}]}
    )

    assert [fact.text for fact in facts] == ["A claim."]
    assert facts[0].source_urls == ["https://source.test/a"]


def test_module_level_builder_and_alias() -> None:
    assert facts_script is build_facts_script
    assert build_facts_script("A fact.").hook == "A fact."
