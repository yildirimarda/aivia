"""A compact fact list with source URLs satisfies the core fact contract."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from aivia.core.models.topic import Claim, Topic
from aivia.core.models.topic import FactList as CoreFactList
from aivia.research.factbase import (
    DEFAULT_MAX_FACTS,
    Fact,
    FactBase,
    FactList,
    FactListContract,
    build_fact_list,
    build_factbase,
    compact_fact_list,
    compact_facts,
    fact_list,
    to_fact_list,
)

_SOURCE = "https://example.com/story"
_OTHER = "https://other.example.org/background"


class _ResearchClaim:
    """A claim shaped like research output without importing a core model."""

    def __init__(
        self,
        text: str,
        source_urls: list[str],
        *,
        confidence: float = 1.0,
        verified: bool = False,
    ) -> None:
        self.text = text
        self.source_urls = source_urls
        self.confidence = confidence
        self.verified = verified


class _ResearchTopic:
    """A topic shaped like research output without importing a core model."""

    def __init__(self, title: str, claims: list[_ResearchClaim]) -> None:
        self.title = title
        self.claims = claims


def test_research_output_becomes_a_core_contract_fact_list() -> None:
    topic = _ResearchTopic(
        "Bridge history",
        [
            _ResearchClaim(
                "The bridge opened in 1937.",
                [_SOURCE, _OTHER],
                confidence=0.75,
                verified=True,
            )
        ],
    )

    fact_list_value = compact_fact_list(topic)

    assert isinstance(fact_list_value, CoreFactList)
    assert fact_list_value.title == "Bridge history"
    assert len(fact_list_value) == 1
    claim = fact_list_value.claims[0]
    assert claim.text == "The bridge opened in 1937."
    assert claim.source_urls == [_SOURCE, _OTHER]
    assert claim.confidence == 0.75
    assert claim.verified is True
    assert fact_list_value.source_urls == [_SOURCE, _OTHER]


def test_core_topic_claims_are_read_structurally() -> None:
    topic = Topic(
        title="Core topic",
        claims=[
            Claim(
                text="A core claim.",
                source_urls=[_SOURCE],
                confidence=0.5,
                verified=True,
            )
        ],
    )

    fact_list_value = compact_fact_list(topic)

    assert isinstance(fact_list_value, CoreFactList)
    assert fact_list_value.claims[0].text == "A core claim."
    assert fact_list_value.claims[0].source_urls == [_SOURCE]
    assert fact_list_value.claims[0].confidence == 0.5
    assert fact_list_value.claims[0].verified is True


def test_unsupported_or_empty_facts_are_dropped() -> None:
    fact_list_value = compact_fact_list(
        [
            _ResearchClaim("Supported fact.", [_SOURCE]),
            _ResearchClaim("Unsourced fact.", []),
            _ResearchClaim("   ", [_SOURCE]),
        ]
    )

    assert [claim.text for claim in fact_list_value.claims] == ["Supported fact."]


def test_duplicate_facts_merge_sources_and_dedupe_urls() -> None:
    fact_list_value = compact_fact_list(
        [
            _ResearchClaim("  The bridge opened in 1937. ", [_SOURCE, _SOURCE]),
            _ResearchClaim("the bridge OPENED in 1937.", [_OTHER]),
        ]
    )

    assert len(fact_list_value) == 1
    claim = fact_list_value.claims[0]
    assert claim.text == "The bridge opened in 1937."
    assert claim.source_urls == [_SOURCE, _OTHER]
    assert fact_list_value.source_urls == [_SOURCE, _OTHER]


def test_mapping_input_accepts_claim_and_source_aliases() -> None:
    fact_list_value = compact_fact_list(
        {
            "title": "Alias topic",
            "facts": [
                {"statement": "Aliased fact.", "sources": [_SOURCE]},
                {"text": "Second fact.", "source_url": _OTHER},
            ],
        }
    )

    assert fact_list_value.title == "Alias topic"
    assert [claim.text for claim in fact_list_value.claims] == [
        "Aliased fact.",
        "Second fact.",
    ]
    assert fact_list_value.claims[1].source_urls == [_OTHER]


def test_max_facts_limits_the_list_and_is_validated() -> None:
    claims = [_ResearchClaim(f"Fact {index}.", [_SOURCE]) for index in range(3)]

    assert DEFAULT_MAX_FACTS is None
    assert len(compact_fact_list(claims, max_facts=2)) == 2
    assert len(compact_fact_list(claims, max_facts=0)) == 0
    with pytest.raises(ValueError):
        compact_fact_list(claims, max_facts=-1)
    with pytest.raises(ValueError):
        compact_fact_list(claims, max_facts=True)


def test_invalid_source_urls_are_rejected() -> None:
    with pytest.raises(ValueError):
        compact_fact_list({"text": "Bad source.", "source_urls": ["example.com"]})
    with pytest.raises(ValueError):
        compact_fact_list({"text": "Bad source.", "source_urls": [123]})


def test_fact_validates_text_sources_and_confidence() -> None:
    with pytest.raises(ValueError):
        Fact(text="   ", source_urls=[_SOURCE])
    with pytest.raises(ValueError):
        Fact(text="Fact.", source_urls=[])
    with pytest.raises(ValueError):
        Fact(text="Fact.", source_urls=[_SOURCE], confidence=1.5)


def test_fact_and_fact_list_aliases_and_serialization() -> None:
    fact_list_value = compact_fact_list(
        {"text": "A fact.", "source_urls": [_SOURCE]},
        title="A topic",
    )

    assert FactBase is FactList
    assert FactListContract is FactList
    assert build_fact_list is compact_fact_list
    assert build_factbase is compact_fact_list
    assert fact_list is compact_fact_list
    assert compact_facts is compact_fact_list
    assert to_fact_list is compact_fact_list
    payload: dict[str, Any] = fact_list_value.to_dict()
    assert payload["title"] == "A topic"
    assert payload["claims"][0]["text"] == "A fact."
    assert payload["source_urls"] == [_SOURCE]
    json.dumps(payload)


def test_empty_input_produces_an_empty_fact_list() -> None:
    assert len(compact_fact_list()) == 0
    assert len(compact_fact_list(None)) == 0
    assert not compact_fact_list([])


def test_factbase_module_does_not_import_core_code() -> None:
    module_path = Path(__file__).resolve().parents[2] / "src" / "aivia" / "research"
    source = (module_path / "factbase.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)

    assert not [
        name for name in imported if name == "aivia" or name.startswith("aivia.")
    ]
