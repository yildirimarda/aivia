"""The quote mode requires attribution and refuses invented real-person credits."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

import aivia.writing.quote as quote_module
from aivia.core.errors import ConfigError
from aivia.writing import (
    BuiltinWritingMode,
    Quote,
    QuoteMode,
    QuotePayload,
    QuoteWritingMode,
    Script,
    WritingMode,
    build_quote_script,
    builtin_writing_mode,
    coerce_quote,
    extract_quote,
    is_anonymous_attribution,
    is_real_person_attribution,
    looks_like_real_person,
    normalise_quote,
    normalize_quote,
    parse_quote,
    quote_script,
    resolve_mode,
    validate_attribution,
    validate_quote,
)


def _payload(script: Script) -> QuotePayload:
    assert isinstance(script.payload, QuotePayload)
    return script.payload


@dataclass
class QuoteRecord:
    """A structural quote object without importing a provider or module."""

    text: str
    author: str
    sources: list[str]
    context: str = ""


def test_quote_mode_replaces_the_builtin_placeholder() -> None:
    assert resolve_mode("quote") is QuoteMode
    assert builtin_writing_mode(WritingMode.QUOTE) is QuoteMode
    assert issubclass(QuoteMode, BuiltinWritingMode)

    mode = QuoteMode()

    assert mode.name == "quote"
    assert mode.mode == "quote"
    assert mode.payload_model is QuotePayload
    assert QuoteWritingMode is QuoteMode
    assert Quote is QuoteMode


def test_quote_mode_does_not_import_research() -> None:
    source = Path(quote_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)

    assert not [name for name in imported if name.startswith("aivia.research")]


def test_mapping_builds_hook_beat_outro_and_payload() -> None:
    script = QuoteMode().build_script(
        {
            "quote": "  The unexamined life is not worth living.  ",
            "attribution": "  Socrates  ",
            "source_urls": ["https://source.test/socrates"],
            "context": "At his trial.",
        }
    )

    assert isinstance(script, Script)
    assert script.mode is WritingMode.QUOTE
    assert script.title == ""
    assert script.language == "en"
    assert script.hook == "The unexamined life is not worth living."
    assert [beat.text for beat in script.beats] == [
        "The unexamined life is not worth living."
    ]
    assert [beat.index for beat in script.beats] == [0]
    assert [line.speaker for line in script.dialogue] == ["narrator"]
    assert script.outro == "— Socrates"
    payload = _payload(script)
    assert payload.quote == "The unexamined life is not worth living."
    assert payload.attribution == "Socrates"
    assert payload.source_urls == ["https://source.test/socrates"]
    assert payload.context == "At his trial."
    assert Script.model_validate_json(script.to_json()) == script


def test_explicit_options_override_the_built_defaults() -> None:
    script = QuoteMode().build_script(
        {
            "quote": "A quotation.",
            "attribution": "Anonymous",
            "context": "A supplied context.",
        },
        title="A short history",
        language="de",
        context="A different context.",
        hook="Open with a question.",
        outro="Close with a smile.",
        speaker="guide",
    )

    assert script.title == "A short history"
    assert script.language == "de"
    assert script.hook == "Open with a question."
    assert script.outro == "Close with a smile."
    assert _payload(script).context == "A different context."
    assert [line.speaker for line in script.dialogue] == ["guide"]


def test_structural_quote_objects_are_read_without_importing_them() -> None:
    record = QuoteRecord(
        text="A sourced quotation.",
        author="Jane Doe",
        sources=["https://one.test/a"],
        context="From an interview.",
    )

    script = QuoteMode().build_script(record)

    assert script.hook == "A sourced quotation."
    assert script.outro == "— Jane Doe"
    payload = _payload(script)
    assert payload.quote == "A sourced quotation."
    assert payload.attribution == "Jane Doe"
    assert payload.source_urls == ["https://one.test/a"]
    assert payload.context == "From an interview."


def test_user_text_can_carry_an_inline_attribution() -> None:
    em_dash = QuoteMode().build_script(
        "The unexamined life is not worth living. — Socrates",
        source_urls=["https://source.test/socrates"],
    )
    hyphen = QuoteMode().build_script('"A quotation." - Anonymous')
    multiline = QuoteMode().build_script("First line.\nSecond line.\n— Traditional")

    assert em_dash.hook == "The unexamined life is not worth living."
    assert _payload(em_dash).attribution == "Socrates"
    assert hyphen.hook == "A quotation."
    assert _payload(hyphen).attribution == "Anonymous"
    assert multiline.hook == "First line.\nSecond line."
    assert _payload(multiline).attribution == "Traditional"


def test_source_alias_url_is_recognised_as_a_source_not_an_author() -> None:
    script = QuoteMode().build_script(
        {
            "quote": "A quotation.",
            "author": "Jane Doe",
            "source": "https://source.test/quote",
        }
    )

    assert _payload(script).attribution == "Jane Doe"
    assert _payload(script).source_urls == ["https://source.test/quote"]


def test_a_quote_is_required() -> None:
    with pytest.raises(ConfigError, match="needs a quote"):
        QuoteMode().build_script("")
    with pytest.raises(ConfigError, match="needs a quote"):
        QuoteMode().build_script({"quote": "   ", "attribution": "Anonymous"})
    with pytest.raises(ConfigError, match="needs a quote"):
        QuoteMode().build_script(None)


def test_an_attribution_is_required() -> None:
    with pytest.raises(ConfigError, match="requires an attribution") as missing:
        QuoteMode().build_script({"quote": "A quotation."})

    assert missing.value.details["reason"] == "attribution_missing"

    with pytest.raises(ConfigError, match="requires an attribution"):
        QuoteMode().build_script({"quote": "A quotation.", "attribution": "   "})


def test_an_invented_real_person_attribution_is_refused_without_a_source() -> None:
    with pytest.raises(ConfigError, match="invented attribution") as refused:
        QuoteMode().build_script(
            {"quote": "A quotation.", "attribution": "Albert Einstein"}
        )

    assert refused.value.details["reason"] == "attribution_unverified"
    assert refused.value.details["attribution"] == "Albert Einstein"

    with pytest.raises(ConfigError, match="invented attribution"):
        QuoteMode().build_script(
            {
                "quote": "A quotation.",
                "attribution": "Jane Doe, 1950",
            }
        )


def test_a_real_person_attribution_is_accepted_with_a_source() -> None:
    script = QuoteMode().build_script(
        {
            "quote": "A quotation.",
            "attribution": "Albert Einstein",
            "source_urls": ["https://source.test/einstein"],
        }
    )

    assert _payload(script).attribution == "Albert Einstein"
    assert script.outro == "— Albert Einstein"


@pytest.mark.parametrize(
    "attribution",
    [
        "Anonymous",
        "anon",
        "Unknown",
        "Unknown author",
        "Author unknown",
        "Traditional",
        "Traditional Irish proverb",
        "Proverb",
        "An unattributed saying",
        "No attribution",
    ],
)
def test_anonymous_and_traditional_attributions_need_no_source(
    attribution: str,
) -> None:
    script = QuoteMode().build_script(
        {"quote": "A quotation.", "attribution": attribution}
    )

    assert _payload(script).attribution == attribution
    assert is_anonymous_attribution(attribution)


@pytest.mark.parametrize(
    "attribution",
    ["The New York Times", "A university study", "Wikipedia", "History"],
)
def test_organisations_and_generic_sources_need_no_source(
    attribution: str,
) -> None:
    script = QuoteMode().build_script(
        {"quote": "A quotation.", "attribution": attribution}
    )

    assert _payload(script).attribution == attribution
    assert not is_anonymous_attribution(attribution)
    assert not looks_like_real_person(attribution)


def test_person_detection_is_exposed_and_narrow() -> None:
    assert looks_like_real_person("Albert Einstein")
    assert is_real_person_attribution is looks_like_real_person
    assert looks_like_real_person("Dr. Jane Doe")
    assert not looks_like_real_person("Anonymous")
    assert not looks_like_real_person("The New York Times")
    assert not looks_like_real_person("https://source.test/quote")


def test_validate_attribution_returns_the_credit_or_raises() -> None:
    assert validate_attribution("Anonymous") == "Anonymous"
    assert validate_attribution("Jane Doe", ["https://source.test/a"]) == "Jane Doe"

    with pytest.raises(ConfigError, match="requires an attribution"):
        validate_attribution("   ")

    with pytest.raises(ConfigError, match="invented attribution"):
        validate_attribution("Jane Doe")


def test_validate_quote_returns_a_payload_and_checks_it() -> None:
    payload = validate_quote({"quote": "A quotation.", "attribution": "Anonymous"})

    assert isinstance(payload, QuotePayload)
    assert payload.quote == "A quotation."

    with pytest.raises(ConfigError, match="requires an attribution"):
        validate_quote({"quote": "A quotation."})


def test_build_payload_keeps_the_typed_contract() -> None:
    mode = QuoteMode()

    payload = mode.build_payload(
        {
            "text": "A quotation.",
            "author": "Anonymous",
            "sources": ["https://source.test/a"],
        }
    )

    assert isinstance(payload, QuotePayload)
    assert payload.quote == "A quotation."
    assert payload.attribution == "Anonymous"
    assert payload.source_urls == ["https://source.test/a"]
    built = mode.build({"quote": "A quotation.", "attribution": "Anonymous"})
    assert isinstance(built, QuotePayload)
    assert built.quote == payload.quote
    assert built.attribution == payload.attribution
    assert mode.build_payload({"quote": "A quotation."}).attribution == ""


def test_extract_quote_aliases_accept_every_input_shape() -> None:
    assert coerce_quote is extract_quote
    assert parse_quote is extract_quote
    assert normalize_quote is extract_quote
    assert normalise_quote is extract_quote

    assert extract_quote("A quotation.").quote == "A quotation."
    assert extract_quote(("A quotation.", "Anonymous")).attribution == "Anonymous"
    assert (
        extract_quote(
            {"payload": {"quote": "A quotation.", "attribution": "Anonymous"}}
        ).quote
        == "A quotation."
    )
    assert (
        extract_quote(QuotePayload(quote="A quotation.", attribution="Anonymous")).quote
        == "A quotation."
    )


def test_individual_builders_can_be_used_on_their_own() -> None:
    mode = QuoteMode()
    value = {"quote": "A quotation.", "attribution": "Anonymous"}

    assert mode.build_hook(value) == "A quotation."
    assert mode.build_hook(value, title="Topic") == "Topic: A quotation."
    assert [beat.text for beat in mode.build_beats(value)] == ["A quotation."]
    assert mode.build_beats(value, max_beats=0) == []
    assert mode.build_outro(value) == "— Anonymous"


def test_building_is_deterministic() -> None:
    mode = QuoteMode()
    value = {
        "quote": "A quotation.",
        "attribution": "Anonymous",
        "source_urls": ["https://source.test/a"],
        "context": "Stable.",
    }

    first = mode.build_script(value)
    second = mode.build_script(value)

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


def test_module_level_builder_and_alias() -> None:
    assert quote_script is build_quote_script
    assert (
        build_quote_script({"quote": "A quotation.", "attribution": "Anonymous"}).hook
        == "A quotation."
    )
