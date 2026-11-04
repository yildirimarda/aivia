"""The lyrics mode keeps sections, counts, rhyme, syllables and keywords."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import aivia.writing.lyrics as lyrics_module
from aivia.core.errors import ConfigError
from aivia.writing import (
    BuiltinWritingMode,
    Lyrics,
    LyricsMode,
    LyricsPayload,
    LyricsSection,
    LyricsWritingMode,
    Script,
    WritingMode,
    build_lyrics_script,
    builtin_writing_mode,
    check_syllable_budget,
    coerce_lyrics,
    count_syllables,
    extract_lyrics,
    insert_keywords,
    lyrics_script,
    normalise_lyrics,
    normalise_section_counts,
    normalize_lyrics,
    normalize_section_counts,
    parse_lyrics,
    resolve_mode,
    validate_lyrics,
)


def _payload(script: Script) -> LyricsPayload:
    assert isinstance(script.payload, LyricsPayload)
    return script.payload


def test_lyrics_mode_replaces_the_builtin_placeholder() -> None:
    assert resolve_mode("lyrics") is LyricsMode
    assert builtin_writing_mode(WritingMode.LYRICS) is LyricsMode
    assert issubclass(LyricsMode, BuiltinWritingMode)

    mode = LyricsMode()

    assert mode.name == "lyrics"
    assert mode.mode == "lyrics"
    assert mode.payload_model is LyricsPayload
    assert LyricsWritingMode is LyricsMode
    assert Lyrics is LyricsMode


def test_lyrics_mode_does_not_import_research() -> None:
    source = Path(lyrics_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)

    assert not [name for name in imported if name.startswith("aivia.research")]


def test_lyrics_mapping_builds_hook_beats_payload_and_round_trips() -> None:
    script = LyricsMode().build_script(
        {
            "title": "City Lights",
            "language": "de",
            "sections": [
                {
                    "kind": "verse",
                    "lines": ["Neon on the river", "A taxi in the rain"],
                    "rhyme_scheme": "aa",
                    "syllable_budget": 8,
                },
                {"name": "chorus", "lines": ["City lights, carry me home"]},
            ],
            "section_counts": {"verse": 1, "chorus": 1},
            "rhyme_scheme": "abab",
            "syllable_budget": 10,
            "genre": "pop",
            "tempo": "fast",
        }
    )

    assert isinstance(script, Script)
    assert script.mode is WritingMode.LYRICS
    assert script.title == "City Lights"
    assert script.language == "de"
    assert script.hook == "City Lights: Neon on the river"
    assert [beat.text for beat in script.beats] == [
        "Neon on the river",
        "A taxi in the rain",
        "City lights, carry me home",
    ]
    assert [beat.index for beat in script.beats] == [0, 1, 2]
    assert [line.speaker for line in script.dialogue] == ["narrator"] * 3
    assert script.outro == ""

    payload = _payload(script)
    assert [section.name for section in payload.sections] == ["verse", "chorus"]
    assert payload.sections[0].rhyme_scheme == "aa"
    assert payload.sections[0].syllable_budget == 8
    assert payload.section_counts == {"verse": 1, "chorus": 1}
    assert payload.rhyme_scheme == "abab"
    assert payload.syllable_budget == 10
    assert payload.genre == "pop"
    assert payload.tempo == "fast"
    assert Script.model_validate_json(script.to_json()) == script


def test_lyrics_text_reads_bracket_and_colon_section_headers() -> None:
    script = LyricsMode().build_script(
        "[Verse 1]\nFirst line\nSecond line\n\n"
        "Chorus: Sing it back\n\n"
        "[Bridge]\nA quieter line"
    )

    payload = _payload(script)
    assert [section.name for section in payload.sections] == [
        "verse 1",
        "chorus",
        "bridge",
    ]
    assert [section.lines for section in payload.sections] == [
        ["First line", "Second line"],
        ["Sing it back"],
        ["A quieter line"],
    ]
    assert payload.section_counts == {"verse 1": 1, "chorus": 1, "bridge": 1}
    assert [beat.text for beat in script.beats] == [
        "First line",
        "Second line",
        "Sing it back",
        "A quieter line",
    ]
    assert script.hook == "First line"


def test_lyrics_text_without_headers_separates_sections_on_blank_lines() -> None:
    payload = extract_lyrics("First block\nSecond line\n\nThird block")

    assert [section.name for section in payload.sections] == ["verse", "verse"]
    assert [section.lines for section in payload.sections] == [
        ["First block", "Second line"],
        ["Third block"],
    ]
    assert payload.section_counts == {"verse": 2}


def test_section_counts_repeat_and_trim_the_requested_sections() -> None:
    value = {
        "sections": [
            {"name": "verse", "lines": ["A verse line"]},
            {"name": "chorus", "lines": ["A chorus line"]},
        ],
        "section_counts": {"verse": 2, "chorus": 1},
    }

    script = LyricsMode().build_script(value)
    payload = _payload(script)

    assert [section.name for section in payload.sections] == [
        "verse",
        "verse",
        "chorus",
    ]
    assert [section.lines for section in payload.sections] == [
        ["A verse line"],
        ["A verse line"],
        ["A chorus line"],
    ]
    assert payload.section_counts == {"verse": 2, "chorus": 1}
    assert [beat.text for beat in script.beats] == [
        "A verse line",
        "A verse line",
        "A chorus line",
    ]

    trimmed = LyricsMode().build_script(
        {
            "sections": [
                {"name": "verse", "lines": ["One"]},
                {"name": "verse", "lines": ["Two"]},
            ],
            "section_counts": {"verse": 1},
        }
    )
    assert [section.lines for section in _payload(trimmed).sections] == [["One"]]


def test_section_counts_accept_a_string_or_pair_list() -> None:
    value = {"sections": [{"name": "verse", "lines": ["One"]}]}

    from_string = LyricsMode().build_script(
        value,
        section_counts="verse: 2, chorus: 1",
    )
    from_pairs = LyricsMode().build_script(
        value,
        section_counts=[("verse", 2), ("chorus", 1)],
    )

    for script in (from_string, from_pairs):
        payload = _payload(script)
        assert payload.section_counts == {"verse": 2, "chorus": 1}
        assert [section.name for section in payload.sections] == [
            "verse",
            "verse",
            "chorus",
        ]


def test_lyrics_explicit_options_override_the_built_defaults() -> None:
    script = LyricsMode().build_script(
        {
            "title": "Old title",
            "language": "en",
            "sections": [{"name": "verse", "lines": ["Old line"]}],
            "rhyme_scheme": "aabb",
            "syllable_budget": 12,
            "keywords": ["old"],
            "genre": "rock",
            "tempo": "slow",
            "hook": "Old hook",
            "outro": "Old outro",
        },
        title="New title",
        language="fr",
        sections=[{"name": "chorus", "lines": ["New line"]}],
        rhyme_scheme="abab",
        syllable_budget=6,
        keywords=["new"],
        genre="pop",
        tempo="fast",
        insert_keywords=False,
        hook="New hook",
        outro="New outro",
        speaker="lead",
    )

    payload = _payload(script)
    assert script.title == "New title"
    assert script.language == "fr"
    assert script.hook == "New hook"
    assert script.outro == "New outro"
    assert [section.name for section in payload.sections] == ["chorus"]
    assert [section.lines for section in payload.sections] == [["New line"]]
    assert payload.rhyme_scheme == "abab"
    assert payload.syllable_budget == 6
    assert payload.keywords == ["new"]
    assert payload.genre == "pop"
    assert payload.tempo == "fast"
    assert [line.speaker for line in script.dialogue] == ["lead"]


def test_keywords_replace_placeholders_and_fill_the_chorus() -> None:
    script = LyricsMode().build_script(
        {
            "sections": [
                {"name": "verse", "lines": ["We walk through the {keyword}"]},
                {"name": "chorus", "lines": ["Sing it now"]},
            ],
            "keywords": ["summer", "harbour"],
        }
    )

    payload = _payload(script)
    lines = [line for section in payload.sections for line in section.lines]

    assert "We walk through the summer, harbour" in lines
    assert payload.keywords == ["summer", "harbour"]
    assert all(keyword in "\n".join(lines) for keyword in payload.keywords)


def test_keyword_insertion_can_be_disabled() -> None:
    script = LyricsMode().build_script(
        {"sections": [{"name": "chorus", "lines": ["A line"]}]},
        keywords=["summer"],
        insert_keywords=False,
    )

    payload = _payload(script)
    assert [section.lines for section in payload.sections] == [["A line"]]
    assert payload.keywords == ["summer"]


def test_insert_keywords_helper_accepts_lines_and_payloads() -> None:
    lines = insert_keywords(["A line"], ["summer", "harbour"])
    assert lines == ["A line", "summer, harbour"]

    payload = insert_keywords(
        {
            "sections": [
                {"name": "verse", "lines": ["A {keyword} line"]},
                {"name": "chorus", "lines": ["Hook"]},
            ]
        },
        ["summer", "harbour"],
    )
    assert isinstance(payload, LyricsPayload)
    assert payload.sections[0].lines == ["A summer, harbour line"]
    assert payload.sections[1].lines == ["Hook"]


def test_lyrics_requires_at_least_one_line() -> None:
    values: tuple[object, ...] = ("", [], None, {"sections": []})
    for value in values:
        with pytest.raises(ConfigError, match="at least one line"):
            LyricsMode().build_script(value)


def test_validate_lyrics_checks_lines_and_returns_a_payload() -> None:
    payload = validate_lyrics({"sections": [{"name": "verse", "lines": ["One"]}]})

    assert isinstance(payload, LyricsPayload)
    assert [section.lines for section in payload.sections] == [["One"]]

    with pytest.raises(ConfigError, match="at least one line"):
        validate_lyrics(None)


def test_syllable_helpers_count_and_enforce_a_budget() -> None:
    assert count_syllables("hello world") == 3
    assert count_syllables("a") == 1
    assert count_syllables("") == 0

    payload = {
        "sections": [
            {
                "name": "verse",
                "lines": ["one two three"],
                "syllable_budget": 3,
            }
        ]
    }
    assert check_syllable_budget(payload) == extract_lyrics(payload)

    with pytest.raises(ConfigError, match="over the budget") as refused:
        check_syllable_budget(
            {
                "sections": [
                    {
                        "name": "verse",
                        "lines": ["one two three four five six"],
                        "syllable_budget": 3,
                    }
                ]
            }
        )

    details = refused.value.details
    assert details["mode"] == "lyrics"
    assert details["section"] == "verse"
    assert details["syllable_budget"] == 3
    assert details["syllables"] > 3


def test_extract_aliases_accept_every_input_shape() -> None:
    assert coerce_lyrics is extract_lyrics
    assert parse_lyrics is extract_lyrics
    assert normalize_lyrics is extract_lyrics
    assert normalise_lyrics is extract_lyrics
    assert normalize_section_counts is normalise_section_counts

    assert extract_lyrics("[Verse]\nOne line").sections[0].name == "verse"
    assert extract_lyrics(("One line", "Two line")).sections[0].lines == [
        "One line",
        "Two line",
    ]
    assert extract_lyrics({"lyrics": "One line\n\nTwo line"}).section_counts == {
        "verse": 2
    }
    assert extract_lyrics(
        {"verse": ["One line"], "chorus": ["Hook"]}
    ).section_counts == {"verse": 1, "chorus": 1}
    assert (
        extract_lyrics(
            {
                "payload": {"sections": [{"name": "verse", "lines": ["One line"]}]},
                "genre": "pop",
            }
        ).genre
        == "pop"
    )
    assert (
        extract_lyrics(
            LyricsPayload(
                sections=[LyricsSection(name="verse", lines=["One line"])],
                genre="pop",
            )
        ).genre
        == "pop"
    )
    assert normalise_section_counts("verse: 2, chorus: 1") == {
        "verse": 2,
        "chorus": 1,
    }


def test_individual_builders_can_be_used_on_their_own() -> None:
    mode = LyricsMode()
    value = {
        "sections": [
            {"name": "verse", "lines": ["One line", "Two line"]},
            {"name": "chorus", "lines": ["Hook"]},
        ]
    }

    assert mode.build_hook(value) == "One line"
    assert mode.build_hook(value, title="Topic") == "Topic: One line"
    assert [beat.text for beat in mode.build_beats(value)] == [
        "One line",
        "Two line",
        "Hook",
    ]
    assert mode.build_beats(value, max_beats=1)[0].text == "One line"
    assert mode.build_outro(value) == ""
    assert mode.build_payload(value).section_counts == {"verse": 1, "chorus": 1}


def test_module_level_builder_and_alias() -> None:
    assert lyrics_script is build_lyrics_script

    script = build_lyrics_script("[Verse]\nOne line")

    assert script.mode is WritingMode.LYRICS
    assert script.hook == "One line"
    assert _payload(script).sections[0].lines == ["One line"]


def test_building_is_deterministic() -> None:
    value = {
        "title": "Stable",
        "sections": [
            {"name": "verse", "lines": ["One line", "Two line"]},
            {"name": "chorus", "lines": ["Hook {keyword}"]},
        ],
        "section_counts": {"verse": 2, "chorus": 1},
        "rhyme_scheme": "aabb",
        "syllable_budget": 8,
        "keywords": ["summer"],
        "genre": "pop",
    }

    def structure(script: Script) -> tuple[object, ...]:
        return (
            script.mode,
            script.title,
            script.hook,
            script.outro,
            script.duration_seconds,
            _payload(script),
            tuple(beat.text for beat in script.beats),
        )

    assert structure(LyricsMode().build_script(value)) == structure(
        LyricsMode().build_script(value)
    )
