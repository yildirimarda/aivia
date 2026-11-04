"""Every beat gets one image prompt and one motion prompt, styled and glossed."""

from __future__ import annotations

import pytest

from aivia.core.models.script import (
    Beat,
    DialogueLine,
    ImagePrompt,
    MediaKind,
    MediaPrompt,
    MotionPrompt,
    Script,
)
from aivia.writing import (
    DEFAULT_IMAGE_GUIDANCE,
    DEFAULT_MOTION_GUIDANCE,
    Glossary,
    GlossaryTerm,
    MediaPromptBuilder,
    MediaPromptPair,
    apply_glossary,
    build_beat_prompts,
    build_image_prompt,
    build_media_prompts,
    build_motion_prompt,
    generate_media_prompts,
    glossary_terms,
    load_style_bible,
    normalise_glossary,
)

STYLE = {
    "name": "house",
    "description": "A calm documentary look.",
    "prompt_suffix": "Muted palette and soft window light.",
    "entries": [
        {
            "key": "visuals",
            "text": "Prefer natural light.",
            "prompt_suffix": "Natural textures, simple composition.",
        },
        {
            "key": "avoid",
            "text": "Avoid visual clutter.",
            "prompt_suffix": "No neon gradients or on-screen text.",
        },
    ],
}


def _script() -> Script:
    return Script(
        title="Bees",
        language="en",
        beats=[
            Beat(
                index=0,
                text="A bee lands on a flower.",
                duration_seconds=3.5,
                dialogue=[DialogueLine(speaker="narrator", text="A bee lands.")],
            ),
            Beat(index=1, text="It carries pollen back to the hive."),
        ],
    )


def test_every_beat_gets_one_image_and_one_motion_prompt() -> None:
    script = build_media_prompts(
        _script(),
        style_bible=STYLE,
    )

    assert len(script.beats) == 2
    for beat in script.beats:
        assert [prompt.kind for prompt in beat.media_prompts] == [
            MediaKind.IMAGE,
            MediaKind.MOTION,
        ]
        assert isinstance(beat.image_prompt, ImagePrompt)
        assert isinstance(beat.motion_prompt, MotionPrompt)
        assert beat.image_prompt is not None
        assert beat.motion_prompt is not None


def test_prompts_are_provider_agnostic_and_deterministic() -> None:
    script = _script()
    first = build_media_prompts(script, style_bible=STYLE)
    second = build_media_prompts(script, style_bible=STYLE)

    assert first == second
    for beat in first.beats:
        for prompt in beat.media_prompts:
            text = prompt.prompt.casefold()
            for vendor in ("openai", "replicate", "fal", "gemini", "anthropic"):
                assert vendor not in text
            assert "provider" not in text
            assert "model id" not in text


def test_the_style_bible_suffix_reaches_both_prompt_kinds() -> None:
    bible = load_style_bible(STYLE)
    script = build_media_prompts(_script(), style_bible=bible)

    for beat in script.beats:
        assert beat.image_prompt is not None
        assert beat.motion_prompt is not None
        for prompt in (beat.image_prompt, beat.motion_prompt):
            assert bible.suffix in prompt.prompt
            assert prompt.style == "house"
        assert "No neon gradients" in (beat.image_prompt.negative_prompt or "")


def test_the_prompt_carries_the_beat_text_and_its_dialogue() -> None:
    beat = _script().beats[0]
    pair = build_beat_prompts(beat, style_bible=STYLE)

    assert "A bee lands on a flower." in pair.image.prompt
    assert "A bee lands." in pair.image.prompt
    assert DEFAULT_IMAGE_GUIDANCE in pair.image.prompt
    assert DEFAULT_MOTION_GUIDANCE in pair.motion.prompt
    assert pair.motion.duration_seconds == 3.5


def test_the_motion_prompt_is_a_motion_and_the_image_prompt_is_a_still() -> None:
    image = build_image_prompt(_script().beats[0], style_bible=STYLE)
    motion = build_motion_prompt(_script().beats[0], style_bible=STYLE)

    assert isinstance(image, ImagePrompt)
    assert image.kind is MediaKind.IMAGE
    assert isinstance(motion, MotionPrompt)
    assert motion.kind is MediaKind.MOTION
    assert DEFAULT_IMAGE_GUIDANCE in image.prompt
    assert DEFAULT_MOTION_GUIDANCE in motion.prompt


def test_a_glossary_rewrites_terms_and_adds_a_definition() -> None:
    glossary = {
        "Aivia": {
            "preferred": "AIVIA",
            "definition": "a small blue robot narrator",
            "avoid": ["Avia"],
        },
        "AI": {
            "preferred": "artificial intelligence",
            "aliases": ["A.I."],
        },
    }
    script = Script(
        beats=[
            Beat(text="Avia uses AI to explain the hive."),
        ]
    )

    result = build_media_prompts(
        script,
        style_bible=STYLE,
        glossary=glossary,
    )
    image = result.beats[0].image_prompt
    motion = result.beats[0].motion_prompt

    assert image is not None
    assert motion is not None
    assert "AIVIA" in image.prompt
    assert "Avia" not in image.prompt
    assert "artificial intelligence" in image.prompt
    assert "AIVIA: a small blue robot narrator" in image.prompt
    assert "AIVIA" in motion.prompt


def test_a_glossary_can_be_scoped_per_language() -> None:
    glossary = Glossary.model_validate(
        {
            "terms": [
                {"term": "honey", "preferred": "Honig", "language": "de"},
                {"term": "honey", "preferred": "bal", "language": "tr"},
            ]
        }
    )
    beat = Beat(text="Honey never spoils.")

    german = build_image_prompt(
        beat, style_bible=STYLE, glossary=glossary, language="de"
    )
    turkish = build_image_prompt(
        beat, style_bible=STYLE, glossary=glossary, language="tr"
    )
    english = build_image_prompt(
        beat, style_bible=STYLE, glossary=glossary, language="en"
    )

    assert "Honig" in german.prompt
    assert "bal" in turkish.prompt
    assert "Honey" in english.prompt


def test_glossary_normalisation_accepts_mappings_lists_and_terms() -> None:
    from_mapping = normalise_glossary({"Aivia": "AIVIA"})
    from_list = normalise_glossary(["Aivia", {"term": "AI", "preferred": "AI"}])
    from_term = normalise_glossary(GlossaryTerm(term="Aivia", preferred="AIVIA"))

    assert from_mapping.terms[0].preferred == "AIVIA"
    assert [term.term for term in from_list.terms] == ["Aivia", "AI"]
    assert from_term.terms[0].term == "Aivia"
    assert glossary_terms({"Aivia": "AIVIA"})[0].display == "AIVIA"
    assert (
        apply_glossary("Avia", {"Aivia": {"preferred": "AIVIA", "avoid": ["Avia"]}})
        == "AIVIA"
    )


def test_rebuilding_replaces_only_the_visual_prompts_and_is_idempotent() -> None:
    music = MediaPrompt(kind=MediaKind.MUSIC, prompt="A quiet drone.")
    script = _script()
    script.beats[0].media_prompts.append(music)

    once = build_media_prompts(script, style_bible=STYLE)
    twice = build_media_prompts(once, style_bible=STYLE)

    assert once == twice
    assert [prompt.kind for prompt in once.beats[0].media_prompts] == [
        MediaKind.IMAGE,
        MediaKind.MOTION,
        MediaKind.MUSIC,
    ]
    assert once.beats[0].media_prompts[-1].prompt == "A quiet drone."
    assert script.beats[0].media_prompts[-1].prompt == "A quiet drone."


def test_a_beat_without_text_falls_back_to_the_script_title() -> None:
    script = Script(title="An empty stage", beats=[Beat()])

    result = build_media_prompts(script, style_bible=STYLE)

    assert result.beats[0].image_prompt is not None
    assert "An empty stage" in result.beats[0].image_prompt.prompt


def test_the_builder_returns_a_pair_and_accepts_a_beat_sequence() -> None:
    builder = MediaPromptBuilder(style_bible=STYLE)
    pair = builder.prompts_for(Beat(text="A single moment."))

    assert isinstance(pair, MediaPromptPair)
    assert isinstance(pair, list)
    assert pair.image is pair[0]
    assert pair.motion is pair[1]
    assert pair.prompts == [pair.image, pair.motion]

    pairs = build_media_prompts(
        [Beat(text="First."), Beat(text="Second.")],
        style_bible=STYLE,
    )
    assert len(pairs) == 2
    assert all(isinstance(item, MediaPromptPair) for item in pairs)


def test_generation_aliases_return_the_same_script() -> None:
    script = _script()

    assert generate_media_prompts(script, style_bible=STYLE) == build_media_prompts(
        script, style_bible=STYLE
    )


def test_media_prompts_reject_an_unsupported_value() -> None:
    with pytest.raises(TypeError):
        build_media_prompts("not a script")  # type: ignore[arg-type]
