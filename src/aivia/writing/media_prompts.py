"""Provider agnostic image and motion prompts for every script beat.

The writing stage produces beats with spoken text and dialogue, but the visual
stage needs one still prompt and one motion prompt per beat. This module builds
both deterministically from the beat, so no model or vendor is involved: the
same script, style bible and glossary always produce the same prompts.

A style bible contributes its visual prompt suffixes, while a glossary rewrites
brand terms to their preferred spelling or translation and can add a short
definition for a term that appears in the beat. Existing non visual prompts on
a beat, such as music or sound effects, are preserved when the media prompts
are rebuilt, which makes the operation safe to run more than once.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any, overload

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from aivia.core.models.script import (
    Beat,
    ImagePrompt,
    MediaKind,
    MediaPrompt,
    MotionPrompt,
    Script,
)
from aivia.writing.style_bible import StyleBible, load_style_bible

__all__ = [
    "DEFAULT_IMAGE_GUIDANCE",
    "DEFAULT_IMAGE_NEGATIVE_PROMPT",
    "DEFAULT_MOTION_GUIDANCE",
    "DEFAULT_MOTION_NEGATIVE_PROMPT",
    "BeatMediaPrompts",
    "BeatPrompts",
    "BrandGlossary",
    "Glossary",
    "GlossaryEntry",
    "GlossaryTerm",
    "GlossaryTerms",
    "ImageMotionPrompts",
    "MediaPromptBuilder",
    "MediaPromptFactory",
    "MediaPromptGenerator",
    "MediaPromptPair",
    "MediaPromptSet",
    "MediaPrompts",
    "PromptBuilder",
    "PromptGenerator",
    "PromptPair",
    "add_media_prompts",
    "apply_glossary",
    "apply_media_prompts",
    "attach_media_prompts",
    "beat_prompts",
    "build_beat_prompts",
    "build_image_prompt",
    "build_media_prompts",
    "build_motion_prompt",
    "build_prompts",
    "generate_beat_prompts",
    "generate_image_prompt",
    "generate_media_prompts",
    "generate_motion_prompt",
    "generate_prompts",
    "glossary_terms",
    "image_prompt_for",
    "media_prompt_pairs",
    "media_prompts",
    "media_prompts_for",
    "motion_prompt_for",
    "normalise_glossary",
    "normalize_glossary",
    "prompts_for_script",
]

DEFAULT_IMAGE_GUIDANCE = (
    "A single clear still frame with natural composition and believable detail."
)
DEFAULT_MOTION_GUIDANCE = (
    "Subtle natural movement with a slow, smooth camera push and no cuts."
)
DEFAULT_IMAGE_NEGATIVE_PROMPT = (
    "text overlays, logos, watermarks, visual clutter, distorted anatomy"
)
DEFAULT_MOTION_NEGATIVE_PROMPT = (
    "text overlays, logos, watermarks, jump cuts, flicker, distorted anatomy"
)
"""Default provider agnostic guidance for the two prompt kinds."""

_WHITESPACE = re.compile(r"\s+", re.UNICODE)
_AVOID_KEYS = frozenset(
    {"avoid", "negative", "negative-prompt", "exclude", "forbidden", "no"}
)
_TERM_KEYS = ("term", "key", "name", "source", "from")
_PREFERRED_KEYS = (
    "preferred",
    "replacement",
    "replace",
    "use",
    "instead",
    "target",
    "translation",
    "translated",
    "canonical",
    "render",
)
_DEFINITION_KEYS = ("definition", "description", "meaning", "gloss", "note", "notes")
_AVOID_TERM_KEYS = (
    "avoid",
    "forbidden",
    "banned",
    "incorrect",
    "wrong",
    "variants",
)
_ALIAS_KEYS = ("aliases", "alias", "synonyms", "also")
_LANGUAGE_KEYS = ("language", "lang", "locale")
_GLOSSARY_LIST_KEYS = (
    "terms",
    "entries",
    "glossary",
    "items",
    "definitions",
    "lexicon",
)


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return _WHITESPACE.sub(" ", str(value)).strip()


def _sentence(value: str) -> str:
    text = _clean(value)
    if text and text[-1] not in ".!?;:":
        return f"{text}."
    return text


def _first(data: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _normalise_language(value: Any) -> str:
    return str(value).strip().casefold().replace("_", "-")


def _same_language(left: str, right: str) -> bool:
    if not left or not right:
        return True
    first = _normalise_language(left)
    second = _normalise_language(right)
    return first == second or first.split("-", 1)[0] == second.split("-", 1)[0]


def _text_sequence(value: Any, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        parts = text.splitlines() if "\n" in text else text.split(",")
        return tuple(part.strip() for part in parts if part.strip())
    if isinstance(value, (bytes, bytearray, Mapping)):
        raise ValueError(f"{field} must be a sequence of strings")
    if isinstance(value, Iterable):
        return tuple(str(item).strip() for item in value if str(item).strip())
    raise ValueError(f"{field} must be a sequence of strings")


def _normalise_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value).casefold()).strip("-")


class GlossaryTerm(BaseModel):
    """One glossary entry with a preferred form and optional definition.

    ``term`` is the spelling the writer used. ``preferred`` is the spelling,
    translation or brand form that prompts should use instead. ``avoid`` and
    ``aliases`` list additional forms that map to the same preferred form.
    A term with no preferred form is still useful when it carries a definition.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    term: str = Field(validation_alias=AliasChoices(*_TERM_KEYS))
    preferred: str = Field(default="", validation_alias=AliasChoices(*_PREFERRED_KEYS))
    definition: str = Field(
        default="",
        validation_alias=AliasChoices(*_DEFINITION_KEYS),
    )
    avoid: tuple[str, ...] = Field(
        default=(),
        validation_alias=AliasChoices(*_AVOID_TERM_KEYS),
    )
    aliases: tuple[str, ...] = Field(
        default=(),
        validation_alias=AliasChoices(*_ALIAS_KEYS),
    )
    language: str = Field(
        default="",
        validation_alias=AliasChoices(*_LANGUAGE_KEYS),
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_string(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"term": value}
        return value

    @field_validator("term", mode="before")
    @classmethod
    def _validate_term(cls, value: Any) -> str:
        text = _clean(value)
        if not text:
            raise ValueError("glossary term must not be empty")
        return text

    @field_validator("preferred", "definition", "language", mode="before")
    @classmethod
    def _clean_text(cls, value: Any) -> str:
        return _clean(value)

    @field_validator("avoid", "aliases", mode="before")
    @classmethod
    def _normalise_sequence(cls, value: Any) -> tuple[str, ...]:
        return _text_sequence(value, "glossary variants")

    @property
    def key(self) -> str:
        """Alias for :attr:`term`."""
        return self.term

    @property
    def replacement(self) -> str:
        """Alias for :attr:`preferred`."""
        return self.preferred

    @property
    def target(self) -> str:
        """Alias for :attr:`preferred`."""
        return self.preferred

    @property
    def display(self) -> str:
        """The form a prompt should show."""
        return self.preferred or self.term

    @property
    def variants(self) -> tuple[str, ...]:
        """Every source form that should be rewritten to :attr:`display`."""
        if not self.preferred:
            return ()
        values = [self.term, *self.aliases, *self.avoid]
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            text = _clean(value)
            if not text or text.casefold() == self.preferred.casefold():
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            result.append(text)
        result.sort(key=len, reverse=True)
        return tuple(result)

    def matches(self, value: Any) -> bool:
        """Whether *value* names this term by term, preferred or alias."""
        text = _clean(value).casefold()
        if not text:
            return False
        candidates = (self.term, self.preferred, *self.aliases, *self.avoid)
        return text in {_clean(candidate).casefold() for candidate in candidates}

    def applies_to(self, language: str) -> bool:
        """Whether this term applies to *language*."""
        return _same_language(self.language, language)

    def apply(self, text: str) -> str:
        """Rewrite every source form in *text* to the preferred form."""
        result = text
        for variant in self.variants:
            pattern = re.compile(
                rf"(?<!\w){re.escape(variant)}(?!\w)",
                re.IGNORECASE,
            )
            result = pattern.sub(self.preferred, result)
        return result

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of this term."""
        return {
            "term": self.term,
            "preferred": self.preferred,
            "definition": self.definition,
            "avoid": list(self.avoid),
            "aliases": list(self.aliases),
            "language": self.language,
        }


GlossaryEntry = GlossaryTerm


class Glossary(BaseModel):
    """A named set of brand terms, optionally split by language."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    name: str = ""
    language: str = ""
    terms: list[GlossaryTerm] = Field(
        default_factory=list,
        validation_alias=AliasChoices(*_GLOSSARY_LIST_KEYS),
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_value(cls, value: Any) -> Any:
        if isinstance(value, Glossary):
            return value.model_dump()
        if value is None:
            return {"terms": []}
        if isinstance(value, (str, bytes, bytearray)):
            text = value.decode() if isinstance(value, (bytes, bytearray)) else value
            return {
                "terms": [
                    {"term": line.strip()} for line in text.splitlines() if line.strip()
                ]
            }
        if isinstance(value, GlossaryTerm):
            return {"terms": [value]}
        if isinstance(value, Mapping):
            data = dict(value)
            if any(key in data for key in _GLOSSARY_LIST_KEYS):
                return data
            if any(key in data for key in _TERM_KEYS):
                return {"terms": [data]}
            language_map = _language_mapping(data)
            if language_map is not None:
                terms: list[Any] = []
                for language, entries in language_map.items():
                    terms.extend(_terms_from_mapping(entries, language=language))
                return {"terms": terms}
            return {"terms": _terms_from_mapping(data)}
        if isinstance(value, Iterable):
            return {"terms": list(value)}
        raise ValueError("a glossary must be a mapping, a list or text")

    @field_validator("name", "language", mode="before")
    @classmethod
    def _clean_text(cls, value: Any) -> str:
        return _clean(value)

    @classmethod
    def from_value(
        cls,
        value: Any = None,
        *,
        language: str | None = None,
    ) -> Glossary:
        """Build a glossary from text, mappings, terms or an existing glossary."""
        glossary = value if isinstance(value, Glossary) else cls.model_validate(value)
        if language is not None:
            glossary = glossary.for_language(language)
        return glossary

    from_mapping = from_value

    def for_language(self, language: str | None) -> Glossary:
        """Return a glossary restricted to terms that apply to *language*."""
        if not language:
            return self
        return self.model_copy(
            update={"terms": [term for term in self.terms if term.applies_to(language)]}
        )

    def apply(self, text: str, *, language: str | None = None) -> str:
        """Rewrite every matching term in *text* for *language*."""
        result = text
        for term in self.for_language(language).terms:
            result = term.apply(result)
        return result

    def matched(
        self,
        text: str,
        *,
        language: str | None = None,
    ) -> tuple[GlossaryTerm, ...]:
        """Return the terms that appear in *text*, in glossary order."""
        matched: list[GlossaryTerm] = []
        for term in self.for_language(language).terms:
            candidates = (term.term, term.preferred, *term.aliases, *term.avoid)
            if any(
                candidate
                and re.search(
                    rf"(?<!\w){re.escape(candidate)}(?!\w)",
                    text,
                    re.IGNORECASE,
                )
                for candidate in candidates
            ):
                matched.append(term)
        return tuple(matched)

    def definitions(
        self,
        text: str,
        *,
        language: str | None = None,
    ) -> tuple[str, ...]:
        """Return ``term: definition`` lines for matched terms with definitions."""
        lines: list[str] = []
        for term in self.matched(text, language=language):
            if term.definition:
                lines.append(f"{term.display}: {term.definition}")
        return tuple(dict.fromkeys(lines))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of this glossary."""
        return {
            "name": self.name,
            "language": self.language,
            "terms": [term.to_dict() for term in self.terms],
        }

    def __len__(self) -> int:
        return len(self.terms)

    def __iter__(self) -> Iterator[GlossaryTerm]:  # type: ignore[override]
        return iter(self.terms)


GlossaryTerms = Glossary
BrandGlossary = Glossary


def _looks_like_language_code(value: Any) -> bool:
    text = _clean(value)
    return bool(re.fullmatch(r"[A-Za-z]{2,3}(?:[-_][A-Za-z0-9]{2,8})?", text))


def _language_mapping(data: Mapping[str, Any]) -> dict[str, Any] | None:
    if not data or not all(isinstance(value, Mapping) for value in data.values()):
        return None
    if not all(_looks_like_language_code(key) for key in data):
        return None
    return dict(data)


def _terms_from_mapping(
    data: Mapping[str, Any],
    *,
    language: str = "",
) -> list[Any]:
    terms: list[Any] = []
    for key, value in data.items():
        if isinstance(value, Mapping):
            item = dict(value)
            item.setdefault("term", key)
            if language:
                item.setdefault("language", language)
            terms.append(item)
        elif isinstance(value, (list, tuple)):
            terms.append(
                {
                    "term": str(key),
                    "aliases": list(value),
                    "language": language,
                }
            )
        else:
            terms.append(
                {
                    "term": str(key),
                    "preferred": "" if value is None else str(value),
                    "language": language,
                }
            )
    return terms


def normalise_glossary(
    value: Any = None,
    *,
    language: str | None = None,
) -> Glossary:
    """Return *value* as a :class:`Glossary`, restricted to *language*."""
    if value is None:
        return Glossary()
    return Glossary.from_value(value, language=language)


normalize_glossary = normalise_glossary


def glossary_terms(value: Any = None) -> tuple[GlossaryTerm, ...]:
    """Return every term in *value* as a tuple."""
    return tuple(normalise_glossary(value).terms)


def apply_glossary(
    text: str,
    glossary: Any = None,
    *,
    language: str | None = None,
) -> str:
    """Rewrite glossary terms in *text*."""
    return normalise_glossary(glossary, language=language).apply(text)


class MediaPromptPair(list[MediaPrompt]):
    """A list holding exactly one image prompt and one motion prompt.

    The pair behaves like the list the rest of the pipeline expects, and also
    exposes named ``image`` and ``motion`` accessors for callers that prefer
    them. It is intentionally small: prompt generation itself lives on
    :class:`MediaPromptBuilder`.
    """

    def __init__(self, image: ImagePrompt, motion: MotionPrompt) -> None:
        super().__init__([image, motion])
        self.image = image
        self.motion = motion

    @property
    def image_prompt(self) -> ImagePrompt:
        """Alias for :attr:`image`."""
        return self.image

    @property
    def motion_prompt(self) -> MotionPrompt:
        """Alias for :attr:`motion`."""
        return self.motion

    @property
    def prompts(self) -> list[MediaPrompt]:
        """Return both prompts as a plain list."""
        return [self.image, self.motion]

    def to_dict(self) -> dict[str, Any]:
        """Return both prompts as JSON-compatible mappings."""
        return {
            "image": self.image.model_dump(mode="json"),
            "motion": self.motion.model_dump(mode="json"),
        }


MediaPrompts = MediaPromptPair
ImageMotionPrompts = MediaPromptPair
PromptPair = MediaPromptPair
BeatMediaPrompts = MediaPromptPair
BeatPrompts = MediaPromptPair
MediaPromptSet = MediaPromptPair


def _beat_subject(beat: Beat, *, fallback: str = "") -> str:
    pieces: list[str] = []
    narration = _clean(beat.text)
    if narration:
        pieces.append(narration)
    for line in beat.dialogue:
        text = _clean(line.text)
        if text and text not in pieces:
            pieces.append(text)
    if not pieces:
        visual = _clean(beat.visual_type)
        if visual and visual.casefold() not in {"none", "text"}:
            pieces.append(f"A {visual} scene")
    if not pieces:
        pieces.append(_clean(fallback) or "A quiet establishing shot")
    return " ".join(_sentence(piece) for piece in pieces)


def _avoid_suffixes(bible: StyleBible) -> tuple[str, ...]:
    values: list[str] = []
    for entry in bible.entries:
        names = {_normalise_key(entry.key), _normalise_key(entry.title)}
        names.update(_normalise_key(alias) for alias in entry.aliases)
        if names & _AVOID_KEYS and entry.prompt_suffix:
            values.append(entry.prompt_suffix)
    return tuple(dict.fromkeys(values))


class MediaPromptBuilder:
    """Build provider agnostic image and motion prompts from beats.

    The builder loads a style bible (the packaged default when none is given)
    and applies its prompt suffixes to every prompt. A glossary may be supplied
    as a :class:`Glossary`, a mapping, a list of terms or plain text. Prompts
    never contain a provider name, model id or price.
    """

    def __init__(
        self,
        *,
        style_bible: Any = None,
        glossary: Any = None,
        language: str = "",
        aspect_ratio: str = "",
        image_guidance: str = DEFAULT_IMAGE_GUIDANCE,
        motion_guidance: str = DEFAULT_MOTION_GUIDANCE,
        image_negative_prompt: str = DEFAULT_IMAGE_NEGATIVE_PROMPT,
        motion_negative_prompt: str = DEFAULT_MOTION_NEGATIVE_PROMPT,
        include_definitions: bool = True,
        config_dir: Any = None,
        defaults_dir: Any = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.style_bible = load_style_bible(
            style_bible,
            config_dir=config_dir,
            defaults_dir=defaults_dir,
            environ=environ,
        )
        self.glossary = normalise_glossary(glossary, language=language or None)
        self.language = _clean(language)
        self.aspect_ratio = _clean(aspect_ratio)
        self.image_guidance = _clean(image_guidance) or DEFAULT_IMAGE_GUIDANCE
        self.motion_guidance = _clean(motion_guidance) or DEFAULT_MOTION_GUIDANCE
        self.image_negative_prompt = (
            _clean(image_negative_prompt) or DEFAULT_IMAGE_NEGATIVE_PROMPT
        )
        self.motion_negative_prompt = (
            _clean(motion_negative_prompt) or DEFAULT_MOTION_NEGATIVE_PROMPT
        )
        self.include_definitions = include_definitions

    @property
    def style(self) -> str:
        """The joined prompt suffix of the active style bible."""
        return self.style_bible.suffix

    @property
    def negative_prompt(self) -> str:
        """The shared style avoid suffix, if the bible defines one."""
        return " ".join(_avoid_suffixes(self.style_bible))

    def _finish(
        self,
        subject: str,
        guidance: str,
        *,
        language: str | None,
    ) -> str:
        parts = [subject, guidance]
        style = self.style
        if style:
            parts.append(style)
        text = _clean(" ".join(part for part in parts if part))
        text = self.glossary.apply(text, language=language or self.language or None)
        if self.include_definitions:
            definitions = self.glossary.definitions(
                text,
                language=language or self.language or None,
            )
            if definitions:
                text = _clean(f"{text} {' '.join(definitions)}")
        return text

    def _negative(
        self,
        base: str,
        *,
        language: str | None,
    ) -> str:
        parts = [base, self.negative_prompt]
        text = _clean(" ".join(part for part in parts if part))
        return self.glossary.apply(
            text,
            language=language or self.language or None,
        )

    def image_prompt_for(
        self,
        beat: Beat,
        *,
        title: str = "",
        aspect_ratio: str | None = None,
    ) -> ImagePrompt:
        """Return the still prompt for *beat*."""
        subject = _beat_subject(beat, fallback=title)
        prompt = self._finish(
            subject,
            self.image_guidance,
            language=self.language or None,
        )
        return ImagePrompt(
            prompt=prompt,
            negative_prompt=self._negative(
                self.image_negative_prompt,
                language=self.language or None,
            ),
            style=self.style_bible.name,
            aspect_ratio=(
                _clean(aspect_ratio) if aspect_ratio is not None else self.aspect_ratio
            ),
            created_at=beat.created_at,
            updated_at=beat.updated_at,
        )

    def motion_prompt_for(
        self,
        beat: Beat,
        *,
        title: str = "",
        aspect_ratio: str | None = None,
    ) -> MotionPrompt:
        """Return the motion prompt for *beat*."""
        subject = _beat_subject(beat, fallback=title)
        prompt = self._finish(
            subject,
            self.motion_guidance,
            language=self.language or None,
        )
        return MotionPrompt(
            prompt=prompt,
            negative_prompt=self._negative(
                self.motion_negative_prompt,
                language=self.language or None,
            ),
            style=self.style_bible.name,
            aspect_ratio=(
                _clean(aspect_ratio) if aspect_ratio is not None else self.aspect_ratio
            ),
            duration_seconds=beat.duration_seconds,
            created_at=beat.created_at,
            updated_at=beat.updated_at,
        )

    def prompts_for(
        self,
        beat: Beat,
        *,
        title: str = "",
        aspect_ratio: str | None = None,
    ) -> MediaPromptPair:
        """Return the image and motion prompts for one *beat*."""
        return MediaPromptPair(
            self.image_prompt_for(
                beat,
                title=title,
                aspect_ratio=aspect_ratio,
            ),
            self.motion_prompt_for(
                beat,
                title=title,
                aspect_ratio=aspect_ratio,
            ),
        )

    def beat_prompts(
        self,
        beat: Beat,
        *,
        title: str = "",
        aspect_ratio: str | None = None,
    ) -> MediaPromptPair:
        """Alias for :meth:`prompts_for`."""
        return self.prompts_for(beat, title=title, aspect_ratio=aspect_ratio)

    def build(self, script: Script) -> Script:
        """Return a copy of *script* with one image and motion prompt per beat.

        Existing image and motion prompts are replaced, so calling twice is
        idempotent. Prompts of other kinds, such as music or sound effects,
        keep their order after the visual pair.
        """
        beats: list[Beat] = []
        for beat in script.beats:
            pair = self.prompts_for(beat, title=script.title)
            others = [
                prompt
                for prompt in beat.media_prompts
                if prompt.kind not in {MediaKind.IMAGE, MediaKind.MOTION}
            ]
            beats.append(
                beat.model_copy(
                    update={"media_prompts": [*pair.prompts, *others]},
                    deep=True,
                )
            )
        return script.model_copy(update={"beats": beats}, deep=True)

    apply = build
    generate = build
    __call__ = build
    image_prompt = image_prompt_for
    motion_prompt = motion_prompt_for
    build_image_prompt = image_prompt_for
    build_motion_prompt = motion_prompt_for

    def prompts(self, script: Script) -> list[MediaPromptPair]:
        """Return one pair per beat without modifying the script."""
        return [self.prompts_for(beat, title=script.title) for beat in script.beats]


MediaPromptGenerator = MediaPromptBuilder
MediaPromptFactory = MediaPromptBuilder
PromptBuilder = MediaPromptBuilder
PromptGenerator = MediaPromptBuilder


def _builder(
    *,
    style_bible: Any = None,
    glossary: Any = None,
    language: str = "",
    aspect_ratio: str = "",
    image_guidance: str = DEFAULT_IMAGE_GUIDANCE,
    motion_guidance: str = DEFAULT_MOTION_GUIDANCE,
    image_negative_prompt: str = DEFAULT_IMAGE_NEGATIVE_PROMPT,
    motion_negative_prompt: str = DEFAULT_MOTION_NEGATIVE_PROMPT,
    include_definitions: bool = True,
    config_dir: Any = None,
    defaults_dir: Any = None,
    environ: Mapping[str, str] | None = None,
) -> MediaPromptBuilder:
    return MediaPromptBuilder(
        style_bible=style_bible,
        glossary=glossary,
        language=language,
        aspect_ratio=aspect_ratio,
        image_guidance=image_guidance,
        motion_guidance=motion_guidance,
        image_negative_prompt=image_negative_prompt,
        motion_negative_prompt=motion_negative_prompt,
        include_definitions=include_definitions,
        config_dir=config_dir,
        defaults_dir=defaults_dir,
        environ=environ,
    )


@overload
def build_media_prompts(value: Script, **kwargs: Any) -> Script: ...


@overload
def build_media_prompts(value: Beat, **kwargs: Any) -> MediaPromptPair: ...


@overload
def build_media_prompts(
    value: Sequence[Beat],
    **kwargs: Any,
) -> list[MediaPromptPair]: ...


def build_media_prompts(
    value: Script | Beat | Sequence[Beat],
    **kwargs: Any,
) -> Script | MediaPromptPair | list[MediaPromptPair]:
    """Attach an image and motion prompt to every beat.

    A :class:`Script` returns a new script, one :class:`Beat` returns its
    prompt pair and a sequence of beats returns one pair per beat. The style
    bible defaults to the packaged default and the glossary defaults to empty.
    """
    builder = _builder(**kwargs)
    if isinstance(value, Script):
        return builder.build(value)
    if isinstance(value, Beat):
        return builder.prompts_for(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [builder.prompts_for(beat) for beat in value]
    raise TypeError("media prompts need a script, a beat or a sequence of beats")


generate_media_prompts = build_media_prompts
media_prompts = build_media_prompts
media_prompts_for = build_media_prompts
apply_media_prompts = build_media_prompts
attach_media_prompts = build_media_prompts
add_media_prompts = build_media_prompts
build_prompts = build_media_prompts
generate_prompts = build_media_prompts
prompts_for_script = build_media_prompts


def build_beat_prompts(beat: Beat, **kwargs: Any) -> MediaPromptPair:
    """Return the image and motion prompts for one beat."""
    builder = _builder(**kwargs)
    return builder.prompts_for(beat)


generate_beat_prompts = build_beat_prompts
beat_prompts = build_beat_prompts


def media_prompt_pairs(script: Script, **kwargs: Any) -> list[MediaPromptPair]:
    """Return one prompt pair per beat without changing the script."""
    return _builder(**kwargs).prompts(script)


def build_image_prompt(beat: Beat, **kwargs: Any) -> ImagePrompt:
    """Return the still prompt for one beat."""
    builder = _builder(**kwargs)
    return builder.image_prompt_for(beat)


image_prompt_for = build_image_prompt
generate_image_prompt = build_image_prompt


def build_motion_prompt(beat: Beat, **kwargs: Any) -> MotionPrompt:
    """Return the motion prompt for one beat."""
    builder = _builder(**kwargs)
    return builder.motion_prompt_for(beat)


motion_prompt_for = build_motion_prompt
generate_motion_prompt = build_motion_prompt
