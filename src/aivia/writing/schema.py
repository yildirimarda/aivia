"""Mode aware script schema for the writing stage.

A script is ordered beats plus a payload whose shape depends on the writing
mode. The core models already describe a beat, its dialogue and its media
prompts; this module adds a typed payload per mode so a joke always carries a
setup and punchline, a riddle always carries a question and answer, and a lyric
always carries its sections. Custom modes keep any extra fields.

Serialization is canonical and versioned. The same script always renders to the
same JSON text, and reading that text back produces an equal script, so stored
scripts survive a round trip without losing their mode specific data.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    SerializeAsAny,
    field_validator,
    model_validator,
)

from aivia.core.models.base import CURRENT_MODEL_SCHEMA_VERSION
from aivia.core.models.script import (
    Beat,
    BeatModel,
    Dialogue,
    DialogueLine,
    DialogueLineModel,
    ImagePrompt,
    MediaKind,
    MediaPrompt,
    MediaPromptKind,
    MediaPromptModel,
    MediaType,
    Mode,
    MotionPrompt,
    ScriptBeat,
    ScriptMode,
    VisualType,
    WritingMode,
    WritingModeName,
)
from aivia.core.models.script import (
    Script as CoreScript,
)
from aivia.core.models.topic import SourceUrls

__all__ = [
    "CURRENT_SCRIPT_SCHEMA_VERSION",
    "MODE_PAYLOADS",
    "PAYLOAD_MODELS",
    "SCRIPT_SCHEMA_VERSION",
    "Beat",
    "BeatModel",
    "CustomModePayload",
    "CustomPayload",
    "Dialogue",
    "DialogueLine",
    "DialogueLineModel",
    "Fact",
    "FactsModePayload",
    "FactsPayload",
    "ImagePrompt",
    "JokeModePayload",
    "JokePayload",
    "LyricsModePayload",
    "LyricsPayload",
    "LyricsSection",
    "MediaKind",
    "MediaPrompt",
    "MediaPromptKind",
    "MediaPromptModel",
    "MediaType",
    "Mode",
    "ModePayload",
    "MotionPrompt",
    "Payload",
    "PoemModePayload",
    "PoemPayload",
    "QuoteModePayload",
    "QuotePayload",
    "RiddleModePayload",
    "RiddlePayload",
    "Script",
    "ScriptBeat",
    "ScriptConfig",
    "ScriptFact",
    "ScriptMode",
    "ScriptModel",
    "ScriptPayload",
    "ScriptSchema",
    "ScriptSpec",
    "StoryModePayload",
    "StoryPayload",
    "VisualType",
    "WritingMode",
    "WritingModeName",
    "WritingScript",
    "build_payload",
    "coerce_payload",
    "parse_payload",
    "payload_for_mode",
    "payload_model_for_mode",
    "validate_payload",
]

CURRENT_SCRIPT_SCHEMA_VERSION = CURRENT_MODEL_SCHEMA_VERSION
"""The script schema version written by this release."""

SCRIPT_SCHEMA_VERSION = CURRENT_SCRIPT_SCHEMA_VERSION
"""Alias for :data:`CURRENT_SCRIPT_SCHEMA_VERSION`."""


def _strip(value: Any) -> Any:
    return value.strip() if isinstance(value, str) else value


def _non_empty(value: Any, field: str) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{field} must be non-empty")
        return text
    return value


def _text_list(value: Any) -> Any:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return list(value)
    return value


class ModePayload(BaseModel):
    """Base class for the fields that only one writing mode understands.

    Unknown fields are kept so a custom mode can add data without a schema
    change. The payload itself is plain data: it never carries a vendor, model
    or price.
    """

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        validate_assignment=True,
        from_attributes=True,
    )

    def to_dict(self) -> dict[str, Any]:
        """Return the payload as a JSON-compatible mapping."""
        return self.model_dump(mode="json")


class ScriptFact(ModePayload):
    """One fact used by the facts mode, with optional provenance."""

    text: str = Field(
        min_length=1,
        validation_alias=AliasChoices(
            "text",
            "statement",
            "claim",
            "fact",
            "content",
        ),
    )
    source_urls: SourceUrls = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "source_urls",
            "sources",
            "urls",
            "source_url",
            "source",
        ),
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    verified: bool = False

    @field_validator("text", mode="before")
    @classmethod
    def _validate_text(cls, value: Any) -> Any:
        return _non_empty(value, "fact text")

    @property
    def statement(self) -> str:
        """Alias for :attr:`text`."""
        return self.text

    @property
    def sources(self) -> list[str]:
        """Alias for :attr:`source_urls`."""
        return self.source_urls


Fact = ScriptFact
"""Alias for :class:`ScriptFact`."""


class FactsPayload(ModePayload):
    """Payload for the facts mode: a sourced fact list and a chosen angle."""

    facts: list[ScriptFact] = Field(default_factory=list)
    source_urls: SourceUrls = Field(default_factory=list)
    angle: str = ""
    notes: str = ""

    @field_validator("facts", mode="before")
    @classmethod
    def _coerce_facts(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, str):
            return [{"text": value}]
        if isinstance(value, (Mapping, ScriptFact)):
            return [value]
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            return [{"text": item} if isinstance(item, str) else item for item in value]
        return value

    @field_validator("angle", "notes", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _strip(value)


class QuotePayload(ModePayload):
    """Payload for the quote mode: the quote, its attribution and context."""

    quote: str = Field(
        default="",
        validation_alias=AliasChoices(
            "quote",
            "text",
            "quote_text",
            "content",
            "line",
        ),
    )
    attribution: str = Field(
        default="",
        validation_alias=AliasChoices(
            "attribution",
            "author",
            "speaker",
            "source",
            "credit",
        ),
    )
    source_urls: SourceUrls = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "source_urls",
            "sources",
            "urls",
            "source_url",
        ),
    )
    context: str = ""

    @field_validator("quote", "attribution", "context", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _strip(value)


class PoemPayload(ModePayload):
    """Payload for the poem mode: form, tone, lines and optional stanzas."""

    form: str = Field(
        default="",
        validation_alias=AliasChoices("form", "style", "poem_form"),
    )
    tone: str = Field(
        default="",
        validation_alias=AliasChoices("tone", "mood", "voice"),
    )
    lines: list[str] = Field(default_factory=list)
    stanzas: list[list[str]] = Field(default_factory=list)
    rhyme_scheme: str = Field(
        default="",
        validation_alias=AliasChoices("rhyme_scheme", "rhyme", "scheme"),
    )
    syllable_budget: int | None = Field(default=None, ge=1)
    length: str = Field(
        default="",
        validation_alias=AliasChoices("length", "target_length", "size"),
    )
    min_seconds: float | None = Field(default=None, ge=0.0)
    max_seconds: float | None = Field(default=None, ge=0.0)
    reading_time_seconds: float | None = Field(
        default=None,
        gt=0.0,
        validation_alias=AliasChoices(
            "reading_time_seconds",
            "reading_time",
            "duration_seconds",
        ),
    )

    @field_validator("lines", mode="before")
    @classmethod
    def _coerce_lines(cls, value: Any) -> Any:
        return _text_list(value)

    @field_validator("stanzas", mode="before")
    @classmethod
    def _coerce_stanzas(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return [_text_list(stanza) for stanza in value]
        return value

    @field_validator("form", "tone", "rhyme_scheme", "length", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _strip(value)

    @model_validator(mode="after")
    def _ordered_reading_time(self) -> PoemPayload:
        if (
            self.min_seconds is not None
            and self.max_seconds is not None
            and self.max_seconds < self.min_seconds
        ):
            raise ValueError("max_seconds must not be below min_seconds")
        return self

    @property
    def reading_time(self) -> float | None:
        """Alias for :attr:`reading_time_seconds`."""
        return self.reading_time_seconds


class StoryPayload(ModePayload):
    """Payload for the story mode: form, tone, scenes and narrative choices."""

    form: str = Field(
        default="",
        validation_alias=AliasChoices("form", "style", "kind", "story_form"),
    )
    tone: str = Field(
        default="",
        validation_alias=AliasChoices("tone", "mood", "voice"),
    )
    scenes: list[str] = Field(default_factory=list)
    characters: list[str] = Field(default_factory=list)
    setting: str = ""
    point_of_view: str = Field(
        default="",
        validation_alias=AliasChoices("point_of_view", "pov", "perspective"),
    )
    moral: str = ""
    length: str = Field(
        default="",
        validation_alias=AliasChoices("length", "target_length", "size"),
    )
    min_seconds: float | None = Field(default=None, ge=0.0)
    max_seconds: float | None = Field(default=None, ge=0.0)
    reading_time_seconds: float | None = Field(
        default=None,
        gt=0.0,
        validation_alias=AliasChoices(
            "reading_time_seconds",
            "reading_time",
            "duration_seconds",
        ),
    )

    @field_validator("scenes", "characters", mode="before")
    @classmethod
    def _coerce_text_list(cls, value: Any) -> Any:
        return _text_list(value)

    @field_validator(
        "form",
        "tone",
        "setting",
        "point_of_view",
        "moral",
        "length",
        mode="before",
    )
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _strip(value)

    @model_validator(mode="after")
    def _ordered_reading_time(self) -> StoryPayload:
        if (
            self.min_seconds is not None
            and self.max_seconds is not None
            and self.max_seconds < self.min_seconds
        ):
            raise ValueError("max_seconds must not be below min_seconds")
        return self

    @property
    def reading_time(self) -> float | None:
        """Alias for :attr:`reading_time_seconds`."""
        return self.reading_time_seconds


class JokePayload(ModePayload):
    """Payload for the joke mode: a setup and a punchline."""

    setup: str = Field(
        default="",
        validation_alias=AliasChoices("setup", "question", "premise"),
    )
    punchline: str = Field(
        default="",
        validation_alias=AliasChoices("punchline", "payoff", "answer"),
    )
    style: str = Field(
        default="",
        validation_alias=AliasChoices("style", "form", "kind"),
    )
    tags: list[str] = Field(default_factory=list)

    @field_validator("setup", "punchline", "style", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _strip(value)

    @field_validator("tags", mode="before")
    @classmethod
    def _coerce_tags(cls, value: Any) -> Any:
        return _text_list(value)


class RiddlePayload(ModePayload):
    """Payload for the riddle mode: a question, its answer and a hint."""

    question: str = Field(
        default="",
        validation_alias=AliasChoices("question", "setup", "prompt"),
    )
    answer: str = Field(
        default="",
        validation_alias=AliasChoices("answer", "solution", "punchline"),
    )
    hint: str = ""
    difficulty: str = ""

    @field_validator("question", "answer", "hint", "difficulty", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _strip(value)


class LyricsSection(ModePayload):
    """One named lyric section such as a verse, chorus or bridge."""

    name: str = Field(
        default="",
        validation_alias=AliasChoices(
            "name",
            "kind",
            "type",
            "section",
            "role",
        ),
    )
    lines: list[str] = Field(default_factory=list)
    rhyme_scheme: str = Field(
        default="",
        validation_alias=AliasChoices("rhyme_scheme", "rhyme", "scheme"),
    )
    syllable_budget: int | None = Field(default=None, ge=1)

    @field_validator("name", "rhyme_scheme", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _strip(value)

    @field_validator("lines", mode="before")
    @classmethod
    def _coerce_lines(cls, value: Any) -> Any:
        return _text_list(value)

    @property
    def kind(self) -> str:
        """Alias for :attr:`name`."""
        return self.name


class LyricsPayload(ModePayload):
    """Payload for the lyrics mode: sections, rhyme, syllables and keywords."""

    sections: list[LyricsSection] = Field(default_factory=list)
    section_counts: dict[str, int] = Field(default_factory=dict)
    rhyme_scheme: str = Field(
        default="",
        validation_alias=AliasChoices("rhyme_scheme", "rhyme", "scheme"),
    )
    syllable_budget: int | None = Field(default=None, ge=1)
    keywords: list[str] = Field(default_factory=list)
    genre: str = ""
    tempo: str = ""

    @field_validator("sections", mode="before")
    @classmethod
    def _coerce_sections(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, (Mapping, LyricsSection, str)):
            return [value]
        return value

    @field_validator("keywords", mode="before")
    @classmethod
    def _coerce_keywords(cls, value: Any) -> Any:
        return _text_list(value)

    @field_validator("rhyme_scheme", "genre", "tempo", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _strip(value)


class CustomPayload(ModePayload):
    """Payload for a custom mode, keeping the mode name and instructions."""

    name: str = ""
    instructions: str = Field(
        default="",
        validation_alias=AliasChoices(
            "instructions",
            "prompt",
            "guidance",
            "rules",
        ),
    )

    @field_validator("name", "instructions", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _strip(value)


MODE_PAYLOADS: dict[WritingMode, type[ModePayload]] = {
    WritingMode.FACTS: FactsPayload,
    WritingMode.QUOTE: QuotePayload,
    WritingMode.POEM: PoemPayload,
    WritingMode.STORY: StoryPayload,
    WritingMode.JOKE: JokePayload,
    WritingMode.RIDDLE: RiddlePayload,
    WritingMode.LYRICS: LyricsPayload,
    WritingMode.CUSTOM: CustomPayload,
}
"""The payload model for every built in writing mode."""

PAYLOAD_MODELS = MODE_PAYLOADS
"""Alias for :data:`MODE_PAYLOADS`."""

Payload = ModePayload
"""Alias for :class:`ModePayload`."""

ScriptPayload = ModePayload
"""Alias for :class:`ModePayload`."""

FactsModePayload = FactsPayload
QuoteModePayload = QuotePayload
PoemModePayload = PoemPayload
StoryModePayload = StoryPayload
JokeModePayload = JokePayload
RiddleModePayload = RiddlePayload
LyricsModePayload = LyricsPayload
CustomModePayload = CustomPayload


def _normalise_mode(mode: Any) -> Any:
    if isinstance(mode, WritingMode):
        return mode
    if isinstance(mode, str):
        text = mode.strip().lower()
        try:
            return WritingMode(text)
        except ValueError:
            return text
    return mode


def payload_model_for_mode(mode: Any) -> type[ModePayload]:
    """Return the payload model that belongs to *mode*.

    An unknown mode falls back to :class:`CustomPayload` so a custom mode file
    can still carry data; the script's own mode field decides whether the mode
    itself is accepted.
    """
    normalised = _normalise_mode(mode)
    if isinstance(normalised, WritingMode):
        return MODE_PAYLOADS[normalised]
    return CustomPayload


def coerce_payload(mode: Any, value: Any = None) -> ModePayload:
    """Validate *value* as the payload model for *mode*."""
    model = payload_model_for_mode(mode)
    if value is None:
        return model()
    if isinstance(value, model):
        return value
    if isinstance(value, ModePayload):
        return model.model_validate(value.model_dump(mode="python"))
    if isinstance(value, Mapping):
        return model.model_validate(dict(value))
    return model.model_validate(value)


parse_payload = coerce_payload
payload_for_mode = coerce_payload
validate_payload = coerce_payload
build_payload = coerce_payload


_PAYLOAD_KEYS = ("payload", "mode_payload", "mode_data", "data", "options")
_MODE_KEYS = ("mode", "writing_mode", "script_mode")


class Script(CoreScript):
    """A mode aware script whose payload is typed for its writing mode.

    Beats, dialogue lines and media prompts are the core models, so the writing
    stage shares one representation with the rest of the pipeline. The payload
    is validated against the selected mode and kept as its concrete class, which
    makes a round trip through canonical JSON lossless. A custom mode that has
    been registered or loaded is accepted by name as well.
    """

    mode: WritingMode | str = Field(  # type: ignore[assignment]
        default=WritingMode.FACTS,
        validation_alias=AliasChoices(
            "mode",
            "writing_mode",
            "script_mode",
        ),
    )
    payload: SerializeAsAny[ModePayload] = Field(  # type: ignore[assignment]
        default_factory=FactsPayload,
        validation_alias=AliasChoices(
            "payload",
            "mode_payload",
            "mode_data",
            "data",
            "options",
        ),
    )

    @field_validator("mode", mode="before")
    @classmethod
    def _accept_registered_custom_mode(cls, value: Any) -> Any:
        if isinstance(value, WritingMode):
            return value
        if isinstance(value, str):
            text = value.strip()
            try:
                return WritingMode(text.lower())
            except ValueError:
                from aivia.writing.modes import is_known_writing_mode

                if is_known_writing_mode(text):
                    return text
                raise ValueError(f"unknown writing mode: {value!r}") from None
        return value

    @model_validator(mode="before")
    @classmethod
    def _coerce_mode_payload(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        mode = next(
            (values[key] for key in _MODE_KEYS if key in values),
            WritingMode.FACTS,
        )
        payload = next(
            (values[key] for key in _PAYLOAD_KEYS if key in values),
            None,
        )
        for key in _PAYLOAD_KEYS:
            values.pop(key, None)
        values["payload"] = coerce_payload(mode, payload)
        return values

    @property
    def typed_payload(self) -> ModePayload:
        """The payload as its concrete per-mode class."""
        return self.payload

    @property
    def payload_dict(self) -> dict[str, Any]:
        """The payload as a JSON-compatible mapping."""
        return self.payload.to_dict()


ScriptSchema = Script
ScriptSpec = Script
ScriptConfig = Script
WritingScript = Script
ScriptModel = Script
"""Aliases for :class:`Script` for callers that use schema or spec wording."""
