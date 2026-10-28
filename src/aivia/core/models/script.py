"""Mode aware scripts, beats and provider agnostic media prompts.

A :class:`Script` is the written result of one writing mode. It is made of
ordered :class:`Beat` units; a beat carries the spoken text, optional dialogue
lines per speaker and the media prompts that will later become its visual and
audio segments. Nothing here calls a provider or knows a vendor name: the
models only describe what a writer produced and what the rest of the pipeline
must consume.

Dialogue is deliberately kept per line with a named speaker. The same speaker
always groups together, so a later voice stage can map each speaker to a
distinct voice without parsing the text again. A beat may be narration only,
dialogue only or a mix of both.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from pydantic import AliasChoices, Field, field_validator, model_validator

from aivia.core.models.base import ModelBase

__all__ = [
    "Beat",
    "BeatModel",
    "Dialogue",
    "DialogueLine",
    "DialogueLineModel",
    "ImagePrompt",
    "MediaKind",
    "MediaPrompt",
    "MediaPromptKind",
    "MediaPromptModel",
    "MediaType",
    "Mode",
    "MotionPrompt",
    "Script",
    "ScriptBeat",
    "ScriptMode",
    "ScriptModel",
    "VisualType",
    "WritingMode",
    "WritingModeName",
]


class WritingMode(StrEnum):
    """The built in ways a script can be written."""

    FACTS = "facts"
    QUOTE = "quote"
    POEM = "poem"
    STORY = "story"
    JOKE = "joke"
    RIDDLE = "riddle"
    LYRICS = "lyrics"
    CUSTOM = "custom"


class MediaKind(StrEnum):
    """The kind of media a prompt is meant to produce."""

    IMAGE = "image"
    MOTION = "motion"
    VIDEO = "video"
    STILL = "still"
    MUSIC = "music"
    SFX = "sfx"
    AUDIO = "audio"
    STOCK = "stock"


class VisualType(StrEnum):
    """The visual source a beat may be resolved to."""

    IMAGE = "image"
    VIDEO = "video"
    STOCK = "stock"
    STILL = "still"
    TEXT = "text"
    NONE = "none"


_MEDIA_KIND_ALIASES = {
    "visual": MediaKind.IMAGE,
    "picture": MediaKind.IMAGE,
    "photo": MediaKind.IMAGE,
    "clip": MediaKind.VIDEO,
    "animation": MediaKind.MOTION,
    "sound": MediaKind.SFX,
    "effect": MediaKind.SFX,
    "speech": MediaKind.AUDIO,
    "voice": MediaKind.AUDIO,
}

_MODE_ALIASES = {
    "freeform": WritingMode.CUSTOM,
    "custom_mode": WritingMode.CUSTOM,
}


def _non_empty(value: Any, field: str) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{field} must be non-empty")
        return text
    return value


def _strip(value: Any) -> Any:
    return value.strip() if isinstance(value, str) else value


def _coerce_dialogue(value: Any) -> Any:
    """Accept a line, a list of lines or a mapping of speaker to lines."""
    if value is None:
        return []
    if isinstance(value, DialogueLine):
        return [value]
    if isinstance(value, str):
        return [{"speaker": "narrator", "text": value}]
    if isinstance(value, Mapping):
        if any(key in value for key in ("speaker", "character", "role", "who", "name")):
            return [value]
        lines: list[Any] = []
        for speaker, raw in value.items():
            if raw is None:
                continue
            if isinstance(raw, Mapping):
                item = dict(raw)
                item.setdefault("speaker", speaker)
                lines.append(item)
            elif isinstance(raw, (list, tuple)):
                for entry in raw:
                    if isinstance(entry, Mapping):
                        item = dict(entry)
                        item.setdefault("speaker", speaker)
                        lines.append(item)
                    else:
                        lines.append({"speaker": speaker, "text": entry})
            else:
                lines.append({"speaker": speaker, "text": raw})
        return lines
    return value


def _coerce_media_prompts(value: Any) -> Any:
    """Accept a prompt, a list of prompts or a mapping of kind to prompt."""
    if value is None:
        return []
    if isinstance(value, MediaPrompt):
        return [value]
    if isinstance(value, str):
        return [{"prompt": value}]
    if isinstance(value, Mapping):
        if any(
            key in value for key in ("prompt", "text", "description", "kind", "type")
        ):
            return [value]
        prompts: list[Any] = []
        for kind, raw in value.items():
            if raw is None:
                continue
            if isinstance(raw, Mapping):
                item = dict(raw)
                item.setdefault("kind", kind)
                prompts.append(item)
            elif isinstance(raw, MediaPrompt):
                prompts.append(raw)
            else:
                prompts.append({"kind": kind, "prompt": raw})
        return prompts
    return value


class DialogueLine(ModelBase):
    """One spoken line attributed to a named speaker.

    ``speaker`` is a stable name such as ``narrator`` or a character id. The
    voice stage maps that name to a configured voice, so the line itself never
    carries a vendor voice id.
    """

    speaker: str = Field(
        min_length=1,
        validation_alias=AliasChoices(
            "speaker",
            "character",
            "role",
            "who",
            "name",
        ),
    )
    text: str = Field(
        min_length=1,
        validation_alias=AliasChoices(
            "text",
            "line",
            "dialogue",
            "content",
            "utterance",
        ),
    )
    voice: str = ""
    emotion: str = Field(
        default="",
        validation_alias=AliasChoices("emotion", "mood", "tone"),
    )
    direction: str = Field(
        default="",
        validation_alias=AliasChoices(
            "direction",
            "stage_direction",
            "action",
            "delivery",
        ),
    )
    start_seconds: float | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices(
            "start_seconds",
            "start",
            "start_time",
            "from_seconds",
        ),
    )
    end_seconds: float | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices(
            "end_seconds",
            "end",
            "end_time",
            "to_seconds",
        ),
    )
    duration_seconds: float | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices(
            "duration_seconds",
            "duration",
            "seconds",
            "length_seconds",
        ),
    )

    @field_validator("speaker", mode="before")
    @classmethod
    def _validate_speaker(cls, value: Any) -> Any:
        return _non_empty(value, "speaker")

    @field_validator("text", mode="before")
    @classmethod
    def _validate_text(cls, value: Any) -> Any:
        return _non_empty(value, "text")

    @field_validator("voice", "emotion", "direction", mode="before")
    @classmethod
    def _strip_optional_text(cls, value: Any) -> Any:
        return _strip(value)

    @model_validator(mode="after")
    def _ordered_times(self) -> DialogueLine:
        if (
            self.start_seconds is not None
            and self.end_seconds is not None
            and self.end_seconds < self.start_seconds
        ):
            raise ValueError("end_seconds must not precede start_seconds")
        return self

    @property
    def character(self) -> str:
        """Alias for :attr:`speaker`."""
        return self.speaker

    @property
    def line(self) -> str:
        """Alias for :attr:`text`."""
        return self.text

    @property
    def mood(self) -> str:
        """Alias for :attr:`emotion`."""
        return self.emotion

    @property
    def stage_direction(self) -> str:
        """Alias for :attr:`direction`."""
        return self.direction

    @property
    def start(self) -> float | None:
        """Alias for :attr:`start_seconds`."""
        return self.start_seconds

    @property
    def end(self) -> float | None:
        """Alias for :attr:`end_seconds`."""
        return self.end_seconds

    @property
    def duration(self) -> float | None:
        """The line duration, derived from its bounds when they are present."""
        if (
            self.duration_seconds is None
            and self.start_seconds is not None
            and self.end_seconds is not None
        ):
            return self.end_seconds - self.start_seconds
        return self.duration_seconds


class MediaPrompt(ModelBase):
    """A provider agnostic prompt for one visual or audio segment.

    ``kind`` distinguishes an image from a motion, music or sound effect
    request. Vendor, model and pricing choices belong to the router, never to
    the prompt.
    """

    kind: MediaKind = Field(
        default=MediaKind.IMAGE,
        validation_alias=AliasChoices(
            "kind",
            "type",
            "media_type",
            "target",
        ),
    )
    prompt: str = Field(
        min_length=1,
        validation_alias=AliasChoices(
            "prompt",
            "text",
            "description",
            "instruction",
        ),
    )
    negative_prompt: str = Field(
        default="",
        validation_alias=AliasChoices(
            "negative_prompt",
            "negative",
            "avoid",
        ),
    )
    style: str = ""
    aspect_ratio: str = Field(
        default="",
        validation_alias=AliasChoices(
            "aspect_ratio",
            "aspect",
            "ratio",
        ),
    )
    seed: int | None = None
    duration_seconds: float | None = Field(
        default=None,
        gt=0,
        validation_alias=AliasChoices(
            "duration_seconds",
            "duration",
            "seconds",
            "length_seconds",
        ),
    )

    @field_validator("kind", mode="before")
    @classmethod
    def _validate_kind(cls, value: Any) -> Any:
        if isinstance(value, MediaKind):
            return value
        if isinstance(value, str):
            text = value.strip().lower()
            return _MEDIA_KIND_ALIASES.get(text, text)
        return value

    @field_validator("prompt", mode="before")
    @classmethod
    def _validate_prompt(cls, value: Any) -> Any:
        return _non_empty(value, "prompt")

    @field_validator(
        "negative_prompt",
        "style",
        "aspect_ratio",
        mode="before",
    )
    @classmethod
    def _strip_optional_text(cls, value: Any) -> Any:
        return _strip(value)

    @property
    def text(self) -> str:
        """Alias for :attr:`prompt`."""
        return self.prompt

    @property
    def negative(self) -> str:
        """Alias for :attr:`negative_prompt`."""
        return self.negative_prompt

    @property
    def media_type(self) -> MediaKind:
        """Alias for :attr:`kind`."""
        return self.kind

    @property
    def target(self) -> MediaKind:
        """Alias for :attr:`kind`."""
        return self.kind

    @property
    def type(self) -> MediaKind:
        """Alias for :attr:`kind`."""
        return self.kind

    @property
    def aspect(self) -> str:
        """Alias for :attr:`aspect_ratio`."""
        return self.aspect_ratio

    @property
    def duration(self) -> float | None:
        """Alias for :attr:`duration_seconds`."""
        return self.duration_seconds


class ImagePrompt(MediaPrompt):
    """A media prompt that defaults to an image request."""

    kind: MediaKind = Field(
        default=MediaKind.IMAGE,
        validation_alias=AliasChoices(
            "kind",
            "type",
            "media_type",
            "target",
        ),
    )


class MotionPrompt(MediaPrompt):
    """A media prompt that defaults to a motion request."""

    kind: MediaKind = Field(
        default=MediaKind.MOTION,
        validation_alias=AliasChoices(
            "kind",
            "type",
            "media_type",
            "target",
        ),
    )


class Beat(ModelBase):
    """One script unit mapped to one visual and one audio segment.

    ``text`` is the spoken narration. ``dialogue`` holds the lines attributed
    to individual speakers and ``media_prompts`` holds the visual or audio
    prompts generated for this beat. ``index`` is assigned from the enclosing
    script when it is omitted, so the beats keep their order without making
    callers repeat it.
    """

    index: int | None = Field(
        default=0,
        ge=0,
        validation_alias=AliasChoices(
            "index",
            "order",
            "number",
            "position",
            "sequence",
        ),
    )
    text: str = Field(
        default="",
        validation_alias=AliasChoices(
            "text",
            "narration",
            "voiceover",
            "voice_over",
            "line",
            "content",
        ),
    )
    dialogue: list[DialogueLine] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "dialogue",
            "dialogue_lines",
            "lines",
        ),
    )
    media_prompts: list[MediaPrompt] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "media_prompts",
            "prompts",
            "media_prompt",
            "visual_prompts",
        ),
    )
    visual_type: str = Field(
        default="",
        validation_alias=AliasChoices(
            "visual_type",
            "visual",
            "shot_type",
            "visual_kind",
        ),
    )
    transition: str = ""
    duration_seconds: float | None = Field(
        default=None,
        gt=0,
        validation_alias=AliasChoices(
            "duration_seconds",
            "duration",
            "seconds",
            "length_seconds",
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_media_shorthand(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        shorthand = (
            ("image_prompt", MediaKind.IMAGE),
            ("motion_prompt", MediaKind.MOTION),
            ("video_prompt", MediaKind.VIDEO),
        )
        extra: list[Any] = []
        for key, kind in shorthand:
            if key not in values:
                continue
            raw = values.pop(key)
            if raw is None:
                continue
            if isinstance(raw, MediaPrompt):
                extra.append(raw)
            elif isinstance(raw, Mapping):
                item = dict(raw)
                item.setdefault("kind", kind)
                extra.append(item)
            else:
                extra.append({"kind": kind, "prompt": raw})
        if extra:
            existing = values.get("media_prompts")
            if existing is None:
                existing = values.get("prompts")
            if existing is None:
                existing = values.get("visual_prompts")
            if existing is None:
                existing = values.get("media_prompt", [])
            values["media_prompts"] = [
                *_coerce_media_prompts(existing),
                *extra,
            ]
            for key in ("prompts", "visual_prompts", "media_prompt"):
                values.pop(key, None)
        return values

    @field_validator("dialogue", mode="before")
    @classmethod
    def _validate_dialogue(cls, value: Any) -> Any:
        return _coerce_dialogue(value)

    @field_validator("media_prompts", mode="before")
    @classmethod
    def _validate_media_prompts(cls, value: Any) -> Any:
        return _coerce_media_prompts(value)

    @field_validator("text", "visual_type", "transition", mode="before")
    @classmethod
    def _strip_optional_text(cls, value: Any) -> Any:
        return _strip(value)

    @property
    def narration(self) -> str:
        """Alias for :attr:`text`."""
        return self.text

    @property
    def voiceover(self) -> str:
        """Alias for :attr:`text`."""
        return self.text

    @property
    def line(self) -> str:
        """Alias for :attr:`text`."""
        return self.text

    @property
    def dialogue_lines(self) -> list[DialogueLine]:
        """Alias for :attr:`dialogue`."""
        return self.dialogue

    @property
    def lines(self) -> list[DialogueLine]:
        """Alias for :attr:`dialogue`."""
        return self.dialogue

    @property
    def order(self) -> int | None:
        """Alias for :attr:`index`."""
        return self.index

    @property
    def prompts(self) -> list[MediaPrompt]:
        """Alias for :attr:`media_prompts`."""
        return self.media_prompts

    @property
    def prompt(self) -> str:
        """The first media prompt text attached to this beat, if any."""
        return self.media_prompts[0].prompt if self.media_prompts else ""

    @property
    def visual(self) -> str:
        """Alias for :attr:`visual_type`."""
        return self.visual_type

    @property
    def duration(self) -> float | None:
        """Alias for :attr:`duration_seconds`."""
        return self.duration_seconds

    @property
    def speakers(self) -> list[str]:
        """The distinct speakers in this beat, in first-seen order."""
        speakers: list[str] = []
        for line in self.dialogue:
            if line.speaker not in speakers:
                speakers.append(line.speaker)
        return speakers

    @property
    def dialogue_by_speaker(self) -> dict[str, list[DialogueLine]]:
        """Group this beat's dialogue lines by speaker, preserving order."""
        grouped: dict[str, list[DialogueLine]] = {}
        for line in self.dialogue:
            grouped.setdefault(line.speaker, []).append(line)
        return grouped

    @property
    def lines_by_speaker(self) -> dict[str, list[DialogueLine]]:
        """Alias for :attr:`dialogue_by_speaker`."""
        return self.dialogue_by_speaker

    def _prompt_of_kind(self, *kinds: MediaKind) -> MediaPrompt | None:
        for prompt in self.media_prompts:
            if prompt.kind in kinds:
                return prompt
        return None

    @property
    def image_prompt(self) -> MediaPrompt | None:
        """The first image prompt attached to this beat, if any."""
        return self._prompt_of_kind(MediaKind.IMAGE, MediaKind.STILL)

    @property
    def motion_prompt(self) -> MediaPrompt | None:
        """The first motion or video prompt attached to this beat, if any."""
        return self._prompt_of_kind(MediaKind.MOTION, MediaKind.VIDEO)

    @property
    def visual_prompt(self) -> MediaPrompt | None:
        """Alias for :attr:`image_prompt`."""
        return self.image_prompt

    @property
    def audio_prompt(self) -> MediaPrompt | None:
        """The first music, sound effect or audio prompt on this beat, if any."""
        return self._prompt_of_kind(
            MediaKind.MUSIC,
            MediaKind.SFX,
            MediaKind.AUDIO,
        )


class Script(ModelBase):
    """A mode aware script made of ordered beats and a mode specific payload.

    ``payload`` carries the fields that only make sense for the selected
    writing mode, for example a joke's setup and punchline or a lyric's
    sections. Keeping it open lets a custom mode add data without changing the
    record schema. ``dialogue`` may hold script level lines; lines inside
    beats are collected by :attr:`all_dialogue`.
    """

    mode: WritingMode = Field(
        default=WritingMode.FACTS,
        validation_alias=AliasChoices(
            "mode",
            "writing_mode",
            "script_mode",
        ),
    )
    title: str = Field(
        default="",
        validation_alias=AliasChoices(
            "title",
            "name",
            "subject",
        ),
    )
    language: str = Field(
        default="en",
        validation_alias=AliasChoices(
            "language",
            "lang",
            "locale",
            "language_code",
        ),
    )
    beats: list[Beat] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "beats",
            "segments",
            "scenes",
            "units",
        ),
    )
    dialogue_lines: list[DialogueLine] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "dialogue_lines",
            "dialogue",
            "lines",
        ),
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices(
            "payload",
            "mode_payload",
            "mode_data",
            "data",
            "options",
        ),
    )
    hook: str = ""
    outro: str = ""
    duration_seconds: float | None = Field(
        default=None,
        gt=0,
        validation_alias=AliasChoices(
            "duration_seconds",
            "duration",
            "seconds",
            "length_seconds",
        ),
    )

    @field_validator("mode", mode="before")
    @classmethod
    def _validate_mode(cls, value: Any) -> Any:
        if isinstance(value, WritingMode):
            return value
        if isinstance(value, str):
            text = value.strip().lower()
            return _MODE_ALIASES.get(text, text)
        return value

    @field_validator("title", "language", "hook", "outro", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _strip(value)

    @field_validator("dialogue_lines", mode="before")
    @classmethod
    def _validate_dialogue(cls, value: Any) -> Any:
        return _coerce_dialogue(value)

    @model_validator(mode="after")
    def _assign_beat_indexes(self) -> Script:
        for position, beat in enumerate(self.beats):
            if beat.index is None or "index" not in beat.model_fields_set:
                beat.index = position
        return self

    @property
    def writing_mode(self) -> WritingMode:
        """Alias for :attr:`mode`."""
        return self.mode

    @property
    def name(self) -> str:
        """Alias for :attr:`title`."""
        return self.title

    @property
    def lang(self) -> str:
        """Alias for :attr:`language`."""
        return self.language

    @property
    def mode_payload(self) -> dict[str, Any]:
        """Alias for :attr:`payload`."""
        return self.payload

    @property
    def dialogue(self) -> list[DialogueLine]:
        """All script level and beat dialogue lines, in script order."""
        return self.all_dialogue

    @property
    def lines(self) -> list[DialogueLine]:
        """Alias for :attr:`dialogue`."""
        return self.dialogue

    @property
    def duration(self) -> float | None:
        """Alias for :attr:`duration_seconds`."""
        return self.duration_seconds

    @property
    def all_dialogue(self) -> list[DialogueLine]:
        """All script level and beat dialogue lines, in script order."""
        lines = list(self.dialogue_lines)
        for beat in self.beats:
            lines.extend(beat.dialogue)
        return lines

    @property
    def speakers(self) -> list[str]:
        """The distinct speakers across the whole script, in first-seen order."""
        speakers: list[str] = []
        for line in self.all_dialogue:
            if line.speaker not in speakers:
                speakers.append(line.speaker)
        return speakers

    @property
    def dialogue_by_speaker(self) -> dict[str, list[DialogueLine]]:
        """Group all dialogue lines by speaker, preserving script order."""
        grouped: dict[str, list[DialogueLine]] = {}
        for line in self.all_dialogue:
            grouped.setdefault(line.speaker, []).append(line)
        return grouped

    @property
    def lines_by_speaker(self) -> dict[str, list[DialogueLine]]:
        """Alias for :attr:`dialogue_by_speaker`."""
        return self.dialogue_by_speaker


ScriptModel = Script
ScriptBeat = Beat
BeatModel = Beat
Dialogue = DialogueLine
DialogueLineModel = DialogueLine
ScriptMode = WritingMode
WritingModeName = WritingMode
Mode = WritingMode
MediaPromptModel = MediaPrompt
MediaPromptKind = MediaKind
MediaType = MediaKind
