"""The story writing mode: prose with a chosen tone and reading time.

A story may arrive as plain text, a list of scenes, a mapping or an existing
payload. Blank lines separate scenes. The mode records the tone, characters,
setting, point of view and moral, builds one beat per scene and estimates a
deterministic reading time. When a length is requested the reading time must
fall inside its inclusive range, otherwise building fails with the measured time
and the target range in the error details.

No model is called here and nothing is imported from research, so the story
structure and its timing stay stable and offline.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from aivia.core.errors import ConfigError
from aivia.core.models.script import Beat, DialogueLine
from aivia.writing.length import (
    DEFAULT_WORDS_PER_SECOND,
    LENGTH_PRESETS,
    LengthRange,
    coerce_length_range,
    count_words,
    reading_time_seconds,
    resolve_length_range,
)
from aivia.writing.modes import (
    BuiltinWritingMode,
    register_builtin_writing_mode,
)
from aivia.writing.schema import Script, StoryPayload, WritingMode

__all__ = [
    "DEFAULT_WORDS_PER_SECOND",
    "LENGTH_PRESETS",
    "LengthRange",
    "Story",
    "StoryInput",
    "StoryMode",
    "StoryWritingMode",
    "build_story_script",
    "coerce_length_range",
    "coerce_story",
    "count_words",
    "extract_story",
    "normalise_story",
    "normalize_story",
    "parse_story",
    "reading_time_seconds",
    "story_script",
]

StoryInput = Any
"""Any value accepted as a story: text, scenes, a mapping or a payload."""

_FORM_KEYS = ("form", "style", "kind", "story_form")
_TONE_KEYS = ("tone", "mood", "voice")
_SCENE_KEYS = ("scenes", "scene", "beats", "parts")
_STORY_KEYS = ("story", "text", "content", "body", "prose")
_CHARACTER_KEYS = ("characters", "character", "cast", "people")
_SETTING_KEYS = ("setting", "place", "location")
_POV_KEYS = ("point_of_view", "pov", "perspective", "viewpoint")
_MORAL_KEYS = ("moral", "lesson", "message")
_LENGTH_KEYS = ("length", "target_length", "duration")
_TITLE_KEYS = ("title", "name", "subject")
_LANGUAGE_KEYS = ("language", "lang", "locale")
_HOOK_KEYS = ("hook", "opening", "intro")
_OUTRO_KEYS = ("outro", "closing", "ending")
_PAYLOAD_KEYS = ("payload", "mode_payload", "mode_data", "data", "options")
_MISSING = object()


def _value_of(value: Any, keys: Sequence[str]) -> Any:
    if isinstance(value, Mapping):
        for key in keys:
            if key in value:
                return value[key]
        return _MISSING
    for key in keys:
        if hasattr(value, key):
            return getattr(value, key)
    return _MISSING


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [
            item.strip() for item in value if isinstance(item, str) and item.strip()
        ]
    return []


def _names(value: Any) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in re.split(r"[,\n]", value) if part.strip()]
    return _string_list(value)


def _scenes(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        blocks = [
            block.strip()
            for block in re.split(r"\n\s*\n", value.strip())
            if block.strip()
        ]
        if len(blocks) > 1:
            return blocks
        return _string_list(value)
    return _string_list(value)


def _length_fields(value: Any) -> tuple[str, float | None, float | None]:
    raw = _value_of(value, _LENGTH_KEYS)
    if raw is _MISSING or raw is None or raw == "":
        return "", None, None
    target = coerce_length_range(raw)
    label = raw.strip() if isinstance(raw, str) else "custom"
    maximum = target.max_seconds if math.isfinite(target.max_seconds) else None
    return label, target.min_seconds, maximum


def _story_from_mapping(value: Mapping[str, Any]) -> StoryPayload:
    scenes = _scenes(_value_of(value, _SCENE_KEYS))
    if not scenes:
        raw = _value_of(value, _STORY_KEYS)
        if raw is not _MISSING and raw is not None:
            scenes = _scenes(raw)
    length, min_seconds, max_seconds = _length_fields(value)
    return StoryPayload(
        form=_text(_value_of(value, _FORM_KEYS)),
        tone=_text(_value_of(value, _TONE_KEYS)),
        scenes=scenes,
        characters=_names(_value_of(value, _CHARACTER_KEYS)),
        setting=_text(_value_of(value, _SETTING_KEYS)),
        point_of_view=_text(_value_of(value, _POV_KEYS)),
        moral=_text(_value_of(value, _MORAL_KEYS)),
        length=length,
        min_seconds=min_seconds,
        max_seconds=max_seconds,
    )


def _merge_metadata(payload: StoryPayload, value: Mapping[str, Any]) -> StoryPayload:
    outer = _story_from_mapping(value)
    return StoryPayload(
        form=outer.form or payload.form,
        tone=outer.tone or payload.tone,
        scenes=outer.scenes or payload.scenes,
        characters=outer.characters or payload.characters,
        setting=outer.setting or payload.setting,
        point_of_view=outer.point_of_view or payload.point_of_view,
        moral=outer.moral or payload.moral,
        length=outer.length or payload.length,
        min_seconds=(
            outer.min_seconds if outer.min_seconds is not None else payload.min_seconds
        ),
        max_seconds=(
            outer.max_seconds if outer.max_seconds is not None else payload.max_seconds
        ),
        reading_time_seconds=payload.reading_time_seconds,
    )


def _story_from(value: Any) -> StoryPayload:
    if isinstance(value, StoryPayload):
        return value
    if isinstance(value, Script) and isinstance(value.payload, StoryPayload):
        return value.payload
    if isinstance(value, str):
        return StoryPayload(scenes=_scenes(value))
    if isinstance(value, (bytes, bytearray)) or value is None:
        return StoryPayload()

    if isinstance(value, Mapping):
        nested = _value_of(value, _PAYLOAD_KEYS)
        if nested is not _MISSING and nested is not None:
            return _merge_metadata(_story_from(nested), value)
        return _story_from_mapping(value)

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return _story_from_mapping({"scenes": list(value)})

    if (
        _value_of(value, _SCENE_KEYS) is not _MISSING
        or _value_of(value, _STORY_KEYS) is not _MISSING
    ):
        fields = {
            key: getattr(value, key)
            for key in (
                *_FORM_KEYS,
                *_TONE_KEYS,
                *_SCENE_KEYS,
                *_STORY_KEYS,
                *_CHARACTER_KEYS,
                *_SETTING_KEYS,
                *_POV_KEYS,
                *_MORAL_KEYS,
                *_LENGTH_KEYS,
            )
            if hasattr(value, key)
        }
        return _story_from_mapping(fields)

    raise ConfigError(
        f"cannot build a story from {type(value).__name__}",
        details={"value": repr(value)},
    )


def extract_story(value: StoryInput = None) -> StoryPayload:
    """Return *value* as a typed :class:`StoryPayload`.

    A string is split into scenes on blank lines. A mapping accepts the schema
    aliases for tone, scenes, characters, setting, point of view, moral and
    length.
    """
    return _story_from(value)


coerce_story = extract_story
parse_story = extract_story
normalize_story = extract_story
normalise_story = extract_story


def _target_range(
    payload: StoryPayload,
    length: Any,
    min_seconds: float | None,
    max_seconds: float | None,
) -> LengthRange | None:
    base = length if length is not None else (payload.length or None)
    if (
        length is None
        and min_seconds is None
        and max_seconds is None
        and payload.min_seconds is not None
    ):
        maximum = payload.max_seconds if payload.max_seconds is not None else math.inf
        return LengthRange(payload.min_seconds, maximum)
    return resolve_length_range(
        base,
        min_seconds=min_seconds,
        max_seconds=max_seconds,
    )


def _check_reading_time(
    mode: str,
    measured: float,
    target: LengthRange | None,
) -> None:
    if target is None or target.contains(measured):
        return
    raise ConfigError(
        f"{mode} reading time {measured:.1f}s is outside the target range "
        f"{target.min_seconds:.1f}-{target.max_seconds:.1f}s",
        details={
            "mode": mode,
            "measured_seconds": round(measured, 3),
            "min_seconds": target.min_seconds,
            "max_seconds": target.max_seconds,
        },
    )


def _optional_text(
    value: Any,
    keys: Sequence[str],
    explicit: str | None,
) -> str | None:
    if explicit is not None:
        return explicit.strip() or None
    found = _text(_value_of(value, keys))
    return found or None


class StoryMode(BuiltinWritingMode):
    """The story writing mode.

    :meth:`build_script` turns prose into a script with one beat per scene, a
    recorded tone and reading time, and an optional length range it must
    satisfy. :meth:`build_payload` keeps the base contract and returns the typed
    :class:`StoryPayload`.
    """

    name = "story"
    description = "Prose with a chosen tone and reading time."
    instructions = (
        "Keep the requested tone and point of view, preserve the scene order "
        "and stay inside the reading time the profile asks for."
    )

    def build_payload(self, value: StoryInput = None) -> StoryPayload:
        """Validate *value* as a :class:`StoryPayload`."""
        return extract_story(value)

    def build_hook(
        self,
        value: StoryInput = None,
        *,
        title: str = "",
        hook: str | None = None,
    ) -> str:
        """Return the opening scene, using an explicit *hook* when given."""
        if hook is not None:
            return hook.strip()
        payload = extract_story(value)
        if not payload.scenes:
            return ""
        subject = title.strip()
        if subject:
            return f"{subject}: {payload.scenes[0]}"
        return payload.scenes[0]

    def build_beats(
        self,
        value: StoryInput = None,
        *,
        speaker: str = "narrator",
        max_beats: int | None = None,
    ) -> list[Beat]:
        """Return one beat per scene, each narrated by *speaker*."""
        scenes = extract_story(value).scenes
        if max_beats is not None:
            scenes = scenes[:max_beats]
        return [
            Beat(
                index=index,
                text=scene,
                dialogue=[DialogueLine(speaker=speaker, text=scene)],
            )
            for index, scene in enumerate(scenes)
        ]

    def build_outro(
        self,
        value: StoryInput = None,
        *,
        outro: str | None = None,
    ) -> str:
        """Return the closing line, using an explicit *outro* when given."""
        if outro is not None:
            return outro.strip()
        return extract_story(value).moral

    def build_script(
        self,
        value: StoryInput = None,
        *,
        title: str | None = None,
        language: str | None = None,
        form: str | None = None,
        tone: str | None = None,
        length: Any = None,
        min_seconds: float | None = None,
        max_seconds: float | None = None,
        words_per_second: float | None = None,
        characters: Sequence[str] | str | None = None,
        setting: str | None = None,
        point_of_view: str | None = None,
        moral: str | None = None,
        hook: str | None = None,
        outro: str | None = None,
        speaker: str = "narrator",
        max_beats: int | None = None,
    ) -> Script:
        """Build a story script from text, scenes, a mapping or a payload."""
        payload = extract_story(value)
        scenes = payload.scenes
        if not scenes:
            raise ConfigError(
                "story mode needs at least one scene",
                details={"mode": self.name},
            )

        resolved_characters = (
            _names(characters) if characters is not None else list(payload.characters)
        )
        resolved_setting = setting.strip() if setting is not None else payload.setting
        resolved_pov = (
            point_of_view.strip()
            if point_of_view is not None
            else payload.point_of_view
        )
        resolved_moral = moral.strip() if moral is not None else payload.moral
        target = _target_range(payload, length, min_seconds, max_seconds)
        measured = reading_time_seconds(
            "\n\n".join(scenes),
            words_per_second=words_per_second,
        )
        if measured <= 0:
            raise ConfigError(
                "story mode needs at least one word",
                details={"mode": self.name},
            )
        _check_reading_time(self.name, measured, target)
        duration = round(measured, 3)
        payload = StoryPayload(
            form=form.strip() if form is not None else payload.form,
            tone=tone.strip() if tone is not None else payload.tone,
            scenes=scenes,
            characters=resolved_characters,
            setting=resolved_setting,
            point_of_view=resolved_pov,
            moral=resolved_moral,
            length=(
                length.strip()
                if isinstance(length, str)
                else (payload.length if length is None else "custom")
            ),
            min_seconds=target.min_seconds if target is not None else None,
            max_seconds=(
                target.max_seconds
                if target is not None and math.isfinite(target.max_seconds)
                else None
            ),
            reading_time_seconds=duration,
        )
        resolved_title = (
            title if title is not None else _text(_value_of(value, _TITLE_KEYS))
        )
        return Script(
            mode=WritingMode.STORY,
            title=resolved_title,
            language=(language or _text(_value_of(value, _LANGUAGE_KEYS)) or "en"),
            hook=self.build_hook(
                payload,
                title=resolved_title,
                hook=_optional_text(value, _HOOK_KEYS, hook),
            ),
            beats=self.build_beats(
                payload,
                speaker=speaker,
                max_beats=max_beats,
            ),
            outro=self.build_outro(
                payload,
                outro=_optional_text(value, _OUTRO_KEYS, outro),
            ),
            payload=payload,
            duration_seconds=duration,
        )

    script = build_script
    make_script = build_script
    build_from_story = build_script


StoryWritingMode = StoryMode
Story = StoryMode


def build_story_script(
    value: StoryInput = None,
    **kwargs: Any,
) -> Script:
    """Build a story script with a default :class:`StoryMode`."""
    return StoryMode().build_script(value, **kwargs)


story_script = build_story_script


register_builtin_writing_mode(WritingMode.STORY, StoryMode)
