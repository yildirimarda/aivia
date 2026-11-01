"""Subtitle cues, timed words and deterministic SRT, VTT and ASS output.

A transcript or a forced alignment produces words with start and end times.
Delivery needs three representations of the same timed text:

* SRT and VTT are plain exchange formats consumed by players and platforms;
* ASS is the styled format a renderer burns into a video.

The subtitle model keeps the timed words and the cues derived from them
together. Export is pure: the same cues, style, canvas and safe area always
produce the same bytes. Word highlighting in ASS is either cumulative karaoke
(``\\k`` tags) or a current-word event per word, and margins come from a
:class:`~aivia.core.platforms.SafeArea` so subtitles stay clear of platform
chrome.

Text is always treated as data. Braces in ASS and markup characters in VTT are
escaped before they reach the output, so a subtitle can never inject an ASS
override tag or a VTT cue setting.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Any, Self

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from aivia.core.config import media as media_defaults
from aivia.core.config.sections import AccessibilityConfig
from aivia.core.errors import ValidationError
from aivia.core.platforms import Platform, SafeArea, get_platform
from aivia.media.text_layout import (
    DEFAULT_LINE_BREAK_LANGUAGE,
    break_lines,
    bundled_font_path,
    font_covers_text,
    hyphenate_word,
    line_break_opportunities,
    missing_glyphs,
    split_syllables,
    wrap_subtitle_text,
    wrap_text,
)
from aivia.providers.protocols.audio import WordTimestamp
from aivia.providers.protocols.stt import STTResponse

__all__ = [
    "DEFAULT_LINE_BREAK_LANGUAGE",
    "DEFAULT_SUBTITLE_FONT",
    "DEFAULT_SUBTITLE_FONT_SIZE",
    "DEFAULT_SUBTITLE_HIGHLIGHT_COLOR",
    "DEFAULT_SUBTITLE_MAX_CHARS",
    "DEFAULT_SUBTITLE_MAX_DURATION_SECONDS",
    "DEFAULT_SUBTITLE_MAX_GAP_SECONDS",
    "DEFAULT_SUBTITLE_PRIMARY_COLOR",
    "DEFAULT_SUBTITLE_STYLE",
    "STYLE_PRESETS",
    "SUBTITLE_STYLE_PRESETS",
    "Subtitle",
    "SubtitleAlignment",
    "SubtitleCue",
    "SubtitleFormat",
    "SubtitleHighlight",
    "SubtitleStyle",
    "SubtitleStylePreset",
    "SubtitleTrack",
    "SubtitleWord",
    "ass_color",
    "ass_for",
    "break_lines",
    "build_ass",
    "build_cues",
    "bundled_font_path",
    "cues_from_words",
    "export_ass",
    "export_srt",
    "export_subtitles",
    "export_vtt",
    "font_covers_text",
    "format_ass_timestamp",
    "format_srt_timestamp",
    "format_vtt_timestamp",
    "from_alignment",
    "from_stt",
    "from_transcript",
    "from_word_timestamps",
    "from_words",
    "generate_ass",
    "generate_srt",
    "generate_vtt",
    "get_style_preset",
    "group_words",
    "hyphenate_word",
    "line_break_opportunities",
    "list_style_presets",
    "missing_glyphs",
    "parse_style",
    "render_ass",
    "resolve_style",
    "split_syllables",
    "style_preset",
    "subtitle_from_alignment",
    "subtitle_from_transcript",
    "subtitle_from_words",
    "to_ass",
    "to_ass_color",
    "to_srt",
    "to_vtt",
    "wrap_subtitle_text",
    "wrap_text",
    "write_ass",
    "write_srt",
    "write_subtitles",
    "write_vtt",
]

DEFAULT_SUBTITLE_MAX_CHARS = media_defaults.DEFAULT_MEDIA_SUBTITLE_MAX_CHARS
"""Longest cue text before the next word starts a new cue."""

DEFAULT_SUBTITLE_MAX_DURATION_SECONDS = (
    media_defaults.DEFAULT_MEDIA_SUBTITLE_MAX_DURATION_SECONDS
)
"""Longest cue duration before the next word starts a new cue."""

DEFAULT_SUBTITLE_MAX_GAP_SECONDS = media_defaults.DEFAULT_MEDIA_SUBTITLE_MAX_GAP_SECONDS
"""A pause longer than this starts a new cue."""

DEFAULT_SUBTITLE_FONT = "Lato"
DEFAULT_SUBTITLE_FONT_SIZE = 32
DEFAULT_SUBTITLE_PRIMARY_COLOR = "#FFFFFF"
DEFAULT_SUBTITLE_HIGHLIGHT_COLOR = "#FFD400"

_NAMED_COLORS: dict[str, str] = {
    "black": "#000000",
    "white": "#FFFFFF",
    "red": "#FF0000",
    "green": "#008000",
    "blue": "#0000FF",
    "yellow": "#FFFF00",
    "cyan": "#00FFFF",
    "magenta": "#FF00FF",
    "transparent": "#00000000",
}

_ASS_COLOR = re.compile(r"^&H(?P<value>[0-9A-Fa-f]{6,8})&$")


class SubtitleFormat(StrEnum):
    """A subtitle file format the model can export."""

    SRT = "srt"
    VTT = "vtt"
    ASS = "ass"

    @classmethod
    def parse(cls, value: str | SubtitleFormat) -> SubtitleFormat:
        """Return a format from a name or file suffix."""
        if isinstance(value, cls):
            return value
        key = value.strip().lower().lstrip(".")
        if key == "subrip":
            key = cls.SRT.value
        elif key in {"webvtt", "web-vtt"}:
            key = cls.VTT.value
        try:
            return cls(key)
        except ValueError as exc:
            allowed = ", ".join(item.value for item in cls)
            raise ValidationError(
                f"unsupported subtitle format {value!r}; expected one of {allowed}"
            ) from exc

    @classmethod
    def from_path(cls, path: Path) -> SubtitleFormat:
        """Return the format implied by a destination file suffix."""
        if not path.suffix:
            raise ValidationError(
                f"cannot infer a subtitle format from {path.name!r}; "
                "pass an explicit format"
            )
        return cls.parse(path.suffix)


class SubtitleAlignment(IntEnum):
    """ASS alignment values, named for readability."""

    BOTTOM_LEFT = 1
    BOTTOM_CENTER = 2
    BOTTOM_RIGHT = 3
    MIDDLE_LEFT = 4
    MIDDLE_CENTER = 5
    MIDDLE_RIGHT = 6
    TOP_LEFT = 7
    TOP_CENTER = 8
    TOP_RIGHT = 9


class SubtitleHighlight(StrEnum):
    """How ASS word highlighting is produced."""

    KARAOKE = "karaoke"
    WORD = "word"
    NONE = "none"

    @classmethod
    def parse(cls, value: str | SubtitleHighlight) -> SubtitleHighlight:
        """Return a highlight mode from a name or alias."""
        if isinstance(value, cls):
            return value
        key = value.strip().lower().replace("-", "_")
        aliases = {
            "karaoke": cls.KARAOKE,
            "k": cls.KARAOKE,
            "word": cls.WORD,
            "current": cls.WORD,
            "highlight": cls.WORD,
            "none": cls.NONE,
            "off": cls.NONE,
            "false": cls.NONE,
        }
        try:
            return aliases[key]
        except KeyError as exc:
            allowed = ", ".join(item.value for item in cls)
            raise ValidationError(
                f"unsupported highlight mode {value!r}; expected one of {allowed}"
            ) from exc


class SubtitleStylePreset(StrEnum):
    """A named subtitle style shipped with the media toolkit."""

    DEFAULT = "default"
    MINIMAL = "minimal"
    BOLD = "bold"
    OUTLINE = "outline"
    BOX = "box"
    KARAOKE = "karaoke"

    @classmethod
    def parse(cls, value: str | SubtitleStylePreset) -> SubtitleStylePreset:
        """Return a preset from a name or alias."""
        if isinstance(value, cls):
            return value
        key = value.strip().lower().replace("_", "-")
        aliases = {
            "default": cls.DEFAULT,
            "plain": cls.DEFAULT,
            "minimal": cls.MINIMAL,
            "clean": cls.MINIMAL,
            "bold": cls.BOLD,
            "outline": cls.OUTLINE,
            "box": cls.BOX,
            "background": cls.BOX,
            "karaoke": cls.KARAOKE,
            "highlight": cls.KARAOKE,
        }
        try:
            return aliases[key]
        except KeyError as exc:
            allowed = ", ".join(item.value for item in cls)
            raise ValidationError(
                f"unknown subtitle style preset {value!r}; expected one of {allowed}"
            ) from exc


def _strip_text(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    return value


def _normalize_color(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        named = _NAMED_COLORS.get(text.lower())
        if named is not None:
            return named
        if text.startswith("#"):
            digits = text[1:]
        elif text.lower().startswith("0x"):
            digits = text[2:]
        elif _ASS_COLOR.match(text):
            match = _ASS_COLOR.match(text)
            assert match is not None
            digits = match.group("value")
            if len(digits) == 8:
                alpha, blue, green, red = (
                    digits[0:2],
                    digits[2:4],
                    digits[4:6],
                    digits[6:8],
                )
                return f"#{red}{green}{blue}{alpha}".upper()
            blue, green, red = digits[0:2], digits[2:4], digits[4:6]
            return f"#{red}{green}{blue}".upper()
        else:
            digits = text
        if len(digits) == 3:
            digits = "".join(character * 2 for character in digits)
        if len(digits) not in {6, 8}:
            raise ValueError("color must be #RRGGBB, #RRGGBBAA or an &HAABBGGRR& value")
        try:
            int(digits, 16)
        except ValueError as exc:
            raise ValueError("color must contain hexadecimal digits") from exc
        return f"#{digits.upper()}"
    if isinstance(value, int) and not isinstance(value, bool):
        if 0 <= value <= 0xFFFFFF:
            return f"#{value:06X}"
        raise ValueError("integer color must be between 0x000000 and 0xFFFFFF")
    if isinstance(value, tuple) and len(value) in {3, 4}:
        channels: list[int] = []
        for channel in value:
            if (
                isinstance(channel, bool)
                or not isinstance(channel, int)
                or not 0 <= channel <= 255
            ):
                raise ValueError("color channels must be integers between 0 and 255")
            channels.append(channel)
        if len(channels) == 3:
            return "#{:02X}{:02X}{:02X}".format(*channels)
        return "#{:02X}{:02X}{:02X}{:02X}".format(*channels)
    raise ValueError("color must be a hex string, integer or RGB tuple")


def _rgba(value: str) -> tuple[int, int, int, int]:
    normalized = _normalize_color(value)
    digits = normalized[1:]
    red = int(digits[0:2], 16)
    green = int(digits[2:4], 16)
    blue = int(digits[4:6], 16)
    alpha = int(digits[6:8], 16) if len(digits) == 8 else 255
    return red, green, blue, alpha


def ass_color(value: str | int | tuple[int, ...]) -> str:
    """Return *value* as an ASS ``&HAABBGGRR&`` color.

    CSS alpha is opaque at ``FF`` while ASS alpha is opaque at ``00``, so the
    two representations are inverted rather than copied.
    """
    red, green, blue, alpha = _rgba(_normalize_color(value))
    ass_alpha = 255 - alpha
    return f"&H{ass_alpha:02X}{blue:02X}{green:02X}{red:02X}&"


to_ass_color = ass_color
"""Alias for :func:`ass_color`."""


def _alignment(value: Any) -> int:
    if isinstance(value, SubtitleAlignment):
        return int(value)
    if isinstance(value, str):
        key = value.strip().lower().replace("-", "_").replace(" ", "_")
        names = {
            "bottom_left": 1,
            "bottom": 2,
            "bottom_center": 2,
            "bottom_centre": 2,
            "center": 2,
            "centre": 2,
            "bottom_right": 3,
            "middle_left": 4,
            "middle": 5,
            "middle_center": 5,
            "middle_centre": 5,
            "middle_right": 6,
            "top_left": 7,
            "top": 8,
            "top_center": 8,
            "top_centre": 8,
            "top_right": 9,
        }
        if key in names:
            return names[key]
        if key.isdigit() and 1 <= int(key) <= 9:
            return int(key)
        raise ValueError("alignment must be 1-9 or a named position")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("alignment must be an integer between 1 and 9")
    if not 1 <= value <= 9:
        raise ValueError("alignment must be between 1 and 9")
    return value


class SubtitleStyle(BaseModel):
    """The visual style of an ASS subtitle line.

    The model mirrors the ASS ``Style`` fields that matter for subtitles while
    keeping colors as familiar hex strings. Presets are ordinary instances, so
    a caller can start from one and override a single field.
    """

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
    )

    font: str = Field(
        default=DEFAULT_SUBTITLE_FONT,
        min_length=1,
        validation_alias=AliasChoices("font", "fontname", "font_name", "family"),
    )
    font_size: int = Field(
        default=DEFAULT_SUBTITLE_FONT_SIZE,
        gt=0,
        validation_alias=AliasChoices("font_size", "fontsize", "size"),
    )
    primary_color: str = Field(
        default=DEFAULT_SUBTITLE_PRIMARY_COLOR,
        validation_alias=AliasChoices("primary_color", "primary", "color"),
    )
    secondary_color: str = Field(
        default=DEFAULT_SUBTITLE_PRIMARY_COLOR,
        validation_alias=AliasChoices("secondary_color", "secondary"),
    )
    highlight_color: str = Field(
        default=DEFAULT_SUBTITLE_HIGHLIGHT_COLOR,
        validation_alias=AliasChoices(
            "highlight_color",
            "highlight",
            "accent_color",
        ),
    )
    outline_color: str = Field(
        default="#000000",
        validation_alias=AliasChoices("outline_color", "border_color"),
    )
    back_color: str = Field(
        default="#000000",
        validation_alias=AliasChoices("back_color", "background_color"),
    )
    bold: bool = False
    italic: bool = False
    underline: bool = False
    strikeout: bool = False
    scale_x: float = Field(default=100.0, gt=0)
    scale_y: float = Field(default=100.0, gt=0)
    spacing: float = 0.0
    angle: float = 0.0
    border_style: int = Field(default=1, ge=1, le=3)
    outline: float = Field(
        default=2.0,
        ge=0.0,
        validation_alias=AliasChoices("outline", "outline_width", "border"),
    )
    shadow: float = Field(default=1.0, ge=0.0)
    alignment: int = Field(default=2, validation_alias=AliasChoices("alignment"))
    margin_l: int = Field(default=0, ge=0)
    margin_r: int = Field(default=0, ge=0)
    margin_v: int = Field(default=0, ge=0)
    encoding: int = Field(default=1, ge=0)
    background: bool = False
    highlight_mode: SubtitleHighlight = Field(
        default=SubtitleHighlight.KARAOKE,
        validation_alias=AliasChoices("highlight_mode", "highlighting"),
    )

    @field_validator("font", mode="before")
    @classmethod
    def _validate_font(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip()
            if not text:
                raise ValueError("font must be non-empty")
            return text
        return value

    @field_validator(
        "primary_color",
        "secondary_color",
        "highlight_color",
        "outline_color",
        "back_color",
        mode="before",
    )
    @classmethod
    def _validate_color(cls, value: Any) -> Any:
        return _normalize_color(value)

    @field_validator("alignment", mode="before")
    @classmethod
    def _validate_alignment(cls, value: Any) -> Any:
        return _alignment(value)

    @field_validator("highlight_mode", mode="before")
    @classmethod
    def _validate_highlight_mode(cls, value: Any) -> Any:
        if value is None:
            return SubtitleHighlight.KARAOKE
        if isinstance(value, bool):
            return SubtitleHighlight.KARAOKE if value else SubtitleHighlight.NONE
        return SubtitleHighlight.parse(value)

    @model_validator(mode="after")
    def _apply_background(self) -> Self:
        if self.background:
            object.__setattr__(self, "border_style", 3)
        return self

    @classmethod
    def preset(cls, name: str | SubtitleStylePreset) -> SubtitleStyle:
        """Return a copy of the named preset."""
        return get_style_preset(name)

    @classmethod
    def from_preset(cls, name: str | SubtitleStylePreset) -> SubtitleStyle:
        """Alias for :meth:`preset`."""
        return cls.preset(name)

    @classmethod
    def resolve(
        cls,
        value: SubtitleStyleInput,
    ) -> SubtitleStyle:
        """Return a style from an instance, preset name or mapping."""
        return resolve_style(value)

    @classmethod
    def from_config(cls, config: AccessibilityConfig) -> SubtitleStyle:
        """Build a style from the accessibility settings."""
        return cls(
            font=config.subtitle_font,
            font_size=config.subtitle_font_size,
            outline=2.0 if config.subtitle_outline else 0.0,
            shadow=1.0 if config.subtitle_outline else 0.0,
            background=config.subtitle_background,
        )

    @property
    def ass_name(self) -> str:
        """A stable ASS style name derived from the font size and weight."""
        weight = "Bold" if self.bold else "Regular"
        return f"Subtitle{weight}{self.font_size}"

    def ass_style_line(
        self,
        *,
        name: str | None = None,
        margins: tuple[int, int, int] | None = None,
    ) -> str:
        """Return the ASS ``Style`` line for this style."""
        style_name = name or self.ass_name
        left, right, vertical = (
            (self.margin_l, self.margin_r, self.margin_v)
            if margins is None
            else margins
        )
        primary = (
            self.highlight_color
            if self.highlight_mode is SubtitleHighlight.KARAOKE
            else self.primary_color
        )
        secondary = (
            self.secondary_color
            if self.highlight_mode is SubtitleHighlight.KARAOKE
            else self.highlight_color
        )
        border_style = 3 if self.background else self.border_style
        return (
            f"Style: {style_name},{self.font},{self.font_size},"
            f"{ass_color(primary)},{ass_color(secondary)},"
            f"{ass_color(self.outline_color)},{ass_color(self.back_color)},"
            f"{-1 if self.bold else 0},{-1 if self.italic else 0},"
            f"{-1 if self.underline else 0},{-1 if self.strikeout else 0},"
            f"{self.scale_x:g},{self.scale_y:g},{self.spacing:g},{self.angle:g},"
            f"{border_style},{self.outline:g},{self.shadow:g},{self.alignment},"
            f"{left},{right},{vertical},{self.encoding}"
        )


DEFAULT_SUBTITLE_STYLE = SubtitleStyle()

type SubtitleStyleInput = (
    SubtitleStyle | str | SubtitleStylePreset | Mapping[str, Any] | None
)
"""A style instance, a preset name, a mapping or ``None`` for the default."""


STYLE_PRESETS: dict[str, SubtitleStyle] = {
    SubtitleStylePreset.DEFAULT.value: DEFAULT_SUBTITLE_STYLE,
    SubtitleStylePreset.MINIMAL.value: SubtitleStyle(
        outline=0.0,
        shadow=0.0,
        highlight_mode=SubtitleHighlight.NONE,
    ),
    SubtitleStylePreset.BOLD.value: SubtitleStyle(
        font_size=42,
        bold=True,
        outline=3.0,
        shadow=1.0,
    ),
    SubtitleStylePreset.OUTLINE.value: SubtitleStyle(
        bold=True,
        outline=3.5,
        shadow=0.0,
    ),
    SubtitleStylePreset.BOX.value: SubtitleStyle(
        bold=True,
        background=True,
        outline=8.0,
        shadow=0.0,
        back_color="#000000CC",
    ),
    SubtitleStylePreset.KARAOKE.value: SubtitleStyle(
        font_size=40,
        bold=True,
        highlight_color="#FFD400",
        highlight_mode=SubtitleHighlight.KARAOKE,
    ),
}

SUBTITLE_STYLE_PRESETS = STYLE_PRESETS
"""Alias for :data:`STYLE_PRESETS`."""


def get_style_preset(name: str | SubtitleStylePreset) -> SubtitleStyle:
    """Return a copy of the named style preset."""
    preset = SubtitleStylePreset.parse(name)
    return STYLE_PRESETS[preset.value].model_copy(deep=True)


style_preset = get_style_preset
"""Alias for :func:`get_style_preset`."""


def list_style_presets() -> tuple[str, ...]:
    """Return every preset name in deterministic order."""
    return tuple(sorted(STYLE_PRESETS))


def resolve_style(
    value: SubtitleStyleInput,
) -> SubtitleStyle:
    """Return a style from an instance, preset name or mapping."""
    if value is None:
        return get_style_preset(SubtitleStylePreset.DEFAULT)
    if isinstance(value, SubtitleStyle):
        return value.model_copy(deep=True)
    if isinstance(value, SubtitleStylePreset):
        return get_style_preset(value)
    if isinstance(value, str):
        return get_style_preset(value)
    if isinstance(value, Mapping):
        return SubtitleStyle.model_validate(dict(value))
    raise TypeError("style must be a SubtitleStyle, preset name or mapping")


parse_style = resolve_style
"""Alias for :func:`resolve_style`."""


class SubtitleWord(BaseModel):
    """One spoken word with its timing in seconds.

    The shape matches :class:`~aivia.providers.protocols.audio.WordTimestamp`
    so an alignment result can be handed to a subtitle without a conversion
    step. ``start``/``end``/``text`` are accepted as aliases.
    """

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
    )

    word: str = Field(
        min_length=1,
        validation_alias=AliasChoices("word", "text", "token"),
    )
    start_seconds: float = Field(
        ge=0.0,
        validation_alias=AliasChoices("start_seconds", "start", "start_time"),
    )
    end_seconds: float = Field(
        ge=0.0,
        validation_alias=AliasChoices("end_seconds", "end", "end_time"),
    )
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    speaker: str | None = None

    @field_validator("word", "speaker", mode="before")
    @classmethod
    def _strip_word(cls, value: Any) -> Any:
        return _strip_text(value)

    @model_validator(mode="after")
    def _validate_span(self) -> Self:
        if self.end_seconds < self.start_seconds:
            raise ValueError("word end_seconds must not be before start_seconds")
        return self

    @property
    def text(self) -> str:
        """Alias for :attr:`word`."""
        return self.word

    @property
    def start(self) -> float:
        """Alias for :attr:`start_seconds`."""
        return self.start_seconds

    @property
    def end(self) -> float:
        """Alias for :attr:`end_seconds`."""
        return self.end_seconds

    @property
    def duration_seconds(self) -> float:
        """The length of the spoken word in seconds."""
        return self.end_seconds - self.start_seconds

    @classmethod
    def from_timestamp(cls, value: WordTimestamp) -> SubtitleWord:
        """Convert a protocol word timestamp into a subtitle word."""
        return cls(
            word=value.word,
            start_seconds=value.start_seconds,
            end_seconds=value.end_seconds,
            confidence=value.confidence,
            speaker=value.speaker,
        )

    def to_timestamp(self) -> WordTimestamp:
        """Convert this word into the shared protocol model."""
        return WordTimestamp(
            word=self.word,
            start_seconds=self.start_seconds,
            end_seconds=self.end_seconds,
            confidence=self.confidence,
            speaker=self.speaker,
        )


def _coerce_word(value: Any) -> SubtitleWord:
    if isinstance(value, SubtitleWord):
        return value
    if isinstance(value, WordTimestamp):
        return SubtitleWord.from_timestamp(value)
    if isinstance(value, Mapping):
        return SubtitleWord.model_validate(dict(value))
    raise TypeError("subtitle words must be SubtitleWord, WordTimestamp or mappings")


def _coerce_words(values: Iterable[Any]) -> list[SubtitleWord]:
    words = [_coerce_word(value) for value in values]
    starts = [word.start_seconds for word in words]
    if starts != sorted(starts):
        raise ValidationError(
            "subtitle words must be ordered by start_seconds",
            details={"field": "words"},
        )
    return words


def _word_text(words: Sequence[SubtitleWord]) -> str:
    return " ".join(word.word for word in words).strip()


def _validate_positive(value: float, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(
            f"{field} must be a positive number",
            details={"field": field, "value": value},
        )
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValidationError(
            f"{field} must be a positive number",
            details={"field": field, "value": value},
        )
    return number


def _validate_gap(value: float, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(
            f"{field} must be a non-negative number",
            details={"field": field, "value": value},
        )
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValidationError(
            f"{field} must be a non-negative number",
            details={"field": field, "value": value},
        )
    return number


def _validate_cue_limit(value: int, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValidationError(
            f"{field} must be a positive integer",
            details={"field": field, "value": value},
        )
    return value


def build_cues(
    words: Iterable[Any],
    *,
    max_chars: int = DEFAULT_SUBTITLE_MAX_CHARS,
    max_duration_seconds: float = DEFAULT_SUBTITLE_MAX_DURATION_SECONDS,
    max_gap_seconds: float = DEFAULT_SUBTITLE_MAX_GAP_SECONDS,
    max_duration: float | None = None,
    max_gap: float | None = None,
) -> list[SubtitleCue]:
    """Group timed *words* into cues at gaps, character and duration limits."""
    if max_duration is not None:
        max_duration_seconds = max_duration
    if max_gap is not None:
        max_gap_seconds = max_gap
    limit_chars = _validate_cue_limit(max_chars, field="max_chars")
    limit_duration = _validate_positive(
        max_duration_seconds,
        field="max_duration_seconds",
    )
    limit_gap = _validate_gap(max_gap_seconds, field="max_gap_seconds")
    ordered = _coerce_words(words)
    if not ordered:
        return []

    groups: list[list[SubtitleWord]] = []
    current: list[SubtitleWord] = []
    for word in ordered:
        if current:
            gap = word.start_seconds - current[-1].end_seconds
            candidate = _word_text([*current, word])
            duration = word.end_seconds - current[0].start_seconds
            if (
                gap > limit_gap
                or len(candidate) > limit_chars
                or duration > limit_duration
            ):
                groups.append(current)
                current = []
        current.append(word)
    if current:
        groups.append(current)

    return [
        SubtitleCue(
            start_seconds=group[0].start_seconds,
            end_seconds=max(word.end_seconds for word in group),
            text=_word_text(group),
            words=group,
        )
        for group in groups
    ]


cues_from_words = build_cues
"""Alias for :func:`build_cues`."""

group_words = build_cues
"""Alias for :func:`build_cues`."""


class SubtitleCue(BaseModel):
    """One subtitle cue with an inclusive start and end time."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
    )

    start_seconds: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices("start_seconds", "start", "start_time"),
    )
    end_seconds: float | None = Field(
        default=None,
        ge=0.0,
        validation_alias=AliasChoices("end_seconds", "end", "end_time"),
    )
    text: str = ""
    words: list[SubtitleWord] = Field(default_factory=list)
    speaker: str | None = None

    @field_validator("text", "speaker", mode="before")
    @classmethod
    def _strip_cue_text(cls, value: Any) -> Any:
        return _strip_text(value)

    @field_validator("words", mode="before")
    @classmethod
    def _coerce_cue_words(cls, value: Any) -> Any:
        if value is None:
            return []
        return [_coerce_word(item) for item in value]

    @model_validator(mode="after")
    def _validate_cue(self) -> Self:
        if not self.text and self.words:
            object.__setattr__(self, "text", _word_text(self.words))
        end = self.end_seconds
        if end is None:
            if self.words:
                end = max(word.end_seconds for word in self.words)
            else:
                end = self.start_seconds
            object.__setattr__(self, "end_seconds", end)
        if not self.text:
            raise ValueError("subtitle cue needs text or words")
        if end < self.start_seconds:
            raise ValueError("cue end_seconds must not be before start_seconds")
        starts = [word.start_seconds for word in self.words]
        if starts != sorted(starts):
            raise ValueError("cue words must be ordered by start_seconds")
        return self

    @classmethod
    def from_words(
        cls,
        words: Iterable[Any],
        *,
        start_seconds: float | None = None,
        end_seconds: float | None = None,
        text: str | None = None,
    ) -> SubtitleCue:
        """Build a cue from words, deriving the span and text when omitted."""
        items = _coerce_words(words)
        if not items:
            raise ValidationError(
                "a cue needs at least one word",
                details={"field": "words"},
            )
        return cls(
            start_seconds=(
                items[0].start_seconds if start_seconds is None else start_seconds
            ),
            end_seconds=(
                max(word.end_seconds for word in items)
                if end_seconds is None
                else end_seconds
            ),
            text=_word_text(items) if text is None else text,
            words=items,
        )

    @property
    def start(self) -> float:
        """Alias for :attr:`start_seconds`."""
        return self.start_seconds

    @property
    def end(self) -> float:
        """Alias for :attr:`end_seconds`."""
        return self.end_seconds or self.start_seconds

    @property
    def duration_seconds(self) -> float:
        """The length of the cue in seconds."""
        return self.end - self.start_seconds

    @property
    def lines(self) -> list[str]:
        """The cue text split into display lines."""
        return self.text.splitlines()

    def wrapped_lines(
        self,
        *,
        max_chars: int,
        language: str = "",
    ) -> list[str]:
        """Return the cue text wrapped with the language line break rules."""
        return break_lines(self.text, max_chars=max_chars, language=language)

    def wrap(self, *, max_chars: int, language: str = "") -> str:
        """Return the cue text wrapped into display lines joined by newlines."""
        return wrap_text(self.text, max_chars=max_chars, language=language)

    @property
    def timed_words(self) -> list[WordTimestamp]:
        """The cue words as shared protocol models."""
        return [word.to_timestamp() for word in self.words]

    def to_srt(self, *, index: int = 1) -> str:
        """Return this cue as one SRT block."""
        return (
            f"{index}\n"
            f"{format_srt_timestamp(self.start_seconds)} --> "
            f"{format_srt_timestamp(self.end)}\n"
            f"{self.text}\n"
        )

    def to_vtt(self) -> str:
        """Return this cue as one VTT block."""
        return (
            f"{format_vtt_timestamp(self.start_seconds)} --> "
            f"{format_vtt_timestamp(self.end)}\n"
            f"{escape_vtt_text(self.text)}\n"
        )


def _coerce_cue(value: Any) -> SubtitleCue:
    if isinstance(value, SubtitleCue):
        return value
    if isinstance(value, (SubtitleWord, WordTimestamp)):
        return SubtitleCue.from_words([value])
    if isinstance(value, Mapping):
        payload = dict(value)
        if "word" in payload and "words" not in payload:
            return SubtitleCue.from_words([payload])
        return SubtitleCue.model_validate(payload)
    raise TypeError("subtitle cues must be SubtitleCue or mappings")


class Subtitle(BaseModel):
    """A subtitle track with timed words, cues, a style and a language."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
    )

    cues: list[SubtitleCue] = Field(default_factory=list)
    words: list[SubtitleWord] = Field(default_factory=list)
    language: str = ""
    style: SubtitleStyle = Field(default_factory=lambda: get_style_preset("default"))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("cues", mode="before")
    @classmethod
    def _coerce_cues(cls, value: Any) -> Any:
        if value is None:
            return []
        return [_coerce_cue(item) for item in value]

    @field_validator("words", mode="before")
    @classmethod
    def _coerce_track_words(cls, value: Any) -> Any:
        if value is None:
            return []
        return [_coerce_word(item) for item in value]

    @field_validator("language", mode="before")
    @classmethod
    def _normalize_language(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("style", mode="before")
    @classmethod
    def _coerce_style(cls, value: Any) -> Any:
        return resolve_style(value)

    @model_validator(mode="after")
    def _derive_track(self) -> Self:
        if not self.cues and self.words:
            object.__setattr__(self, "cues", build_cues(self.words))
        if not self.words and self.cues:
            object.__setattr__(
                self,
                "words",
                [word for cue in self.cues for word in cue.words],
            )
        return self

    @classmethod
    def from_words(
        cls,
        words: Iterable[Any],
        *,
        language: str = "",
        style: SubtitleStyleInput = None,
        preset: str | SubtitleStylePreset | None = None,
        style_preset: str | SubtitleStylePreset | None = None,
        max_chars: int = DEFAULT_SUBTITLE_MAX_CHARS,
        max_duration_seconds: float = DEFAULT_SUBTITLE_MAX_DURATION_SECONDS,
        max_gap_seconds: float = DEFAULT_SUBTITLE_MAX_GAP_SECONDS,
        max_duration: float | None = None,
        max_gap: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Subtitle:
        """Build a track from timed words, grouping them into cues."""
        items = _coerce_words(words)
        selected = preset if preset is not None else style_preset
        chosen = resolve_style(style if selected is None else selected)
        return cls(
            cues=build_cues(
                items,
                max_chars=max_chars,
                max_duration_seconds=max_duration_seconds,
                max_gap_seconds=max_gap_seconds,
                max_duration=max_duration,
                max_gap=max_gap,
            ),
            words=items,
            language=language,
            style=chosen,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_transcript(
        cls,
        transcript: STTResponse | Mapping[str, Any] | Iterable[Any],
        **kwargs: Any,
    ) -> Subtitle:
        """Build a track from an STT response, mapping or word sequence."""
        if isinstance(transcript, STTResponse):
            words: list[Any] = transcript.words or [
                word for segment in transcript.segments for word in segment.words
            ]
            kwargs.setdefault("language", transcript.language or "")
            metadata = dict(kwargs.pop("metadata", {}) or {})
            metadata.setdefault("provider", transcript.provider)
            metadata.setdefault("model", transcript.model)
            kwargs["metadata"] = metadata
        elif isinstance(transcript, Mapping):
            payload = dict(transcript)
            words = list(payload.get("words") or [])
            if not words:
                words = [
                    word
                    for segment in payload.get("segments") or []
                    for word in (
                        segment.get("words", [])
                        if isinstance(segment, Mapping)
                        else getattr(segment, "words", [])
                    )
                ]
            if "language" not in kwargs and payload.get("language"):
                kwargs["language"] = payload["language"]
        else:
            words = list(transcript)
        return cls.from_words(words, **kwargs)

    @classmethod
    def from_stt(cls, transcript: Any, **kwargs: Any) -> Subtitle:
        """Alias for :meth:`from_transcript`."""
        return cls.from_transcript(transcript, **kwargs)

    @classmethod
    def from_word_timestamps(cls, words: Iterable[Any], **kwargs: Any) -> Subtitle:
        """Alias for :meth:`from_words`."""
        return cls.from_words(words, **kwargs)

    @classmethod
    def from_alignment(cls, alignment: Any, **kwargs: Any) -> Subtitle:
        """Build a track from an alignment, keeping only timed words."""
        words = getattr(alignment, "word_timestamps", None)
        if words is None:
            words = getattr(alignment, "words", None)
        if words is None:
            raise TypeError("alignment must expose word_timestamps or words")
        timed: list[WordTimestamp] = []
        for word in words:
            if isinstance(word, WordTimestamp):
                timed.append(word)
                continue
            start = getattr(word, "start_seconds", None)
            end = getattr(word, "end_seconds", None)
            if start is None or end is None:
                continue
            timed.append(
                WordTimestamp(
                    word=word.word,
                    start_seconds=start,
                    end_seconds=end,
                    confidence=getattr(word, "confidence", None),
                )
            )
        if "language" not in kwargs:
            language = getattr(alignment, "language", None)
            if language:
                kwargs["language"] = language
        return cls.from_words(timed, **kwargs)

    @property
    def text(self) -> str:
        """The concatenated cue text."""
        return "\n".join(cue.text for cue in self.cues)

    @property
    def start_seconds(self) -> float:
        """The earliest cue start, or zero for an empty track."""
        return min((cue.start_seconds for cue in self.cues), default=0.0)

    @property
    def end_seconds(self) -> float:
        """The latest cue end, or zero for an empty track."""
        return max((cue.end for cue in self.cues), default=0.0)

    @property
    def duration_seconds(self) -> float:
        """The span of the whole track in seconds."""
        return self.end_seconds - self.start_seconds

    @property
    def srt(self) -> str:
        """The track as an SRT document."""
        return self.to_srt()

    @property
    def vtt(self) -> str:
        """The track as a WebVTT document."""
        return self.to_vtt()

    @property
    def timed_words(self) -> list[WordTimestamp]:
        """Every word as a shared protocol model."""
        return [word.to_timestamp() for word in self.words]

    @property
    def word_timestamps(self) -> list[WordTimestamp]:
        """Alias for :attr:`timed_words`."""
        return self.timed_words

    def to_srt(self) -> str:
        """Return the track as an SRT document."""
        if not self.cues:
            return ""
        blocks = [cue.to_srt(index=index) for index, cue in enumerate(self.cues, 1)]
        return "\n".join(blocks).rstrip() + "\n"

    def to_vtt(self) -> str:
        """Return the track as a WebVTT document."""
        blocks = [cue.to_vtt() for cue in self.cues]
        body = "\n".join(blocks).rstrip()
        return "WEBVTT\n\n" + (body + "\n" if body else "")

    def generate_ass(self, **kwargs: Any) -> str:
        """Alias for :meth:`to_ass`."""
        return self.to_ass(**kwargs)

    def to_ass(
        self,
        *,
        width: int = 1920,
        height: int = 1080,
        style: SubtitleStyleInput = None,
        preset: str | SubtitleStylePreset | None = None,
        style_preset: str | SubtitleStylePreset | None = None,
        safe_area: SafeArea | Mapping[str, float] | None = None,
        platform: Platform | str | None = None,
        highlight: bool | SubtitleHighlight | str = True,
    ) -> str:
        """Return the track as an ASS document rendered for one canvas.

        ``safe_area`` and ``platform`` both describe the margins kept clear of
        platform chrome. An explicit ``safe_area`` wins over the platform.
        """
        return generate_ass(
            self,
            width=width,
            height=height,
            style=style,
            preset=preset,
            style_preset=style_preset,
            safe_area=safe_area,
            platform=platform,
            highlight=highlight,
        )

    def export(self, format: str | SubtitleFormat, **kwargs: Any) -> str:
        """Return the track in the requested format."""
        target = SubtitleFormat.parse(format)
        if target is SubtitleFormat.SRT:
            return self.to_srt()
        if target is SubtitleFormat.VTT:
            return self.to_vtt()
        return self.to_ass(**kwargs)

    def write(
        self,
        path: str | Path,
        *,
        format: str | SubtitleFormat | None = None,
        **kwargs: Any,
    ) -> Path:
        """Write the track to *path*, inferring the format from its suffix."""
        destination = Path(path)
        target = (
            SubtitleFormat.from_path(destination)
            if format is None
            else SubtitleFormat.parse(format)
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            self.export(target, **kwargs),
            encoding="utf-8",
            newline="\n",
        )
        return destination

    def export_srt(self) -> str:
        """Return the track as an SRT document."""
        return self.to_srt()

    def export_vtt(self) -> str:
        """Return the track as a WebVTT document."""
        return self.to_vtt()

    def export_ass(self, **kwargs: Any) -> str:
        """Return the track as an ASS document."""
        return self.to_ass(**kwargs)

    def write_srt(self, path: str | Path) -> Path:
        """Write the track as SRT."""
        return self.write(path, format=SubtitleFormat.SRT)

    def write_vtt(self, path: str | Path) -> Path:
        """Write the track as VTT."""
        return self.write(path, format=SubtitleFormat.VTT)

    def write_ass(self, path: str | Path, **kwargs: Any) -> Path:
        """Write the track as ASS."""
        return self.write(path, format=SubtitleFormat.ASS, **kwargs)


SubtitleTrack = Subtitle
"""Alias for :class:`Subtitle`."""


def _milliseconds(seconds: float) -> int:
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise ValidationError(
            "subtitle time must be a number",
            details={"value": seconds},
        )
    value = float(seconds)
    if not math.isfinite(value) or value < 0:
        raise ValidationError(
            "subtitle time must be a finite, non-negative number",
            details={"value": seconds},
        )
    return round(value * 1000)


def _centiseconds(seconds: float) -> int:
    return round(_milliseconds(seconds) / 10)


def format_srt_timestamp(seconds: float) -> str:
    """Format *seconds* as an SRT ``HH:MM:SS,mmm`` timestamp."""
    total = _milliseconds(seconds)
    hours, remainder = divmod(total, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole:02d},{millis:03d}"


def format_vtt_timestamp(seconds: float) -> str:
    """Format *seconds* as a WebVTT ``HH:MM:SS.mmm`` timestamp."""
    return format_srt_timestamp(seconds).replace(",", ".")


def format_ass_timestamp(seconds: float) -> str:
    """Format *seconds* as an ASS ``H:MM:SS.cc`` timestamp."""
    total = _centiseconds(seconds)
    hours, remainder = divmod(total, 360_000)
    minutes, remainder = divmod(remainder, 6000)
    whole, centis = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{whole:02d}.{centis:02d}"


def escape_vtt_text(text: str) -> str:
    """Escape markup characters in a VTT cue text."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def escape_ass_text(text: str) -> str:
    """Escape braces, backslashes and newlines in an ASS text field."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    escaped = normalized.replace("\\", "\\\\")
    escaped = escaped.replace("{", "\\{").replace("}", "\\}")
    return escaped.replace("\n", "\\N")


def _coerce_safe_area(
    safe_area: SafeArea | Mapping[str, float] | None,
    *,
    platform: Platform | str | None,
) -> SafeArea:
    from_platform: SafeArea | None = None
    if isinstance(platform, Platform):
        from_platform = platform.safe_area
    elif isinstance(platform, str):
        from_platform = get_platform(platform).safe_area
    elif platform is not None:
        raise TypeError("platform must be a Platform, platform id or None")
    if safe_area is None:
        return _validate_safe_area(from_platform or SafeArea())
    if isinstance(safe_area, SafeArea):
        return _validate_safe_area(safe_area)
    if isinstance(safe_area, Mapping):
        try:
            area = SafeArea(
                top=float(safe_area.get("top", 0.0)),
                bottom=float(safe_area.get("bottom", 0.0)),
                left=float(safe_area.get("left", 0.0)),
                right=float(safe_area.get("right", 0.0)),
            )
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "safe area margins must be numbers",
                details={"field": "safe_area"},
            ) from exc
        return _validate_safe_area(area)
    raise TypeError("safe_area must be a SafeArea, mapping or None")


def _validate_safe_area(area: SafeArea) -> SafeArea:
    values = (area.top, area.bottom, area.left, area.right)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) < 1.0
        for value in values
    ):
        raise ValidationError(
            "safe area margins must be fractions between 0 and 1",
            details={"field": "safe_area"},
        )
    if area.horizontal >= 1.0 or area.vertical >= 1.0:
        raise ValidationError(
            "safe area margins must leave a visible area",
            details={"field": "safe_area"},
        )
    return area


def _validate_canvas(width: int, height: int) -> tuple[int, int]:
    for value, label in ((width, "width"), (height, "height")):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValidationError(
                f"subtitle canvas {label} must be a positive integer",
                details={"field": label},
            )
    return width, height


def _ass_margins(
    style: SubtitleStyle,
    area: SafeArea,
    *,
    width: int,
    height: int,
) -> tuple[int, int, int]:
    top, right, bottom, left = area.to_pixels(width, height)
    if style.alignment in {1, 2, 3}:
        vertical = bottom
    elif style.alignment in {7, 8, 9}:
        vertical = top
    else:
        vertical = max(top, bottom)
    return left + style.margin_l, right + style.margin_r, vertical + style.margin_v


def _karaoke_text(
    cue: SubtitleCue,
    *,
    style: SubtitleStyle,
) -> str:
    words = cue.words
    if not words:
        color = ass_color(style.primary_color)
        return f"{{\\1c{color}}}{escape_ass_text(cue.text)}"
    start_cs = _centiseconds(cue.start_seconds)
    cursor = start_cs
    parts: list[str] = []
    for index, word in enumerate(words):
        word_start = _centiseconds(word.start_seconds)
        word_end = _centiseconds(word.end_seconds)
        if word_start > cursor:
            parts.append(f"{{\\k{word_start - cursor}}}")
            cursor = word_start
        duration = max(0, word_end - max(word_start, cursor))
        if index:
            parts.append(" ")
        parts.append(f"{{\\k{duration}}}{escape_ass_text(word.word)}")
        cursor = max(cursor, word_end)
    return "".join(parts)


def _word_events(
    cue: SubtitleCue,
    *,
    style: SubtitleStyle,
    start: str,
    end: str,
    name: str,
    margins: tuple[int, int, int],
) -> list[str]:
    words = cue.words
    if not words:
        return [
            _dialogue_line(
                start=start,
                end=end,
                name=name,
                margins=margins,
                text=escape_ass_text(cue.text),
            )
        ]
    normal = ass_color(style.primary_color)
    highlight = ass_color(style.highlight_color)
    events: list[str] = []
    for index, word in enumerate(words):
        spans: list[str] = []
        for other_index, other in enumerate(words):
            text = escape_ass_text(other.word)
            if other_index == index:
                spans.append(f"{{\\1c{highlight}}}{text}{{\\1c{normal}}}")
            else:
                spans.append(text)
        events.append(
            _dialogue_line(
                start=format_ass_timestamp(word.start_seconds),
                end=format_ass_timestamp(word.end_seconds),
                name=name,
                margins=margins,
                text=" ".join(spans),
            )
        )
    return events


def _dialogue_line(
    *,
    start: str,
    end: str,
    name: str,
    margins: tuple[int, int, int],
    text: str,
) -> str:
    left, right, vertical = margins
    return f"Dialogue: 0,{start},{end},{name},,{left},{right},{vertical},,{text}"


def _highlight_mode(
    highlight: bool | SubtitleHighlight | str,
    style: SubtitleStyle,
) -> SubtitleHighlight:
    if isinstance(highlight, bool):
        return style.highlight_mode if highlight else SubtitleHighlight.NONE
    return SubtitleHighlight.parse(highlight)


def generate_ass(
    subtitle: Subtitle | Iterable[SubtitleCue] | Iterable[SubtitleWord],
    *,
    width: int = 1920,
    height: int = 1080,
    style: SubtitleStyleInput = None,
    preset: str | SubtitleStylePreset | None = None,
    style_preset: str | SubtitleStylePreset | None = None,
    safe_area: SafeArea | Mapping[str, float] | None = None,
    platform: Platform | str | None = None,
    highlight: bool | SubtitleHighlight | str = True,
) -> str:
    """Render a subtitle track as an ASS document.

    The canvas size sets ``PlayResX``/``PlayResY``. Safe area margins are
    converted to ASS pixel margins for the style's alignment, so a bottom
    aligned subtitle never enters the platform's bottom chrome.
    """
    track = _as_subtitle(subtitle)
    chosen = resolve_style(track.style if style is None else style)
    selected = preset if preset is not None else style_preset
    if selected is not None:
        chosen = resolve_style(selected)
    mode = _highlight_mode(highlight, chosen)
    canvas_width, canvas_height = _validate_canvas(width, height)
    area = _validate_safe_area(_coerce_safe_area(safe_area, platform=platform))
    margins = _ass_margins(chosen, area, width=canvas_width, height=canvas_height)
    name = chosen.ass_name

    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {canvas_width}",
        f"PlayResY: {canvas_height}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        (
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding"
        ),
        chosen.ass_style_line(name=name, margins=margins),
        "",
        "[Events]",
        (
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, "
            "MarginV, Effect, Text"
        ),
    ]

    for cue in track.cues:
        start = format_ass_timestamp(cue.start_seconds)
        end = format_ass_timestamp(cue.end)
        if mode is SubtitleHighlight.WORD:
            lines.extend(
                _word_events(
                    cue,
                    style=chosen,
                    start=start,
                    end=end,
                    name=name,
                    margins=margins,
                )
            )
            continue
        if mode is SubtitleHighlight.KARAOKE:
            text = _karaoke_text(cue, style=chosen)
        else:
            text = escape_ass_text(cue.text)
        lines.append(
            _dialogue_line(
                start=start,
                end=end,
                name=name,
                margins=margins,
                text=text,
            )
        )
    return "\n".join(lines).rstrip() + "\n"


build_ass = generate_ass
"""Alias for :func:`generate_ass`."""

render_ass = generate_ass
"""Alias for :func:`generate_ass`."""

ass_for = generate_ass
"""Alias for :func:`generate_ass`."""


def _as_subtitle(
    source: Subtitle | Iterable[SubtitleCue] | Iterable[SubtitleWord],
) -> Subtitle:
    if isinstance(source, Subtitle):
        return source
    items = list(source)
    if not items:
        return Subtitle()
    if all(isinstance(item, SubtitleCue) for item in items):
        return Subtitle(cues=[item for item in items if isinstance(item, SubtitleCue)])
    return Subtitle.from_words(items)


def to_srt(
    source: Subtitle | Iterable[SubtitleCue] | Iterable[SubtitleWord],
) -> str:
    """Return *source* as an SRT document."""
    return _as_subtitle(source).to_srt()


generate_srt = to_srt
"""Alias for :func:`to_srt`."""

export_srt = to_srt
"""Alias for :func:`to_srt`."""


def to_vtt(
    source: Subtitle | Iterable[SubtitleCue] | Iterable[SubtitleWord],
) -> str:
    """Return *source* as a WebVTT document."""
    return _as_subtitle(source).to_vtt()


generate_vtt = to_vtt
"""Alias for :func:`to_vtt`."""

export_vtt = to_vtt
"""Alias for :func:`to_vtt`."""


def to_ass(
    source: Subtitle | Iterable[SubtitleCue] | Iterable[SubtitleWord],
    **kwargs: Any,
) -> str:
    """Return *source* as an ASS document."""
    return generate_ass(source, **kwargs)


export_ass = to_ass
"""Alias for :func:`to_ass`."""


def from_words(words: Iterable[Any], **kwargs: Any) -> Subtitle:
    """Build a subtitle from timed words."""
    return Subtitle.from_words(words, **kwargs)


subtitle_from_words = from_words
"""Alias for :func:`from_words`."""


def from_transcript(transcript: Any, **kwargs: Any) -> Subtitle:
    """Build a subtitle from a transcript response or word sequence."""
    return Subtitle.from_transcript(transcript, **kwargs)


subtitle_from_transcript = from_transcript
"""Alias for :func:`from_transcript`."""

from_stt = from_transcript
"""Alias for :func:`from_transcript`."""

from_word_timestamps = from_words
"""Alias for :func:`from_words`."""


def from_alignment(alignment: Any, **kwargs: Any) -> Subtitle:
    """Build a subtitle from an alignment, keeping timed words only."""
    return Subtitle.from_alignment(alignment, **kwargs)


subtitle_from_alignment = from_alignment
"""Alias for :func:`from_alignment`."""


def export_subtitles(
    subtitle: Subtitle | Iterable[SubtitleCue] | Iterable[SubtitleWord],
    format: str | SubtitleFormat,
    **kwargs: Any,
) -> str:
    """Return *subtitle* in the requested format."""
    return _as_subtitle(subtitle).export(format, **kwargs)


def write_subtitles(
    subtitle: Subtitle | Iterable[SubtitleCue] | Iterable[SubtitleWord],
    path: str | Path,
    *,
    format: str | SubtitleFormat | None = None,
    **kwargs: Any,
) -> Path:
    """Write *subtitle* to *path*, inferring the format from its suffix."""
    return _as_subtitle(subtitle).write(path, format=format, **kwargs)


def write_srt(
    subtitle: Subtitle | Iterable[SubtitleCue] | Iterable[SubtitleWord],
    path: str | Path,
) -> Path:
    """Write *subtitle* as SRT."""
    return _as_subtitle(subtitle).write_srt(path)


def write_vtt(
    subtitle: Subtitle | Iterable[SubtitleCue] | Iterable[SubtitleWord],
    path: str | Path,
) -> Path:
    """Write *subtitle* as VTT."""
    return _as_subtitle(subtitle).write_vtt(path)


def write_ass(
    subtitle: Subtitle | Iterable[SubtitleCue] | Iterable[SubtitleWord],
    path: str | Path,
    **kwargs: Any,
) -> Path:
    """Write *subtitle* as ASS."""
    return _as_subtitle(subtitle).write_ass(path, **kwargs)
