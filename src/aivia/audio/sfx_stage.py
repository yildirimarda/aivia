"""Place rule matched sound effects on a timeline without changing its length.

A sound effect is usually shorter than a beat and is triggered by something
that happens in the script: a door slams, a camera clicks, a crowd cheers. This
stage turns a small ordered rule set into concrete timeline tracks. A rule may
name a beat directly, match literal terms in a beat's text, or use a regular
expression. It chooses a placement (start, middle, end, matched word or an
explicit offset), a gain and optionally an existing audio file. When no file is
supplied the request goes through the single screened, cached and budgeted sound
effect service.

Placement is deterministic and bounded by the timeline. Effects that would run
past the end are shortened and effects that start past the end are moved back,
so adding sound effects never changes the total duration. Existing effects are
replaced by default while music and voice tracks are left untouched.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from pydantic import (
    ValidationError as PydanticValidationError,
)

from aivia.compose.schema import AudioTrackRole as TimelineTrackRole
from aivia.compose.timeline import Timeline, TimelineBeat, TimelineMusic
from aivia.core.config.media import (
    DEFAULT_SFX_DURATION_SECONDS,
    DEFAULT_SFX_VOLUME_DB,
)
from aivia.core.errors import ConfigError
from aivia.core.models.base import canonical_json
from aivia.core.models.run import StageResult, StageState
from aivia.core.stages.base import Stage, StageContext
from aivia.providers.protocols.audio import AudioFormat
from aivia.providers.protocols.sfx import SFXRequest

__all__ = [
    "DEFAULT_SFX_DURATION_SECONDS",
    "DEFAULT_SFX_FILENAME",
    "DEFAULT_SFX_VOLUME_DB",
    "DEFAULT_TIMELINE_FILENAME",
    "SFX_POSITIONS",
    "SFX_STAGE_NAME",
    "SFXPlacement",
    "SFXRule",
    "SFXStage",
    "SFXStageResult",
    "SelectedSFX",
    "SfxPlacement",
    "SfxRule",
    "SfxStage",
    "SfxStageResult",
    "SoundEffectRule",
    "SoundEffectStage",
    "SoundEffectsStage",
    "build_sfx_stage",
    "load_sfx_rules",
    "match_sfx_rules",
    "normalise_sfx_rules",
    "normalize_sfx_rules",
    "place_sound_effects",
    "run_sfx_stage",
    "sfx_stage",
]

SFX_STAGE_NAME = "sfx"
"""The recipe stage name this implementation serves."""

DEFAULT_TIMELINE_FILENAME = "timeline.json"
"""The timeline file the voice and music stages write under the run directory."""

DEFAULT_SFX_FILENAME = "sfx.json"
"""The manifest that records the rules and their placed effects."""

SFX_POSITIONS = ("start", "center", "end", "word", "offset", "absolute")
"""Where a rule may place its effect relative to a beat or the timeline."""

_EPSILON = 1e-9
_MIN_DURATION = 1e-6
_SECTION_KEYS = ("sfx", "sound_effects", "sound-effects", "effects")
_RULE_KEYS = ("rules", "effects", "sound_effects", "sound-effects", "cues")
_POSITION_ALIASES = {
    "absolute": "absolute",
    "after": "end",
    "anchor": "word",
    "at_end": "end",
    "at_start": "start",
    "begin": "start",
    "beginning": "start",
    "beat_end": "end",
    "beat_start": "start",
    "before": "start",
    "center": "center",
    "centre": "center",
    "close": "end",
    "delay": "offset",
    "end": "end",
    "finish": "end",
    "global": "absolute",
    "intro": "start",
    "keyword": "word",
    "match": "word",
    "mid": "center",
    "middle": "center",
    "offset": "offset",
    "open": "start",
    "outro": "end",
    "relative": "offset",
    "start": "start",
    "timeline": "absolute",
    "word": "word",
}


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _finite(
    value: Any,
    name: str,
    *,
    non_negative: bool = False,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number", details={"value": repr(value)})
    number = float(value)
    if not math.isfinite(number):
        raise ConfigError(f"{name} must be finite", details={"value": repr(value)})
    if non_negative and number < 0.0:
        raise ConfigError(
            f"{name} must not be negative", details={"value": repr(value)}
        )
    if positive and number <= 0.0:
        raise ConfigError(f"{name} must be positive", details={"value": repr(value)})
    return number


def _optional_finite(
    value: Any,
    name: str,
    *,
    non_negative: bool = False,
    positive: bool = False,
) -> float | None:
    if value is None:
        return None
    return _finite(value, name, non_negative=non_negative, positive=positive)


def _section(config: Any, name: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name, None)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _config_value(context: StageContext, *names: str) -> Any:
    for section_name in _SECTION_KEYS:
        section = _section(context.config, section_name)
        if section is None:
            continue
        for name in names:
            raw = _field(section, name)
            if raw is not None:
                return raw
    return None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray, Mapping, BaseModel)):
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    return [value]


def _normalise_text(value: Any) -> str:
    return " ".join(_text(value).split()).casefold()


def _term_pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(term)
    if term and (term[0].isalnum() or term[0] == "_"):
        escaped = rf"(?<!\w){escaped}"
    if term and (term[-1].isalnum() or term[-1] == "_"):
        escaped = rf"{escaped}(?!\w)"
    return re.compile(escaped, re.IGNORECASE | re.UNICODE)


def _matches_term(term: str, text: str) -> bool:
    return bool(term) and bool(_term_pattern(term).search(text))


def _terms(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items: Iterable[Any] = value.split(",") if "," in value else [value]
    elif isinstance(value, Mapping):
        for key in ("terms", "keywords", "words", "all", "any", "text", "value"):
            if key in value:
                return _terms(value[key])
        items = value.values()
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        items = value
    else:
        raise ValueError("match terms must be a string or a sequence of strings")
    result: list[str] = []
    for item in items:
        if isinstance(item, (Mapping, list, tuple, set, frozenset)):
            for text in _terms(item):
                if text not in result:
                    result.append(text)
            continue
        text = _text(item)
        if text and text not in result:
            result.append(text)
    return result


def _patterns(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items: Iterable[Any] = [value]
    elif isinstance(value, Mapping):
        for key in ("patterns", "regex", "regexes", "pattern"):
            if key in value:
                return _patterns(value[key])
        items = value.values()
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        items = value
    else:
        raise ValueError("patterns must be a string or a sequence of strings")
    result: list[str] = []
    for item in items:
        if isinstance(item, (Mapping, list, tuple, set, frozenset)):
            for text in _patterns(item):
                if text not in result:
                    result.append(text)
            continue
        text = _text(item)
        if not text:
            continue
        try:
            re.compile(text)
        except re.error as exc:
            raise ValueError(f"invalid sfx rule pattern {text!r}: {exc}") from exc
        if text not in result:
            result.append(text)
    return result


def _languages(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items: Iterable[Any] = value.split(",")
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        items = value
    else:
        raise ValueError("languages must be a language code or a sequence")
    result: list[str] = []
    for item in items:
        code = _text(item).replace("_", "-").casefold()
        if code and code not in result:
            result.append(code)
    return result


def _position(value: Any, start_seconds: float | None = None) -> str:
    if value is None:
        return "offset" if start_seconds is not None else "start"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "offset"
    text = _text(value).casefold().replace("-", "_").replace(" ", "_")
    if not text:
        return "offset" if start_seconds is not None else "start"
    if text not in _POSITION_ALIASES:
        raise ValueError(
            f"unknown sfx placement: {value!r} (choose from {', '.join(SFX_POSITIONS)})"
        )
    return _POSITION_ALIASES[text]


class SfxRule(BaseModel):
    """One rule that matches a beat and describes the effect to place there.

    A rule matches by ``beat``/``index``, by literal ``match`` terms or by a
    ``pattern`` regular expression. A rule with no match criteria applies to
    every beat. ``source`` places an existing file without a provider call;
    otherwise ``prompt`` is sent to the sound effect service.
    """

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        validate_assignment=True,
    )

    id: str = Field(
        default="",
        validation_alias=AliasChoices("id", "name", "rule", "rule_id", "key", "label"),
    )
    prompt: str = Field(
        default="",
        validation_alias=AliasChoices(
            "prompt",
            "description",
            "effect",
            "sfx",
            "sound",
            "sound_effect",
            "sound-effect",
            "text",
            "instruction",
        ),
    )
    source: str = Field(
        default="",
        validation_alias=AliasChoices(
            "source",
            "path",
            "file",
            "audio",
            "asset",
            "asset_id",
            "clip",
            "file_path",
        ),
    )
    beat: str = Field(
        default="",
        validation_alias=AliasChoices("beat", "beat_id", "segment", "segment_id"),
    )
    index: int | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices("index", "beat_index", "order", "number"),
    )
    terms: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "match",
            "matches",
            "terms",
            "keywords",
            "contains",
            "trigger",
            "triggers",
            "when",
        ),
    )
    patterns: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("pattern", "patterns", "regex", "regexes"),
    )
    position: str = Field(
        default="",
        validation_alias=AliasChoices(
            "position",
            "place",
            "placement",
            "anchor",
        ),
    )
    start_seconds: float | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "start_seconds",
            "start",
            "offset_seconds",
            "offset",
            "at_seconds",
        ),
    )
    volume_db: float | None = Field(
        default=None,
        validation_alias=AliasChoices("volume_db", "volume", "gain_db", "gain"),
    )
    duration_seconds: float | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "duration_seconds",
            "duration",
            "length_seconds",
            "length",
        ),
    )
    fade_in_seconds: float | None = Field(
        default=None,
        validation_alias=AliasChoices("fade_in_seconds", "fade_in"),
    )
    fade_out_seconds: float | None = Field(
        default=None,
        validation_alias=AliasChoices("fade_out_seconds", "fade_out"),
    )
    loop: bool = False
    languages: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("languages", "language", "lang", "locale"),
    )
    once: bool = False
    enabled: bool = Field(
        default=True, validation_alias=AliasChoices("enabled", "active", "visible")
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _accept_shorthands(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"prompt": data}
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        at = values.pop("at", None)
        if at is not None:
            if isinstance(at, (int, float)) and not isinstance(at, bool):
                values.setdefault("start_seconds", float(at))
                values.setdefault("position", "offset")
            else:
                text = _text(at)
                try:
                    number = float(text)
                except ValueError:
                    values.setdefault("position", text)
                else:
                    values.setdefault("start_seconds", number)
                    values.setdefault("position", "offset")
        return values

    @field_validator("id", "prompt", "source", "beat", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _text(value)

    @field_validator("terms", mode="before")
    @classmethod
    def _coerce_terms(cls, value: Any) -> Any:
        return _terms(value)

    @field_validator("patterns", mode="before")
    @classmethod
    def _coerce_patterns(cls, value: Any) -> Any:
        return _patterns(value)

    @field_validator("languages", mode="before")
    @classmethod
    def _coerce_languages(cls, value: Any) -> Any:
        return _languages(value)

    @field_validator(
        "start_seconds",
        "fade_in_seconds",
        "fade_out_seconds",
        mode="before",
    )
    @classmethod
    def _validate_seconds(cls, value: Any) -> Any:
        return _optional_finite(value, "sfx time", non_negative=True)

    @field_validator("duration_seconds", mode="before")
    @classmethod
    def _validate_duration(cls, value: Any) -> Any:
        return _optional_finite(value, "duration_seconds", positive=True)

    @field_validator("volume_db", mode="before")
    @classmethod
    def _validate_volume(cls, value: Any) -> Any:
        return _optional_finite(value, "volume_db")

    @field_validator("metadata", mode="before")
    @classmethod
    def _coerce_metadata(cls, value: Any) -> Any:
        return {} if value is None else value

    @model_validator(mode="after")
    def _validate_rule(self) -> Self:
        if not self.prompt and not self.source:
            raise ValueError("sfx rule needs a prompt or a source")
        object.__setattr__(
            self,
            "position",
            _position(self.position, self.start_seconds),
        )
        return self

    @property
    def triggers(self) -> tuple[str, ...]:
        """Every literal term and pattern this rule matches."""
        return (*self.terms, *self.patterns)


def _load_rule_file(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(
            f"could not read the sfx rules: {path}",
            details={"path": str(path), "error": str(exc)},
        ) from exc
    try:
        if path.suffix.casefold() in {".json"}:
            return json.loads(text)
        return yaml.safe_load(text)
    except (ValueError, yaml.YAMLError) as exc:
        raise ConfigError(
            f"invalid sfx rules file: {path}",
            details={"path": str(path), "error": str(exc)},
        ) from exc


def _rule_from_value(value: Any, *, index: int) -> SfxRule:
    try:
        if isinstance(value, SfxRule):
            rule = value.model_copy(deep=True)
        elif isinstance(value, Mapping):
            rule = SfxRule.model_validate(dict(value))
        elif isinstance(value, str):
            rule = SfxRule(prompt=value)
        else:
            raise ConfigError(
                "an sfx rule must be a mapping, a prompt string or an SfxRule",
                details={"value": type(value).__name__},
            )
    except PydanticValidationError as exc:
        raise ConfigError(
            f"invalid sfx rule: {exc}",
            details={"index": index, "error": str(exc)},
        ) from exc
    if not rule.id:
        object.__setattr__(rule, "id", f"rule_{index}")
    return rule


def load_sfx_rules(value: Any) -> tuple[SfxRule, ...]:
    """Load an ordered rule set from a mapping, sequence or YAML/JSON file."""
    if value is None:
        return ()
    if isinstance(value, SfxRule):
        return (value.model_copy(deep=True),)
    if isinstance(value, (str, Path)):
        text = str(value).strip()
        path = Path(text).expanduser()
        if path.is_file():
            return load_sfx_rules(_load_rule_file(path))
        if path.suffix.casefold() in {".yaml", ".yml", ".json"}:
            raise ConfigError(
                f"sfx rules file does not exist: {path}",
                details={"path": str(path)},
            )
        return (SfxRule(prompt=text, id="rule_0"),)

    rules: list[Any]
    if isinstance(value, Mapping):
        values = dict(value)
        nested = next(
            (values[key] for key in _RULE_KEYS if key in values),
            None,
        )
        if nested is not None:
            return load_sfx_rules(nested)
        rules = []
        for key, raw in values.items():
            if isinstance(raw, Mapping):
                item = dict(raw)
                item.setdefault("id", key)
            elif isinstance(raw, str):
                item = {"id": key, "prompt": raw}
            else:
                item = raw
            rules.append(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        rules = list(value)
    else:
        raise ConfigError(
            "sfx rules must be a mapping, a sequence or a file path",
            details={"value": type(value).__name__},
        )

    loaded: list[SfxRule] = []
    seen: set[str] = set()
    for index, raw in enumerate(rules):
        rule = _rule_from_value(raw, index=index)
        key = rule.id.casefold()
        if key in seen:
            raise ConfigError(
                f"duplicate sfx rule id: {rule.id!r}",
                details={"rule": rule.id},
            )
        seen.add(key)
        loaded.append(rule)
    return tuple(loaded)


normalise_sfx_rules = load_sfx_rules
"""Alias for :func:`load_sfx_rules`."""

normalize_sfx_rules = load_sfx_rules
"""Alias for :func:`load_sfx_rules`."""


def _matched_terms(rule: SfxRule, text: str) -> tuple[str, ...] | None:
    if not rule.triggers:
        return ()
    matched: list[str] = []
    for term in rule.terms:
        if _matches_term(term, text):
            matched.append(term)
    for pattern in rule.patterns:
        found = re.search(pattern, text, re.IGNORECASE | re.UNICODE)
        if found is not None:
            matched.append(found.group(0))
    return tuple(matched) if matched else None


def match_sfx_rules(text: Any, rules: Iterable[Any]) -> list[SfxRule]:
    """Return the rules whose literal terms or patterns occur in *text*."""
    normalised = _text(text)
    result: list[SfxRule] = []
    for raw in rules:
        rule = raw if isinstance(raw, SfxRule) else _rule_from_value(raw, index=0)
        if not rule.enabled:
            continue
        if _matched_terms(rule, normalised) is None:
            continue
        result.append(rule)
    return result


@dataclass(slots=True)
class SfxPlacement:
    """One placed effect with its rule, timing, gain and provenance."""

    id: str
    rule: str
    source: str
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    volume_db: float
    origin: str
    beat: str = ""
    prompt: str = ""
    provider: str = ""
    model: str = ""
    cost_usd: float = 0.0
    matched_terms: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for manifests."""
        return {
            "id": self.id,
            "rule": self.rule,
            "beat": self.beat,
            "prompt": self.prompt,
            "source": self.source,
            "origin": self.origin,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "duration_seconds": self.duration_seconds,
            "volume_db": self.volume_db,
            "provider": self.provider,
            "model": self.model,
            "cost_usd": self.cost_usd,
            "matched_terms": list(self.matched_terms),
            "metadata": dict(self.metadata),
        }


SfxRuleModel = SfxRule
SFXRule = SfxRule
SoundEffectRule = SfxRule
SfxPlacementModel = SfxPlacement
SFXPlacement = SfxPlacement
SelectedSFX = SfxPlacement


def _sources(value: Any) -> tuple[Any, ...]:
    result: list[Any] = []
    for candidate in (
        value,
        getattr(value, "outcome", None),
        getattr(value, "result", None),
        getattr(value, "response", None),
        getattr(value, "asset", None),
        getattr(value, "clip", None),
    ):
        if candidate is not None and all(candidate is not item for item in result):
            result.append(candidate)
    return tuple(result)


def _path_of(value: Any) -> str:
    if isinstance(value, (str, Path)):
        return str(value)
    if isinstance(value, Mapping):
        for name in ("path", "source", "url", "file"):
            raw = value.get(name)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
        for name in ("paths", "assets", "clips"):
            raw = value.get(name)
            if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
                for item in raw:
                    found = _path_of(item)
                    if found:
                        return found
        return ""
    for source in _sources(value):
        for name in ("path", "source", "url", "file"):
            raw = getattr(source, name, None)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
        for name in ("paths", "assets", "clips"):
            raw = getattr(source, name, None)
            if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
                for item in raw:
                    found = _path_of(item)
                    if found:
                        return found
    return ""


def _data_of(value: Any) -> bytes | None:
    if isinstance(value, Mapping):
        for name in ("data", "audio"):
            raw = value.get(name)
            if isinstance(raw, (bytes, bytearray, memoryview)):
                return bytes(raw)
    for source in _sources(value):
        for name in ("data", "audio"):
            raw = getattr(source, name, None)
            if isinstance(raw, (bytes, bytearray, memoryview)):
                return bytes(raw)
        for name in ("clips", "assets"):
            raw = getattr(source, name, None)
            if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
                for item in raw:
                    found = _data_of(item)
                    if found is not None:
                        return found
    return None


def _format_of(value: Any) -> str:
    for source in _sources(value):
        raw = getattr(source, "format", None)
        if isinstance(raw, AudioFormat):
            return raw.value
        if isinstance(raw, str) and raw.strip():
            return raw.strip().casefold().lstrip(".")
        for name in ("clips", "assets"):
            raw = getattr(source, name, None)
            if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
                for item in raw:
                    found = _format_of(item)
                    if found:
                        return found
    path = _path_of(value)
    suffix = Path(path).suffix.casefold().lstrip(".") if path else ""
    return suffix or "mp3"


def _duration_of(value: Any) -> float | None:
    if isinstance(value, Mapping):
        for name in ("duration_seconds", "duration"):
            raw = value.get(name)
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                number = float(raw)
                if math.isfinite(number) and number > 0.0:
                    return number
    for source in _sources(value):
        for name in ("duration_seconds", "duration"):
            raw = getattr(source, name, None)
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                number = float(raw)
                if math.isfinite(number) and number > 0.0:
                    return number
        for name in ("assets", "probes", "clips"):
            raw = getattr(source, name, None)
            if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
                for item in raw:
                    found = _duration_of(item)
                    if found is not None:
                        return found
        analysis = getattr(source, "analysis", None)
        if analysis is not None:
            found = _duration_of(analysis)
            if found is not None:
                return found
    return None


def _cost_of(value: Any) -> float:
    if isinstance(value, Mapping):
        for name in ("cost_usd", "amount_usd", "total_cost_usd"):
            raw = value.get(name)
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                return max(0.0, float(raw))
        return 0.0
    for source in _sources(value):
        for name in ("cost_usd", "amount_usd", "total_cost_usd"):
            raw = getattr(source, name, None)
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                return max(0.0, float(raw))
        events = getattr(source, "events", None)
        if isinstance(events, Sequence) and not isinstance(events, (str, bytes)):
            total = sum(
                float(getattr(event, "amount_usd", 0.0) or 0.0) for event in events
            )
            if total:
                return max(0.0, total)
    return 0.0


def _provider_of(value: Any) -> str:
    for source in _sources(value):
        raw = getattr(source, "provider", None)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return ""


def _model_of(value: Any) -> str:
    for source in _sources(value):
        raw = getattr(source, "model", None)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return ""


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = canonical_json(value, indent=2) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _timeline_duration(timeline: Timeline) -> float:
    if timeline.duration_seconds is not None:
        return float(timeline.duration_seconds)
    ends: list[float] = []
    for position, beat in enumerate(timeline.beats):
        if not beat.enabled:
            continue
        if beat.end_seconds is not None:
            ends.append(float(beat.end_seconds))
        elif beat.duration_seconds is not None:
            ends.append(beat.start_seconds + float(beat.duration_seconds))
        elif position + 1 < len(timeline.beats):
            ends.append(timeline.beats[position + 1].start_seconds)
    for language in timeline.resolved_languages():
        for voice in language.voices:
            if not voice.enabled:
                continue
            if voice.end_seconds is not None:
                ends.append(float(voice.end_seconds))
            elif voice.duration_seconds is not None:
                ends.append(
                    (voice.start_seconds or 0.0) + float(voice.duration_seconds)
                )
        if language.subtitles is not None:
            ends.append(float(language.subtitles.end_seconds))
    for track in timeline.music:
        if not track.enabled:
            continue
        if track.end_seconds is not None:
            ends.append(float(track.end_seconds))
        elif track.duration_seconds is not None:
            ends.append(track.start_seconds + float(track.duration_seconds))
    return max(ends, default=0.0)


def _beat_end(
    beat: TimelineBeat,
    position: int,
    beats: Sequence[TimelineBeat],
    duration: float,
) -> float:
    if beat.end_seconds is not None:
        return min(float(beat.end_seconds), duration)
    if beat.duration_seconds is not None:
        return min(beat.start_seconds + float(beat.duration_seconds), duration)
    if position + 1 < len(beats):
        return min(beats[position + 1].start_seconds, duration)
    return duration


def _slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_")
    return text or "sfx"


def _word_start(
    rule: SfxRule,
    matched: tuple[str, ...],
    timeline: Timeline,
    beat: TimelineBeat,
) -> float | None:
    terms = [term for term in rule.terms if term]
    patterns = list(rule.patterns)
    for language in timeline.resolved_languages():
        for voice in language.voices:
            if not voice.enabled or (voice.beat and voice.beat != beat.id):
                continue
            for word in voice.words:
                text = word.word
                if any(_matches_term(term, text) for term in terms) or any(
                    re.search(pattern, text, re.IGNORECASE | re.UNICODE)
                    for pattern in patterns
                ):
                    return word.start_seconds
    if matched:
        return None
    return None


class SfxStage(Stage):
    """Place rule matched sound effects on a timeline.

    ``options`` may carry ``rules`` (a list, mapping or YAML/JSON path),
    ``timeline``/``timeline_filename``, ``sfx_filename``, ``volume_db``,
    ``position``, ``duration_seconds``, ``prompt``, ``languages`` and
    ``replace``. A caller may inject an ``sfx`` callable or service instead of
    calling the configured provider.
    """

    name = SFX_STAGE_NAME
    artifacts: tuple[str, ...] = (
        DEFAULT_TIMELINE_FILENAME,
        DEFAULT_SFX_FILENAME,
    )

    def __init__(
        self,
        name: str | None = None,
        *,
        options: Mapping[str, Any] | None = None,
        rules: Any = None,
        sound_effects: Any = None,
        effects: Any = None,
        cues: Any = None,
        timeline: Any = None,
        timeline_filename: str | None = None,
        sfx_filename: str | None = None,
        manifest_filename: str | None = None,
        volume_db: float | None = None,
        position: str | None = None,
        duration_seconds: float | None = None,
        prompt: str | None = None,
        languages: Any = None,
        replace: bool | None = None,
        sfx: Any = None,
        generator: Any = None,
        provider: Any = None,
        **kwargs: Any,
    ) -> None:
        merged = dict(options or {})
        explicit = {
            "rules": rules,
            "sound_effects": sound_effects,
            "effects": effects,
            "cues": cues,
            "timeline": timeline,
            "timeline_filename": timeline_filename,
            "sfx_filename": sfx_filename,
            "manifest_filename": manifest_filename,
            "volume_db": volume_db,
            "position": position,
            "duration_seconds": duration_seconds,
            "prompt": prompt,
            "languages": languages,
            "replace": replace,
            "sfx": sfx,
            "generator": generator,
            "provider": provider,
        }
        for key, value in explicit.items():
            if value is not None:
                merged[key] = value
        merged.update(kwargs)
        timeline_name = str(
            merged.get("timeline_filename") or DEFAULT_TIMELINE_FILENAME
        )
        manifest_name = str(
            merged.get("sfx_filename")
            or merged.get("manifest_filename")
            or DEFAULT_SFX_FILENAME
        )
        super().__init__(
            name=name or SFX_STAGE_NAME,
            options=merged,
            artifacts=(timeline_name, manifest_name),
        )
        self.rules = next(
            (
                merged[key]
                for key in ("rules", "sound_effects", "effects", "cues")
                if merged.get(key) is not None
            ),
            None,
        )
        if self.rules is not None and not isinstance(self.rules, (str, Path)):
            self.rules = load_sfx_rules(self.rules)
        self.timeline = merged.get("timeline")
        self.timeline_filename = timeline_name
        self.sfx_filename = manifest_name
        self.volume_db = _finite(
            merged.get("volume_db", DEFAULT_SFX_VOLUME_DB),
            "volume_db",
        )
        self.position = _position(
            merged.get("position") or merged.get("placement") or merged.get("anchor")
        )
        self.duration_seconds = _optional_finite(
            merged.get("duration_seconds"),
            "duration_seconds",
            positive=True,
        )
        self.prompt = _text(merged.get("prompt"))
        self.languages = _languages(merged.get("languages") or merged.get("language"))
        self.replace = bool(merged.get("replace", True))
        self.sfx = (
            merged.get("sfx") or merged.get("generator") or merged.get("provider")
        )
        if self.timeline is None:
            self.inputs = (self.timeline_filename,)

    def _rules_source(self, context: StageContext, timeline: Timeline) -> Any:
        if self.rules is not None:
            return self.rules
        configured = _config_value(context, *_RULE_KEYS)
        if configured is not None:
            return configured
        for key in (*_RULE_KEYS, "sfx_rules"):
            raw = timeline.metadata.get(key)
            if raw is not None:
                return raw
        section = timeline.metadata.get("sfx")
        if isinstance(section, Mapping):
            for key in _RULE_KEYS:
                raw = section.get(key)
                if raw is not None:
                    return raw
        return None

    def _load_timeline(self, context: StageContext) -> Timeline:
        value = self.timeline
        if value is None:
            path = context.run_dir / self.timeline_filename
            if not path.is_file():
                raise ConfigError(
                    "sfx stage needs a timeline",
                    details={"timeline": self.timeline_filename},
                )
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ConfigError(
                    f"could not read the timeline: {path}",
                    details={"path": str(path), "error": str(exc)},
                ) from exc
            return Timeline.model_validate(payload)
        if isinstance(value, Timeline):
            return value.model_copy(deep=True)
        if isinstance(value, Mapping):
            return Timeline.model_validate(dict(value))
        if isinstance(value, (str, Path)):
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = context.run_dir / path
            if not path.is_file():
                raise ConfigError(
                    "sfx stage timeline file does not exist",
                    details={"path": str(path)},
                )
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ConfigError(
                    f"could not read the timeline: {path}",
                    details={"path": str(path), "error": str(exc)},
                ) from exc
            return Timeline.model_validate(payload)
        raise ConfigError(
            "timeline must be a Timeline, mapping or file path",
            details={"value": type(value).__name__},
        )

    def _languages_match(self, rule: SfxRule, timeline: Timeline) -> bool:
        if not rule.languages:
            return True
        available: list[str] = []
        for language in timeline.resolved_languages():
            code = language.language.casefold()
            if code and code not in available:
                available.append(code)
                base = code.split("-", 1)[0]
                if base and base not in available:
                    available.append(base)
        return any(code in available for code in rule.languages)

    def _targets(
        self,
        rule: SfxRule,
        timeline: Timeline,
    ) -> list[tuple[TimelineBeat | None, tuple[str, ...], float | None]]:
        beats = timeline.beats
        if not beats:
            if rule.beat or rule.index is not None:
                raise ConfigError(
                    f"sfx rule {rule.id!r} references a beat but the timeline is empty",
                    details={"rule": rule.id},
                )
            if rule.terms or rule.patterns:
                return []
            return [(None, (), None)]

        targets: list[tuple[TimelineBeat | None, tuple[str, ...], float | None]] = []
        for beat in beats:
            if not beat.enabled:
                continue
            if rule.index is not None and beat.index != rule.index:
                continue
            if rule.beat and beat.id != rule.beat:
                continue
            matched = _matched_terms(rule, beat.text)
            if matched is None:
                continue
            word_start = _word_start(rule, matched, timeline, beat)
            targets.append((beat, matched, word_start))

        if (rule.beat or rule.index is not None) and not targets:
            reference = rule.beat if rule.beat else str(rule.index)
            raise ConfigError(
                f"sfx rule {rule.id!r} references unknown beat {reference!r}",
                details={"rule": rule.id, "beat": reference},
            )
        return targets

    async def _call_sfx(
        self,
        request: SFXRequest,
        context: StageContext,
        *,
        run_id: str,
    ) -> Any:
        target = self.sfx
        if target is None:
            if context.router is None:
                raise ConfigError(
                    "sfx stage requires a sound effect provider or an injected "
                    "sfx service",
                    details={"stage": self.name},
                )
            return await context.media.sfx(
                request,
                run_id=run_id,
                operation=self.name,
            )
        method = target
        if not callable(target):
            for name in ("generate", "sfx", "generate_sfx"):
                candidate = getattr(target, name, None)
                if callable(candidate):
                    method = candidate
                    break
        if not callable(method):
            raise ConfigError(
                "sfx must be callable or expose a generate method",
                details={"value": type(target).__name__},
            )
        value = method(request)
        if inspect.isawaitable(value):
            value = await value
        return value

    def _requested_duration(
        self,
        rule: SfxRule,
        beat: TimelineBeat | None,
        beat_end: float,
    ) -> float | None:
        duration = rule.duration_seconds or self.duration_seconds
        if duration is None:
            return None
        if beat is not None:
            available = max(_EPSILON, beat_end - beat.start_seconds)
            duration = min(duration, available)
        return max(_MIN_DURATION, duration)

    async def _materialise(
        self,
        rule: SfxRule,
        beat: TimelineBeat | None,
        matched: tuple[str, ...],
        context: StageContext,
        *,
        run_id: str,
        identifier: str,
        beat_end: float,
    ) -> tuple[str, float | None, str, str, str, float, int]:
        if rule.source:
            duration = rule.duration_seconds or self.duration_seconds
            return (
                rule.source,
                duration,
                "asset",
                "",
                "",
                0.0,
                0,
            )

        prompt = rule.prompt or self.prompt
        if not prompt:
            raise ConfigError(
                f"sfx rule {rule.id!r} needs a prompt or a source",
                details={"rule": rule.id},
            )
        request = SFXRequest(
            prompt=prompt,
            duration_seconds=self._requested_duration(rule, beat, beat_end),
            loop=rule.loop,
            count=1,
            metadata={
                "stage": self.name,
                "rule": rule.id,
                "beat": beat.id if beat is not None else "",
                "matched_terms": list(matched),
                **dict(rule.metadata),
            },
        )
        result = await self._call_sfx(request, context, run_id=run_id)
        source = _path_of(result)
        if not source:
            payload = _data_of(result)
            if payload is None:
                raise ConfigError(
                    "sound effect result has no stored audio path or data"
                )
            suffix = _format_of(result)
            path = context.run_dir / "sfx" / f"{_slug(identifier)}.{suffix}"
            _write_bytes(path, payload)
            source = str(path)
        return (
            source,
            _duration_of(result),
            "provider",
            _provider_of(result),
            _model_of(result),
            _cost_of(result),
            1,
        )

    def _start_seconds(
        self,
        rule: SfxRule,
        position: str,
        beat: TimelineBeat | None,
        beat_start: float,
        beat_end: float,
        length: float,
        word_start: float | None,
        total: float,
    ) -> float:
        if position == "absolute":
            raw = rule.start_seconds or 0.0
        elif position == "offset":
            raw = beat_start + (rule.start_seconds or 0.0)
        elif position == "end":
            raw = beat_end - length
        elif position == "center":
            raw = beat_start + max(0.0, (beat_end - beat_start - length) / 2.0)
        elif position == "word":
            raw = word_start if word_start is not None else beat_start
        else:
            raw = beat_start
        return min(max(0.0, raw), total)

    async def run(self, context: StageContext) -> StageResult:
        timeline = self._load_timeline(context)
        original_duration = _timeline_duration(timeline)
        if original_duration <= _EPSILON:
            raise ConfigError(
                "sfx stage needs a timeline with a positive duration",
                details={"timeline": timeline.id},
            )
        pinned_duration = timeline.duration_seconds is None
        if pinned_duration:
            timeline.duration_seconds = original_duration

        source = self._rules_source(context, timeline)
        rules = load_sfx_rules(source)
        if self.languages:
            rules = tuple(
                rule
                for rule in rules
                if not rule.languages
                or any(code in self.languages for code in rule.languages)
            )

        if self.replace:
            timeline.music = [
                track
                for track in timeline.music
                if track.role not in (TimelineTrackRole.SFX, TimelineTrackRole.EFFECTS)
            ]
            for language in timeline.languages:
                language.music = [
                    track
                    for track in language.music
                    if track.role
                    not in (TimelineTrackRole.SFX, TimelineTrackRole.EFFECTS)
                ]

        used_ids = {track.id for track in timeline.music if track.id}
        run_id = context.run.id if context.run is not None else ""
        placements: list[SfxPlacement] = []
        tracks: list[TimelineMusic] = []
        calls = 0
        total_cost = 0.0
        provider = ""
        model = ""

        for rule_index, rule in enumerate(rules):
            if not rule.enabled:
                continue
            if not self._languages_match(rule, timeline):
                continue
            targets = self._targets(rule, timeline)
            if rule.once:
                targets = targets[:1]
            for target_index, (beat, matched, word_start) in enumerate(targets):
                if beat is None:
                    beat_start = 0.0
                    beat_end = original_duration
                    beat_id = ""
                    beat_position = 0
                else:
                    beat_position = beat.index if beat.index is not None else 0
                    beat_start = beat.start_seconds
                    beat_end = _beat_end(
                        beat,
                        beat_position,
                        timeline.beats,
                        original_duration,
                    )
                    beat_id = beat.id

                identifier = _slug(f"{rule.id}_{beat_id or target_index}_{rule_index}")
                (
                    effect_source,
                    measured_duration,
                    origin,
                    effect_provider,
                    effect_model,
                    cost,
                    call_count,
                ) = await self._materialise(
                    rule,
                    beat,
                    matched,
                    context,
                    run_id=run_id,
                    identifier=identifier,
                    beat_end=beat_end,
                )
                duration = (
                    measured_duration
                    or rule.duration_seconds
                    or self.duration_seconds
                    or DEFAULT_SFX_DURATION_SECONDS
                )
                length = max(_MIN_DURATION, min(float(duration), original_duration))
                start = self._start_seconds(
                    rule,
                    rule.position or self.position,
                    beat,
                    beat_start,
                    beat_end,
                    length,
                    word_start,
                    original_duration,
                )
                if start >= original_duration:
                    length = min(length, original_duration)
                    start = max(0.0, original_duration - length)
                else:
                    length = min(length, original_duration - start)
                length = max(_MIN_DURATION, length)
                end = min(original_duration, start + length)
                length = max(_MIN_DURATION, end - start)
                fade_in = min(rule.fade_in_seconds or 0.0, length)
                fade_out = min(rule.fade_out_seconds or 0.0, length)
                volume = (
                    self.volume_db if rule.volume_db is None else float(rule.volume_db)
                )

                track_id = identifier
                suffix = 2
                while track_id in used_ids:
                    track_id = f"{identifier}_{suffix}"
                    suffix += 1
                used_ids.add(track_id)

                placement = SfxPlacement(
                    id=track_id,
                    rule=rule.id,
                    beat=beat_id,
                    prompt=rule.prompt or self.prompt,
                    source=effect_source,
                    origin=origin,
                    start_seconds=start,
                    end_seconds=end,
                    duration_seconds=length,
                    volume_db=volume,
                    provider=effect_provider,
                    model=effect_model,
                    cost_usd=cost,
                    matched_terms=matched,
                    metadata=dict(rule.metadata),
                )
                track = TimelineMusic(
                    id=track_id,
                    source=effect_source,
                    role=TimelineTrackRole.SFX,
                    start_seconds=start,
                    end_seconds=end,
                    duration_seconds=length,
                    volume_db=volume,
                    fade_in_seconds=fade_in,
                    fade_out_seconds=fade_out,
                    loop=rule.loop,
                    duck=False,
                    metadata={
                        **placement.to_dict(),
                        "effect_id": track_id,
                        "rule_id": rule.id,
                    },
                )
                placements.append(placement)
                tracks.append(track)
                timeline.music.append(track)
                calls += call_count
                total_cost += cost
                if effect_provider:
                    provider = effect_provider
                if effect_model:
                    model = effect_model

        new_duration = _timeline_duration(timeline)
        duration_unchanged = abs(new_duration - original_duration) <= _EPSILON
        summary = {
            "version": 1,
            "stage": self.name,
            "timeline": self.timeline_filename,
            "rules": len(rules),
            "placements": len(placements),
            "tracks": [placement.to_dict() for placement in placements],
            "effects": [placement.to_dict() for placement in placements],
            "calls": calls,
            "cost_usd": total_cost,
            "provider": provider,
            "model": model,
            "duration_seconds": original_duration,
            "total_duration_seconds": original_duration,
            "duration_unchanged": duration_unchanged,
        }
        timeline.metadata["sfx"] = {
            "rules": len(rules),
            "placements": len(placements),
            "cost_usd": total_cost,
            "duration_seconds": original_duration,
            "duration_unchanged": duration_unchanged,
        }
        _write_json(
            context.run_dir / self.timeline_filename,
            timeline.model_dump(mode="json"),
        )
        _write_json(context.run_dir / self.sfx_filename, summary)
        self.artifacts = (self.timeline_filename, self.sfx_filename)

        metadata = {
            "rules": len(rules),
            "placements": len(placements),
            "tracks": [placement.to_dict() for placement in placements],
            "effects": [placement.to_dict() for placement in placements],
            "calls": calls,
            "cost_usd": total_cost,
            "provider": provider,
            "model": model,
            "duration_seconds": original_duration,
            "total_duration_seconds": original_duration,
            "duration_unchanged": duration_unchanged,
            "pinned_duration": pinned_duration,
            "timeline": self.timeline_filename,
            "manifest": self.sfx_filename,
        }
        return StageResult(
            run_id=run_id,
            stage=self.name,
            status=StageState.SUCCEEDED,
            artifacts=[self.timeline_filename, self.sfx_filename],
            cost_usd=total_cost,
            metadata=metadata,
        )


SFXStage = SfxStage
"""Alias for :class:`SfxStage`."""

SoundEffectStage = SfxStage
"""Alias for :class:`SfxStage`."""

SoundEffectsStage = SfxStage
"""Alias for :class:`SfxStage`."""

sfx_stage = SfxStage
"""Alias for :class:`SfxStage`."""

SfxStageResult = StageResult
SFXStageResult = StageResult
"""Aliases for the stage result returned by :class:`SfxStage`."""


def build_sfx_stage(
    *,
    options: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> SfxStage:
    """Build a :class:`SfxStage` from recipe options and overrides."""
    return SfxStage(options=options, **kwargs)


async def run_sfx_stage(
    context: StageContext,
    *,
    stage: SfxStage | None = None,
    **options: Any,
) -> StageResult:
    """Build a stage unless one is supplied, then run it asynchronously."""
    if stage is not None and options:
        raise ConfigError("pass either a stage or options, not both")
    active = stage if stage is not None else SfxStage(**options)
    return await active.run(context)


def place_sound_effects(
    timeline: Timeline | Mapping[str, Any],
    placements: Iterable[SfxPlacement | Mapping[str, Any]],
    *,
    replace: bool = True,
) -> list[TimelineMusic]:
    """Add already materialised effects to *timeline* without extending it.

    This is the deterministic part of the stage exposed for callers that have
    audio files already. It never calls a provider and never changes the
    timeline's total duration.
    """
    value = (
        timeline
        if isinstance(timeline, Timeline)
        else Timeline.model_validate(dict(timeline))
    )
    total = _timeline_duration(value)
    if total <= _EPSILON:
        raise ConfigError("sfx placement needs a timeline with a positive duration")
    if value.duration_seconds is None:
        value.duration_seconds = total
    if replace:
        value.music = [
            track
            for track in value.music
            if track.role not in (TimelineTrackRole.SFX, TimelineTrackRole.EFFECTS)
        ]
    tracks: list[TimelineMusic] = []
    for raw in placements:
        placement = (
            raw
            if isinstance(raw, SfxPlacement)
            else SfxPlacement(
                id=str(raw.get("id") or "sfx"),
                rule=str(raw.get("rule") or ""),
                source=str(raw.get("source") or ""),
                start_seconds=float(raw.get("start_seconds") or 0.0),
                end_seconds=float(raw.get("end_seconds") or 0.0),
                duration_seconds=float(raw.get("duration_seconds") or 0.0),
                volume_db=float(
                    raw.get("volume_db", DEFAULT_SFX_VOLUME_DB) or DEFAULT_SFX_VOLUME_DB
                ),
                origin=str(raw.get("origin") or "asset"),
                beat=str(raw.get("beat") or ""),
                prompt=str(raw.get("prompt") or ""),
                matched_terms=tuple(raw.get("matched_terms") or ()),
                metadata=dict(raw.get("metadata") or {}),
            )
        )
        length = max(
            _MIN_DURATION,
            min(placement.duration_seconds, total),
        )
        start = min(max(0.0, placement.start_seconds), total)
        if start >= total:
            length = min(length, total)
            start = max(0.0, total - length)
        else:
            length = min(length, total - start)
        length = max(_MIN_DURATION, length)
        end = min(total, start + length)
        length = max(_MIN_DURATION, end - start)
        track = TimelineMusic(
            id=placement.id,
            source=placement.source,
            role=TimelineTrackRole.SFX,
            start_seconds=start,
            end_seconds=end,
            duration_seconds=length,
            volume_db=placement.volume_db,
            duck=False,
            metadata=placement.to_dict(),
        )
        value.music.append(track)
        tracks.append(track)
    return tracks
