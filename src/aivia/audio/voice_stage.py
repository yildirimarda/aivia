"""The voice stage: scripts in, measured speech and a timeline out.

The stage reads one script per language and synthesizes speech through the
single screened, cached and budgeted speech path. A script may be voiced one
beat at a time or as one whole script:

* ``beat`` mode makes one speech request per non-empty beat and leaves a
  configurable gap of silence between voiced segments. When a beat carries
  dialogue, every line is voiced separately so a named speaker can use a
  distinct voice while the narration keeps the language's narrator voice.
* ``script`` mode makes one request per language for the whole script.

Either way the timeline records the *measured* duration returned by the speech
service, not an estimate from the word count. Beat timings, speaker names,
voice starts, gaps and word timings are written to a
:class:`~aivia.compose.timeline.Timeline` JSON file, which is the hand-off the
subtitle and render stages consume.

An injected speech callable or service keeps the stage offline in tests. The
default path calls ``context.media.speech`` so the call still goes through the
router, cache, budget, cost ledger and sidecars.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from aivia.audio.qa import AudioQAReport, check_audio_quality
from aivia.compose.timeline import Timeline
from aivia.core.config.media import DEFAULT_MEDIA_VOICE_GAP_SECONDS
from aivia.core.errors import ConfigError
from aivia.core.models.base import canonical_json
from aivia.core.models.run import StageResult, StageState
from aivia.core.models.script import Beat, DialogueLine, Script
from aivia.core.stages.base import Stage, StageContext, _normalise_inputs
from aivia.providers.protocols.audio import AudioFormat
from aivia.providers.protocols.tts import TTSRequest

__all__ = [
    "DEFAULT_GAP_SECONDS",
    "DEFAULT_SCRIPT_FILENAME_PATTERN",
    "DEFAULT_TIMELINE_FILENAME",
    "SYNTHESIS_MODES",
    "VOICE_STAGE_NAME",
    "Voice",
    "VoiceStage",
    "VoiceStageResult",
    "build_voice_stage",
    "run_voice_stage",
    "voice_stage",
]

VOICE_STAGE_NAME = "voice"
"""The recipe stage name this implementation serves."""

DEFAULT_TIMELINE_FILENAME = "timeline.json"
"""The timeline file the stage writes under the run directory."""

DEFAULT_SCRIPT_FILENAME_PATTERN = "script.{language}.json"
"""The per-language script file name pattern the stage reads."""

DEFAULT_GAP_SECONDS = DEFAULT_MEDIA_VOICE_GAP_SECONDS
"""The silence inserted between beats unless a caller configures one."""

SYNTHESIS_MODES = ("beat", "script")
"""The supported synthesis modes."""

_BEAT_ALIASES = frozenset(
    {"beat", "beats", "per-beat", "per_beat", "perbeat", "segment", "segments"}
)
_SCRIPT_ALIASES = frozenset(
    {"script", "scripts", "whole", "whole-script", "whole_script", "full", "single"}
)
_NARRATOR_SPEAKERS = frozenset(
    {"narrator", "narration", "voiceover", "voice-over", "voice_over"}
)
_SPEAKER_VOICE_KEYS = (
    "speaker_voices",
    "speaker_voice",
    "voices_by_speaker",
    "voices_by_role",
    "role_voices",
    "character_voices",
    "character_voice",
    "speaker_map",
    "character_map",
    "voice_map",
)
_SPEAKER_SECTION_KEYS = ("speakers", "characters", "roles", "voices")
_VOICE_VALUE_KEYS = ("voice", "voice_id", "id", "name")


def _finite(value: Any, name: str, *, non_negative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number", details={"value": repr(value)})
    number = float(value)
    if not math.isfinite(number):
        raise ConfigError(f"{name} must be finite", details={"value": repr(value)})
    if non_negative and number < 0.0:
        raise ConfigError(
            f"{name} must not be negative", details={"value": repr(value)}
        )
    return number


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _language_key(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ConfigError("language must be a string", details={"value": repr(value)})
    return value.strip().replace("_", "-").casefold()


def _base_language(value: str) -> str:
    return value.split("-", 1)[0]


def _speaker_key(value: Any) -> str:
    return _text(value).casefold()


def _comparison_text(value: Any) -> str:
    return " ".join(_text(value).split()).casefold()


def _voice_value(value: Any) -> str:
    if isinstance(value, Mapping):
        for name in _VOICE_VALUE_KEYS:
            text = _text(value.get(name))
            if text:
                return text
        return ""
    return _text(value)


def _lookup_voice(mapping: Any, key: str) -> str:
    if not key or not isinstance(mapping, Mapping):
        return ""
    text = _voice_value(mapping.get(key))
    if text:
        return text
    folded = key.casefold()
    for raw_key, raw_value in mapping.items():
        if isinstance(raw_key, str) and raw_key.strip().casefold() == folded:
            text = _voice_value(raw_value)
            if text:
                return text
    return ""


def _nested_mapping(value: Any, keys: Iterable[str]) -> Mapping[Any, Any]:
    if not isinstance(value, Mapping):
        return {}
    for key in keys:
        nested = value.get(key)
        if isinstance(nested, Mapping):
            return nested
    return {}


def _section(config: Any, name: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name, None)


def _normalise_languages(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw: Iterable[Any] = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        raw = value
    else:
        raise ConfigError(
            "languages must be a language code or a list of language codes",
            details={"value": repr(value)},
        )
    codes: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            raise ConfigError(
                "language codes must be strings",
                details={"value": repr(item)},
            )
        for part in item.split(","):
            code = _language_key(part)
            if code and code not in codes:
                codes.append(code)
    return codes


def _configured_languages(config: Any) -> list[str]:
    section = _section(config, "languages")
    if section is None:
        return []
    raw = getattr(section, "languages", None)
    if raw is None and isinstance(section, Mapping):
        raw = section.get("languages", section)
    if not isinstance(raw, Mapping):
        return []
    codes: list[str] = []
    for key in raw:
        code = _language_key(key)
        if code and code not in {"default", "schema_version"} and code not in codes:
            codes.append(code)
    return codes


def _default_language(config: Any) -> str:
    section = _section(config, "languages")
    raw = getattr(section, "default", None)
    if raw is None and isinstance(section, Mapping):
        raw = section.get("default")
    return _language_key(raw) if isinstance(raw, str) and raw.strip() else "en"


def _mode(value: Any) -> str:
    text = _text(value).casefold().replace(" ", "-")
    if not text:
        return "beat"
    if text in _BEAT_ALIASES:
        return "beat"
    if text in _SCRIPT_ALIASES:
        return "script"
    raise ConfigError(
        f"unknown voice synthesis mode: {value!r}",
        details={"modes": list(SYNTHESIS_MODES)},
    )


def _script_value(value: Any, language: str, mode: str) -> Script:
    if isinstance(value, Script):
        return value
    if isinstance(value, str):
        return Script.model_validate(
            {"mode": mode, "language": language, "hook": value.strip()}
        )
    if isinstance(value, Mapping):
        payload = dict(value)
        payload.setdefault("language", language)
        payload.setdefault("mode", mode)
        return Script.model_validate(payload)
    raise ConfigError(
        "script must be a Script, mapping or string",
        details={"language": language, "value": type(value).__name__},
    )


def _mapping_scripts(value: Any) -> Mapping[Any, Any]:
    if hasattr(value, "scripts"):
        value = value.scripts
    if isinstance(value, Mapping) and isinstance(value.get("scripts"), Mapping):
        value = value["scripts"]
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "items"):
        return cast(Mapping[Any, Any], value.items())
    raise ConfigError(
        "scripts must be a mapping of language code to script",
        details={"value": type(value).__name__},
    )


def _beat_text(beat: Beat) -> str:
    text = _text(beat.text)
    if text:
        return text
    parts = [_text(line.text) for line in beat.dialogue if _text(line.text)]
    return " ".join(parts)


def _dialogue_units(
    lines: Iterable[DialogueLine],
    *,
    fallback_speaker: str = "narrator",
) -> list[tuple[str, str, str, DialogueLine | None]]:
    units: list[tuple[str, str, str, DialogueLine | None]] = []
    for line in lines:
        text = _text(line.text)
        if not text:
            continue
        speaker = _text(line.speaker) or fallback_speaker
        units.append((text, speaker, _text(line.voice), line))
    return units


def _beat_units(beat: Beat) -> list[tuple[str, str, str, DialogueLine | None]]:
    """Return the spoken units of *beat* in script order.

    A beat that repeats its narration as a narrator dialogue line is voiced
    once, from the dialogue, so writing modes that populate both fields do not
    produce duplicate audio.
    """
    dialogue = _dialogue_units(beat.dialogue)
    narration = _text(beat.text)
    if not narration:
        return dialogue
    if not dialogue:
        return [(narration, "narrator", "", None)]
    target = _comparison_text(narration)
    joined = _comparison_text(
        " ".join(text for text, _speaker, _voice, _line in dialogue)
    )
    if target == joined or any(
        _comparison_text(text) == target for text, _speaker, _voice, _line in dialogue
    ):
        return dialogue
    return [(narration, "narrator", "", None), *dialogue]


def _full_script_text(script: Script) -> str:
    parts: list[str] = []
    hook = _text(script.hook)
    if hook:
        parts.append(hook)
    for beat in script.beats:
        text = _beat_text(beat)
        if text:
            parts.append(text)
    outro = _text(script.outro)
    if outro:
        parts.append(outro)
    if not parts:
        for line in script.dialogue_lines:
            text = _text(line.text)
            if text:
                parts.append(text)
    return "\n\n".join(parts)


def _result_sources(value: Any) -> tuple[Any, ...]:
    return (value, getattr(value, "outcome", None), getattr(value, "response", None))


def _duration_of(value: Any) -> float | None:
    for source in _result_sources(value):
        if source is None:
            continue
        for name in ("duration_seconds", "duration"):
            raw = getattr(source, name, None)
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                number = float(raw)
                if math.isfinite(number) and number >= 0.0:
                    return number
    if isinstance(value, Mapping):
        for name in ("duration_seconds", "duration"):
            raw = value.get(name)
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                number = float(raw)
                if math.isfinite(number) and number >= 0.0:
                    return number
    return None


def _path_of(value: Any) -> str:
    if isinstance(value, (str, Path)):
        return str(value)
    if isinstance(value, Mapping):
        for name in ("path", "source", "url", "file"):
            raw = value.get(name)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
        paths = value.get("paths")
        if isinstance(paths, Sequence) and not isinstance(paths, (str, bytes)):
            for item in paths:
                if isinstance(item, str) and item.strip():
                    return item.strip()
        return ""
    for source in (*_result_sources(value), getattr(value, "asset", None)):
        if source is None:
            continue
        for name in ("path", "source", "url", "file"):
            raw = getattr(source, name, None)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
        paths = getattr(source, "paths", None)
        if isinstance(paths, Sequence) and not isinstance(paths, (str, bytes)):
            for item in paths:
                if isinstance(item, str) and item.strip():
                    return item.strip()
    return ""


def _cost_of(value: Any) -> float:
    if isinstance(value, Mapping):
        for name in ("cost_usd", "amount_usd", "total_cost_usd"):
            raw = value.get(name)
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                return max(0.0, float(raw))
        return 0.0
    for name in ("cost_usd", "amount_usd", "total_cost_usd"):
        raw = getattr(value, name, None)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return max(0.0, float(raw))
    return 0.0


def _provider_of(value: Any) -> str:
    for source in _result_sources(value):
        if source is None:
            continue
        raw = getattr(source, "provider", None)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return ""


def _model_of(value: Any) -> str:
    for source in _result_sources(value):
        if source is None:
            continue
        raw = getattr(source, "model", None)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return ""


def _word_records(value: Any, offset: float) -> list[dict[str, Any]]:
    candidates: Any = None
    for source in _result_sources(value):
        if source is None:
            continue
        for name in ("word_timestamps", "words", "timed_words"):
            raw = getattr(source, name, None)
            if raw:
                candidates = raw
                break
        if candidates:
            break
    if candidates is None and isinstance(value, Mapping):
        for name in ("word_timestamps", "words", "timed_words"):
            raw = value.get(name)
            if raw:
                candidates = raw
                break
    if not candidates:
        return []
    records: list[dict[str, Any]] = []
    for item in candidates:
        if isinstance(item, Mapping):
            word = _text(item.get("word") or item.get("text"))
            start = item.get("start_seconds", item.get("start", 0.0))
            end = item.get("end_seconds", item.get("end"))
            confidence = item.get("confidence")
        else:
            word = _text(getattr(item, "word", None) or getattr(item, "text", None))
            start = getattr(item, "start_seconds", getattr(item, "start", 0.0))
            end = getattr(item, "end_seconds", getattr(item, "end", None))
            confidence = getattr(item, "confidence", None)
        if not word or not isinstance(start, (int, float)):
            continue
        record: dict[str, Any] = {
            "word": word,
            "start_seconds": float(start) + offset,
        }
        if isinstance(end, (int, float)):
            record["end_seconds"] = float(end) + offset
        if isinstance(confidence, (int, float)):
            record["confidence"] = float(confidence)
        records.append(record)
    return records


def _script_glob(pattern: str) -> str:
    return pattern.replace("{language}", "*").replace("{code}", "*")


def _script_input_declarations(
    languages: Iterable[str],
    pattern: str,
) -> tuple[str, ...]:
    """Return the per-language script files the stage reads as inputs.

    Relative names are resolved by the resume hash against the run directory,
    so editing a script between runs changes the stage input hash. A custom
    ``script_filename_pattern`` is honoured just as it is when loading.
    """
    declared: list[str] = []
    for language in languages:
        relative = pattern.format(language=language, code=language)
        if relative and relative not in declared:
            declared.append(relative)
    return tuple(declared)


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


_QUALITY_OPTION_KEYS = ("audio_qa", "audio_quality", "qa", "quality")
_QUALITY_THRESHOLD_KEYS = frozenset(
    {
        "expected_words_per_second",
        "max_internal_silence_seconds",
        "max_leading_silence_seconds",
        "max_mispronunciation_ratio",
        "max_silence_ratio",
        "max_trailing_silence_seconds",
        "min_duration_ratio",
        "min_word_confidence",
        "silent_ratio",
    }
)
_QUALITY_EXTRA_KEYS = ("transcripts", "silences", "pronunciations")


def _quality_options(
    options: Mapping[str, Any],
) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    """Resolve the quality gate toggle, thresholds and extra check inputs."""
    raw: Any = True
    for key in _QUALITY_OPTION_KEYS:
        if key in options:
            raw = options[key]
            break
    enabled = True
    values: dict[str, Any] = {}
    if isinstance(raw, Mapping):
        enabled = bool(raw.get("enabled", True))
        values.update(raw)
    elif raw is not None:
        enabled = bool(raw)

    extras: dict[str, Any] = {}
    for key in _QUALITY_EXTRA_KEYS:
        if key in options:
            extras[key] = options[key]
        elif key in values:
            extras[key] = values.pop(key)
    thresholds = {
        key: options[key] for key in _QUALITY_THRESHOLD_KEYS if key in options
    }
    thresholds.update(
        {key: value for key, value in values.items() if key in _QUALITY_THRESHOLD_KEYS}
    )
    return enabled, thresholds, extras


class VoiceStageResult(StageResult):
    """A stage result whose metadata carries per-language durations."""

    @property
    def durations(self) -> dict[str, float]:
        raw = self.metadata.get("durations")
        if not isinstance(raw, Mapping):
            return {}
        result: dict[str, float] = {}
        for language, value in raw.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                result[str(language)] = float(value)
        return result

    @property
    def timeline(self) -> str:
        raw = self.metadata.get("timeline")
        return raw if isinstance(raw, str) else ""

    @property
    def quality(self) -> dict[str, Any]:
        """The serialized audio quality report, empty when checks were off."""
        raw = self.metadata.get("quality")
        return dict(raw) if isinstance(raw, Mapping) else {}

    @property
    def quality_ok(self) -> bool:
        """Whether the audio quality pass found no errors."""
        return bool(self.metadata.get("quality_ok", True))

    @property
    def quality_findings(self) -> list[dict[str, Any]]:
        """The serialized quality findings in report order."""
        raw = self.metadata.get("quality_findings")
        if not isinstance(raw, list):
            return []
        return [dict(finding) for finding in raw if isinstance(finding, Mapping)]


class VoiceStage(Stage):
    """Synthesize a script per language and write a measured timeline.

    ``options`` may carry ``languages``/``language``, ``mode`` (``beat`` or
    ``script``), ``gap_seconds``, ``voice``/``voices``, ``speaker_voices``
    (a speaker or character name to voice mapping), ``model``,
    ``instructions``, ``speed``, ``format``, ``timeline_filename``,
    ``script_filename_pattern`` and ``word_timestamps``. A caller may inject
    ``scripts``, a single ``script`` or a ``speech`` callable/service instead
    of reading files and calling the configured provider.
    """

    name = VOICE_STAGE_NAME
    artifacts: tuple[str, ...] = (DEFAULT_TIMELINE_FILENAME,)

    def __init__(
        self,
        name: str | None = None,
        *,
        options: Mapping[str, Any] | None = None,
        scripts: Any = None,
        script: Any = None,
        speech: Any = None,
        synthesizer: Any = None,
        languages: Any = None,
        mode: str | None = None,
        gap_seconds: float | None = None,
        voice: str | None = None,
        voices: Mapping[str, Any] | None = None,
        speaker_voices: Mapping[str, Any] | None = None,
        character_voices: Mapping[str, Any] | None = None,
        timeline_filename: str | None = None,
        script_filename_pattern: str | None = None,
        **kwargs: Any,
    ) -> None:
        merged = dict(options or {})
        if scripts is not None:
            merged["scripts"] = scripts
        if script is not None:
            merged["script"] = script
        if speech is not None:
            merged["speech"] = speech
        if synthesizer is not None:
            merged["synthesizer"] = synthesizer
        if languages is not None:
            merged["languages"] = languages
        if mode is not None:
            merged["mode"] = mode
        if gap_seconds is not None:
            merged["gap_seconds"] = gap_seconds
        if voice is not None:
            merged["voice"] = voice
        if voices is not None:
            merged["voices"] = voices
        if speaker_voices is not None:
            merged["speaker_voices"] = speaker_voices
        if character_voices is not None:
            merged["character_voices"] = character_voices
        if timeline_filename is not None:
            merged["timeline_filename"] = timeline_filename
        if script_filename_pattern is not None:
            merged["script_filename_pattern"] = script_filename_pattern
        merged.update(kwargs)
        super().__init__(
            name=name or VOICE_STAGE_NAME,
            options=merged,
            artifacts=(merged.get("timeline_filename") or DEFAULT_TIMELINE_FILENAME,),
            inputs=(),
        )
        self.scripts = merged.get("scripts")
        self.script = merged.get("script")
        self.speech = merged.get("speech") or merged.get("synthesizer")
        self.languages = merged.get("languages")
        self.mode = _mode(merged.get("mode"))
        self.gap_seconds = _finite(
            merged.get("gap_seconds", merged.get("gap", DEFAULT_GAP_SECONDS)),
            "gap_seconds",
            non_negative=True,
        )
        self.voice = _text(
            merged.get("voice")
            or merged.get("narrator_voice")
            or merged.get("narration_voice")
            or merged.get("default_voice")
        )
        self.voices = merged.get("voices")
        self.speaker_voices = next(
            (
                merged[name]
                for name in _SPEAKER_VOICE_KEYS
                if isinstance(merged.get(name), Mapping)
            ),
            None,
        )
        if self.speaker_voices is None:
            self.speaker_voices = merged.get("speakers") or merged.get("characters")
        self.timeline_filename = str(
            merged.get("timeline_filename") or DEFAULT_TIMELINE_FILENAME
        )
        self.script_filename_pattern = str(
            merged.get("script_filename_pattern") or DEFAULT_SCRIPT_FILENAME_PATTERN
        )
        (
            self.quality_enabled,
            self.quality_thresholds,
            self.quality_extras,
        ) = _quality_options(merged)
        self.scripts_by_language: dict[str, Script] = {}
        self._declared_input_override: tuple[str, ...] | None = None

    def _explicit_languages(self) -> list[str]:
        explicit = self.languages
        if explicit is None:
            explicit = self.options.get("languages")
        if explicit is None:
            explicit = self.options.get("language")
        return _normalise_languages(explicit)

    def _discover_script_languages(self, run_dir: Path) -> list[str]:
        discovered: list[str] = []
        if not run_dir.is_dir():
            return discovered
        pattern = _script_glob(self.script_filename_pattern)
        for path in sorted(run_dir.glob(pattern)):
            name = path.name
            prefix, _, suffix = self.script_filename_pattern.partition("{")
            if not name.startswith(prefix):
                continue
            remainder = name[len(prefix) :]
            if suffix:
                _token, _, tail = suffix.partition("}")
                if tail and remainder.endswith(tail):
                    remainder = remainder[: -len(tail)]
            code = _language_key(remainder)
            if code and code not in discovered:
                discovered.append(code)
        return discovered

    def _resolved_languages(
        self,
        run_dir: Path | None,
        config: Any,
    ) -> list[str]:
        codes = self._explicit_languages()
        if codes:
            return codes
        if isinstance(self.scripts, Mapping):
            return [_language_key(key) for key in self.scripts if _language_key(key)]
        if run_dir is not None:
            discovered = self._discover_script_languages(run_dir)
            if discovered:
                return discovered
        configured = _configured_languages(config)
        if configured:
            return configured
        return [_default_language(config)]

    @property
    def inputs(self) -> tuple[str, ...]:
        """The per-language script files this stage reads from disk.

        Computed from the current languages so a language filter applied after
        construction still declares exactly the scripts that will be read. An
        injected script or scripts mapping reads no files and declares none.
        """
        override = self._declared_input_override
        if override is not None:
            return override
        if self.scripts is not None or self.script is not None:
            return ()
        return _script_input_declarations(
            self._explicit_languages(),
            self.script_filename_pattern,
        )

    @inputs.setter
    def inputs(self, value: Iterable[str | Path] | str | Path) -> None:
        self._declared_input_override = _normalise_inputs(value)

    def resolve_inputs(
        self,
        *,
        base_dir: str | Path | None = None,
        config: Any = None,
    ) -> tuple[str, ...]:
        """Resolve the script files this run will read, including discovery."""
        if self.scripts is not None or self.script is not None:
            return ()
        run_dir = Path(base_dir).expanduser() if base_dir is not None else None
        return _script_input_declarations(
            self._resolved_languages(run_dir, config),
            self.script_filename_pattern,
        )

    def _target_languages(self, context: StageContext) -> list[str]:
        return self._resolved_languages(context.run_dir, context.config)

    def _load_scripts(self, context: StageContext) -> dict[str, Script]:
        languages = self._target_languages(context)
        if self.scripts is not None:
            values = _mapping_scripts(self.scripts)
            scripts: dict[str, Script] = {}
            for key, value in values.items():
                code = _language_key(key)
                if code in languages:
                    scripts[code] = _script_value(value, code, self.mode)
            missing_scripts = [code for code in languages if code not in scripts]
            if missing_scripts:
                raise ConfigError(
                    "scripts are missing for some languages",
                    details={
                        "languages": languages,
                        "missing": missing_scripts,
                    },
                )
            return scripts
        if self.script is not None:
            code = _language_key(getattr(self.script, "language", "")) or languages[0]
            if code not in languages:
                languages = [code]
            return {code: _script_value(self.script, code, self.mode)}

        scripts = {}
        missing_files: list[str] = []
        for code in languages:
            relative = self.script_filename_pattern.format(language=code, code=code)
            path = context.run_dir / relative
            if not path.is_file():
                missing_files.append(relative)
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ConfigError(
                    f"invalid script JSON: {path}",
                    details={"path": str(path)},
                ) from exc
            scripts[code] = _script_value(payload, code, self.mode)
        if missing_files:
            raise ConfigError(
                "voice stage needs a script for every language",
                details={
                    "missing": missing_files,
                    "run_dir": str(context.run_dir),
                },
            )
        return scripts

    def _speaker_voice_for(self, speaker: str, config: Any) -> str:
        key = _speaker_key(speaker)
        if not key:
            return ""
        mappings: list[Any] = [self.speaker_voices]
        if isinstance(self.voices, Mapping):
            mappings.append(self.voices)
        configured = _section(config, "voices")
        if isinstance(configured, Mapping):
            mappings.append(_nested_mapping(configured, _SPEAKER_SECTION_KEYS))
            mappings.append(configured)
        for name in _SPEAKER_VOICE_KEYS:
            mappings.append(_section(config, name))
        for mapping in mappings:
            text = _lookup_voice(mapping, key)
            if text:
                return text
        return ""

    def _voice_for(self, language: str, config: Any, *, speaker: str = "") -> str:
        speaker_voice = self._speaker_voice_for(speaker, config)
        if speaker_voice:
            return speaker_voice
        key = _speaker_key(speaker)
        if key and key not in _NARRATOR_SPEAKERS:
            return _text(speaker)
        mapping = self.voices
        for name in (language, _base_language(language), "default"):
            text = _lookup_voice(mapping, name)
            if text:
                return text
        configured = _section(config, "voices")
        for name in (language, _base_language(language), "default"):
            text = _lookup_voice(configured, name)
            if text:
                return text
        if self.voice:
            return self.voice
        return "default"

    def _request(
        self,
        text: str,
        language: str,
        *,
        beat: str,
        config: Any,
        speaker: str = "",
        voice: str = "",
    ) -> TTSRequest:
        options = self.options
        resolved = _text(voice) or self._voice_for(language, config, speaker=speaker)
        values: dict[str, Any] = {
            "text": text,
            "voice": resolved,
            "language": language,
            "word_timestamps": bool(options.get("word_timestamps", True)),
            "metadata": {
                "stage": self.name,
                "language": language,
                "beat": beat,
                "speaker": _text(speaker),
            },
        }
        for key in ("model", "instructions", "speed", "sample_rate"):
            value = options.get(key)
            if value is not None:
                values[key] = value
        fmt = options.get("format")
        if fmt is not None:
            if isinstance(fmt, AudioFormat):
                values["format"] = fmt
            else:
                try:
                    values["format"] = AudioFormat(str(fmt).strip().lower())
                except ValueError as exc:
                    raise ConfigError(
                        f"unknown audio format: {fmt!r}",
                        details={"formats": [item.value for item in AudioFormat]},
                    ) from exc
        return TTSRequest(**values)

    async def _speak(
        self,
        request: TTSRequest,
        context: StageContext,
        *,
        run_id: str,
    ) -> Any:
        target = self.speech
        if target is None:
            if context.router is None:
                raise ConfigError(
                    "voice stage requires a speech provider or an injected "
                    "speech service",
                    details={"stage": self.name},
                )
            return await context.media.speech(
                request,
                run_id=run_id,
                operation=self.name,
            )
        method = target
        if not callable(target):
            for name in ("synthesize", "generate", "speech", "generate_speech"):
                candidate = getattr(target, name, None)
                if callable(candidate):
                    method = candidate
                    break
        if not callable(method):
            raise ConfigError(
                "speech must be callable or expose a synthesize method",
                details={"value": type(target).__name__},
            )
        value = method(request)
        if inspect.isawaitable(value):
            value = await value
        return value

    async def _synthesize_unit(
        self,
        text: str,
        language: str,
        context: StageContext,
        *,
        run_id: str,
        beat_id: str,
        speaker: str,
        voice: str,
        start: float,
        unit_id: str,
        line: DialogueLine | None = None,
    ) -> tuple[dict[str, Any], float, float, str, str]:
        resolved = _text(voice) or self._voice_for(
            language, context.config, speaker=speaker
        )
        result = await self._speak(
            self._request(
                text,
                language,
                beat=beat_id,
                config=context.config,
                speaker=speaker,
                voice=resolved,
            ),
            context,
            run_id=run_id,
        )
        measured = _duration_of(result)
        if measured is None:
            raise ConfigError(
                "speech result has no measured duration",
                details={"language": language, "beat": beat_id, "speaker": speaker},
            )
        source = _path_of(result)
        if not source:
            raise ConfigError(
                "speech result has no stored audio path",
                details={"language": language, "beat": beat_id, "speaker": speaker},
            )
        duration = measured
        if line is not None:
            if line.duration_seconds is not None:
                duration = float(line.duration_seconds)
            elif line.end_seconds is not None and line.end_seconds >= start:
                duration = float(line.end_seconds) - start
        end = start + duration
        record = {
            "id": unit_id,
            "beat": beat_id,
            "source": source,
            "start_seconds": start,
            "end_seconds": end,
            "language": language,
            "speaker": speaker,
            "voice": resolved,
            "words": _word_records(result, start),
        }
        return (
            record,
            end,
            _cost_of(result),
            _provider_of(result),
            _model_of(result),
        )

    async def _synthesize_language(
        self,
        language: str,
        script: Script,
        context: StageContext,
        *,
        run_id: str,
    ) -> tuple[list[dict[str, Any]], list[float], float, float, str, str]:
        voices: list[dict[str, Any]] = []
        beat_starts: list[float] = []
        cursor = 0.0
        total_cost = 0.0
        provider = ""
        model = ""
        if self.mode == "script":
            text = _full_script_text(script)
            if not text:
                raise ConfigError(
                    "script has no spoken text",
                    details={"language": language},
                )
            result = await self._speak(
                self._request(text, language, beat="", config=context.config),
                context,
                run_id=run_id,
            )
            duration = _duration_of(result)
            if duration is None:
                raise ConfigError(
                    "speech result has no measured duration",
                    details={"language": language},
                )
            source = _path_of(result)
            if not source:
                raise ConfigError(
                    "speech result has no stored audio path",
                    details={"language": language},
                )
            voices.append(
                {
                    "id": f"{language}_script",
                    "source": source,
                    "start_seconds": 0.0,
                    "end_seconds": duration,
                    "language": language,
                    "speaker": "narrator",
                    "words": _word_records(result, 0.0),
                }
            )
            total_cost += _cost_of(result)
            provider = provider or _provider_of(result)
            model = model or _model_of(result)
            beat_starts = self._script_beat_starts(script, duration)
            return voices, beat_starts, duration, total_cost, provider, model

        starts_by_beat: dict[str, float] = {}
        ends_by_beat: dict[str, float] = {}
        for unit_position, (text, speaker, unit_voice, line) in enumerate(
            _dialogue_units(script.dialogue_lines)
        ):
            start = cursor
            if line is not None and line.start_seconds is not None:
                start = float(line.start_seconds)
            (
                record,
                end,
                cost,
                unit_provider,
                unit_model,
            ) = await self._synthesize_unit(
                text,
                language,
                context,
                run_id=run_id,
                beat_id="",
                speaker=speaker,
                voice=unit_voice,
                start=start,
                unit_id=f"{language}_dialogue_{unit_position}",
                line=line,
            )
            voices.append(record)
            cursor = end + self.gap_seconds
            total_cost += cost
            provider = provider or unit_provider
            model = model or unit_model
            ends_by_beat[""] = max(ends_by_beat.get("", end), end)

        for position, beat in enumerate(script.beats):
            beat_id = f"beat_{position}"
            units = _beat_units(beat)
            for unit_position, (text, speaker, unit_voice, line) in enumerate(units):
                start = cursor
                if line is not None and line.start_seconds is not None:
                    start = float(line.start_seconds)
                unit_id = (
                    f"{language}_{beat_id}"
                    if len(units) == 1
                    else f"{language}_{beat_id}_{unit_position}"
                )
                (
                    record,
                    end,
                    cost,
                    unit_provider,
                    unit_model,
                ) = await self._synthesize_unit(
                    text,
                    language,
                    context,
                    run_id=run_id,
                    beat_id=beat_id,
                    speaker=speaker,
                    voice=unit_voice,
                    start=start,
                    unit_id=unit_id,
                    line=line,
                )
                voices.append(record)
                starts_by_beat.setdefault(beat_id, start)
                ends_by_beat[beat_id] = max(ends_by_beat.get(beat_id, end), end)
                cursor = end + self.gap_seconds
                total_cost += cost
                provider = provider or unit_provider
                model = model or unit_model

        if not voices:
            raise ConfigError(
                "script has no spoken text",
                details={"language": language},
            )
        duration = max(ends_by_beat.values())
        beat_starts = [
            starts_by_beat.get(f"beat_{position}", 0.0)
            for position in range(len(script.beats))
        ]
        return voices, beat_starts, duration, total_cost, provider, model

    @staticmethod
    def _script_beat_starts(script: Script, duration: float) -> list[float]:
        beats = script.beats
        if not beats:
            return []
        durations = [beat.duration_seconds for beat in beats]
        if all(isinstance(value, (int, float)) and value > 0 for value in durations):
            starts: list[float] = []
            cursor = 0.0
            for value in durations:
                starts.append(cursor)
                cursor += float(cast(float, value))
            return starts
        step = duration / len(beats)
        return [position * step for position in range(len(beats))]

    @staticmethod
    def _event_ids(context: StageContext, run_id: str) -> set[str]:
        tracker = context.cost
        if tracker is None:
            return set()
        try:
            events = list(tracker.events(run_id=run_id or None))
        except TypeError:
            events = list(tracker.events())
        return {str(getattr(event, "id", "")) for event in events}

    @staticmethod
    def _new_events(
        context: StageContext,
        run_id: str,
        before: set[str],
    ) -> list[Any]:
        tracker = context.cost
        if tracker is None:
            return []
        try:
            events = list(tracker.events(run_id=run_id or None))
        except TypeError:
            events = list(tracker.events())
        return [
            event for event in events if str(getattr(event, "id", "")) not in before
        ]

    def _quality_report(
        self,
        timeline: Timeline,
        scripts: Mapping[str, Script],
    ) -> AudioQAReport | None:
        if not self.quality_enabled:
            return None
        extras = self.quality_extras
        return check_audio_quality(
            timeline,
            scripts=scripts,
            transcripts=extras.get("transcripts"),
            silences=extras.get("silences"),
            pronunciations=extras.get("pronunciations"),
            **self.quality_thresholds,
        )

    async def run(self, context: StageContext) -> StageResult:
        scripts = self._load_scripts(context)
        self.scripts_by_language = scripts
        run_id = context.run.id if context.run is not None else ""
        before = self._event_ids(context, run_id)
        languages = list(scripts)

        timeline_languages: list[dict[str, Any]] = []
        durations: dict[str, float] = {}
        beat_starts_by_language: dict[str, list[float]] = {}
        calls = 0
        total_cost = 0.0
        provider = ""
        model = ""
        for language in languages:
            script = scripts[language]
            (
                voices,
                beat_starts,
                duration,
                cost,
                language_provider,
                language_model,
            ) = await self._synthesize_language(
                language,
                script,
                context,
                run_id=run_id,
            )
            timeline_languages.append({"language": language, "voices": voices})
            durations[language] = duration
            beat_starts_by_language[language] = beat_starts
            calls += len(voices) if self.mode == "beat" else 1
            total_cost += cost
            provider = provider or language_provider
            model = model or language_model

        events = self._new_events(context, run_id, before)
        if total_cost == 0.0:
            total_cost = sum(
                max(0.0, float(getattr(event, "amount_usd", 0.0))) for event in events
            )

        first_language = languages[0] if languages else ""
        first_starts = beat_starts_by_language.get(first_language, [])
        beat_count = max((len(script.beats) for script in scripts.values()), default=0)
        beats: list[dict[str, Any]] = []
        for position in range(beat_count):
            text = ""
            for script in scripts.values():
                if position < len(script.beats):
                    text = _beat_text(script.beats[position]) or text
                    if text:
                        break
            beats.append(
                {
                    "id": f"beat_{position}",
                    "index": position,
                    "start_seconds": (
                        first_starts[position] if position < len(first_starts) else 0.0
                    ),
                    "text": text,
                }
            )

        timeline_payload: dict[str, Any] = {
            "id": (context.run.id if context.run is not None else "") or "timeline",
            "beats": beats,
            "languages": timeline_languages,
            "metadata": {
                "mode": self.mode,
                "gap_seconds": self.gap_seconds,
            },
        }
        timeline = Timeline.model_validate(timeline_payload)
        relative = self.timeline_filename
        _write_json(context.run_dir / relative, timeline.model_dump(mode="json"))
        self.artifacts = (relative,)

        quality = self._quality_report(timeline, scripts)
        metadata = {
            "mode": self.mode,
            "gap_seconds": self.gap_seconds,
            "languages": languages,
            "durations": durations,
            "calls": calls,
            "timeline": relative,
            "cost_usd": total_cost,
            "provider": provider,
            "model": model,
        }
        if quality is not None:
            report = quality.to_dict()
            metadata["quality"] = report
            metadata["audio_quality"] = report
            metadata["quality_ok"] = quality.passed
            metadata["quality_findings"] = report["findings"]
        return VoiceStageResult(
            run_id=run_id,
            stage=self.name,
            status=StageState.SUCCEEDED,
            artifacts=[relative],
            cost_usd=total_cost,
            metadata=metadata,
        )


Voice = VoiceStage
voice_stage = VoiceStage


def build_voice_stage(
    *,
    options: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> VoiceStage:
    """Build a :class:`VoiceStage` from recipe options and overrides."""
    return VoiceStage(options=options, **kwargs)


async def run_voice_stage(
    context: StageContext,
    *,
    stage: VoiceStage | None = None,
    **options: Any,
) -> StageResult:
    """Build a stage unless one is supplied, then run it asynchronously."""
    if stage is not None and options:
        raise ConfigError("pass either a stage or options, not both")
    active = stage if stage is not None else VoiceStage(**options)
    return await active.run(context)
