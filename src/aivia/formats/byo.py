"""Provided text and script inputs that replace research and writing.

A caller may already have the facts or the finished script. Passing either file
lets a run skip the paid stage that would have produced it, while the remaining
stages read the same artifacts they normally would. The input is validated
before a run is created so an unreadable or malformed file never starts a paid
pipeline.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from aivia.core.config.formats import FormatRecipeConfig
from aivia.core.errors import ConfigError, ValidationError
from aivia.core.models.base import canonical_json
from aivia.core.models.script import Script
from aivia.formats.input_topic import (
    DEFAULT_FACT_LIST_FILENAME,
    DEFAULT_TOPIC_FILENAME,
    InputTopicStage,
    ResolvedTopic,
    resolve_input_topic,
)
from aivia.llm.languages import normalize_language_code
from aivia.writing.stage import DEFAULT_SCRIPT_FILENAME_PATTERN

__all__ = [
    "ProvidedInputs",
    "apply_provided_inputs",
    "apply_stage_options",
    "load_provided_inputs",
    "provided_stage_options",
    "script_pattern",
    "seed_provided_inputs",
]

_SCRIPT_KEYS = frozenset(
    {
        "beats",
        "dialogue",
        "dialogue_lines",
        "hook",
        "lang",
        "language",
        "language_code",
        "locale",
        "mode",
        "name",
        "outro",
        "payload",
        "scenes",
        "segments",
        "subject",
        "title",
        "units",
        "writing_mode",
    }
)
_LANGUAGE_PART = re.compile(r"[A-Za-z]{2,3}(?:[-_][A-Za-z]{2,4})?")


def _error(message: str, path: Path) -> ConfigError:
    return ConfigError(message, details={"path": str(path)})


def _existing(path: str | Path, kind: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_file():
        raise _error(f"provided {kind} file not found: {candidate}", candidate)
    return candidate


def _read_text(path: Path, kind: str) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise _error(f"cannot read provided {kind} file {path}: {exc}", path) from exc
    if not text.strip():
        raise _error(f"provided {kind} file is empty: {path}", path)
    return text


def _language_from_path(path: Path) -> str:
    parts = path.stem.replace("_", ".").split(".")
    for part in reversed(parts):
        if _LANGUAGE_PART.fullmatch(part):
            try:
                return normalize_language_code(part)
            except (ValueError, ValidationError):
                continue
    return ""


def _language(value: Any, fallback: str = "") -> str:
    if isinstance(value, str) and value.strip():
        try:
            return normalize_language_code(value)
        except (ValueError, ValidationError) as exc:
            raise ConfigError(str(exc), details={"language": value}) from exc
    return fallback or "en"


def _has_content(script: Script) -> bool:
    if script.hook.strip() or script.outro.strip() or script.dialogue_lines:
        return True
    if script.payload:
        return True
    return any(
        beat.text.strip() or beat.dialogue or beat.media_prompts
        for beat in script.beats
    )


def _script_from_value(
    value: Any,
    *,
    language: str,
    path: Path,
) -> Script:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise _error(f"provided script file has an empty script: {path}", path)
        script = Script.model_validate(
            {
                "language": language,
                "hook": text,
                "beats": [{"text": text}],
            }
        )
    elif isinstance(value, Mapping):
        payload = dict(value)
        payload.setdefault("language", language)
        try:
            script = Script.model_validate(payload)
        except ValueError as exc:
            raise _error(f"invalid provided script: {path}: {exc}", path) from exc
    else:
        raise _error(
            f"provided script must be a mapping or text, got "
            f"{type(value).__name__}: {path}",
            path,
        )
    if not _has_content(script):
        raise _error(f"provided script has no content: {path}", path)
    return script


def _plain_text_scripts(text: str, path: Path, language: str) -> dict[str, Script]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise _error(f"provided script file is empty: {path}", path)
    script = Script.model_validate(
        {
            "language": language,
            "title": lines[0],
            "hook": lines[0],
            "beats": [{"text": line} for line in lines],
        }
    )
    return {language: script}


def _document(path: Path, kind: str) -> Any:
    text = _read_text(path, kind)
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise _error(f"invalid provided script JSON: {path}", path) from exc
    if suffix in {".yaml", ".yml"}:
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise _error(f"invalid provided script YAML: {path}", path) from exc
    return text


def _scripts_from_payload(payload: Any, path: Path) -> dict[str, Script]:
    fallback = _language_from_path(path)
    if isinstance(payload, str):
        return _plain_text_scripts(payload, path, fallback or "en")
    if payload is None:
        raise _error(f"provided script file is empty: {path}", path)

    if isinstance(payload, Mapping):
        nested = payload.get("scripts")
        if isinstance(nested, Mapping):
            payload = nested
        elif not (_SCRIPT_KEYS & set(payload)):
            return _scripts_from_language_map(payload, path, fallback)

    if isinstance(payload, Mapping):
        language = _language(
            payload.get("language") or payload.get("lang") or payload.get("locale"),
            fallback,
        )
        return {language: _script_from_value(payload, language=language, path=path)}

    if isinstance(payload, Sequence) and not isinstance(payload, (bytes, bytearray)):
        scripts: dict[str, Script] = {}
        for item in payload:
            if not isinstance(item, Mapping):
                raise _error(
                    f"provided script list must contain mappings: {path}",
                    path,
                )
            raw = item.get("language") or item.get("lang") or item.get("locale")
            if raw is None:
                raise _error(
                    f"provided script list needs a language per script: {path}",
                    path,
                )
            language = _language(raw, fallback)
            if language in scripts:
                raise _error(
                    f"duplicate provided script language {language!r}: {path}",
                    path,
                )
            scripts[language] = _script_from_value(
                item,
                language=language,
                path=path,
            )
        if not scripts:
            raise _error(f"provided script file is empty: {path}", path)
        return scripts

    raise _error(
        f"provided script must be a mapping, list or text, got "
        f"{type(payload).__name__}: {path}",
        path,
    )


def _scripts_from_language_map(
    payload: Mapping[Any, Any],
    path: Path,
    fallback: str,
) -> dict[str, Script]:
    scripts: dict[str, Script] = {}
    for key, value in payload.items():
        language = _language(str(key), fallback)
        if language in scripts:
            raise _error(
                f"duplicate provided script language {language!r}: {path}",
                path,
            )
        scripts[language] = _script_from_value(
            value,
            language=language,
            path=path,
        )
    if not scripts:
        raise _error(f"provided script file is empty: {path}", path)
    return scripts


def _load_scripts(path: str | Path) -> dict[str, Script]:
    target = _existing(path, "script")
    payload = _document(target, "script")
    return _scripts_from_payload(payload, target)


def _topic_from_scripts(scripts: Mapping[str, Script]) -> ResolvedTopic | None:
    if not scripts:
        return None
    script = next(iter(scripts.values()))
    texts: list[str] = []
    for beat in script.beats:
        if beat.text.strip():
            texts.append(beat.text.strip())
        for line in beat.dialogue:
            if line.text.strip():
                texts.append(line.text.strip())
    for line in script.dialogue_lines:
        if line.text.strip():
            texts.append(line.text.strip())
    title = script.title.strip() or script.hook.strip()
    if not title and texts:
        title = texts[0]
    if not title:
        return None
    facts = [{"text": text} for text in texts] or [{"text": title}]
    return resolve_input_topic({"title": title, "facts": facts})


@dataclass(frozen=True, slots=True)
class ProvidedInputs:
    """Validated files a caller supplied instead of research or writing."""

    text_file: Path | None = None
    script_file: Path | None = None
    topic: ResolvedTopic | None = None
    scripts: Mapping[str, Script] = field(default_factory=dict)

    @property
    def supplied(self) -> bool:
        """Whether any input file was supplied."""
        return self.text_file is not None or self.script_file is not None

    @property
    def has_script(self) -> bool:
        """Whether a prepared script replaces the write stage."""
        return self.script_file is not None


def load_provided_inputs(
    *,
    text_file: str | Path | None = None,
    script_file: str | Path | None = None,
) -> ProvidedInputs:
    """Validate and load the files a run should use instead of paid stages."""
    topic: ResolvedTopic | None = None
    scripts: dict[str, Script] = {}
    text_path: Path | None = None
    script_path: Path | None = None

    if text_file is not None:
        text_path = _existing(text_file, "text")
        _read_text(text_path, "text")
        topic = InputTopicStage(file=text_path).resolve()

    if script_file is not None:
        script_path = _existing(script_file, "script")
        scripts = _load_scripts(script_path)
        if topic is None:
            topic = _topic_from_scripts(scripts)

    return ProvidedInputs(
        text_file=text_path,
        script_file=script_path,
        topic=topic,
        scripts=scripts,
    )


def provided_stage_options(
    provided: ProvidedInputs,
) -> dict[str, dict[str, Any]]:
    """Return the recipe options that skip or feed the replaced stages."""
    options: dict[str, dict[str, Any]] = {}
    if provided.text_file is not None or provided.script_file is not None:
        options["research"] = {"skip": True}
    if provided.text_file is not None:
        options["input_topic"] = {"file": str(provided.text_file)}
    if provided.script_file is not None:
        options["write"] = {"skip": True}
    return options


def apply_stage_options(
    recipe: FormatRecipeConfig,
    options: Mapping[str, Mapping[str, Any]],
) -> FormatRecipeConfig:
    """Merge per-stage option overrides into *recipe* without mutating it."""
    stages = []
    for item in recipe.stages:
        extra = options.get(item.stage)
        if extra:
            merged = dict(item.options)
            merged.update(extra)
            item = item.model_copy(update={"options": merged})
        stages.append(item)
    return recipe.model_copy(update={"stages": stages})


def apply_provided_inputs(
    recipe: FormatRecipeConfig,
    provided: ProvidedInputs,
) -> FormatRecipeConfig:
    """Apply the skipped and fed stages a provided input implies."""
    if not provided.supplied:
        return recipe
    return apply_stage_options(recipe, provided_stage_options(provided))


def script_pattern(recipe: FormatRecipeConfig) -> str:
    """Return the write stage's script file pattern, or the default."""
    for item in recipe.stages:
        if item.stage == "write":
            value = item.options.get("filename_pattern")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return DEFAULT_SCRIPT_FILENAME_PATTERN


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(dict(payload), indent=2) + "\n", encoding="utf-8")


def seed_provided_inputs(
    run_dir: str | Path,
    provided: ProvidedInputs,
    *,
    pattern: str = DEFAULT_SCRIPT_FILENAME_PATTERN,
) -> tuple[str, ...]:
    """Write the provided topic and scripts into the run directory.

    The remaining stages read exactly the artifacts the skipped stages would
    have written, so no downstream stage needs to know where the input came
    from.
    """
    directory = Path(run_dir).expanduser()
    written: list[str] = []

    if provided.topic is not None:
        payload = provided.topic.to_payload()
        topic_relative = DEFAULT_TOPIC_FILENAME
        _write_json(directory / topic_relative, payload)
        written.append(topic_relative)
        facts_relative = DEFAULT_FACT_LIST_FILENAME
        if facts_relative != topic_relative:
            _write_json(directory / facts_relative, payload)
            written.append(facts_relative)

    for language, script in provided.scripts.items():
        relative = pattern.format(language=language, code=language)
        _write_json(directory / relative, script.model_dump(mode="json"))
        written.append(relative)

    return tuple(dict.fromkeys(written))
