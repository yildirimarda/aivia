"""The concrete write stage: facts in, one native script per language out.

The stage is the bridge between the research fact list and every later stage.
It reads the run's fact list, asks the configured writing model for a script in
each target language and writes one JSON script per language into the run
directory. The model is never asked to translate an existing script: every
language is written natively from the same facts by
:mod:`aivia.writing.multilingual`.

Cost and tokens are reported on the :class:`~aivia.core.models.run.StageResult`
and in its metadata, so ``aivia write`` can print them without re-reading the
ledger. A caller may inject a provider, a structured helper or a writer object;
the default path selects an LLM provider through the stage context router.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Awaitable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from aivia.core.config.llm import LLMConfig, LLMRole
from aivia.core.errors import ConfigError
from aivia.core.models.base import canonical_json
from aivia.core.models.run import StageResult, StageState
from aivia.core.models.script import Script
from aivia.core.stages.base import Stage, StageContext
from aivia.llm.structured import StructuredOutputHelper
from aivia.writing.facts import extract_facts
from aivia.writing.multilingual import (
    MULTILINGUAL_OPERATION,
    generate_multilingual_scripts,
)

__all__ = [
    "DEFAULT_FACT_LIST_FILENAME",
    "DEFAULT_SCRIPT_FILENAME_PATTERN",
    "FACT_LIST_FILENAMES",
    "SCRIPT_FILENAME_PATTERN",
    "WRITE_STAGE_NAME",
    "Write",
    "WriteStage",
    "WriteStageResult",
    "WritingStage",
    "build_write_stage",
    "run_write_stage",
    "write_stage",
]

WRITE_STAGE_NAME = "write"
"""The recipe stage name this implementation serves."""

DEFAULT_FACT_LIST_FILENAME = "facts.json"
"""The fact list file the stage reads when no explicit path is configured."""

FACT_LIST_FILENAMES = (
    DEFAULT_FACT_LIST_FILENAME,
    "fact_list.json",
    "fact-list.json",
    "facts.yaml",
    "fact_list.yaml",
    "topic.json",
    "research.json",
)
"""Fact list names tried in order under the run directory."""

DEFAULT_SCRIPT_FILENAME_PATTERN = "script.{language}.json"
"""The per-language script file name pattern."""

SCRIPT_FILENAME_PATTERN = DEFAULT_SCRIPT_FILENAME_PATTERN

_FACT_LIST_SUFFIXES = {".json", ".yaml", ".yml"}
_SOURCE_KEYS = ("source_urls", "sources", "urls", "source_url", "source")
_TITLE_KEYS = ("title", "name", "topic", "subject")
_ANGLE_KEYS = ("angle", "perspective", "viewpoint")
_NOTES_KEYS = ("notes", "note", "summary")
_FACT_CONTAINER_KEYS = ("fact_list", "fact-list", "factlist", "fact_base", "factbase")
_SCRIPT_MODE_DEFAULT = "facts"


@dataclass(frozen=True, slots=True)
class _Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: _Usage) -> _Usage:
        return _Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
        }


class WriteStageResult(StageResult):
    """A stage result whose metadata carries the token breakdown."""

    @property
    def input_tokens(self) -> int:
        return _as_int(self.metadata.get("input_tokens"))

    @property
    def output_tokens(self) -> int:
        return _as_int(self.metadata.get("output_tokens"))

    @property
    def total_tokens(self) -> int:
        return _as_int(self.metadata.get("total_tokens"))


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


def _section(config: Any, name: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name, None)


def _llm_config(config: Any) -> LLMConfig | None:
    section = _section(config, "llm")
    if isinstance(section, LLMConfig):
        return section
    if isinstance(section, Mapping):
        try:
            return LLMConfig.from_mapping(section)
        except (TypeError, ValueError):
            return None
    return None


def _languages_section(config: Any) -> Any:
    return _section(config, "languages")


def _language_codes(config: Any) -> list[str]:
    section = _languages_section(config)
    if section is None:
        return []
    raw = getattr(section, "languages", None)
    if raw is None and isinstance(section, Mapping):
        raw = section.get("languages", section)
    if not isinstance(raw, Mapping):
        return []
    codes: list[str] = []
    for key in raw:
        code = str(key).strip()
        if code in {"default", "schema_version"}:
            continue
        if code and code not in codes:
            codes.append(code)
    return codes


def _default_language(config: Any) -> str:
    section = _languages_section(config)
    raw = getattr(section, "default", None)
    if raw is None and isinstance(section, Mapping):
        raw = section.get("default")
    return str(raw).strip() if isinstance(raw, str) and raw.strip() else "en"


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
            code = part.strip().casefold().replace("_", "-")
            if code and code not in codes:
                codes.append(code)
    return codes


def _value_of(value: Any, keys: Sequence[str]) -> Any:
    if isinstance(value, Mapping):
        for key in keys:
            if key in value:
                return value[key]
        return None
    for key in keys:
        item = getattr(value, key, None)
        if item is not None:
            return item
    return None


def _title_of(value: Any) -> str:
    raw = _value_of(value, _TITLE_KEYS)
    return raw.strip() if isinstance(raw, str) else ""


def _angle_of(value: Any) -> str:
    raw = _value_of(value, _ANGLE_KEYS)
    return raw.strip() if isinstance(raw, str) else ""


def _notes_of(value: Any) -> str:
    raw = _value_of(value, _NOTES_KEYS)
    return raw.strip() if isinstance(raw, str) else ""


def _source_urls(facts: Sequence[Any]) -> list[str]:
    urls: list[str] = []
    for fact in facts:
        for url in getattr(fact, "source_urls", ()) or ():
            if isinstance(url, str) and url not in urls:
                urls.append(url)
    return urls


def _iter_facts(value: Any) -> Iterable[Any]:
    """Read a fact list structurally, including research selection output."""
    if value is None:
        return
    if isinstance(value, Mapping):
        for key in _FACT_CONTAINER_KEYS:
            if key in value:
                yield from _iter_facts(value[key])
                return
        if "topic" in value and isinstance(value["topic"], Mapping):
            yield from _iter_facts(value["topic"])
            return
        if "selected" in value:
            for item in value.get("selected") or ():
                yield from _iter_facts(item)
            return
        for key in ("topics", "items", "results"):
            nested = value.get(key)
            if isinstance(nested, Sequence) and not isinstance(
                nested, (str, bytes, bytearray)
            ):
                for item in nested:
                    yield from _iter_facts(item)
                return
        yield value
        return
    if isinstance(value, (str, bytes, bytearray)):
        yield value
        return
    for key in _FACT_CONTAINER_KEYS:
        nested = getattr(value, key, None)
        if nested is not None:
            yield from _iter_facts(nested)
            return
    selected = getattr(value, "selected", None)
    if selected is not None:
        for item in selected:
            yield from _iter_facts(item)
        return
    yield value


def _facts_from(value: Any) -> list[Any]:
    """Return fact records from a fact list, research output or user text."""
    if value is None:
        return []
    if isinstance(value, Mapping) and "selected" in value:
        facts: list[Any] = []
        for item in value.get("selected") or ():
            facts.extend(_facts_from(item))
        return facts
    facts = list(extract_facts(value))
    if facts:
        return facts
    collected: list[Any] = []
    for item in _iter_facts(value):
        if item is value:
            continue
        collected.extend(extract_facts(item))
    if collected:
        return collected
    for item in _iter_facts(value):
        if item is value:
            continue
        title = _title_of(item)
        summary = _notes_of(item)
        text = summary or title
        urls = _value_of(item, _SOURCE_KEYS) or []
        if isinstance(urls, str):
            urls = [urls]
        if isinstance(text, str) and text.strip():
            collected.extend(extract_facts([{"text": text, "source_urls": urls}]))
    return collected


def _read_document(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(
                f"invalid fact list JSON: {path}",
                details={"path": str(path)},
            ) from exc
    if suffix in {".yaml", ".yml"}:
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigError(
                f"invalid fact list YAML: {path}",
                details={"path": str(path)},
            ) from exc
    return text


def _candidate_paths(run_dir: Path, explicit: Any) -> list[Path]:
    paths: list[Path] = []
    if isinstance(explicit, (str, Path)) and str(explicit).strip():
        candidate = Path(str(explicit)).expanduser()
        paths.append(candidate if candidate.is_absolute() else run_dir / candidate)
    for name in FACT_LIST_FILENAMES:
        paths.append(run_dir / name)
        paths.append(run_dir / "research" / name)
        paths.append(run_dir / "input" / name)
    unique: list[Path] = []
    for path in paths:
        if path not in unique:
            unique.append(path)
    return unique


def _registered_fact_paths(run_dir: Path) -> list[Path]:
    manifest = run_dir / "artifacts.json"
    if not manifest.is_file():
        return []
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    stages = payload.get("stages") if isinstance(payload, Mapping) else None
    if not isinstance(stages, Mapping):
        return []
    paths: list[Path] = []
    for stage in ("research", "input_topic", "input", "write"):
        entries = stages.get(stage)
        if not isinstance(entries, Sequence):
            continue
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            raw = entry.get("path")
            if not isinstance(raw, str):
                continue
            candidate = run_dir / raw
            if candidate.suffix.lower() in _FACT_LIST_SUFFIXES:
                paths.append(candidate)
    return paths


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


class _RecordingHelper:
    """A structured helper proxy that keeps every result for usage totals."""

    def __init__(self, helper: StructuredOutputHelper) -> None:
        self._helper = helper
        self.results: list[Any] = []

    async def complete(self, *args: Any, **kwargs: Any) -> Any:
        result = await self._helper.complete(*args, **kwargs)
        self.results.append(result)
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._helper, name)


def _event_usage(events: Iterable[Any]) -> _Usage:
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    cost = _as_float(0)
    for event in events:
        if getattr(event, "unit_kind", "") != "token":
            continue
        metadata = getattr(event, "metadata", None)
        metadata = metadata if isinstance(metadata, Mapping) else {}
        input_tokens += _as_int(metadata.get("input_tokens"))
        output_tokens += _as_int(metadata.get("output_tokens"))
        total_tokens += _as_int(getattr(event, "units", 0))
        cost += _as_float(getattr(event, "amount_usd", 0.0))
    if total_tokens == 0 and (input_tokens or output_tokens):
        total_tokens = input_tokens + output_tokens
    return _Usage(input_tokens, output_tokens, total_tokens, cost)


def _result_usage(result: Any) -> _Usage:
    if result is None:
        return _Usage()
    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, Mapping):
        nested = metadata.get("usage")
        values = nested if isinstance(nested, Mapping) else metadata
        input_tokens = _as_int(values.get("input_tokens"))
        output_tokens = _as_int(values.get("output_tokens"))
        total_tokens = _as_int(values.get("total_tokens"))
        cost = _as_float(values.get("cost_usd"))
        if cost == 0.0:
            cost = _as_float(metadata.get("cost_usd"))
        if input_tokens or output_tokens or total_tokens or cost:
            if total_tokens == 0 and (input_tokens or output_tokens):
                total_tokens = input_tokens + output_tokens
            return _Usage(input_tokens, output_tokens, total_tokens, cost)
    usage = getattr(result, "usage", None)
    if usage is not None:
        return _Usage(
            _as_int(getattr(usage, "input_tokens", 0)),
            _as_int(getattr(usage, "output_tokens", 0)),
            _as_int(getattr(usage, "total_tokens", 0)),
            _as_float(getattr(result, "amount_usd", getattr(result, "cost_usd", 0.0))),
        )
    input_tokens = _as_int(
        getattr(result, "input_tokens", None)
        if getattr(result, "input_tokens", None) is not None
        else _mapping_value(result, "input_tokens")
    )
    output_tokens = _as_int(
        getattr(result, "output_tokens", None)
        if getattr(result, "output_tokens", None) is not None
        else _mapping_value(result, "output_tokens")
    )
    total_tokens = _as_int(
        getattr(result, "total_tokens", None)
        if getattr(result, "total_tokens", None) is not None
        else _mapping_value(result, "total_tokens")
    )
    if total_tokens == 0 and (input_tokens or output_tokens):
        total_tokens = input_tokens + output_tokens
    cost = _as_float(
        getattr(result, "amount_usd", None)
        if getattr(result, "amount_usd", None) is not None
        else getattr(result, "cost_usd", 0.0)
    )
    if cost == 0.0:
        cost = _as_float(_mapping_value(result, "cost_usd"))
    return _Usage(input_tokens, output_tokens, total_tokens, cost)


def _mapping_value(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return None


def _usage_from_results(results: Iterable[Any]) -> _Usage:
    total = _Usage()
    for result in results:
        total += _result_usage(result)
    return total


def _scripts_from(value: Any, languages: Sequence[str], mode: str) -> dict[str, Script]:
    if value is None:
        return {}
    if hasattr(value, "scripts"):
        value = value.scripts
    if isinstance(value, Mapping) and isinstance(value.get("scripts"), Mapping):
        value = value["scripts"]
    if isinstance(value, Mapping):
        items = dict(value)
    elif hasattr(value, "items"):
        items = dict(cast(Any, value).items())
    else:
        raise ConfigError(
            "writer must return a mapping of language code to script",
            details={"value": type(value).__name__},
        )
    scripts: dict[str, Script] = {}
    for key, item in items.items():
        code = str(key).strip().casefold().replace("_", "-")
        if code not in languages:
            continue
        if isinstance(item, Script):
            scripts[code] = item
        elif isinstance(item, Mapping):
            payload = dict(item)
            payload.setdefault("language", code)
            payload.setdefault("mode", mode)
            scripts[code] = Script.model_validate(payload)
        elif isinstance(item, str):
            scripts[code] = Script.model_validate(
                {"mode": mode, "language": code, "hook": item.strip()}
            )
        else:
            raise ConfigError(
                "writer returned an unsupported script value",
                details={"language": code, "value": type(item).__name__},
            )
    return scripts


class WriteStage(Stage):
    """Write one script per target language from the run fact list.

    ``options`` may carry ``languages``/``language``, ``mode``, ``facts`` or
    ``fact_list``, ``fact_list_path``, ``title``, ``angle``, ``tone``,
    ``audience``, ``instructions``, ``model`` and ``role``. An explicit
    ``facts``/``fact_list`` value wins over a file. A caller may inject a
    ``writer`` (callable or object), a ``provider`` or a structured ``helper``;
    without one the stage selects an LLM provider through ``context.router``.
    """

    name = WRITE_STAGE_NAME
    artifacts: tuple[str, ...] = ()

    def __init__(
        self,
        name: str | None = None,
        *,
        options: Mapping[str, Any] | None = None,
        fact_list: Any = None,
        facts: Any = None,
        writer: Any = None,
        provider: Any = None,
        helper: Any = None,
        languages: Any = None,
        mode: str | None = None,
        fact_list_path: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        merged = dict(options or {})
        if fact_list is not None:
            merged["fact_list"] = fact_list
        if facts is not None:
            merged["facts"] = facts
        if writer is not None:
            merged["writer"] = writer
        if provider is not None:
            merged["provider"] = provider
        if helper is not None:
            merged["helper"] = helper
        if languages is not None:
            merged["languages"] = languages
        if mode is not None:
            merged["mode"] = mode
        if fact_list_path is not None:
            merged["fact_list_path"] = fact_list_path
        merged.update(kwargs)
        super().__init__(name=name or WRITE_STAGE_NAME, options=merged)
        self.fact_list = merged.get("fact_list")
        self.facts = merged.get("facts")
        self.writer = merged.get("writer")
        self.provider = merged.get("provider")
        self.helper = merged.get("helper")
        self.languages = merged.get("languages")
        self.mode = str(merged.get("mode") or _SCRIPT_MODE_DEFAULT).strip()
        self.fact_list_path = merged.get("fact_list_path") or merged.get("facts_path")
        self.scripts: dict[str, Script] = {}
        self.source_facts: list[Any] = []

    def _load_facts(self, context: StageContext) -> tuple[list[Any], Any]:
        for value in (
            self.fact_list,
            self.facts,
            self.options.get("fact_list"),
            self.options.get("facts"),
            self.options.get("topic"),
        ):
            if value is not None:
                facts = _facts_from(value)
                if not facts:
                    raise ConfigError(
                        "the supplied fact list is empty",
                        details={"stage": self.name},
                    )
                return facts, value

        explicit = (
            self.fact_list_path
            or self.options.get("fact_list_path")
            or self.options.get("facts_path")
            or self.options.get("input")
        )
        for path in _candidate_paths(context.run_dir, explicit):
            if not path.is_file():
                continue
            value = _read_document(path)
            facts = _facts_from(value)
            if facts:
                return facts, value
        for path in _registered_fact_paths(context.run_dir):
            if not path.is_file():
                continue
            value = _read_document(path)
            facts = _facts_from(value)
            if facts:
                return facts, value

        run = context.run
        if run is not None:
            for source in (run.config_snapshot, getattr(run, "metadata", None)):
                if isinstance(source, Mapping):
                    for key in ("fact_list", "facts", "topic"):
                        if key in source:
                            facts = _facts_from(source[key])
                            if facts:
                                return facts, source[key]

        searched = ", ".join(
            str(path) for path in _candidate_paths(context.run_dir, explicit)
        )
        raise ConfigError(
            "write stage needs a fact list",
            details={
                "stage": self.name,
                "run_dir": str(context.run_dir),
                "searched": searched,
            },
        )

    def _target_languages(self, context: StageContext) -> list[str]:
        explicit = self.languages
        if explicit is None:
            explicit = self.options.get("languages")
        if explicit is None:
            explicit = self.options.get("language")
        codes = _normalise_languages(explicit)
        configured = _language_codes(context.config)
        if codes:
            if configured:
                unknown = [code for code in codes if code not in configured]
                if unknown:
                    available = ", ".join(configured)
                    raise ConfigError(
                        f"unknown language(s): {', '.join(unknown)} "
                        f"(available: {available})",
                        details={
                            "languages": unknown,
                            "available": configured,
                        },
                    )
            return codes
        if configured:
            return configured
        return [_default_language(context.config)]

    def _writer_kwargs(
        self,
        facts: Sequence[Any],
        languages: Sequence[str],
        context: StageContext,
        *,
        title: str,
        angle: str,
        notes: str,
        source_urls: Sequence[str],
    ) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "title": title,
            "angle": angle,
            "notes": notes,
            "source_urls": list(source_urls),
            "tone": str(self.options.get("tone") or ""),
            "audience": str(self.options.get("audience") or ""),
            "instructions": str(self.options.get("instructions") or ""),
            "model": self.options.get("model"),
            "role": str(self.options.get("role") or LLMRole.WRITING.value),
            "run_id": context.run.id if context.run is not None else "",
            "context": context,
        }

    async def _write_with_writer(
        self,
        writer: Any,
        facts: Sequence[Any],
        languages: Sequence[str],
        stage_context: StageContext,
        **kwargs: Any,
    ) -> tuple[Any, list[Any]]:
        target = writer
        method = None
        for name in ("write", "generate", "write_sync", "generate_sync"):
            candidate = getattr(target, name, None)
            if callable(candidate):
                method = candidate
                break
        if method is None and callable(target):
            method = target
        if method is None:
            raise ConfigError(
                "writer must be callable or expose a write method",
                details={"value": type(writer).__name__},
            )
        call_kwargs = {key: value for key, value in kwargs.items() if key != "facts"}
        try:
            value = method(facts, list(languages), **call_kwargs)
        except TypeError:
            call_kwargs.pop("context", None)
            try:
                value = method(facts, list(languages), **call_kwargs)
            except TypeError:
                value = method(facts, list(languages))
        if isinstance(value, Awaitable):
            value = await value
        results = getattr(value, "results", None)
        captured = list(results) if isinstance(results, Sequence) else []
        return value, captured

    async def _default_write(
        self,
        facts: Sequence[Any],
        languages: Sequence[str],
        stage_context: StageContext,
        **kwargs: Any,
    ) -> tuple[Any, list[Any]]:
        provider = self.provider
        if provider is None and stage_context.router is not None:
            provider = stage_context.router.select("llm")
        if provider is None:
            raise ConfigError(
                "write stage requires an LLM provider",
                details={"stage": self.name},
            )
        helper = self.helper
        if helper is None:
            config = _llm_config(stage_context.config)
            priced = config is not None and bool(config.pricing)
            tracker = stage_context.cost if priced else None
            helper = StructuredOutputHelper(
                provider,
                config=config if priced else None,
                tracker=tracker,
                run_id=(stage_context.run.id if stage_context.run is not None else ""),
                operation=MULTILINGUAL_OPERATION,
            )
        recording = _RecordingHelper(helper)
        languages_section = _languages_section(stage_context.config)
        result = await generate_multilingual_scripts(
            facts,
            list(languages),
            languages=languages_section or stage_context.config,
            helper=cast(Any, recording),
            mode=self.mode,
            title=kwargs.get("title", ""),
            angle=kwargs.get("angle", ""),
            tone=kwargs.get("tone", ""),
            audience=kwargs.get("audience", ""),
            instructions=kwargs.get("instructions", ""),
            model=kwargs.get("model"),
            role=kwargs.get("role", LLMRole.WRITING.value),
            run_id=kwargs.get("run_id", ""),
            operation=MULTILINGUAL_OPERATION,
        )
        return result, recording.results

    async def run(self, context: StageContext) -> StageResult:
        facts, source = self._load_facts(context)
        self.source_facts = facts
        languages = self._target_languages(context)
        title = str(self.options.get("title") or _title_of(source))
        angle = str(self.options.get("angle") or _angle_of(source))
        notes = str(self.options.get("notes") or _notes_of(source))
        source_urls = _source_urls(facts)
        kwargs = self._writer_kwargs(
            facts,
            languages,
            context,
            title=title,
            angle=angle,
            notes=notes,
            source_urls=source_urls,
        )

        run_id = context.run.id if context.run is not None else ""
        before = self._event_ids(context, run_id)
        if self.writer is not None:
            value, captured = await self._write_with_writer(
                self.writer, facts, languages, context, **kwargs
            )
        else:
            value, captured = await self._default_write(
                facts, languages, context, **kwargs
            )

        scripts = _scripts_from(value, languages, self.mode)
        missing = [code for code in languages if code not in scripts]
        if missing:
            raise ConfigError(
                "writer did not return a script for every language",
                details={"languages": list(languages), "missing": missing},
            )

        artifacts: list[str] = []
        pattern = str(
            self.options.get("filename_pattern") or DEFAULT_SCRIPT_FILENAME_PATTERN
        )
        for code in languages:
            script = scripts[code]
            relative = pattern.format(language=code, code=code)
            _write_json(context.run_dir / relative, script.model_dump(mode="json"))
            artifacts.append(relative)
        self.scripts = scripts
        self.artifacts = tuple(artifacts)

        usage = _usage_from_results(captured)
        if usage.total_tokens == 0 and usage.cost_usd == 0.0:
            usage = _result_usage(value)
        events = self._new_events(context, run_id, before)
        event_usage = _event_usage(events)
        if usage.total_tokens == 0 and usage.cost_usd == 0.0:
            usage = event_usage
        elif usage.cost_usd == 0.0 and event_usage.cost_usd:
            usage = _Usage(
                usage.input_tokens,
                usage.output_tokens,
                usage.total_tokens,
                event_usage.cost_usd,
            )

        provider = str(getattr(value, "provider", "") or "")
        model = str(getattr(value, "model", "") or "")
        metadata = {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
            "usage": usage.to_dict(),
            "cost_usd": usage.cost_usd,
            "languages": list(languages),
            "scripts": dict(zip(languages, artifacts, strict=False)),
            "mode": self.mode,
            "fact_count": len(facts),
            "source_urls": source_urls,
            "provider": provider,
            "model": model,
        }
        return WriteStageResult(
            run_id=run_id,
            stage=self.name,
            status=StageState.SUCCEEDED,
            artifacts=artifacts,
            cost_usd=usage.cost_usd,
            metadata=metadata,
        )

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


WritingStage = WriteStage
Write = WriteStage
write_stage = WriteStage


def build_write_stage(
    *,
    options: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> WriteStage:
    """Build a :class:`WriteStage` from recipe options and overrides."""
    return WriteStage(options=options, **kwargs)


async def run_write_stage(
    context: StageContext,
    *,
    stage: WriteStage | None = None,
    **options: Any,
) -> StageResult:
    """Build a stage unless one is supplied, then run it asynchronously."""
    active = stage if stage is not None else WriteStage(**options)
    if stage is not None and options:
        raise ConfigError("pass either a stage or options, not both")
    return await active.run(context)
