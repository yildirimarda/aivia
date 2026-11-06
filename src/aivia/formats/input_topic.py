"""The input topic stage: build the fact list every format starts from.

A format recipe needs a topic before writing can start. Research is optional:
the same stage accepts a topic supplied on the command line, a text or JSON/YAML
file, or the output of the research stage. Whatever the source, it normalizes
the result into one canonical ``topic.json`` fact list that the writing stage
already knows how to read.

The stage is deterministic and offline. It never calls a provider and therefore
costs nothing. It records where the topic came from and keeps every source URL
that the input carried so the accuracy invariant is not lost when research is
skipped.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

from aivia.core.errors import ConfigError
from aivia.core.models.base import canonical_json
from aivia.core.models.run import StageResult, StageState
from aivia.core.models.topic import Claim, Topic
from aivia.core.stages.base import Stage, StageContext

__all__ = [
    "DEFAULT_FACT_LIST_FILENAME",
    "DEFAULT_INPUT_FILENAMES",
    "DEFAULT_TOPIC_FILENAME",
    "INPUT_TOPIC_STAGE_NAME",
    "InputStage",
    "InputTopic",
    "InputTopicResult",
    "InputTopicStage",
    "ResolvedTopic",
    "TopicInput",
    "TopicInputStage",
    "TopicResult",
    "TopicStage",
    "build_input_topic_stage",
    "input_topic",
    "input_topic_stage",
    "resolve_input_topic",
    "run_input_topic_stage",
]

INPUT_TOPIC_STAGE_NAME = "input_topic"
"""The recipe stage name this implementation serves."""

DEFAULT_TOPIC_FILENAME = "topic.json"
"""The canonical topic artifact written into the run directory."""

DEFAULT_FACT_LIST_FILENAME = "facts.json"
"""The fact-list alias the writing stage looks for first."""

DEFAULT_INPUT_FILENAMES = (
    "research/facts.json",
    "research/claims.json",
    "research/topic.json",
    "research/selected.json",
    "research/topics.json",
    "research.json",
    "facts.json",
    "fact_list.json",
    "fact-list.json",
    "facts.yaml",
    "fact_list.yaml",
    "input/facts.json",
    "input/topic.json",
    "input/topic.txt",
    "input/topic.md",
    "topic.txt",
    "topic.md",
)
"""Research or topic files tried under the run directory when no source is given."""

_TITLE_KEYS = ("title", "name", "subject", "topic_title", "headline", "query")
_SUMMARY_KEYS = ("summary", "description", "excerpt", "abstract", "notes", "note")
_TEXT_KEYS = (
    "text",
    "statement",
    "claim",
    "fact",
    "content",
    "body",
    "quote",
    "narration",
    "voiceover",
)
_URL_KEYS = ("source_urls", "sources", "urls", "source_url", "source")
_MODE_KEYS = ("mode", "topic_mode", "source_mode")
_CLAIM_KEYS = ("claims", "facts")
_SELECTED_KEYS = ("selected", "topics", "items", "results")
_BEAT_KEYS = ("beats", "scenes")
_PAYLOAD_KEYS = ("payload", "mode_payload", "mode_data", "data")
_TITLE_MAX = 160
_MAX_UNIT = 1.0
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")

_SOURCE_KEYS = {
    "file": (
        "file",
        "topic_file",
        "input_file",
        "source_file",
        "input_path",
        "path",
        "from_file",
        "input",
    ),
    "research": (
        "research",
        "research_output",
        "research_input",
        "topic_input",
        "researched",
        "source",
    ),
    "facts": ("facts", "fact_list", "claims"),
    "text": ("text", "content", "body", "prompt", "topic_text", "quote", "script"),
    "topic": ("topic",),
    "title": ("title", "name", "subject", "topic_title", "headline"),
}


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _value_of(value: Any, keys: Sequence[str]) -> Any:
    if isinstance(value, Mapping):
        for key in keys:
            if key in value:
                return value[key]
        return None
    for key in keys:
        if hasattr(value, key):
            candidate = getattr(value, key)
            if not callable(candidate):
                return candidate
    return None


def _clean_urls(value: Any, *, source: str = "") -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_values: Iterable[Any] = (value,)
    elif isinstance(value, Mapping):
        url = value.get("url") or value.get("source_url")
        raw_values = () if url is None else (url,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        raw_values = value
    else:
        raise ConfigError(
            "source URLs must be a URL, a mapping or a list",
            details={"stage": INPUT_TOPIC_STAGE_NAME, "source": source},
        )
    urls: list[str] = []
    for raw in raw_values:
        if not isinstance(raw, str):
            raise ConfigError(
                "source URL must be a string",
                details={"stage": INPUT_TOPIC_STAGE_NAME, "source": source},
            )
        url = raw.strip()
        if not url:
            continue
        parsed = urlsplit(url)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            raise ConfigError(
                f"source URL must be an absolute http(s) URL: {url!r}",
                details={"stage": INPUT_TOPIC_STAGE_NAME, "source": source},
            )
        if url not in urls:
            urls.append(url)
    return urls


def _as_fact(value: Any, *, source: str = "") -> dict[str, Any] | None:
    if isinstance(value, str):
        text = value.strip()
        return None if not text else _fact_payload(text)
    if value is None or isinstance(value, (bytes, bytearray)):
        return None
    text = _text(_value_of(value, _TEXT_KEYS))
    if not text:
        return None
    confidence = _value_of(value, ("confidence",))
    verified = _value_of(value, ("verified",))
    return _fact_payload(
        text,
        _clean_urls(_value_of(value, _URL_KEYS), source=source),
        confidence=_MAX_UNIT if confidence is None else confidence,
        verified=False if verified is None else bool(verified),
    )


def _fact_payload(
    text: str,
    source_urls: Sequence[str] = (),
    *,
    confidence: Any = 1.0,
    verified: bool = False,
) -> dict[str, Any]:
    try:
        number = float(confidence)
    except (TypeError, ValueError):
        number = 1.0
    if number < 0.0:
        number = 0.0
    if number > 1.0:
        number = 1.0
    return {
        "text": text.strip(),
        "source_urls": list(source_urls),
        "confidence": number,
        "verified": bool(verified),
    }


def _split_text(value: str) -> list[str]:
    lines = [line.strip() for line in value.splitlines()]
    lines = [_BULLET.sub("", line).strip() for line in lines if line.strip()]
    if len(lines) > 1:
        return lines
    if not lines:
        return []
    return [part.strip() for part in _SENTENCE_BREAK.split(lines[0]) if part.strip()]


def _iter_facts(value: Any, *, source: str = "") -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        return [_fact_payload(item) for item in _split_text(value)]
    if isinstance(value, (bytes, bytearray)):
        return []
    if isinstance(value, Mapping):
        payload = _value_of(value, _PAYLOAD_KEYS)
        if payload is not None:
            nested = _iter_facts(payload, source=source)
            if nested:
                return nested
        for key in _CLAIM_KEYS:
            if key not in value:
                continue
            nested = _iter_facts(value[key], source=source)
            if nested:
                return nested
        selected = _value_of(value, _SELECTED_KEYS)
        if selected is not None:
            nested = _iter_facts(selected, source=source)
            if nested:
                return nested
        beats = _value_of(value, _BEAT_KEYS)
        if beats is not None:
            nested = _iter_facts(beats, source=source)
            if nested:
                return nested
        fact = _as_fact(value, source=source)
        if fact is not None:
            return [fact]
        text = _text(_value_of(value, _SUMMARY_KEYS)) or _text(
            _value_of(value, _TITLE_KEYS)
        )
        if text:
            return [
                _fact_payload(
                    text,
                    _clean_urls(_value_of(value, _URL_KEYS), source=source),
                    verified=bool(_value_of(value, ("verified",))),
                )
            ]
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        facts: list[dict[str, Any]] = []
        for item in value:
            facts.extend(_iter_facts(item, source=source))
        return facts

    payload = _value_of(value, _PAYLOAD_KEYS)
    if payload is not None:
        nested = _iter_facts(payload, source=source)
        if nested:
            return nested
    for key in _CLAIM_KEYS:
        nested_value = getattr(value, key, None)
        if nested_value is None or callable(nested_value):
            continue
        nested = _iter_facts(nested_value, source=source)
        if nested:
            return nested
    selected = _value_of(value, _SELECTED_KEYS)
    if selected is not None:
        nested = _iter_facts(selected, source=source)
        if nested:
            return nested
    beats = _value_of(value, _BEAT_KEYS)
    if beats is not None:
        nested = _iter_facts(beats, source=source)
        if nested:
            return nested
    fact = _as_fact(value, source=source)
    if fact is not None:
        return [fact]
    summary = _text(_value_of(value, _SUMMARY_KEYS))
    title = _text(_value_of(value, _TITLE_KEYS))
    text = summary or title
    if text:
        return [
            _fact_payload(
                text,
                _clean_urls(_value_of(value, _URL_KEYS), source=source),
                verified=bool(_value_of(value, ("verified",))),
            )
        ]
    return []


def _title_from(value: Any, facts: Sequence[Mapping[str, Any]]) -> str:
    title = _text(_value_of(value, _TITLE_KEYS))
    if title:
        return title
    selected = _value_of(value, _SELECTED_KEYS)
    if selected is not None and not isinstance(selected, (str, bytes, bytearray)):
        if isinstance(selected, Mapping):
            if any(key in selected for key in _TITLE_KEYS):
                title = _text(_value_of(selected, _TITLE_KEYS))
                if title:
                    return title
            selected = list(selected.values())
        if isinstance(selected, Sequence):
            for item in selected:
                title = _text(_value_of(item, _TITLE_KEYS))
                if title:
                    return title
    if isinstance(value, str):
        lines = [line.strip() for line in value.splitlines() if line.strip()]
        if lines:
            return lines[0][:_TITLE_MAX].strip()
    if facts:
        text = str(facts[0].get("text", "")).strip()
        if text:
            first = _SENTENCE_BREAK.split(text)[0].strip()
            return (first or text)[:_TITLE_MAX].strip()
    return ""


def _summary_from(value: Any) -> str:
    return _text(_value_of(value, _SUMMARY_KEYS))


def _mode_from(value: Any) -> str:
    return _text(_value_of(value, _MODE_KEYS))


def _selected_first(value: Any) -> Any:
    selected = _value_of(value, ("selected",))
    if selected is None:
        return value
    if isinstance(selected, Mapping):
        if any(key in selected for key in _TITLE_KEYS):
            return selected
        selected = list(selected.values())
    if isinstance(selected, Sequence) and not isinstance(
        selected, (str, bytes, bytearray)
    ):
        return selected[0] if selected else value
    return value


@dataclass(frozen=True, slots=True)
class ResolvedTopic:
    """The normalized topic and the provenance of the input that produced it."""

    title: str
    summary: str = ""
    facts: tuple[dict[str, Any], ...] = ()
    source_urls: tuple[str, ...] = ()
    mode: str = ""
    origin: str = ""
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def claims(self) -> list[Claim]:
        """Return claims for the facts that carry at least one source URL."""
        if any(not fact.get("source_urls") for fact in self.facts):
            return []
        return [
            Claim(
                text=str(fact["text"]),
                source_urls=list(fact["source_urls"]),
                confidence=float(fact.get("confidence", 1.0)),
                verified=bool(fact.get("verified", False)),
            )
            for fact in self.facts
        ]

    @property
    def topic(self) -> Topic:
        """Return the core :class:`Topic` view of this input."""
        return Topic(
            title=self.title,
            summary=self.summary,
            claims=self.claims,
            source_urls=list(self.source_urls),
            mode=self.mode,
            verified=bool(self.facts)
            and all(bool(fact.get("verified")) for fact in self.facts),
        )

    @property
    def fact_count(self) -> int:
        """The number of facts in the topic."""
        return len(self.facts)

    def to_payload(self) -> dict[str, Any]:
        """Return the canonical JSON payload written as ``topic.json``."""
        payload: dict[str, Any] = {
            "title": self.title,
            "summary": self.summary,
            "facts": [dict(fact) for fact in self.facts],
            "source_urls": list(self.source_urls),
            "mode": self.mode,
            "origin": self.origin,
            "metadata": dict(self.metadata),
        }
        claims = self.claims
        if claims:
            payload["claims"] = [claim.model_dump(mode="json") for claim in claims]
        return payload


InputTopicResult = StageResult
"""The input stage returns a plain :class:`StageResult`."""

TopicResult = StageResult
"""Alias for :class:`InputTopicResult`."""


def _section(config: Any, name: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name, None)


def _path_from(value: Any, context: StageContext | None) -> Path | None:
    if not isinstance(value, (str, Path)):
        return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute() and context is not None:
        candidate = context.run_dir / candidate
    return candidate


def _read_file(path: Path) -> Any:
    if not path.is_file():
        raise ConfigError(
            f"input topic file not found: {path}",
            details={"stage": INPUT_TOPIC_STAGE_NAME, "path": str(path)},
        )
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(
            f"cannot read input topic file {path}: {exc}",
            details={"stage": INPUT_TOPIC_STAGE_NAME, "path": str(path)},
        ) from exc
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(
                f"invalid topic JSON: {path}",
                details={"stage": INPUT_TOPIC_STAGE_NAME, "path": str(path)},
            ) from exc
    if suffix in {".yaml", ".yml"}:
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigError(
                f"invalid topic YAML: {path}",
                details={"stage": INPUT_TOPIC_STAGE_NAME, "path": str(path)},
            ) from exc
    return text


def _existing_file(value: Any, context: StageContext | None) -> Path | None:
    path = _path_from(value, context)
    if path is not None and path.is_file():
        return path
    return None


def _pick_option(options: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = options.get(key)
        if value is not None:
            return value
    return None


def _candidate_files(context: StageContext | None) -> list[Path]:
    if context is None:
        return []
    return [context.run_dir / name for name in DEFAULT_INPUT_FILENAMES]


def _resolve_source(
    options: Mapping[str, Any],
    context: StageContext | None,
) -> tuple[Any, str, str]:
    """Return ``(value, origin, source)`` for the highest-priority input."""
    file_value = _pick_option(options, _SOURCE_KEYS["file"])
    if file_value is not None:
        path = _path_from(file_value, context)
        if path is None:
            raise ConfigError(
                "input topic file must be a path",
                details={"stage": INPUT_TOPIC_STAGE_NAME},
            )
        return _read_file(path), "file", str(path)

    for key in _SOURCE_KEYS["research"]:
        value = options.get(key)
        if value is None:
            continue
        path = _existing_file(value, context)
        if path is not None:
            return _read_file(path), "research", str(path)
        if isinstance(value, (str, Path)) and Path(value).suffix.lower() in {
            ".json",
            ".yaml",
            ".yml",
            ".txt",
            ".md",
        }:
            raise ConfigError(
                f"research output file not found: {value}",
                details={
                    "stage": INPUT_TOPIC_STAGE_NAME,
                    "path": str(value),
                },
            )
        return value, "research", key

    facts_value = _pick_option(options, _SOURCE_KEYS["facts"])
    if facts_value is not None:
        return facts_value, "input", "facts"

    text_value = _pick_option(options, _SOURCE_KEYS["text"])
    if text_value is not None:
        path = _existing_file(text_value, context)
        if path is not None:
            return _read_file(path), "file", str(path)
        return text_value, "cli", "text"

    topic_value = _pick_option(options, _SOURCE_KEYS["topic"])
    if topic_value is not None:
        path = _existing_file(topic_value, context)
        if path is not None:
            return _read_file(path), "file", str(path)
        return topic_value, "cli", "topic"

    for path in _candidate_files(context):
        if path.is_file():
            return _read_file(path), "research", str(path)
    return None, "", ""


def resolve_input_topic(
    value: Any = None,
    *,
    context: StageContext | None = None,
    title: str | None = None,
    summary: str | None = None,
    mode: str | None = None,
    origin: str = "",
    source: str = "",
) -> ResolvedTopic:
    """Normalize *value* into a :class:`ResolvedTopic` without writing files."""
    facts = _iter_facts(value, source=source)
    resolved_title = _text(title) or _title_from(value, facts)
    if not resolved_title and facts:
        resolved_title = _title_from(facts[0], facts)
    if not resolved_title:
        raise ConfigError(
            "input topic needs a topic, a file or research output",
            details={"stage": INPUT_TOPIC_STAGE_NAME},
        )
    if not facts:
        facts = [_fact_payload(resolved_title)]
    resolved_summary = _text(summary) or _summary_from(value)
    resolved_mode = _text(mode) or _mode_from(value)
    urls: list[str] = []
    for url in _clean_urls(_value_of(value, _URL_KEYS), source=source):
        if url not in urls:
            urls.append(url)
    for fact in facts:
        for url in fact.get("source_urls", ()):
            if url not in urls:
                urls.append(url)
    return ResolvedTopic(
        title=resolved_title,
        summary=resolved_summary,
        facts=tuple(facts),
        source_urls=tuple(urls),
        mode=resolved_mode,
        origin=origin,
        source=source,
        metadata={
            "fact_count": len(facts),
            "source_urls": list(urls),
        },
    )


class InputTopicStage(Stage):
    """Build the run's topic from CLI text, a file or research output.

    ``options`` may carry ``topic``/``text``/``title`` for command-line input,
    ``file``/``topic_file`` for a text, JSON or YAML file, ``facts``/``claims``
    for an inline fact list, or ``research``/``research_output`` for the
    research stage result. When none is supplied, the stage looks for research
    and topic files under the run directory. The result is written as
    ``topic.json`` and costs nothing.
    """

    name = INPUT_TOPIC_STAGE_NAME
    artifacts: tuple[str, ...] = (DEFAULT_TOPIC_FILENAME,)

    def __init__(
        self,
        name: str | None = None,
        *,
        options: Mapping[str, Any] | None = None,
        topic: Any = None,
        title: str | None = None,
        text: Any = None,
        content: Any = None,
        body: Any = None,
        query: Any = None,
        file: str | Path | None = None,
        topic_file: str | Path | None = None,
        input_file: str | Path | None = None,
        source_file: str | Path | None = None,
        path: str | Path | None = None,
        facts: Any = None,
        fact_list: Any = None,
        claims: Any = None,
        research: Any = None,
        research_output: Any = None,
        topic_input: Any = None,
        summary: str | None = None,
        mode: str | None = None,
        filename: str | None = None,
        output: str | None = None,
        **kwargs: Any,
    ) -> None:
        merged = dict(options or {})
        supplied = {
            "topic": topic,
            "title": title,
            "text": text,
            "content": content,
            "body": body,
            "query": query,
            "file": file,
            "topic_file": topic_file,
            "input_file": input_file,
            "source_file": source_file,
            "path": path,
            "facts": facts,
            "fact_list": fact_list,
            "claims": claims,
            "research": research,
            "research_output": research_output,
            "topic_input": topic_input,
            "summary": summary,
            "mode": mode,
            "filename": filename,
            "output": output,
        }
        for key, value in supplied.items():
            if value is not None:
                merged[key] = value
        merged.update(kwargs)
        self.filename = _text(
            merged.get("filename")
            or merged.get("topic_filename")
            or merged.get("output")
            or merged.get("output_filename")
            or DEFAULT_TOPIC_FILENAME
        )
        if not self.filename:
            self.filename = DEFAULT_TOPIC_FILENAME
        self.artifacts = (self.filename,)
        declared_inputs: list[str] = []
        for key in _SOURCE_KEYS["file"]:
            value = merged.get(key)
            if isinstance(value, (str, Path)) and str(value).strip():
                declared = str(Path(str(value)).expanduser())
                if declared not in declared_inputs:
                    declared_inputs.append(declared)
        super().__init__(
            name=name or INPUT_TOPIC_STAGE_NAME,
            options=merged,
            inputs=declared_inputs,
        )
        self.resolved: ResolvedTopic | None = None
        self.topic: Topic | None = None
        self.facts: list[dict[str, Any]] = []
        self.payload: dict[str, Any] = {}

    def _title_option(self) -> str:
        return _text(_pick_option(self.options, _SOURCE_KEYS["title"]))

    def resolve(self, context: StageContext | None = None) -> ResolvedTopic:
        """Resolve the configured input into a normalized topic."""
        value, origin, source = _resolve_source(self.options, context)
        title = self._title_option()
        if value is None and title:
            value = title
            origin = origin or "cli"
            source = source or "title"
        resolved = resolve_input_topic(
            value,
            context=context,
            title=title or None,
            summary=_text(_pick_option(self.options, ("summary",))),
            mode=_text(_pick_option(self.options, ("mode", "topic_mode"))),
            origin=origin,
            source=source,
        )
        first = _selected_first(value)
        if first is not value:
            resolved = resolve_input_topic(
                first,
                context=context,
                title=title or resolved.title,
                summary=_text(_pick_option(self.options, ("summary",)))
                or resolved.summary,
                mode=_text(_pick_option(self.options, ("mode", "topic_mode")))
                or resolved.mode,
                origin=origin,
                source=source,
            )
        return resolved

    build = resolve
    load = resolve

    async def run(self, context: StageContext) -> StageResult:
        resolved = self.resolve(context)
        self.resolved = resolved
        self.topic = resolved.topic
        self.facts = [dict(fact) for fact in resolved.facts]
        payload = resolved.to_payload()
        self.payload = payload
        relative = self.filename
        path = context.run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        text = canonical_json(payload, indent=2) + "\n"
        path.write_text(text, encoding="utf-8")
        facts_relative = DEFAULT_FACT_LIST_FILENAME
        facts_path = context.run_dir / facts_relative
        source_path = Path(resolved.source).expanduser()
        if not source_path.is_absolute():
            source_path = context.run_dir / source_path
        if (
            facts_relative != relative
            and facts_path != path
            and facts_path != source_path
        ):
            facts_path.parent.mkdir(parents=True, exist_ok=True)
            facts_path.write_text(text, encoding="utf-8")
        else:
            facts_relative = relative
        metadata = {
            "title": resolved.title,
            "summary": resolved.summary,
            "fact_count": resolved.fact_count,
            "source_urls": list(resolved.source_urls),
            "origin": resolved.origin,
            "source": resolved.source,
            "mode": resolved.mode,
            "topic_file": relative,
            "facts_file": facts_relative,
        }
        return StageResult(
            run_id=context.run.id if context.run is not None else "",
            stage=self.name,
            status=StageState.SUCCEEDED,
            artifacts=[relative],
            cost_usd=0.0,
            metadata=metadata,
        )


InputStage = InputTopicStage
InputTopic = InputTopicStage
TopicStage = InputTopicStage
TopicInputStage = InputTopicStage
TopicInput = ResolvedTopic
input_topic_stage = InputTopicStage
input_topic = InputTopicStage


def build_input_topic_stage(
    *,
    options: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> InputTopicStage:
    """Build an :class:`InputTopicStage` from recipe options and overrides."""
    return InputTopicStage(options=options, **kwargs)


async def run_input_topic_stage(
    context: StageContext,
    *,
    stage: InputTopicStage | None = None,
    **options: Any,
) -> StageResult:
    """Build a stage unless one is supplied, then run it asynchronously."""
    if stage is not None and options:
        raise ConfigError("pass either a stage or options, not both")
    active = stage if stage is not None else InputTopicStage(**options)
    return await active.run(context)
