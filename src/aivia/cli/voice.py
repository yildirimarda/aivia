"""``aivia voice``: run the audio stages for an existing run.

The command reuses the run's format recipe, selects its ``voice``, ``subtitle``
and ``timeline`` stages and executes them in recipe order against the existing
run directory. A language filter narrows every selected stage to the requested
languages, so one locale can be voiced and subtitled without paying for every
configured language again. The report prints the total cost and the measured
duration per language, read from the stage metadata or the timeline file the
voice stage writes.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError

import aivia.cli.run as run_command
from aivia.cli.registry import register_command
from aivia.cli.run import build_stages
from aivia.core import paths
from aivia.core.config.database import DatabaseConfig
from aivia.core.config.formats import FormatRecipeConfig, FormatsConfig
from aivia.core.config.resolve import resolve_config
from aivia.core.cost import CostTracker
from aivia.core.db import RunRepository, create_database_engine
from aivia.core.dry_run import apply_dry_run
from aivia.core.errors import ConfigError, ValidationError
from aivia.core.models import Run
from aivia.core.orchestrator import OrchestrationResult, Orchestrator
from aivia.core.stages import Stage
from aivia.llm.languages import normalize_language_code

__all__ = ["configure", "run_voice"]

VOICE_STAGE_NAMES = frozenset({"voice", "subtitle", "timeline"})
"""The recipe stages the voice command executes, in recipe order."""

_ORIGINAL_BUILD_STAGES = build_stages


def _build_stages(recipe: FormatRecipeConfig) -> list[Stage]:
    """Use a local override when a test or module replaced this function."""
    if build_stages is not _ORIGINAL_BUILD_STAGES:
        return build_stages(recipe)
    return run_command.build_stages(recipe)


def _configure_resolution(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        metavar="PATH",
        help="config file to merge instead of the default location",
    )
    parser.add_argument(
        "--env",
        "--environment",
        dest="environment",
        metavar="NAME",
        help="merge config.<NAME>.yaml beside the user config",
    )
    parser.add_argument(
        "--profile",
        metavar="NAME",
        help="merge the overrides of a configured profile",
    )
    parser.add_argument(
        "--set",
        dest="set_values",
        action="append",
        metavar="KEY=VALUE",
        help="override a dotted key; repeatable and applied after the profile",
    )


def configure(parser: argparse.ArgumentParser) -> None:
    """Add the voice command's options."""
    parser.add_argument(
        "run_id",
        nargs="?",
        metavar="RUN_ID",
        help="existing run whose audio stages to run",
    )
    parser.add_argument(
        "--run",
        "--run-id",
        dest="run_option",
        metavar="RUN_ID",
        help="existing run whose audio stages to run",
    )
    parser.add_argument(
        "--language",
        "--languages",
        dest="languages",
        action="append",
        metavar="CODE",
        help="language to voice; repeatable or comma separated",
    )
    parser.add_argument(
        "--database",
        "--db",
        dest="database",
        metavar="PATH",
        help="database file to update instead of the one in the user data directory",
    )
    _configure_resolution(parser)


def _run_id(args: argparse.Namespace) -> str:
    positional = (getattr(args, "run_id", None) or "").strip()
    option = (getattr(args, "run_option", None) or "").strip()
    return positional or option


def _load_config(args: argparse.Namespace) -> object:
    resolved = resolve_config(
        config_file=args.config,
        environment=args.environment,
        profile=args.profile,
        preset=args.preset,
        set_values=args.set_values,
    )
    config = resolved.to_app_config()
    if getattr(args, "dry_run", False):
        config = apply_dry_run(config)
    return config


def _database_path(args: argparse.Namespace, config: object) -> Path:
    explicit = getattr(args, "database", None)
    if explicit:
        return Path(explicit).expanduser()
    database = getattr(config, "database", None)
    if not isinstance(database, DatabaseConfig):
        database = None
    filename = database.filename if database is not None else DatabaseConfig().filename
    return paths.resolve_paths(args.config).data_dir / filename


def _recipe(config: object, run: Run) -> FormatRecipeConfig:
    section = getattr(config, "formats", None)
    formats = section.formats if isinstance(section, FormatsConfig) else {}
    name = run.format.strip()
    if not name:
        raise ConfigError(f"run has no format: {run.id}")
    recipe = formats.get(name)
    if recipe is None:
        available = ", ".join(sorted(formats)) or "none"
        raise ConfigError(
            f"unknown format: {name} (available: {available})",
            details={"format": name, "available": sorted(formats)},
        )
    return recipe


def _voice_stages(recipe: FormatRecipeConfig) -> list[Stage]:
    stages = [
        stage for stage in _build_stages(recipe) if stage.name in VOICE_STAGE_NAMES
    ]
    if not any(stage.name == "voice" for stage in stages):
        raise ConfigError(
            "format has no voice stage",
            details={"stages": list(recipe.stages)},
        )
    return stages


def _available_languages(config: object) -> dict[str, Any]:
    section = getattr(config, "languages", None)
    languages = getattr(section, "languages", None)
    return dict(languages) if isinstance(languages, Mapping) else {}


def _requested_languages(args: argparse.Namespace, config: object) -> list[str]:
    raw: list[str] = []
    for value in getattr(args, "languages", None) or ():
        raw.extend(part.strip() for part in str(value).split(",") if part.strip())
    if not raw:
        return []
    try:
        codes: list[str] = []
        for value in raw:
            code = normalize_language_code(value)
            if code not in codes:
                codes.append(code)
    except (ValidationError, ValueError) as exc:
        raise ConfigError(str(exc), details={"languages": raw}) from exc

    available = _available_languages(config)
    unknown = [code for code in codes if available and code not in available]
    if unknown:
        names = ", ".join(sorted(available))
        raise ConfigError(
            f"unknown language(s): {', '.join(unknown)} (available: {names})",
            details={"languages": unknown, "available": sorted(available)},
        )
    return codes


def _apply_language_filter(stage: Stage, languages: list[str]) -> None:
    if not languages:
        return
    stage.options["languages"] = list(languages)
    if len(languages) == 1:
        stage.options["language"] = languages[0]
    else:
        stage.options.pop("language", None)


def _stage_languages(stage: Stage) -> list[str]:
    raw = stage.options.get("languages")
    if raw is None:
        raw = stage.options.get("language")
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, (list, tuple)):
        values = [str(value) for value in raw]
    else:
        return []
    languages: list[str] = []
    for value in values:
        try:
            code = normalize_language_code(value)
        except (ValidationError, ValueError):
            continue
        if code not in languages:
            languages.append(code)
    return languages


def _execute(
    engine: Engine,
    run: Run,
    config: object,
    stages: list[Stage],
    *,
    data_dir: Path,
    tracker: CostTracker,
) -> OrchestrationResult:
    return Orchestrator(
        stages=stages,
        run=run.id,
        repository=engine,
        run_dir=run.run_dir or None,
        data_dir=data_dir,
        config=config,
        cost=tracker,
        raise_on_failure=False,
    ).execute_sync()


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _metadata_durations(result: Any) -> tuple[dict[str, float], float | None]:
    durations: dict[str, float] = {}
    total: float | None = None
    for stage_result in getattr(result, "stages", ()):
        metadata = getattr(stage_result, "metadata", None)
        if not isinstance(metadata, Mapping):
            continue
        raw = metadata.get("durations")
        if isinstance(raw, Mapping):
            for language, value in raw.items():
                number = _number(value)
                if number is not None:
                    durations[str(language)] = number
        for key in ("duration_seconds", "duration", "total_duration_seconds"):
            number = _number(metadata.get(key))
            if number is not None:
                total = number if total is None else max(total, number)
    return durations, total


def _timeline_source(stages: list[Stage], result: Any, run: Run) -> str:
    for stage_result in getattr(result, "stages", ()):
        metadata = getattr(stage_result, "metadata", None)
        if not isinstance(metadata, Mapping):
            continue
        raw = metadata.get("timeline")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    for stage_result in getattr(result, "stages", ()):
        artifacts = getattr(stage_result, "artifacts", None)
        for artifact in artifacts if isinstance(artifacts, list) else ():
            if isinstance(artifact, str) and Path(artifact).name == "timeline.json":
                return artifact
    for stage in stages:
        raw = stage.options.get("timeline_filename")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    if run.run_dir and (Path(run.run_dir) / "timeline.json").is_file():
        return "timeline.json"
    return ""


def _timeline_durations(path: Path) -> tuple[dict[str, float], float | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, None
    if not isinstance(payload, Mapping):
        return {}, None

    durations: dict[str, float] = {}
    entries = payload.get("languages")
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            language = entry.get("language")
            code = language.strip() if isinstance(language, str) else ""
            ends: list[float] = []
            voices = entry.get("voices")
            for voice in voices if isinstance(voices, list) else ():
                if not isinstance(voice, Mapping):
                    continue
                end = _number(voice.get("end_seconds"))
                if end is None:
                    end = _number(voice.get("end"))
                if end is None:
                    duration = _number(voice.get("duration_seconds"))
                    if duration is None:
                        duration = _number(voice.get("duration"))
                    start = _number(voice.get("start_seconds"))
                    if start is None:
                        start = _number(voice.get("start")) or 0.0
                    if duration is not None:
                        end = start + duration
                if end is not None:
                    ends.append(end)
            if code and ends:
                durations[code] = max(ends)

    total = _number(payload.get("duration_seconds"))
    if total is None:
        total = _number(payload.get("duration"))
    return durations, total


def _durations(
    stages: list[Stage],
    result: Any,
    run: Run,
) -> tuple[dict[str, float], float | None, str]:
    durations, total = _metadata_durations(result)
    timeline = _timeline_source(stages, result, run)
    if timeline and run.run_dir:
        from_timeline, timeline_total = _timeline_durations(
            Path(run.run_dir) / timeline
        )
        for language, value in from_timeline.items():
            durations.setdefault(language, value)
        if total is None:
            total = timeline_total
    if total is None and durations:
        total = max(durations.values())
    return durations, total, timeline


def _report_languages(stages: list[Stage], result: Any) -> list[str]:
    languages: list[str] = []
    for stage in stages:
        for code in _stage_languages(stage):
            if code not in languages:
                languages.append(code)
    if languages:
        return languages
    for stage_result in getattr(result, "stages", ()):
        metadata = getattr(stage_result, "metadata", None)
        if not isinstance(metadata, Mapping):
            continue
        raw = metadata.get("languages")
        if isinstance(raw, (list, tuple)):
            for value in raw:
                if isinstance(value, str) and value.strip() not in languages:
                    languages.append(value.strip())
    return languages


def _report(
    run: Run,
    stages: list[Stage],
    result: Any,
    events: list[Any],
) -> tuple[str, float]:
    names = {stage.name for stage in stages}
    cost = sum(
        float(getattr(stage_result, "cost_usd", 0.0) or 0.0)
        for stage_result in getattr(result, "stages", ())
        if getattr(stage_result, "stage", "") in names
    )
    if cost == 0.0 and events:
        cost = sum(float(event.amount_usd) for event in events)

    durations, total, timeline = _durations(stages, result, run)
    languages = _report_languages(stages, result)

    lines = [f"Run: {run.id}", "Stages: " + ", ".join(stage.name for stage in stages)]
    for stage in stages:
        lines.append(f"Stage: {stage.name}")
    if languages:
        lines.append("Languages: " + ", ".join(languages))
    lines.append(f"Cost: ${cost:.6f}")
    if durations:
        lines.append(
            "Durations: "
            + ", ".join(f"{code} {value:.3f}s" for code, value in durations.items())
        )
    lines.append(f"Duration: {total or 0.0:.3f}s")
    if timeline:
        lines.append(f"Timeline: {timeline}")
    return "\n".join(lines), cost


def run_voice(args: argparse.Namespace) -> int:
    """Run the voice, subtitle and timeline stages for an existing run."""
    run_id = _run_id(args)
    if not run_id:
        print("error: a run id is required", file=sys.stderr)
        return 2

    try:
        config = _load_config(args)
        languages = _requested_languages(args, config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    database_path = _database_path(args, config)
    if not database_path.exists():
        print(f"error: database not found: {database_path}", file=sys.stderr)
        return 1

    engine = create_database_engine(path=database_path)
    tracker = CostTracker(engine)
    try:
        run = RunRepository(engine).get(run_id)
        if run is None:
            print(f"error: run not found: {run_id}", file=sys.stderr)
            return 1

        try:
            stages = _voice_stages(_recipe(config, run))
            for stage in stages:
                _apply_language_filter(stage, languages)
        except ConfigError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        before = {event.id for event in tracker.events(run_id=run.id)}
        result = _execute(
            engine,
            run,
            config,
            stages,
            data_dir=database_path.parent,
            tracker=tracker,
        )
        events = [
            event for event in tracker.events(run_id=run.id) if event.id not in before
        ]
        names = {stage.name for stage in stages}
        stage_results = [item for item in result.stages if item.stage in names]
        if not stage_results:
            print(
                f"error: voice stages produced no result for run {run.id}",
                file=sys.stderr,
            )
            return 1

        report, _cost = _report(run, stages, result, events)
        print(report)
        if result.failed:
            print(
                f"error: run {run.id} failed at stage "
                f"{result.failed_stage!r}: {result.error_message}",
                file=sys.stderr,
            )
            return 1
        return 0
    except SQLAlchemyError as exc:
        print(f"error: cannot run the voice stages: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()


register_command(
    "voice",
    run_voice,
    help="Run the voice, subtitle and timeline stages for an existing run",
    configure=configure,
)
