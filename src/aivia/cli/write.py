"""``aivia write``: run the write stage for an existing run.

The command reuses the run's format recipe, selects its ``write`` stage and
executes only that stage against the existing run directory. A language filter
narrows the stage to the requested languages, so a run can be rewritten for one
locale without paying for every configured language again. The report prints
the stage cost and the input, output and total token counts.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError

import aivia.cli.run as run_command
from aivia.cli.registry import register_command
from aivia.cli.run import build_router, build_stages
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

__all__ = ["configure", "run_write"]

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
    """Add the write command's options."""
    parser.add_argument(
        "run_id",
        nargs="?",
        metavar="RUN_ID",
        help="existing run whose write stage to run",
    )
    parser.add_argument(
        "--run",
        "--run-id",
        dest="run_option",
        metavar="RUN_ID",
        help="existing run whose write stage to run",
    )
    parser.add_argument(
        "--language",
        "--languages",
        dest="languages",
        action="append",
        metavar="CODE",
        help="language to write; repeatable or comma separated",
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


def _write_stage(recipe: FormatRecipeConfig) -> Stage:
    for stage in _build_stages(recipe):
        if stage.name == "write":
            return stage
    raise ConfigError(
        "format has no write stage",
        details={"stages": list(recipe.stages)},
    )


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
    stage.preserve_artifacts = True
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
    stage: Stage,
    *,
    data_dir: Path,
    tracker: CostTracker,
) -> OrchestrationResult:
    return Orchestrator(
        stages=[stage],
        run=run.id,
        repository=engine,
        run_dir=run.run_dir or None,
        data_dir=data_dir,
        config=config,
        cost=tracker,
        router=build_router(config),
        raise_on_failure=False,
    ).execute_sync()


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _metadata_tokens(result: Any) -> tuple[int, int, int]:
    metadata = getattr(result, "metadata", None)
    if not isinstance(metadata, Mapping):
        return 0, 0, 0
    usage = metadata.get("usage")
    if isinstance(usage, Mapping):
        metadata = {**usage, **metadata}
    input_tokens = _as_int(metadata.get("input_tokens"))
    output_tokens = _as_int(metadata.get("output_tokens"))
    total_tokens = _as_int(metadata.get("total_tokens"))
    if total_tokens == 0 and (input_tokens or output_tokens):
        total_tokens = input_tokens + output_tokens
    return input_tokens, output_tokens, total_tokens


def _event_tokens(events: list[Any]) -> tuple[int, int, int]:
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    for event in events:
        if event.unit_kind != "token":
            continue
        metadata = event.metadata if isinstance(event.metadata, Mapping) else {}
        input_tokens += _as_int(metadata.get("input_tokens"))
        output_tokens += _as_int(metadata.get("output_tokens"))
        total_tokens += _as_int(event.units)
    return input_tokens, output_tokens, total_tokens


def _report(
    run: Run,
    stage: Stage,
    result: Any,
    events: list[Any],
) -> tuple[str, float, int, int, int]:
    input_tokens, output_tokens, total_tokens = _metadata_tokens(result)
    if total_tokens == 0:
        input_tokens, output_tokens, total_tokens = _event_tokens(events)

    cost = float(getattr(result, "cost_usd", 0.0) or 0.0)
    if cost == 0.0 and events:
        cost = sum(float(event.amount_usd) for event in events)

    languages = _stage_languages(stage)
    lines = [f"Run: {run.id}", f"Stage: {stage.name}"]
    if languages:
        lines.append("Languages: " + ", ".join(languages))
    lines.append(f"Cost: ${cost:.6f}")
    if total_tokens:
        lines.append(
            f"Tokens: {total_tokens} (input {input_tokens}, output {output_tokens})"
        )
    else:
        lines.append("Tokens: 0")
    return "\n".join(lines), cost, input_tokens, output_tokens, total_tokens


def run_write(args: argparse.Namespace) -> int:
    """Run the write stage for an existing run and print its cost and tokens."""
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
            stage = _write_stage(_recipe(config, run))
            _apply_language_filter(stage, languages)
        except ConfigError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        before = {event.id for event in tracker.events(run_id=run.id)}
        result = _execute(
            engine,
            run,
            config,
            stage,
            data_dir=database_path.parent,
            tracker=tracker,
        )
        events = [
            event for event in tracker.events(run_id=run.id) if event.id not in before
        ]
        stage_result = next(
            (item for item in result.stages if item.stage == stage.name),
            None,
        )
        if stage_result is None:
            print(
                f"error: write stage produced no result for run {run.id}",
                file=sys.stderr,
            )
            return 1

        report, _cost, _input, _output, _total = _report(
            run, stage, stage_result, events
        )
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
        print(f"error: cannot run the write stage: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()


register_command(
    "write",
    run_write,
    help="Run the write stage for an existing run, optionally filtered by language",
    configure=configure,
)
