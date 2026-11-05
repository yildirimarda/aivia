"""``aivia sfx``: place rule matched sound effects on an existing timeline.

The command reads the ``sfx`` config section for rule files, a default gain and
placement, lets flags override them, then runs the recipe's ``sfx`` stage for an
existing run. It prints every placement with its rule, beat, timing, gain and
source, plus the total cost and whether the timeline duration stayed unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from sqlalchemy.exc import SQLAlchemyError

import aivia.cli.run as run_command
from aivia.audio.sfx_stage import (
    DEFAULT_SFX_FILENAME,
    SFX_STAGE_NAME,
    SfxStage,
    load_sfx_rules,
)
from aivia.cli.registry import register_command
from aivia.cli.run import build_stages
from aivia.core import paths
from aivia.core.config.database import DatabaseConfig
from aivia.core.config.formats import FormatRecipeConfig, FormatsConfig
from aivia.core.config.providers import ProvidersConfig
from aivia.core.config.resolve import resolve_config
from aivia.core.config.sfx import SfxConfig
from aivia.core.cost import CostTracker
from aivia.core.db import RunRepository, create_database_engine
from aivia.core.dry_run import apply_dry_run
from aivia.core.errors import ConfigError
from aivia.core.models import Run
from aivia.core.orchestrator import OrchestrationResult, Orchestrator
from aivia.core.router import Router
from aivia.core.stages import Stage
from aivia.providers.provider_factory import build_providers
from aivia.providers.storage_factory import build_storage_provider

__all__ = [
    "configure",
    "run_sfx",
]

_RULE_KEYS = ("rules", "sound_effects", "effects", "cues")
_VOLUME_KEYS = ("volume_db", "volume", "gain_db", "gain")
_POSITION_KEYS = ("position", "placement", "anchor")

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
    """Add the sfx command's options."""
    parser.add_argument(
        "run_id",
        nargs="?",
        metavar="RUN_ID",
        help="existing run whose timeline receives the effects",
    )
    parser.add_argument(
        "--run",
        "--run-id",
        dest="run_option",
        metavar="RUN_ID",
        help="existing run whose timeline receives the effects",
    )
    parser.add_argument(
        "--rules",
        "--rule",
        "--rule-file",
        "--rule-files",
        dest="rules",
        action="append",
        metavar="PATH",
        help="sfx rule file to load; repeatable or comma separated",
    )
    parser.add_argument(
        "--volume-db",
        "--volume",
        "--gain-db",
        "--gain",
        dest="volume_db",
        type=float,
        metavar="DB",
        help="default gain for effects whose rule sets no volume",
    )
    parser.add_argument(
        "--placement",
        "--position",
        "--anchor",
        dest="placement",
        metavar="WHERE",
        help="default placement: start, center, end, word, offset or absolute",
    )
    parser.add_argument(
        "--timeline",
        "--timeline-file",
        dest="timeline",
        metavar="PATH",
        help="timeline file to edit instead of timeline.json in the run directory",
    )
    parser.add_argument(
        "--language",
        "--languages",
        dest="languages",
        action="append",
        metavar="CODE",
        help="language whose rules to place; repeatable or comma separated",
    )
    parser.add_argument(
        "--replace",
        dest="replace",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="replace existing sfx tracks or keep and append to them",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="print the placements as JSON instead of text",
    )
    parser.add_argument(
        "--database",
        "--db",
        dest="database",
        metavar="PATH",
        help="database file to read instead of the one in the user data directory",
    )
    _configure_resolution(parser)


def _run_id(args: argparse.Namespace) -> str:
    positional = (getattr(args, "run_id", None) or "").strip()
    option = (getattr(args, "run_option", None) or "").strip()
    return positional or option


def _load_config(args: argparse.Namespace) -> Any:
    resolved = resolve_config(
        config_file=args.config,
        environment=args.environment,
        profile=args.profile,
        preset=getattr(args, "preset", None),
        set_values=args.set_values,
    )
    config = resolved.to_app_config()
    if getattr(args, "dry_run", False):
        config = apply_dry_run(config)
    return config


def _sfx_config(config: Any) -> SfxConfig:
    section = getattr(config, "sfx", None)
    if isinstance(section, SfxConfig):
        return section
    if isinstance(section, Mapping):
        return SfxConfig.from_mapping(section)
    if isinstance(config, Mapping):
        raw = config.get("sfx")
        if isinstance(raw, SfxConfig):
            return raw
        if isinstance(raw, Mapping):
            return SfxConfig.from_mapping(raw)
    return SfxConfig()


def _database_path(args: argparse.Namespace, config: Any) -> Path:
    explicit = getattr(args, "database", None)
    if explicit:
        return Path(explicit).expanduser()
    database = getattr(config, "database", None)
    if not isinstance(database, DatabaseConfig):
        database = None
    filename = database.filename if database is not None else DatabaseConfig().filename
    return paths.resolve_paths(args.config).data_dir / filename


def _providers_config(config: Any) -> ProvidersConfig:
    section = getattr(config, "providers", None)
    if isinstance(section, ProvidersConfig):
        return section
    if isinstance(section, Mapping):
        return ProvidersConfig.from_mapping(section)
    if isinstance(config, Mapping):
        raw = config.get("providers")
        if isinstance(raw, ProvidersConfig):
            return raw
        if isinstance(raw, Mapping):
            return ProvidersConfig.from_mapping(raw)
    return ProvidersConfig()


def _build_router(config: Any) -> Router:
    """Build the configured media router for provider sound effects."""
    return Router(build_providers(_providers_config(config)))


def _recipe(config: Any, run: Run) -> FormatRecipeConfig:
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


def _sfx_stage(recipe: FormatRecipeConfig) -> Stage:
    for stage in _build_stages(recipe):
        if stage.name == SFX_STAGE_NAME:
            return stage
    raise ConfigError(
        "format has no sfx stage",
        details={"stages": list(recipe.stages)},
    )


def _values(args: argparse.Namespace, name: str) -> list[str]:
    values: list[str] = []
    for value in getattr(args, name, None) or ():
        for part in str(value).split(","):
            text = part.strip()
            if text and text not in values:
                values.append(text)
    return values


def _has_option(stage: Stage, names: Sequence[str]) -> bool:
    return any(stage.options.get(name) is not None for name in names)


def _load_rule_files(files: Sequence[str]) -> tuple[Any, ...]:
    rules: list[Any] = []
    for filename in files:
        rules.extend(load_sfx_rules(filename))
    if not rules:
        return ()
    return load_sfx_rules(rules)


def _with_placement(rules: Sequence[Any], placement: str | None) -> tuple[Any, ...]:
    if not placement:
        return tuple(rules)
    default = SfxConfig.from_mapping({"placement": placement}).placement.value
    adjusted: list[Any] = []
    for rule in rules:
        fields = cast(set[str], getattr(rule, "model_fields_set", set()))
        if "position" in fields or "start_seconds" in fields:
            adjusted.append(rule)
            continue
        copy = rule.model_copy(deep=True)
        object.__setattr__(copy, "position", default)
        adjusted.append(copy)
    return tuple(adjusted)


def _stage_options(
    args: argparse.Namespace,
    section: SfxConfig,
    stage: Stage,
) -> dict[str, Any]:
    options: dict[str, Any] = {}

    stage_placement = next(
        (
            str(stage.options[name])
            for name in _POSITION_KEYS
            if stage.options.get(name) is not None
        ),
        None,
    )
    explicit_placement = getattr(args, "placement", None)
    if explicit_placement is not None:
        placement = explicit_placement
        options["position"] = explicit_placement
    else:
        placement = stage_placement
        if placement is None:
            placement = section.placement.value
            options["position"] = placement

    rules = _values(args, "rules")
    if rules:
        options["rules"] = _with_placement(_load_rule_files(rules), placement)
    elif not _has_option(stage, _RULE_KEYS) and section.rule_files:
        options["rules"] = _with_placement(
            _load_rule_files(section.rule_files),
            placement,
        )

    volume = getattr(args, "volume_db", None)
    if volume is not None:
        options["volume_db"] = volume
    elif not _has_option(stage, _VOLUME_KEYS):
        options["volume_db"] = section.volume_db

    timeline = getattr(args, "timeline", None)
    if timeline:
        options["timeline"] = timeline

    languages = _values(args, "languages")
    if languages:
        options["languages"] = languages

    replace = getattr(args, "replace", None)
    if replace is not None:
        options["replace"] = bool(replace)
    return options


def _apply_options(stage: Stage, options: Mapping[str, Any]) -> Stage:
    if not options:
        return stage
    merged = {**stage.options, **options}
    if type(stage) is SfxStage:
        replacement = SfxStage(name=stage.name, options=merged, sfx=stage.sfx)
        replacement.version = stage.version
        replacement.inputs = stage.inputs
        replacement.retry_policy = stage.retry_policy
        return replacement
    stage.options.update(options)
    return stage


def _execute(
    engine: Any,
    run: Run,
    config: Any,
    stage: Stage,
    *,
    data_dir: Path,
    tracker: CostTracker,
    router: Router | None = None,
    storage: Any = None,
) -> OrchestrationResult:
    return Orchestrator(
        stages=[stage],
        run=run.id,
        repository=engine,
        run_dir=run.run_dir or None,
        data_dir=data_dir,
        config=config,
        cost=tracker,
        router=router,
        storage=storage,
        raise_on_failure=False,
    ).execute_sync()


def _manifest(run: Run, stage: Stage) -> dict[str, Any]:
    if not run.run_dir:
        return {}
    filename = stage.options.get("sfx_filename") or DEFAULT_SFX_FILENAME
    path = Path(run.run_dir) / str(filename)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _payload(
    run: Run,
    stage: Stage,
    result: Any,
) -> dict[str, Any]:
    stage_result = next(
        (item for item in result.stages if item.stage == stage.name),
        None,
    )
    metadata = getattr(stage_result, "metadata", None)
    payload = dict(metadata) if isinstance(metadata, Mapping) else {}
    manifest = _manifest(run, stage)
    for key, value in manifest.items():
        payload.setdefault(key, value)
    if stage_result is not None:
        payload.setdefault("cost_usd", getattr(stage_result, "cost_usd", 0.0))
    payload.setdefault("cost_usd", 0.0)
    payload.setdefault("timeline", stage.options.get("timeline") or "timeline.json")
    payload.setdefault(
        "manifest",
        stage.options.get("sfx_filename") or DEFAULT_SFX_FILENAME,
    )
    return payload


def _placements(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for key in ("tracks", "effects", "placements"):
        value = payload.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [item for item in value if isinstance(item, Mapping)]
    return []


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def _report(payload: Mapping[str, Any], *, json_output: bool = False) -> None:
    if json_output:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return

    placements = _placements(payload)
    count = payload.get("placements")
    if not isinstance(count, int):
        count = len(placements)
    print(f"Run: {payload.get('run', '')}".rstrip())
    print(f"Stage: {SFX_STAGE_NAME}")
    print(f"Rules: {int(_number(payload.get('rules'), 0))}")
    print(f"Placements: {count}")
    print(f"Effects: {count}")
    print(f"Calls: {int(_number(payload.get('calls'), 0))}")
    print(f"Cost: ${_number(payload.get('cost_usd')):.6f}")
    duration = payload.get("duration_seconds", payload.get("total_duration_seconds"))
    if duration is not None:
        print(f"Duration: {_number(duration):.3f}s")
    if "duration_unchanged" in payload:
        unchanged = bool(payload["duration_unchanged"])
        print(f"Duration unchanged: {'yes' if unchanged else 'no'}")
    if payload.get("timeline"):
        print(f"Timeline: {payload['timeline']}")
    if payload.get("manifest"):
        print(f"Manifest: {payload['manifest']}")

    for placement in placements:
        print(f"Placement: {placement.get('id', '')}")
        if placement.get("rule"):
            print(f"  Rule: {placement['rule']}")
        if placement.get("beat"):
            print(f"  Beat: {placement['beat']}")
        if placement.get("prompt"):
            print(f"  Prompt: {placement['prompt']}")
        if placement.get("source"):
            print(f"  Source: {placement['source']}")
        print(f"  Start: {_number(placement.get('start_seconds')):.3f}s")
        print(f"  End: {_number(placement.get('end_seconds')):.3f}s")
        print(f"  Duration: {_number(placement.get('duration_seconds')):.3f}s")
        print(f"  Volume: {_number(placement.get('volume_db')):.1f} dB")
        if placement.get("origin"):
            print(f"  Origin: {placement['origin']}")
        if placement.get("provider"):
            print(f"  Provider: {placement['provider']}")
        if placement.get("model"):
            print(f"  Model: {placement['model']}")
        print(f"  Effect cost: ${_number(placement.get('cost_usd')):.6f}")


def run_sfx(args: argparse.Namespace) -> int:
    """Place effects on an existing run timeline and report the result."""
    run_id = _run_id(args)
    if not run_id:
        print("error: a run id is required", file=sys.stderr)
        return 2

    try:
        config = _load_config(args)
        section = _sfx_config(config)
    except (ConfigError, ValueError, TypeError) as exc:
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
            base_stage = _sfx_stage(_recipe(config, run))
            stage = _apply_options(
                base_stage,
                _stage_options(args, section, base_stage),
            )
        except (ConfigError, ValueError, TypeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        result = _execute(
            engine,
            run,
            config,
            stage,
            data_dir=database_path.parent,
            tracker=tracker,
            router=_build_router(config),
            storage=build_storage_provider(
                getattr(config, "storage", None),
                data_dir=database_path.parent,
            ),
        )
        payload = _payload(run, stage, result)
        payload["run"] = run.id
        _report(payload, json_output=bool(getattr(args, "json_output", False)))
        if result.failed:
            print(
                f"error: run {run.id} failed at stage "
                f"{result.failed_stage!r}: {result.error_message}",
                file=sys.stderr,
            )
            return 1
        return 0
    except SQLAlchemyError as exc:
        print(f"error: cannot run the sfx stage: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()


register_command(
    "sfx",
    run_sfx,
    help="Place sound effects on an existing timeline and print placements and cost",
    configure=configure,
)
