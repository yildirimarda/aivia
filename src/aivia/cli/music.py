"""``aivia music``: select a background music bed and report its rights.

The command reads the ``music`` config section and lets a few flags override the
source, library, mood and ducking settings. Without a run id it selects the best
local library track and prints its source, license and ducking settings. With a
run id it runs the recipe's ``music`` stage against the run timeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sqlalchemy.exc import SQLAlchemyError

import aivia.cli.run as run_command
from aivia.audio.music_select import (
    BackgroundMusic,
    MusicSelectStage,
    select_library_music,
)
from aivia.cli.registry import register_command
from aivia.cli.run import build_stages
from aivia.core import paths
from aivia.core.config.database import DatabaseConfig
from aivia.core.config.formats import FormatRecipeConfig, FormatsConfig
from aivia.core.config.music import MusicConfig
from aivia.core.config.resolve import resolve_config
from aivia.core.cost import CostTracker
from aivia.core.db import RunRepository, create_database_engine
from aivia.core.errors import ConfigError
from aivia.core.models import Run
from aivia.core.orchestrator import OrchestrationResult, Orchestrator
from aivia.core.stages import Stage
from aivia.media.audio_mix import DuckingSettings
from aivia.media.music_library import MusicLibraryError

__all__ = [
    "configure",
    "run_music",
    "select_music_bed",
]

MUSIC_STAGE_NAME = "music"
"""The recipe stage this command runs when a run id is supplied."""

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
    """Add the music command's options."""
    parser.add_argument(
        "run_id",
        nargs="?",
        metavar="RUN_ID",
        help="existing run whose music stage to run",
    )
    parser.add_argument(
        "--run",
        "--run-id",
        dest="run_option",
        metavar="RUN_ID",
        help="existing run whose music stage to run",
    )
    parser.add_argument(
        "--source",
        metavar="MODE",
        help="where the bed may come from: auto, library or provider",
    )
    parser.add_argument(
        "--library",
        "--library-path",
        dest="library",
        metavar="PATH",
        help="music library manifest to select from",
    )
    parser.add_argument(
        "--mood",
        metavar="NAME",
        help="mood to match against the library or provider prompt",
    )
    parser.add_argument(
        "--genre",
        metavar="NAME",
        help="genre to match against the library or provider prompt",
    )
    parser.add_argument(
        "--tag",
        "--tags",
        dest="tags",
        action="append",
        metavar="NAME",
        help="style tag to match; repeatable or comma separated",
    )
    parser.add_argument(
        "--tempo",
        "--tempo-bpm",
        dest="tempo_bpm",
        type=float,
        metavar="BPM",
        help="preferred tempo in beats per minute",
    )
    parser.add_argument(
        "--duration",
        "--duration-seconds",
        dest="duration_seconds",
        type=float,
        metavar="SECONDS",
        help="requested bed duration",
    )
    parser.add_argument(
        "--match-duration",
        action="store_true",
        default=None,
        help="reject library tracks shorter than the requested duration",
    )
    parser.add_argument(
        "--commercial-only",
        action="store_true",
        default=None,
        help="only select a track whose license allows commercial use",
    )
    parser.add_argument(
        "--ducking",
        dest="ducking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable or disable voice ducking",
    )
    parser.add_argument(
        "--duck",
        dest="ducking",
        action="store_true",
        default=None,
        help="alias for --ducking",
    )
    parser.add_argument(
        "--no-duck",
        dest="ducking",
        action="store_false",
        default=None,
        help="alias for --no-ducking",
    )
    parser.add_argument(
        "--threshold-db",
        "--ducking-threshold",
        dest="ducking_threshold_db",
        type=float,
        metavar="DB",
        help="sidechain threshold where ducking starts",
    )
    parser.add_argument(
        "--ratio",
        "--ducking-ratio",
        dest="ducking_ratio",
        type=float,
        metavar="RATIO",
        help="sidechain compression ratio",
    )
    parser.add_argument(
        "--attack-ms",
        "--ducking-attack",
        dest="ducking_attack_ms",
        type=float,
        metavar="MS",
        help="sidechain attack in milliseconds",
    )
    parser.add_argument(
        "--release-ms",
        "--ducking-release",
        dest="ducking_release_ms",
        type=float,
        metavar="MS",
        help="sidechain release in milliseconds",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="print the selection as JSON instead of text",
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
    return resolved.to_app_config()


def _music_config(config: Any) -> MusicConfig:
    section = getattr(config, "music", None)
    if isinstance(section, MusicConfig):
        return section
    if isinstance(section, Mapping):
        return MusicConfig.from_mapping(section)
    if isinstance(config, Mapping):
        raw = config.get("music")
        if isinstance(raw, MusicConfig):
            return raw
        if isinstance(raw, Mapping):
            return MusicConfig.from_mapping(raw)
    return MusicConfig()


def _database_path(args: argparse.Namespace, config: Any) -> Path:
    explicit = getattr(args, "database", None)
    if explicit:
        return Path(explicit).expanduser()
    database = getattr(config, "database", None)
    if not isinstance(database, DatabaseConfig):
        database = None
    filename = database.filename if database is not None else DatabaseConfig().filename
    return paths.resolve_paths(args.config).data_dir / filename


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


def _music_stage(recipe: FormatRecipeConfig) -> Stage:
    for stage in _build_stages(recipe):
        if stage.name == MUSIC_STAGE_NAME:
            return stage
    raise ConfigError(
        "format has no music stage",
        details={"stages": list(recipe.stages)},
    )


def _tags(value: Any) -> list[str]:
    tags: list[str] = []
    for item in value or ():
        for part in str(item).split(","):
            tag = part.strip()
            if tag and tag not in tags:
                tags.append(tag)
    return tags


def _source(value: Any, fallback: str) -> str:
    if value is None:
        return fallback
    return str(MusicConfig.from_mapping({"source": value}).source)


def _ducking(section: MusicConfig, args: argparse.Namespace) -> DuckingSettings:
    values = section.ducking.model_dump()
    if getattr(args, "ducking", None) is not None:
        values["enabled"] = bool(args.ducking)
    for name in ("threshold_db", "ratio", "attack_ms", "release_ms"):
        value = getattr(args, f"ducking_{name}", None)
        if value is not None:
            values[name] = value
    return DuckingSettings.from_value(values)


def _options(args: argparse.Namespace, section: MusicConfig) -> dict[str, Any]:
    library = (
        args.library
        if getattr(args, "library", None) is not None
        else section.library_path
    )
    options: dict[str, Any] = {
        "source": _source(getattr(args, "source", None), str(section.source)),
        "mood": (
            args.mood if getattr(args, "mood", None) is not None else section.mood
        ),
        "genre": getattr(args, "genre", None) or "",
        "tags": _tags(getattr(args, "tags", None)),
        "volume_db": section.volume_db,
        "ducking": _ducking(section, args),
    }
    if library:
        options["library"] = library
    if getattr(args, "tempo_bpm", None) is not None:
        options["tempo_bpm"] = args.tempo_bpm
    if getattr(args, "duration_seconds", None) is not None:
        options["duration_seconds"] = args.duration_seconds
    if getattr(args, "match_duration", None) is not None:
        options["match_duration"] = bool(args.match_duration)
    if getattr(args, "commercial_only", None) is not None:
        options["commercial_only"] = bool(args.commercial_only)
    return options


def _select_library(options: Mapping[str, Any]) -> BackgroundMusic:
    library = options.get("library")
    if not library:
        raise ConfigError(
            "music selection needs a local library; set music.library_path or "
            "pass --library"
        )
    if options.get("source") == "provider":
        raise ConfigError(
            "provider music needs a run id; pass RUN_ID or use a local library"
        )
    try:
        return select_library_music(
            library,
            mood=options.get("mood"),
            genre=options.get("genre"),
            tags=options.get("tags"),
            tempo_bpm=options.get("tempo_bpm"),
            duration_seconds=options.get("duration_seconds"),
            match_duration=bool(options.get("match_duration", False)),
            commercial_only=bool(options.get("commercial_only", False)),
        )
    except MusicLibraryError as exc:
        raise ConfigError(
            "no local music track matched the request",
            details={"error": exc.message},
        ) from exc


def select_music_bed(
    config: Any,
    *,
    source: Any = None,
    library: Any = None,
    mood: Any = None,
    genre: Any = None,
    tags: Any = None,
    tempo_bpm: float | None = None,
    duration_seconds: float | None = None,
    match_duration: bool | None = None,
    commercial_only: bool | None = None,
    ducking: DuckingSettings | Mapping[str, Any] | None = None,
) -> tuple[BackgroundMusic, DuckingSettings]:
    """Select a local bed from *config* and return it with the ducking settings."""
    section = _music_config(config)
    options: dict[str, Any] = {
        "source": _source(source, str(section.source)),
        "mood": section.mood if mood is None else mood,
        "genre": genre or "",
        "tags": _tags(tags),
        "library": section.library_path if library is None else library,
    }
    if tempo_bpm is not None:
        options["tempo_bpm"] = tempo_bpm
    if duration_seconds is not None:
        options["duration_seconds"] = duration_seconds
    if match_duration is not None:
        options["match_duration"] = bool(match_duration)
    if commercial_only is not None:
        options["commercial_only"] = bool(commercial_only)
    settings = DuckingSettings.from_value(
        section.ducking.model_dump() if ducking is None else ducking
    )
    return _select_library(options), settings


def _payload(choice: BackgroundMusic, ducking: DuckingSettings) -> dict[str, Any]:
    settings = ducking.model_dump(mode="json")
    settings["applied"] = ducking.enabled
    return {
        "source": choice.origin,
        "path": choice.source,
        "id": choice.id,
        "title": choice.title,
        "mood": choice.mood,
        "genre": choice.genre,
        "duration_seconds": choice.duration_seconds,
        "license": choice.license.model_dump(mode="json"),
        "license_name": choice.license_name,
        "attribution": choice.attribution,
        "requires_attribution": choice.requires_attribution,
        "commercial_cleared": choice.is_cleared,
        "ducking": settings,
        "cost_usd": choice.cost_usd,
    }


def _ducking_line(ducking: Mapping[str, Any]) -> str:
    enabled = bool(ducking.get("enabled", False))
    parts = [
        f"threshold {float(ducking.get('threshold_db', 0.0)):.1f} dB",
        f"ratio {float(ducking.get('ratio', 1.0)):g}",
        f"attack {float(ducking.get('attack_ms', 0.0)):.1f} ms",
        f"release {float(ducking.get('release_ms', 0.0)):.1f} ms",
    ]
    state = "enabled" if enabled else "disabled"
    return f"Ducking: {state} ({', '.join(parts)})"


def _report(payload: Mapping[str, Any], *, json_output: bool = False) -> None:
    if json_output:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(f"Source: {payload.get('source', '')}")
    print(f"Path: {payload.get('path', '')}")
    track = payload.get("id") or payload.get("track")
    if track:
        print(f"Track: {track}")
    if payload.get("title"):
        print(f"Title: {payload['title']}")
    print(f"License: {payload.get('license_name', '')}")
    print(f"Commercial use: {'yes' if payload.get('commercial_cleared') else 'no'}")
    if payload.get("attribution"):
        print(f"Attribution: {payload['attribution']}")
    ducking = payload.get("ducking")
    if isinstance(ducking, Mapping):
        print(_ducking_line(ducking))
        if "applied" in ducking:
            print(f"Ducking applied: {'yes' if ducking['applied'] else 'no'}")
        if "verified" in ducking:
            print(f"Ducking verified: {'yes' if ducking['verified'] else 'no'}")
    if payload.get("cost_usd") is not None:
        print(f"Cost: ${float(payload['cost_usd']):.6f}")


def _apply_options(stage: Stage, options: Mapping[str, Any]) -> Stage:
    merged = {**stage.options, **options}
    if isinstance(stage, MusicSelectStage):
        replacement = MusicSelectStage(name=stage.name, options=merged)
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


def _run_stage(
    args: argparse.Namespace, config: Any, options: Mapping[str, Any]
) -> int:
    run_id = _run_id(args)
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

        stage = _apply_options(_music_stage(_recipe(config, run)), options)
        result = _execute(
            engine,
            run,
            config,
            [stage],
            data_dir=database_path.parent,
            tracker=tracker,
        )
        stage_result = next(
            (item for item in result.stages if item.stage == stage.name),
            None,
        )
        if stage_result is None:
            print(
                f"error: music stage produced no result for run {run.id}",
                file=sys.stderr,
            )
            return 1
        payload = dict(stage_result.metadata or {})
        payload.setdefault("cost_usd", stage_result.cost_usd)
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
        print(f"error: cannot run the music stage: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()


def run_music(args: argparse.Namespace) -> int:
    """Select a background bed and print its source, license and ducking."""
    try:
        config = _load_config(args)
        section = _music_config(config)
        options = _options(args, section)
        run_id = _run_id(args)
        if run_id:
            return _run_stage(args, config, options)
        choice = _select_library(options)
        settings = DuckingSettings.from_value(options["ducking"])
        _report(
            _payload(choice, settings),
            json_output=bool(getattr(args, "json_output", False)),
        )
        return 0
    except (ConfigError, MusicLibraryError, ValueError, TypeError) as exc:
        message = exc.message if isinstance(exc, MusicLibraryError) else str(exc)
        print(f"error: {message}", file=sys.stderr)
        return 1


register_command(
    "music",
    run_music,
    help="Select a background music bed and print its source, license and ducking",
    configure=configure,
)
