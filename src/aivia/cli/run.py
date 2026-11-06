"""``aivia run``: create, execute or enqueue a format recipe and inspect it."""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
from pathlib import Path

from sqlalchemy import Connection, Engine
from sqlalchemy.exc import SQLAlchemyError

from aivia.audio.lyric_stage import ALIGN_STAGE_NAME, LyricAlignStage
from aivia.audio.music_select import MUSIC_STAGE_NAME, MusicSelectStage
from aivia.cli.confirm import confirm_spend, confirmation_required
from aivia.cli.registry import register_command
from aivia.core import paths
from aivia.core.config.budget import BudgetConfig
from aivia.core.config.capability import CapabilityChainConfig
from aivia.core.config.database import DatabaseConfig
from aivia.core.config.embeddings import EmbeddingsConfig
from aivia.core.config.formats import FormatRecipeConfig, OutputType
from aivia.core.config.llm import LLMConfig
from aivia.core.config.ops import JobsConfig
from aivia.core.config.providers import ProvidersConfig
from aivia.core.config.resolve import resolve_app_config, resolve_config
from aivia.core.cost import CostTracker
from aivia.core.db import (
    RunRepository,
    StageRunRepository,
    create_database_engine,
    upgrade_database,
)
from aivia.core.dry_run import apply_dry_run
from aivia.core.errors import ConfigError, QuotaError
from aivia.core.estimate import Estimator, RecipeEstimate, render_estimate
from aivia.core.idempotency import IdempotencyConflict, IdempotentRun, create_run_once
from aivia.core.jobs import JobQueue
from aivia.core.models import Job, Run, RunKind, RunState, StageResult
from aivia.core.models.provider import ProviderMeta
from aivia.core.orchestrator import OrchestrationResult, Orchestrator
from aivia.core.router import Router
from aivia.core.run_dirs import RunDirectories, RunDirectory
from aivia.core.stages import Stage
from aivia.core.toolchain import detect_toolchain
from aivia.formats.audio_only import (
    COVER_ART_STAGE_NAME,
    AudioOnlyStage,
    CoverArtStage,
)
from aivia.formats.byo import (
    ProvidedInputs,
    apply_provided_inputs,
    load_provided_inputs,
    provided_stage_options,
    script_pattern,
    seed_provided_inputs,
)
from aivia.formats.catalog import load_formats
from aivia.formats.image_post import CarouselStage, ImagePostStage
from aivia.formats.input_topic import INPUT_TOPIC_STAGE_NAME, InputTopicStage
from aivia.formats.quote_card import QuoteCardRenderStage
from aivia.formats.song_video import SONG_VIDEO_STAGE_NAME, SongVideoStage
from aivia.providers import docs as provider_docs
from aivia.providers.provider_factory import build_providers
from aivia.providers.registry import registry as provider_registry
from aivia.providers.storage_factory import build_storage_provider
from aivia.visuals.reuse import VISUAL_REUSE_STAGE_NAME, VisualReuseStage
from aivia.writing.stage import WRITE_STAGE_NAME, WriteStage

__all__ = [
    "build_router",
    "build_stages",
    "capability_configs",
    "configure",
    "provider_metadata",
    "run_run",
    "run_show",
]


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


def _configure_database(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--database",
        "--db",
        dest="database",
        metavar="PATH",
        help="database file to use instead of the one in the user data directory",
    )


def _configure_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--format",
        dest="format_name",
        metavar="NAME",
        help="format recipe to run or estimate",
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="print the stage cost estimate and exit without running anything",
    )
    parser.add_argument(
        "--text-file",
        "--text",
        "--topic-file",
        "--input-file",
        dest="text_file",
        metavar="PATH",
        help="read the topic from a text, JSON or YAML file instead of research",
    )
    parser.add_argument(
        "--script-file",
        "--script",
        dest="script_file",
        metavar="PATH",
        help="use a prepared JSON, YAML or text script instead of writing",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--enqueue",
        action="store_true",
        help="queue the run for a worker instead of executing it now",
    )
    mode.add_argument(
        "--execute",
        "--sync",
        "--wait",
        "--now",
        dest="execute",
        action="store_true",
        help="execute the run immediately (the default)",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="confirm spending above the configured threshold without prompting",
    )
    parser.add_argument(
        "--idempotency-key",
        "--idempotency",
        dest="idempotency_key",
        metavar="KEY",
        help="return the existing run when KEY was already used for this request",
    )
    _configure_database(parser)
    _configure_resolution(parser)
    parser.set_defaults(_run_handler=run_run)


def _configure_show(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "run_id",
        nargs="?",
        metavar="RUN_ID",
        help="run whose stages, cost and artifacts to show",
    )
    parser.add_argument(
        "--run",
        "--run-id",
        dest="run_option",
        metavar="RUN_ID",
        help="run whose stages, cost and artifacts to show",
    )
    _configure_database(parser)
    _configure_resolution(parser)
    parser.set_defaults(_run_handler=run_show)


def configure(parser: argparse.ArgumentParser) -> None:
    """Add the run command, its options and the ``show`` subcommand."""
    _configure_run_options(parser)
    subparsers = parser.add_subparsers(dest="run_command", metavar="COMMAND")
    show_parser = subparsers.add_parser(
        "show",
        help="Show a run's stages, cost and artifacts",
        description="Show a run's stages, cost and artifacts",
    )
    _configure_show(show_parser)


def _configured_provider_metadata(
    section: ProvidersConfig,
) -> dict[str, ProviderMeta]:
    metadata: dict[str, ProviderMeta] = {}
    for name, entry in section.providers.items():
        if not entry.enabled:
            continue
        provider_type = provider_registry.resolve(entry.type)
        static = provider_docs.static_metadata(provider_type)
        if static is None:
            raise ConfigError(
                f"provider {name!r} does not declare static metadata",
                details={"provider": name, "type": entry.type},
            )
        metadata[name] = static
    return metadata


def provider_metadata(config: object) -> dict[str, ProviderMeta]:
    """Return provider metadata keyed by configured name.

    Configured providers win. When no provider is configured at all, the
    registered provider types are used instead, so a fresh installation can
    still preview an estimate with the deterministic mock providers.
    """
    section = getattr(config, "providers", None)
    if isinstance(section, ProvidersConfig) and section.providers:
        return _configured_provider_metadata(section)

    metadata: dict[str, ProviderMeta] = {}
    for registration in provider_registry.registrations():
        static = provider_docs.static_metadata(registration.provider)
        if static is not None:
            metadata[registration.name] = static
    return metadata


def capability_configs(config: object) -> dict[str, CapabilityChainConfig]:
    """Return configured capability chains for the estimator.

    The LLM and embeddings sections own a provider chain but not the shared
    strategy fields, so each maps to a chain config with the default selection
    strategy. Embedding model prices become pricing overrides so the estimate
    uses the same table as the service. An empty chain and price table is
    omitted to keep the estimator's historical cheapest default.
    """
    configs: dict[str, CapabilityChainConfig] = {}
    llm = getattr(config, "llm", None)
    if isinstance(llm, LLMConfig) and llm.chain:
        configs["llm"] = CapabilityChainConfig(chain=llm.chain)
    embeddings = getattr(config, "embeddings", None)
    if isinstance(embeddings, EmbeddingsConfig) and (
        embeddings.chain or embeddings.pricing
    ):
        configs["embeddings"] = embeddings.to_capability_config()
    return configs


def _resolve_recipe(
    config: object,
    name: str,
    *,
    config_dir: str | Path | None = None,
    defaults_dir: str | Path | None = None,
) -> tuple[str, FormatRecipeConfig]:
    catalog = load_formats(
        config=config,
        config_dir=config_dir,
        defaults_dir=defaults_dir,
    )
    recipe = catalog.get(name)
    return name, recipe


def _print_missing(estimate: RecipeEstimate) -> None:
    for stage in estimate.missing_stages:
        print(
            f"error: cannot estimate stage {stage.stage!r}: {stage.note}",
            file=sys.stderr,
        )


def _confirmation_threshold(config: object) -> float:
    """Return the configured spend above which a run needs confirmation."""
    budget = getattr(config, "budget", None)
    if isinstance(budget, BudgetConfig):
        return budget.confirm_above
    return 0.0


def _jobs_config(config: object) -> JobsConfig:
    section = getattr(config, "jobs", None)
    return section if isinstance(section, JobsConfig) else JobsConfig()


def _providers_config(config: object) -> ProvidersConfig:
    section = getattr(config, "providers", None)
    return section if isinstance(section, ProvidersConfig) else ProvidersConfig()


def build_router(config: object) -> Router:
    return Router(build_providers(_providers_config(config)))


def _resolve_plan(
    args: argparse.Namespace,
) -> tuple[object, str, FormatRecipeConfig, RecipeEstimate, str, ProvidedInputs]:
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
    resolved_paths = paths.resolve_paths(args.config)
    name, recipe = _resolve_recipe(
        config,
        args.format_name,
        config_dir=resolved_paths.config_dir,
        defaults_dir=resolved_paths.defaults_dir,
    )
    provided = load_provided_inputs(
        text_file=getattr(args, "text_file", None),
        script_file=getattr(args, "script_file", None),
    )
    recipe = apply_provided_inputs(recipe, provided)
    estimate = Estimator(
        provider_metadata(config),
        configs=capability_configs(config) or None,
    ).estimate(name, recipe)
    return config, name, recipe, estimate, resolved.environment or "", provided


def _estimate_run(args: argparse.Namespace) -> tuple[object, RecipeEstimate]:
    config, _name, _recipe, estimate, _environment, _provided = _resolve_plan(args)
    return config, estimate


def _database_path(args: argparse.Namespace, config: object) -> Path:
    explicit = getattr(args, "database", None)
    if explicit:
        return Path(explicit).expanduser()
    database = getattr(config, "database", None)
    if not isinstance(database, DatabaseConfig):
        database = None
    filename = database.filename if database is not None else DatabaseConfig().filename
    return paths.resolve_paths(args.config).data_dir / filename


def _data_dir(args: argparse.Namespace) -> Path:
    return paths.resolve_paths(args.config).data_dir


def _open_database(path: Path) -> Engine:
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_database_engine(path=path)
    try:
        upgrade_database(engine=engine)
    except BaseException:
        engine.dispose()
        raise
    return engine


def _create_run(
    engine: Engine,
    args: argparse.Namespace,
    name: str,
    estimate: RecipeEstimate,
    environment: str = "",
    *,
    connection: Connection | None = None,
    created_dirs: list[Path] | None = None,
) -> IdempotentRun:
    run = Run(
        kind=RunKind.PIPELINE,
        state=RunState.QUEUED,
        idempotency_key=(getattr(args, "idempotency_key", None) or "").strip(),
        profile=(getattr(args, "profile", None) or "").strip(),
        environment=environment.strip(),
        format=name,
        toolchain_fingerprint=detect_toolchain(),
        estimated_cost_usd=estimate.total_usd,
    )

    def prepare(candidate: Run) -> None:
        directory = RunDirectories(_data_dir(args)).create(candidate)
        if created_dirs is not None:
            created_dirs.append(directory.path)

    return create_run_once(
        RunRepository(engine),
        run,
        prepare=prepare,
        connection=connection,
    )


def _discard_run_directories(directories: list[Path]) -> None:
    """Remove run directories created for a submission that did not persist."""
    while directories:
        shutil.rmtree(directories.pop(), ignore_errors=True)


RENDER_STAGE_NAME = SONG_VIDEO_STAGE_NAME
"""The recipe stage name served by the concrete render implementations."""

VISUALS_STAGE_NAMES = frozenset({"visual", "visuals", VISUAL_REUSE_STAGE_NAME})
"""The recipe stage names served by the concrete visual reuse implementation."""

_RENDER_STAGES: dict[str, type[Stage]] = {
    "image_post": ImagePostStage,
    "carousel": CarouselStage,
    "quote_card": QuoteCardRenderStage,
    "song_video": SongVideoStage,
    "audio_only": AudioOnlyStage,
    "personalized_song_card": AudioOnlyStage,
}


def _render_stage_class(
    recipe: FormatRecipeConfig,
    format_name: str | None = None,
) -> type[Stage]:
    """Choose the concrete renderer for a recipe's output and format.

    A named built-in format pins its own renderer. Otherwise the output type
    selects it: a carousel uses the carousel renderer, an image the image post
    renderer unless a typography template marks it as a quote card, audio the
    audio-only renderer and video the timeline renderer. An unknown output
    type keeps the base stage so the run is queued instead of failing.
    """
    name = (format_name or "").strip().lower()
    if name in _RENDER_STAGES:
        return _RENDER_STAGES[name]

    options = next(
        (item.options for item in recipe.stages if item.stage == RENDER_STAGE_NAME),
        {},
    )
    output = recipe.output_type
    if output is OutputType.CAROUSEL or options.get("carousel") is True:
        return CarouselStage
    if output is OutputType.IMAGE:
        if "template" in options or "typography" in options:
            return QuoteCardRenderStage
        return ImagePostStage
    if output is OutputType.AUDIO:
        return AudioOnlyStage
    if output is OutputType.VIDEO:
        return SongVideoStage
    return Stage


def build_stages(
    recipe: FormatRecipeConfig,
    format_name: str | None = None,
) -> list[Stage]:
    """Build the stage objects a recipe executes.

    Concrete stages arrive with the stage registry. Until then the base stage
    keeps the execution path wired and testable, and a caller may replace this
    function to supply implementations.
    """
    concrete: dict[str, type[Stage]] = {
        INPUT_TOPIC_STAGE_NAME: InputTopicStage,
        WRITE_STAGE_NAME: WriteStage,
        MUSIC_STAGE_NAME: MusicSelectStage,
        ALIGN_STAGE_NAME: LyricAlignStage,
        COVER_ART_STAGE_NAME: CoverArtStage,
        **dict.fromkeys(VISUALS_STAGE_NAMES, VisualReuseStage),
    }
    stages: list[Stage] = []
    for item in recipe.stages:
        if item.stage == RENDER_STAGE_NAME:
            stage_class = _render_stage_class(recipe, format_name)
        else:
            stage_class = concrete.get(item.stage, Stage)
        stages.append(stage_class(name=item.stage, **item.to_stage_kwargs()))
    return stages


def _execute_run(
    engine: Engine,
    run: Run,
    config: object,
    stages: list[Stage],
) -> OrchestrationResult:
    orchestrator = Orchestrator(
        stages=stages,
        run=run.id,
        repository=engine,
        run_dir=run.run_dir or None,
        data_dir=_data_dir_from_run(run),
        config=config,
        cost=CostTracker(engine),
        router=build_router(config),
        storage=build_storage_provider(
            getattr(config, "storage", None),
            data_dir=_data_dir_from_run(run),
        ),
        raise_on_failure=False,
    )
    return asyncio.run(orchestrator.execute())


def _data_dir_from_run(run: Run) -> Path | None:
    if not run.run_dir:
        return None
    return Path(run.run_dir).expanduser().parent.parent


def _stages_are_implemented(stages: list[Stage]) -> bool:
    """Whether every stage in the recipe overrides the base implementation."""
    return all(type(stage).run is not Stage.run for stage in stages)


def _print_created(
    run: Run,
    estimate: RecipeEstimate,
    *,
    job: Job | None = None,
) -> None:
    print(f"Run: {run.id}")
    print(f"Estimate: ${estimate.total_usd:.6f}")
    print(f"State: {run.state.value}")
    if job is not None:
        print(f"Job: {job.id}")


def _print_existing(run: Run) -> None:
    print(f"Run: {run.id}")
    if run.estimated_cost_usd is not None:
        print(f"Estimate: ${run.estimated_cost_usd:.6f}")
    print(f"State: {run.state.value}")


def _seed_provided(
    run: Run,
    recipe: FormatRecipeConfig,
    provided: ProvidedInputs,
) -> None:
    if not provided.supplied or not run.run_dir:
        return
    seed_provided_inputs(
        run.run_dir,
        provided,
        pattern=script_pattern(recipe),
    )


def _job_payload(
    name: str,
    run: Run,
    provided: ProvidedInputs,
) -> dict[str, object]:
    payload: dict[str, object] = {"format": name, "profile": run.profile}
    options = provided_stage_options(provided) if provided.supplied else {}
    if options:
        payload["stage_options"] = options
    return payload


def run_run(args: argparse.Namespace) -> int:
    """Create a run, execute or enqueue it, and print its id and estimate."""
    if not args.format_name:
        suffix = " with --estimate" if args.estimate else ""
        print(f"error: --format is required{suffix}", file=sys.stderr)
        return 2

    try:
        config, name, recipe, estimate, environment, provided = _resolve_plan(args)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.estimate:
        print(render_estimate(estimate), end="")
        if estimate.missing_stages:
            _print_missing(estimate)
            return 1
        return 0

    if estimate.missing_stages:
        print(render_estimate(estimate), end="")
        _print_missing(estimate)
        return 1

    threshold = _confirmation_threshold(config)
    if confirmation_required(estimate.total_usd, threshold) and not confirm_spend(
        estimate.total_usd,
        threshold,
        assume_yes=args.yes,
    ):
        return 1

    database_path = _database_path(args, config)
    stages = build_stages(recipe)
    enqueue = getattr(args, "enqueue", False) or (
        not getattr(args, "execute", False) and not _stages_are_implemented(stages)
    )
    try:
        engine = _open_database(database_path)
    except (OSError, SQLAlchemyError) as exc:
        print(f"error: cannot open the database: {exc}", file=sys.stderr)
        return 1

    try:
        queue = JobQueue(engine, config=_jobs_config(config))
        if enqueue:
            created_dirs: list[Path] = []
            try:
                with queue.write_transaction() as connection:
                    created = _create_run(
                        engine,
                        args,
                        name,
                        estimate,
                        environment,
                        connection=connection,
                        created_dirs=created_dirs,
                    )
                    run = created.run
                    if not created.created:
                        _discard_run_directories(created_dirs)
                        _print_existing(run)
                        return 0
                    _seed_provided(run, recipe, provided)
                    job = queue.enqueue(
                        Job(
                            kind="run",
                            run_id=run.id,
                            payload=_job_payload(name, run, provided),
                        ),
                        connection=connection,
                    )
            except BaseException:
                _discard_run_directories(created_dirs)
                raise
            _print_created(run, estimate, job=job)
            return 0

        created = _create_run(engine, args, name, estimate, environment)
        run = created.run
        if not created.created:
            _print_existing(run)
            return 0

        _seed_provided(run, recipe, provided)
        result = _execute_run(engine, run, config, stages)
        _print_created(result.run, estimate)
        if result.failed:
            print(
                f"error: run {result.run.id} failed at stage "
                f"{result.failed_stage!r}: {result.error_message}",
                file=sys.stderr,
            )
            return 1
        return 0
    except QuotaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except IdempotencyConflict as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (OSError, SQLAlchemyError) as exc:
        print(f"error: cannot run the recipe: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()


def _run_id(args: argparse.Namespace) -> str:
    positional = (getattr(args, "run_id", None) or "").strip()
    option = (getattr(args, "run_option", None) or "").strip()
    return positional or option


def _stage_rows(stages: list[StageResult]) -> list[StageResult]:
    return sorted(
        stages,
        key=lambda stage: (stage.index is None, stage.index or 0, stage.stage),
    )


def _artifact_rows(
    run: Run,
    stages: list[StageResult],
) -> list[tuple[str, str, str]]:
    rows: dict[tuple[str, str], str] = {}
    for stage in stages:
        for artifact in stage.artifacts:
            rows.setdefault((stage.stage, artifact), "-")
    if run.run_dir:
        directory = RunDirectory(run.run_dir, run_id=run.id)
        try:
            records = directory.artifacts()
        except (OSError, ValueError):
            records = ()
        for record in records:
            rows[(record.stage, record.path)] = str(record.size_bytes)
    return [
        (stage, path, size)
        for (stage, path), size in sorted(rows.items(), key=lambda item: item[0])
    ]


def _money(value: float | None) -> str:
    return "-" if value is None else f"${value:.6f}"


def run_show(args: argparse.Namespace) -> int:
    """Print a run's metadata, stage results, cost and artifacts."""
    run_id = _run_id(args)
    if not run_id:
        print("error: a run id is required", file=sys.stderr)
        return 2

    try:
        config = resolve_app_config(
            config_file=args.config,
            environment=args.environment,
            profile=args.profile,
            preset=args.preset,
            set_values=args.set_values,
        )
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    database_path = _database_path(args, config)
    if not database_path.exists():
        print(f"error: database not found: {database_path}", file=sys.stderr)
        return 1

    engine = create_database_engine(path=database_path)
    try:
        run = RunRepository(engine).get(run_id)
        if run is None:
            print(f"error: run not found: {run_id}", file=sys.stderr)
            return 1

        stages = _stage_rows(StageRunRepository(engine).list(run_id=run_id))
        ledger_total = CostTracker(engine).total(run_id=run_id)
        cost = ledger_total if ledger_total else run.cost_usd

        lines = [
            f"Run: {run.id}",
            f"Format: {run.format or '-'}",
            f"State: {run.state.value}",
            f"Cost: {_money(cost)}",
            f"Estimate: {_money(run.estimated_cost_usd)}",
        ]
        if run.profile:
            lines.insert(2, f"Profile: {run.profile}")
        if run.run_dir:
            lines.append(f"Run dir: {run.run_dir}")
        if run.error_code:
            lines.append(f"Error: {run.error_code}: {run.error_message}")

        lines.extend(["", "STAGES", "STAGE\tSTATUS\tCOST\tESTIMATE\tERROR"])
        for stage in stages:
            lines.append(
                "\t".join(
                    (
                        stage.stage,
                        stage.status.value,
                        _money(stage.cost_usd),
                        _money(stage.estimated_cost_usd),
                        stage.error_code,
                    )
                )
            )

        lines.extend(["", "ARTIFACTS", "STAGE\tPATH\tSIZE"])
        for artifact_stage, path, size in _artifact_rows(run, stages):
            lines.append("\t".join((artifact_stage, path, size)))

        print("\n".join(lines))
    except SQLAlchemyError as exc:
        print(f"error: cannot read the run: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()
    return 0


def _dispatch(args: argparse.Namespace) -> int:
    handler = getattr(args, "_run_handler", None)
    return int(handler(args)) if handler is not None else int(run_run(args))


register_command(
    "run",
    _dispatch,
    help="Run a format recipe; use --estimate to preview the cost",
    configure=configure,
)
