"""``aivia resume``: continue a run paused by the budget guard."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError

from aivia.cli.confirm import confirm_action
from aivia.cli.registry import register_command
from aivia.core import paths
from aivia.core.config.database import DatabaseConfig
from aivia.core.config.resolve import resolve_app_config
from aivia.core.db import RunRepository, create_database_engine
from aivia.core.errors import ConfigError
from aivia.core.models import RunState, utcnow

__all__ = ["configure", "run_resume"]


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
    """Add the resume command's options."""
    parser.add_argument(
        "run_id",
        nargs="?",
        metavar="RUN_ID",
        help="paused run to resume",
    )
    parser.add_argument(
        "--run",
        "--run-id",
        dest="run_option",
        metavar="RUN_ID",
        help="paused run to resume",
    )
    parser.add_argument(
        "--by",
        "--who",
        "--resumed-by",
        dest="resumed_by",
        metavar="NAME",
        help="person or service resuming the run; recorded on the run",
    )
    parser.add_argument(
        "--reason",
        "--why",
        "--resume-reason",
        dest="resume_reason",
        metavar="TEXT",
        help="why the run is being resumed; recorded on the run",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="confirm resuming without prompting",
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
    positional = (args.run_id or "").strip()
    option = (args.run_option or "").strip()
    return positional or option


def _resolve_database(
    args: argparse.Namespace,
) -> tuple[Path, DatabaseConfig | None]:
    explicit = getattr(args, "database", None)
    if explicit:
        return Path(explicit).expanduser(), None

    config = resolve_app_config(
        config_file=args.config,
        environment=args.environment,
        profile=args.profile,
        set_values=args.set_values,
    )
    database = getattr(config, "database", None)
    if not isinstance(database, DatabaseConfig):
        database = None
    filename = database.filename if database is not None else DatabaseConfig().filename
    resolved = paths.resolve_paths(args.config)
    return resolved.data_dir / filename, database


def _resume(
    engine: Engine,
    run_id: str,
    *,
    resumed_by: str,
    resume_reason: str,
    assume_yes: bool,
) -> int:
    repository = RunRepository(engine)
    run = repository.get(run_id)
    if run is None:
        print(f"error: run not found: {run_id}", file=sys.stderr)
        return 1
    if run.state is not RunState.PAUSED_BUDGET:
        print(
            f"error: run {run.id} is not paused on budget (state: {run.state.value})",
            file=sys.stderr,
        )
        return 1

    if not confirm_action(
        f"Resume paused budget run {run.id}?",
        assume_yes=assume_yes,
    ):
        return 1

    moment = utcnow()
    resumed = repository.update(
        run.id,
        state=RunState.RUNNING,
        resumed_by=resumed_by,
        resume_reason=resume_reason,
        resumed_at=moment,
        updated_at=moment,
    )
    if resumed is None:
        print(f"error: cannot resume run: {run.id}", file=sys.stderr)
        return 1

    print(f"Resumed run {resumed.id} as {resumed.resumed_by}: {resumed.resume_reason}")
    return 0


def run_resume(args: argparse.Namespace) -> int:
    """Resume a paused budget run after recording who confirmed and why."""
    run_id = _run_id(args)
    if not run_id:
        print("error: a run id is required", file=sys.stderr)
        return 2

    resumed_by = (args.resumed_by or "").strip()
    if not resumed_by:
        print("error: --by is required to record who resumed the run", file=sys.stderr)
        return 2
    resume_reason = (args.resume_reason or "").strip()
    if not resume_reason:
        print(
            "error: --reason is required to record why the run was resumed",
            file=sys.stderr,
        )
        return 2

    try:
        database_path, database_config = _resolve_database(args)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not database_path.exists():
        print(f"error: database not found: {database_path}", file=sys.stderr)
        return 1

    engine = create_database_engine(path=database_path, config=database_config)
    try:
        return _resume(
            engine,
            run_id,
            resumed_by=resumed_by,
            resume_reason=resume_reason,
            assume_yes=args.yes,
        )
    except SQLAlchemyError as exc:
        print(f"error: cannot resume the run: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()


register_command(
    "resume",
    run_resume,
    help="Resume a budget-paused run after recording who confirmed and why",
    configure=configure,
)
