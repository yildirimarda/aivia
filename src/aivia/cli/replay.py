"""``aivia replay``: re-execute a stored run and report artifact drift.

A replay reads the snapshot recorded in the original run directory, applies the
recorded seeds to that configuration and executes the recipe again in a fresh
run directory. The original run, its artifacts and its manifest are never
modified. After the replay finishes the registered artifacts of both runs are
compared by path and content hash so a changed, added or removed output is
reported as drift.
"""

from __future__ import annotations

import argparse
import copy
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
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
from aivia.core.config.resolve import resolve_app_config
from aivia.core.cost import CostTracker
from aivia.core.db import RunRepository, create_database_engine
from aivia.core.errors import ConfigError
from aivia.core.models import Run, RunState
from aivia.core.orchestrator import OrchestrationResult, Orchestrator
from aivia.core.run_dirs import ArtifactRecord, RunDirectories, RunDirectory
from aivia.core.snapshot import (
    ConfigSnapshot,
    build_config_snapshot,
    read_config_snapshot,
    write_config_snapshot,
)
from aivia.core.stages import Stage

__all__ = [
    "ArtifactDrift",
    "configure",
    "diff_artifacts",
    "run_replay",
]

_ORIGINAL_BUILD_STAGES = build_stages


def _build_stages(recipe: FormatRecipeConfig) -> list[Stage]:
    """Use a local override when a test or module replaced this function."""
    if build_stages is not _ORIGINAL_BUILD_STAGES:
        return build_stages(recipe)
    return run_command.build_stages(recipe)


@dataclass(frozen=True, slots=True)
class ArtifactDrift:
    """One artifact compared between the original run and its replay."""

    stage: str
    path: str
    status: str
    original_sha256: str
    replay_sha256: str

    @property
    def drifted(self) -> bool:
        """Whether this artifact differs between the two runs."""
        return self.status != "unchanged"

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable representation of the comparison."""
        return {
            "stage": self.stage,
            "path": self.path,
            "status": self.status,
            "original_sha256": self.original_sha256,
            "replay_sha256": self.replay_sha256,
        }


def diff_artifacts(
    original: Iterable[ArtifactRecord],
    replay: Iterable[ArtifactRecord],
) -> tuple[ArtifactDrift, ...]:
    """Compare two artifact registries by stage, path, hash and size.

    The result is sorted by stage and path and contains every artifact from
    either side. An artifact only present in the replay is ``added``, one only
    present in the original is ``removed`` and a shared path with a different
    digest or size is ``changed``.
    """
    before = {(record.stage, record.path): record for record in original}
    after = {(record.stage, record.path): record for record in replay}
    rows: list[ArtifactDrift] = []
    for key in sorted(set(before) | set(after)):
        original_record = before.get(key)
        replay_record = after.get(key)
        stage, path = key
        if original_record is None:
            status = "added"
        elif replay_record is None:
            status = "removed"
        elif (
            original_record.sha256 == replay_record.sha256
            and original_record.size_bytes == replay_record.size_bytes
        ):
            status = "unchanged"
        else:
            status = "changed"
        rows.append(
            ArtifactDrift(
                stage=stage,
                path=path,
                status=status,
                original_sha256=(
                    original_record.sha256 if original_record is not None else ""
                ),
                replay_sha256=(
                    replay_record.sha256 if replay_record is not None else ""
                ),
            )
        )
    return tuple(rows)


def _configure_resolution(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        metavar="PATH",
        help="config file used only to locate the database and data directory",
    )
    parser.add_argument(
        "--env",
        "--environment",
        dest="environment",
        metavar="NAME",
        help="environment used only to locate the database and data directory",
    )
    parser.add_argument(
        "--profile",
        metavar="NAME",
        help="profile used only to locate the database and data directory",
    )
    parser.add_argument(
        "--set",
        dest="set_values",
        action="append",
        metavar="KEY=VALUE",
        help="override a dotted key when locating the database; repeatable",
    )


def configure(parser: argparse.ArgumentParser) -> None:
    """Add the replay command's options."""
    parser.add_argument(
        "run_id",
        nargs="?",
        metavar="RUN_ID",
        help="stored run to replay",
    )
    parser.add_argument(
        "--run",
        "--run-id",
        dest="run_option",
        metavar="RUN_ID",
        help="stored run to replay",
    )
    parser.add_argument(
        "--database",
        "--db",
        dest="database",
        metavar="PATH",
        help="database file to read instead of the one in the user data directory",
    )
    parser.add_argument(
        "--fail-on-drift",
        action="store_true",
        help="exit non-zero when any artifact differs from the original",
    )
    _configure_resolution(parser)


def _run_id(args: argparse.Namespace) -> str:
    positional = (getattr(args, "run_id", None) or "").strip()
    option = (getattr(args, "run_option", None) or "").strip()
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


def _load_snapshot(run: Run) -> ConfigSnapshot:
    if run.run_dir:
        directory = Path(run.run_dir).expanduser()
        if directory.is_dir():
            try:
                return read_config_snapshot(directory)
            except FileNotFoundError:
                pass
    if run.config_snapshot:
        return ConfigSnapshot(copy.deepcopy(run.config_snapshot))
    raise FileNotFoundError(f"run has no replay snapshot: {run.id}")


def _snapshot_config(snapshot: ConfigSnapshot) -> dict[str, Any]:
    raw = snapshot.get("config")
    if not isinstance(raw, Mapping):
        raise ValueError("snapshot config must be an object")
    return copy.deepcopy(dict(raw))


def _apply_seeds(config: dict[str, Any], seeds: Any) -> dict[str, Any]:
    """Apply recorded seeds to *config*, using dotted keys as nested paths."""
    if not isinstance(seeds, Mapping):
        if seeds is not None:
            config["seed"] = copy.deepcopy(seeds)
        return config
    for key, value in seeds.items():
        path = str(key).strip()
        if not path:
            continue
        parts = path.split(".")
        target = config
        for part in parts[:-1]:
            child = target.get(part)
            if not isinstance(child, dict):
                child = {}
                target[part] = child
            target = child
        target[parts[-1]] = copy.deepcopy(value)
    return config


def _recipe(config: Mapping[str, Any], run: Run) -> FormatRecipeConfig:
    raw = config.get("formats")
    formats = FormatsConfig.model_validate(raw if raw is not None else {})
    name = run.format.strip()
    if not name:
        raise ValueError(f"run has no format to replay: {run.id}")
    recipe = formats.formats.get(name)
    if recipe is None:
        available = ", ".join(sorted(formats.formats)) or "none"
        raise ValueError(
            f"snapshot has no format recipe {name!r} (available: {available})"
        )
    return recipe


def _original_artifacts(run: Run) -> tuple[ArtifactRecord, ...]:
    if not run.run_dir:
        return ()
    directory = RunDirectory(run.run_dir, run_id=run.id)
    if not directory.path.is_dir():
        return ()
    try:
        return directory.artifacts()
    except (OSError, ValueError):
        return ()


def _recorded_toolchain(snapshot: ConfigSnapshot) -> dict[str, str] | None:
    raw = snapshot.get("toolchain_fingerprint")
    if not isinstance(raw, Mapping):
        return None
    return {
        key: str(raw.get(key, ""))
        for key in ("renderer_version", "ffmpeg_version", "font_set")
    }


def _create_replay_run(
    engine: Engine,
    original: Run,
    snapshot: ConfigSnapshot,
    config: Mapping[str, Any],
    database_path: Path,
) -> Run:
    replay = Run(
        kind=original.kind,
        state=RunState.QUEUED,
        profile=original.profile,
        format=original.format,
        toolchain_fingerprint=original.toolchain_fingerprint,
        estimated_cost_usd=original.estimated_cost_usd,
    )
    directory = RunDirectories(database_path.parent).create(replay)
    payload = build_config_snapshot(
        config,
        run=replay,
        seeds=snapshot.get("seeds"),
        providers=snapshot.get("providers"),
        models=snapshot.get("models"),
        model_roles=snapshot.get("model_roles"),
        toolchain_fingerprint=_recorded_toolchain(snapshot),
        profile=original.profile,
        environment=snapshot.get("environment"),
    )
    write_config_snapshot(directory.path, payload)
    replay.config_snapshot = dict(payload)
    return RunRepository(engine).create(replay)


def _execute_replay(
    engine: Engine,
    replay: Run,
    config: Mapping[str, Any],
    stages: list[Stage],
    database_path: Path,
) -> OrchestrationResult:
    orchestrator = Orchestrator(
        stages=stages,
        run=replay,
        repository=engine,
        run_dir=replay.run_dir or None,
        data_dir=database_path.parent,
        config=config,
        cost=CostTracker(engine),
        raise_on_failure=False,
    )
    return orchestrator.execute_sync()


def _short_digest(value: str) -> str:
    return value[:12] if value else "-"


def _render_report(
    original: Run,
    result: OrchestrationResult,
    drifts: tuple[ArtifactDrift, ...],
) -> str:
    changed = sum(1 for item in drifts if item.status == "changed")
    added = sum(1 for item in drifts if item.status == "added")
    removed = sum(1 for item in drifts if item.status == "removed")
    unchanged = sum(1 for item in drifts if item.status == "unchanged")
    drifted = [item for item in drifts if item.drifted]

    lines = [
        f"Replay: {original.id} -> {result.run.id}",
        f"State: {result.run.state.value}",
        (
            f"Artifacts: {unchanged} unchanged, {changed} changed, "
            f"{added} added, {removed} removed"
        ),
        "Drift: none" if not drifted else f"Drift: {len(drifted)}",
    ]
    if drifted:
        lines.extend(["", "STAGE\tPATH\tSTATUS\tORIGINAL\tREPLAY"])
        for item in drifted:
            lines.append(
                "\t".join(
                    (
                        item.stage,
                        item.path,
                        item.status,
                        _short_digest(item.original_sha256),
                        _short_digest(item.replay_sha256),
                    )
                )
            )
    if result.failed:
        lines.append(f"Error: {result.error_code}: {result.error_message}")
    return "\n".join(lines)


def run_replay(args: argparse.Namespace) -> int:
    """Replay a stored run from its snapshot and report artifact drift."""
    run_id = _run_id(args)
    if not run_id:
        print("error: a run id is required", file=sys.stderr)
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
        original = RunRepository(engine).get(run_id)
        if original is None:
            print(f"error: run not found: {run_id}", file=sys.stderr)
            return 1

        try:
            snapshot = _load_snapshot(original)
            config = _apply_seeds(_snapshot_config(snapshot), snapshot.get("seeds"))
            recipe = _recipe(config, original)
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        try:
            replay = _create_replay_run(
                engine,
                original,
                snapshot,
                config,
                database_path,
            )
            result = _execute_replay(
                engine,
                replay,
                config,
                _build_stages(recipe),
                database_path,
            )
        except (OSError, ValueError, SQLAlchemyError) as exc:
            print(f"error: cannot replay the run: {exc}", file=sys.stderr)
            return 1

        drifts = diff_artifacts(
            _original_artifacts(original),
            _original_artifacts(result.run),
        )
        print(_render_report(original, result, drifts))
        if result.failed:
            return 1
        if getattr(args, "fail_on_drift", False) and any(
            item.drifted for item in drifts
        ):
            return 1
        return 0
    except SQLAlchemyError as exc:
        print(f"error: cannot replay the run: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()


register_command(
    "replay",
    run_replay,
    help="Replay a stored run from its snapshot and report artifact drift",
    configure=configure,
)
