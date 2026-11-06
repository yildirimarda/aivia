"""``aivia add-language``: extend an existing run with another language.

The command runs only the run's write stage for the requested languages, so the
existing fact list is reused instead of being collected again. The visuals of a
run belong to the run rather than to one language, so the shared visual manifest
is extended with the new language without running a visual stage or calling a
visual provider. A language that needs more beats than the shared set has still
uses the existing prefix; no visual is ever recharged.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from sqlalchemy.exc import SQLAlchemyError

import aivia.cli.run as run_command
from aivia.cli.registry import register_command
from aivia.cli.run import build_stages
from aivia.cli.write import (
    _apply_language_filter,
    _database_path,
    _execute,
    _load_config,
    _recipe,
    _report,
    _requested_languages,
    _run_id,
)
from aivia.core.config.formats import FormatRecipeConfig
from aivia.core.cost import CostTracker
from aivia.core.db import RunRepository, create_database_engine
from aivia.core.errors import ConfigError
from aivia.core.models import Run
from aivia.core.orchestrator import OrchestrationResult
from aivia.core.run_dirs import RunDirectory
from aivia.core.stages import Stage
from aivia.visuals.image_stage import _write_json
from aivia.visuals.reuse import (
    DEFAULT_SHARED_VISUALS_FILENAME,
    DEFAULT_VISUAL_REUSE_FILENAME,
)

__all__ = [
    "ADD_LANGUAGE_STAGE_NAMES",
    "VISUAL_MANIFEST_NAMES",
    "configure",
    "run_add_language",
]

ADD_LANGUAGE_STAGE_NAMES = frozenset({"write", "writing"})
"""Recipe stage names that produce one script per language."""

VISUAL_MANIFEST_NAMES = (
    DEFAULT_VISUAL_REUSE_FILENAME,
    DEFAULT_SHARED_VISUALS_FILENAME,
    "visuals.json",
    "visual.json",
    "image.json",
    "video.json",
    "stock.json",
)
"""Manifest names an existing run may use for its shared visual set."""

_VISUAL_STAGE_NAMES = frozenset(
    {
        "visual",
        "visuals",
        "visual_reuse",
        "visual-reuse",
        "image",
        "stock",
        "video",
        "visual_qa",
        "visual-qa",
    }
)
"""Recipe stage names whose options may name a custom visual manifest."""

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
    """Add the add-language command's options."""
    parser.add_argument(
        "run_id",
        nargs="?",
        metavar="RUN_ID",
        help="existing run to extend with another language",
    )
    parser.add_argument(
        "--run",
        "--run-id",
        dest="run_option",
        metavar="RUN_ID",
        help="existing run to extend with another language",
    )
    parser.add_argument(
        "language",
        nargs="?",
        metavar="CODE",
        help="language to add (the same as --language)",
    )
    parser.add_argument(
        "--language",
        "--languages",
        dest="languages",
        action="append",
        metavar="CODE",
        help="language to add; repeatable or comma separated",
    )
    parser.add_argument(
        "--database",
        "--db",
        dest="database",
        metavar="PATH",
        help="database file to update instead of the one in the user data directory",
    )
    _configure_resolution(parser)


def _write_stage(recipe: FormatRecipeConfig) -> Stage:
    for stage in _build_stages(recipe):
        if stage.name in ADD_LANGUAGE_STAGE_NAMES:
            return stage
    raise ConfigError(
        "format has no write stage",
        details={"stages": list(recipe.stages)},
    )


def _manifest_candidates(recipe: FormatRecipeConfig, run_dir: Path) -> list[Path]:
    """Return every existing visual manifest the run may carry."""
    names: list[str] = list(VISUAL_MANIFEST_NAMES)
    for stage in recipe.stages:
        if stage.stage not in _VISUAL_STAGE_NAMES:
            continue
        for key in ("filename", "manifest_filename", "visual_manifest"):
            value = stage.options.get(key)
            if isinstance(value, str) and value.strip():
                names.append(value.strip())

    candidates: list[Path] = []
    for name in names:
        path = Path(name)
        if not path.is_absolute():
            path = run_dir / path
        if path not in candidates:
            candidates.append(path)
    return [path for path in candidates if path.is_file()]


def _manifest_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _script_beat_count(
    run_dir: Path,
    language: str,
    artifacts: Iterable[str],
) -> int | None:
    """Return how many beats the written script for *language* has."""
    names = (
        f"script.{language}.json",
        f"script_{language}.json",
        "script.json",
    )
    paths = [run_dir / name for name in names]
    for artifact in artifacts:
        path = Path(artifact)
        candidate = path if path.is_absolute() else run_dir / path
        if language in candidate.name and candidate.suffix.lower() == ".json":
            paths.append(candidate)

    seen: set[Path] = set()
    for path in paths:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        beats = payload.get("beats") if isinstance(payload, Mapping) else None
        if isinstance(beats, Sequence) and not isinstance(
            beats, (str, bytes, bytearray)
        ):
            return len(beats)
    return None


def _add_language_to_manifest(
    path: Path,
    language: str,
    beat_count: int | None,
) -> bool:
    """Share the manifest's existing beats with *language* without new cost."""
    payload = _manifest_payload(path)
    if payload is None:
        return False

    languages = payload.get("languages")
    if not isinstance(languages, list):
        languages = []
    if language not in languages:
        languages.append(language)
    payload["languages"] = languages

    beats = payload.get("beats")
    if isinstance(beats, list):
        for beat in beats:
            if not isinstance(beat, dict):
                continue
            beat_languages = beat.get("languages")
            if not isinstance(beat_languages, list):
                beat_languages = []
            if language not in beat_languages:
                beat_languages.append(language)
            beat["languages"] = beat_languages
        default_count = len(beats)
    else:
        default_count = 0

    counts = payload.get("beat_counts")
    if not isinstance(counts, dict):
        counts = {}
    counts[language] = default_count if beat_count is None else beat_count
    payload["beat_counts"] = counts

    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        metadata["languages"] = list(languages)
        metadata["beat_counts"] = dict(counts)
        metadata["reused_across"] = len(languages)

    _write_json(path, payload)
    return True


def _refresh_visual_artifacts(run_dir: Path, updated: Sequence[Path]) -> None:
    """Re-hash modified manifests so a resume does not rerun the visual stage."""
    if not updated:
        return
    directory = RunDirectory(run_dir)
    if not directory.manifest_path.is_file():
        return
    try:
        records = directory.artifacts()
    except (OSError, ValueError):
        return
    changed = {path.resolve(strict=False) for path in updated}
    stages = {
        record.stage
        for record in records
        if directory.resolve(record.path).resolve(strict=False) in changed
    }
    for stage in stages:
        paths = [record.path for record in records if record.stage == stage]
        try:
            directory.register(stage, paths)
        except (OSError, ValueError):
            continue


def _reuse_visuals(
    recipe: FormatRecipeConfig,
    run: Run,
    languages: Sequence[str],
    artifacts: Iterable[str],
) -> int:
    """Extend every visual manifest with *languages* and return the count."""
    if not run.run_dir:
        return 0
    run_dir = Path(run.run_dir).expanduser()
    manifests = _manifest_candidates(recipe, run_dir)
    if not manifests:
        return 0

    updated: list[Path] = []
    for language in languages:
        count = _script_beat_count(run_dir, language, artifacts)
        for path in manifests:
            if _add_language_to_manifest(path, language, count):
                updated.append(path)
    _refresh_visual_artifacts(run_dir, updated)
    return len(set(updated))


def _result_for(
    result: OrchestrationResult,
    stage: Stage,
) -> Any:
    return next(
        (item for item in result.stages if item.stage == stage.name),
        None,
    )


def run_add_language(args: argparse.Namespace) -> int:
    """Write extra languages for an existing run and reuse its visuals."""
    run_id = _run_id(args)
    if not run_id:
        print("error: a run id is required", file=sys.stderr)
        return 2

    try:
        config = _load_config(args)
        raw_languages = list(getattr(args, "languages", None) or ())
        positional = (getattr(args, "language", None) or "").strip()
        if positional:
            raw_languages.append(positional)
        args.languages = raw_languages
        languages = _requested_languages(args, config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not languages:
        print("error: at least one --language is required", file=sys.stderr)
        return 2

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
            recipe = _recipe(config, run)
            stage = _write_stage(recipe)
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
        stage_result = _result_for(result, stage)
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

        reused = _reuse_visuals(
            recipe,
            result.run,
            languages,
            stage_result.artifacts,
        )
        if reused:
            print("Visuals: reused without a provider call")
        else:
            print("Visuals: none")
        return 0
    except SQLAlchemyError as exc:
        print(f"error: cannot add the language: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()


register_command(
    "add-language",
    run_add_language,
    help="Add a language to an existing run, reusing its facts and visuals",
    configure=configure,
)
register_command(
    "add_language",
    run_add_language,
    help="Alias for add-language",
    configure=configure,
)
register_command(
    "language",
    run_add_language,
    help="Add a language to an existing run, reusing its facts and visuals",
    configure=configure,
)
