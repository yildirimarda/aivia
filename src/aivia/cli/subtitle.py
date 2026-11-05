"""``aivia subtitle``: inspect and edit subtitle words and timings.

``subtitle edit`` exports the effective words of a timeline as YAML, then reads
an edited copy back. Applying an edit updates the timeline, regenerates only
the changed languages' ASS files and re-burns only the affected compositions
through the shared render cache. Audio synthesis is never repeated.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from sqlalchemy.exc import SQLAlchemyError

from aivia.audio.subtitle_edit import (
    apply_subtitle_edits,
    subtitle_edit_yaml,
)
from aivia.audio.subtitle_stage import SubtitleStage, SubtitleStageResult
from aivia.cli.registry import register_command
from aivia.compose import (
    Composition,
    RenderCache,
    Timeline,
    render_video,
    timeline_to_compositions,
)
from aivia.core import paths
from aivia.core.config.database import DatabaseConfig
from aivia.core.config.resolve import resolve_config
from aivia.core.config.video import VideoConfig, VideoOverridesConfig
from aivia.core.db import RunRepository, create_database_engine
from aivia.core.errors import AiviaError, ConfigError
from aivia.core.models import Run
from aivia.core.platforms import get_platform
from aivia.core.run_dirs import RunDirectories, RunDirectory
from aivia.core.stages import StageContext

__all__ = [
    "configure",
    "render_affected_outputs",
    "run_edit",
    "run_subtitle",
]

DEFAULT_TIMELINE_NAME = "timeline.json"
"""The timeline file the voice stage writes into a run directory."""

_FORMAT_SUFFIXES = {
    "video": "mp4",
    "audio": "mp3",
    "still": "png",
    "image": "png",
    "web-card": "html",
    "web_card": "html",
}


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


def _configure_edit(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "run_id",
        nargs="?",
        metavar="RUN_ID",
        help="run whose timeline to edit",
    )
    parser.add_argument(
        "--run",
        "--run-id",
        dest="run_option",
        metavar="RUN_ID",
        help="run whose timeline to edit",
    )
    parser.add_argument(
        "--timeline",
        metavar="PATH",
        help=(
            "timeline file to edit instead of RUN_ID/timeline.json; a relative "
            "path is resolved against the run directory when RUN_ID is given"
        ),
    )
    parser.add_argument(
        "--language",
        "--languages",
        dest="languages",
        action="append",
        metavar="CODE",
        help="language to export; repeatable or comma separated",
    )
    parser.add_argument(
        "--output",
        "-o",
        metavar="PATH",
        help="write the YAML export to PATH instead of standard output",
    )
    parser.add_argument(
        "--input",
        "--apply",
        "--from",
        dest="input",
        metavar="PATH",
        help="apply an edited YAML document instead of exporting",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        help="directory for re-burnt outputs (default: RUN_DIR/renders)",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        metavar="NAME",
        help="output container for re-burnt files (default: mp4)",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="update the timeline and ASS files without re-burning outputs",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="ignore the render cache and burn every affected output again",
    )
    parser.add_argument(
        "--database",
        "--db",
        dest="database",
        metavar="PATH",
        help="database file to read instead of the one in the user data directory",
    )
    _configure_resolution(parser)


def configure(parser: argparse.ArgumentParser) -> None:
    """Add the subtitle group and its ``edit`` subcommand."""
    subparsers = parser.add_subparsers(dest="subtitle_command", metavar="COMMAND")
    edit = subparsers.add_parser(
        "edit",
        help="Expose subtitle words and timings as YAML and re-burn edited outputs",
        description=(
            "Export a timeline's words and timings as YAML, or apply an edited "
            "document and re-burn only the affected outputs"
        ),
    )
    _configure_edit(edit)
    edit.set_defaults(_subtitle_handler=run_edit)


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


def _database_path(args: argparse.Namespace, config: Any) -> Path:
    explicit = getattr(args, "database", None)
    if explicit:
        return Path(explicit).expanduser()
    database = getattr(config, "database", None)
    if not isinstance(database, DatabaseConfig):
        database = None
    filename = database.filename if database is not None else DatabaseConfig().filename
    return paths.resolve_paths(args.config).data_dir / filename


def _load_run(args: argparse.Namespace, config: Any) -> Run:
    run_id = _run_id(args)
    database_path = _database_path(args, config)
    if not database_path.exists():
        raise ConfigError(
            f"database not found: {database_path}",
            details={"database": str(database_path)},
        )
    engine = create_database_engine(path=database_path)
    try:
        run = RunRepository(engine).get(run_id)
    finally:
        engine.dispose()
    if run is None:
        raise ConfigError(f"run not found: {run_id}", details={"run": run_id})
    return run


def _run_directory(args: argparse.Namespace, config: Any, run: Run) -> Path:
    if run.run_dir:
        return Path(run.run_dir).expanduser()
    data_dir = _database_path(args, config).parent
    return RunDirectories(data_dir).path_for(run)


def _resolve_timeline(
    args: argparse.Namespace,
) -> tuple[Path, Run | None, Any]:
    run_id = _run_id(args)
    explicit = getattr(args, "timeline", None)
    if explicit is None and not run_id:
        raise ConfigError("a run id or --timeline is required")

    config: Any = {}
    if run_id or args.config or args.environment or args.profile or args.set_values:
        config = _load_config(args)

    run: Run | None = None
    if run_id:
        run = _load_run(args, config)

    if explicit is not None:
        path = Path(explicit).expanduser()
        if not path.is_absolute() and run is not None:
            path = _run_directory(args, config, run) / path
        return path, run, config

    assert run is not None
    return _run_directory(args, config, run) / DEFAULT_TIMELINE_NAME, run, config


def _read_timeline(path: Path) -> Timeline:
    if not path.is_file():
        raise ConfigError(
            f"timeline not found: {path}",
            details={"timeline": str(path)},
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(
            f"invalid timeline JSON: {path}",
            details={"timeline": str(path)},
        ) from exc
    try:
        return Timeline.model_validate(payload)
    except ValueError as exc:
        raise ConfigError(
            f"invalid timeline: {path}",
            details={"timeline": str(path)},
        ) from exc


def _write_timeline(path: Path, timeline: Timeline) -> None:
    payload = json.dumps(
        timeline.model_dump(mode="json"),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with open(handle, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.write("\n")
            stream.flush()
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _requested_languages(args: argparse.Namespace) -> list[str]:
    values = getattr(args, "languages", None) or []
    if isinstance(values, str):
        values = [values]
    codes: list[str] = []
    for value in values:
        for part in str(value).split(","):
            code = part.strip().replace("_", "-").lower()
            if code and code not in codes:
                codes.append(code)
    return codes


def _read_input(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    return Path(path).expanduser().read_text(encoding="utf-8")


def _platform_video(video: Any, platforms: Sequence[str]) -> Any:
    """Fill missing video platform overrides from the packaged registry."""
    if not platforms:
        return video
    if isinstance(video, VideoConfig):
        base = video
    elif isinstance(video, Mapping):
        base = VideoConfig.from_mapping(video)
    elif video is None:
        base = VideoConfig()
    else:
        return video

    overrides = dict(base.platforms)
    for name in platforms:
        if name in overrides:
            continue
        try:
            platform = get_platform(name)
        except (KeyError, ValueError):
            continue
        limit = platform.limit("video")
        if limit is None:
            continue
        values: dict[str, Any] = {}
        if limit.resolutions:
            values["resolution"] = limit.resolutions[0]
        if limit.loudness_lufs is not None:
            values["loudness_lufs"] = limit.loudness_lufs
        if values:
            overrides[name] = VideoOverridesConfig.from_mapping(values)
    if overrides == base.platforms:
        return base
    return base.model_copy(update={"platforms": overrides})


def _compositions(timeline: Timeline, config: Any) -> list[Composition]:
    video = (
        config.get("video")
        if isinstance(config, Mapping)
        else getattr(config, "video", None)
    )
    return timeline_to_compositions(
        timeline,
        video=_platform_video(video, timeline.platforms),
    )


def _format_suffix(value: str | None) -> str:
    name = (value or "mp4").strip().lower().lstrip(".")
    name = _FORMAT_SUFFIXES.get(name, name)
    if not name:
        raise ConfigError("output format must not be empty")
    return f".{name}"


def _regenerate_subtitles(
    timeline: Timeline,
    changed: Sequence[str],
    run_dir: Path,
    config: Any,
) -> dict[str, str]:
    """Regenerate every ASS file of each changed language and platform.

    The stage owns platform expansion so a multi-platform timeline writes one
    file per platform with that platform's canvas and safe area. Forcing a
    single platform here would leave the other platforms' ASS files stale.
    """
    files: dict[str, str] = {}
    for code in changed:
        stage = SubtitleStage(options={"languages": [code]}, timeline=timeline)
        result = cast(
            SubtitleStageResult,
            asyncio.run(stage.run(StageContext(run_dir=run_dir, config=config))),
        )
        for key, relative in result.files.items():
            language, _separator, _platform = key.partition("/")
            if language == code:
                files[key] = relative
    return files


def render_affected_outputs(
    compositions: Sequence[Composition],
    changed: Sequence[str],
    *,
    output_dir: Path,
    output_format: str | None = None,
    cache_root: Path | None = None,
    no_cache: bool = False,
) -> list[tuple[Path, bool]]:
    """Render only compositions for *changed* languages, reusing the cache.

    Returns ``(path, cached)`` pairs. A cache hit copies the previous artifact
    instead of running the renderer again.
    """
    affected = set(changed)
    suffix = _format_suffix(output_format)
    root = cache_root or (paths.data_dir() / "cache" / "renders" / "subtitle")
    cache = RenderCache(root, no_cache=no_cache)
    rendered: list[tuple[Path, bool]] = []
    for composition in compositions:
        if composition.language not in affected:
            continue
        output = output_dir / f"{composition.id or 'timeline'}{suffix}"
        output.parent.mkdir(parents=True, exist_ok=True)
        cached = cache.get(composition)
        if cached is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cached.path, output)
            rendered.append((output, True))
            continue
        render_video(composition, output)
        cache.store(
            composition,
            output,
            metadata={
                "kind": "video",
                "language": composition.language,
                "platform": composition.platform,
            },
        )
        rendered.append((output, False))
    return rendered


def _register_edit_artifacts(
    timeline_path: Path,
    run: Run | None,
    subtitles: Mapping[str, str],
    rendered: Sequence[tuple[Path, bool]],
) -> None:
    """Record edited subtitle files and re-burnt outputs for resume."""
    directory = RunDirectory(
        timeline_path.parent,
        run_id=run.id if run is not None else "",
    )
    subtitle_files: list[Path] = []
    for relative in subtitles.values():
        if not relative:
            continue
        try:
            subtitle_files.append(directory.resolve(relative))
        except ValueError:
            continue
    if subtitle_files:
        directory.register("subtitle", subtitle_files)
    render_files: list[Path] = []
    for path, _cached in rendered:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = candidate.resolve()
        try:
            render_files.append(directory.resolve(candidate))
        except ValueError:
            continue
    if render_files:
        directory.register("render", render_files)
    directory.register("voice", timeline_path)


def _print_apply_report(
    timeline_path: Path,
    changed: Sequence[str],
    subtitles: Mapping[str, str],
    rendered: Sequence[tuple[Path, bool]],
) -> None:
    print(f"Timeline: {timeline_path}")
    print("Affected: " + (", ".join(changed) if changed else "none"))
    for relative in subtitles.values():
        if relative:
            print(f"Subtitle: {relative}")
    for path, cached in rendered:
        print(f"{'Cached' if cached else 'Rendered'}: {path}")


def run_edit(args: argparse.Namespace) -> int:
    """Export subtitle words as YAML, or apply an edited document."""
    try:
        timeline_path, _run, config = _resolve_timeline(args)
        timeline = _read_timeline(timeline_path)

        if getattr(args, "input", None):
            text = _read_input(args.input)
            updated, changed = apply_subtitle_edits(timeline, text)
            compositions = _compositions(updated, config)
            subtitles: dict[str, str] = {}
            rendered: list[tuple[Path, bool]] = []
            if changed:
                _write_timeline(timeline_path, updated)
                subtitles = _regenerate_subtitles(
                    updated,
                    changed,
                    timeline_path.parent,
                    config,
                )
                if not getattr(args, "no_render", False):
                    output_dir = getattr(args, "output_dir", None)
                    directory = (
                        Path(output_dir).expanduser()
                        if output_dir
                        else timeline_path.parent / "renders"
                    )
                    rendered = render_affected_outputs(
                        compositions,
                        changed,
                        output_dir=directory,
                        output_format=getattr(args, "output_format", None),
                        no_cache=bool(getattr(args, "no_cache", False)),
                    )
                _register_edit_artifacts(timeline_path, _run, subtitles, rendered)
            _print_apply_report(timeline_path, changed, subtitles, rendered)
            return 0

        document = subtitle_edit_yaml(
            timeline,
            languages=_requested_languages(args),
        )
        output = getattr(args, "output", None)
        if output:
            target = Path(output).expanduser()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(document, encoding="utf-8")
            print(f"Wrote: {target}")
        else:
            print(document, end="")
        return 0
    except (AiviaError, OSError, ValueError, SQLAlchemyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def run_subtitle(args: argparse.Namespace) -> int:
    """Dispatch a subtitle subcommand."""
    handler = getattr(args, "_subtitle_handler", None)
    if handler is None:
        print("usage: aivia subtitle {edit}", file=sys.stderr)
        return 2
    return int(handler(args))


register_command(
    "subtitle",
    run_subtitle,
    help="Inspect and edit subtitle words and timings",
    configure=configure,
)
