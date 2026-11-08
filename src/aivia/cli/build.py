"""``aivia build poses``: generate, check and store a standard pose set.

Each ``POSE=SOURCE`` argument names a pose and its source image. The source is
turned into a transparent cutout with local background removal (generates), one
cost event is recorded in the ledger (cost recorded), the cutout is stored next
to its pose record (stores) and the stored pose is reloaded to confirm its
schema and asset are valid (checks). Nothing is written unless every source
validates, and an existing pose is never overwritten without ``--force``.
"""

from __future__ import annotations

import argparse
import contextlib
import re
import sys
from pathlib import Path

from aivia.characters import (
    Pose,
    PoseError,
    PoseLibrary,
    PoseSchemaError,
    cleanup_edges,
)
from aivia.cli.registry import register_command
from aivia.core import paths
from aivia.core.cost import CostTracker
from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.errors import ValidationError

__all__ = ["configure", "run_build"]

_POSE = re.compile(r"^([^=]+)=(.+)$")
_PROVIDER = "rembg"
_MODEL = "rembg"
_OPERATION = "remove_background"
_UNIT_KIND = "image"


def configure(parser: argparse.ArgumentParser) -> None:
    """Add the build group and its poses subcommand."""
    subparsers = parser.add_subparsers(dest="build_command", metavar="COMMAND")
    poses = subparsers.add_parser(
        "poses",
        help="Generate, check and store a standard pose and expression set",
        description=(
            "Turn each POSE=SOURCE image into a transparent cutout, record its "
            "cost, store it in the pose library and reload it to verify it."
        ),
    )
    poses.add_argument(
        "poses",
        metavar="POSE=SOURCE",
        nargs="+",
        help="named pose and its source image, as POSE=SOURCE; repeatable",
    )
    poses.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing pose with the same name",
    )
    poses.set_defaults(_build_handler=run_build_poses)


def _parse_pose(value: str) -> tuple[str, str]:
    match = _POSE.match(value.strip())
    if not match:
        raise ValidationError(
            f"pose must be POSE=SOURCE: {value!r}",
            details={"pose": value},
        )
    name = match.group(1).strip()
    source = match.group(2).strip()
    if not name or not source:
        raise ValidationError(
            f"pose must be POSE=SOURCE with both sides non-empty: {value!r}",
            details={"pose": value},
        )
    return name, source


def _image_name(name: str) -> str:
    safe = "".join(char if char.isalnum() or char in "-_." else "_" for char in name)
    if not safe or safe in {".", ".."}:
        safe = "pose"
    if not safe.lower().endswith(".png"):
        safe = f"{safe}.png"
    return safe


def _remove_pose(library: PoseLibrary, name: str) -> None:
    """Drop an existing pose file and its cutout image so --force can replace it."""
    pose_file = library.pose_path(name)
    try:
        existing = library.load(name)
        image_file = library.poses_dir / existing.image
    except PoseError:
        image_file = None
    with contextlib.suppress(OSError):
        pose_file.unlink()
    if image_file is not None:
        with contextlib.suppress(OSError):
            image_file.unlink()


def _read_source(source: str) -> bytes:
    path = Path(source).expanduser()
    if not path.is_file():
        raise ValidationError(
            f"pose source image not found: {path}",
            details={"source": source, "path": str(path)},
        )
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ValidationError(
            f"cannot read pose source {path}: {exc}",
            details={"source": source, "path": str(path)},
        ) from exc


def run_build(args: argparse.Namespace) -> int:
    """Dispatch to the selected build subcommand."""
    handler = getattr(args, "_build_handler", None)
    if handler is None:
        print("error: missing build command", file=sys.stderr)
        print("usage: aivia build poses POSE=SOURCE [options]", file=sys.stderr)
        return 2
    return int(handler(args))


def run_build_poses(args: argparse.Namespace) -> int:
    """Generate, check and store every requested pose and record its cost."""
    dry_run = bool(getattr(args, "dry_run", False))
    force = bool(getattr(args, "force", False))
    character = getattr(args, "character", None) or None

    try:
        parsed = [_parse_pose(value) for value in args.poses]
    except ValidationError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        return 1

    sources: list[tuple[str, str, bytes]] = []
    for name, source in parsed:
        try:
            sources.append((name, source, _read_source(source)))
        except ValidationError as exc:
            print(f"error: {exc.message}", file=sys.stderr)
            return 1

    library = PoseLibrary(data_dir=paths.data_dir(), character=character)

    if dry_run:
        print("Would build the following poses:")
        for name, source, _ in sources:
            print(f"  {name} <- {source} (as {_image_name(name)})")
        print(f"Character: {character or '-'}")
        print(f"Total: {len(sources)} pose(s)")
        return 0

    database_path = paths.data_dir() / "aivia.db"
    database_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_database_engine(path=database_path)
    try:
        upgrade_database(engine=engine)
        tracker = CostTracker(engine)

        built: list[str] = []
        for name, _source, data in sources:
            if library.exists(name) and not force:
                print(
                    f"error: pose already exists at {library.pose_path(name)}; "
                    "use --force to overwrite",
                    file=sys.stderr,
                )
                return 1
            if force and library.exists(name):
                _remove_pose(library, name)
            try:
                cutout = cleanup_edges(data)
                pose = Pose(
                    name=name,
                    image=_image_name(name),
                    alpha=True,
                    tags=([character] if character else []) + ["pose"],
                    edge_cleanup="rembg",
                )
                library.store(pose, image_bytes=cutout, edge_cleanup=False)
                library.load(name)
                tracker.record(
                    provider=_PROVIDER,
                    model=_MODEL,
                    operation=_OPERATION,
                    units=1.0,
                    unit_kind=_UNIT_KIND,
                    amount_usd=0.0,
                    estimated=False,
                )
            except (PoseError, PoseSchemaError, OSError) as exc:
                print(f"error: cannot build pose {name!r}: {exc}", file=sys.stderr)
                return 1
            built.append(name)

        print(f"Built {len(built)} pose(s): {', '.join(built)}")
        for name in built:
            print(f"  {name} -> {_image_name(name)} ({_PROVIDER}, free)")
        return 0
    finally:
        engine.dispose()


register_command(
    "build",
    run_build,
    help="Build stored assets such as pose libraries",
    configure=configure,
)
