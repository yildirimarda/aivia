"""``aivia character create``: validate supplied images and store version one.

A character is a YAML file in ``<data dir>/characters/<name>.yaml`` next to the
reference images and pose cutouts it references. This command validates every
supplied image with the shared media probe, stages the images next to the
character file and writes a schema-valid character document whose ``version`` is
one. Nothing is written unless every image validates, and an existing character
is never overwritten without ``--force``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml
from pydantic import ValidationError as PydanticValidationError

from aivia.characters import (
    CURRENT_CHARACTER_SCHEMA_VERSION,
    Character,
    CharacterError,
    CharacterLibrary,
)
from aivia.cli.registry import register_command
from aivia.core.errors import ValidationError
from aivia.core.media_probe import MediaCategory, MediaLimits, validate_media

__all__ = ["configure", "run_character"]


def configure(parser: argparse.ArgumentParser) -> None:
    """Add the character group and its create subcommand."""
    subparsers = parser.add_subparsers(dest="character_command", metavar="COMMAND")
    create = subparsers.add_parser(
        "create",
        help="Validate supplied images and store a version one character",
        description="Validate supplied images and store a version one character",
    )
    create.add_argument(
        "name",
        metavar="NAME",
        help="character name to create",
    )
    create.add_argument(
        "--image",
        action="append",
        dest="images",
        metavar="PATH",
        help="reference image to validate and store; repeatable",
    )
    create.add_argument(
        "--pose",
        action="append",
        dest="poses",
        metavar="NAME=PATH",
        help="named pose cutout as NAME=PATH; repeatable",
    )
    create.add_argument(
        "--voice",
        metavar="VOICE",
        default="",
        help="character voice id",
    )
    create.add_argument(
        "--art-style",
        metavar="STYLE",
        default="",
        help="character art style",
    )
    create.add_argument(
        "--brand-kit",
        metavar="KIT",
        default="",
        help="bound brand kit name",
    )
    create.add_argument(
        "--negatives",
        action="append",
        dest="negatives",
        metavar="TEXT",
        help="negative prompt term; repeatable",
    )
    create.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing character with the same name",
    )
    create.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the planned files without writing",
    )
    create.set_defaults(_character_handler=run_character_create)


def _parse_poses(values: list[str] | None) -> dict[str, str]:
    poses: dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            raise ValidationError(
                f"pose must be NAME=PATH: {value!r}",
                details={"pose": value},
            )
        name, _, path = value.partition("=")
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValidationError(
                f"pose must be NAME=PATH with both sides non-empty: {value!r}",
                details={"pose": value},
            )
        poses[name] = path
    return poses


def _validate_image(path: str) -> None:
    """Confirm *path* is a readable image within the default media limits."""
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise ValidationError(
            f"cannot read image {path}: {exc}",
            details={"path": path},
        ) from exc
    try:
        probe = validate_media(data, path=path, limits=MediaLimits())
    except ValidationError:
        raise
    except (
        Exception
    ) as exc:  # pragma: no cover - probe failures become validation errors
        raise ValidationError(
            f"cannot validate image {path}: {exc}",
            details={"path": path},
        ) from exc
    if probe.category is not MediaCategory.IMAGE:
        raise ValidationError(
            f"{path} is not an image (detected {probe.category.value})",
            details={"path": path, "category": probe.category.value},
        )


def _stage_image(path: str, destination_dir: Path) -> str:
    """Copy *path* into *destination_dir* and return the stored file name."""
    source = Path(path)
    name = source.name
    if not name:
        raise ValidationError(
            f"image path has no file name: {path}", details={"path": path}
        )
    target = destination_dir / name
    try:
        shutil.copyfile(source, target)
    except OSError as exc:
        raise ValidationError(
            f"cannot stage image {path} as {name}: {exc}",
            details={"path": path, "staged_as": name},
        ) from exc
    return name


def run_character(args: argparse.Namespace) -> int:
    """Dispatch to the selected character subcommand."""
    handler = getattr(args, "_character_handler", None)
    if handler is None:
        print("error: missing character command", file=sys.stderr)
        print("usage: aivia character create NAME [options]", file=sys.stderr)
        return 2
    return int(handler(args))


def run_character_create(args: argparse.Namespace) -> int:
    """Validate supplied images and store a version one character file."""
    library = CharacterLibrary()
    characters_dir = library.characters_dir
    try:
        character_path = library.character_path(args.name)
    except CharacterError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        return 1

    if character_path.is_file() and not args.force:
        print(
            f"error: character already exists at {character_path}",
            file=sys.stderr,
        )
        print(
            "refusing to overwrite without --force; use --dry-run to preview",
            file=sys.stderr,
        )
        return 1

    try:
        images = args.images or []
        poses = _parse_poses(args.poses)

        # Validate every image before writing anything.
        for image in images:
            _validate_image(image)
        for image_path in poses.values():
            _validate_image(image_path)
    except ValidationError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        return 1

    reference_images = [Path(image).name for image in images]
    pose_library = {
        pose_name: Path(image_path).name for pose_name, image_path in poses.items()
    }

    character: dict[str, object] = {
        "name": args.name,
        "version": CURRENT_CHARACTER_SCHEMA_VERSION,
    }
    if args.voice:
        character["voice"] = args.voice
    if args.art_style:
        character["art_style"] = args.art_style
    if args.brand_kit:
        character["brand_kit"] = args.brand_kit
    if args.negatives:
        character["negatives"] = args.negatives
    if reference_images:
        character["reference_images"] = reference_images
    if pose_library:
        character["pose_library"] = pose_library

    # Validate the assembled document against the schema before storing it.
    try:
        Character.model_validate(character)
    except PydanticValidationError as exc:
        print(f"error: invalid character {args.name!r}: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(f"Would create: {character_path}")
        for image in reference_images:
            print(f"Would stage: {characters_dir / image}")
        for pose_name, image in pose_library.items():
            print(f"Would stage pose {pose_name!r}: {characters_dir / image}")
        print(f"Version: {CURRENT_CHARACTER_SCHEMA_VERSION}")
        return 0

    characters_dir.mkdir(parents=True, exist_ok=True)
    for image in images:
        _stage_image(image, characters_dir)
    for image_path in poses.values():
        _stage_image(image_path, characters_dir)

    try:
        character_path.write_text(
            yaml.safe_dump(character, sort_keys=False), encoding="utf-8"
        )
    except OSError as exc:
        print(f"error: cannot write {character_path}: {exc}", file=sys.stderr)
        return 1

    print(
        f"Created character {args.name!r} at {character_path} "
        f"(version {CURRENT_CHARACTER_SCHEMA_VERSION})"
    )
    return 0


register_command(
    "character",
    run_character,
    help="Create and manage recurring characters",
    configure=configure,
)
