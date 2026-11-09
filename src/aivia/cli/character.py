"""``aivia character create``: validate supplied images and store version one.
``aivia character test``: render sample scenes across the configured consistency
strategies and print each scene's consistency score and cost.

A character is a YAML file in ``<data dir>/characters/<name>.yaml`` next to the
reference images and pose cutouts it references. This command validates every
supplied image with the shared media probe, stages the images next to the
character file and writes a schema-valid character document whose ``version`` is
one. Nothing is written unless every image validates, and an existing character
is never overwritten without ``--force``.

The test command renders one image per sample scene for every consistency
strategy in the configured chain, checks each produced asset against the
character's reference images and records a cost event per render. It prints a
per-scene table (scene, strategy, consistency score, cost) followed by the
total cost. Nothing is sent to a real provider unless a real provider is
configured; pass ``--dry-run`` to substitute the deterministic mock providers.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml
from PIL import Image
from pydantic import ValidationError as PydanticValidationError
from sqlalchemy.exc import SQLAlchemyError

from aivia.characters import (
    CURRENT_CHARACTER_SCHEMA_VERSION,
    Character,
    CharacterError,
    CharacterLibrary,
    CharacterNotFoundError,
    Scene,
    SceneResult,
    Strategy,
    run_strategy_chain,
)
from aivia.cli.registry import register_command
from aivia.core import paths
from aivia.core.config.character import CharacterConfig, ConsistencyStrategy
from aivia.core.config.providers import ProvidersConfig
from aivia.core.config.resolve import resolve_config
from aivia.core.cost import CostTracker
from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.dry_run import apply_dry_run
from aivia.core.errors import AiviaError, ConfigError, ValidationError
from aivia.core.media_probe import MediaCategory, MediaLimits, validate_media
from aivia.providers.protocols.image import CAPABILITY_IMAGE_GEN, ImageRequest
from aivia.providers.protocols.media import ImageFormat
from aivia.providers.provider_factory import build_providers

__all__ = ["configure", "run_character", "run_character_test"]

_DEFAULT_SCENES: tuple[str, ...] = (
    "The character waves hello in a sunlit room.",
    "The character stands ready, arms at their sides.",
    "The character laughs, head tilted back.",
    "The character points at something off-screen.",
)
"""Sample scenes rendered by ``aivia character test`` when none are supplied."""

_TEST_RUN_ID = "character-test"
"""Stable run id stamped on every cost event written by the test command."""

_DEFAULT_COST_PER_IMAGE_USD = 0.02
"""Fallback cost used when no image provider carries a price."""


def _default_scenes() -> list[str]:
    """Return the built-in sample scenes used when ``--scene`` is not given."""
    return list(_DEFAULT_SCENES)


def _load_config(args: argparse.Namespace) -> Any:
    """Resolve the merged application config, applying dry-run when requested."""
    resolved = resolve_config(
        config_file=getattr(args, "config", None),
        environment=getattr(args, "environment", None),
        profile=getattr(args, "profile", None),
        preset=getattr(args, "preset", None),
        set_values=getattr(args, "set_values", None),
    )
    config = resolved.to_app_config()
    if getattr(args, "dry_run", False):
        config = apply_dry_run(config)
    return config


def _character_config(config: Any) -> CharacterConfig:
    """Return the character section of the resolved config."""
    section = getattr(config, "character", None)
    if isinstance(section, CharacterConfig):
        return section
    return CharacterConfig()


def _data_dir(args: argparse.Namespace) -> Path:
    """Return the user data directory for the selected config file."""
    return paths.resolve_paths(getattr(args, "config", None)).data_dir


def _load_character(args: argparse.Namespace) -> Character | None:
    """Load the named character, or ``None`` when no name is given or not found."""
    name = getattr(args, "name", None)
    if not name:
        return None
    library = CharacterLibrary(data_dir=_data_dir(args))
    try:
        return library.load(name)
    except CharacterNotFoundError:
        return None


def _image_provider(config: Any) -> Any | None:
    """Return the first configured provider able to generate images."""
    section = getattr(config, "providers", None)
    if not isinstance(section, ProvidersConfig):
        return None
    providers = build_providers(section)
    for provider in providers.values():
        if CAPABILITY_IMAGE_GEN in getattr(provider, "capabilities", ()):
            return provider
    return None


def _cost_per_image(config: Any) -> float:
    """Return the simulated USD cost of one rendered image."""
    provider = _image_provider(config)
    if provider is not None:
        price = getattr(provider, "estimate_cost", None)
        if callable(price):
            try:
                value = float(price(CAPABILITY_IMAGE_GEN, 1.0))
                if math.isfinite(value) and value >= 0.0:
                    return value
            except (TypeError, ValueError):
                pass
    return _DEFAULT_COST_PER_IMAGE_USD


def _make_image_strategy(provider: Any, cost_per_image: float) -> Strategy:
    """Build one strategy that renders a scene with the image provider."""

    async def _render(scene: str) -> bytes | None:
        request = ImageRequest(prompt=scene, format=ImageFormat.PNG)
        response = await provider.generate(request)
        images = getattr(response, "images", None) or ()
        for image in images:
            data = getattr(image, "data", None)
            if isinstance(data, (bytes, bytearray)) and data:
                return bytes(data)
        return None

    def strategy(scene: Scene) -> bytes | None:
        return asyncio.run(_render(scene))

    return strategy


def _build_strategies(
    config: Any, character: Character | None
) -> dict[ConsistencyStrategy, Strategy]:
    """Build one rendering strategy per configured consistency strategy."""
    strategies: dict[ConsistencyStrategy, Strategy] = {}
    provider = _image_provider(config)
    if provider is None:
        return strategies
    cost_per_image = _cost_per_image(config)
    for strategy in _character_config(config).consistency_strategies:
        strategies[strategy] = _make_image_strategy(provider, cost_per_image)
    return strategies


def _thumbnail_similarity(a_bytes: bytes, b_bytes: bytes) -> float:
    """Compare two images with a cheap deterministic thumbnail metric."""
    try:
        im_a = Image.open(io.BytesIO(a_bytes)).convert("L").resize((32, 32))
        im_b = Image.open(io.BytesIO(b_bytes)).convert("L").resize((32, 32))
    except Exception:
        return 0.0
    pixels_a = list(im_a.getdata())
    pixels_b = list(im_b.getdata())
    if not pixels_a:
        return 0.0
    mse = float(
        sum((x - y) ** 2 for x, y in zip(pixels_a, pixels_b, strict=False))
    ) / float(len(pixels_a))
    return max(0.0, 1.0 - mse / (255.0 * 255.0))


def _build_scorer(config: Any, character: Character | None, data_dir: Path) -> Any:
    """Return a vision scorer comparing an asset to the character's references."""
    references: list[bytes] = []
    if character is not None:
        base = data_dir / "characters"
        for reference in character.reference_images:
            candidate = base / reference
            if candidate.is_file():
                with contextlib.suppress(OSError):
                    references.append(candidate.read_bytes())

    def scorer(asset: Any) -> float | None:
        if not isinstance(asset, (bytes, bytearray)) or not references:
            return None
        best = 0.0
        for reference in references:
            score = _thumbnail_similarity(bytes(asset), reference)
            if score > best:
                best = score
        return best

    return scorer


def _open_tracker(
    args: argparse.Namespace, config: Any
) -> tuple[Any | None, CostTracker | None]:
    """Open the cost database and return ``(engine, tracker)``."""
    explicit = getattr(args, "database", None)
    if explicit:
        db_path = Path(explicit).expanduser()
    else:
        database = getattr(config, "database", None)
        filename = getattr(database, "filename", None)
        db_path = paths.resolve_paths(getattr(args, "config", None)).data_dir / (
            filename or "aivia.db"
        )
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_database_engine(path=db_path)
    try:
        upgrade_database(engine=engine)
    except BaseException:
        engine.dispose()
        raise
    tracker = CostTracker(engine)
    return engine, tracker


def _record_cost(
    tracker: CostTracker | None,
    strategy: ConsistencyStrategy | None,
    cost_per_image: float,
) -> float:
    """Record one estimated image-gen cost event and return its amount.

    When *tracker* is ``None`` (an estimate run) the amount is still returned
    for display but nothing is persisted.
    """
    if strategy is None:
        return 0.0
    if tracker is not None:
        tracker.record_estimate(
            provider="character-test",
            operation=str(strategy.value),
            units=1.0,
            unit_kind="image",
            amount_usd=cost_per_image,
            pricing_version="mock",
            run_id=_TEST_RUN_ID,
        )
    return cost_per_image


def _format_score(score: float | None) -> str:
    """Render one consistency score for display."""
    if score is None:
        return "-"
    return f"{score:.3f}"


def _print_test_results(
    rows: list[tuple[int, str, SceneResult, float | None, float]],
    total_cost: float,
) -> None:
    """Print the per-scene results and the total cost."""
    print(f"{'Scene':<6} {'Strategy':<14} {'Score':<8} {'Cost'}")
    print("-" * 44)
    for index, _scene, result, score, cost in rows:
        name = result.strategy.value if result.strategy is not None else "-"
        print(f"{index:<6} {name:<14} {_format_score(score):<8} ${cost:.6f}")
    print("-" * 44)
    print(f"Total cost: ${total_cost:.6f}")


def run_character_test(args: argparse.Namespace) -> int:
    """Render sample scenes across strategies and print scores and cost."""
    try:
        config = _load_config(args)
        character = _load_character(args)
        character_config = _character_config(config)
        cost_per_image = _cost_per_image(config)

        if getattr(args, "estimate", False):
            for strategy in character_config.consistency_strategies:
                print(f"{strategy.value:<14} estimated ${cost_per_image:.6f}")
            total = cost_per_image * len(character_config.consistency_strategies)
            print(f"Total estimated cost: ${total:.6f}")
            return 0

        name = getattr(args, "name", None)
        if name is not None and character is None:
            print(f"error: unknown character: {name}", file=sys.stderr)
            return 1
        scenes = getattr(args, "scenes", None) or _default_scenes()
        strategies = _build_strategies(config, character)
        scorer = _build_scorer(config, character, _data_dir(args))

        engine: Any | None = None
        tracker: CostTracker | None = None
        engine, tracker = _open_tracker(args, config)
        if tracker is None:
            print("error: unable to open the cost database", file=sys.stderr)
            return 1

        rows: list[tuple[int, str, SceneResult, float | None, float]] = []
        total_cost = 0.0
        for index, scene in enumerate(scenes):
            result = run_strategy_chain(
                [scene], config=character_config, strategies=strategies
            )[0]
            score: float | None = None
            cost = 0.0
            if result.asset is not None:
                try:
                    score = scorer(result.asset)
                except Exception:
                    score = None
                cost = _record_cost(tracker, result.strategy, cost_per_image)
            total_cost += cost
            rows.append((index, scene, result, score, cost))

        _print_test_results(rows, total_cost)

        if engine is not None:
            engine.dispose()
        return 0
    except (AiviaError, ConfigError, OSError, SQLAlchemyError, ValidationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


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

    test = subparsers.add_parser(
        "test",
        help=(
            "Render sample scenes across the configured consistency strategies "
            "and print each scene's consistency score and cost"
        ),
        description=(
            "Render one image per sample scene for every consistency strategy in "
            "the configured chain, check each produced asset against the "
            "character's reference images and print a per-scene table of strategy, "
            "consistency score and cost followed by the total cost."
        ),
    )
    test.add_argument(
        "name",
        nargs="?",
        metavar="NAME",
        help="character name to test consistency against (optional)",
    )
    test.add_argument(
        "--scene",
        action="append",
        dest="scenes",
        metavar="TEXT",
        help="sample scene prompt to render; repeatable",
    )
    test.add_argument(
        "--strategy",
        choices=[strategy.value for strategy in ConsistencyStrategy],
        metavar="NAME",
        help=(
            "limit the test to one consistency strategy "
            f"({', '.join(s.value for s in ConsistencyStrategy)})"
        ),
    )
    test.add_argument(
        "--estimate",
        action="store_true",
        help="print the estimated cost without rendering",
    )
    test.add_argument(
        "--dry-run",
        action="store_true",
        help="render with the deterministic mock providers (no spend, no network)",
    )
    test.add_argument(
        "--output",
        metavar="DIR",
        help="copy each rendered asset under DIR instead of leaving it in storage",
    )
    test.add_argument(
        "--database",
        "--db",
        dest="database",
        metavar="PATH",
        help="cost database file to use instead of the one in the user data directory",
    )
    test.set_defaults(_character_handler=run_character_test)


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
        print("usage: aivia character {create,test} [options]", file=sys.stderr)
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
