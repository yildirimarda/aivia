"""``aivia series``: view and edit a series continuity log.

A series continuity log lives at ``<data dir>/series/<name>.yaml`` and holds
established facts (category, key, value) that keep a series consistent across
outputs. The ``view`` subcommand prints the log and the ``edit`` subcommand
adds, replaces or removes facts. Nothing is written unless the assembled
document validates against the series schema.
"""

from __future__ import annotations

import argparse
import sys

from pydantic import ValidationError as PydanticValidationError

from aivia.characters import (
    Series,
    SeriesError,
    SeriesFact,
    SeriesLibrary,
    SeriesNotFoundError,
    check_contradictions,
)
from aivia.cli.registry import register_command

__all__ = ["configure", "run_series"]


def configure(parser: argparse.ArgumentParser) -> None:
    """Add the series group and its view and edit subcommands."""
    subparsers = parser.add_subparsers(dest="series_command", metavar="COMMAND")

    view = subparsers.add_parser(
        "view",
        help="Print a series continuity log",
        description="Print a series continuity log",
    )
    view.add_argument(
        "name",
        metavar="NAME",
        help="series name to view",
    )
    view.set_defaults(_series_handler=run_series_view)

    edit = subparsers.add_parser(
        "edit",
        help="Add, replace or remove series continuity facts",
        description="Add, replace or remove series continuity facts",
    )
    edit.add_argument(
        "name",
        metavar="NAME",
        help="series name to edit",
    )
    edit.add_argument(
        "--add",
        action="append",
        dest="adds",
        metavar="CATEGORY=KEY=VALUE",
        help="add a continuity fact as CATEGORY=KEY=VALUE; repeatable",
    )
    edit.add_argument(
        "--set",
        action="append",
        dest="sets",
        metavar="CATEGORY=KEY=VALUE",
        help="replace a continuity fact as CATEGORY=KEY=VALUE; repeatable",
    )
    edit.add_argument(
        "--remove",
        action="append",
        dest="removes",
        metavar="CATEGORY=KEY",
        help="remove the continuity fact for CATEGORY=KEY; repeatable",
    )
    edit.add_argument(
        "--source",
        metavar="SOURCE",
        default="",
        help="source recorded on every added or replaced fact",
    )
    edit.add_argument(
        "--confidence",
        metavar="CONFIDENCE",
        default="1.0",
        help="confidence recorded on every added or replaced fact (0 to 1)",
    )
    edit.add_argument(
        "--force",
        action="store_true",
        help="create the series when it does not exist yet",
    )
    edit.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the planned changes without writing",
    )
    edit.set_defaults(_series_handler=run_series_edit)


def _parse_fact(values: list[str] | None) -> dict[tuple[str, str], str]:
    facts: dict[tuple[str, str], str] = {}
    for value in values or []:
        parts = value.split("=")
        if len(parts) != 3 or not all(part.strip() for part in parts):
            raise ValueError(f"fact must be CATEGORY=KEY=VALUE: {value!r}")
        category, key, value_text = (part.strip() for part in parts)
        facts[(category, key)] = value_text
    return facts


def _parse_remove(values: list[str] | None) -> set[tuple[str, str]]:
    removes: set[tuple[str, str]] = set()
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"remove must be CATEGORY=KEY: {value!r}")
        category, _, key = value.partition("=")
        category = category.strip()
        key = key.strip()
        if not category or not key:
            raise ValueError(
                "remove must be CATEGORY=KEY with non-empty category and key: "
                f"{value!r}"
            )
        removes.add((category, key))
    return removes


def _parse_confidence(value: str) -> float:
    try:
        confidence = float(value)
    except ValueError as exc:
        raise ValueError(f"confidence must be a number: {value!r}") from exc
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence must be between 0 and 1: {value!r}")
    return confidence


def run_series(args: argparse.Namespace) -> int:
    """Dispatch to the selected series subcommand."""
    handler = getattr(args, "_series_handler", None)
    if handler is None:
        print("error: missing series command", file=sys.stderr)
        print("usage: aivia series view|edit NAME [options]", file=sys.stderr)
        return 2
    return int(handler(args))


def _format_series(series: Series) -> str:
    lines = [
        f"series: {series.name}",
        f"version: {series.version}",
        f"facts: {len(series.facts)}",
    ]
    if series.facts:
        lines.append("")
        lines.append("facts:")
        for fact in series.facts:
            source = f" (source: {fact.source})" if fact.source else ""
            lines.append(f"  {fact.category}.{fact.key} = {fact.value!r}{source}")
    return "\n".join(lines)


def run_series_view(args: argparse.Namespace) -> int:
    """Print a series continuity log."""
    library = SeriesLibrary()
    try:
        series = library.load(args.name)
    except SeriesNotFoundError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        return 1
    except SeriesError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        return 1
    print(_format_series(series))
    return 0


def run_series_edit(args: argparse.Namespace) -> int:
    """Add, replace or remove series continuity facts."""
    library = SeriesLibrary()
    series_path = library.series_path(args.name)
    exists = series_path.is_file()

    if exists:
        try:
            series = library.load(args.name)
        except SeriesError as exc:
            print(f"error: {exc.message}", file=sys.stderr)
            return 1
    elif args.force:
        series = Series(name=args.name)
    else:
        print(
            f"error: series {args.name!r} does not exist; use --force to create it",
            file=sys.stderr,
        )
        return 1

    try:
        adds = _parse_fact(args.adds)
        sets = _parse_fact(args.sets)
        removes = _parse_remove(args.removes)
        confidence = _parse_confidence(args.confidence)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    overlaps = set(adds) & set(sets)
    if overlaps:
        listed = ", ".join(f"{category}.{key}" for category, key in sorted(overlaps))
        print(
            f"error: fact(s) given to both --add and --set: {listed}",
            file=sys.stderr,
        )
        return 1

    for (category, key), _value in adds.items():
        if series.fact_for(category, key) is not None:
            print(
                f"error: {category}.{key} already exists; use --set to replace it",
                file=sys.stderr,
            )
            return 1

    for category, key in removes:
        if series.fact_for(category, key) is None:
            print(
                f"error: no such fact {category}.{key} to remove",
                file=sys.stderr,
            )
            return 1

    updated = series
    for category, key in removes:
        updated = updated.remove_fact(category, key)
    for (category, key), _value in adds.items():
        fact = SeriesFact(
            category=category,
            key=key,
            value=_value,
            source=args.source,
            confidence=confidence,
        )
        updated = updated.add_or_replace_fact(fact)
    for (category, key), _value in sets.items():
        fact = SeriesFact(
            category=category,
            key=key,
            value=_value,
            source=args.source,
            confidence=confidence,
        )
        updated = updated.add_or_replace_fact(fact)

    try:
        Series.model_validate(updated.model_dump(mode="python"))
    except PydanticValidationError as exc:
        print(f"error: invalid series {args.name!r}: {exc}", file=sys.stderr)
        return 1

    contradictions = check_contradictions(
        updated,
        "\n".join(f"{fact.category} {fact.key} {fact.value}" for fact in updated.facts),
    )
    if contradictions:
        print(
            "warning: edited series contains internal contradictions:",
            file=sys.stderr,
        )
        for contradiction in contradictions:
            print(
                f"  {contradiction.fact.category}.{contradiction.fact.key}: "
                f"{contradiction.reason}",
                file=sys.stderr,
            )

    # Normalise the stored name to the lookup name so the file path stays
    # stable even when the document's name field differs in case.
    updated = updated.model_copy(update={"name": args.name})

    if args.dry_run:
        print(f"Would write: {series_path}")
        print(_format_series(updated))
        return 0

    try:
        library.save(updated)
    except SeriesError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        return 1

    print(
        f"Updated series {args.name!r} at {series_path} "
        f"(version {updated.version}, {len(updated.facts)} facts)"
    )
    return 0


register_command(
    "series",
    run_series,
    help="View and edit a series continuity log",
    configure=configure,
)
