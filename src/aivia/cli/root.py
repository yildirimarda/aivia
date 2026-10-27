"""Root command line interface for AIVIA."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from aivia import __version__
from aivia.cli.registry import CommandRegistry
from aivia.cli.registry import registry as default_registry

__all__ = ["build_parser", "main", "version_string"]


def version_string() -> str:
    """Return the text printed by the version flag."""
    return f"aivia {__version__}"


def build_parser(registry: CommandRegistry | None = None) -> argparse.ArgumentParser:
    """Build the root parser with the version flag and registered commands."""
    active = registry if registry is not None else default_registry
    parser = argparse.ArgumentParser(
        prog="aivia",
        description="Generate, compose and deliver AI-backed media.",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=version_string(),
        help="show the installed version and exit",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    active.add_to(subparsers)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    registry: CommandRegistry | None = None,
) -> int:
    """Run the root CLI and return its exit code."""
    active = registry if registry is not None else default_registry
    parser = build_parser(active)
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)

    if args.command is None:
        print(version_string())
        return 0

    command = active.get(args.command)
    result = command.handler(args)
    return 0 if result is None else int(result)


if __name__ == "__main__":
    sys.exit(main())
