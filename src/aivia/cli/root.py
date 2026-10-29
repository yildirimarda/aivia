"""Root command line interface for AIVIA."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from aivia import __version__
from aivia.cli.globals import (
    add_global_options,
    apply_global_options,
    split_global_options,
)
from aivia.cli.registry import Command, CommandRegistry
from aivia.cli.registry import registry as default_registry
from aivia.core.bootstrap import bootstrap
from aivia.core.logging import configure_logging

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
    add_global_options(parser)
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    active.add_to(subparsers)
    return parser


def _bootstrap_before(command: Command) -> int | None:
    """Populate the user config directory before a command runs.

    ``init`` is the explicit bootstrap command and already reports what it
    creates, so it is left alone.  Every other command silently ensures the
    packaged defaults exist, which makes a fresh installation usable without
    asking the user to run ``init`` first.
    """
    if command.name == "init":
        return None
    try:
        bootstrap()
    except OSError as exc:
        print(f"error: cannot populate the config directory: {exc}", file=sys.stderr)
        return 1
    return None


def main(
    argv: Sequence[str] | None = None,
    *,
    registry: CommandRegistry | None = None,
) -> int:
    """Run the root CLI and return its exit code."""
    active = registry if registry is not None else default_registry
    parser = build_parser(active)
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        provided, remaining = split_global_options(arguments)
        args = parser.parse_args(remaining)
    except SystemExit as exc:
        return int(exc.code or 0)

    apply_global_options(args, provided)
    if args.verbose:
        configure_logging(logging.DEBUG)

    if args.command is None:
        print(version_string())
        return 0

    command = active.get(args.command)
    bootstrap_error = _bootstrap_before(command)
    if bootstrap_error is not None:
        return bootstrap_error
    result = command.handler(args)
    return 0 if result is None else int(result)


if __name__ == "__main__":
    sys.exit(main())
