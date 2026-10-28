"""``aivia config``: inspect the layered configuration."""

from __future__ import annotations

import argparse
import sys

from aivia.cli.registry import register_command
from aivia.core import config_show
from aivia.core.errors import ConfigError

__all__ = ["configure", "run_config", "run_show"]


def _configure_show(parser: argparse.ArgumentParser) -> None:
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
    parser.add_argument(
        "--resolved",
        action="store_true",
        help="show the fully merged configuration (the default)",
    )
    parser.add_argument(
        "--origins",
        action="store_true",
        help="print the layer that supplied each value",
    )
    parser.set_defaults(_config_handler=run_show)


def configure(parser: argparse.ArgumentParser) -> None:
    """Add the config group and its subcommands."""
    subparsers = parser.add_subparsers(dest="config_command", metavar="COMMAND")
    show = subparsers.add_parser(
        "show",
        help="Print the merged configuration with profile and origin tracking",
        description="Print the merged configuration with profile and origin tracking",
    )
    _configure_show(show)


def run_config(args: argparse.Namespace) -> int:
    """Dispatch to the selected config subcommand."""
    handler = getattr(args, "_config_handler", None)
    if handler is None:
        print("error: missing config command", file=sys.stderr)
        print("usage: aivia config {show}", file=sys.stderr)
        return 2
    return int(handler(args))


def run_show(args: argparse.Namespace) -> int:
    """Print the merged configuration, masking secrets."""
    try:
        resolved = config_show.resolve_config(
            config_file=args.config,
            environment=args.environment,
            profile=args.profile,
            set_values=args.set_values,
        )
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        config_show.dump_config(resolved, include_origins=args.origins),
        end="",
    )
    return 0


register_command(
    "config",
    run_config,
    help="Inspect the layered configuration",
    configure=configure,
)
