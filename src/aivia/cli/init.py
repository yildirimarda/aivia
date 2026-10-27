"""``aivia init``: create the user config skeleton from packaged defaults."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from aivia.cli.registry import register_command
from aivia.core import paths
from aivia.core.bootstrap import BootstrapResult, bootstrap

__all__ = ["configure", "run_init"]


def configure(parser: argparse.ArgumentParser) -> None:
    """Add the init command's options."""
    parser.add_argument(
        "--config",
        metavar="PATH",
        help="config file to create instead of the default location",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing config file with the packaged skeleton",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved paths and planned changes without writing",
    )


def _print_paths(resolved: paths.Paths) -> None:
    print(f"Config directory: {resolved.config_dir}")
    print(f"Config file: {resolved.config_file}")
    print(f"Data directory: {resolved.data_dir}")
    print(f"Defaults directory: {resolved.defaults_dir}")


def _print_result(
    result: BootstrapResult, *, overwrote: bool, config_file: Path
) -> None:
    if overwrote:
        print(f"Overwrote: {config_file}")
    for path in result.created:
        print(f"Created: {path}")
    for path in result.preserved:
        if overwrote and path == config_file:
            continue
        print(f"Preserved: {path}")


def run_init(args: argparse.Namespace) -> int:
    """Create the config skeleton, refusing to overwrite without force."""
    resolved = paths.resolve_paths(args.config)
    _print_paths(resolved)

    config_exists = os.path.lexists(resolved.config_file)
    if args.dry_run:
        result = bootstrap(
            config_dir=resolved.config_dir,
            data_dir=resolved.data_dir,
            dry_run=True,
        )
        print("Dry run: no files written.")
        if args.force and config_exists:
            print(f"Would overwrite: {resolved.config_file}")
        for path in result.created:
            print(f"Would create: {path}")
        for path in result.preserved:
            if args.force and path == resolved.config_file:
                continue
            print(f"Would preserve: {path}")
        return 0

    if config_exists and not args.force:
        print(
            f"error: config already exists at {resolved.config_file}",
            file=sys.stderr,
        )
        print(
            "refusing to overwrite without --force; use --dry-run to preview",
            file=sys.stderr,
        )
        return 1

    if config_exists and args.force:
        try:
            resolved.config_file.unlink()
        except OSError as exc:
            print(
                f"error: cannot overwrite {resolved.config_file}: {exc}",
                file=sys.stderr,
            )
            return 1

    try:
        result = bootstrap(
            config_dir=resolved.config_dir,
            data_dir=resolved.data_dir,
        )
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    _print_result(
        result,
        overwrote=config_exists and args.force,
        config_file=resolved.config_file,
    )
    return 0


register_command(
    "init",
    run_init,
    help="Create the user config skeleton and print resolved paths",
    configure=configure,
)
