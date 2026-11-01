"""``aivia presets``: list and show named configuration presets.

The commands read the same layered configuration as ``config show``. They
resolve with the default preset disabled, so every configured definition stays
visible even when one of them is the default. ``show`` renders a preset as
deterministic YAML with secrets masked, which makes it safe to paste into an
issue or compare across environments.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import yaml

from aivia.cli.registry import register_command
from aivia.core.config.masking import mask_secrets
from aivia.core.config.resolve import resolve_config
from aivia.core.errors import ConfigError
from aivia.core.presets import (
    describe_preset,
    get_preset,
    iter_presets,
    presets_config,
)

__all__ = [
    "configure",
    "run_list",
    "run_presets",
    "run_show",
]


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


def _configure_list(parser: argparse.ArgumentParser) -> None:
    _configure_resolution(parser)
    parser.add_argument(
        "--all",
        "--include-disabled",
        dest="include_disabled",
        action="store_true",
        help="include presets when the presets section is disabled",
    )
    parser.set_defaults(_presets_handler=run_list)


def _configure_show(parser: argparse.ArgumentParser) -> None:
    _configure_resolution(parser)
    parser.add_argument("name", metavar="NAME", help="preset name to show")
    parser.add_argument(
        "--overrides",
        "--overrides-only",
        dest="overrides_only",
        action="store_true",
        help="print only the preset's overrides mapping",
    )
    parser.set_defaults(_presets_handler=run_show)


def configure(parser: argparse.ArgumentParser) -> None:
    """Add the presets group and its subcommands."""
    subparsers = parser.add_subparsers(dest="presets_command", metavar="COMMAND")
    list_parser = subparsers.add_parser(
        "list",
        help="List named presets and mark the default",
        description="List named presets and mark the default",
    )
    _configure_list(list_parser)
    show_parser = subparsers.add_parser(
        "show",
        help="Show one named preset as YAML",
        description="Show one named preset as YAML",
    )
    _configure_show(show_parser)


def _load(args: argparse.Namespace) -> Any:
    resolved = resolve_config(
        config_file=args.config,
        environment=args.environment,
        profile=args.profile,
        preset="",
        set_values=args.set_values,
    )
    return presets_config(resolved.values)


def run_list(args: argparse.Namespace) -> int:
    """Print every preset name, its description and whether it is the default."""
    try:
        section = _load(args)
        presets = iter_presets(
            section,
            include_disabled=bool(args.include_disabled),
        )
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print("NAME\tDESCRIPTION\tDEFAULT")
    for preset in presets:
        description = " ".join(preset.description.split()) or "-"
        default = "yes" if preset.name == section.default else "no"
        print(f"{preset.name}\t{description}\t{default}")
    return 0


def run_show(args: argparse.Namespace) -> int:
    """Print one preset as deterministic, masked YAML."""
    try:
        section = _load(args)
        preset = get_preset(section, args.name)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    payload: Any = preset.overrides if args.overrides_only else describe_preset(preset)
    text = yaml.safe_dump(
        mask_secrets(payload),
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )
    print(text, end="")
    return 0


def run_presets(args: argparse.Namespace) -> int:
    """Dispatch to the selected presets subcommand."""
    handler = getattr(args, "_presets_handler", None)
    if handler is None:
        print("error: missing presets command", file=sys.stderr)
        print("usage: aivia presets {list, show}", file=sys.stderr)
        return 2
    return int(handler(args))


register_command(
    "presets",
    run_presets,
    help="List named presets and show one as YAML",
    configure=configure,
)
