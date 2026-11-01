"""``aivia prompts``: list prompt versions and the selected one.

The selection comes from the layered configuration, so ``--profile`` and
``--set`` work exactly as they do for ``config show``. Prompts are read beside
the selected config file, falling back to the packaged defaults.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from aivia.cli.registry import register_command
from aivia.core import config_show, paths
from aivia.core.config.prompts import PromptsConfig, PromptVersionSource
from aivia.core.errors import ConfigError
from aivia.llm import PROMPT_EXTENSIONS, PromptLibrary, PromptSelection

__all__ = [
    "configure",
    "run_list",
    "run_prompts",
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
        "names",
        nargs="*",
        metavar="NAME",
        help="only list these prompts (default: every visible prompt)",
    )
    parser.add_argument(
        "--paths",
        action="store_true",
        help="show the selected file paths instead of yes/no",
    )
    parser.add_argument(
        "--all",
        "--versions",
        dest="all_versions",
        action="store_true",
        help="print one row per user and default version",
    )
    parser.set_defaults(_prompts_handler=run_list)


def _configure_show(parser: argparse.ArgumentParser) -> None:
    _configure_resolution(parser)
    parser.add_argument("name", metavar="NAME", help="prompt name to show")
    parser.add_argument(
        "--source",
        "--version",
        dest="source",
        choices=[source.value for source in PromptVersionSource],
        help="override the configured version for this lookup",
    )
    parser.add_argument(
        "--content",
        action="store_true",
        help="print the prompt text instead of its path",
    )
    parser.set_defaults(_prompts_handler=run_show)


def configure(parser: argparse.ArgumentParser) -> None:
    """Add the prompts group and its subcommands."""
    subparsers = parser.add_subparsers(dest="prompts_command", metavar="COMMAND")
    list_parser = subparsers.add_parser(
        "list",
        help="List prompts with their user, default and selected versions",
        description="List prompts with their user, default and selected versions",
    )
    _configure_list(list_parser)
    show_parser = subparsers.add_parser(
        "show",
        help="Show the selected version of one prompt",
        description="Show the selected version of one prompt",
    )
    _configure_show(show_parser)
    parser.set_defaults(
        _prompts_handler=run_list,
        names=[],
        paths=False,
        all_versions=False,
    )


def _library(args: argparse.Namespace) -> PromptLibrary:
    if args.config is not None:
        config_dir = Path(args.config).expanduser().resolve().parent
    else:
        config_dir = paths.config_dir()
    return PromptLibrary(config_dir=config_dir)


def _selections(args: argparse.Namespace) -> dict[str, PromptVersionSource]:
    resolved = config_show.resolve_config(
        config_file=args.config,
        environment=args.environment,
        profile=args.profile,
        preset=args.preset,
        set_values=args.set_values,
    )
    section = resolved.values.get("prompts")
    if section is None:
        return {}
    if not isinstance(section, Mapping):
        raise ConfigError("prompts must be a mapping")
    return PromptsConfig.from_mapping(section).versions


def _candidate_keys(name: str) -> tuple[str, ...]:
    keys = [name]
    for extension in PROMPT_EXTENSIONS:
        if name.endswith(extension):
            keys.append(name[: -len(extension)])
    return tuple(keys)


def _source_for(
    selections: Mapping[str, PromptVersionSource],
    name: str,
) -> PromptVersionSource:
    for key in _candidate_keys(name):
        if key in selections:
            return selections[key]
    return PromptVersionSource.AUTO


def _logical_name(name: str) -> str:
    for extension in PROMPT_EXTENSIONS:
        if name.endswith(extension):
            return name[: -len(extension)]
    return name


def _prompt_groups(library: PromptLibrary) -> dict[str, tuple[str, ...]]:
    groups: dict[str, list[str]] = {}
    for name in library.names():
        groups.setdefault(_logical_name(name), []).append(name)
    return {name: tuple(files) for name, files in groups.items()}


def _requested_names(
    library: PromptLibrary,
    requested: Sequence[str],
) -> tuple[str, ...]:
    groups = _prompt_groups(library)
    available = library.names()
    if not requested:
        return tuple(groups)
    selected: list[str] = []
    for want in requested:
        matches = [
            name for name, files in groups.items() if want == name or want in files
        ]
        if not matches:
            raise ConfigError(
                f"unknown prompt: {want} (available: {', '.join(available) or 'none'})"
            )
        for name in matches:
            if name not in selected:
                selected.append(name)
    return tuple(selected)


def _check_selections(
    library: PromptLibrary,
    names: Sequence[str],
    selections: Mapping[str, PromptVersionSource],
) -> None:
    known = {key for name in names for key in _candidate_keys(name)}
    known.update(library.names())
    unknown = sorted(key for key in selections if key not in known)
    if unknown:
        raise ConfigError(
            "unknown prompt version: "
            f"{', '.join(unknown)} (available: {', '.join(names) or 'none'})"
        )


def _yes_no(value: object) -> str:
    return "yes" if value is not None else "no"


def _print_selection(selection: PromptSelection, *, paths: bool) -> None:
    user = selection.user.path if paths and selection.user is not None else None
    default = (
        selection.default.path if paths and selection.default is not None else None
    )
    user_text = str(user) if user is not None else _yes_no(selection.user)
    default_text = str(default) if default is not None else _yes_no(selection.default)
    print(f"{selection.name}\t{user_text}\t{default_text}\t{selection.source}")


def run_list(args: argparse.Namespace) -> int:
    """Print every prompt and its user, default and selected versions."""
    try:
        library = _library(args)
        selections = _selections(args)
        names = _requested_names(library, args.names)
        _check_selections(library, names, selections)
        rendered: list[PromptSelection] = [
            library.selection(name, _source_for(selections, name)) for name in names
        ]
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.all_versions:
        print("NAME\tSOURCE\tSELECTED\tPATH")
        for selection in rendered:
            for version in selection.versions:
                selected = "yes" if version.path == selection.selected.path else "no"
                print(f"{selection.name}\t{version.source}\t{selected}\t{version.path}")
        return 0

    print("NAME\tUSER\tDEFAULT\tSELECTED")
    for selection in rendered:
        _print_selection(selection, paths=args.paths)
    return 0


def run_show(args: argparse.Namespace) -> int:
    """Print the selected version of one prompt."""
    try:
        library = _library(args)
        selections = _selections(args)
        source = args.source or _source_for(selections, args.name)
        selection = library.selection(args.name, source)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.content:
        print(selection.selected.text, end="")
    else:
        print(f"{selection.name}\t{selection.source}\t{selection.selected.path}")
    return 0


def run_prompts(args: argparse.Namespace) -> int:
    """Dispatch to the selected prompts subcommand, defaulting to list."""
    handler = getattr(args, "_prompts_handler", run_list)
    return int(handler(args))


register_command(
    "prompts",
    run_prompts,
    help="List prompt versions and select one per config or profile",
    configure=configure,
)
