"""Global options accepted before or after any subcommand.

The root parser owns the help text, but a small parser also recognises the
global flags wherever they appear in the argument list. That parser removes
them before the command parser runs, so ``aivia --verbose run`` and
``aivia run --verbose`` mean the same thing, and nested commands such as
``aivia config show --profile short`` keep working without every command
redeclaring the flags.

Commands read the merged values from their :class:`argparse.Namespace` as
usual; :attr:`argparse.Namespace.global_options` holds the same values as one
typed object for code that prefers that shape.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "GlobalOptions",
    "add_global_options",
    "apply_global_options",
    "build_global_parser",
]

_GLOBAL_DESTINATIONS = (
    "config",
    "environment",
    "profile",
    "set_values",
    "estimate",
    "dry_run",
    "yes",
    "no_cache",
    "live",
    "verbose",
)


@dataclass(frozen=True, slots=True)
class GlobalOptions:
    """The options that apply to every command."""

    config: str | None = None
    environment: str | None = None
    profile: str | None = None
    set_values: tuple[str, ...] = ()
    estimate: bool = False
    dry_run: bool = False
    yes: bool = False
    no_cache: bool = False
    live: bool = False
    verbose: bool = False

    @classmethod
    def from_namespace(cls, namespace: argparse.Namespace) -> GlobalOptions:
        """Read global options from a parsed namespace, tolerating omissions."""
        raw_set = getattr(namespace, "set_values", None)
        if isinstance(raw_set, str):
            set_values: tuple[str, ...] = (raw_set,)
        elif raw_set:
            set_values = tuple(str(value) for value in raw_set)
        else:
            set_values = ()
        return cls(
            config=_text(getattr(namespace, "config", None)),
            environment=_text(getattr(namespace, "environment", None)),
            profile=_text(getattr(namespace, "profile", None)),
            set_values=set_values,
            estimate=bool(getattr(namespace, "estimate", False)),
            dry_run=bool(getattr(namespace, "dry_run", False)),
            yes=bool(getattr(namespace, "yes", False)),
            no_cache=bool(getattr(namespace, "no_cache", False)),
            live=bool(getattr(namespace, "live", False)),
            verbose=bool(getattr(namespace, "verbose", False)),
        )

    def resolution_options(self) -> dict[str, Any]:
        """Return the keywords accepted by :func:`resolve_app_config`."""
        return {
            "config_file": self.config,
            "environment": self.environment,
            "profile": self.profile,
            "set_values": list(self.set_values),
        }


def _text(value: object) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _default(value: Any, *, suppress: bool) -> Any:
    return argparse.SUPPRESS if suppress else value


def _existing_options(parser: argparse.ArgumentParser) -> set[str]:
    options: set[str] = set()
    for action in parser._actions:
        options.update(action.option_strings)
    return options


def add_global_options(
    parser: argparse.ArgumentParser,
    *,
    suppress_defaults: bool = False,
    only_missing: bool = False,
) -> None:
    """Add the global flags to *parser*.

    ``suppress_defaults`` leaves an attribute unset when the flag is absent, so
    a command parser cannot overwrite a value already parsed by the root.
    ``only_missing`` skips a group whose option strings a command already
    defines, which keeps command-specific aliases working.
    """
    existing = _existing_options(parser) if only_missing else set()

    def missing(*option_strings: str) -> bool:
        return not only_missing or not any(
            option in existing for option in option_strings
        )

    def default(value: Any) -> Any:
        return _default(value, suppress=suppress_defaults)

    if missing("--config"):
        parser.add_argument(
            "--config",
            metavar="PATH",
            default=default(None),
            help="config file to merge instead of the default location",
        )
    if missing("--env", "--environment"):
        parser.add_argument(
            "--env",
            "--environment",
            dest="environment",
            metavar="NAME",
            default=default(None),
            help="merge config.<NAME>.yaml beside the user config",
        )
    if missing("--profile"):
        parser.add_argument(
            "--profile",
            metavar="NAME",
            default=default(None),
            help="merge the overrides of a configured profile",
        )
    if missing("--set"):
        parser.add_argument(
            "--set",
            dest="set_values",
            action="append",
            metavar="KEY=VALUE",
            default=default(None),
            help="override a dotted key; repeatable and applied after the profile",
        )
    if missing("--estimate"):
        parser.add_argument(
            "--estimate",
            action="store_true",
            default=default(False),
            help="print the cost estimate and exit without running anything",
        )
    if missing("--dry-run"):
        parser.add_argument(
            "--dry-run",
            action="store_true",
            default=default(False),
            help="preview the work without making paid calls or writing outputs",
        )
    if missing("-y", "--yes"):
        parser.add_argument(
            "-y",
            "--yes",
            action="store_true",
            default=default(False),
            help="confirm consequential actions without prompting",
        )
    if missing("--no-cache"):
        parser.add_argument(
            "--no-cache",
            action="store_true",
            default=default(False),
            help="ignore cached assets and generated responses",
        )
    if missing("--live"):
        parser.add_argument(
            "--live",
            action="store_true",
            default=default(False),
            help="allow calls to external services, subject to the live budget",
        )
    if missing("-v", "--verbose"):
        parser.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            default=default(False),
            help="print verbose diagnostic logging",
        )


def build_global_parser(*, prog: str = "aivia") -> argparse.ArgumentParser:
    """Build a parser that only recognises global flags, for argv filtering."""
    parser = argparse.ArgumentParser(prog=prog, add_help=False)
    add_global_options(parser, suppress_defaults=True)
    return parser


def apply_global_options(
    namespace: argparse.Namespace,
    provided: argparse.Namespace,
) -> argparse.Namespace:
    """Copy explicitly provided globals onto *namespace* and type them."""
    for destination in _GLOBAL_DESTINATIONS:
        if hasattr(provided, destination):
            setattr(namespace, destination, getattr(provided, destination))
    namespace.global_options = GlobalOptions.from_namespace(namespace)
    return namespace


def split_global_options(
    argv: Sequence[str],
    *,
    prog: str = "aivia",
) -> tuple[argparse.Namespace, list[str]]:
    """Return the explicitly provided globals and the command arguments."""
    parser = build_global_parser(prog=prog)
    return parser.parse_known_args(list(argv))
