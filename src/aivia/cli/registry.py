"""Subcommand registry for the AIVIA root command line interface.

Later milestones extend the CLI by registering commands here instead of editing
the root parser. A command owns its name, help text, argument configuration and
handler, so modules can add functionality without importing the root module.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from aivia.cli.globals import add_global_options

__all__ = [
    "Command",
    "CommandHandler",
    "CommandRegistry",
    "ParserConfigurator",
    "register_command",
    "registry",
]

CommandHandler = Callable[[argparse.Namespace], int | None]
ParserConfigurator = Callable[[argparse.ArgumentParser], None]


@dataclass(frozen=True, slots=True)
class Command:
    """One registered subcommand."""

    name: str
    handler: CommandHandler
    help: str = ""
    configure: ParserConfigurator | None = None

    def add_parser(
        self,
        subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    ) -> argparse.ArgumentParser:
        """Attach this command's parser to the root subparsers action."""
        parser = subparsers.add_parser(
            self.name,
            help=self.help or None,
            description=self.help or None,
        )
        if self.configure is not None:
            self.configure(parser)
        add_global_options(parser, suppress_defaults=True, only_missing=True)
        parser.set_defaults(_command=self)
        return parser


class CommandRegistry:
    """An ordered, collision-checked collection of subcommands."""

    def __init__(self) -> None:
        self._commands: dict[str, Command] = {}

    def register(self, command: Command) -> Command:
        """Register a command, rejecting duplicate or malformed names."""
        name = command.name.strip()
        if not name:
            raise ValueError("command name must not be empty")
        if name.startswith("-") or any(character.isspace() for character in name):
            raise ValueError(f"invalid command name: {command.name!r}")
        if name in self._commands:
            raise ValueError(f"command already registered: {name}")
        stored = command
        if name != command.name:
            stored = Command(
                name=name,
                handler=command.handler,
                help=command.help,
                configure=command.configure,
            )
        self._commands[name] = stored
        return stored

    def command(
        self,
        name: str,
        *,
        help: str = "",
        configure: ParserConfigurator | None = None,
    ) -> Callable[[CommandHandler], CommandHandler]:
        """Return a decorator that registers the decorated handler."""

        def decorator(handler: CommandHandler) -> CommandHandler:
            self.register(
                Command(name=name, handler=handler, help=help, configure=configure)
            )
            return handler

        return decorator

    def register_command(
        self,
        name: str,
        handler: CommandHandler,
        *,
        help: str = "",
        configure: ParserConfigurator | None = None,
    ) -> Command:
        """Register a command from an explicit handler."""
        return self.register(
            Command(name=name, handler=handler, help=help, configure=configure)
        )

    def get(self, name: str) -> Command:
        """Return a registered command by name."""
        try:
            return self._commands[name]
        except KeyError:
            raise KeyError(f"unknown command: {name}") from None

    def names(self) -> tuple[str, ...]:
        """Return command names in deterministic, sorted order."""
        return tuple(sorted(self._commands))

    def add_to(
        self,
        subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    ) -> None:
        """Install every registered command into a subparsers action."""
        for name in self.names():
            self._commands[name].add_parser(subparsers)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._commands

    def __iter__(self) -> Iterator[Command]:
        return iter(self._commands[name] for name in self.names())

    def __len__(self) -> int:
        return len(self._commands)


registry = CommandRegistry()
register_command = registry.register_command
