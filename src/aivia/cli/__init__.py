"""Command-line entry point for AIVIA."""

from __future__ import annotations

from aivia.cli import config, confirm, costs, init, providers, resume, run
from aivia.cli.globals import GlobalOptions
from aivia.cli.registry import (
    Command,
    CommandHandler,
    CommandRegistry,
    ParserConfigurator,
    register_command,
    registry,
)
from aivia.cli.root import build_parser, main, version_string

__all__ = [
    "Command",
    "CommandHandler",
    "CommandRegistry",
    "GlobalOptions",
    "ParserConfigurator",
    "build_parser",
    "config",
    "confirm",
    "costs",
    "init",
    "main",
    "providers",
    "register_command",
    "registry",
    "resume",
    "run",
    "version_string",
]
