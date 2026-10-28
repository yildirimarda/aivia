"""Command-line entry point for AIVIA."""

from __future__ import annotations

from aivia.cli import config, init, providers, run
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
    "ParserConfigurator",
    "build_parser",
    "config",
    "init",
    "main",
    "providers",
    "register_command",
    "registry",
    "run",
    "version_string",
]
