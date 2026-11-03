"""Command-line entry point for AIVIA."""

from __future__ import annotations

from aivia.cli import (
    api,
    compose,
    config,
    confirm,
    costs,
    init,
    media,
    presets,
    profile,
    prompts,
    providers,
    replay,
    research,
    resume,
    run,
    worker,
)
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
    "api",
    "build_parser",
    "compose",
    "config",
    "confirm",
    "costs",
    "init",
    "main",
    "media",
    "presets",
    "profile",
    "prompts",
    "providers",
    "register_command",
    "registry",
    "replay",
    "research",
    "resume",
    "run",
    "version_string",
    "worker",
]
