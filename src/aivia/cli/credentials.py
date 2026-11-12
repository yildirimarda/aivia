"""``aivia credentials``: reload secrets without restarting."""

from __future__ import annotations

import argparse
import sys

from aivia.cli.registry import register_command
from aivia.core.logging import get_logger
from aivia.core.secrets import SecretsLoader

logger = get_logger(__name__)


def _configure_reload(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--secrets-file",
        metavar="PATH",
        help="secrets file to reload instead of the default location",
    )


def _run_reload(args: argparse.Namespace) -> int:
    """Reload credentials and report without exposing secret values."""
    secrets_file = getattr(args, "secrets_file", None)
    loader = SecretsLoader(secrets_file=secrets_file)
    try:
        loader.reload()
    except Exception as exc:
        print(f"error: cannot reload credentials: {exc}", file=sys.stderr)
        return 1
    logger.info("credentials reloaded")
    print("credentials reloaded")
    return 0


def _configure(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(dest="credentials_command", metavar="COMMAND")
    reload_parser = subparsers.add_parser(
        "reload",
        help="Reload secrets without restarting",
        description="Reload secrets from the environment, secrets file and keyring",
    )
    _configure_reload(reload_parser)
    reload_parser.set_defaults(_credentials_handler=_run_reload)


def configure(parser: argparse.ArgumentParser) -> None:
    _configure(parser)


def run_credentials(args: argparse.Namespace) -> int:
    """Dispatch to the credentials subcommand."""
    handler = getattr(args, "_credentials_handler", None)
    if handler is None:
        print(
            "error: missing credentials command (use 'aivia credentials reload')",
            file=sys.stderr,
        )
        return 2
    return int(handler(args))


register_command(
    "credentials",
    run_credentials,
    help="Reload secrets without restarting",
    configure=configure,
)
