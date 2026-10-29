"""``aivia api``: run the local HTTP API on a validated loopback address."""

from __future__ import annotations

import argparse
import sys

from aivia.api.app import serve
from aivia.api.security import validate_bind_host
from aivia.cli.registry import register_command
from aivia.core.config import ApiConfig, JobsConfig
from aivia.core.config.resolve import resolve_app_config
from aivia.core.errors import ConfigError

__all__ = [
    "configure",
    "run_api",
    "run_serve",
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


def _configure_serve(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--host",
        metavar="ADDRESS",
        help="loopback address to bind instead of the configured one",
    )
    parser.add_argument(
        "--port",
        type=int,
        metavar="PORT",
        help="port to listen on instead of the configured one",
    )
    _configure_resolution(parser)
    parser.set_defaults(_api_handler=run_serve)


def configure(parser: argparse.ArgumentParser) -> None:
    """Add the api group and its ``serve`` subcommand."""
    subparsers = parser.add_subparsers(dest="api_command", metavar="COMMAND")
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the local API on a loopback address",
        description="Start the local API on a loopback address",
    )
    _configure_serve(serve_parser)


def run_api(args: argparse.Namespace) -> int:
    """Dispatch to the selected api subcommand."""
    handler = getattr(args, "_api_handler", None)
    if handler is None:
        print("error: missing api command", file=sys.stderr)
        print("usage: aivia api serve", file=sys.stderr)
        return 2
    return int(handler(args))


def _resolve_config(args: argparse.Namespace) -> tuple[ApiConfig, JobsConfig]:
    config = resolve_app_config(
        config_file=getattr(args, "config", None),
        environment=getattr(args, "environment", None),
        profile=getattr(args, "profile", None),
        set_values=getattr(args, "set_values", None),
    )
    section = getattr(config, "api", None)
    api = section if isinstance(section, ApiConfig) else ApiConfig()
    jobs_section = getattr(config, "jobs", None)
    jobs = jobs_section if isinstance(jobs_section, JobsConfig) else JobsConfig()

    overrides: dict[str, object] = {}
    host = getattr(args, "host", None)
    port = getattr(args, "port", None)
    if host is not None:
        overrides["host"] = host
    if port is not None:
        overrides["port"] = port
    if overrides:
        api = ApiConfig.from_mapping({**api.model_dump(), **overrides})
    return api, jobs


def _resolve_api(args: argparse.Namespace) -> ApiConfig:
    return _resolve_config(args)[0]


def run_serve(args: argparse.Namespace) -> int:
    """Validate the bind address and run the local API until it stops."""
    try:
        api, jobs = _resolve_config(args)
        validate_bind_host(api.host)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Serving AIVIA API on http://{api.host}:{api.port}", flush=True)
    serve(
        api,
        jobs=jobs,
        environment=getattr(args, "environment", None),
        profile=getattr(args, "profile", None),
    )
    return 0


register_command(
    "api",
    run_api,
    help="Run the local HTTP API on a loopback address",
    configure=configure,
)
