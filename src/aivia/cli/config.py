"""``aivia config``: inspect and validate the layered configuration."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from aivia.cli.registry import register_command
from aivia.core import config_show, model_ids, paths
from aivia.core.config import AppConfig, parse_app_config
from aivia.core.config import docs as config_docs
from aivia.core.config import migrations as config_migrations
from aivia.core.config import schema as config_schema
from aivia.core.config.resolve import load_config_file
from aivia.core.errors import ConfigError
from aivia.llm import PromptLibrary

__all__ = [
    "configure",
    "run_config",
    "run_docs",
    "run_migrate",
    "run_schema",
    "run_show",
    "run_validate",
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


def _configure_show(parser: argparse.ArgumentParser) -> None:
    _configure_resolution(parser)
    parser.add_argument(
        "--resolved",
        action="store_true",
        help="show the fully merged configuration (the default)",
    )
    parser.add_argument(
        "--origins",
        action="store_true",
        help="print the layer that supplied each value",
    )
    parser.set_defaults(_config_handler=run_show)


def _configure_validate(parser: argparse.ArgumentParser) -> None:
    _configure_resolution(parser)
    parser.set_defaults(_config_handler=run_validate)


def _configure_schema(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="write the schema to PATH instead of standard output",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when the schema at PATH (or the packaged copy) is stale",
    )
    parser.set_defaults(_config_handler=run_schema)


def _configure_docs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="write the reference to PATH instead of standard output",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when the reference at PATH (or docs/CONFIG.md) is stale",
    )
    parser.set_defaults(_config_handler=run_docs)


def _configure_migrate(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        metavar="PATH",
        help="config file to migrate instead of the default location",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the diff and the planned version without writing",
    )
    parser.set_defaults(_config_handler=run_migrate)


def configure(parser: argparse.ArgumentParser) -> None:
    """Add the config group and its subcommands."""
    subparsers = parser.add_subparsers(dest="config_command", metavar="COMMAND")
    show = subparsers.add_parser(
        "show",
        help="Print the merged configuration with profile and origin tracking",
        description="Print the merged configuration with profile and origin tracking",
    )
    _configure_show(show)
    validate = subparsers.add_parser(
        "validate",
        help="Validate the merged configuration and print every error",
        description="Validate the merged configuration and print every error",
    )
    _configure_validate(validate)
    schema = subparsers.add_parser(
        "schema",
        help="Export the configuration JSON Schema",
        description="Export the configuration JSON Schema",
    )
    _configure_schema(schema)
    docs = subparsers.add_parser(
        "docs",
        help="Generate the configuration reference document",
        description="Generate the configuration reference document",
    )
    _configure_docs(docs)
    migrate = subparsers.add_parser(
        "migrate",
        help="Upgrade an older user config and print a diff",
        description="Upgrade an older user config and print a diff",
    )
    _configure_migrate(migrate)


def run_config(args: argparse.Namespace) -> int:
    """Dispatch to the selected config subcommand."""
    handler = getattr(args, "_config_handler", None)
    if handler is None:
        print("error: missing config command", file=sys.stderr)
        print(
            "usage: aivia config {show, validate, schema, docs, migrate}",
            file=sys.stderr,
        )
        return 2
    return int(handler(args))


def run_docs(args: argparse.Namespace) -> int:
    """Print, write or check the generated configuration reference."""
    text = config_docs.dump_config_docs()
    target = Path(args.output) if args.output is not None else config_docs.docs_path()
    if args.check:
        try:
            existing = target.read_text(encoding="utf-8")
        except OSError:
            existing = None
        if existing != text:
            print(
                f"error: configuration reference is stale: {target}",
                file=sys.stderr,
            )
            print(
                f"regenerate it with: aivia config docs --output {target}",
                file=sys.stderr,
            )
            return 1
        print(f"configuration reference is up to date: {target}")
        return 0
    if args.output is None:
        print(text, end="")
        return 0
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    except OSError as exc:
        print(f"error: cannot write {target}: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {target}")
    return 0


def run_schema(args: argparse.Namespace) -> int:
    """Print, write or check the deterministic configuration JSON Schema."""
    text = config_schema.dump_config_schema()
    target = (
        Path(args.output)
        if args.output is not None
        else config_schema.packaged_schema_path()
    )
    if args.check:
        try:
            existing = target.read_text(encoding="utf-8")
        except OSError:
            existing = None
        if existing != text:
            print(
                f"error: configuration schema is stale: {target}",
                file=sys.stderr,
            )
            print(
                f"regenerate it with: aivia config schema --output {target}",
                file=sys.stderr,
            )
            return 1
        print(f"configuration schema is up to date: {target}")
        return 0
    if args.output is None:
        print(text, end="")
        return 0
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    except OSError as exc:
        print(f"error: cannot write {target}: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {target}")
    return 0


def run_migrate(args: argparse.Namespace) -> int:
    """Upgrade the user config, printing the diff and preserving unknown keys."""
    config_file = paths.resolve_config_file(args.config)
    if not config_file.exists():
        print(f"error: config file not found: {config_file}", file=sys.stderr)
        return 1
    try:
        original = config_file.read_text(encoding="utf-8")
        values = load_config_file(config_file, origin="user")
        result = config_migrations.migrate_config(values)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: cannot read {config_file}: {exc}", file=sys.stderr)
        return 1

    if not result.changed:
        print(f"{config_file}: already at schema_version {result.to_version}")
        return 0

    migrated = config_migrations.dump_migrated_config(
        result.values,
        before=values,
        original_text=original,
    )
    diff = config_migrations.render_migration_diff(
        str(config_file),
        original,
        migrated,
        from_version=result.from_version,
        to_version=result.to_version,
    )
    if diff:
        print(diff, end="" if diff.endswith("\n") else "\n")

    if args.dry_run:
        print(
            f"Dry run: no files written. Would migrate {config_file}: "
            f"schema_version {result.from_version} -> {result.to_version}"
        )
        return 0

    try:
        config_file.write_text(migrated, encoding="utf-8")
    except OSError as exc:
        print(f"error: cannot write {config_file}: {exc}", file=sys.stderr)
        return 1
    print(
        f"Migrated {config_file}: schema_version "
        f"{result.from_version} -> {result.to_version}"
    )
    return 0


def run_show(args: argparse.Namespace) -> int:
    """Print the merged configuration, masking secrets."""
    try:
        resolved = config_show.resolve_config(
            config_file=args.config,
            environment=args.environment,
            profile=args.profile,
            preset=args.preset,
            set_values=args.set_values,
        )
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        config_show.dump_config(resolved, include_origins=args.origins),
        end="",
    )
    return 0


def _error_path(details: Mapping[str, Any], message: str) -> str:
    path = details.get("path")
    if isinstance(path, str) and path:
        return path
    preset = details.get("preset")
    if isinstance(preset, str):
        if "unknown platform account" in message:
            return f"presets.{preset}.accounts"
        if "unknown format" in message:
            return f"presets.{preset}.format"
        if "unknown profile" in message:
            return f"presets.{preset}.profile"
        return f"presets.{preset}"
    schedule = details.get("schedule")
    if isinstance(schedule, str):
        return f"schedule.schedules.{schedule}.profiles"
    profile = details.get("profile")
    if isinstance(profile, str):
        if f"profile {profile} references unknown platform account" in message:
            return f"profiles.{profile}.accounts"
        if f"profile {profile} references unknown format" in message:
            return f"profiles.{profile}.format"
    account = details.get("account")
    if isinstance(account, str):
        if "unknown credential" in message:
            return f"publish.accounts.{account}.credential_refs"
        if "unknown profile" in message:
            return f"publish.accounts.{account}.profile"
        return f"publish.accounts.{account}"
    if isinstance(profile, str):
        return "publish.default_profile"
    return ""


def _error_items(error: ConfigError) -> list[tuple[str, str]]:
    """Return every issue carried by *error* as a ``(path, message)`` pair."""
    details = error.details
    raw_errors = details.get("errors")
    if isinstance(raw_errors, Sequence) and not isinstance(raw_errors, (str, bytes)):
        items: list[tuple[str, str]] = []
        for entry in raw_errors:
            if not isinstance(entry, Mapping):
                continue
            path = entry.get("path")
            message = entry.get("message")
            items.append(
                (
                    str(path) if path is not None else "",
                    str(message) if message is not None else str(error),
                )
            )
        if items:
            return items
    unknown = details.get("unknown")
    if isinstance(unknown, Sequence) and not isinstance(unknown, (str, bytes)):
        keys = [str(key) for key in unknown]
        if keys:
            return [(key, f"unknown config section: {key}") for key in keys]
    message = str(error)
    return [(_error_path(details, message), message)]


def _format_error(path: str, message: str) -> str:
    if path and path not in message:
        return f"{path}: {message}"
    return message


def _schema_version_issue(values: Mapping[str, Any]) -> tuple[str, str] | None:
    if "schema_version" not in values:
        return None
    version = values["schema_version"]
    if isinstance(version, int) and not isinstance(version, bool) and version >= 1:
        if version > config_migrations.CURRENT_SCHEMA_VERSION:
            return (
                "schema_version",
                f"config schema version {version} is newer than supported "
                f"version {config_migrations.CURRENT_SCHEMA_VERSION}; "
                "upgrade aivia",
            )
        return None
    return (
        "schema_version",
        "invalid value for schema_version: "
        f"expected a positive integer, got {version!r}",
    )


def _prompt_version_issues(
    parsed: Any,
    config_dir: Path,
) -> list[tuple[str, str]]:
    """Check every selected prompt version against the available prompt files."""
    prompts = getattr(parsed, "prompts", None)
    if prompts is None:
        return []
    library = PromptLibrary(config_dir=config_dir)
    issues: list[tuple[str, str]] = []
    for name, source in sorted(prompts.versions.items()):
        try:
            library.resolve(name, source)
        except ConfigError as exc:
            issues.append((f"prompts.versions.{name}", str(exc)))
    return issues


def run_validate(args: argparse.Namespace) -> int:
    """Validate the merged configuration and report every error with its path."""
    try:
        resolved = config_show.resolve_config(
            config_file=args.config,
            environment=args.environment,
            profile=args.profile,
            preset=args.preset,
            set_values=args.set_values,
        )
    except ConfigError as exc:
        for path, message in _error_items(exc):
            print(f"error: {_format_error(path, message)}", file=sys.stderr)
        return 1

    values = dict(resolved.values)
    issues: list[tuple[str, str]] = []
    schema_issue = _schema_version_issue(values)
    if schema_issue is not None:
        issues.append(schema_issue)
    values.pop("schema_version", None)

    registered = set(AppConfig.model_fields)
    unknown = sorted(str(key) for key in values if key not in registered)
    if unknown:
        available = ", ".join(sorted(registered)) or "none"
        for key in unknown:
            issues.append(
                (key, f"unknown config section: {key} (registered: {available})")
            )
            values.pop(key)

    try:
        parsed = parse_app_config(values, context=resolved.error_context())
    except ConfigError as exc:
        issues.extend(_error_items(exc))
    else:
        config_dir = (
            Path(args.config).expanduser().resolve().parent
            if getattr(args, "config", None)
            else paths.config_dir()
        )
        try:
            model_ids.warn_model_ids(parsed, config_dir=config_dir)
        except ConfigError as exc:
            issues.append(("models_file", str(exc)))
        issues.extend(_prompt_version_issues(parsed, config_dir))

    if issues:
        for path, message in issues:
            print(f"error: {_format_error(path, message)}", file=sys.stderr)
        return 1
    print("configuration is valid")
    return 0


register_command(
    "config",
    run_config,
    help="Inspect the layered configuration",
    configure=configure,
)
