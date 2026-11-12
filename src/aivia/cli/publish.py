"""``aivia publish dry-run`` and ``aivia publish check``: pre-flight publish validation.

Both commands validate every dimension that can fail at publish time -- tokens,
media, formatting, compatibility, scopes and remaining quota -- without calling
a platform API or spending money. ``dry-run`` additionally submits a dry-run
publish request through the matched provider to confirm it accepts the request.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aivia.cli.registry import register_command
from aivia.core import paths
from aivia.core.budget import BudgetGuard
from aivia.core.config.budget import BudgetConfig
from aivia.core.config.providers import ProvidersConfig
from aivia.core.config.resolve import resolve_config
from aivia.core.config.schedule import PublishAccountConfig, PublishConfig
from aivia.core.cost import CostTracker
from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.dry_run import apply_dry_run
from aivia.core.errors import AiviaError, ConfigError
from aivia.core.platforms import (
    CaptionLimits,
    Platform,
    PlatformCatalog,
    load_platforms,
)
from aivia.core.secrets import MissingSecretError, SecretsLoader
from aivia.formats import load_packaged_formats
from aivia.providers.protocols.publish import (
    CAPABILITY_PUBLISH,
    PublishMedia,
    PublishRequest,
)
from aivia.providers.provider_factory import build_providers

__all__ = ["configure", "run_publish"]

_PUBLISH_COMMAND_HELP = "Validate a publish target before posting."


@dataclass(frozen=True, slots=True)
class CheckResult:
    """The outcome of one publish pre-flight check."""

    name: str
    """The check dimension: tokens, media, formatting, compatibility, scopes, quota."""

    ok: bool
    """Whether the check passed."""

    message: str = ""
    """A human readable description of the result."""

    detail: str = ""
    """Optional supporting detail printed beneath the result line."""


@dataclass(slots=True)
class _Runtime:
    config: Any
    publish: PublishConfig
    platforms: PlatformCatalog
    providers: dict[str, Any]
    secrets: SecretsLoader
    budget: BudgetGuard | None
    engine: Any | None


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


def _configure_database(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--database",
        "--db",
        dest="database",
        metavar="PATH",
        help="database file to use instead of the one in the user data directory",
    )


def _configure_publish(parser: argparse.ArgumentParser) -> None:
    _configure_resolution(parser)
    _configure_database(parser)
    parser.add_argument(
        "--platform",
        metavar="NAME",
        help="platform to publish to (x, youtube, instagram, tiktok, telegram)",
    )
    parser.add_argument(
        "--account",
        metavar="KEY",
        help=(
            "publish account key (platform/language); defaults to the first "
            "account for the platform"
        ),
    )
    parser.add_argument(
        "--language",
        metavar="CODE",
        help="language used with --platform to resolve the publish account",
    )
    parser.add_argument(
        "--media",
        metavar="PATH",
        help="media file to validate against the platform media limits",
    )
    parser.add_argument(
        "--caption",
        metavar="TEXT",
        help="caption text to validate against the platform caption limit",
    )
    parser.add_argument(
        "--title",
        metavar="TEXT",
        help="title text to validate against the platform title limit",
    )
    parser.add_argument(
        "--format",
        metavar="NAME",
        help=(
            "format recipe whose output type is checked against the platform "
            "media types"
        ),
    )


def configure(parser: argparse.ArgumentParser) -> None:
    """Add the ``publish`` group with ``dry-run`` and ``check`` commands."""
    subparsers = parser.add_subparsers(dest="publish_command", metavar="COMMAND")
    for name in ("dry-run", "check"):
        command = subparsers.add_parser(
            name,
            help="Validate a publish target before posting",
            description="Pre-flight validation of tokens, media, formatting, "
            "compatibility, scopes and remaining quota.",
        )
        _configure_publish(command)
        command.set_defaults(_publish_command=name)
    parser.set_defaults(_publish_handler=run_publish)


def _load_config(args: argparse.Namespace) -> Any:
    resolved = resolve_config(
        config_file=args.config,
        environment=args.environment,
        profile=args.profile,
        preset=getattr(args, "preset", None),
        set_values=args.set_values,
    )
    config = resolved.to_app_config()
    dry_run = (
        bool(getattr(args, "dry_run", False))
        or getattr(args, "_publish_command", "") == "dry-run"
    )
    if dry_run:
        config = apply_dry_run(config)
    return config


def _providers_config(config: Any) -> ProvidersConfig:
    section = getattr(config, "providers", None)
    return section if isinstance(section, ProvidersConfig) else ProvidersConfig()


def _database_path(args: argparse.Namespace, config: Any) -> Path:
    explicit = getattr(args, "database", None)
    if explicit:
        return Path(explicit).expanduser()
    from aivia.core.config.database import DatabaseConfig

    database = getattr(config, "database", None)
    if not isinstance(database, DatabaseConfig):
        database = None
    filename = database.filename if database is not None else DatabaseConfig().filename
    return paths.resolve_paths(args.config).data_dir / filename


def _open_database(path: Path) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_database_engine(path=path)
    try:
        upgrade_database(engine=engine)
    except BaseException:
        engine.dispose()
        raise
    return engine


def _budget_guard(config: Any, engine: Any | None) -> BudgetGuard | None:
    budget_config = getattr(config, "budget", None)
    if not isinstance(budget_config, BudgetConfig):
        return None
    tracker: Any = None
    if engine is not None:
        tracker = CostTracker(engine)
    return BudgetGuard(budget_config, tracker)


def _build_runtime(args: argparse.Namespace) -> _Runtime:
    config = _load_config(args)
    publish = getattr(config, "publish", None)
    if not isinstance(publish, PublishConfig):
        publish = PublishConfig()
    platforms = load_platforms()
    section = _providers_config(config)
    providers = build_providers(section)
    secrets = SecretsLoader()
    database_path = _database_path(args, config)
    engine: Any = None
    try:
        engine = _open_database(database_path)
    except (OSError, ConfigError):
        engine = None
    budget = _budget_guard(config, engine)
    return _Runtime(
        config=config,
        publish=publish,
        platforms=platforms,
        providers=providers,
        secrets=secrets,
        budget=budget,
        engine=engine,
    )


def _resolve_account(
    runtime: _Runtime,
    platform_name: str | None,
    account_key: str | None,
    language: str | None,
) -> tuple[str, PublishAccountConfig, Platform]:
    """Return (account_key, publish_config, platform) for the publish target."""
    publish = runtime.publish
    if not publish.accounts:
        raise ConfigError(
            "no publish accounts are configured; add a publish section",
            details={"section": "publish"},
        )
    if account_key:
        if account_key not in publish.names:
            raise ConfigError(
                f"unknown publish account: {account_key!r}",
                details={"account": account_key, "available": sorted(publish.names)},
            )
        account = publish.accounts[account_key]
        platform_name = account.platform or platform_name
    elif platform_name:
        matches = [
            key
            for key in publish.names
            if key.partition("/")[0] == platform_name.strip()
        ]
        if not matches:
            raise ConfigError(
                f"no publish account configured for platform {platform_name!r}",
                details={
                    "platform": platform_name,
                    "available": sorted(publish.names),
                },
            )
        if language:
            matches = [
                key for key in matches if key.partition("/")[2] == language.strip()
            ]
            if not matches:
                raise ConfigError(
                    f"no publish account for {platform_name!r}/{language!r}",
                    details={
                        "platform": platform_name,
                        "language": language,
                        "available": sorted(publish.names),
                    },
                )
        account_key = matches[0]
        account = publish.accounts[account_key]
    else:
        account_key = next(iter(publish.names))
        account = publish.accounts[account_key]
        platform_name = account.platform

    if not platform_name:
        raise ConfigError(
            "publish account has no platform",
            details={"account": account_key},
        )
    try:
        platform = runtime.platforms.require(platform_name)
    except KeyError as exc:
        raise ConfigError(
            f"unknown platform: {platform_name!r}",
            details={
                "platform": platform_name,
                "available": sorted(runtime.platforms.ids()),
            },
        ) from exc
    return account_key, account, platform


def _check_tokens(runtime: _Runtime, account: Any) -> CheckResult:
    """Resolve every credential reference for the account."""
    missing: list[str] = []
    for reference in account.credential_refs:
        try:
            runtime.secrets.get(reference)
        except MissingSecretError:
            missing.append(reference)
        except ConfigError as exc:
            missing.append(f"{reference} ({exc})")
    if missing:
        names = ", ".join(missing)
        return CheckResult(
            "tokens",
            False,
            f"unresolvable credential reference(s) for account: {names}",
        )
    count = len(account.credential_refs)
    return CheckResult(
        "tokens",
        True,
        f"{count} credential reference(s) resolved",
        detail=", ".join(account.credential_refs) or "(none configured)",
    )


def _file_size(path: Path) -> int:
    return path.stat().st_size


def _check_media(
    runtime: _Runtime, media_path: str | None, platform: Platform
) -> CheckResult:
    """Validate the media file against the platform media limits."""
    if not media_path:
        return CheckResult("media", True, "no media supplied; media check skipped")
    path = Path(media_path).expanduser()
    if not path.is_file():
        return CheckResult(
            "media",
            False,
            f"media file not found: {path}",
            detail=str(path),
        )
    size = _file_size(path)
    problems: list[str] = []
    media_type = (
        "video"
        if str(path.suffix).lower() in {".mp4", ".mov", ".mkv", ".webm"}
        else "image"
    )
    media_limits = platform.limit(media_type)
    if (
        media_limits is not None
        and media_limits.max_file_size_bytes is not None
        and size > media_limits.max_file_size_bytes
    ):
        problems.append(
            f"media size {size} bytes exceeds the {media_type} limit of "
            f"{media_limits.max_file_size_bytes} bytes"
        )
    if problems:
        return CheckResult("media", False, "; ".join(problems), detail=str(path))
    return CheckResult(
        "media",
        True,
        f"media fits the {media_type} limits ({size} bytes)",
        detail=str(path),
    )


def _check_formatting(
    caption: str | None,
    title: str | None,
    platform: Platform,
) -> CheckResult:
    """Validate caption and title lengths against the platform caption limits."""
    limits: CaptionLimits = platform.caption_limits
    problems: list[str] = []
    if caption is not None:
        max_length = limits.max_length
        if max_length is not None and len(caption) > max_length:
            problems.append(
                f"caption is {len(caption)} characters, limit is {max_length}"
            )
    if title is not None:
        max_title = limits.max_title_length
        if max_title is not None and len(title) > max_title:
            problems.append(f"title is {len(title)} characters, limit is {max_title}")
    if problems:
        return CheckResult("formatting", False, "; ".join(problems))
    parts: list[str] = []
    if caption is not None:
        parts.append(f"caption {len(caption)}ch")
    if title is not None:
        parts.append(f"title {len(title)}ch")
    if not parts:
        return CheckResult(
            "formatting",
            True,
            "no caption or title supplied; formatting check skipped",
        )
    return CheckResult("formatting", True, "within limits: " + ", ".join(parts))


def _check_compatibility(
    runtime: _Runtime,
    format_name: str | None,
    platform: Platform,
) -> CheckResult:
    """Validate the format output type against the platform media types."""
    if not format_name:
        return CheckResult(
            "compatibility",
            True,
            "no format supplied; compatibility check skipped",
        )
    formats = load_packaged_formats()
    try:
        recipe = formats.get(format_name)
    except KeyError as exc:
        return CheckResult(
            "compatibility",
            False,
            f"unknown format: {format_name!r}",
            detail=str(exc),
        )
    output_type = recipe.output_type
    if output_type not in platform.media_types:
        return CheckResult(
            "compatibility",
            False,
            f"format {format_name!r} outputs {output_type.value} which "
            f"{platform.id!r} does not support "
            f"({', '.join(platform.media_types) or 'none'})",
        )
    return CheckResult(
        "compatibility",
        True,
        (
            f"format {format_name!r} outputs {output_type.value} supported by "
            f"{platform.id!r}"
        ),
    )


def _check_scopes(runtime: _Runtime, platform: Platform) -> CheckResult:
    """Validate that a configured provider serves the publish capability."""
    provider = runtime.providers.get(platform.id)
    if provider is not None and provider.metadata.supports(CAPABILITY_PUBLISH):
        return CheckResult(
            "scopes",
            True,
            f"{platform.id!r} provider declares the publish capability",
            detail=provider.name,
        )
    for name, candidate in runtime.providers.items():
        if candidate.metadata.supports(CAPABILITY_PUBLISH):
            return CheckResult(
                "scopes",
                True,
                (
                    f"publish capability served by {name!r} "
                    f"(no provider named {platform.id!r})"
                ),
            )
    return CheckResult(
        "scopes",
        False,
        (f"no configured provider serves the publish capability for {platform.id!r}"),
    )


def _check_quota(
    runtime: _Runtime,
    platform: Platform,
    media_path: str | None,
) -> CheckResult:
    """Validate the estimated publish cost against the remaining budget."""
    guard = runtime.budget
    if guard is None:
        return CheckResult(
            "quota", True, "no budget cap configured; quota check skipped"
        )
    estimate_usd = 0.0
    try:
        result = guard.check(
            estimate_usd,
            run_id="publish-check",
            provider=platform.id,
            operation="publish",
        )
    except ConfigError as exc:
        return CheckResult("quota", False, f"budget check could not run: {exc}")
    remaining = result.remaining_usd
    remaining_text = "unbounded" if remaining is None else f"${remaining:.6f}"
    return CheckResult(
        "quota",
        True,
        (
            f"estimated publish cost ${estimate_usd:.6f} within budget; "
            f"remaining {remaining_text}"
        ),
    )


def _dry_run_publish(
    runtime: _Runtime,
    account: Any,
    platform: Platform,
    caption: str | None,
    media_path: str | None,
) -> CheckResult:
    """Submit a dry-run publish request through the matched provider."""
    provider = runtime.providers.get(platform.id)
    if provider is None:
        candidates = runtime.providers.items()
        for _, candidate in candidates:
            if candidate.metadata.supports(CAPABILITY_PUBLISH):
                provider = candidate
                break
    if provider is None:
        return CheckResult(
            "dry-run",
            False,
            "no publish-capable provider to simulate the request with",
        )
    media: list[PublishMedia] = []
    if media_path:
        path = Path(media_path).expanduser()
        if not path.is_file():
            return CheckResult(
                "dry-run",
                False,
                f"media file not found for dry-run publish: {path}",
            )
        media_type = (
            "video"
            if path.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm"}
            else "image"
        )
        media.append(
            PublishMedia(
                path=str(path),
                media_type=f"{media_type}/{path.suffix.lstrip('.')}",
                filename=path.name,
            )
        )
    request = PublishRequest(
        platform=platform.id,
        account=getattr(account, "key", account.platform or ""),
        caption=caption or "",
        media=media,
        dry_run=True,
    )
    try:
        import asyncio

        result = asyncio.run(provider.publish(request))
    except (AiviaError, ConfigError, OSError, ValueError, TypeError) as exc:
        return CheckResult(
            "dry-run",
            False,
            f"dry-run publish was rejected: {exc}",
        )
    if result.dry_run:
        return CheckResult(
            "dry-run",
            True,
            "dry-run publish accepted by the provider",
        )
    return CheckResult(
        "dry-run",
        False,
        "dry-run publish did not report a dry run",
    )


def _run_checks(
    runtime: _Runtime,
    args: argparse.Namespace,
) -> list[CheckResult]:
    platform_name = getattr(args, "platform", None)
    account_key = getattr(args, "account", None)
    language = getattr(args, "language", None)
    media_path = getattr(args, "media", None)
    caption = getattr(args, "caption", None)
    title = getattr(args, "title", None)
    format_name = getattr(args, "format", None)

    _account_key, account, platform = _resolve_account(
        runtime, platform_name, account_key, language
    )

    results: list[CheckResult] = [
        _check_tokens(runtime, account),
        _check_media(runtime, media_path, platform),
        _check_formatting(caption, title, platform),
        _check_compatibility(runtime, format_name, platform),
        _check_scopes(runtime, platform),
        _check_quota(runtime, platform, media_path),
    ]
    command = getattr(args, "_publish_command", "check")
    if command == "dry-run":
        results.append(
            _dry_run_publish(runtime, account, platform, caption, media_path)
        )
    return results


def _print_report(results: list[CheckResult], command: str) -> None:
    width = max((len(result.name) for result in results), default=0)
    lines: list[str] = [f"publish {command}:"]
    for result in results:
        status = "ok" if result.ok else "FAIL"
        lines.append(f"  {result.name:<{width}} [{status}] {result.message}")
        if result.detail:
            lines.append(f"              detail: {result.detail}")
    print("\n".join(lines))


def run_publish(args: argparse.Namespace) -> int:
    """Run the publish pre-flight validation and return an exit code."""
    command = getattr(args, "_publish_command", "check")
    try:
        runtime = _build_runtime(args)
    except (AiviaError, ConfigError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        results = _run_checks(runtime, args)
    except (AiviaError, ConfigError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        if runtime.engine is not None:
            runtime.engine.dispose()
        return 1
    finally:
        if runtime.engine is not None:
            runtime.engine.dispose()
    _print_report(results, command)
    failed = [result for result in results if not result.ok]
    if failed:
        print(
            f"error: {len(failed)} check(s) failed; publish would not succeed",
            file=sys.stderr,
        )
        return 1
    return 0


def _dispatch(args: argparse.Namespace) -> int:
    handler = getattr(args, "_publish_handler", None)
    return int(handler(args)) if handler is not None else int(run_publish(args))


register_command(
    "publish",
    _dispatch,
    help=_PUBLISH_COMMAND_HELP,
    configure=configure,
)
