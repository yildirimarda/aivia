"""``aivia providers``: inspect registered provider types and check health."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import math
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from aivia.cli.registry import register_command
from aivia.core.dry_run import DRY_RUN_PROVIDER_TYPE
from aivia.core.errors import ConfigError
from aivia.core.models.provider import ProviderMeta
from aivia.core.redact import redact
from aivia.providers import docs as provider_docs
from aivia.providers.base import HealthStatus, Provider, ProviderHealth
from aivia.providers.mock import register_mock_providers
from aivia.providers.registry import registry as provider_registry

__all__ = [
    "LIVE_BUDGET_ENV",
    "configure",
    "run_check",
    "run_docs",
    "run_list",
    "run_providers",
]

LIVE_BUDGET_ENV = "AIVA_LIVE_BUDGET_USD"
_NO_SPEND = 0.0


def _configure_list(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--capability",
        metavar="NAME",
        help="only list providers that declare this capability",
    )
    parser.set_defaults(_providers_handler=run_list)


def _configure_check(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "names",
        nargs="*",
        metavar="PROVIDER",
        help="provider type or dotted path to check (default: every registered type)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="allow the health checks to contact external services",
    )
    parser.add_argument(
        "--budget",
        "--max-budget",
        dest="budget",
        type=float,
        metavar="USD",
        help=f"positive spend cap for this check (default: ${LIVE_BUDGET_ENV})",
    )
    parser.set_defaults(_providers_handler=run_check)


def _configure_docs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="write the reference to PATH instead of standard output",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when the reference at PATH (or docs/PROVIDERS.md) is stale",
    )
    parser.set_defaults(_providers_handler=run_docs)


def configure(parser: argparse.ArgumentParser) -> None:
    """Add the providers group and its subcommands."""
    subparsers = parser.add_subparsers(dest="providers_command", metavar="COMMAND")
    list_parser = subparsers.add_parser(
        "list",
        help="List registered provider types and their capabilities",
        description="List registered provider types and their capabilities",
    )
    _configure_list(list_parser)
    check_parser = subparsers.add_parser(
        "check",
        help="Run live health checks under an explicit budget cap",
        description="Run live health checks under an explicit budget cap",
    )
    _configure_check(check_parser)
    docs_parser = subparsers.add_parser(
        "docs",
        help="Generate the provider capability matrix",
        description="Generate the provider capability matrix",
    )
    _configure_docs(docs_parser)


def run_providers(args: argparse.Namespace) -> int:
    """Dispatch to the selected providers subcommand."""
    handler = getattr(args, "_providers_handler", None)
    if handler is None:
        print("error: missing providers command", file=sys.stderr)
        print("usage: aivia providers {list, check, docs}", file=sys.stderr)
        return 2
    return int(handler(args))


def _static_metadata(provider: type[Provider]) -> ProviderMeta | None:
    """Return metadata without constructing the provider when possible."""
    return provider_docs.static_metadata(provider)


def run_list(args: argparse.Namespace) -> int:
    """Print every registered provider type and its declared capabilities."""
    capability = args.capability.strip() if args.capability else None
    print("NAME\tCAPABILITIES")
    for registration in provider_registry.registrations():
        metadata = _static_metadata(registration.provider)
        capabilities = tuple(metadata.capabilities) if metadata is not None else ()
        if capability is not None and capability not in capabilities:
            continue
        rendered = ", ".join(capabilities) if capabilities else "-"
        print(f"{registration.name}\t{rendered}")
    return 0


def run_docs(args: argparse.Namespace) -> int:
    """Print, write or check the generated provider capability matrix."""
    text = provider_docs.dump_provider_docs()
    target = Path(args.output) if args.output is not None else provider_docs.docs_path()
    if args.check:
        try:
            existing = target.read_text(encoding="utf-8")
        except OSError:
            existing = None
        if existing != text:
            print(
                f"error: provider reference is stale: {target}",
                file=sys.stderr,
            )
            print(
                f"regenerate it with: aivia providers docs --output {target}",
                file=sys.stderr,
            )
            return 1
        print(f"provider reference is up to date: {target}")
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


@dataclass(frozen=True, slots=True)
class _CheckResult:
    """The outcome of checking one provider, including setup failures."""

    name: str
    health: ProviderHealth | None = None
    error: str | None = None


def _error_text(exc: BaseException) -> str:
    message = redact(str(exc)) or type(exc).__name__
    return f"{type(exc).__name__}: {message}"


async def _check_one(name: str, provider: type[Provider]) -> _CheckResult:
    try:
        instance = provider()
    except Exception as exc:
        return _CheckResult(name=name, error=_error_text(exc))

    try:
        health = await instance.health()
    except Exception as exc:
        health = ProviderHealth(
            status=HealthStatus.UNHEALTHY,
            message=_error_text(exc),
        )
    finally:
        with contextlib.suppress(Exception):
            await instance.close()
    return _CheckResult(name=name, health=health)


async def _run_checks(
    items: Sequence[tuple[str, type[Provider]]],
) -> list[_CheckResult]:
    results: list[_CheckResult] = []
    for name, provider in items:
        results.append(await _check_one(name, provider))
    return results


def _is_positive_amount(value: float) -> bool:
    return math.isfinite(value) and value > 0


def _resolve_budget(value: float | None) -> float:
    if value is None:
        raw = os.environ.get(LIVE_BUDGET_ENV)
        if raw is None or not raw.strip():
            raise ValueError(
                "providers check requires a budget cap; pass --budget USD "
                f"or set {LIVE_BUDGET_ENV}"
            )
        try:
            budget = float(raw)
        except ValueError as exc:
            raise ValueError(
                f"{LIVE_BUDGET_ENV} must be a positive amount, got {raw!r}"
            ) from exc
    else:
        budget = value

    if not _is_positive_amount(budget):
        raise ValueError(f"budget cap must be a positive amount, got {budget!r}")
    return budget


def _mock_provider_type() -> type[Provider]:
    """Register the mock types if needed and return the all-in-one mock."""
    register_mock_providers(provider_registry)
    return provider_registry.resolve(DRY_RUN_PROVIDER_TYPE)


def _resolve_names(
    names: Sequence[str],
    *,
    dry_run: bool = False,
) -> list[tuple[str, type[Provider]]]:
    if dry_run:
        registrations = provider_registry.registrations()
        provider = _mock_provider_type()
        if names:
            for name in names:
                provider_registry.resolve(name)
            return [(name, provider) for name in names]
        if not registrations:
            registrations = provider_registry.registrations()
        return [(registration.name, provider) for registration in registrations]

    if not names:
        return [
            (registration.name, registration.provider)
            for registration in provider_registry.registrations()
        ]
    return [(name, provider_registry.resolve(name)) for name in names]


def run_check(args: argparse.Namespace) -> int:
    """Run health checks, refusing to touch the network without live mode."""
    dry_run = bool(getattr(args, "dry_run", False))
    if not args.live and not dry_run:
        print(
            "error: providers check requires live mode; pass --live or --dry-run",
            file=sys.stderr,
        )
        return 1

    budget: float
    if dry_run:
        budget = _NO_SPEND
    else:
        try:
            budget = _resolve_budget(args.budget)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    try:
        items = _resolve_names(args.names, dry_run=dry_run)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    results = asyncio.run(_run_checks(items))
    failures = 0
    for result in results:
        if result.error is not None:
            failures += 1
            print(f"{result.name}: error: {result.error}", file=sys.stderr)
            continue
        health = result.health
        assert health is not None
        line = f"{result.name}: {health.status.value}"
        if health.latency_ms is not None:
            line += f" ({health.latency_ms:.1f} ms)"
        if health.message:
            line += f" {health.message}"
        print(line)
        if health.status is HealthStatus.UNHEALTHY:
            failures += 1

    suffix = " (in dry run; mock providers)" if dry_run else ""
    print(
        f"checked {len(results)} provider(s) with a ${budget:.2f} budget cap: "
        f"{failures} failed{suffix}"
    )
    return 1 if failures else 0


register_command(
    "providers",
    run_providers,
    help="List provider types, run health checks and generate the reference",
    configure=configure,
)
