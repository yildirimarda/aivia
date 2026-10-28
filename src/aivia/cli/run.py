"""``aivia run``: preview the cost of a format recipe before running it."""

from __future__ import annotations

import argparse
import sys

from aivia.cli.confirm import confirm_spend, confirmation_required
from aivia.cli.registry import register_command
from aivia.core.config.budget import BudgetConfig
from aivia.core.config.formats import FormatRecipeConfig, FormatsConfig
from aivia.core.config.providers import ProvidersConfig
from aivia.core.config.resolve import resolve_app_config
from aivia.core.errors import ConfigError
from aivia.core.estimate import Estimator, RecipeEstimate, render_estimate
from aivia.core.models.provider import ProviderMeta
from aivia.providers import docs as provider_docs
from aivia.providers.registry import registry as provider_registry

__all__ = [
    "configure",
    "provider_metadata",
    "run_run",
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


def configure(parser: argparse.ArgumentParser) -> None:
    """Add the run command's options."""
    parser.add_argument(
        "--format",
        dest="format_name",
        metavar="NAME",
        help="format recipe to run or estimate",
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="print the stage cost estimate and exit without running anything",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="confirm spending above the configured threshold without prompting",
    )
    _configure_resolution(parser)


def _configured_provider_metadata(
    section: ProvidersConfig,
) -> dict[str, ProviderMeta]:
    metadata: dict[str, ProviderMeta] = {}
    for name, entry in section.providers.items():
        if not entry.enabled:
            continue
        provider_type = provider_registry.resolve(entry.type)
        static = provider_docs.static_metadata(provider_type)
        if static is None:
            raise ConfigError(
                f"provider {name!r} does not declare static metadata",
                details={"provider": name, "type": entry.type},
            )
        metadata[name] = static
    return metadata


def provider_metadata(config: object) -> dict[str, ProviderMeta]:
    """Return provider metadata keyed by configured name.

    Configured providers win. When no provider is configured at all, the
    registered provider types are used instead, so a fresh installation can
    still preview an estimate with the deterministic mock providers.
    """
    section = getattr(config, "providers", None)
    if isinstance(section, ProvidersConfig) and section.providers:
        return _configured_provider_metadata(section)

    metadata: dict[str, ProviderMeta] = {}
    for registration in provider_registry.registrations():
        static = provider_docs.static_metadata(registration.provider)
        if static is not None:
            metadata[registration.name] = static
    return metadata


def _resolve_recipe(config: object, name: str) -> tuple[str, FormatRecipeConfig]:
    section = getattr(config, "formats", None)
    formats: dict[str, FormatRecipeConfig] = (
        section.formats if isinstance(section, FormatsConfig) else {}
    )
    recipe = formats.get(name)
    if recipe is None:
        available = ", ".join(sorted(formats)) or "none"
        raise ConfigError(
            f"unknown format: {name} (available: {available})",
            details={"format": name, "available": sorted(formats)},
        )
    return name, recipe


def _print_missing(estimate: RecipeEstimate) -> None:
    for stage in estimate.missing_stages:
        print(
            f"error: cannot estimate stage {stage.stage!r}: {stage.note}",
            file=sys.stderr,
        )


def _confirmation_threshold(config: object) -> float:
    """Return the configured spend above which a run needs confirmation."""
    budget = getattr(config, "budget", None)
    if isinstance(budget, BudgetConfig):
        return budget.confirm_above
    return 0.0


def _estimate_run(args: argparse.Namespace) -> tuple[object, RecipeEstimate]:
    config = resolve_app_config(
        config_file=args.config,
        environment=args.environment,
        profile=args.profile,
        set_values=args.set_values,
    )
    name, recipe = _resolve_recipe(config, args.format_name)
    estimate = Estimator(provider_metadata(config)).estimate(name, recipe)
    return config, estimate


def run_run(args: argparse.Namespace) -> int:
    """Print a pre-run estimate, confirm a costly run, or explain it is pending."""
    if not args.format_name:
        suffix = " with --estimate" if args.estimate else ""
        print(f"error: --format is required{suffix}", file=sys.stderr)
        return 2

    try:
        config, estimate = _estimate_run(args)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.estimate:
        print(render_estimate(estimate), end="")
        if estimate.missing_stages:
            _print_missing(estimate)
            return 1
        return 0

    if estimate.missing_stages:
        print(render_estimate(estimate), end="")
        _print_missing(estimate)
        return 1

    threshold = _confirmation_threshold(config)
    if confirmation_required(estimate.total_usd, threshold) and not confirm_spend(
        estimate.total_usd,
        threshold,
        assume_yes=args.yes,
    ):
        return 1

    print(
        "error: run execution is not available yet; "
        "pass --estimate to preview costs",
        file=sys.stderr,
    )
    return 2


register_command(
    "run",
    run_run,
    help="Run a format recipe; use --estimate to preview the cost",
    configure=configure,
)
