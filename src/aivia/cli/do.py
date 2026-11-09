"""``aivia do``: preview and approve a format recipe plan.

The command loads a format recipe, prints a cost preview and then asks for
approval according to the planner's ``approval_mode``:

* ``always`` -- every plan must be approved before it can run
* ``ask`` -- the default; prompt the user for confirmation
* ``never`` -- skip prompting and approve the plan immediately

A ``--edit`` flag opens an edit path so the plan can be reviewed before
approval. Without ``--yes``, non-interactive runs are refused instead of
hanging.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aivia.cli.confirm import confirm_action
from aivia.cli.registry import register_command
from aivia.cli.run import provider_metadata
from aivia.core.config import PlannerConfig
from aivia.core.config.formats import FormatRecipeConfig, FormatsConfig
from aivia.core.config.planner import PlannerApprovalMode
from aivia.core.config.resolve import resolve_config
from aivia.core.dry_run import apply_dry_run
from aivia.core.errors import ConfigError
from aivia.core.estimate import Estimator, render_estimate


def _configure_edit(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--format",
        dest="format",
        metavar="FORMAT",
        help="format recipe to preview and approve",
    )
    parser.add_argument(
        "--edit",
        action="store_true",
        help="open the plan for editing before approval",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="write the cost breakdown to PATH instead of only standard output",
    )


def configure(parser: argparse.ArgumentParser) -> None:
    """Add the do command's options."""
    _configure_edit(parser)
    parser.set_defaults(_do_handler=run_do)


def _load_config(args: argparse.Namespace) -> object:
    resolved = resolve_config(
        config_file=args.config,
        environment=args.environment,
        profile=args.profile,
        preset=args.preset,
        set_values=args.set_values,
    )
    config = resolved.to_app_config()
    if getattr(args, "dry_run", False):
        config = apply_dry_run(config)
    return config


def _format_recipe(config: object, name: str) -> FormatRecipeConfig | None:
    section = getattr(config, "formats", None)
    if not isinstance(section, FormatsConfig):
        return None
    recipe = section.formats.get(name)
    if not isinstance(recipe, FormatRecipeConfig):
        return None
    return recipe


def run_do(args: argparse.Namespace) -> int:
    """Preview a plan, print its cost breakdown and request approval before running it.

    With ``--output PATH`` the breakdown is also written to PATH after approval.
    """
    if not args.format:
        print("error: a format is required", file=sys.stderr)
        return 2

    try:
        config = _load_config(args)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    planner = getattr(config, "planner", None)
    if not isinstance(planner, PlannerConfig):
        planner = PlannerConfig()

    recipe = _format_recipe(config, args.format)
    if recipe is None:
        section = getattr(config, "formats", None)
        formats = section.formats if isinstance(section, FormatsConfig) else {}
        available = ", ".join(sorted(formats)) or "none"
        print(
            f"error: unknown format: {args.format} (available: {available})",
            file=sys.stderr,
        )
        return 1

    try:
        estimate = Estimator(provider_metadata(config)).estimate(args.format, recipe)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    breakdown = f"Format: {args.format}\n{render_estimate(estimate)}"
    print(breakdown, end="")

    if args.edit:
        print(f"Edit path: plan '{args.format}' is open for editing before approval.")

    if planner.approval_mode is PlannerApprovalMode.NEVER:
        if args.output:
            return _write_output(args.output, breakdown)
        return 0

    if not args.yes and not confirm_action(
        f"Approve plan {args.format}?", assume_yes=args.yes
    ):
        print("error: plan approval declined", file=sys.stderr)
        return 1

    if args.output:
        return _write_output(args.output, breakdown)

    return 0


def _write_output(path: str, breakdown: str) -> int:
    target = Path(path).expanduser()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(breakdown, encoding="utf-8")
    except OSError as exc:
        print(f"error: cannot write output {target}: {exc}", file=sys.stderr)
        return 1
    print(f"Output: {target}")
    return 0


register_command(
    "do",
    run_do,
    help="Preview and approve a format recipe plan",
    configure=configure,
)
