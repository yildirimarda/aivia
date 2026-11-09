"""``aivia template``: extract, run and batch templates from sessions.

A template is a saved generation with variables and asset slots, extracted from
a session with suggested variables. It can be run once or in batch from CSV,
with per-row cost and isolated bad rows.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import re
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import yaml

from aivia.cli import media as media_module
from aivia.cli.registry import register_command
from aivia.core.config.capability import CapabilityStrategy
from aivia.core.errors import ConfigError
from aivia.core.models.template import (
    KINDS as TEMPLATE_KINDS,
)
from aivia.core.models.template import (
    Template,
    extract_template,
    load_template,
    save_template,
)
from aivia.media.batch_runner import BatchRunner

__all__ = ["configure", "run_template"]

_STRATEGIES = tuple(strategy.value for strategy in CapabilityStrategy)
_PLACEHOLDER = re.compile(r"{{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*}}")


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _batch_field(name: str) -> str:
    return str(name).strip().lstrip("-").replace("-", "_").lower()


def _substitute(text: str, values: Mapping[str, str]) -> str:
    return _PLACEHOLDER.sub(
        lambda match: values.get(match.group(1), match.group(0)), text
    )


def _parse_var(values: list[str] | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in values or []:
        if "=" not in item:
            raise ConfigError(f"--var value must be VAR=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ConfigError("--var variable name must not be empty")
        result[key] = value
    return result


def _base_args() -> argparse.Namespace:
    args = argparse.Namespace()
    args.config = None
    args.environment = None
    args.profile = None
    args.preset = None
    args.set_values = None
    args.database = None
    args.dry_run = False
    args.no_cache = False
    args.verbose = False
    args.provider = None
    args.strategy = None
    args.model = None
    args.count = 1
    args.seed = None
    args.output = None
    args.output_dir = None
    return args


def _template_args(
    template: Template, row: Mapping[str, Any], *, position: int
) -> argparse.Namespace:
    args = _base_args()
    args.kind = template.kind
    values: dict[str, str] = {}
    for name in template.variable_names:
        raw = row.get(name)
        text = _text(raw)
        values[name] = text if text else template.variable_default(name)
    prompt = _substitute(template.prompt, values)
    args.prompt = None
    args.prompt_option = prompt
    args.prompt_file = None
    for key, value in dict(template.options).items():
        if value is None:
            continue
        setattr(args, _batch_field(str(key)), value)
    for raw_name, raw_value in row.items():
        name = _batch_field(raw_name)
        if name in {"id", "key", "kind", "output"}:
            continue
        if raw_value is None or _text(raw_value) == "":
            continue
        setattr(args, name, raw_value)
    output = _text(row.get("output"))
    if output:
        args.output = output
    return args


def _configure_template_resolution(parser: argparse.ArgumentParser) -> None:
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
        "--database",
        "--db",
        dest="database",
        metavar="PATH",
        help="database file to use instead of the one in the user data directory",
    )


def _run_extract(args: argparse.Namespace) -> int:
    try:
        path = Path(args.session).expanduser()
        if not path.is_file():
            raise ConfigError(f"session file not found: {path}")
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            session = json.loads(text)
        else:
            session = yaml.safe_load(text)
        if not isinstance(session, Mapping):
            raise ConfigError("a session file must contain a mapping")
        template = extract_template(
            session,
            name=_text(args.name) or "",
            kind=_text(args.kind) or None,
            source_session_id=_text(args.source_session) or "",
        )
    except (OSError, yaml.YAMLError) as exc:
        print(f"error: cannot read session {args.session}: {exc}", file=sys.stderr)
        return 1
    except (ValueError, ConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    output = _text(args.output)
    if output:
        try:
            save_template(template, output)
        except OSError as exc:
            print(f"error: cannot write template {output}: {exc}", file=sys.stderr)
            return 1
        print(f"Wrote template: {output}")

    if template.variables:
        print("Suggested variables:")
        for variable in template.variables:
            line = f"  {variable.name} = {variable.default!r}"
            if variable.description:
                line += f"  # {variable.description}"
            print(line)
    else:
        print("No variables suggested.")
    if template.asset_slots:
        print("Asset slots:")
        for slot in template.asset_slots:
            print(f"  {slot.name} <- {slot.asset_label}")
    print(f"Kind: {template.kind}")
    print(f"Prompt: {template.prompt!r}")
    return 0


def _run_single(template: Template, args: argparse.Namespace) -> int:
    try:
        runtime = media_module._runtime(args, template.kind)
    except (ConfigError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        outcome = asyncio.run(media_module._generate(runtime, args, template.kind))
    finally:
        runtime.engine.dispose()
    return media_module._print_outcome(outcome, args, template.kind)


def _run_run(args: argparse.Namespace) -> int:
    try:
        template = load_template(args.template)
    except (OSError, yaml.YAMLError, ValueError) as exc:
        print(f"error: cannot load template {args.template}: {exc}", file=sys.stderr)
        return 1
    set_values = _parse_var(getattr(args, "template_vars", None))
    missing = [
        name
        for name in template.variable_names
        if name not in set_values and not template.variable_default(name)
    ]
    if missing:
        print(
            f"error: missing required variables: {', '.join(missing)}", file=sys.stderr
        )
        return 2
    values = dict(set_values)
    for name in template.variable_names:
        values.setdefault(name, template.variable_default(name))
    prompt = _substitute(template.prompt, values)
    base = _base_args()
    base.config = args.config
    base.environment = args.environment
    base.profile = args.profile
    base.preset = args.preset
    base.database = args.database
    base.dry_run = args.dry_run
    base.no_cache = args.no_cache
    base.provider = args.provider
    base.strategy = args.strategy
    base.model = args.model
    base.count = args.count
    base.seed = args.seed
    base.output = _text(args.output)
    base.output_dir = _text(args.output_dir)
    base.kind = template.kind
    base.prompt = None
    base.prompt_option = prompt
    base.prompt_file = None
    for key, value in dict(template.options).items():
        if value is None:
            continue
        setattr(base, _batch_field(str(key)), value)
    return _run_single(template, base)


def _batch_rows(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read batch file {path}: {exc}") from exc
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ConfigError("a batch CSV needs a header row")
    rows: list[dict[str, Any]] = []
    for position, row in enumerate(reader, start=2):
        if None in row:
            raise ConfigError(f"batch CSV row {position} has extra values")
        rows.append(dict(row))
    if not rows:
        raise ConfigError("batch file has no items")
    return rows


def _template_batch_handler(
    template: Template,
    runtime: Any,
    output_dir: str,
    set_values: dict[str, str],
) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    def handler(row: Mapping[str, Any]) -> dict[str, Any]:
        values = dict(set_values)
        for name in template.variable_names:
            raw = row.get(name)
            text = _text(raw)
            values.setdefault(name, text if text else template.variable_default(name))
        args = _template_args(template, row, position=0)
        # Re-substitute with row values taking precedence over --var defaults.
        args.prompt_option = _substitute(template.prompt, values)
        if output_dir:
            args.output_dir = output_dir
        outcome = asyncio.run(media_module._generate(runtime, args, template.kind))
        outputs = media_module._batch_write_output(
            outcome,
            args,
            template.kind,
            key=str(row.get("id") or row.get("key") or ""),
        )
        return {
            "provider": getattr(outcome, "provider", "") or "",
            "model": getattr(outcome, "model", "") or "",
            "cost_usd": float(getattr(outcome, "amount_usd", 0.0)),
            "cached": bool(getattr(outcome, "cached", False)),
            "outputs": [str(path) for path in outputs],
        }

    return handler


def _run_batch(args: argparse.Namespace) -> int:
    try:
        template = load_template(args.template)
    except (OSError, yaml.YAMLError, ValueError) as exc:
        print(f"error: cannot load template {args.template}: {exc}", file=sys.stderr)
        return 1
    try:
        path = Path(args.file).expanduser()
        if not path.is_file():
            raise ConfigError(f"batch file not found: {path}")
        rows = _batch_rows(path)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    set_values = _parse_var(getattr(args, "template_vars", None))
    try:
        base = _base_args()
        base.config = args.config
        base.environment = args.environment
        base.profile = args.profile
        base.preset = args.preset
        base.database = args.database
        base.dry_run = args.dry_run
        base.no_cache = args.no_cache
        base.provider = args.provider
        base.strategy = args.strategy
        base.model = args.model
        base.count = args.count
        base.seed = args.seed
        base.kind = template.kind
        runtime = media_module._runtime(base, template.kind)
    except (ConfigError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        runner = BatchRunner(
            _template_batch_handler(
                template, runtime, _text(args.output_dir), set_values
            ),
            concurrency=args.concurrency,
            max_cost_usd=args.max_cost,
            manifest=None,
            resume=False,
            stop_on_error=False,
        )
        result = runner.run_sync(rows)
    finally:
        runtime.engine.dispose()
    media_module._print_batch_result(result)
    return 0 if result.ok else 1


def run_template(args: argparse.Namespace) -> int:
    """Dispatch to the selected template subcommand."""
    handler = getattr(args, "_template_handler", None)
    if handler is None:
        print("error: missing template command", file=sys.stderr)
        print("usage: aivia template {extract,run,batch}", file=sys.stderr)
        return 2
    return int(handler(args))


def configure(parser: argparse.ArgumentParser) -> None:
    """Add the template group and its ``extract``, ``run`` and ``batch`` commands."""
    subparsers = parser.add_subparsers(dest="template_command", metavar="COMMAND")

    extract = subparsers.add_parser(
        "extract",
        help="Extract a template from a session with suggested variables",
        description="Extract a reusable template from a session file",
    )
    extract.add_argument(
        "session",
        metavar="FILE",
        help="session YAML or JSON file to extract the template from",
    )
    extract.add_argument(
        "--output",
        metavar="PATH",
        help="write the extracted template to PATH instead of standard output",
    )
    extract.add_argument(
        "--name",
        metavar="NAME",
        help="template name",
    )
    extract.add_argument(
        "--kind",
        choices=TEMPLATE_KINDS,
        help="media kind (default: taken from the session or image)",
    )
    extract.add_argument(
        "--source-session",
        metavar="ID",
        help="source session id recorded on the template",
    )
    extract.set_defaults(_template_handler=_run_extract)

    run_parser = subparsers.add_parser(
        "run",
        help="Run a template once with variable values",
        description="Run one template and report its cost",
    )
    run_parser.add_argument(
        "template",
        metavar="FILE",
        help="template YAML or JSON file",
    )
    run_parser.add_argument(
        "--output",
        metavar="PATH",
        help="copy the first output to PATH",
    )
    run_parser.add_argument(
        "--output-dir",
        metavar="DIR",
        help="write the output under DIR",
    )
    _configure_template_resolution(run_parser)
    run_parser.add_argument(
        "--var",
        dest="template_vars",
        action="append",
        default=None,
        metavar="VAR=VALUE",
        help="template variable value; repeatable",
    )
    run_parser.add_argument(
        "--provider",
        action="append",
        default=None,
        metavar="NAME",
        help="provider name to use",
    )
    run_parser.add_argument(
        "--strategy",
        choices=_STRATEGIES,
        metavar="NAME",
        help="provider selection strategy",
    )
    run_parser.add_argument("--model", metavar="NAME", help="provider model id")
    run_parser.add_argument("--count", type=int, default=1, metavar="N")
    run_parser.add_argument("--seed", type=int, metavar="N", help="reproducible seed")
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="swap providers for the deterministic mock",
    )
    run_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="ignore the media cache",
    )
    run_parser.set_defaults(_template_handler=_run_run)

    batch = subparsers.add_parser(
        "batch",
        help="Run a template once per CSV row with per-row cost",
        description=(
            "Run a template per CSV row, reporting per-row cost and isolating bad rows"
        ),
    )
    batch.add_argument(
        "file",
        metavar="FILE",
        help="CSV file with one row per batch item",
    )
    batch.add_argument(
        "--template",
        metavar="FILE",
        required=True,
        help="template YAML or JSON file applied to every row",
    )
    batch.add_argument(
        "--output-dir",
        metavar="DIR",
        help="write each item output under DIR",
    )
    batch.add_argument(
        "--concurrency",
        type=int,
        default=1,
        metavar="N",
        help="maximum number of items in flight (default: 1)",
    )
    batch.add_argument(
        "--max-cost",
        "--budget",
        dest="max_cost",
        type=float,
        metavar="USD",
        help="stop starting items once the batch cost reaches this amount",
    )
    _configure_template_resolution(batch)
    batch.add_argument(
        "--var",
        dest="template_vars",
        action="append",
        default=None,
        metavar="VAR=VALUE",
        help="template variable value; repeatable",
    )
    batch.add_argument(
        "--provider",
        action="append",
        default=None,
        metavar="NAME",
        help="default provider chain",
    )
    batch.add_argument(
        "--strategy",
        choices=_STRATEGIES,
        metavar="NAME",
        help="provider selection strategy",
    )
    batch.add_argument("--model", metavar="NAME", help="default provider model id")
    batch.add_argument("--count", type=int, default=1, metavar="N")
    batch.add_argument("--seed", type=int, metavar="N", help="default seed")
    batch.add_argument(
        "--dry-run",
        action="store_true",
        help="swap providers for the deterministic mock",
    )
    batch.add_argument(
        "--no-cache",
        action="store_true",
        help="ignore the media cache",
    )
    batch.set_defaults(_template_handler=_run_batch)


register_command(
    "template",
    run_template,
    help="Extract, run and batch templates from sessions",
    configure=configure,
)
