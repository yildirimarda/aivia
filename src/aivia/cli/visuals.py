"""``aivia visuals``: run or report the visual decisions of an existing run.

The command executes the concrete visual stages a run's format recipe selects
and then prints one line group per beat: the resolved decision, the provider
and model that served it, the cost it incurred and the quality score the QA
gate assigned. A run that already has visual manifests can be inspected
without executing anything, which keeps the report useful after a resumed or
partially completed pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError

import aivia.cli.run as run_command
from aivia.cli.registry import register_command
from aivia.cli.run import build_stages
from aivia.core import paths
from aivia.core.config.database import DatabaseConfig
from aivia.core.config.formats import FormatRecipeConfig, FormatsConfig
from aivia.core.config.providers import ProvidersConfig
from aivia.core.config.resolve import resolve_config
from aivia.core.cost import CostTracker
from aivia.core.db import AssetRepository, RunRepository, create_database_engine
from aivia.core.dry_run import apply_dry_run
from aivia.core.errors import ConfigError
from aivia.core.models import Asset, Run
from aivia.core.orchestrator import OrchestrationResult, Orchestrator
from aivia.core.router import Router
from aivia.core.stages import Stage
from aivia.providers.provider_factory import build_providers
from aivia.visuals.reuse import VisualReuseStage

__all__ = [
    "VISUAL_STAGE_NAMES",
    "BeatVisual",
    "configure",
    "run_visuals",
    "visual_beat_reports",
]

VISUAL_STAGE_NAMES = frozenset(
    {"visual", "visuals", "image", "stock", "video", "visual_qa", "visual-qa"}
)
"""The recipe stages whose per beat results this command reports."""

_DEFAULT_MANIFESTS = (
    "visual-qa.json",
    "visual_qa.json",
    "visuals.json",
    "visual.json",
    "video.json",
    "videos.json",
    "image.json",
    "images.json",
    "stock.json",
    "stocks.json",
)
"""Manifest names a visual stage writes when it is not configured otherwise."""

_MANIFEST_HINTS = ("visual", "image", "video", "stock", "qa")
_SOURCE_PRIORITY = {
    "": 0,
    "asset": 0,
    "metadata": 1,
    "visuals": 1,
    "visual": 1,
    "image": 2,
    "stock": 2,
    "video": 2,
    "qa": 3,
    "visual_qa": 3,
}
_ORIGINAL_BUILD_STAGES = build_stages


@dataclass(slots=True)
class BeatVisual:
    """The reported decision, provider, cost and quality for one beat."""

    index: int
    decision: str = ""
    provider: str = ""
    model: str = ""
    cost_usd: float = 0.0
    quality_score: float | None = None
    cached: bool = False
    fallback: str = ""
    error: str = ""
    source: str = ""

    @property
    def cost(self) -> float:
        """Alias for :attr:`cost_usd`."""
        return self.cost_usd

    @property
    def quality(self) -> float | None:
        """Alias for :attr:`quality_score`."""
        return self.quality_score

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this beat."""
        return {
            "index": self.index,
            "decision": self.decision,
            "visual": self.decision,
            "provider": self.provider,
            "model": self.model,
            "cost_usd": self.cost_usd,
            "cost": self.cost_usd,
            "quality_score": self.quality_score,
            "quality": self.quality_score,
            "score": self.quality_score,
            "cached": self.cached,
            "fallback": self.fallback,
            "error": self.error,
            "source": self.source,
        }


def _build_stages(recipe: FormatRecipeConfig) -> list[Stage]:
    """Use a local override when a test or module replaced this function."""
    if build_stages is not _ORIGINAL_BUILD_STAGES:
        return build_stages(recipe)
    return run_command.build_stages(recipe)


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
    """Add the visuals command's options."""
    parser.add_argument(
        "run_id",
        nargs="?",
        metavar="RUN_ID",
        help="existing run whose visual beats to run or report",
    )
    parser.add_argument(
        "--run",
        "--run-id",
        dest="run_option",
        metavar="RUN_ID",
        help="existing run whose visual beats to run or report",
    )
    parser.add_argument(
        "--database",
        "--db",
        dest="database",
        metavar="PATH",
        help="database file to read instead of the one in the user data directory",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="print the per beat report as JSON",
    )
    _configure_resolution(parser)


def _run_id(args: argparse.Namespace) -> str:
    positional = (getattr(args, "run_id", None) or "").strip()
    option = (getattr(args, "run_option", None) or "").strip()
    return positional or option


def _load_config(args: argparse.Namespace) -> object:
    resolved = resolve_config(
        config_file=getattr(args, "config", None),
        environment=getattr(args, "environment", None),
        profile=getattr(args, "profile", None),
        preset=getattr(args, "preset", None),
        set_values=getattr(args, "set_values", None),
    )
    config = resolved.to_app_config()
    if getattr(args, "dry_run", False):
        config = apply_dry_run(config)
    return config


def _database_path(args: argparse.Namespace, config: object) -> Path:
    explicit = getattr(args, "database", None)
    if explicit:
        return Path(explicit).expanduser()
    database = getattr(config, "database", None)
    if not isinstance(database, DatabaseConfig):
        database = None
    filename = database.filename if database is not None else DatabaseConfig().filename
    return paths.resolve_paths(getattr(args, "config", None)).data_dir / filename


def _providers_config(config: object) -> ProvidersConfig:
    section = getattr(config, "providers", None)
    if isinstance(section, ProvidersConfig):
        return section
    if isinstance(section, Mapping):
        return ProvidersConfig.from_mapping(section)
    if isinstance(config, Mapping):
        raw = config.get("providers")
        if isinstance(raw, ProvidersConfig):
            return raw
        if isinstance(raw, Mapping):
            return ProvidersConfig.from_mapping(raw)
    return ProvidersConfig()


def _build_router(config: object) -> Router:
    """Build the configured media router for visual generation."""
    return Router(build_providers(_providers_config(config)))


def _recipe(config: object, run: Run) -> FormatRecipeConfig:
    section = getattr(config, "formats", None)
    formats = section.formats if isinstance(section, FormatsConfig) else {}
    name = run.format.strip()
    if not name:
        raise ConfigError(f"run has no format: {run.id}")
    recipe = formats.get(name)
    if recipe is None:
        available = ", ".join(sorted(formats)) or "none"
        raise ConfigError(
            f"unknown format: {name} (available: {available})",
            details={"format": name, "available": sorted(formats)},
        )
    return recipe


def _visual_stages(recipe: FormatRecipeConfig) -> list[Stage]:
    return [
        stage for stage in _build_stages(recipe) if stage.name in VISUAL_STAGE_NAMES
    ]


def _is_implemented(stage: Stage) -> bool:
    run_method = getattr(stage, "run", None)
    if run_method is None:
        return False
    function = getattr(run_method, "__func__", run_method)
    return function is not Stage.run


def _has_visual_input(run: Run, stage: Stage) -> bool:
    """Whether the run directory holds the input a concrete visual stage reads."""
    run_dir = Path(run.run_dir).expanduser() if run.run_dir else None
    if run_dir is None or not run_dir.is_dir():
        return False
    if isinstance(stage, VisualReuseStage):
        return any(run_dir.glob("script.*.json"))
    return True


def _execute(
    engine: Engine,
    run: Run,
    config: object,
    stages: list[Stage],
    *,
    data_dir: Path,
) -> OrchestrationResult:
    return Orchestrator(
        stages=stages,
        run=run.id,
        repository=engine,
        run_dir=run.run_dir or None,
        data_dir=data_dir,
        config=config,
        cost=CostTracker(engine),
        router=_build_router(config),
        raise_on_failure=False,
    ).execute_sync()


def _text(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "value") and not isinstance(value, str):
        value = value.value
    return str(value).strip()


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _field(value: Any, *names: str) -> Any:
    for name in names:
        if isinstance(value, Mapping):
            if name in value and value[name] is not None:
                return value[name]
        else:
            found = getattr(value, name, None)
            if found is not None:
                return found
    return None


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _items(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _asset_of(value: Any) -> Any:
    asset = _field(value, "asset", "image", "video", "clip")
    if isinstance(asset, Mapping):
        return asset
    if isinstance(asset, Asset):
        return asset
    return asset


def _stock_of(value: Any) -> Any:
    return _field(value, "stock")


def _decision_of(value: Any) -> str:
    request = _field(value, "request", "visual_request")
    decision = _field(
        value,
        "decision",
        "chosen_type",
        "visual_type",
        "visual",
        "source_type",
        "visual_kind",
    )
    if not _text(decision) and request is not None:
        decision = _field(
            request,
            "visual_type",
            "visual",
            "source_type",
            "visual_kind",
            "kind",
        )
    if not _text(decision):
        decision = _field(value, "kind", "media_kind", "media_type", "type")
    if not _text(decision) and _stock_of(value) is not None:
        decision = "stock"
    if not _text(decision):
        asset = _asset_of(value)
        if asset is not None:
            decision = _field(asset, "kind", "media_type", "type")
    return _text(decision).lower()


def _provider_of(value: Any) -> str:
    provider = _text(_field(value, "provider", "vendor"))
    if provider:
        return provider
    for nested in (_asset_of(value), _stock_of(value)):
        if nested is not None:
            provider = _text(_field(nested, "provider", "vendor"))
            if provider:
                return provider
    return ""


def _model_of(value: Any) -> str:
    model = _text(_field(value, "model", "model_id"))
    if model:
        return model
    for nested in (_asset_of(value), _stock_of(value)):
        if nested is not None:
            model = _text(_field(nested, "model", "model_id"))
            if model:
                return model
            metadata = _mapping(_field(nested, "metadata"))
            if metadata is not None:
                model = _text(metadata.get("model"))
                if model:
                    return model
    return ""


def _cost_of(value: Any) -> float:
    for name in ("cost_usd", "total_cost_usd", "cost", "amount_usd", "amount"):
        number = _number(_field(value, name))
        if number is not None:
            return number
    asset = _asset_of(value)
    if asset is not None:
        number = _number(_field(asset, "total_cost_usd", "cost_usd"))
        if number is not None:
            return number
        links = _items(_field(asset, "cost_links", "costs"))
        total = 0.0
        for link in links:
            amount = _number(_field(link, "amount_usd", "amount", "cost"))
            if amount is not None:
                total += amount
        if links:
            return total
    return 0.0


def _quality_of(value: Any) -> float | None:
    for name in ("score", "quality_score", "qa_score", "quality"):
        number = _number(_field(value, name))
        if number is not None:
            return number
    asset = _asset_of(value)
    if asset is not None:
        number = _number(_field(asset, "quality_score", "score", "qa_score"))
        if number is not None:
            return number
    metadata = _mapping(_field(value, "metadata"))
    if metadata is not None:
        for name in ("score", "quality_score", "qa_score", "quality"):
            number = _number(metadata.get(name))
            if number is not None:
                return number
    return None


def _index_of(value: Any, fallback: int) -> int:
    for name in ("index", "beat_index", "beat", "position", "number"):
        raw = _field(value, name)
        if isinstance(raw, bool) or raw is None:
            continue
        if isinstance(raw, int):
            return max(0, raw)
        if isinstance(raw, str) and raw.strip().isdigit():
            return max(0, int(raw.strip()))
    return fallback


def _source_of(value: str) -> str:
    text = _text(value).lower().replace("-", "_")
    if text in {"visual_qa", "qa", "quality"}:
        return "qa"
    if text in {"image", "images", "still"}:
        return "image"
    if text in {"video", "videos", "motion"}:
        return "video"
    if text in {"stock", "stocks"}:
        return "stock"
    if text in {"visuals", "visual"}:
        return "visuals"
    return text


def _beat_from(value: Any, *, source: str, fallback_index: int) -> BeatVisual:
    return BeatVisual(
        index=_index_of(value, fallback_index),
        decision=_decision_of(value),
        provider=_provider_of(value),
        model=_model_of(value),
        cost_usd=_cost_of(value),
        quality_score=_quality_of(value),
        cached=bool(_field(value, "cached", "cache_hit")),
        fallback=_text(_field(value, "fallback")),
        error=_text(_field(value, "error", "error_message")),
        source=_source_of(source),
    )


def _merge(target: BeatVisual, incoming: BeatVisual) -> None:
    current_priority = _SOURCE_PRIORITY.get(target.source, 0)
    incoming_priority = _SOURCE_PRIORITY.get(incoming.source, 0)
    higher = incoming_priority >= current_priority
    if incoming.decision and (not target.decision or higher):
        target.decision = incoming.decision
    if incoming.provider and (not target.provider or higher):
        target.provider = incoming.provider
    if incoming.model and (not target.model or higher):
        target.model = incoming.model
    if incoming.cost_usd and (not target.cost_usd or higher):
        target.cost_usd = incoming.cost_usd
    if incoming.quality_score is not None and (target.quality_score is None or higher):
        target.quality_score = incoming.quality_score
    if incoming.cached:
        target.cached = True
    if incoming.fallback and (not target.fallback or higher):
        target.fallback = incoming.fallback
    if incoming.error and (not target.error or higher):
        target.error = incoming.error
    if higher:
        target.source = incoming.source


def _payload_beats(payload: Any) -> list[Any]:
    if isinstance(payload, Mapping):
        for name in ("beats", "results", "items", "decisions"):
            value = payload.get(name)
            if isinstance(value, Mapping):
                return list(value.values())
            beats = _items(value)
            if beats:
                return beats
        if "index" in payload:
            return [payload]
        return []
    return _items(payload)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _manifest_names(
    run_dir: Path,
    stages: Sequence[Stage],
    result: OrchestrationResult | None,
) -> list[Path]:
    names: list[str] = []
    for stage in stages:
        for key in (
            "filename",
            "manifest_filename",
            "manifest",
            f"{stage.name}_filename",
            f"{stage.name}s_filename",
        ):
            value = stage.options.get(key)
            if isinstance(value, str) and value.strip():
                names.append(value.strip())
        for artifact in stage.artifacts:
            names.append(artifact)
    if result is not None:
        for stage_result in result.stages:
            if stage_result.stage not in VISUAL_STAGE_NAMES:
                continue
            names.extend(stage_result.artifacts)
    names.extend(_DEFAULT_MANIFESTS)

    paths: list[Path] = []
    seen: set[Path] = set()
    for name in names:
        path = run_dir / name
        resolved = path.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        paths.append(path)
    for path in sorted(run_dir.glob("*.json")):
        resolved = path.resolve(strict=False)
        if resolved in seen:
            continue
        lowered = path.name.lower()
        payload = _read_json(path)
        stage_name = _text(_field(payload, "stage")).lower()
        if any(hint in lowered for hint in _MANIFEST_HINTS) or (
            stage_name in VISUAL_STAGE_NAMES
        ):
            seen.add(resolved)
            paths.append(path)
    return paths


def _stage_beats(stages: Sequence[Stage]) -> list[tuple[Any, str]]:
    entries: list[tuple[Any, str]] = []
    for stage in stages:
        for attribute in ("results", "beats", "decisions"):
            values = _items(getattr(stage, attribute, None))
            if not values:
                continue
            entries.extend((value, stage.name) for value in values)
            break
    return entries


def _result_beats(result: OrchestrationResult | None) -> list[tuple[Any, str]]:
    entries: list[tuple[Any, str]] = []
    if result is None:
        return entries
    for stage_result in result.stages:
        if stage_result.stage not in VISUAL_STAGE_NAMES:
            continue
        direct = _items(getattr(stage_result, "beats", None))
        if direct:
            entries.extend((value, stage_result.stage) for value in direct)
        metadata = _mapping(stage_result.metadata)
        if metadata is None:
            continue
        for name in ("beats", "results", "items", "decisions"):
            values = _items(metadata.get(name))
            if values:
                entries.extend((value, stage_result.stage) for value in values)
                break
    return entries


def _merge_entries(entries: Iterable[tuple[Any, str]]) -> list[BeatVisual]:
    beats: dict[int, BeatVisual] = {}
    for position, (value, source) in enumerate(entries):
        beat = _beat_from(value, source=source, fallback_index=position)
        current = beats.get(beat.index)
        if current is None:
            beats[beat.index] = beat
        else:
            _merge(current, beat)
    return [beats[index] for index in sorted(beats)]


def _asset_reports(engine: Engine | None, run_id: str) -> list[BeatVisual]:
    if engine is None:
        return []
    reports: list[BeatVisual] = []
    for asset in AssetRepository(engine).list(run_id=run_id):
        decision = _text(_field(asset, "kind", "media_type", "type")).lower()
        if decision not in {"image", "video", "stock"}:
            continue
        reports.append(
            BeatVisual(
                index=len(reports),
                decision=decision,
                provider=_text(_field(asset, "provider", "vendor")),
                model=_text(_field(asset, "model", "model_id")),
                cost_usd=_cost_of(asset),
                quality_score=_quality_of(asset),
                source="asset",
            )
        )
    return reports


def visual_beat_reports(
    run: Run,
    *,
    stages: Sequence[Stage] = (),
    result: OrchestrationResult | None = None,
    engine: Engine | None = None,
) -> list[BeatVisual]:
    """Return the per beat visual report for *run*.

    Results come from the executed stage objects first, then from their
    persisted metadata and manifests, and finally from asset rows. Each source
    is merged by beat index so a QA score can enrich the decision a generation
    stage produced without creating a second beat.
    """
    entries = _stage_beats(stages)
    entries.extend(_result_beats(result))
    run_dir = Path(run.run_dir).expanduser() if run.run_dir else None
    if run_dir is not None and run_dir.is_dir():
        for path in _manifest_names(run_dir, stages, result):
            payload = _read_json(path)
            if payload is None:
                continue
            source = _text(_field(payload, "stage")) or path.stem
            entries.extend((value, source) for value in _payload_beats(payload))
    reports = _merge_entries(entries)
    if reports:
        return reports
    return _asset_reports(engine, run.id)


def _report(
    run: Run,
    beats: Sequence[BeatVisual],
    *,
    stages: Sequence[Stage] = (),
    cost_usd: float | None = None,
    json_output: bool = False,
) -> None:
    total = sum(beat.cost_usd for beat in beats) if cost_usd is None else cost_usd
    payload = {
        "run": run.id,
        "stages": [stage.name for stage in stages],
        "beats": [beat.to_dict() for beat in beats],
        "cost_usd": total,
        "cost": total,
    }
    if json_output:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return

    print(f"Run: {run.id}".rstrip())
    if stages:
        print("Stages: " + ", ".join(stage.name for stage in stages))
    print(f"Beats: {len(beats)}")
    print(f"Cost: ${total:.6f}")
    for beat in beats:
        print(f"Beat: {beat.index}")
        print(f"  Decision: {beat.decision or 'none'}")
        if beat.provider:
            print(f"  Provider: {beat.provider}")
        if beat.model:
            print(f"  Model: {beat.model}")
        print(f"  Cost: ${beat.cost_usd:.6f}")
        if beat.quality_score is None:
            print("  Quality: n/a")
        else:
            print(f"  Quality: {beat.quality_score:.3f}")
            print(f"  Quality score: {beat.quality_score:.3f}")
        if beat.cached:
            print("  Cached: yes")
        if beat.fallback:
            print(f"  Fallback: {beat.fallback}")
        if beat.error:
            print(f"  Error: {beat.error}")


def run_visuals(args: argparse.Namespace) -> int:
    """Run the visual stages for an existing run and print the per beat report."""
    run_id = _run_id(args)
    if not run_id:
        print("error: a run id is required", file=sys.stderr)
        return 2

    try:
        config = _load_config(args)
    except (ConfigError, ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    database_path = _database_path(args, config)
    if not database_path.exists():
        print(f"error: database not found: {database_path}", file=sys.stderr)
        return 1

    engine = create_database_engine(path=database_path)
    try:
        run = RunRepository(engine).get(run_id)
        if run is None:
            print(f"error: run not found: {run_id}", file=sys.stderr)
            return 1

        stages: list[Stage] = []
        try:
            stages = _visual_stages(_recipe(config, run))
        except ConfigError:
            stages = []

        executable = [
            stage
            for stage in stages
            if _is_implemented(stage) and _has_visual_input(run, stage)
        ]
        result: OrchestrationResult | None = None
        if executable:
            try:
                result = _execute(
                    engine,
                    run,
                    config,
                    executable,
                    data_dir=database_path.parent,
                )
            except (ConfigError, SQLAlchemyError, ValueError, TypeError) as exc:
                print(f"error: cannot run the visual stages: {exc}", file=sys.stderr)
                return 1
            refreshed = RunRepository(engine).get(run.id)
            if refreshed is not None:
                run = refreshed

        beats = visual_beat_reports(run, stages=stages, result=result, engine=engine)
        if not beats:
            print(
                f"error: no visual beats found for run {run.id}",
                file=sys.stderr,
            )
            return 1

        total = (
            sum(float(getattr(item, "cost_usd", 0.0) or 0.0) for item in result.stages)
            if result is not None
            else None
        )
        _report(
            run,
            beats,
            stages=stages,
            cost_usd=total,
            json_output=bool(getattr(args, "json_output", False)),
        )
        if result is not None and result.failed_stage:
            print(
                f"error: run {run.id} failed at stage "
                f"{result.failed_stage!r}: {result.error_message}",
                file=sys.stderr,
            )
            return 1
        return 0
    except (ConfigError, SQLAlchemyError, ValueError, TypeError) as exc:
        print(f"error: cannot report the visual beats: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()


register_command(
    "visuals",
    run_visuals,
    help="Run or report the visual decisions, provider, cost and quality per beat",
    configure=configure,
)
