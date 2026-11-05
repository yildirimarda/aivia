"""Generate one shared visual set per run and reuse it for every language.

A run writes one script per language from the same fact base, but the visual
assets belong to the run, not to a language. This module resolves the visual
requests once from a single source language, generates each asset once and then
exposes the same results to every language. No provider is called for a second
language and no cost is charged twice. When scripts have different beat counts,
the shared set covers the longest script: positions missing from the source
language fall back to the first language that has them, and each language uses
only the prefix matching its own beat count.

The decision is deterministic: the source language is explicit or the first
language in input order, the requests come from the existing strategy resolver
and every language references the same ordered beats. A caller may also supply
already resolved results, for example when adding a language to an existing
run, and the stage will share those without touching a provider at all.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from aivia.core.config.visuals import VisualsConfig
from aivia.core.errors import ConfigError
from aivia.core.models.run import StageResult, StageState
from aivia.core.models.script import Script, VisualType
from aivia.core.stages.base import Stage, StageContext
from aivia.visuals.image_stage import (
    _text,
    _write_json,
    resolve_image_request,
)
from aivia.visuals.stock_stage import resolve_stock_request
from aivia.visuals.strategy import (
    VisualRequest,
    resolve_visual_requests,
)
from aivia.visuals.video_stage import resolve_video_request

__all__ = [
    "DEFAULT_REUSE_FILENAME",
    "DEFAULT_REUSE_MANIFEST_FILENAME",
    "DEFAULT_SHARED_VISUALS_FILENAME",
    "DEFAULT_VISUAL_REUSE_FILENAME",
    "REUSE_STAGE_NAME",
    "SHARED_VISUALS_FILENAME",
    "VISUAL_REUSE_STAGE_NAME",
    "ResolvedSharedVisuals",
    "SharedVisualBeat",
    "SharedVisuals",
    "SharedVisualsResult",
    "VisualReuse",
    "VisualReuseBeat",
    "VisualReuseResolver",
    "VisualReuseStage",
    "VisualReuseStageResult",
    "VisualReuseStore",
    "build_visual_reuse_stage",
    "generate_shared_visuals",
    "generate_visuals_once",
    "resolve_shared_visuals",
    "resolve_visual_reuse",
    "resolve_visual_reuse_sync",
    "reuse_visuals",
    "run_visual_reuse_stage",
    "share_visual_requests",
    "share_visuals",
    "shared_visual_requests",
    "shared_visuals",
    "visual_reuse",
    "visual_reuse_stage",
]

VISUAL_REUSE_STAGE_NAME = "visual_reuse"
"""The recipe stage name this implementation serves."""

REUSE_STAGE_NAME = VISUAL_REUSE_STAGE_NAME
"""Alias for :data:`VISUAL_REUSE_STAGE_NAME`."""

DEFAULT_VISUAL_REUSE_FILENAME = "visual-reuse.json"
"""The manifest that records the shared visual set under the run directory."""

DEFAULT_REUSE_FILENAME = DEFAULT_VISUAL_REUSE_FILENAME
"""Alias for :data:`DEFAULT_VISUAL_REUSE_FILENAME`."""

DEFAULT_REUSE_MANIFEST_FILENAME = DEFAULT_VISUAL_REUSE_FILENAME
"""Alias for :data:`DEFAULT_VISUAL_REUSE_FILENAME`."""

DEFAULT_SHARED_VISUALS_FILENAME = "shared-visuals.json"
"""A second manifest name kept for callers that prefer the shared vocabulary."""

SHARED_VISUALS_FILENAME = DEFAULT_SHARED_VISUALS_FILENAME
"""Alias for :data:`DEFAULT_SHARED_VISUALS_FILENAME`."""

DEFAULT_SHARED_FILENAME = DEFAULT_SHARED_VISUALS_FILENAME
"""Alias for :data:`DEFAULT_SHARED_VISUALS_FILENAME`."""

SHARED_FILENAME = DEFAULT_SHARED_FILENAME
"""Alias for :data:`DEFAULT_SHARED_FILENAME`."""

DEFAULT_SHARED_VISUALS_MANIFEST_FILENAME = DEFAULT_SHARED_VISUALS_FILENAME
"""Alias for :data:`DEFAULT_SHARED_VISUALS_FILENAME`."""

_IMAGE_TYPES = frozenset(
    {
        VisualType.IMAGE,
        VisualType.STILL,
        VisualType.TEXT,
    }
)
_VIDEO_TYPES = frozenset({VisualType.VIDEO})
_STOCK_TYPES = frozenset({VisualType.STOCK})


def _configured_visual_options(
    config: Any,
    strategy: Any,
    max_ai_video_beats: int | None,
) -> tuple[Any, int]:
    """Resolve the visuals config defaults, keeping explicit options on top."""
    configured = VisualsConfig.from_config(config)
    chosen_strategy = (
        configured.strategy if strategy is None or strategy == "" else strategy
    )
    chosen_limit = (
        configured.max_ai_video_beats
        if max_ai_video_beats is None
        else max_ai_video_beats
    )
    return chosen_strategy, chosen_limit


def _config_default_language(config: Any) -> str:
    section = (
        config.get("languages")
        if isinstance(config, Mapping)
        else getattr(config, "languages", None)
    )
    raw = getattr(section, "default", None)
    if raw is None and isinstance(section, Mapping):
        raw = section.get("default")
    return _language_key(raw)


def _language_key(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("language must be a string")
    return value.strip().replace("_", "-").lower()


def _normalise_languages(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values: Sequence[Any] = [value]
    elif isinstance(value, Mapping):
        values = list(value)
    elif isinstance(value, Sequence):
        values = value
    else:
        raise ConfigError(
            "languages must be a language code or a list of language codes",
            details={"value": type(value).__name__},
        )
    languages: list[str] = []
    for item in values:
        code = _language_key(item)
        if code and code not in languages:
            languages.append(code)
    return languages


def _script_from(value: Any) -> Script | None:
    if isinstance(value, Script):
        return value
    if isinstance(value, Mapping):
        try:
            return Script.model_validate(value)
        except Exception:
            return None
    return None


def _scripts_from(value: Any) -> dict[str, Script]:
    """Coerce a mapping, sequence or single script into language -> script."""
    if value is None:
        return {}
    if isinstance(value, Script):
        code = _language_key(value.language) or "default"
        return {code: value}
    if isinstance(value, Mapping):
        scripts: dict[str, Script] = {}
        for key, item in value.items():
            script = _script_from(item)
            if script is None:
                continue
            code = _language_key(key) or _language_key(script.language)
            if not code:
                code = "default"
            scripts[code] = script
        if scripts:
            return scripts
        return {}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        scripts = {}
        for position, item in enumerate(value):
            script = _script_from(item)
            if script is None:
                continue
            code = _language_key(script.language) or f"language-{position}"
            scripts[code] = script
        return scripts
    nested = getattr(value, "scripts", None)
    if isinstance(nested, Mapping):
        return _scripts_from(nested)
    return {}


def _is_request(value: Any) -> bool:
    if isinstance(value, VisualRequest):
        return True
    if isinstance(value, Mapping):
        return "prompt" in value or "visual_type" in value or "kind" in value
    return False


def _requests_from(value: Any) -> list[VisualRequest]:
    if isinstance(value, VisualRequest):
        return [value]
    if isinstance(value, Mapping):
        if _is_request(value):
            return [VisualRequest.model_validate(value)]
        requests: list[VisualRequest] = []
        for item in value.values():
            requests.extend(_requests_from(item))
        return requests
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        requests = []
        for item in value:
            requests.extend(_requests_from(item))
        return requests
    return []


def _source_language(
    value: Any,
    scripts: Mapping[str, Script],
    languages: Sequence[str],
) -> str:
    requested = _language_key(value)
    if requested:
        if scripts and requested not in scripts:
            available = ", ".join(scripts)
            raise ConfigError(
                f"unknown source language: {requested} (available: {available})",
                details={"language": requested, "available": list(scripts)},
            )
        return requested
    if languages:
        return languages[0]
    if scripts:
        return next(iter(scripts))
    return ""


def _beat_counts(value: Any) -> dict[str, int]:
    """Return the beat count of every script in *value* keyed by language."""
    return {code: len(script.beats) for code, script in _scripts_from(value).items()}


def _requests_by_language(
    scripts: Mapping[str, Script],
    *,
    strategy: Any,
    max_ai_video_beats: int,
    aspect_ratio: str,
    duration_seconds: float | None,
) -> dict[str, list[VisualRequest]]:
    return {
        code: resolve_visual_requests(
            script,
            strategy=strategy,
            max_ai_video_beats=max_ai_video_beats,
            aspect_ratio=aspect_ratio,
            duration_seconds=duration_seconds,
        )
        for code, script in scripts.items()
    }


def _merge_requests(
    by_language: Mapping[str, list[VisualRequest]],
    *,
    source: str,
    order: Sequence[str],
) -> list[VisualRequest]:
    """Build one shared request list covering the longest language script.

    A position comes from the source language when it has a beat there and
    falls back to the first other language that does. Equal length scripts
    therefore keep using exactly the source language's requests.
    """
    preferred = [source, *(code for code in order if code != source)]
    count = max((len(requests) for requests in by_language.values()), default=0)
    merged: list[VisualRequest] = []
    for position in range(count):
        for code in preferred:
            requests = by_language.get(code, [])
            if position < len(requests):
                merged.append(requests[position])
                break
    return merged


def _request_input(
    value: Any,
    *,
    source_language: str,
    strategy: Any,
    max_ai_video_beats: int,
    aspect_ratio: str,
    duration_seconds: float | None,
) -> list[VisualRequest]:
    requests = _requests_from(value)
    if requests:
        return requests
    scripts = _scripts_from(value)
    if scripts:
        source = _source_language(source_language, scripts, list(scripts))
        by_language = _requests_by_language(
            scripts,
            strategy=strategy,
            max_ai_video_beats=max_ai_video_beats,
            aspect_ratio=aspect_ratio,
            duration_seconds=duration_seconds,
        )
        return _merge_requests(
            by_language,
            source=source,
            order=list(scripts),
        )
    if isinstance(value, (Script, Mapping)) or (
        isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
    ):
        return resolve_visual_requests(
            cast(Any, value),
            strategy=strategy,
            max_ai_video_beats=max_ai_video_beats,
            aspect_ratio=aspect_ratio,
            duration_seconds=duration_seconds,
        )
    raise ConfigError(
        "visual reuse needs scripts or visual requests",
        details={"value": type(value).__name__},
    )


def shared_visual_requests(
    value: Any,
    *,
    source_language: Any = None,
    language: Any = None,
    strategy: Any = None,
    max_ai_video_beats: int | None = None,
    aspect_ratio: str = "",
    duration_seconds: float | None = None,
    config: Any = None,
) -> list[VisualRequest]:
    """Resolve the shared requests, covering the longest language script."""
    scripts = _scripts_from(value)
    source = _source_language(
        source_language if source_language is not None else language,
        scripts,
        list(scripts),
    )
    strategy, max_ai_video_beats = _configured_visual_options(
        config,
        strategy,
        max_ai_video_beats,
    )
    return _request_input(
        value,
        source_language=source,
        strategy=strategy,
        max_ai_video_beats=max_ai_video_beats,
        aspect_ratio=aspect_ratio,
        duration_seconds=duration_seconds,
    )


def share_visual_requests(
    value: Any,
    *,
    requests: Any = None,
    source_language: Any = None,
    language: Any = None,
    languages: Any = None,
    strategy: Any = None,
    max_ai_video_beats: int | None = None,
    aspect_ratio: str = "",
    duration_seconds: float | None = None,
    config: Any = None,
) -> dict[str, list[VisualRequest]]:
    """Return the shared requests for every language in *value*.

    The shared list covers the longest script. Each language receives only the
    prefix that matches its own beat count, so a shorter script never sees
    requests for beats it does not have.
    """
    scripts = _scripts_from(value)
    codes = _normalise_languages(languages)
    if not codes:
        codes = list(scripts)
    source = _source_language(
        source_language if source_language is not None else language,
        scripts,
        codes,
    )
    strategy, max_ai_video_beats = _configured_visual_options(
        config,
        strategy,
        max_ai_video_beats,
    )
    active = (
        _requests_from(requests)
        if requests is not None
        else _request_input(
            value,
            source_language=source,
            strategy=strategy,
            max_ai_video_beats=max_ai_video_beats,
            aspect_ratio=aspect_ratio,
            duration_seconds=duration_seconds,
        )
    )
    return {
        code: list(active[: len(scripts[code].beats)])
        if code in scripts
        else list(active)
        for code in codes
    }


def _asset_of(value: Any) -> Any:
    if value is None:
        return None
    asset = getattr(value, "asset", None)
    if asset is not None:
        return asset
    if isinstance(value, Mapping):
        return value.get("asset")
    return value


def _result_cost(value: Any) -> float:
    amount = getattr(value, "cost_usd", None)
    if isinstance(amount, bool) or not isinstance(amount, (int, float)):
        return 0.0
    return max(0.0, float(amount))


def _result_index(value: Any, position: int) -> int:
    index = getattr(value, "index", None)
    if isinstance(index, int) and not isinstance(index, bool) and index >= 0:
        return index
    return position


def _result_request(value: Any) -> VisualRequest | None:
    request = getattr(value, "request", None)
    return request if isinstance(request, VisualRequest) else None


def _result_dict(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def _coerce_results(value: Any, source_language: str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, SharedVisuals):
        return [beat.visual for beat in value.beats if beat.visual is not None]
    if isinstance(value, Mapping):
        if _is_request(value):
            return [value]
        selected = value.get(source_language)
        if selected is None:
            selected = next(iter(value.values()), None)
        return _coerce_results(selected, source_language)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return [value]


class SharedVisualBeat(BaseModel):
    """One generated visual beat shared by every language."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    index: int = Field(default=0, ge=0)
    request: VisualRequest | None = None
    visual: Any = Field(default=None, exclude=True)
    languages: list[str] = Field(default_factory=list)
    source_language: str = ""
    cost_usd: float = Field(default=0.0, ge=0.0)
    reused: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def result(self) -> Any:
        """The resolved visual result that every language reuses."""
        return self.visual

    @property
    def asset(self) -> Any:
        """The stored asset of the resolved result, when there is one."""
        return _asset_of(self.visual)

    @property
    def selected(self) -> Any:
        """Alias for :attr:`asset`."""
        return self.asset

    @property
    def first(self) -> Any:
        """Alias for :attr:`asset`."""
        return self.asset

    @property
    def resolved(self) -> bool:
        """Whether the shared beat produced an asset."""
        return self.asset is not None

    @property
    def passed(self) -> bool:
        """Alias for :attr:`resolved`."""
        return self.resolved

    @property
    def provider(self) -> str:
        """The provider reported by the shared result."""
        return _text(getattr(self.visual, "provider", "")) or _text(
            getattr(self.asset, "provider", "")
        )

    @property
    def model(self) -> str:
        """The model reported by the shared result."""
        return _text(getattr(self.visual, "model", "")) or _text(
            getattr(self.asset, "model", "")
        )

    @property
    def prompt_hash(self) -> str:
        """The prompt hash reported by the shared result."""
        return _text(getattr(self.visual, "prompt_hash", ""))

    @property
    def language(self) -> str:
        """Alias for :attr:`source_language`."""
        return self.source_language

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for the manifest."""
        asset = self.asset
        return {
            "index": self.index,
            "request": self.request.to_dict() if self.request is not None else None,
            "result": _result_dict(self.visual),
            "asset": _result_dict(asset),
            "languages": list(self.languages),
            "source_language": self.source_language,
            "cost_usd": self.cost_usd,
            "reused": self.reused,
            "provider": self.provider,
            "model": self.model,
            "prompt_hash": self.prompt_hash,
            "metadata": dict(self.metadata),
        }


VisualReuseBeat = SharedVisualBeat
SharedVisualBeatResult = SharedVisualBeat
ResolvedSharedVisuals = SharedVisualBeat


class SharedVisuals(BaseModel):
    """The run-level visual set and the languages that share it."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    source_language: str = ""
    languages: list[str] = Field(default_factory=list)
    beats: list[SharedVisualBeat] = Field(
        default_factory=list,
        validation_alias=AliasChoices("beats", "results", "items", "visuals"),
    )
    cost_usd: float = Field(default=0.0, ge=0.0)
    beat_counts: dict[str, int] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def requests(self) -> list[VisualRequest]:
        """The shared requests in beat order."""
        return [beat.request for beat in self.beats if beat.request is not None]

    @property
    def results(self) -> list[Any]:
        """The shared visual results in beat order."""
        return [beat.visual for beat in self.beats if beat.visual is not None]

    @property
    def assets(self) -> list[Any]:
        """Every shared asset in beat order."""
        return [beat.asset for beat in self.beats if beat.asset is not None]

    @property
    def resolved(self) -> list[SharedVisualBeat]:
        """Beats whose shared result produced an asset."""
        return [beat for beat in self.beats if beat.resolved]

    @property
    def unresolved(self) -> list[SharedVisualBeat]:
        """Beats whose shared result produced no asset."""
        return [beat for beat in self.beats if not beat.resolved]

    @property
    def total_cost_usd(self) -> float:
        """The total cost charged once for the whole run."""
        return math.fsum(beat.cost_usd for beat in self.beats)

    @property
    def by_language(self) -> dict[str, list[SharedVisualBeat]]:
        """The shared beats for each language, trimmed to its beat count."""
        return {code: self.for_language(code) for code in self.languages}

    def for_language(self, language: str) -> list[SharedVisualBeat]:
        """Return the shared beats for *language*.

        The shared set covers the longest script. A language with fewer beats
        receives only the prefix it can use; a language with no recorded count
        receives the whole set.
        """
        code = _language_key(language)
        count = self.beat_counts.get(code)
        if count is None:
            return list(self.beats)
        return list(self.beats[:count])

    def visuals_for(self, language: str) -> list[SharedVisualBeat]:
        """Alias for :meth:`for_language`."""
        return self.for_language(language)

    def get_language(self, language: str) -> list[SharedVisualBeat] | None:
        """Return the shared beats only when *language* is part of the run."""
        code = _language_key(language)
        return self.for_language(code) if code in self.languages else None

    def add_language(
        self,
        language: str,
        beat_count: int | None = None,
    ) -> SharedVisuals:
        """Return a copy that also shares this set with *language*.

        *beat_count* records how many of the shared beats the language can use.
        When omitted the whole set is assumed, which keeps the set usable for
        callers that only add a language name.
        """
        code = _language_key(language)
        languages = list(self.languages)
        if code and code not in languages:
            languages.append(code)
        counts = dict(self.beat_counts)
        if code:
            counts[code] = len(self.beats) if beat_count is None else max(0, beat_count)
        beats = [
            beat.model_copy(
                update={"languages": list(dict.fromkeys([*beat.languages, code]))}
            )
            for beat in self.beats
        ]
        return self.model_copy(
            update={"languages": languages, "beats": beats, "beat_counts": counts}
        )

    def __getitem__(self, language: str) -> list[SharedVisualBeat]:
        return self.for_language(language)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for the manifest."""
        return {
            "source_language": self.source_language,
            "languages": list(self.languages),
            "beats": [beat.to_dict() for beat in self.beats],
            "beat_counts": dict(self.beat_counts),
            "cost_usd": self.cost_usd,
            "metadata": dict(self.metadata),
        }


SharedVisualsResult = SharedVisuals
SharedVisualPlan = SharedVisuals
SharedVisualSet = SharedVisuals
VisualReusePlan = SharedVisuals
VisualReuseResult = SharedVisuals


def _build_shared(
    requests: Sequence[VisualRequest],
    results: Sequence[Any],
    *,
    source_language: str,
    languages: Sequence[str],
    beat_counts: Mapping[str, int] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SharedVisuals:
    beats: list[SharedVisualBeat] = []
    for position, request in enumerate(requests):
        result = results[position] if position < len(results) else None
        beats.append(
            SharedVisualBeat(
                index=_result_index(result, request.index),
                request=request,
                visual=result,
                languages=list(languages),
                source_language=source_language,
                cost_usd=_result_cost(result),
                metadata={"position": position},
            )
        )
    for position in range(len(requests), len(results)):
        result = results[position]
        beats.append(
            SharedVisualBeat(
                index=_result_index(result, position),
                request=_result_request(result),
                visual=result,
                languages=list(languages),
                source_language=source_language,
                cost_usd=_result_cost(result),
                metadata={"position": position},
            )
        )
    return SharedVisuals(
        source_language=source_language,
        languages=list(languages),
        beats=beats,
        cost_usd=math.fsum(beat.cost_usd for beat in beats),
        beat_counts=dict(beat_counts or {}),
        metadata=dict(metadata or {}),
    )


def _extend_shared(
    shared: SharedVisuals,
    requests: Sequence[VisualRequest],
    results: Sequence[Any],
    *,
    language: str,
    beat_count: int,
) -> SharedVisuals:
    """Append newly generated beats to an existing shared visual set."""
    code = _language_key(language)
    languages = list(dict.fromkeys([*shared.languages, code]))
    beats = [
        beat.model_copy(
            update={"languages": list(dict.fromkeys([*beat.languages, code]))}
        )
        for beat in shared.beats
    ]
    for offset, (request, result) in enumerate(zip(requests, results, strict=False)):
        position = len(shared.beats) + offset
        beats.append(
            SharedVisualBeat(
                index=_result_index(result, request.index),
                request=request,
                visual=result,
                languages=languages,
                source_language=shared.source_language,
                cost_usd=_result_cost(result),
                metadata={"position": position},
            )
        )
    counts = dict(shared.beat_counts)
    counts[code] = max(0, beat_count)
    return shared.model_copy(
        update={
            "languages": languages,
            "beats": beats,
            "beat_counts": counts,
            "cost_usd": math.fsum(beat.cost_usd for beat in beats),
        }
    )


def share_visuals(
    value: Any = None,
    *,
    requests: Any = None,
    results: Any = None,
    visuals: Any = None,
    source_language: Any = None,
    language: Any = None,
    languages: Any = None,
    strategy: Any = None,
    max_ai_video_beats: int | None = None,
    aspect_ratio: str = "",
    duration_seconds: float | None = None,
    metadata: Mapping[str, Any] | None = None,
    config: Any = None,
) -> SharedVisuals:
    """Share requests or already resolved results across every language.

    No provider is called. When *results* (or *visuals*) is supplied the source
    language's results become the shared set; otherwise the shared set only
    carries the requests so a later step can resolve them once.
    """
    scripts = _scripts_from(value)
    codes = _normalise_languages(languages)
    if not codes:
        codes = list(scripts)
    source = _source_language(
        source_language if source_language is not None else language,
        scripts,
        codes,
    )
    strategy, max_ai_video_beats = _configured_visual_options(
        config,
        strategy,
        max_ai_video_beats,
    )
    selected_results = results if results is not None else visuals
    active_results = _coerce_results(selected_results, source)
    if requests is not None:
        active_requests = _requests_from(requests)
    elif selected_results is not None:
        active_requests = _requests_with_results(
            value,
            active_results,
            source_language=source,
            strategy=strategy,
            max_ai_video_beats=max_ai_video_beats,
            aspect_ratio=aspect_ratio,
            duration_seconds=duration_seconds,
        )
    else:
        active_requests = _request_input(
            value,
            source_language=source,
            strategy=strategy,
            max_ai_video_beats=max_ai_video_beats,
            aspect_ratio=aspect_ratio,
            duration_seconds=duration_seconds,
        )
    return _build_shared(
        active_requests,
        active_results,
        source_language=source,
        languages=codes,
        beat_counts=_beat_counts(value),
        metadata=metadata,
    )


def _requests_from_results(results: Sequence[Any]) -> list[VisualRequest]:
    requests: list[VisualRequest] = []
    for result in results:
        request = _result_request(result)
        if request is not None:
            requests.append(request)
    return requests


def _requests_with_results(
    value: Any,
    results: Sequence[Any],
    *,
    source_language: str,
    strategy: Any,
    max_ai_video_beats: int,
    aspect_ratio: str,
    duration_seconds: float | None,
) -> list[VisualRequest]:
    """Keep result requests and add fallback requests for missing beats."""
    result_requests = _requests_from_results(results)
    if not _scripts_from(value):
        return result_requests
    script_requests = _request_input(
        value,
        source_language=source_language,
        strategy=strategy,
        max_ai_video_beats=max_ai_video_beats,
        aspect_ratio=aspect_ratio,
        duration_seconds=duration_seconds,
    )
    if len(script_requests) <= len(result_requests):
        return result_requests
    return [*result_requests, *script_requests[len(result_requests) :]]


def _callable_kwargs(method: Any, values: Mapping[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return {}
    parameters = signature.parameters
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return dict(values)
    return {name: value for name, value in values.items() if name in parameters}


class VisualReuse:
    """Resolve one shared visual set and hand it to every language.

    ``image``/``video``/``stock`` (and their aliases) are the collaborators the
    existing resolvers expect. A single ``router`` or ``media`` facade may be
    supplied instead. ``generator`` (or ``resolver``) overrides them all with a
    callable that receives a :class:`VisualRequest`, which keeps tests and
    custom pipelines independent of a provider.
    """

    def __init__(
        self,
        image: Any = None,
        *,
        image_service: Any = None,
        images: Any = None,
        video: Any = None,
        video_service: Any = None,
        videos: Any = None,
        stock: Any = None,
        stock_service: Any = None,
        search: Any = None,
        stock_search: Any = None,
        router: Any = None,
        media: Any = None,
        config: Any = None,
        generator: Any = None,
        resolver: Any = None,
        results: Any = None,
        visuals: Any = None,
        source_language: Any = None,
        language: Any = None,
        strategy: Any = None,
        max_ai_video_beats: int | None = None,
        aspect_ratio: str = "",
        duration_seconds: float | None = None,
        **options: Any,
    ) -> None:
        self._image = image if image is not None else image_service
        if self._image is None:
            self._image = images
        self._video = video if video is not None else video_service
        if self._video is None:
            self._video = videos
        self._stock = stock if stock is not None else stock_service
        if self._stock is None:
            self._stock = search if search is not None else stock_search
        self._router = router
        self._media = media
        self._config = config
        self._generator = generator if generator is not None else resolver
        self._results = results if results is not None else visuals
        self.source_language = (
            source_language if source_language is not None else language
        )
        self._strategy = strategy
        self._max_ai_video_beats = max_ai_video_beats
        self.strategy, self.max_ai_video_beats = _configured_visual_options(
            config,
            strategy,
            max_ai_video_beats,
        )
        self.aspect_ratio = aspect_ratio
        self.duration_seconds = duration_seconds
        self.options = dict(options)

    def _visual_options(self, context: StageContext | None) -> tuple[Any, int]:
        config = self._config
        if config is None and context is not None:
            config = context.config
        return _configured_visual_options(
            config,
            self._strategy,
            self._max_ai_video_beats,
        )

    def _context_media(self, context: StageContext | None) -> Any:
        if context is None:
            return None
        try:
            return context.media
        except ValueError:
            return None

    def _service_for(self, request: VisualRequest, context: StageContext | None) -> Any:
        if request.visual_type in _VIDEO_TYPES:
            return (
                self._video if self._video is not None else self._context_media(context)
            )
        if request.visual_type in _STOCK_TYPES:
            return (
                self._stock if self._stock is not None else self._context_media(context)
            )
        return self._image if self._image is not None else self._context_media(context)

    async def _generate(
        self,
        request: VisualRequest,
        *,
        context: StageContext | None,
        run_id: str,
        metadata: Mapping[str, Any] | None,
    ) -> Any:
        values: dict[str, Any] = {
            "context": context,
            "run_id": run_id,
            "metadata": metadata,
            **self.options,
        }
        if self._generator is not None:
            kwargs = _callable_kwargs(self._generator, values)
            result = self._generator(request, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            return result
        service = self._service_for(request, context)
        config = (
            self._config
            if self._config is not None
            else getattr(context, "config", None)
        )
        if request.visual_type in _VIDEO_TYPES:
            return await resolve_video_request(
                request,
                service=service,
                router=self._router,
                context=context,
                config=config,
                run_id=run_id,
                metadata=metadata,
            )
        if request.visual_type in _STOCK_TYPES:
            return await resolve_stock_request(
                request,
                search=service,
                router=self._router,
                context=context,
                config=config,
                run_id=run_id,
                metadata=metadata,
            )
        return await resolve_image_request(
            request,
            service=service,
            router=self._router,
            context=context,
            config=config,
            run_id=run_id,
            metadata=metadata,
        )

    async def resolve(
        self,
        value: Any = None,
        *,
        requests: Any = None,
        results: Any = None,
        visuals: Any = None,
        source_language: Any = None,
        language: Any = None,
        languages: Any = None,
        context: StageContext | None = None,
        run_id: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> SharedVisuals:
        """Generate the source language's visuals once and share them."""
        scripts = _scripts_from(value)
        codes = _normalise_languages(languages)
        if not codes:
            codes = list(scripts)
        source = _source_language(
            source_language
            if source_language is not None
            else language
            if language is not None
            else self.source_language,
            scripts,
            codes,
        )
        if not codes and source:
            codes = [source]
        self.strategy, self.max_ai_video_beats = self._visual_options(context)
        selected = results if results is not None else visuals
        if selected is None:
            selected = self._results
        if selected is not None:
            active_results = _coerce_results(selected, source)
            if requests is not None:
                active_requests = _requests_from(requests)
            else:
                active_requests = _requests_with_results(
                    value,
                    active_results,
                    source_language=source,
                    strategy=self.strategy,
                    max_ai_video_beats=self.max_ai_video_beats,
                    aspect_ratio=self.aspect_ratio,
                    duration_seconds=self.duration_seconds,
                )
        else:
            if requests is not None:
                active_requests = _requests_from(requests)
            else:
                active_requests = _request_input(
                    value,
                    source_language=source,
                    strategy=self.strategy,
                    max_ai_video_beats=self.max_ai_video_beats,
                    aspect_ratio=self.aspect_ratio,
                    duration_seconds=self.duration_seconds,
                )
            generated: list[Any] = []
            for position, request in enumerate(active_requests):
                active_meta = dict(metadata or {})
                active_meta.setdefault("position", position)
                active_meta.setdefault("source_language", source)
                generated.append(
                    await self._generate(
                        request,
                        context=context,
                        run_id=run_id,
                        metadata=active_meta,
                    )
                )
            active_results = generated
        return _build_shared(
            active_requests,
            active_results,
            source_language=source,
            languages=codes,
            beat_counts=_beat_counts(value),
            metadata=metadata,
        )

    async def resolve_requests(
        self,
        requests: Any,
        *,
        languages: Any = None,
        source_language: Any = None,
        context: StageContext | None = None,
        run_id: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> SharedVisuals:
        """Resolve explicit requests once and share them."""
        return await self.resolve(
            requests=requests,
            languages=languages,
            source_language=source_language,
            context=context,
            run_id=run_id,
            metadata=metadata,
        )

    def resolve_sync(self, value: Any = None, **kwargs: Any) -> SharedVisuals:
        """Run :meth:`resolve` from synchronous code."""
        import asyncio

        return asyncio.run(self.resolve(value, **kwargs))

    __call__ = resolve


VisualReuseResolver = VisualReuse
VisualReusePlanner = VisualReuse


class VisualReuseStore:
    """Cache one shared visual set and answer later languages from it.

    The first call resolves and generates the source language's visuals. Every
    later call, for any language, returns the same beats without calling a
    provider again. The store keeps the shared set available through
    :attr:`shared`.
    """

    def __init__(
        self,
        image: Any = None,
        *,
        video: Any = None,
        stock: Any = None,
        generator: Any = None,
        resolver: Any = None,
        router: Any = None,
        media: Any = None,
        config: Any = None,
        source_language: Any = None,
        strategy: Any = None,
        max_ai_video_beats: int | None = None,
        aspect_ratio: str = "",
        duration_seconds: float | None = None,
        **options: Any,
    ) -> None:
        self._resolver = VisualReuse(
            image,
            video=video,
            stock=stock,
            generator=generator,
            resolver=resolver,
            router=router,
            media=media,
            config=config,
            source_language=source_language,
            strategy=strategy,
            max_ai_video_beats=max_ai_video_beats,
            aspect_ratio=aspect_ratio,
            duration_seconds=duration_seconds,
            **options,
        )
        self.shared: SharedVisuals | None = None

    async def resolve(
        self,
        value: Any = None,
        *,
        language: Any = None,
        languages: Any = None,
        context: StageContext | None = None,
        run_id: str = "",
        metadata: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[SharedVisualBeat]:
        """Return this language's shared beats, generating only on first use.

        When a later language has more beats than the shared set, only the
        missing beats are generated and appended. Beats that already exist are
        never requested or charged again.
        """
        code = _language_key(language)
        if self.shared is None:
            self.shared = await self._resolver.resolve(
                value,
                languages=languages or ([code] if code else None),
                context=context,
                run_id=run_id,
                metadata=metadata,
                **kwargs,
            )
        elif code and code not in self.shared.languages:
            self.shared = await self._add_language(
                self.shared,
                value,
                code,
                context=context,
                run_id=run_id,
                metadata=metadata,
            )
        return self.shared.for_language(code)

    async def _add_language(
        self,
        shared: SharedVisuals,
        value: Any,
        code: str,
        *,
        context: StageContext | None,
        run_id: str,
        metadata: Mapping[str, Any] | None,
    ) -> SharedVisuals:
        scripts = _scripts_from(value)
        script = scripts.get(code)
        if script is None and len(scripts) == 1:
            script = next(iter(scripts.values()))
        if script is None:
            return shared.add_language(code)
        count = len(script.beats)
        if count <= len(shared.beats):
            return shared.add_language(code, count)
        requests = resolve_visual_requests(
            script,
            strategy=self._resolver.strategy,
            max_ai_video_beats=self._resolver.max_ai_video_beats,
            aspect_ratio=self._resolver.aspect_ratio,
            duration_seconds=self._resolver.duration_seconds,
        )
        extra = requests[len(shared.beats) : count]
        generated: list[Any] = []
        for offset, request in enumerate(extra, start=len(shared.beats)):
            active_meta = dict(metadata or {})
            active_meta.setdefault("position", offset)
            active_meta.setdefault("source_language", shared.source_language)
            generated.append(
                await self._resolver._generate(
                    request,
                    context=context,
                    run_id=run_id,
                    metadata=active_meta,
                )
            )
        return _extend_shared(
            shared,
            extra,
            generated,
            language=code,
            beat_count=count,
        )

    def resolve_sync(self, value: Any = None, **kwargs: Any) -> list[SharedVisualBeat]:
        """Run :meth:`resolve` from synchronous code."""
        import asyncio

        return asyncio.run(self.resolve(value, **kwargs))


class VisualReuseStageResult(StageResult):
    """A stage result carrying the shared visual beats and their languages."""

    beats: list[SharedVisualBeat] = Field(
        default_factory=list,
        validation_alias=AliasChoices("beats", "results", "items", "visuals"),
    )
    source_language: str = ""
    languages: list[str] = Field(default_factory=list)
    beat_counts: dict[str, int] = Field(default_factory=dict)

    @property
    def results(self) -> list[SharedVisualBeat]:
        """Alias for :attr:`beats`."""
        return self.beats

    @property
    def visual_results(self) -> list[Any]:
        """The raw shared visual results in beat order."""
        return [beat.visual for beat in self.beats if beat.visual is not None]

    @property
    def requests(self) -> list[VisualRequest]:
        """The shared requests in beat order."""
        return [beat.request for beat in self.beats if beat.request is not None]

    @property
    def assets(self) -> list[Any]:
        """Every shared asset in beat order."""
        return [beat.asset for beat in self.beats if beat.asset is not None]

    @property
    def resolved(self) -> list[SharedVisualBeat]:
        """Beats whose shared result produced an asset."""
        return [beat for beat in self.beats if beat.resolved]

    @property
    def unresolved(self) -> list[SharedVisualBeat]:
        """Beats whose shared result produced no asset."""
        return [beat for beat in self.beats if not beat.resolved]

    @property
    def by_language(self) -> dict[str, list[SharedVisualBeat]]:
        """The shared beats for each language, trimmed to its beat count."""
        return {
            code: list(self.beats[: self.beat_counts[code]])
            if code in self.beat_counts
            else list(self.beats)
            for code in self.languages
        }

    @property
    def total_cost_usd(self) -> float:
        """The total cost charged once for the whole run."""
        return math.fsum(beat.cost_usd for beat in self.beats)


class VisualReuseStage(Stage):
    """Generate the shared visual set once and write its manifest.

    ``scripts`` may be a mapping of language to script, a sequence of scripts
    or a single script. ``requests`` may override the resolved requests and
    ``results`` may supply already resolved visuals so adding a language never
    charges the visual stage again. Without an explicit input the stage loads
    ``script.<language>.json`` files from the run directory.
    """

    name = VISUAL_REUSE_STAGE_NAME
    artifacts: tuple[str, ...] = (
        DEFAULT_VISUAL_REUSE_FILENAME,
        DEFAULT_SHARED_VISUALS_FILENAME,
    )

    def __init__(
        self,
        name: str | None = None,
        *,
        options: Mapping[str, Any] | None = None,
        scripts: Any = None,
        script: Any = None,
        languages: Any = None,
        language: Any = None,
        source_language: Any = None,
        requests: Any = None,
        visuals: Any = None,
        visual_requests: Any = None,
        results: Any = None,
        resolved: Any = None,
        image: Any = None,
        image_service: Any = None,
        images: Any = None,
        video: Any = None,
        video_service: Any = None,
        videos: Any = None,
        stock: Any = None,
        stock_service: Any = None,
        search: Any = None,
        stock_search: Any = None,
        router: Any = None,
        media: Any = None,
        config: Any = None,
        generator: Any = None,
        resolver: Any = None,
        strategy: Any = None,
        max_ai_video_beats: int | None = None,
        aspect_ratio: str = "",
        duration_seconds: float | None = None,
        filename: str | None = None,
        manifest_filename: str | None = None,
        **kwargs: Any,
    ) -> None:
        merged = dict(options or {})
        supplied = {
            "scripts": scripts if scripts is not None else script,
            "languages": languages,
            "language": language,
            "source_language": source_language,
            "requests": requests if requests is not None else visual_requests,
            "visuals": visuals,
            "results": results if results is not None else resolved,
            "image": image,
            "image_service": image_service,
            "images": images,
            "video": video,
            "video_service": video_service,
            "videos": videos,
            "stock": stock,
            "stock_service": stock_service,
            "search": search,
            "stock_search": stock_search,
            "router": router,
            "media": media,
            "config": config,
            "generator": generator,
            "resolver": resolver,
            "strategy": strategy,
            "max_ai_video_beats": max_ai_video_beats,
            "aspect_ratio": aspect_ratio,
            "duration_seconds": duration_seconds,
            "filename": filename,
            "manifest_filename": manifest_filename,
        }
        for key, value in supplied.items():
            if value is not None:
                merged[key] = value
        self.filename = _text(
            merged.pop("filename", None)
            or merged.pop("manifest_filename", None)
            or DEFAULT_VISUAL_REUSE_FILENAME
        )
        if not self.filename:
            self.filename = DEFAULT_VISUAL_REUSE_FILENAME
        self.artifacts = tuple(
            dict.fromkeys([self.filename, DEFAULT_SHARED_VISUALS_FILENAME])
        )

        self._scripts = merged.pop("scripts", None)
        self._languages = merged.pop("languages", None)
        self._language = merged.pop("language", None)
        self._source_language = merged.pop("source_language", None)
        self._requests = merged.pop("requests", None)
        if self._requests is None:
            self._requests = merged.pop("visuals", None)
        self._results = merged.pop("results", None)
        self._image = merged.pop("image", None)
        if self._image is None:
            self._image = merged.pop("image_service", None)
        if self._image is None:
            self._image = merged.pop("images", None)
        self._video = merged.pop("video", None)
        if self._video is None:
            self._video = merged.pop("video_service", None)
        if self._video is None:
            self._video = merged.pop("videos", None)
        self._stock = merged.pop("stock", None)
        if self._stock is None:
            self._stock = merged.pop("stock_service", None)
        if self._stock is None:
            self._stock = merged.pop("search", None)
        if self._stock is None:
            self._stock = merged.pop("stock_search", None)
        self._router = merged.pop("router", None)
        self._media = merged.pop("media", None)
        self._config = merged.pop("config", None)
        self._generator = merged.pop("generator", None)
        if self._generator is None:
            self._generator = merged.pop("resolver", None)
        self.strategy = merged.pop("strategy", None)
        self.max_ai_video_beats = merged.pop("max_ai_video_beats", None)
        self.aspect_ratio = merged.pop("aspect_ratio", "")
        self.duration_seconds = merged.pop("duration_seconds", None)
        self.results: list[SharedVisualBeat] = []
        self.shared: SharedVisuals | None = None
        super().__init__(name=name, options=merged, **kwargs)

    def _scripts_from_run(self, context: StageContext) -> dict[str, Script]:
        import json

        scripts: dict[str, Script] = {}
        for path in sorted(context.run_dir.glob("script.*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            script = _script_from(payload)
            if script is None:
                continue
            code = _language_key(path.name[len("script.") : -len(".json")])
            if not code:
                code = _language_key(script.language) or "default"
            scripts[code] = script
        return scripts

    def _resolver_for(self, context: StageContext) -> VisualReuse:
        return VisualReuse(
            self._image,
            video=self._video,
            stock=self._stock,
            router=self._router if self._router is not None else context.router,
            media=self._media,
            config=self._config if self._config is not None else context.config,
            generator=self._generator,
            source_language=self._source_language,
            strategy=self.strategy,
            max_ai_video_beats=self.max_ai_video_beats,
            aspect_ratio=self.aspect_ratio,
            duration_seconds=self.duration_seconds,
        )

    def _input(self, context: StageContext) -> Any:
        if self._scripts is not None:
            return self._scripts
        if self._requests is not None:
            return self._requests
        if self._results is not None:
            return None
        return self._scripts_from_run(context)

    async def run(self, context: StageContext) -> StageResult:
        """Generate the shared visuals once and write both manifest names."""
        resolver = self._resolver_for(context)
        run_id = context.run.id if context.run is not None else ""
        value = self._input(context)
        source = self._source_language
        if source is None and self._language is None:
            default = _config_default_language(context.config)
            scripts = _scripts_from(value)
            if default and default in scripts:
                source = default
        shared = await resolver.resolve(
            value,
            requests=self._requests,
            results=self._results,
            source_language=source,
            language=self._language,
            languages=self._languages,
            context=context,
            run_id=run_id,
        )
        self.shared = shared
        self.results = shared.beats

        resolved_count = len(shared.resolved)
        total_cost = shared.total_cost_usd
        providers = [beat.provider for beat in shared.beats if beat.provider]
        models = [beat.model for beat in shared.beats if beat.model]
        metadata = {
            "beats": len(shared.beats),
            "resolved": resolved_count,
            "unresolved": len(shared.beats) - resolved_count,
            "languages": list(shared.languages),
            "source_language": shared.source_language,
            "beat_counts": dict(shared.beat_counts),
            "cost_usd": total_cost,
            "providers": list(dict.fromkeys(providers)),
            "models": list(dict.fromkeys(models)),
            "reused_across": len(shared.languages),
        }
        payload: dict[str, Any] = {
            "version": 1,
            "stage": self.name,
            "source_language": shared.source_language,
            "languages": list(shared.languages),
            "beat_counts": dict(shared.beat_counts),
            "beats": [beat.to_dict() for beat in shared.beats],
            "metadata": metadata,
        }
        for relative in self.artifacts:
            _write_json(context.run_dir / relative, payload)
        return VisualReuseStageResult(
            run_id=run_id,
            stage=self.name,
            status=StageState.SUCCEEDED,
            artifacts=list(self.artifacts),
            cost_usd=total_cost,
            beats=shared.beats,
            source_language=shared.source_language,
            languages=list(shared.languages),
            beat_counts=dict(shared.beat_counts),
            metadata={**metadata, "manifest": self.filename},
        )


visual_reuse_stage = VisualReuseStage


def build_visual_reuse_stage(
    *,
    options: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> VisualReuseStage:
    """Build a :class:`VisualReuseStage` from recipe options and overrides."""
    return VisualReuseStage(options=options, **kwargs)


async def run_visual_reuse_stage(
    context: StageContext,
    *,
    stage: VisualReuseStage | None = None,
    **options: Any,
) -> StageResult:
    """Build a stage unless one is supplied, then run it asynchronously."""
    if stage is not None and options:
        raise ConfigError("pass either a stage or options, not both")
    active = stage if stage is not None else VisualReuseStage(**options)
    return await active.run(context)


async def resolve_visual_reuse(
    value: Any = None,
    *,
    image: Any = None,
    image_service: Any = None,
    images: Any = None,
    video: Any = None,
    video_service: Any = None,
    videos: Any = None,
    stock: Any = None,
    stock_service: Any = None,
    search: Any = None,
    stock_search: Any = None,
    router: Any = None,
    media: Any = None,
    config: Any = None,
    generator: Any = None,
    resolver: Any = None,
    results: Any = None,
    visuals: Any = None,
    requests: Any = None,
    source_language: Any = None,
    language: Any = None,
    languages: Any = None,
    context: StageContext | None = None,
    run_id: str = "",
    metadata: Mapping[str, Any] | None = None,
    strategy: Any = None,
    max_ai_video_beats: int | None = None,
    aspect_ratio: str = "",
    duration_seconds: float | None = None,
    **options: Any,
) -> SharedVisuals:
    """Generate the source language's visuals once and share them."""
    active = VisualReuse(
        image,
        image_service=image_service,
        images=images,
        video=video,
        video_service=video_service,
        videos=videos,
        stock=stock,
        stock_service=stock_service,
        search=search,
        stock_search=stock_search,
        router=router,
        media=media,
        config=config,
        generator=generator,
        resolver=resolver,
        results=results,
        visuals=visuals,
        source_language=source_language,
        language=language,
        strategy=strategy,
        max_ai_video_beats=max_ai_video_beats,
        aspect_ratio=aspect_ratio,
        duration_seconds=duration_seconds,
        **options,
    )
    return await active.resolve(
        value,
        requests=requests,
        source_language=source_language,
        language=language,
        languages=languages,
        context=context,
        run_id=run_id,
        metadata=metadata,
    )


def resolve_visual_reuse_sync(value: Any = None, **kwargs: Any) -> SharedVisuals:
    """Run :func:`resolve_visual_reuse` from synchronous code."""
    import asyncio

    return asyncio.run(resolve_visual_reuse(value, **kwargs))


reuse_visuals = resolve_visual_reuse
resolve_shared_visuals = resolve_visual_reuse
resolve_reuse = resolve_visual_reuse
reuse_shared_visuals = resolve_visual_reuse
generate_shared_visuals = resolve_visual_reuse
generate_visuals_once = resolve_visual_reuse
generate_once = resolve_visual_reuse
shared_visuals = share_visuals
shared_requests = share_visual_requests
reuse_requests = share_visual_requests
visual_reuse = resolve_visual_reuse
