"""Resolve stock beats from ordered queries, top results and relevance scores.

A stock beat carries a search query rather than a generation prompt. This stage
turns one query into a small number of provider results, scores those results
against the beat's intended description and keeps the best one. When the first
query finds nothing relevant it moves on to the next query instead of failing
the beat, so a narrow phrase can fall back to a broader description.

Resolution is deterministic for the same search responses, queries and scores.
The stage never picks a vendor: callers inject a stock provider, a router or a
plain search callable. Relevance scoring reuses the shared
:mod:`aivia.media.stock_relevance` ranker, and an explicitly injected ranker or
scorer takes precedence over the vision provider settings.
"""

from __future__ import annotations

import contextlib
import inspect
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from aivia.core.errors import ConfigError
from aivia.core.models.base import canonical_json
from aivia.core.models.run import StageResult, StageState
from aivia.core.models.script import MediaKind, VisualType
from aivia.core.stages.base import Stage, StageContext
from aivia.media.stock_relevance import (
    StockRanking,
    StockRelevance,
    rank_stock_candidates,
    relevance_threshold,
)
from aivia.providers.protocols.stock import (
    CAPABILITY_STOCK,
    StockAsset,
    StockMediaKind,
    StockOrientation,
    StockRequest,
    StockResponse,
)
from aivia.visuals.strategy import VisualRequest, resolve_visual_requests

__all__ = [
    "DEFAULT_QUERY_LIMIT",
    "DEFAULT_STOCK_FILENAME",
    "DEFAULT_TOP_RESULTS",
    "STOCK_STAGE_NAME",
    "ResolvedStock",
    "StockAttempt",
    "StockBeat",
    "StockBeatResolver",
    "StockBeatResult",
    "StockDecision",
    "StockQueryResolver",
    "StockResolver",
    "StockResult",
    "StockSelection",
    "StockStage",
    "StockStageResolver",
    "StockStageResult",
    "build_stock_queries",
    "build_stock_stage",
    "queries_for",
    "resolve_stock",
    "resolve_stock_beat",
    "resolve_stock_beats",
    "resolve_stock_request",
    "resolve_stock_request_sync",
    "resolve_stock_requests",
    "resolve_stock_requests_sync",
    "resolve_stocks",
    "run_stock_stage",
    "stock_queries",
    "stock_stage",
]

STOCK_STAGE_NAME = "stock"
"""The recipe stage name this implementation serves."""

DEFAULT_STOCK_FILENAME = "stock.json"
"""The manifest that records every resolved stock beat under the run directory."""

DEFAULT_TOP_RESULTS = 5
"""How many provider results one query contributes before relevance scoring."""

MAX_STOCK_QUERIES = 4
"""The largest number of queries one stock beat tries before giving up."""

DEFAULT_QUERY_LIMIT = MAX_STOCK_QUERIES
"""Alias for :data:`MAX_STOCK_QUERIES` used by callers and stages."""

FULL_RELEVANCE = 1.0
"""The score used when a candidate is accepted without a vision judgement."""

_QUERY_KEYS = ("queries", "search_queries", "stock_queries")
_FALLBACK_KEYS = (
    "fallback_queries",
    "query_alternatives",
    "alternate_queries",
    "next_queries",
)


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _texts(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items: Sequence[Any] = (value,)
    elif isinstance(value, Mapping):
        items = tuple(value.values())
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        items = value
    else:
        return []
    result: list[str] = []
    for item in items:
        text = _text(item)
        if text and text not in result:
            result.append(text)
    return result


def _request_metadata(request: VisualRequest) -> Mapping[str, Any]:
    metadata = getattr(request, "metadata", None)
    return metadata if isinstance(metadata, Mapping) else {}


def _coerce_request(value: Any) -> VisualRequest:
    if isinstance(value, VisualRequest):
        return value
    if isinstance(value, Mapping):
        return VisualRequest.model_validate(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ConfigError("stock resolution needs a non-empty query")
        return VisualRequest(
            visual_type=VisualType.STOCK,
            kind=MediaKind.STOCK,
            prompt=text,
            query=text,
            text=text,
        )
    requests = resolve_visual_requests(value, strategy="stock")
    return requests[0]


def stock_queries(
    value: Any,
    *,
    queries: Any = None,
    fallback_queries: Any = None,
    limit: int | None = None,
) -> list[str]:
    """Return the ordered, de-duplicated queries to try for one stock beat.

    An explicit ``queries`` mapping may key a beat by its index, query, prompt
    or text. A sequence supplies extra queries for the beat. ``fallback_queries``
    is always appended after the request's own query, prompt and text, and a
    ``queries``/``search_queries``/``fallback_queries`` list in the request
    metadata is honoured first. An empty query list means the beat has nothing
    to search.
    """
    request = _coerce_request(value)
    metadata = _request_metadata(request)
    ordered: list[str] = []

    def add(values: Any) -> None:
        for text in _texts(values):
            if text not in ordered:
                ordered.append(text)

    for key in _QUERY_KEYS:
        add(metadata.get(key))
        add(getattr(request, key, None))
    add(request.query)

    selected: Any = None
    if isinstance(queries, Mapping):
        for candidate in (
            request.index,
            str(request.index),
            request.query,
            request.prompt,
            request.text,
        ):
            if candidate in queries:
                selected = queries[candidate]
                break
    elif queries is not None:
        selected = queries
    add(selected)
    add(request.prompt)
    add(request.text)
    for key in _FALLBACK_KEYS:
        add(metadata.get(key))
        add(getattr(request, key, None))
    add(fallback_queries)

    if limit is None:
        return ordered
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ConfigError("query limit must be an integer")
    count = limit
    if count < 0:
        raise ConfigError("query limit must not be negative")
    return ordered[:count]


build_stock_queries = stock_queries
queries_for = stock_queries


def _orientation(value: str) -> StockOrientation | None:
    text = value.strip().lower()
    if not text:
        return None
    if text in {"landscape", "16:9", "16/9", "1.777", "1.78"}:
        return StockOrientation.LANDSCAPE
    if text in {"portrait", "9:16", "9/16", "0.5625"}:
        return StockOrientation.PORTRAIT
    if text in {"square", "1:1", "1/1", "1.0"}:
        return StockOrientation.SQUARE
    parts = text.replace("/", ":").split(":")
    if len(parts) == 2:
        try:
            width = float(parts[0])
            height = float(parts[1])
        except ValueError:
            return None
        if width <= 0 or height <= 0:
            return None
        if width > height:
            return StockOrientation.LANDSCAPE
        if width < height:
            return StockOrientation.PORTRAIT
        return StockOrientation.SQUARE
    return None


def _stock_kind(value: Any, request: VisualRequest) -> StockMediaKind:
    if isinstance(value, StockMediaKind):
        return value
    text = _text(value).lower()
    if not text:
        text = _text(_request_metadata(request).get("stock_kind")).lower()
    if text in {"video", "clip", "motion"}:
        return StockMediaKind.VIDEO
    return StockMediaKind.IMAGE


def _as_response(value: Any, request: StockRequest) -> StockResponse:
    if value is None:
        return StockResponse(query=request.query)
    if isinstance(value, StockResponse):
        return value
    if isinstance(value, StockAsset):
        return StockResponse(
            query=request.query,
            assets=[value],
            provider=value.provider,
        )
    if isinstance(value, Mapping):
        if "assets" in value:
            return StockResponse.model_validate(value)
        return StockResponse(
            query=request.query,
            assets=[StockAsset.model_validate(value)],
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        assets = [
            item if isinstance(item, StockAsset) else StockAsset.model_validate(item)
            for item in value
        ]
        return StockResponse(query=request.query, assets=assets)
    found = getattr(value, "assets", None)
    if isinstance(found, Sequence):
        return StockResponse(
            query=_text(getattr(value, "query", "")) or request.query,
            assets=list(found),
            provider=_text(getattr(value, "provider", "")),
        )
    raise TypeError(
        "a stock search must return a StockResponse, StockAsset, mapping or sequence"
    )


def _callable_arguments(
    function: Any,
    request: StockRequest,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return (request,), {}
    names = {
        name
        for name, parameter in parameters.items()
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    if not names:
        return (), {}
    if names <= {"request", "stock_request", "search_request"}:
        return (request,), {}
    if names <= {"query", "text", "description", "prompt"}:
        return (request.query,), {}
    return (request,), {}


async def _invoke_search(target: Any, request: StockRequest) -> Any:
    method = getattr(target, "search", None)
    if not callable(method):
        if not callable(target):
            raise TypeError("stock search must be callable or expose search()")
        method = target
    arguments, keywords = _callable_arguments(method, request)
    result = method(*arguments, **keywords)
    if inspect.isawaitable(result):
        result = await result
    return result


async def _call_search(
    target: Any,
    request: StockRequest,
    *,
    run_id: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> StockResponse:
    if target is None:
        raise ConfigError("stock resolution needs a stock provider or search callable")
    execute = getattr(target, "execute", None)
    if callable(execute) and not hasattr(target, "search"):
        result = execute(
            CAPABILITY_STOCK,
            lambda provider: _invoke_search(provider, request),
        )
        if inspect.isawaitable(result):
            result = await result
        return _as_response(result, request)
    result = await _invoke_search(target, request)
    return _as_response(result, request)


def _relevance(item: Any, *, threshold: float) -> StockRelevance:
    if isinstance(item, StockRelevance):
        return StockRelevance(
            asset=item.asset,
            score=item.score,
            description=item.description,
            reason=item.reason,
            threshold=threshold,
            metadata=dict(item.metadata),
        )
    if isinstance(item, StockAsset):
        return StockRelevance(asset=item, score=FULL_RELEVANCE, threshold=threshold)
    return StockRelevance(
        asset=StockAsset.model_validate(item),
        score=FULL_RELEVANCE,
        threshold=threshold,
    )


def _ranking_from(
    items: Sequence[Any],
    *,
    description: str,
    threshold: float,
) -> StockRanking:
    scored = [_relevance(item, threshold=threshold) for item in items]
    kept = [item for item in scored if item.score >= threshold]
    dropped = [item for item in scored if item.score < threshold]
    kept.sort(key=lambda item: item.score, reverse=True)
    return StockRanking(
        kept,
        dropped=dropped,
        description=description,
        min_relevance=threshold,
    )


def _as_ranking(
    value: Any,
    assets: Sequence[StockAsset],
    *,
    description: str,
    threshold: float,
) -> StockRanking:
    if isinstance(value, StockRanking):
        return _ranking_from(
            list(value),
            description=description,
            threshold=threshold,
        )
    if isinstance(value, StockRelevance):
        return _ranking_from(
            [value],
            description=description,
            threshold=threshold,
        )
    if isinstance(value, StockAsset):
        return _ranking_from(
            [value],
            description=description,
            threshold=threshold,
        )
    if isinstance(value, Mapping):
        for key in ("kept", "assets", "selected", "results", "items"):
            if key in value:
                return _as_ranking(
                    value[key],
                    assets,
                    description=description,
                    threshold=threshold,
                )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return _ranking_from(
            list(value),
            description=description,
            threshold=threshold,
        )
    kept = getattr(value, "kept", None)
    dropped = getattr(value, "dropped", None)
    if kept is not None:
        ranked = _ranking_from(
            list(kept),
            description=description,
            threshold=threshold,
        )
        rejected = [_relevance(item, threshold=threshold) for item in (dropped or ())]
        return StockRanking(
            list(ranked),
            dropped=[item for item in rejected if item.score < threshold],
            description=description,
            min_relevance=threshold,
        )
    first = getattr(value, "first", None)
    if first is not None:
        return _as_ranking(first, assets, description=description, threshold=threshold)
    raise TypeError("ranker must return a StockRanking, relevance list or asset list")


async def _call_ranker(
    ranker: Any,
    assets: Sequence[StockAsset],
    description: str,
    *,
    run_id: str,
    metadata: Mapping[str, Any] | None,
    config: Any,
    threshold: float,
) -> StockRanking:
    method = None
    for name in ("rank", "filter", "select", "rank_assets"):
        candidate = getattr(ranker, name, None)
        if callable(candidate):
            method = candidate
            break
    if method is None:
        if not callable(ranker):
            raise TypeError("ranker must be callable or expose rank()")
        method = ranker
    arguments: tuple[Any, ...] = (list(assets), description)
    keywords: dict[str, Any] = {}
    parameters: Mapping[str, inspect.Parameter]
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        parameters = {}
    names = list(parameters)
    if names and names[0] in {"description", "intended", "query", "prompt", "text"}:
        arguments = (description, list(assets))
    if "run_id" in parameters:
        keywords["run_id"] = run_id
    if metadata is not None and "metadata" in parameters:
        keywords["metadata"] = metadata
    if "config" in parameters:
        keywords["config"] = config
    if "min_relevance" in parameters:
        keywords["min_relevance"] = threshold
    result = method(*arguments, **keywords)
    if inspect.isawaitable(result):
        result = await result
    return _as_ranking(
        result,
        assets,
        description=description,
        threshold=threshold,
    )


async def _rank_assets(
    assets: Sequence[StockAsset],
    description: str,
    *,
    ranker: Any,
    scorer: Any,
    vision: Any,
    vision_provider: Any,
    helper: Any,
    model: str | None,
    config: Any,
    threshold: float,
    run_id: str,
    metadata: Mapping[str, Any] | None,
) -> StockRanking:
    if not assets:
        return StockRanking([], description=description, min_relevance=threshold)
    if ranker is not None:
        return await _call_ranker(
            ranker,
            assets,
            description,
            run_id=run_id,
            metadata=metadata,
            config=config,
            threshold=threshold,
        )
    active_scorer = scorer
    if active_scorer is not None or vision is not None or vision_provider is not None:
        return await rank_stock_candidates(
            list(assets),
            description,
            scorer=active_scorer,
            provider=vision_provider or vision,
            helper=helper,
            model=model,
            config=config,
            min_relevance=threshold,
            run_id=run_id,
            metadata=metadata,
        )
    return StockRanking(
        [
            StockRelevance(asset=asset, score=FULL_RELEVANCE, threshold=threshold)
            for asset in assets
        ],
        description=description,
        min_relevance=threshold,
    )


def _cost_of(response: StockResponse) -> float:
    value = response.metadata.get("cost_usd", 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return max(0.0, float(value))


class StockAttempt(BaseModel):
    """One query tried for a stock beat and what it produced."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    query: str
    provider: str = ""
    result_count: int = 0
    scores: dict[str, float] = Field(default_factory=dict)
    selected_id: str | None = None
    passed: bool = False
    error: str = ""
    response: StockResponse | None = None
    ranking: StockRanking | None = None

    @property
    def selected(self) -> StockAsset | None:
        """The asset this query selected, or ``None`` when it did not pass."""
        if self.ranking is None or self.ranking.first is None:
            return None
        return self.ranking.first.asset


class StockBeatResult(BaseModel):
    """The resolved stock asset for one beat, or why no query worked.

    Attribute access falls through to the selected :class:`StockAsset`, so a
    caller may read ``result.id`` or ``result.url`` as well as
    ``result.asset``. ``attempts`` records every query in the order it was
    tried and ``fallback_used`` is true once the first query did not win.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    index: int = Field(default=0, ge=0)
    request: VisualRequest | None = None
    query: str = ""
    queries: list[str] = Field(default_factory=list)
    asset: StockAsset | None = None
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    ranking: StockRanking | None = None
    attempts: list[StockAttempt] = Field(default_factory=list)
    fallback_used: bool = False
    provider: str = ""
    cost_usd: float = Field(default=0.0, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        asset = self.__dict__.get("asset")
        if asset is None:
            raise AttributeError(name)
        return getattr(asset, name)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, StockAsset):
            return self.asset == other
        return super().__eq__(other)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    @property
    def resolved(self) -> bool:
        """Whether a stock asset was selected."""
        return self.asset is not None

    @property
    def passed(self) -> bool:
        """Alias for :attr:`resolved`."""
        return self.resolved

    @property
    def selected(self) -> StockAsset | None:
        """Alias for :attr:`asset`."""
        return self.asset

    @property
    def first(self) -> StockAsset | None:
        """Alias for :attr:`asset`."""
        return self.asset

    @property
    def first_asset(self) -> StockAsset | None:
        """Alias for :attr:`asset`."""
        return self.asset

    @property
    def relevance(self) -> float | None:
        """Alias for :attr:`score`."""
        return self.score

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for the manifest."""
        return {
            "index": self.index,
            "request": (self.request.to_dict() if self.request is not None else None),
            "query": self.query,
            "queries": list(self.queries),
            "asset": (
                self.asset.model_dump(mode="json") if self.asset is not None else None
            ),
            "score": self.score,
            "attempts": [
                attempt.model_dump(mode="json", exclude={"response", "ranking"})
                for attempt in self.attempts
            ],
            "fallback_used": self.fallback_used,
            "provider": self.provider,
            "cost_usd": self.cost_usd,
            "metadata": dict(self.metadata),
        }


StockSelection = StockBeatResult
StockBeat = StockBeatResult
StockResult = StockBeatResult
ResolvedStock = StockBeatResult
StockDecision = StockBeatResult


class StockStageResult(StageResult):
    """A stage result carrying one :class:`StockBeatResult` per stock beat."""

    beats: list[StockBeatResult] = Field(
        default_factory=list,
        validation_alias=AliasChoices("beats", "results", "items", "selections"),
    )

    @property
    def results(self) -> list[StockBeatResult]:
        """Alias for :attr:`beats`."""
        return self.beats

    @property
    def assets(self) -> list[StockAsset]:
        """Every selected asset in beat order."""
        return [beat.asset for beat in self.beats if beat.asset is not None]

    @property
    def selected(self) -> list[StockAsset]:
        """Alias for :attr:`assets`."""
        return self.assets

    @property
    def unresolved(self) -> list[StockBeatResult]:
        """Beats for which no query produced a relevant asset."""
        return [beat for beat in self.beats if beat.asset is None]

    @property
    def queries(self) -> dict[int, str]:
        """The winning query keyed by beat index."""
        return {beat.index: beat.query for beat in self.beats}


async def resolve_stock_request(
    value: Any,
    *,
    search: Any = None,
    provider: Any = None,
    router: Any = None,
    ranker: Any = None,
    scorer: Any = None,
    judge: Any = None,
    analyzer: Any = None,
    vision: Any = None,
    vision_provider: Any = None,
    llm: Any = None,
    helper: Any = None,
    model: str | None = None,
    config: Any = None,
    min_relevance: float | None = None,
    threshold: float | None = None,
    top_results: int = DEFAULT_TOP_RESULTS,
    per_page: int | None = None,
    queries: Any = None,
    fallback_queries: Any = None,
    query_limit: int | None = DEFAULT_QUERY_LIMIT,
    stock_kind: Any = None,
    kind: Any = None,
    orientation: Any = None,
    min_width: int | None = None,
    min_height: int | None = None,
    min_duration_seconds: float | None = None,
    max_duration_seconds: float | None = None,
    run_id: str = "",
    metadata: Mapping[str, Any] | None = None,
    **values: Any,
) -> StockBeatResult:
    """Resolve one stock beat through its ordered queries.

    ``search`` (or ``provider``) may be a stock provider, a router or an async
    callable. The first query whose top results contain an asset at or above
    the relevance threshold wins; otherwise the next query is tried. The
    returned result is unresolved only when every query failed to produce a
    relevant asset.
    """
    request = _coerce_request(value)
    query_values = stock_queries(
        request,
        queries=queries,
        fallback_queries=fallback_queries,
        limit=query_limit,
    )
    if not query_values:
        return StockBeatResult(
            index=request.index,
            request=request,
            metadata={"reason": "no query"},
        )

    if isinstance(top_results, bool) or not isinstance(top_results, int):
        raise ConfigError("top_results must be an integer")
    if top_results < 1:
        raise ConfigError("top_results must be positive")
    page_size = per_page if per_page is not None else top_results
    if isinstance(page_size, bool) or not isinstance(page_size, int):
        raise ConfigError("per_page must be an integer")
    if page_size < 1 or page_size > 200:
        raise ConfigError("per_page must be between 1 and 200")

    target = search if search is not None else provider
    if target is None:
        target = router
    if target is None:
        raise ConfigError("stock resolution needs a stock provider or search callable")

    active_kind = _stock_kind(stock_kind if stock_kind is not None else kind, request)
    request_meta = _request_metadata(request)
    active_orientation = orientation
    if active_orientation is None:
        active_orientation = request_meta.get("orientation")
    if active_orientation is None:
        active_orientation = _orientation(request.aspect_ratio)
    elif isinstance(active_orientation, str):
        active_orientation = _orientation(active_orientation)
    active_metadata = dict(metadata or {})
    active_metadata.update(
        {
            "beat_index": request.index,
            "query": query_values[0],
        }
    )
    description = request.prompt or request.query or query_values[0]
    resolved_threshold = relevance_threshold(
        config,
        min_relevance=min_relevance,
        threshold=threshold,
    )
    active_scorer = scorer or judge or analyzer
    active_vision = vision or vision_provider or llm
    if active_vision is not None and hasattr(active_vision, "search"):
        active_vision = None
    if vision_provider is not None and hasattr(vision_provider, "search"):
        vision_provider = None

    attempts: list[StockAttempt] = []
    total_cost = 0.0
    last_ranking: StockRanking | None = None
    for position, query in enumerate(query_values):
        stock_request = StockRequest(
            query=query,
            kind=active_kind,
            orientation=active_orientation,
            min_width=min_width,
            min_height=min_height,
            min_duration_seconds=min_duration_seconds,
            max_duration_seconds=max_duration_seconds,
            per_page=page_size,
            metadata=dict(active_metadata),
        )
        try:
            response = await _call_search(
                target,
                stock_request,
                run_id=run_id,
                metadata=metadata,
            )
        except Exception as exc:
            attempts.append(
                StockAttempt(
                    query=query,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        total_cost += _cost_of(response)
        assets = list(response.assets)[:top_results]
        ranking = await _rank_assets(
            assets,
            description,
            ranker=ranker,
            scorer=active_scorer,
            vision=active_vision,
            vision_provider=vision_provider,
            helper=helper,
            model=model,
            config=config,
            threshold=resolved_threshold,
            run_id=run_id,
            metadata=metadata,
        )
        last_ranking = ranking
        selected = ranking.first
        attempt = StockAttempt(
            query=query,
            provider=response.provider,
            result_count=len(assets),
            scores=ranking.scores,
            selected_id=selected.asset.id if selected is not None else None,
            passed=selected is not None,
            response=response,
            ranking=ranking,
        )
        attempts.append(attempt)
        if selected is not None:
            return StockBeatResult(
                index=request.index,
                request=request,
                query=query,
                queries=query_values,
                asset=selected.asset,
                score=selected.score,
                ranking=ranking,
                attempts=attempts,
                fallback_used=position > 0,
                provider=selected.asset.provider or response.provider,
                cost_usd=total_cost,
                metadata={
                    "description": description,
                    "min_relevance": resolved_threshold,
                    "top_results": top_results,
                },
            )

    return StockBeatResult(
        index=request.index,
        request=request,
        query=query_values[-1],
        queries=query_values,
        asset=None,
        score=None,
        ranking=last_ranking,
        attempts=attempts,
        fallback_used=len(attempts) > 1,
        provider=attempts[-1].provider if attempts else "",
        cost_usd=total_cost,
        metadata={
            "description": description,
            "min_relevance": resolved_threshold,
            "top_results": top_results,
        },
    )


resolve_stock_beat = resolve_stock_request
resolve_stock = resolve_stock_request
"""Aliases for :func:`resolve_stock_request`."""


async def resolve_stock_requests(
    values: Any,
    **kwargs: Any,
) -> list[StockBeatResult]:
    """Resolve every stock request in *values*, preserving input order."""
    if isinstance(values, VisualRequest):
        requests = [values]
    elif isinstance(values, Mapping):
        requests = [_coerce_request(values)]
    elif isinstance(values, Sequence) and not isinstance(
        values, (str, bytes, bytearray)
    ):
        requests = [_coerce_request(item) for item in values]
    else:
        requests = resolve_visual_requests(values, strategy="stock")
    results: list[StockBeatResult] = []
    for position, request in enumerate(requests):
        result = await resolve_stock_request(request, **kwargs)
        if result.index != position and "index" not in request.model_fields_set:
            result = result.model_copy(update={"index": position})
        results.append(result)
    return results


resolve_stock_beats = resolve_stock_requests
resolve_stocks = resolve_stock_requests
"""Aliases for :func:`resolve_stock_requests`."""


def resolve_stock_request_sync(value: Any, **kwargs: Any) -> StockBeatResult:
    """Run :func:`resolve_stock_request` from synchronous code."""
    import asyncio

    return asyncio.run(resolve_stock_request(value, **kwargs))


def resolve_stock_requests_sync(values: Any, **kwargs: Any) -> list[StockBeatResult]:
    """Run :func:`resolve_stock_requests` from synchronous code."""
    import asyncio

    return asyncio.run(resolve_stock_requests(values, **kwargs))


class StockResolver:
    """Reusable stock resolver carrying the search, ranker and limits.

    A stage builds one of these, but callers that already have visual requests
    may use it directly without a run directory or a database.
    """

    def __init__(
        self,
        search: Any = None,
        *,
        provider: Any = None,
        router: Any = None,
        ranker: Any = None,
        scorer: Any = None,
        judge: Any = None,
        analyzer: Any = None,
        vision: Any = None,
        vision_provider: Any = None,
        llm: Any = None,
        helper: Any = None,
        model: str | None = None,
        config: Any = None,
        min_relevance: float | None = None,
        threshold: float | None = None,
        top_results: int = DEFAULT_TOP_RESULTS,
        per_page: int | None = None,
        queries: Any = None,
        fallback_queries: Any = None,
        query_limit: int | None = DEFAULT_QUERY_LIMIT,
        stock_kind: Any = None,
        orientation: Any = None,
    ) -> None:
        self._search = search if search is not None else provider
        self._router = router
        self._ranker = ranker
        self._scorer = scorer or judge or analyzer
        self._vision = vision or vision_provider or llm
        self._helper = helper
        self._model = model
        self._config = config
        self._min_relevance = min_relevance
        self._threshold = threshold
        self.top_results = top_results
        self.per_page = per_page
        self.queries = queries
        self.fallback_queries = fallback_queries
        self.query_limit = query_limit
        self.stock_kind = stock_kind
        self.orientation = orientation

    async def resolve(
        self,
        value: Any,
        *,
        run_id: str = "",
        metadata: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> StockBeatResult:
        """Resolve one request or beat with this resolver's settings."""
        values = {
            "search": self._search,
            "router": self._router,
            "ranker": self._ranker,
            "scorer": self._scorer,
            "vision": self._vision,
            "helper": self._helper,
            "model": self._model,
            "config": self._config,
            "min_relevance": self._min_relevance,
            "threshold": self._threshold,
            "top_results": self.top_results,
            "per_page": self.per_page,
            "queries": self.queries,
            "fallback_queries": self.fallback_queries,
            "query_limit": self.query_limit,
            "stock_kind": self.stock_kind,
            "orientation": self.orientation,
            "run_id": run_id,
            "metadata": metadata,
        }
        values.update(kwargs)
        return await resolve_stock_request(value, **values)

    async def resolve_beat(self, value: Any, **kwargs: Any) -> StockBeatResult:
        """Alias for :meth:`resolve`."""
        return await self.resolve(value, **kwargs)

    async def resolve_requests(
        self,
        values: Any,
        *,
        run_id: str = "",
        metadata: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[StockBeatResult]:
        """Resolve every request in *values* with this resolver's settings."""
        options = {
            "search": self._search,
            "router": self._router,
            "ranker": self._ranker,
            "scorer": self._scorer,
            "vision": self._vision,
            "helper": self._helper,
            "model": self._model,
            "config": self._config,
            "min_relevance": self._min_relevance,
            "threshold": self._threshold,
            "top_results": self.top_results,
            "per_page": self.per_page,
            "queries": self.queries,
            "fallback_queries": self.fallback_queries,
            "query_limit": self.query_limit,
            "stock_kind": self.stock_kind,
            "orientation": self.orientation,
            "run_id": run_id,
            "metadata": metadata,
        }
        options.update(kwargs)
        return await resolve_stock_requests(values, **options)

    def resolve_sync(self, value: Any, **kwargs: Any) -> StockBeatResult:
        """Run :meth:`resolve` from synchronous code."""
        import asyncio

        return asyncio.run(self.resolve(value, **kwargs))

    __call__ = resolve


StockStageResolver = StockResolver
StockBeatResolver = StockResolver
StockQueryResolver = StockResolver


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = canonical_json(value, indent=2)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


class StockStage(Stage):
    """Resolve every stock beat into a selected, relevance scored asset.

    ``requests`` may be a script, a beat sequence, one or more visual requests
    or mappings. A ``script``/``beats`` option is resolved with the stock
    strategy first. ``search``/``provider``/``router`` supplies the stock
    search, while ``ranker``/``scorer``/``vision`` scores its top results.
    ``queries`` and ``fallback_queries`` extend each beat's own query plan.
    """

    name = STOCK_STAGE_NAME
    artifacts: tuple[str, ...] = (DEFAULT_STOCK_FILENAME,)

    def __init__(
        self,
        name: str | None = None,
        *,
        options: Mapping[str, Any] | None = None,
        requests: Any = None,
        visuals: Any = None,
        visual_requests: Any = None,
        script: Any = None,
        beats: Any = None,
        search: Any = None,
        provider: Any = None,
        router: Any = None,
        ranker: Any = None,
        scorer: Any = None,
        judge: Any = None,
        analyzer: Any = None,
        vision: Any = None,
        vision_provider: Any = None,
        llm: Any = None,
        helper: Any = None,
        model: str | None = None,
        config: Any = None,
        min_relevance: float | None = None,
        threshold: float | None = None,
        top_results: int | None = None,
        per_page: int | None = None,
        queries: Any = None,
        fallback_queries: Any = None,
        query_limit: int | None = None,
        stock_kind: Any = None,
        kind: Any = None,
        orientation: Any = None,
        min_width: int | None = None,
        min_height: int | None = None,
        min_duration_seconds: float | None = None,
        max_duration_seconds: float | None = None,
        filename: str | None = None,
        manifest_filename: str | None = None,
        **kwargs: Any,
    ) -> None:
        merged = dict(options or {})
        supplied = {
            "requests": requests if requests is not None else visual_requests,
            "visuals": visuals,
            "script": script,
            "beats": beats,
            "search": search,
            "provider": provider,
            "router": router,
            "ranker": ranker,
            "scorer": scorer,
            "judge": judge,
            "analyzer": analyzer,
            "vision": vision,
            "vision_provider": vision_provider,
            "llm": llm,
            "helper": helper,
            "model": model,
            "config": config,
            "min_relevance": min_relevance,
            "threshold": threshold,
            "top_results": top_results,
            "per_page": per_page,
            "queries": queries,
            "fallback_queries": fallback_queries,
            "query_limit": query_limit,
            "stock_kind": stock_kind,
            "kind": kind,
            "orientation": orientation,
            "min_width": min_width,
            "min_height": min_height,
            "min_duration_seconds": min_duration_seconds,
            "max_duration_seconds": max_duration_seconds,
            "filename": filename,
            "manifest_filename": manifest_filename,
        }
        for key, value in supplied.items():
            if value is not None:
                merged[key] = value
        self.filename = _text(
            merged.pop("filename", None)
            or merged.pop("manifest_filename", None)
            or DEFAULT_STOCK_FILENAME
        )
        if not self.filename:
            self.filename = DEFAULT_STOCK_FILENAME
        self.artifacts = (self.filename,)

        self._requests_input = merged.pop("requests", None)
        if self._requests_input is None:
            self._requests_input = merged.pop("visuals", None)
        self._script = merged.pop("script", None)
        self._beats = merged.pop("beats", None)
        self._search = merged.pop("search", None)
        self._provider = merged.pop("provider", None)
        self._router = merged.pop("router", None)
        self._ranker = merged.pop("ranker", None)
        self._scorer = (
            merged.pop("scorer", None)
            or merged.pop("judge", None)
            or merged.pop("analyzer", None)
        )
        self._vision = (
            merged.pop("vision", None)
            or merged.pop("vision_provider", None)
            or merged.pop("llm", None)
        )
        self._helper = merged.pop("helper", None)
        self._model = merged.pop("model", None)
        self._config = merged.pop("config", None)
        self._min_relevance = merged.pop("min_relevance", None)
        self._threshold = merged.pop("threshold", None)
        self.top_results = int(merged.pop("top_results", DEFAULT_TOP_RESULTS))
        self.per_page = merged.pop("per_page", None)
        self.queries = merged.pop("queries", None)
        self.fallback_queries = merged.pop("fallback_queries", None)
        self.query_limit = merged.pop("query_limit", DEFAULT_QUERY_LIMIT)
        self.stock_kind = merged.pop("stock_kind", None)
        if self.stock_kind is None:
            self.stock_kind = merged.pop("kind", None)
        self.orientation = merged.pop("orientation", None)
        self.min_width = merged.pop("min_width", None)
        self.min_height = merged.pop("min_height", None)
        self.min_duration_seconds = merged.pop("min_duration_seconds", None)
        self.max_duration_seconds = merged.pop("max_duration_seconds", None)
        self.results: list[StockBeatResult] = []
        super().__init__(name=name, options=merged, **kwargs)

    def _requests(self) -> list[VisualRequest]:
        value = self._requests_input
        if value is None:
            value = self._script if self._script is not None else self._beats
        if value is None:
            return []
        if isinstance(value, VisualRequest):
            return [value]
        if isinstance(value, Mapping):
            return [_coerce_request(value)]
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return [_coerce_request(item) for item in value]
        return resolve_visual_requests(value, strategy="stock")

    async def run(self, context: StageContext) -> StageResult:
        """Resolve every stock beat and write the ``stock.json`` manifest."""
        requests = self._requests()
        stock_requests = [
            request for request in requests if request.visual_type.value == "stock"
        ]
        search = self._search if self._search is not None else self._provider
        if search is None:
            search = self._router if self._router is not None else context.router
        run_id = context.run.id if context.run is not None else ""
        results: list[StockBeatResult] = []
        for position, request in enumerate(stock_requests):
            result = await resolve_stock_request(
                request,
                search=search,
                ranker=self._ranker,
                scorer=self._scorer,
                vision=self._vision,
                helper=self._helper,
                model=self._model,
                config=self._config if self._config is not None else context.config,
                min_relevance=self._min_relevance,
                threshold=self._threshold,
                top_results=self.top_results,
                per_page=self.per_page,
                queries=self.queries,
                fallback_queries=self.fallback_queries,
                query_limit=self.query_limit,
                stock_kind=self.stock_kind,
                orientation=self.orientation,
                min_width=self.min_width,
                min_height=self.min_height,
                min_duration_seconds=self.min_duration_seconds,
                max_duration_seconds=self.max_duration_seconds,
                run_id=run_id,
                metadata={"position": position},
            )
            results.append(result)
        self.results = results

        resolved_count = sum(1 for result in results if result.resolved)
        unresolved_count = len(results) - resolved_count
        payload: dict[str, Any] = {
            "version": 1,
            "stage": self.name,
            "beats": [result.to_dict() for result in results],
            "metadata": {
                "queries": sum(len(result.queries) for result in results),
                "resolved": resolved_count,
                "unresolved": unresolved_count,
            },
        }
        relative = self.filename
        _write_json(context.run_dir / relative, payload)
        total_cost = sum(result.cost_usd for result in results)
        providers = [result.provider for result in results if result.provider]
        return StockStageResult(
            run_id=run_id,
            stage=self.name,
            status=StageState.SUCCEEDED,
            artifacts=[relative],
            cost_usd=total_cost,
            beats=results,
            metadata={
                "beats": len(results),
                "resolved": resolved_count,
                "unresolved": unresolved_count,
                "queries": sum(len(result.queries) for result in results),
                "providers": list(dict.fromkeys(providers)),
                "manifest": relative,
            },
        )


stock_stage = StockStage
"""Alias for :class:`StockStage`."""


def build_stock_stage(
    *,
    options: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> StockStage:
    """Build a :class:`StockStage` from recipe options and overrides."""
    return StockStage(options=options, **kwargs)


async def run_stock_stage(
    context: StageContext,
    *,
    stage: StockStage | None = None,
    **options: Any,
) -> StageResult:
    """Build a stage unless one is supplied, then run it asynchronously."""
    if stage is not None and options:
        raise ConfigError("pass either a stage or options, not both")
    active = stage if stage is not None else StockStage(**options)
    return await active.run(context)
