"""Run one speech to text request through the single paid path.

The service screens any context prompt, answers an identical audio file from
the asset cache when possible, applies the remaining budget in audio minutes,
asks the router for a provider, records the per-minute cost and stores the
transcript as a text asset with a cost sidecar. Callers never choose a vendor
or write a cost record themselves.

The order matches the project invariant: screen -> cache -> budget -> router ->
provider -> store -> cost -> sidecar. A cache hit short-circuits the paid path
but still records a zero-amount event, so reused transcripts are visible in the
ledger without counting as spend.

The cache key is derived from the audio bytes (or the audio URL) plus the
transcription options, so the same recording is never paid for twice even when
the caller passes a different filename. The budget and the cost both use the
audio duration in minutes; a duration hint in the request metadata avoids a
probe, otherwise inline audio is measured with ffprobe before the call.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote, urlsplit

from sqlalchemy import Engine

from aivia.core.asset_cache import AssetCache, AssetCacheKey
from aivia.core.budget import BudgetGuard
from aivia.core.config.budget import BudgetConfig
from aivia.core.cost import CostTracker
from aivia.core.errors import ValidationError
from aivia.core.models.asset import Asset, AssetCostLink, AssetKind
from aivia.core.models.base import canonical_json
from aivia.core.models.cost import CostEvent
from aivia.core.models.run import ToolchainFingerprint
from aivia.core.pricing import ResolvedPrice
from aivia.core.router import Router, RouteSelection
from aivia.core.sidecar import (
    SIDECAR_SUFFIX,
    build_cost_sidecar,
    write_cost_sidecar,
)
from aivia.media.audio_utils import measure_audio_duration
from aivia.media.screening import (
    DEFAULT_PROMPT_KIND,
    ScreeningToken,
    require_screened,
    screen_prompt,
)
from aivia.providers.base import Provider
from aivia.providers.protocols.storage import (
    StorageObject,
    StorageProvider,
    StoragePutRequest,
)
from aivia.providers.protocols.stt import (
    CAPABILITY_STT,
    STTProvider,
    STTRequest,
    STTResponse,
)
from aivia.providers.stt.common import response_duration_seconds
from aivia.providers.stt.cost import (
    MINUTE_UNIT,
    STT_OPERATION,
    STTCostCalculator,
)

__all__ = [
    "MINUTE_UNIT",
    "STT_OPERATION",
    "STTOutcome",
    "STTService",
    "STTServiceResult",
    "stt_cache_key",
    "transcribe",
]

ToolchainLike = ToolchainFingerprint | Mapping[str, Any] | str | None
BudgetLike = BudgetGuard | BudgetConfig | Mapping[str, Any] | None


def _file_path(url: str) -> Path | None:
    parsed = urlsplit(url)
    if parsed.scheme.lower() != "file":
        return None
    return Path(unquote(parsed.path))


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _audio_hash(request: STTRequest, audio_hash: str | None = None) -> str:
    if audio_hash is not None:
        digest = audio_hash.strip().lower()
        if not digest:
            raise ValueError("audio_hash must be a non-empty string")
        return digest
    if request.audio is not None:
        return _sha256(request.audio)
    if request.audio_url is not None:
        return _sha256(request.audio_url.encode("utf-8"))
    raise ValueError("speech to text request has no audio source")


def stt_cache_key(
    request: STTRequest,
    *,
    audio_hash: str | None = None,
    model: str | None = None,
) -> AssetCacheKey:
    """Return the cache key for *request* and its transcription options.

    The provider is deliberately left empty so identical audio is answered
    before the router has selected a provider. The stored asset still records
    who actually served it. The filename is excluded because the audio bytes
    identify the recording; changing only the display name must not cost money.
    """
    if not isinstance(request, STTRequest):
        raise TypeError("request must be a STTRequest")
    digest = _audio_hash(request, audio_hash)
    parameters: dict[str, Any] = {
        "source": "audio" if request.audio is not None else "audio_url",
        "model": request.model,
        "language": request.language,
        "prompt": request.prompt,
        "temperature": request.temperature,
        "word_timestamps": request.word_timestamps,
        "diarize": request.diarize,
    }
    return AssetCacheKey(
        provider="",
        model=(request.model if model is None else model) or "",
        prompt=request.prompt or "",
        parameters=parameters,
        reference_hashes=(f"audio:{digest}",),
    )


def _duration_hint(
    request: STTRequest,
    duration_seconds: float | None = None,
) -> float | None:
    candidate: Any = duration_seconds
    if candidate is None:
        candidate = request.metadata.get("duration_seconds")
    if candidate is None:
        candidate = request.metadata.get("duration")
    if isinstance(candidate, bool) or not isinstance(candidate, (int, float)):
        return None
    value = float(candidate)
    if not math.isfinite(value) or value < 0:
        return None
    return value


@dataclass(frozen=True, slots=True)
class STTOutcome:
    """The transcript together with its stored asset, price and sidecar."""

    request: STTRequest
    response: STTResponse
    assets: tuple[Asset, ...] = ()
    storage_objects: tuple[StorageObject, ...] = ()
    events: tuple[CostEvent, ...] = ()
    price: ResolvedPrice | None = None
    selection: RouteSelection | None = None
    sidecars: tuple[Path, ...] = ()
    cache_hit: bool = False

    @property
    def cached(self) -> bool:
        """Whether this outcome came from the asset cache."""
        return self.cache_hit

    @property
    def event(self) -> CostEvent | None:
        """The first recorded cost event, or ``None``."""
        return self.events[0] if self.events else None

    @property
    def text(self) -> str:
        """The transcript text."""
        return self.response.text

    @property
    def words(self) -> list[Any]:
        """Every timed word in the transcript."""
        return self.response.all_words

    @property
    def asset(self) -> Asset | None:
        """The stored transcript asset, or ``None``."""
        return self.assets[0] if self.assets else None

    @property
    def paths(self) -> tuple[str, ...]:
        """Every stored asset path in response order."""
        return tuple(asset.path for asset in self.assets)

    @property
    def sidecar(self) -> Path | None:
        """The first written sidecar, or ``None``."""
        return self.sidecars[0] if self.sidecars else None

    @property
    def provider(self) -> str:
        """The provider that served the call."""
        return self.response.provider

    @property
    def model(self) -> str:
        """The model that served the call."""
        return self.response.model

    @property
    def amount_usd(self) -> float:
        """The recorded cost, or zero when no cost was recorded."""
        if self.events:
            return sum(event.amount_usd for event in self.events)
        if self.price is not None:
            return self.price.amount_usd
        return 0.0


STTServiceResult = STTOutcome
"""Alias for :class:`STTOutcome`."""


class STTService:
    """Execute speech to text through a :class:`~aivia.core.router.Router`."""

    def __init__(
        self,
        router: Router,
        *,
        tracker: CostTracker | None = None,
        engine: Engine | None = None,
        cache: AssetCache | None = None,
        storage: StorageProvider | None = None,
        budget: BudgetLike = None,
        safety: Any = None,
        pricing_overrides: Mapping[str, float] | None = None,
        pricing_version: str = "",
        operation: str = STT_OPERATION,
        toolchain: ToolchainLike = None,
        key_prefix: str | None = None,
        record_cost: bool = True,
        record_sidecar: bool = True,
        no_cache: bool = False,
        ffprobe_executable: str = "ffprobe",
        ffprobe_runner: Any = None,
        ffprobe_timeout_seconds: float | None = None,
    ) -> None:
        if not isinstance(router, Router):
            raise TypeError("router must be a Router")
        if tracker is not None and not isinstance(tracker, CostTracker):
            raise TypeError("tracker must be a CostTracker")
        if engine is not None and not isinstance(engine, Engine):
            raise TypeError("engine must be a SQLAlchemy Engine")
        if cache is not None and not isinstance(cache, AssetCache):
            raise TypeError("cache must be an AssetCache")
        if budget is not None and not isinstance(
            budget, (BudgetGuard, BudgetConfig, Mapping)
        ):
            raise TypeError("budget must be a BudgetGuard, BudgetConfig or mapping")
        if key_prefix is not None:
            if not isinstance(key_prefix, str) or not key_prefix.strip():
                raise ValueError("key_prefix must be a non-empty string")
            key_prefix = key_prefix.strip().strip("/")
        if not isinstance(operation, str) or not operation.strip():
            raise ValueError("operation must be a non-empty string")

        selected_cache = cache
        if selected_cache is None and engine is not None:
            selected_cache = AssetCache(engine, no_cache=no_cache)
        elif (
            selected_cache is None
            and tracker is not None
            and tracker.repository.engine is not None
        ):
            selected_cache = AssetCache(
                tracker.repository.engine,
                cost=tracker,
                no_cache=no_cache,
            )
        if selected_cache is not None and no_cache:
            selected_cache.no_cache = True
        if tracker is None and selected_cache is not None:
            tracker = selected_cache.cost

        self._router = router
        self._tracker = tracker
        self._cache = selected_cache
        self._storage = storage
        self._budget = (
            BudgetGuard(budget, tracker)
            if isinstance(budget, (BudgetConfig, Mapping))
            else budget
        )
        self._safety = safety
        self._operation = operation.strip()
        self._toolchain = toolchain
        self._key_prefix = key_prefix
        self._record_cost = bool(record_cost)
        self._record_sidecar = bool(record_sidecar)
        self._no_cache = bool(no_cache)
        self._ffprobe_executable = ffprobe_executable
        self._ffprobe_runner = ffprobe_runner
        self._ffprobe_timeout = ffprobe_timeout_seconds
        self._payloads: dict[str, bytes] = {}
        self._calculator = STTCostCalculator(
            pricing_overrides,
            pricing_version=pricing_version,
        )
        self._last_outcome: STTOutcome | None = None

    @property
    def router(self) -> Router:
        """The router every call goes through."""
        return self._router

    @property
    def cache(self) -> AssetCache | None:
        """The asset cache, when one was configured."""
        return self._cache

    @property
    def last_outcome(self) -> STTOutcome | None:
        """The most recent outcome."""
        return self._last_outcome

    def cache_key(
        self,
        request: STTRequest,
        *,
        audio_hash: str | None = None,
    ) -> AssetCacheKey:
        """Return the cache key for one speech to text request."""
        if not isinstance(request, STTRequest):
            raise TypeError("request must be a STTRequest")
        return stt_cache_key(request, audio_hash=audio_hash)

    def screen(self, text: str) -> ScreeningToken:
        """Screen a context prompt and return a token."""
        return screen_prompt(text, self._safety)

    def _require_screened(self, token: Any, text: str) -> ScreeningToken:
        return require_screened(
            token,
            text,
            kind=DEFAULT_PROMPT_KIND,
            config=self._safety,
        )

    async def generate(
        self,
        request: STTRequest,
        *,
        token: Any = None,
        screening_token: Any = None,
        run_id: str = "",
        tracker: CostTracker | None = None,
        operation: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        remaining_budget: float | None = None,
        duration_seconds: float | None = None,
        audio_hash: str | None = None,
        storage: StorageProvider | None = None,
        key: str | None = None,
        key_prefix: str | None = None,
        pinned: bool = True,
        estimated: bool = False,
        no_cache: bool | None = None,
    ) -> STTOutcome:
        """Transcribe *request* through the screened paid path."""
        if not isinstance(request, STTRequest):
            raise TypeError("request must be a STTRequest")
        if request.prompt:
            active_token = token if token is not None else screening_token
            self._require_screened(active_token, request.prompt)

        active_tracker = tracker if tracker is not None else self._tracker
        operation_name = self._operation if operation is None else operation.strip()
        if not operation_name:
            raise ValueError("operation must be a non-empty string")
        active_storage = storage if storage is not None else self._storage
        active_prefix = (
            key_prefix.strip().strip("/")
            if key_prefix is not None
            else self._key_prefix
        )
        if key_prefix is not None and not active_prefix:
            raise ValueError("key_prefix must be a non-empty string")
        use_cache = (
            self._cache is not None and not self._no_cache and not bool(no_cache)
        )

        if use_cache:
            cached = await self._cached_outcome(
                request,
                run_id=run_id,
                operation=operation_name,
                storage=active_storage,
                write_sidecar=self._record_sidecar,
                audio_hash=audio_hash,
            )
            if cached is not None:
                self._last_outcome = cached
                return cached

        budget = remaining_budget
        if budget is None and self._budget is not None:
            budget = self._budget.remaining(run_id, operation=operation_name)
        estimate_seconds = _duration_hint(request, duration_seconds)
        if estimate_seconds is None and budget is not None:
            estimate_seconds = self._probe_duration(request)
        budget_minutes = max(0.0, estimate_seconds or 0.0) / 60.0

        async def run(provider: Provider) -> STTResponse:
            return await cast(STTProvider, provider).transcribe(request)

        response = await self._router.execute(
            CAPABILITY_STT,
            run,
            remaining_budget=budget,
            units=budget_minutes,
            unit_kind=MINUTE_UNIT,
        )
        if not isinstance(response, STTResponse):
            raise TypeError("speech to text provider must return a STTResponse")
        selection = self._router.last_selection
        if selection is None:
            raise RuntimeError("router completed a transcription without a selection")
        provider_name = response.provider or selection.name
        model_name = response.model or request.model or ""
        duration = response_duration_seconds(response)
        response = response.model_copy(
            update={
                "provider": provider_name,
                "model": model_name,
                "duration_seconds": (
                    duration if duration is not None else response.duration_seconds
                ),
            }
        )

        payload = (
            canonical_json(
                {
                    "text": response.text,
                    "language": response.language,
                    "duration_seconds": response.duration_seconds,
                    "segments": [
                        segment.model_dump(mode="json") for segment in response.segments
                    ],
                    "words": [word.model_dump(mode="json") for word in response.words],
                    "model": response.model,
                    "provider": provider_name,
                },
                indent=2,
            )
            + "\n"
        ).encode("utf-8")
        stored: StorageObject | None = None
        if active_storage is not None:
            storage_key = key.strip() if isinstance(key, str) else None
            if storage_key == "":
                raise ValueError("storage key must be non-empty")
            if storage_key is None and active_prefix:
                digest = _sha256(payload)
                storage_key = f"{active_prefix}/{digest[:2]}/{digest}.json"
            stored = await active_storage.put(
                StoragePutRequest(
                    data=payload,
                    key=storage_key,
                    content_type="application/json",
                    metadata={
                        "kind": "transcript",
                        "provider": provider_name,
                        "model": model_name,
                        "run_id": run_id,
                    },
                )
            )

        price = self._calculator.resolve(
            selection.provider,
            _minutes(response),
            provider_name=selection.name,
            operation=operation_name,
        )
        event = self._calculator.cost_event(
            selection.provider,
            response=response,
            provider_name=selection.name,
            run_id=run_id,
            operation=operation_name,
            estimated=estimated,
            metadata=metadata,
            price=price,
        )
        recorded: CostEvent | None = None
        if self._record_cost and active_tracker is not None:
            recorded = (
                active_tracker.record_estimate(event)
                if estimated
                else active_tracker.record_actual(event)
            )
            event = recorded

        cost_links = (
            [
                AssetCostLink(
                    cost_event_id=event.id,
                    amount_usd=event.amount_usd,
                    provider=event.provider,
                    model=event.model,
                    operation=event.operation,
                )
            ]
            if recorded is not None
            else []
        )
        payload_digest = _sha256(payload)
        self._payloads[payload_digest] = payload
        asset = Asset(
            kind=AssetKind.TEXT,
            provider=provider_name,
            model=model_name,
            prompt=request.prompt or "",
            path=(stored.url or stored.key) if stored is not None else "",
            content_hash=payload_digest,
            run_id=run_id,
            pinned=pinned,
            cost_links=cost_links,
        )
        if use_cache and self._cache is not None:
            asset = self._cache.store(
                asset,
                key=self.cache_key(request, audio_hash=audio_hash),
            )
            if asset.content_hash:
                self._payloads.setdefault(asset.content_hash, payload)
        sidecars: tuple[Path, ...] = ()
        if self._record_sidecar:
            sidecars = await self._write_sidecars(
                asset,
                event,
                run_id=run_id,
                storage=active_storage,
                stored=stored,
            )

        outcome = STTOutcome(
            request=request,
            response=response,
            assets=(asset,),
            storage_objects=(stored,) if stored is not None else (),
            events=(event,),
            price=price,
            selection=selection,
            sidecars=sidecars,
            cache_hit=False,
        )
        self._last_outcome = outcome
        return outcome

    def _probe_duration(self, request: STTRequest) -> float | None:
        if request.audio is None:
            return None
        timeout = self._ffprobe_timeout if self._ffprobe_timeout is not None else 10.0
        try:
            return measure_audio_duration(
                request.audio,
                ffprobe_executable=self._ffprobe_executable,
                ffprobe_runner=self._ffprobe_runner,
                timeout_seconds=timeout,
            )
        except (ValidationError, ValueError, TypeError):
            return None

    async def _cached_outcome(
        self,
        request: STTRequest,
        *,
        run_id: str,
        operation: str,
        storage: StorageProvider | None,
        write_sidecar: bool,
        audio_hash: str | None,
    ) -> STTOutcome | None:
        cache = self._cache
        if cache is None:
            return None
        key = self.cache_key(request, audio_hash=audio_hash)
        entry = cache.entry(key)
        if entry is None:
            return None
        asset = cache.assets.get(entry.asset_id)
        if asset is None or asset.kind is not AssetKind.TEXT:
            return None
        payload = await self._cached_payload(asset, storage)
        if payload is None:
            return None
        try:
            data = json.loads(payload)
        except (TypeError, ValueError):
            return None
        if not isinstance(data, Mapping):
            return None
        try:
            response = STTResponse.model_validate(data)
        except ValueError:
            return None
        metadata = dict(response.metadata)
        metadata["cache"] = "hit"
        response = response.model_copy(update={"metadata": metadata})
        event = cache.record_hit(key, asset, run_id=run_id, operation=operation)
        sidecars: tuple[Path, ...] = ()
        if write_sidecar:
            sidecars = await self._write_sidecars(
                asset,
                event,
                run_id=run_id,
                storage=storage,
                stored=None,
            )
        return STTOutcome(
            request=request,
            response=response,
            assets=(asset,),
            events=(event,),
            sidecars=sidecars,
            cache_hit=True,
        )

    async def _cached_payload(
        self,
        asset: Asset,
        storage: StorageProvider | None,
    ) -> bytes | None:
        if asset.content_hash:
            cached = self._payloads.get(asset.content_hash)
            if cached is not None:
                return cached
        if asset.path:
            path = _file_path(asset.path)
            if path is not None and path.is_file():
                return path.read_bytes()
            if storage is not None:
                try:
                    return await storage.get(asset.path)
                except KeyError:
                    pass
        if storage is not None and asset.content_hash:
            digest = asset.content_hash
            try:
                return await storage.get(f"objects/{digest[:2]}/{digest}")
            except KeyError:
                return None
        return None

    async def _write_sidecars(
        self,
        asset: Asset,
        event: CostEvent,
        *,
        run_id: str,
        storage: StorageProvider | None,
        stored: StorageObject | None,
    ) -> tuple[Path, ...]:
        path = _file_path(asset.path) if asset.path else None
        if path is not None:
            return (
                write_cost_sidecar(
                    path,
                    [event],
                    toolchain_fingerprint=self._toolchain,
                    run_id=run_id or None,
                ),
            )
        if storage is None or stored is None:
            return ()
        sidecar = build_cost_sidecar(
            stored.url or stored.key,
            [event],
            toolchain_fingerprint=self._toolchain,
            run_id=run_id or None,
        )
        payload = (canonical_json(sidecar.to_dict(), indent=2) + "\n").encode("utf-8")
        await storage.put(
            StoragePutRequest(
                data=payload,
                key=f"{stored.key}{SIDECAR_SUFFIX}",
                content_type="application/json",
                overwrite=True,
            )
        )
        return ()


def _minutes(response: STTResponse) -> float:
    duration = response_duration_seconds(response)
    if duration is None:
        return 0.0
    return float(duration) / 60.0


async def transcribe(
    request: STTRequest,
    router: Router,
    **values: Any,
) -> STTOutcome:
    """Execute one transcription without constructing a long-lived service."""
    service = STTService(router)
    return await service.generate(request, **values)
