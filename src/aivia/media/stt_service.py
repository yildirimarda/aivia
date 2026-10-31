"""Run one speech to text request through the single paid path.

The service screens any context prompt, applies the remaining budget, asks the
router for a provider, records the per-minute cost and stores the transcript as
a text asset with a cost sidecar. Callers never choose a vendor or write a cost
record themselves.

The order matches the project invariant: screen -> budget -> router -> provider
-> store -> cost -> sidecar.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote, urlsplit

from sqlalchemy import Engine

from aivia.core.budget import BudgetGuard
from aivia.core.config.budget import BudgetConfig
from aivia.core.cost import CostTracker
from aivia.core.models.asset import Asset, AssetCostLink, AssetKind
from aivia.core.models.base import canonical_json
from aivia.core.models.cost import CostEvent
from aivia.core.models.run import ToolchainFingerprint
from aivia.core.pricing import ResolvedPrice
from aivia.core.router import Router, RouteSelection
from aivia.core.sidecar import write_cost_sidecar
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
from aivia.providers.stt.cost import STT_OPERATION, STTCostCalculator

__all__ = [
    "STT_OPERATION",
    "STTOutcome",
    "STTService",
    "STTServiceResult",
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
    ) -> None:
        if not isinstance(router, Router):
            raise TypeError("router must be a Router")
        if tracker is not None and not isinstance(tracker, CostTracker):
            raise TypeError("tracker must be a CostTracker")
        if engine is not None and not isinstance(engine, Engine):
            raise TypeError("engine must be a SQLAlchemy Engine")
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

        self._router = router
        self._tracker = tracker
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
    def last_outcome(self) -> STTOutcome | None:
        """The most recent outcome."""
        return self._last_outcome

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
        storage: StorageProvider | None = None,
        key: str | None = None,
        key_prefix: str | None = None,
        pinned: bool = True,
        estimated: bool = False,
    ) -> STTOutcome:
        """Transcribe *request* through the screened paid path."""
        if not isinstance(request, STTRequest):
            raise TypeError("request must be a STTRequest")
        if request.prompt:
            active_token = token if token is not None else screening_token
            self._require_screened(active_token, request.prompt)

        active_tracker = tracker if tracker is not None else self._tracker
        operation_name = self._operation if operation is None else operation.strip()
        active_storage = storage if storage is not None else self._storage
        active_prefix = (
            key_prefix.strip().strip("/")
            if key_prefix is not None
            else self._key_prefix
        )
        budget = remaining_budget
        if budget is None and self._budget is not None:
            budget = self._budget.remaining(run_id, operation=operation_name)

        async def run(provider: Provider) -> STTResponse:
            return await cast(STTProvider, provider).transcribe(request)

        response = await self._router.execute(
            CAPABILITY_STT,
            run,
            remaining_budget=budget,
            units=1.0,
            unit_kind="call",
        )
        if not isinstance(response, STTResponse):
            raise TypeError("speech to text provider must return a STTResponse")
        selection = self._router.last_selection
        if selection is None:
            raise RuntimeError("router completed a transcription without a selection")
        provider_name = response.provider or selection.name
        model_name = response.model or request.model or ""
        response = response.model_copy(
            update={"provider": provider_name, "model": model_name}
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
        asset = Asset(
            kind=AssetKind.TEXT,
            provider=provider_name,
            model=model_name,
            prompt=request.prompt or "",
            path=(stored.url or stored.key) if stored is not None else "",
            content_hash=_sha256(payload),
            run_id=run_id,
            pinned=pinned,
            cost_links=cost_links,
        )
        sidecars: tuple[Path, ...] = ()
        if self._record_sidecar:
            path = _file_path(asset.path) if asset.path else None
            if path is not None:
                sidecars = (
                    write_cost_sidecar(
                        path,
                        [event],
                        toolchain_fingerprint=self._toolchain,
                        run_id=run_id or None,
                    ),
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
        )
        self._last_outcome = outcome
        return outcome


def _minutes(response: STTResponse) -> float:
    duration = response.duration_seconds
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
