"""Run one sound effect request through the single paid path.

The service screens the prompt, applies the remaining budget, asks the router
for a provider, validates and stores every returned clip, records one cost
event and writes a cost sidecar next to each output. Callers never choose a
vendor or write a cost record themselves.

The order matches the project invariant: screen -> budget -> router -> provider
-> validate -> store -> cost -> sidecar.
"""

from __future__ import annotations

import hashlib
import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote, urlsplit

import httpx
from sqlalchemy import Engine

from aivia.core.budget import BudgetGuard
from aivia.core.config.budget import BudgetConfig
from aivia.core.cost import CostTracker
from aivia.core.media_probe import (
    MediaDownloader,
    MediaLimits,
    MediaProbe,
    validate_media,
)
from aivia.core.models.asset import Asset, AssetAnalysis, AssetCostLink, AssetKind
from aivia.core.models.cost import CostEvent
from aivia.core.models.run import ToolchainFingerprint
from aivia.core.pricing import PricingResolver, ResolvedPrice
from aivia.core.router import Router, RouteSelection
from aivia.core.sidecar import write_cost_sidecar
from aivia.media.accessibility import write_accessibility_sidecars
from aivia.media.screening import (
    DEFAULT_PROMPT_KIND,
    ScreeningToken,
    require_screened,
    screen_prompt,
)
from aivia.providers.base import Provider
from aivia.providers.protocols.sfx import (
    CAPABILITY_SFX,
    SFXProvider,
    SFXRequest,
    SFXResponse,
)
from aivia.providers.protocols.storage import (
    StorageObject,
    StorageProvider,
    StoragePutRequest,
)

__all__ = [
    "SFX_OPERATION",
    "SFX_UNIT",
    "SFXOutcome",
    "SFXService",
    "SFXServiceResult",
    "generate_sfx",
]

SFX_OPERATION = "sfx"
"""The cost event operation for one sound effect call."""

SFX_UNIT = "clip"
"""The cost unit used for one generated sound effect clip."""

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
class SFXOutcome:
    """The response together with its stored clips, price and sidecars."""

    request: SFXRequest
    response: SFXResponse
    assets: tuple[Asset, ...] = ()
    probes: tuple[MediaProbe, ...] = ()
    storage_objects: tuple[StorageObject, ...] = ()
    events: tuple[CostEvent, ...] = ()
    price: ResolvedPrice | None = None
    selection: RouteSelection | None = None
    sidecars: tuple[Path, ...] = ()
    accessibility_sidecars: tuple[Any, ...] = ()

    @property
    def accessibility(self) -> Any:
        """The first accessibility sidecar result, or ``None``."""
        return self.accessibility_sidecars[0] if self.accessibility_sidecars else None

    @property
    def accessibility_sidecar(self) -> Any:
        """Alias for :attr:`accessibility`."""
        return self.accessibility

    @property
    def accessibility_paths(self) -> tuple[Path, ...]:
        """Every written accessibility sidecar path."""
        return tuple(
            path
            for result in self.accessibility_sidecars
            for path in getattr(result, "paths", ())
        )

    @property
    def all_sidecars(self) -> tuple[Path, ...]:
        """Cost and accessibility sidecar paths together."""
        return (*self.sidecars, *self.accessibility_paths)

    @property
    def event(self) -> CostEvent | None:
        """The first recorded cost event, or ``None``."""
        return self.events[0] if self.events else None

    @property
    def clips(self) -> list[Any]:
        """Every generated clip in response order."""
        return list(self.response.clips)

    @property
    def asset(self) -> Asset | None:
        """The first stored asset, or ``None``."""
        return self.assets[0] if self.assets else None

    @property
    def outputs(self) -> tuple[Asset, ...]:
        """Alias for :attr:`assets`."""
        return self.assets

    @property
    def paths(self) -> tuple[str, ...]:
        """Every stored asset path in response order."""
        return tuple(asset.path for asset in self.assets)

    @property
    def urls(self) -> tuple[str, ...]:
        """Alias for :attr:`paths`."""
        return self.paths

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
    def cached(self) -> bool:
        """Sound effects are never served from the asset cache."""
        return False

    @property
    def amount_usd(self) -> float:
        """The recorded cost, or zero when no cost was recorded."""
        if self.events:
            return sum(event.amount_usd for event in self.events)
        if self.price is not None:
            return self.price.amount_usd
        return 0.0


SFXServiceResult = SFXOutcome
"""Alias for :class:`SFXOutcome`."""


class SFXService:
    """Execute sound effect generation through a :class:`~aivia.core.router.Router`."""

    def __init__(
        self,
        router: Router,
        *,
        tracker: CostTracker | None = None,
        engine: Engine | None = None,
        storage: StorageProvider | None = None,
        budget: BudgetLike = None,
        safety: Any = None,
        limits: MediaLimits | None = None,
        downloader: MediaDownloader | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        pricing_overrides: Mapping[str, float] | None = None,
        pricing_version: str = "",
        operation: str = SFX_OPERATION,
        toolchain: ToolchainLike = None,
        key_prefix: str | None = None,
        record_cost: bool = True,
        record_sidecar: bool = True,
        accessibility: Any = None,
        accessibility_config: Any = None,
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
        self._limits = limits or MediaLimits()
        self._downloader = downloader
        self._transport = transport
        self._operation = operation.strip()
        self._toolchain = toolchain
        self._key_prefix = key_prefix
        self._record_cost = bool(record_cost)
        self._record_sidecar = bool(record_sidecar)
        self._accessibility = (
            accessibility if accessibility is not None else accessibility_config
        )
        self._resolver = PricingResolver(
            pricing_overrides,
            pricing_version=pricing_version,
        )
        self._last_outcome: SFXOutcome | None = None

    def _write_accessibility_sidecars(
        self,
        assets: list[Asset],
        request: SFXRequest,
    ) -> tuple[Any, ...]:
        if self._accessibility is None:
            return ()
        metadata = request.metadata or {}
        written: list[Any] = []
        for asset in assets:
            if not asset.path:
                continue
            path = _file_path(asset.path)
            if path is None:
                continue
            result = write_accessibility_sidecars(
                path,
                self._accessibility,
                alt_text=(
                    metadata.get("alt_text")
                    or metadata.get("alt")
                    or request.prompt
                ),
                subtitle=metadata.get("subtitle"),
                words=metadata.get("words"),
                transcript=metadata.get("transcript"),
                text=metadata.get("text") or request.prompt,
                duration_seconds=asset.analysis.duration_seconds,
            )
            if result:
                written.append(result)
        return tuple(written)

    @property
    def router(self) -> Router:
        """The router every call goes through."""
        return self._router

    @property
    def last_outcome(self) -> SFXOutcome | None:
        """The most recent outcome."""
        return self._last_outcome

    def screen(self, prompt: str) -> ScreeningToken:
        """Screen *prompt* with this service's configuration and return a token."""
        return screen_prompt(prompt, self._safety)

    def _require_screened(self, token: Any, prompt: str) -> ScreeningToken:
        return require_screened(
            token,
            prompt,
            kind=DEFAULT_PROMPT_KIND,
            config=self._safety,
        )

    def _downloader_for(self) -> MediaDownloader:
        if self._downloader is None:
            self._downloader = MediaDownloader(
                transport=self._transport,
                limits=self._limits,
            )
        return self._downloader

    async def generate(
        self,
        request: SFXRequest,
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
    ) -> SFXOutcome:
        """Generate *request* through the screened paid path."""
        if not isinstance(request, SFXRequest):
            raise TypeError("request must be a SFXRequest")
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

        base_seed = request.seed if request.seed is not None else secrets.randbits(32)
        provider_request = (
            request
            if request.seed is not None
            else request.model_copy(update={"seed": base_seed})
        )

        async def run(provider: Provider) -> SFXResponse:
            return await cast(SFXProvider, provider).generate(provider_request)

        response = await self._router.execute(
            CAPABILITY_SFX,
            run,
            remaining_budget=budget,
            units=float(request.count),
            unit_kind=SFX_UNIT,
        )
        if not isinstance(response, SFXResponse):
            raise TypeError("sound effect provider must return a SFXResponse")
        selection = self._router.last_selection
        if selection is None:
            raise RuntimeError(
                "router completed a sound effect call without a selection"
            )
        provider_name = response.provider or selection.name
        model_name = response.model or request.model or ""
        response = response.model_copy(
            update={"provider": provider_name, "model": model_name}
        )

        materialised: list[tuple[MediaProbe, StorageObject | None]] = []
        for index, clip in enumerate(response.clips):
            if clip.data is not None:
                payload = clip.data
                probe = validate_media(
                    payload,
                    limits=self._limits,
                    expected_format=clip.format.value,
                )
            else:
                if clip.url is None:
                    raise ValueError("sound effect clip has neither data nor url")
                probe = await self._downloader_for().download(
                    clip.url,
                    limits=self._limits,
                    expected_format=clip.format.value,
                )
                payload = probe.data
            stored: StorageObject | None = None
            if active_storage is not None:
                storage_key = key.strip() if isinstance(key, str) else None
                if storage_key == "":
                    raise ValueError("storage key must be non-empty")
                if storage_key is None and active_prefix:
                    digest = _sha256(payload)
                    storage_key = (
                        f"{active_prefix}/{digest[:2]}/{digest}.{probe.extension}"
                    )
                stored = await active_storage.put(
                    StoragePutRequest(
                        data=payload,
                        key=storage_key,
                        content_type=probe.mime_type,
                        metadata={
                            "kind": "sfx",
                            "format": probe.format.value,
                            "provider": provider_name,
                            "model": model_name,
                            "run_id": run_id,
                            "index": index,
                        },
                    )
                )
            materialised.append((probe, stored))

        price = self._resolver.resolve(
            selection.provider,
            operation_name,
            float(response.count),
            unit_kind=SFX_UNIT,
            provider_name=selection.name,
        )
        details: dict[str, Any] = {
            "clip_count": response.count,
            "price_source": price.source.value,
        }
        if response.request_id:
            details["request_id"] = response.request_id
        if metadata is not None:
            details.update(metadata)
        event = CostEvent(
            provider=price.provider,
            model=model_name,
            operation=operation_name,
            units=price.units,
            unit_kind=price.unit_kind,
            amount_usd=price.amount_usd,
            estimated=estimated,
            run_id=run_id,
            pricing_version=price.pricing_version,
            metadata=details,
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
        assets: list[Asset] = []
        for index, (probe, stored) in enumerate(materialised):
            assets.append(
                Asset(
                    kind=AssetKind.SFX,
                    provider=provider_name,
                    model=model_name,
                    prompt=request.prompt,
                    path=(stored.url or stored.key) if stored is not None else "",
                    content_hash=probe.sha256,
                    run_id=run_id,
                    pinned=pinned,
                    seed=(
                        base_seed + index
                        if request.seed is None
                        else request.seed + index
                    ),
                    analysis=AssetAnalysis(duration_seconds=probe.duration_seconds),
                    cost_links=cost_links,
                )
            )

        sidecars: tuple[Path, ...] = ()
        if self._record_sidecar:
            written: list[Path] = []
            for asset in assets:
                path = _file_path(asset.path) if asset.path else None
                if path is not None:
                    written.append(
                        write_cost_sidecar(
                            path,
                            [event],
                            toolchain_fingerprint=self._toolchain,
                            run_id=run_id or None,
                        )
                    )
            sidecars = tuple(written)

        accessibility = self._write_accessibility_sidecars(assets, request)
        outcome = SFXOutcome(
            request=request,
            response=response,
            assets=tuple(assets),
            probes=tuple(probe for probe, _ in materialised),
            storage_objects=tuple(
                stored for _, stored in materialised if stored is not None
            ),
            events=(event,),
            price=price,
            selection=selection,
            sidecars=sidecars,
            accessibility_sidecars=accessibility,
        )
        self._last_outcome = outcome
        return outcome


async def generate_sfx(
    request: SFXRequest,
    router: Router,
    **values: Any,
) -> SFXOutcome:
    """Execute one sound effect request without constructing a long-lived service."""
    service = SFXService(router)
    return await service.generate(request, **values)
