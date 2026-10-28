"""Cost sidecars written next to every generated output.

A sidecar is the per-output reproducibility record for the money a run spent.
It sits beside the media file it describes, so an output can be audited without
loading the cost ledger: the totals, every cost event that paid for it, the
providers and models involved, the pricing version and the renderer toolchain
fingerprint are all in one JSON document.

The sidecar only describes recorded events. It never resolves a price, calls a
provider or enforces a budget; those responsibilities belong to the pricing
resolver, the media service and the budget guard. Events are copied through a
small allowlist, which keeps prompts, metadata and credentials out of the file.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aivia.core.models.base import canonical_json, utcnow
from aivia.core.models.cost import CostEvent
from aivia.core.models.run import ToolchainFingerprint

__all__ = [
    "SIDECAR_SUFFIX",
    "CostSidecar",
    "SidecarCostItem",
    "SidecarTotals",
    "build_cost_sidecar",
    "sidecar_path",
    "write_cost_sidecar",
    "write_sidecar",
]

SIDECAR_SUFFIX = ".cost.json"
"""The suffix appended to an output path, for example ``clip.mp4.cost.json``."""


def _as_utc(value: datetime | None) -> datetime:
    if value is None:
        return utcnow()
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _coerce_events(
    events: Iterable[CostEvent | Mapping[str, Any]],
) -> list[CostEvent]:
    coerced: list[CostEvent] = []
    for event in events:
        if isinstance(event, CostEvent):
            coerced.append(event)
        elif isinstance(event, Mapping):
            coerced.append(CostEvent.model_validate(event))
        else:
            raise TypeError("a sidecar event must be a CostEvent or a mapping")
    return coerced


@dataclass(frozen=True, slots=True)
class SidecarCostItem:
    """One cost event as it appears in a sidecar, without its free-form metadata.

    ``metadata`` is deliberately absent: it can hold prompt text or provider
    diagnostics that must not be copied next to a published output. Everything
    needed to explain the amount remains.
    """

    cost_event_id: str
    provider: str
    model: str
    operation: str
    units: float
    unit_kind: str
    amount_usd: float
    estimated: bool
    pricing_version: str

    @classmethod
    def from_event(cls, event: CostEvent) -> SidecarCostItem:
        """Copy the auditable fields of *event* into an item."""
        return cls(
            cost_event_id=event.id,
            provider=event.provider,
            model=event.model,
            operation=event.operation,
            units=event.units,
            unit_kind=event.unit_kind,
            amount_usd=event.amount_usd,
            estimated=event.estimated,
            pricing_version=event.pricing_version,
        )

    @property
    def id(self) -> str:
        """Alias for :attr:`cost_event_id`."""
        return self.cost_event_id

    @property
    def amount(self) -> float:
        """Alias for :attr:`amount_usd`."""
        return self.amount_usd

    @property
    def is_estimated(self) -> bool:
        """Whether this item is a pre-call estimate."""
        return self.estimated

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the item."""
        return {
            "cost_event_id": self.cost_event_id,
            "provider": self.provider,
            "model": self.model,
            "operation": self.operation,
            "units": self.units,
            "unit_kind": self.unit_kind,
            "amount_usd": self.amount_usd,
            "estimated": self.estimated,
            "pricing_version": self.pricing_version,
        }


@dataclass(frozen=True, slots=True)
class SidecarTotals:
    """The summed cost of every item in a sidecar."""

    total_usd: float
    estimated_usd: float
    actual_usd: float
    units: float
    count: int

    @property
    def total(self) -> float:
        """Alias for :attr:`total_usd`."""
        return self.total_usd

    @property
    def estimated(self) -> float:
        """Alias for :attr:`estimated_usd`."""
        return self.estimated_usd

    @property
    def actual(self) -> float:
        """Alias for :attr:`actual_usd`."""
        return self.actual_usd

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the totals."""
        return {
            "total_usd": self.total_usd,
            "estimated_usd": self.estimated_usd,
            "actual_usd": self.actual_usd,
            "units": self.units,
            "count": self.count,
        }


@dataclass(frozen=True, slots=True)
class CostSidecar:
    """The cost record written next to one output file.

    ``pricing_version`` is the single version that priced every item, or an
    empty string when the items used more than one. ``pricing_versions`` always
    lists every distinct version so a mixed ledger stays auditable.
    """

    output: str
    run_id: str
    totals: SidecarTotals
    items: tuple[SidecarCostItem, ...]
    providers: tuple[str, ...]
    models: tuple[str, ...]
    pricing_version: str
    pricing_versions: tuple[str, ...]
    toolchain_fingerprint: ToolchainFingerprint
    generated_at: datetime

    @property
    def total_usd(self) -> float:
        """Alias for :attr:`SidecarTotals.total_usd`."""
        return self.totals.total_usd

    @property
    def total(self) -> float:
        """Alias for :attr:`total_usd`."""
        return self.total_usd

    @property
    def estimated_usd(self) -> float:
        """Alias for :attr:`SidecarTotals.estimated_usd`."""
        return self.totals.estimated_usd

    @property
    def actual_usd(self) -> float:
        """Alias for :attr:`SidecarTotals.actual_usd`."""
        return self.totals.actual_usd

    @property
    def toolchain_digest(self) -> str:
        """The digest of the recorded toolchain fingerprint."""
        return self.toolchain_fingerprint.digest

    def toolchain_payload(self) -> dict[str, object]:
        """Return the toolchain components and their stable digest."""
        fingerprint = self.toolchain_fingerprint
        return {
            "renderer_version": fingerprint.renderer_version,
            "ffmpeg_version": fingerprint.ffmpeg_version,
            "font_set": fingerprint.font_set,
            "identity": fingerprint.identity,
            "digest": fingerprint.digest,
            "is_known": fingerprint.is_known,
        }

    def to_dict(self) -> dict[str, object]:
        """Return the canonical JSON payload written to disk."""
        return {
            "output": self.output,
            "run_id": self.run_id,
            "generated_at": self.generated_at.isoformat(),
            "totals": self.totals.to_dict(),
            "items": [item.to_dict() for item in self.items],
            "providers": list(self.providers),
            "models": list(self.models),
            "pricing_version": self.pricing_version,
            "pricing_versions": list(self.pricing_versions),
            "toolchain_fingerprint": self.toolchain_payload(),
        }


def sidecar_path(output: str | Path) -> Path:
    """Return the sidecar path that sits next to *output*.

    The full file name is kept and the suffix appended, so ``image.png`` and
    ``image.jpg`` never share a sidecar.
    """
    target = Path(output).expanduser()
    return target.with_name(target.name + SIDECAR_SUFFIX)


def _common_run_id(events: list[CostEvent]) -> str:
    run_ids = {event.run_id.strip() for event in events if event.run_id.strip()}
    return next(iter(run_ids)) if len(run_ids) == 1 else ""


def _pricing_versions(events: list[CostEvent]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                event.pricing_version.strip()
                for event in events
                if event.pricing_version.strip()
            }
        )
    )


def _resolve_pricing_version(
    versions: tuple[str, ...],
    requested: str | None,
) -> tuple[str, tuple[str, ...]]:
    if requested is not None:
        chosen = requested.strip()
        if chosen and chosen not in versions:
            versions = tuple(sorted((*versions, chosen)))
        return chosen, versions
    if len(versions) == 1:
        return versions[0], versions
    return "", versions


def build_cost_sidecar(
    output: str | Path,
    events: Iterable[CostEvent | Mapping[str, Any]],
    *,
    toolchain_fingerprint: ToolchainFingerprint | Mapping[str, Any] | str | None = None,
    pricing_version: str | None = None,
    run_id: str | None = None,
    generated_at: datetime | None = None,
) -> CostSidecar:
    """Build the cost sidecar for *output* from its paid calls.

    *events* may hold validated :class:`~aivia.core.models.cost.CostEvent`
    records or mappings using the model's aliases. When *pricing_version* is
    omitted it is derived from the events; a single version is recorded as-is
    and a mixed ledger leaves it empty while filling ``pricing_versions``.
    """
    records = _coerce_events(events)
    items = tuple(SidecarCostItem.from_event(event) for event in records)
    versions = _pricing_versions(records)
    chosen_version, versions = _resolve_pricing_version(versions, pricing_version)
    fingerprint = ToolchainFingerprint.model_validate(toolchain_fingerprint)
    selected_run = run_id.strip() if run_id is not None else _common_run_id(records)

    totals = SidecarTotals(
        total_usd=math.fsum(item.amount_usd for item in items),
        estimated_usd=math.fsum(item.amount_usd for item in items if item.estimated),
        actual_usd=math.fsum(item.amount_usd for item in items if not item.estimated),
        units=math.fsum(item.units for item in items),
        count=len(items),
    )

    return CostSidecar(
        output=str(Path(output).expanduser()),
        run_id=selected_run,
        totals=totals,
        items=items,
        providers=tuple(sorted({item.provider for item in items if item.provider})),
        models=tuple(sorted({item.model for item in items if item.model})),
        pricing_version=chosen_version,
        pricing_versions=versions,
        toolchain_fingerprint=fingerprint,
        generated_at=_as_utc(generated_at),
    )


def write_cost_sidecar(
    output: str | Path,
    events: Iterable[CostEvent | Mapping[str, Any]],
    *,
    toolchain_fingerprint: ToolchainFingerprint | Mapping[str, Any] | str | None = None,
    pricing_version: str | None = None,
    run_id: str | None = None,
    generated_at: datetime | None = None,
) -> Path:
    """Write the cost sidecar next to *output* and return its path.

    The file is written as canonical, indented JSON with a trailing newline.
    Re-running the same output overwrites its sidecar instead of accumulating
    duplicate files.
    """
    sidecar = build_cost_sidecar(
        output,
        events,
        toolchain_fingerprint=toolchain_fingerprint,
        pricing_version=pricing_version,
        run_id=run_id,
        generated_at=generated_at,
    )
    target = sidecar_path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(canonical_json(sidecar.to_dict(), indent=2) + "\n", "utf-8")
    return target


def write_sidecar(
    output: str | Path,
    events: Iterable[CostEvent | Mapping[str, Any]],
    *,
    toolchain_fingerprint: ToolchainFingerprint | Mapping[str, Any] | str | None = None,
    pricing_version: str | None = None,
    run_id: str | None = None,
    generated_at: datetime | None = None,
) -> Path:
    """Alias for :func:`write_cost_sidecar`."""
    return write_cost_sidecar(
        output,
        events,
        toolchain_fingerprint=toolchain_fingerprint,
        pricing_version=pricing_version,
        run_id=run_id,
        generated_at=generated_at,
    )
