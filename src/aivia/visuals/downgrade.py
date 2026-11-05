"""Downgrade visual beats from video to image to stock under a budget cap.

A beat may ask for the most expensive visual source while a run only has room
for a cheaper one. This module keeps that decision deterministic: it estimates
each tier, walks the requested tier and every cheaper tier in order, and picks
the first one that still fits the remaining budget. When nothing fits, the beat
is reported as unresolved instead of spending over the cap.

The actual resolution still goes through the existing image, video and stock
resolvers, so every paid call keeps the same screen -> cache -> budget -> router
-> provider -> validate -> store -> cost -> sidecar path. A service that raises
:class:`~aivia.core.errors.BudgetError` is treated as a signal to try the next
cheaper tier, which makes the cap hold even when the configured estimates are
too optimistic.

Every downgrade emits a warning through the supplied notifier (the console
notifier by default) and is also recorded on the result, so a caller can inspect
what happened without parsing logs. Stock is the final tier: a stock beat is
never upgraded and an image beat only falls back to stock.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from aivia.core.config.budget import BudgetConfig
from aivia.core.errors import BudgetError, ConfigError
from aivia.core.models.run import StageResult, StageState
from aivia.core.models.script import MediaKind, VisualType
from aivia.core.stages.base import Stage, StageContext
from aivia.providers.console import ConsoleNotifier
from aivia.providers.protocols.notify import (
    NotificationLevel,
    NotificationMessage,
)
from aivia.visuals.image_stage import _write_json, resolve_image_request
from aivia.visuals.stock_stage import resolve_stock_request
from aivia.visuals.strategy import VisualRequest, resolve_visual_requests
from aivia.visuals.video_stage import resolve_video_request

__all__ = [
    "DEFAULT_DOWNGRADE_FILENAME",
    "DEFAULT_DOWNGRADE_MANIFEST_FILENAME",
    "DEFAULT_VISUAL_DOWNGRADE_FILENAME",
    "DOWNGRADE_FILENAME",
    "DOWNGRADE_MANIFEST_FILENAME",
    "DOWNGRADE_STAGE_NAME",
    "VISUAL_DOWNGRADE_FILENAME",
    "VISUAL_DOWNGRADE_STAGE_NAME",
    "Downgrade",
    "DowngradeBeat",
    "DowngradeBeatResult",
    "DowngradeDecision",
    "DowngradePolicy",
    "DowngradeResolver",
    "DowngradeResult",
    "DowngradeService",
    "DowngradeStage",
    "DowngradeStageResult",
    "ResolvedVisualDowngrade",
    "VisualDowngrade",
    "VisualDowngradeBeat",
    "VisualDowngradeDecision",
    "VisualDowngradePolicy",
    "VisualDowngradeResolver",
    "VisualDowngradeResult",
    "VisualDowngradeService",
    "VisualDowngradeStage",
    "VisualDowngradeStageResult",
    "VisualTier",
    "build_downgrade_stage",
    "build_visual_downgrade_request",
    "build_visual_downgrade_stage",
    "downgrade",
    "downgrade_visual_beats",
    "downgrade_visual_request",
    "downgrade_visual_requests",
    "downgrade_visuals",
    "resolve_downgrade",
    "resolve_downgrade_beat",
    "resolve_downgrade_beats",
    "resolve_downgrade_request",
    "resolve_downgrade_requests",
    "resolve_visual_downgrade",
    "resolve_visual_downgrade_beat",
    "resolve_visual_downgrade_request",
    "resolve_visual_downgrade_requests",
    "resolve_visual_downgrade_requests_sync",
    "resolve_visual_downgrade_sync",
    "resolve_visual_downgrades",
    "run_downgrade_stage",
    "run_visual_downgrade_stage",
    "visual_downgrade_stage",
]

DOWNGRADE_STAGE_NAME = "downgrade"
"""The recipe stage name this implementation serves."""

VISUAL_DOWNGRADE_STAGE_NAME = DOWNGRADE_STAGE_NAME
"""Alias for :data:`DOWNGRADE_STAGE_NAME`."""

DEFAULT_DOWNGRADE_FILENAME = "downgrade.json"
"""The manifest that records every downgrade decision under the run directory."""

DOWNGRADE_FILENAME = DEFAULT_DOWNGRADE_FILENAME
"""Alias for :data:`DEFAULT_DOWNGRADE_FILENAME`."""

DEFAULT_VISUAL_DOWNGRADE_FILENAME = "visual-downgrade.json"
"""Alias manifest name kept for callers that prefer the visual prefix."""

VISUAL_DOWNGRADE_FILENAME = DEFAULT_VISUAL_DOWNGRADE_FILENAME
"""Alias for :data:`DEFAULT_VISUAL_DOWNGRADE_FILENAME`."""

DEFAULT_DOWNGRADE_MANIFEST_FILENAME = DEFAULT_DOWNGRADE_FILENAME
"""Alias for :data:`DEFAULT_DOWNGRADE_FILENAME`."""

DOWNGRADE_MANIFEST_FILENAME = DEFAULT_DOWNGRADE_FILENAME
"""Alias for :data:`DEFAULT_DOWNGRADE_FILENAME`."""

_EPSILON = 1e-9
"""Float tolerance used when comparing an estimate with the remaining budget."""

_TIER_RANK = {VisualType.STOCK: 0, VisualType.IMAGE: 1, VisualType.VIDEO: 2}
"""The cheap-to-expensive order used to decide whether a tier is a downgrade."""

_TIER_KIND = {
    VisualType.VIDEO: MediaKind.VIDEO,
    VisualType.IMAGE: MediaKind.IMAGE,
    VisualType.STOCK: MediaKind.STOCK,
}


class VisualTier(StrEnum):
    """The three paid visual sources in downgrade order."""

    VIDEO = "video"
    IMAGE = "image"
    STOCK = "stock"


_TIER_ALIASES: dict[str, VisualTier] = {
    "video": VisualTier.VIDEO,
    "videos": VisualTier.VIDEO,
    "clip": VisualTier.VIDEO,
    "clips": VisualTier.VIDEO,
    "motion": VisualTier.VIDEO,
    "video_gen": VisualTier.VIDEO,
    "ai_video": VisualTier.VIDEO,
    "image": VisualTier.IMAGE,
    "images": VisualTier.IMAGE,
    "still": VisualTier.IMAGE,
    "stills": VisualTier.IMAGE,
    "text": VisualTier.IMAGE,
    "image_gen": VisualTier.IMAGE,
    "ai_image": VisualTier.IMAGE,
    "stock": VisualTier.STOCK,
    "stock_image": VisualTier.STOCK,
    "library": VisualTier.STOCK,
}

_TIER_VISUAL = {
    VisualTier.VIDEO: VisualType.VIDEO,
    VisualTier.IMAGE: VisualType.IMAGE,
    VisualTier.STOCK: VisualType.STOCK,
}

_VISUAL_TIER = {
    VisualType.VIDEO: VisualTier.VIDEO,
    VisualType.IMAGE: VisualTier.IMAGE,
    VisualType.STILL: VisualTier.IMAGE,
    VisualType.TEXT: VisualTier.IMAGE,
    VisualType.STOCK: VisualTier.STOCK,
}


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _amount(value: Any, *, label: str = "estimate_usd") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(
            f"{label} must be a finite, non-negative number",
            details={label: value},
        )
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ConfigError(
            f"{label} must be a finite, non-negative number",
            details={label: value},
        )
    return number


def _tier(value: Any) -> VisualTier:
    if isinstance(value, VisualTier):
        return value
    if isinstance(value, VisualType):
        tier = _VISUAL_TIER.get(value)
        if tier is not None:
            return tier
    if isinstance(value, str):
        name = value.strip().lower().replace("-", "_").replace(" ", "_")
        tier = _TIER_ALIASES.get(name)
        if tier is not None:
            return tier
    allowed = ", ".join(item.value for item in VisualTier)
    raise ConfigError(
        f"unknown visual tier: {value!r} (allowed: {allowed})",
        details={"tier": value, "allowed": [item.value for item in VisualTier]},
    )


def _coerce_request(value: Any) -> VisualRequest:
    if isinstance(value, VisualRequest):
        return value
    if isinstance(value, Mapping):
        return VisualRequest.model_validate(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ConfigError("visual downgrade needs a non-empty prompt")
        return VisualRequest(
            visual_type=VisualType.VIDEO,
            kind=MediaKind.VIDEO,
            prompt=text,
            text=text,
        )
    requests = resolve_visual_requests(value, strategy="video")
    return requests[0]


def _coerce_requests(value: Any) -> list[VisualRequest]:
    if isinstance(value, VisualRequest):
        return [value]
    if isinstance(value, Mapping):
        return [_coerce_request(value)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_coerce_request(item) for item in value]
    return resolve_visual_requests(value, strategy="video")


def _tiers_for(request: VisualRequest) -> tuple[VisualTier, ...]:
    requested = _VISUAL_TIER.get(request.visual_type)
    if requested is None:
        return ()
    rank = _TIER_RANK[_TIER_VISUAL[requested]]
    return tuple(
        tier
        for tier in (VisualTier.VIDEO, VisualTier.IMAGE, VisualTier.STOCK)
        if _TIER_RANK[_TIER_VISUAL[tier]] <= rank
    )


def _request_for_tier(request: VisualRequest, tier: VisualTier) -> VisualRequest:
    visual_type = _TIER_VISUAL[tier]
    values: dict[str, Any] = {
        "visual_type": visual_type,
        "kind": _TIER_KIND[visual_type],
    }
    if tier is VisualTier.STOCK:
        values["query"] = (
            _text(request.query) or _text(request.prompt) or _text(request.text)
        )
    return request.model_copy(update=values)


def _budget_cap(budget: Any) -> float | None:
    if budget is None:
        return None
    if isinstance(budget, bool):
        raise ConfigError(
            "budget must be a BudgetConfig, mapping or number",
            details={"budget": budget},
        )
    if isinstance(budget, (int, float)):
        return _amount(budget, label="budget")
    if isinstance(budget, Mapping):
        budget = BudgetConfig.from_mapping(budget)
    if isinstance(budget, BudgetConfig):
        for name in ("per_run", "per_call", "per_video", "per_day", "per_month"):
            value = float(getattr(budget, name, 0.0) or 0.0)
            if value > 0:
                return value
        return None
    raise ConfigError(
        "budget must be a BudgetConfig, mapping or number",
        details={"budget": type(budget).__name__},
    )


def _config_budget(config: Any) -> Any:
    if isinstance(config, BudgetConfig):
        return config
    if isinstance(config, Mapping):
        return config.get("budget")
    return getattr(config, "budget", None)


def _coerce_estimates(
    estimates: Any = None,
    costs: Any = None,
    prices: Any = None,
) -> dict[VisualTier, Any]:
    result: dict[VisualTier, Any] = {}
    for source in (estimates, costs, prices):
        if source is None:
            continue
        if not isinstance(source, Mapping):
            if isinstance(source, Sequence) and not isinstance(
                source, (str, bytes, bytearray)
            ):
                try:
                    source = dict(source)
                except (TypeError, ValueError) as exc:
                    raise ConfigError(
                        "visual estimates must be tier/amount pairs",
                        details={"estimates": source},
                    ) from exc
            else:
                raise ConfigError(
                    "visual estimates must be a mapping of tier to amount",
                    details={"estimates": type(source).__name__},
                )
        for key, value in source.items():
            result[_tier(key)] = value
    return result


def _estimate_for(
    estimates: Mapping[VisualTier, Any],
    request: VisualRequest,
    tier: VisualTier,
    *,
    default: float = 0.0,
) -> float:
    value = estimates.get(tier)
    if isinstance(value, Mapping):
        for key in (request.index, str(request.index), request.prompt, request.query):
            if key in value:
                nested = value[key]
                if isinstance(nested, Mapping):
                    nested = nested.get(tier.value, nested.get(tier, default))
                return _amount(nested, label=f"{tier.value} estimate")
        return _amount(default, label=f"{tier.value} estimate")
    if value is None:
        return _amount(default, label=f"{tier.value} estimate")
    return _amount(value, label=f"{tier.value} estimate")


class VisualDowngradeDecision(BaseModel):
    """The tier chosen for one beat and why the cheaper ones were not used.

    ``candidates`` keeps every tier in downgrade order with its estimate and
    whether it fit the remaining budget, so a manifest can explain the decision
    without re-running the estimates. ``chosen_type`` is ``None`` only when no
    tier fits.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    index: int = Field(default=0, ge=0)
    requested_type: VisualType = Field(
        default=VisualType.VIDEO,
        validation_alias=AliasChoices("requested_type", "requested", "from_type"),
    )
    chosen_type: VisualType | None = Field(
        default=None,
        validation_alias=AliasChoices("chosen_type", "chosen", "to_type"),
    )
    estimate_usd: float = Field(default=0.0, ge=0.0)
    remaining_usd: float | None = None
    cap_usd: float | None = None
    downgraded: bool = False
    reason: str = ""
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    message: str = ""

    @property
    def requested(self) -> VisualType:
        """Alias for :attr:`requested_type`."""
        return self.requested_type

    @property
    def chosen(self) -> VisualType | None:
        """Alias for :attr:`chosen_type`."""
        return self.chosen_type

    @property
    def visual_type(self) -> VisualType | None:
        """The tier the beat will actually use, if any."""
        return self.chosen_type

    @property
    def type(self) -> VisualType | None:
        """Alias for :attr:`chosen_type`."""
        return self.chosen_type

    @property
    def from_type(self) -> VisualType:
        """Alias for :attr:`requested_type`."""
        return self.requested_type

    @property
    def to_type(self) -> VisualType | None:
        """Alias for :attr:`chosen_type`."""
        return self.chosen_type

    @property
    def allowed(self) -> bool:
        """Whether some tier fits the remaining budget."""
        return self.chosen_type is not None

    @property
    def blocked(self) -> bool:
        """Whether no tier fits the remaining budget."""
        return self.chosen_type is None

    @property
    def cost_usd(self) -> float:
        """Alias for :attr:`estimate_usd`."""
        return self.estimate_usd

    @property
    def reason_text(self) -> str:
        """Alias for :attr:`message`."""
        return self.message

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for the manifest."""
        return {
            "index": self.index,
            "requested_type": self.requested_type.value,
            "chosen_type": (
                self.chosen_type.value if self.chosen_type is not None else None
            ),
            "estimate_usd": self.estimate_usd,
            "remaining_usd": self.remaining_usd,
            "cap_usd": self.cap_usd,
            "downgraded": self.downgraded,
            "allowed": self.allowed,
            "blocked": self.blocked,
            "reason": self.reason,
            "message": self.message,
            "candidates": [dict(item) for item in self.candidates],
        }


class VisualDowngradePolicy:
    """Choose the most expensive visual tier that fits the remaining budget.

    ``budget`` may be a :class:`~aivia.core.config.budget.BudgetConfig`, a
    mapping using its aliases or a plain number. When a budget has more than one
    active cap the tightest practical per-run cap is used, falling back to the
    per-call and per-video caps. An explicit ``cap_usd`` wins. A cap of ``0``
    means only free tiers may run, while ``None`` means no cap at all.

    ``estimates`` maps a tier to its expected cost. Keys may be
    :class:`VisualTier`, :class:`VisualType` or their string aliases. A value
    may also be a mapping keyed by beat index or prompt, which lets a caller
    price each beat differently.
    """

    def __init__(
        self,
        budget: Any = None,
        *,
        cap_usd: float | None = None,
        cap: float | None = None,
        max_usd: float | None = None,
        budget_usd: float | None = None,
        limit: float | None = None,
        estimates: Any = None,
        costs: Any = None,
        prices: Any = None,
        order: Any = None,
        tiers: Any = None,
        default_estimate: float = 0.0,
    ) -> None:
        explicit = [
            value
            for value in (cap_usd, cap, max_usd, budget_usd, limit)
            if value is not None
        ]
        if len(explicit) > 1:
            raise ConfigError("pass only one explicit visual budget cap")
        selected: float | None
        if explicit:
            selected = _amount(explicit[0], label="cap_usd")
        else:
            selected = _budget_cap(budget)
        self._cap = selected
        self._estimates = _coerce_estimates(estimates, costs, prices)
        self._default_estimate = _amount(default_estimate, label="default_estimate")
        requested_order = order if order is not None else tiers
        if requested_order is None:
            self._order: tuple[VisualTier, ...] | None = None
        else:
            if isinstance(requested_order, (str, bytes)):
                requested_order = (requested_order,)
            self._order = tuple(_tier(item) for item in requested_order)

    @property
    def cap(self) -> float | None:
        """The configured budget cap, or ``None`` when unlimited."""
        return self._cap

    @property
    def cap_usd(self) -> float | None:
        """Alias for :attr:`cap`."""
        return self._cap

    @property
    def estimates(self) -> dict[VisualTier, Any]:
        """The tier estimates, keyed by :class:`VisualTier`."""
        return dict(self._estimates)

    @property
    def order(self) -> tuple[VisualTier, ...]:
        """The configured downgrade order, or the default video/image/stock."""
        if self._order is None:
            return (VisualTier.VIDEO, VisualTier.IMAGE, VisualTier.STOCK)
        return self._order

    def tiers_for(self, value: Any) -> tuple[VisualTier, ...]:
        """Return the tiers to try for *value* without ever upgrading it."""
        request = _coerce_request(value)
        requested = _VISUAL_TIER.get(request.visual_type)
        if requested is None:
            return ()
        rank = _TIER_RANK[_TIER_VISUAL[requested]]
        ordered: tuple[VisualTier, ...]
        if self._order is None:
            ordered = (VisualTier.VIDEO, VisualTier.IMAGE, VisualTier.STOCK)
        else:
            ordered = self._order
        result: list[VisualTier] = []
        for tier in ordered:
            if _TIER_RANK[_TIER_VISUAL[tier]] > rank:
                continue
            if tier not in result:
                result.append(tier)
        return tuple(result)

    def estimate_for(self, value: Any, tier: Any) -> float:
        """Return the expected cost of *tier* for one request."""
        request = _coerce_request(value)
        return _estimate_for(
            self._estimates,
            request,
            _tier(tier),
            default=self._default_estimate,
        )

    def remaining(
        self,
        *,
        remaining_usd: float | None = None,
        spent_usd: float = 0.0,
    ) -> float | None:
        """Return the amount still available for one beat."""
        if remaining_usd is not None:
            return _amount(remaining_usd, label="remaining_usd")
        if self._cap is None:
            return None
        spent = _amount(spent_usd, label="spent_usd")
        return max(self._cap - spent, 0.0)

    def candidates(
        self,
        value: Any,
        *,
        remaining_usd: float | None = None,
        spent_usd: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Return every tier with its estimate and whether it fits."""
        request = _coerce_request(value)
        remaining = self.remaining(
            remaining_usd=remaining_usd,
            spent_usd=spent_usd,
        )
        result: list[dict[str, Any]] = []
        for tier in self.tiers_for(request):
            estimate = self.estimate_for(request, tier)
            affordable = remaining is None or estimate <= remaining + _EPSILON
            result.append(
                {
                    "tier": tier.value,
                    "visual_type": _TIER_VISUAL[tier].value,
                    "estimate_usd": estimate,
                    "affordable": affordable,
                }
            )
        return result

    def decide(
        self,
        value: Any,
        *,
        remaining_usd: float | None = None,
        spent_usd: float = 0.0,
    ) -> VisualDowngradeDecision:
        """Return the chosen tier for one request without calling a provider."""
        request = _coerce_request(value)
        remaining = self.remaining(
            remaining_usd=remaining_usd,
            spent_usd=spent_usd,
        )
        candidates = self.candidates(
            request,
            remaining_usd=remaining,
        )
        selected = next(
            (item for item in candidates if item["affordable"]),
            None,
        )
        requested = _VISUAL_TIER.get(request.visual_type)
        requested_type = (
            _TIER_VISUAL[requested] if requested is not None else request.visual_type
        )
        if selected is None:
            if not candidates:
                reason = "no visual tier is available"
            else:
                reason = "no visual tier fits the remaining budget"
            return VisualDowngradeDecision(
                index=request.index,
                requested_type=requested_type,
                chosen_type=None,
                estimate_usd=0.0,
                remaining_usd=remaining,
                cap_usd=self._cap,
                downgraded=False,
                reason=reason,
                candidates=candidates,
                message=(
                    f"beat {request.index} needs {requested_type.value} but {reason}"
                ),
            )

        chosen = _tier(selected["tier"])
        chosen_type = _TIER_VISUAL[chosen]
        estimate = float(selected["estimate_usd"])
        downgraded = requested is not None and chosen is not requested
        if downgraded:
            assert requested is not None
            requested_estimate = self.estimate_for(request, requested)
            reason = (
                f"{requested.value} estimate {requested_estimate:.6g} USD does "
                f"not fit the remaining budget; using {chosen.value}"
            )
        else:
            reason = f"{chosen.value} estimate {estimate:.6g} USD fits the budget"
        if remaining is None:
            message = reason
        else:
            message = f"{reason} ({remaining:.6g} USD remaining)"
        return VisualDowngradeDecision(
            index=request.index,
            requested_type=requested_type,
            chosen_type=chosen_type,
            estimate_usd=estimate,
            remaining_usd=remaining,
            cap_usd=self._cap,
            downgraded=downgraded,
            reason=reason,
            candidates=candidates,
            message=message,
        )

    choose = decide
    select = decide
    evaluate = decide
    resolve = decide

    def alert_for(
        self,
        decision: VisualDowngradeDecision,
    ) -> NotificationMessage | None:
        """Build the warning for a downgrade, or ``None`` when nothing changed."""
        if decision.chosen_type is not None and not decision.downgraded:
            return None
        if decision.chosen_type is None and not decision.candidates:
            return None
        requested = decision.requested_type.value
        chosen = (
            decision.chosen_type.value if decision.chosen_type is not None else "none"
        )
        cap = (
            f"{decision.cap_usd:.6g} USD"
            if decision.cap_usd is not None
            else "the configured budget"
        )
        text = (
            f"Beat {decision.index} was downgraded from {requested} to {chosen} "
            f"to stay within {cap}."
        )
        if decision.chosen_type is None:
            text = (
                f"Beat {decision.index} could not use {requested}: no visual "
                f"tier fits {cap}."
            )
        return NotificationMessage(
            title=f"Visual downgraded: {requested} -> {chosen}",
            text=text,
            level=NotificationLevel.WARNING,
            metadata={
                "index": decision.index,
                "requested_type": requested,
                "chosen_type": chosen,
                "estimate_usd": decision.estimate_usd,
                "remaining_usd": decision.remaining_usd,
                "cap_usd": decision.cap_usd,
                "reason": decision.reason,
            },
        )


class VisualDowngradeResult(BaseModel):
    """One beat after the downgrade decision and its actual resolution."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    index: int = Field(default=0, ge=0)
    request: VisualRequest | None = None
    decision: VisualDowngradeDecision | None = None
    visual: Any = Field(default=None, exclude=True)
    asset: Any = Field(default=None, exclude=True)
    requested_type: VisualType = VisualType.VIDEO
    chosen_type: VisualType | None = None
    cost_usd: float = Field(default=0.0, ge=0.0)
    error: str = ""
    attempts: list[dict[str, Any]] = Field(default_factory=list)
    alerts: list[NotificationMessage] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        for source in (
            self.__dict__.get("visual"),
            self.__dict__.get("asset"),
        ):
            if source is not None and hasattr(source, name):
                return getattr(source, name)
        raise AttributeError(name)

    @property
    def resolved(self) -> bool:
        """Whether an asset was stored for the chosen tier."""
        return self.asset is not None

    @property
    def passed(self) -> bool:
        """Alias for :attr:`resolved`."""
        return self.resolved

    @property
    def downgraded(self) -> bool:
        """Whether the chosen tier is cheaper than the requested one."""
        return bool(self.decision is not None and self.decision.downgraded)

    @property
    def visual_type(self) -> VisualType | None:
        """The tier that actually ran, if any."""
        return self.chosen_type

    @property
    def type(self) -> VisualType | None:
        """Alias for :attr:`chosen_type`."""
        return self.chosen_type

    @property
    def from_type(self) -> VisualType:
        """Alias for :attr:`requested_type`."""
        return self.requested_type

    @property
    def to_type(self) -> VisualType | None:
        """Alias for :attr:`chosen_type`."""
        return self.chosen_type

    @property
    def alert(self) -> NotificationMessage | None:
        """The first alert emitted for this beat, if any."""
        return self.alerts[0] if self.alerts else None

    @property
    def provider(self) -> str:
        """The provider reported by the resolved tier."""
        return _text(getattr(self.visual, "provider", "")) or _text(
            getattr(self.asset, "provider", "")
        )

    @property
    def model(self) -> str:
        """The model reported by the resolved tier."""
        return _text(getattr(self.visual, "model", "")) or _text(
            getattr(self.asset, "model", "")
        )

    @property
    def prompt_hash(self) -> str:
        """The prompt hash reported by the resolved tier."""
        return _text(getattr(self.visual, "prompt_hash", ""))

    @property
    def query(self) -> str:
        """The winning stock query, or an empty string."""
        return _text(getattr(self.visual, "query", ""))

    @property
    def score(self) -> float | None:
        """The stock relevance score, when the chosen tier is stock."""
        value = getattr(self.visual, "score", None)
        return float(value) if isinstance(value, (int, float)) else None

    @property
    def selected(self) -> Any:
        """Alias for :attr:`asset`."""
        return self.asset

    @property
    def first(self) -> Any:
        """Alias for :attr:`asset`."""
        return self.asset

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for the manifest."""
        visual = self.visual
        if hasattr(visual, "to_dict"):
            visual_payload: Any = visual.to_dict()
        elif hasattr(visual, "model_dump"):
            visual_payload = visual.model_dump(mode="json")
        else:
            visual_payload = None
        asset = self.asset
        if hasattr(asset, "model_dump"):
            asset_payload: Any = asset.model_dump(mode="json")
        else:
            asset_payload = None
        return {
            "index": self.index,
            "request": (self.request.to_dict() if self.request is not None else None),
            "decision": (
                self.decision.to_dict() if self.decision is not None else None
            ),
            "requested_type": self.requested_type.value,
            "chosen_type": (
                self.chosen_type.value if self.chosen_type is not None else None
            ),
            "asset": asset_payload,
            "visual": visual_payload,
            "cost_usd": self.cost_usd,
            "error": self.error,
            "downgraded": self.downgraded,
            "attempts": [dict(item) for item in self.attempts],
            "alerts": [alert.model_dump(mode="json") for alert in self.alerts],
            "metadata": dict(self.metadata),
        }


class VisualDowngradeStageResult(StageResult):
    """A stage result carrying one :class:`VisualDowngradeResult` per beat."""

    beats: list[VisualDowngradeResult] = Field(
        default_factory=list,
        validation_alias=AliasChoices("beats", "results", "items", "selections"),
    )
    alerts: list[NotificationMessage] = Field(default_factory=list)

    @property
    def results(self) -> list[VisualDowngradeResult]:
        """Alias for :attr:`beats`."""
        return self.beats

    @property
    def assets(self) -> list[Any]:
        """Every resolved asset in beat order."""
        return [beat.asset for beat in self.beats if beat.asset is not None]

    @property
    def selected(self) -> list[Any]:
        """Alias for :attr:`assets`."""
        return self.assets

    @property
    def unresolved(self) -> list[VisualDowngradeResult]:
        """Beats for which no tier produced an asset."""
        return [beat for beat in self.beats if beat.asset is None]

    @property
    def downgrades(self) -> list[VisualDowngradeResult]:
        """Beats whose chosen tier is cheaper than the requested one."""
        return [beat for beat in self.beats if beat.downgraded]

    @property
    def requested_types(self) -> dict[int, VisualType]:
        """The requested visual type keyed by beat index."""
        return {beat.index: beat.requested_type for beat in self.beats}

    @property
    def chosen_types(self) -> dict[int, VisualType | None]:
        """The chosen visual type keyed by beat index."""
        return {beat.index: beat.chosen_type for beat in self.beats}

    @property
    def total_cost_usd(self) -> float:
        """The total cost of every resolved beat."""
        return math.fsum(beat.cost_usd for beat in self.beats)


DowngradeDecision = VisualDowngradeDecision
DowngradePolicy = VisualDowngradePolicy
DowngradeResult = VisualDowngradeResult
DowngradeStageResult = VisualDowngradeStageResult
DowngradeBeat = VisualDowngradeResult
DowngradeBeatResult = VisualDowngradeResult
VisualDowngradeBeat = VisualDowngradeResult
ResolvedVisualDowngrade = VisualDowngradeResult


def _notifier_send(notifier: Any) -> Any:
    for name in ("send", "notify", "emit"):
        method = getattr(notifier, name, None)
        if callable(method):
            return method
    if callable(notifier):
        return notifier
    return None


class VisualDowngrade:
    """Resolve beats through the cheapest tier that still fits the budget.

    ``video``/``image``/``stock`` (and their aliases) are the collaborators the
    existing resolvers expect. A single ``router`` or ``media`` facade may be
    supplied instead. ``policy`` decides the tier; without one a policy is
    built from ``budget``/``cap_usd`` and ``estimates``. Downgrades are sent to
    ``notifier`` when one is supplied; by default they go to the console.
    """

    def __init__(
        self,
        video: Any = None,
        *,
        video_service: Any = None,
        videos: Any = None,
        image: Any = None,
        image_service: Any = None,
        images: Any = None,
        stock: Any = None,
        stock_service: Any = None,
        search: Any = None,
        stock_search: Any = None,
        services: Mapping[Any, Any] | None = None,
        visual_services: Mapping[Any, Any] | None = None,
        router: Any = None,
        media: Any = None,
        config: Any = None,
        budget: Any = None,
        cap_usd: float | None = None,
        cap: float | None = None,
        max_usd: float | None = None,
        budget_usd: float | None = None,
        limit: float | None = None,
        estimates: Any = None,
        costs: Any = None,
        prices: Any = None,
        policy: VisualDowngradePolicy | None = None,
        notifier: Any = None,
        alert_notifier: Any = None,
        emitter: Any = None,
        fallback_on_failure: bool = False,
        default_estimate: float = 0.0,
        **options: Any,
    ) -> None:
        self._video = video if video is not None else video_service
        if self._video is None:
            self._video = videos
        self._image = image if image is not None else image_service
        if self._image is None:
            self._image = images
        self._stock = stock if stock is not None else stock_service
        if self._stock is None:
            self._stock = search if search is not None else stock_search
        selected_services = services if services is not None else visual_services
        if selected_services is not None:
            if not isinstance(selected_services, Mapping):
                raise ConfigError(
                    "visual services must be a mapping of tier to collaborator",
                    details={"services": type(selected_services).__name__},
                )
            for key, value in selected_services.items():
                tier = _tier(key)
                if tier is VisualTier.VIDEO and self._video is None:
                    self._video = value
                elif tier is VisualTier.IMAGE and self._image is None:
                    self._image = value
                elif tier is VisualTier.STOCK and self._stock is None:
                    self._stock = value
        self._router = router
        self._media = media
        self._config = config
        self._fallback_on_failure = bool(fallback_on_failure)
        self._options = dict(options)

        if policy is not None:
            if any(
                value is not None
                for value in (
                    budget,
                    cap_usd,
                    cap,
                    max_usd,
                    budget_usd,
                    limit,
                    estimates,
                    costs,
                    prices,
                )
            ):
                raise ConfigError(
                    "pass either a visual downgrade policy or budget settings, not both"
                )
            self._policy = policy
        else:
            self._policy = VisualDowngradePolicy(
                budget,
                cap_usd=cap_usd,
                cap=cap,
                max_usd=max_usd,
                budget_usd=budget_usd,
                limit=limit,
                estimates=estimates,
                costs=costs,
                prices=prices,
                default_estimate=default_estimate,
            )

        selected = notifier if notifier is not None else alert_notifier
        if selected is None:
            selected = emitter
        if selected is None:
            selected = ConsoleNotifier()
        elif selected is False:
            selected = None
        self._notifier = selected

    @property
    def policy(self) -> VisualDowngradePolicy:
        """The policy that decides each beat's tier."""
        return self._policy

    @property
    def cap_usd(self) -> float | None:
        """Alias for :attr:`VisualDowngradePolicy.cap_usd`."""
        return self._policy.cap_usd

    @property
    def notifier(self) -> Any:
        """The notifier downgrade alerts are sent through, if any."""
        return self._notifier

    def _config_for(self, context: StageContext | None) -> Any:
        if self._config is not None:
            return self._config
        return context.config if context is not None else None

    async def _send(self, message: NotificationMessage) -> Any:
        if self._notifier is None:
            return None
        send = _notifier_send(self._notifier)
        if send is None:
            raise ConfigError(
                "visual downgrade notifier must expose a send coroutine",
                details={"notifier": type(self._notifier).__name__},
            )
        result = send(message)
        if inspect.isawaitable(result):
            result = await result
        return result

    def _targets(self) -> tuple[Any, Any, Any]:
        video = self._video
        image = self._image
        stock = self._stock
        if self._media is not None:
            if video is None:
                video = self._media
            if image is None:
                image = self._media
            if stock is None:
                stock = getattr(self._media, "search", None) or self._media
        return video, image, stock

    def _resolver_options(self) -> dict[str, Any]:
        blocked = {
            "budget",
            "cap",
            "cap_usd",
            "max_usd",
            "budget_usd",
            "limit",
            "estimates",
            "costs",
            "prices",
            "policy",
            "notifier",
            "alert_notifier",
            "emitter",
            "fallback_on_failure",
            "remaining_usd",
            "spent_usd",
            "requests",
            "visuals",
            "visual_requests",
            "script",
            "beats",
            "video",
            "video_service",
            "videos",
            "image",
            "image_service",
            "images",
            "stock",
            "stock_service",
            "search",
            "stock_search",
            "services",
            "visual_services",
            "media",
            "router",
            "config",
            "filename",
            "manifest_filename",
        }
        return {
            key: value for key, value in self._options.items() if key not in blocked
        }

    async def _resolve_tier(
        self,
        tier: VisualTier,
        request: VisualRequest,
        *,
        context: StageContext | None,
        run_id: str,
        metadata: Mapping[str, Any] | None,
        overrides: Mapping[str, Any],
    ) -> Any:
        video, image, stock = self._targets()
        config = self._config_for(context)
        options = self._resolver_options()
        options.update(overrides)
        if tier is VisualTier.VIDEO:
            return await resolve_video_request(
                request,
                service=video,
                router=self._router,
                context=context,
                config=config,
                run_id=run_id,
                metadata=metadata,
                **options,
            )
        if tier is VisualTier.IMAGE:
            return await resolve_image_request(
                request,
                service=image,
                router=self._router,
                context=context,
                config=config,
                run_id=run_id,
                metadata=metadata,
                **options,
            )
        return await resolve_stock_request(
            request,
            search=stock,
            router=self._router,
            config=config,
            run_id=run_id,
            metadata=metadata,
            **options,
        )

    async def resolve(
        self,
        value: Any,
        *,
        context: StageContext | None = None,
        run_id: str = "",
        metadata: Mapping[str, Any] | None = None,
        remaining_usd: float | None = None,
        spent_usd: float = 0.0,
        **overrides: Any,
    ) -> VisualDowngradeResult:
        """Resolve one visual beat, downgrading until it fits the budget."""
        request = _coerce_request(value)
        remaining = self._policy.remaining(
            remaining_usd=remaining_usd,
            spent_usd=spent_usd,
        )
        decision = self._policy.decide(request, remaining_usd=remaining)
        active_meta = dict(metadata or {})
        active_meta.setdefault("downgrade", True)
        attempts: list[dict[str, Any]] = []
        chosen_tier: VisualTier | None = None
        chosen_beat: Any = None
        alerts: list[NotificationMessage] = []
        error = ""

        for candidate in decision.candidates:
            tier = _tier(candidate["tier"])
            estimate = float(candidate["estimate_usd"])
            if not candidate["affordable"]:
                attempts.append(
                    {
                        "tier": tier.value,
                        "estimate_usd": estimate,
                        "selected": False,
                        "reason": "over budget",
                    }
                )
                continue
            tier_request = _request_for_tier(request, tier)
            try:
                beat = await self._resolve_tier(
                    tier,
                    tier_request,
                    context=context,
                    run_id=run_id,
                    metadata=active_meta,
                    overrides=overrides,
                )
            except BudgetError as exc:
                attempts.append(
                    {
                        "tier": tier.value,
                        "estimate_usd": estimate,
                        "selected": False,
                        "reason": f"budget error: {exc}",
                    }
                )
                continue
            except Exception as exc:
                attempts.append(
                    {
                        "tier": tier.value,
                        "estimate_usd": estimate,
                        "selected": False,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                raise
            cost = float(getattr(beat, "cost_usd", 0.0) or 0.0)
            resolved = getattr(beat, "asset", None) is not None
            attempts.append(
                {
                    "tier": tier.value,
                    "estimate_usd": estimate,
                    "actual_usd": cost,
                    "selected": resolved or not self._fallback_on_failure,
                    "resolved": resolved,
                    "reason": "" if resolved else "no asset returned",
                }
            )
            if resolved or not self._fallback_on_failure:
                chosen_tier = tier
                chosen_beat = beat
                break

        if chosen_tier is None:
            error = "no visual tier could be resolved within the budget"
            chosen_beat = None

        chosen_type = _TIER_VISUAL[chosen_tier] if chosen_tier is not None else None
        if chosen_type is not None and chosen_type != decision.chosen_type:
            decision = self._policy.decide(request, remaining_usd=remaining)
            decision = decision.model_copy(
                update={
                    "chosen_type": chosen_type,
                    "estimate_usd": float(getattr(chosen_beat, "cost_usd", 0.0) or 0.0),
                    "downgraded": chosen_type != decision.requested_type,
                    "reason": "resolved a cheaper tier after the first choice failed",
                }
            )
        if chosen_type is None:
            decision = decision.model_copy(update={"chosen_type": None})

        alert = self._policy.alert_for(decision)
        if alert is not None:
            alerts.append(alert)
            await self._send(alert)

        asset = getattr(chosen_beat, "asset", None) if chosen_beat is not None else None
        cost = float(getattr(chosen_beat, "cost_usd", 0.0) or 0.0)
        return VisualDowngradeResult(
            index=request.index,
            request=request,
            decision=decision,
            visual=chosen_beat,
            asset=asset,
            requested_type=decision.requested_type,
            chosen_type=chosen_type,
            cost_usd=cost,
            error=error,
            attempts=attempts,
            alerts=alerts,
            metadata=active_meta,
        )

    async def resolve_requests(
        self,
        values: Any,
        *,
        context: StageContext | None = None,
        run_id: str = "",
        metadata: Mapping[str, Any] | None = None,
        remaining_usd: float | None = None,
        spent_usd: float = 0.0,
        **overrides: Any,
    ) -> list[VisualDowngradeResult]:
        """Resolve every beat while sharing one budget across the sequence."""
        requests = _coerce_requests(values)
        cap = self._policy.cap_usd
        spent = _amount(spent_usd, label="spent_usd")
        start_remaining = (
            self._policy.remaining(
                remaining_usd=remaining_usd,
                spent_usd=spent,
            )
            if remaining_usd is not None or cap is not None
            else None
        )
        if start_remaining is not None:
            spent = max((cap if cap is not None else 0.0) - start_remaining, 0.0)
        results: list[VisualDowngradeResult] = []
        for position, request in enumerate(requests):
            active_remaining = (
                max(start_remaining - spent, 0.0)
                if start_remaining is not None
                else None
            )
            active_meta = dict(metadata or {})
            active_meta.setdefault("position", position)
            result = await self.resolve(
                request,
                context=context,
                run_id=run_id,
                metadata=active_meta,
                remaining_usd=active_remaining,
                **overrides,
            )
            results.append(result)
            spent += result.cost_usd
        return results

    __call__ = resolve


VisualDowngradeResolver = VisualDowngrade
DowngradeResolver = VisualDowngrade
VisualDowngradeService = VisualDowngrade
DowngradeService = VisualDowngrade
Downgrade = VisualDowngrade


async def resolve_visual_downgrade(
    value: Any,
    *,
    video: Any = None,
    video_service: Any = None,
    image: Any = None,
    image_service: Any = None,
    stock: Any = None,
    search: Any = None,
    services: Mapping[Any, Any] | None = None,
    visual_services: Mapping[Any, Any] | None = None,
    router: Any = None,
    media: Any = None,
    config: Any = None,
    budget: Any = None,
    cap_usd: float | None = None,
    cap: float | None = None,
    estimates: Any = None,
    costs: Any = None,
    prices: Any = None,
    policy: VisualDowngradePolicy | None = None,
    notifier: Any = None,
    alert_notifier: Any = None,
    fallback_on_failure: bool = False,
    context: StageContext | None = None,
    run_id: str = "",
    metadata: Mapping[str, Any] | None = None,
    remaining_usd: float | None = None,
    spent_usd: float = 0.0,
    **options: Any,
) -> VisualDowngradeResult:
    """Resolve one beat, downgrading video -> image -> stock as needed."""
    resolver = VisualDowngrade(
        video,
        video_service=video_service,
        image=image,
        image_service=image_service,
        stock=stock,
        search=search,
        services=services,
        visual_services=visual_services,
        router=router,
        media=media,
        config=config,
        budget=budget,
        cap_usd=cap_usd,
        cap=cap,
        estimates=estimates,
        costs=costs,
        prices=prices,
        policy=policy,
        notifier=notifier,
        alert_notifier=alert_notifier,
        fallback_on_failure=fallback_on_failure,
        **options,
    )
    return await resolver.resolve(
        value,
        context=context,
        run_id=run_id,
        metadata=metadata,
        remaining_usd=remaining_usd,
        spent_usd=spent_usd,
    )


async def resolve_visual_downgrade_requests(
    values: Any,
    *,
    video: Any = None,
    video_service: Any = None,
    image: Any = None,
    image_service: Any = None,
    stock: Any = None,
    search: Any = None,
    services: Mapping[Any, Any] | None = None,
    visual_services: Mapping[Any, Any] | None = None,
    router: Any = None,
    media: Any = None,
    config: Any = None,
    budget: Any = None,
    cap_usd: float | None = None,
    cap: float | None = None,
    estimates: Any = None,
    costs: Any = None,
    prices: Any = None,
    policy: VisualDowngradePolicy | None = None,
    notifier: Any = None,
    alert_notifier: Any = None,
    fallback_on_failure: bool = False,
    context: StageContext | None = None,
    run_id: str = "",
    metadata: Mapping[str, Any] | None = None,
    remaining_usd: float | None = None,
    spent_usd: float = 0.0,
    **options: Any,
) -> list[VisualDowngradeResult]:
    """Resolve every beat, sharing the budget across the whole sequence."""
    resolver = VisualDowngrade(
        video,
        video_service=video_service,
        image=image,
        image_service=image_service,
        stock=stock,
        search=search,
        services=services,
        visual_services=visual_services,
        router=router,
        media=media,
        config=config,
        budget=budget,
        cap_usd=cap_usd,
        cap=cap,
        estimates=estimates,
        costs=costs,
        prices=prices,
        policy=policy,
        notifier=notifier,
        alert_notifier=alert_notifier,
        fallback_on_failure=fallback_on_failure,
        **options,
    )
    return await resolver.resolve_requests(
        values,
        context=context,
        run_id=run_id,
        metadata=metadata,
        remaining_usd=remaining_usd,
        spent_usd=spent_usd,
    )


downgrade_visual_request = resolve_visual_downgrade
resolve_downgrade_request = resolve_visual_downgrade
resolve_visual_downgrade_request = resolve_visual_downgrade
resolve_downgrade = resolve_visual_downgrade
resolve_downgrade_beat = resolve_visual_downgrade
resolve_visual_downgrade_beat = resolve_visual_downgrade
downgrade = resolve_visual_downgrade
build_visual_downgrade_request = resolve_visual_downgrade
downgrade_visual_requests = resolve_visual_downgrade_requests
resolve_downgrade_requests = resolve_visual_downgrade_requests
downgrade_visual_beats = resolve_visual_downgrade_requests
resolve_downgrade_beats = resolve_visual_downgrade_requests
resolve_visual_downgrades = resolve_visual_downgrade_requests
downgrade_visuals = resolve_visual_downgrade_requests


def resolve_visual_downgrade_sync(value: Any, **kwargs: Any) -> VisualDowngradeResult:
    """Run :func:`resolve_visual_downgrade` from synchronous code."""
    import asyncio

    return asyncio.run(resolve_visual_downgrade(value, **kwargs))


def resolve_visual_downgrade_requests_sync(
    values: Any,
    **kwargs: Any,
) -> list[VisualDowngradeResult]:
    """Run :func:`resolve_visual_downgrade_requests` from synchronous code."""
    import asyncio

    return asyncio.run(resolve_visual_downgrade_requests(values, **kwargs))


class VisualDowngradeStage(Stage):
    """Resolve visual beats while downgrading to stay inside a budget cap.

    ``requests`` may be a script, a beat sequence, one or more visual requests
    or mappings. ``video``/``image``/``stock`` (and their aliases) supply the
    collaborators; without them the stage uses ``context.media``. ``budget`` or
    ``cap_usd`` sets the shared cap and ``estimates`` prices each tier. The
    stage writes ``downgrade.json`` (and the ``visual-downgrade.json`` alias)
    with every decision and alert.
    """

    name = DOWNGRADE_STAGE_NAME
    artifacts: tuple[str, ...] = (DEFAULT_DOWNGRADE_FILENAME,)

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
        video: Any = None,
        video_service: Any = None,
        videos: Any = None,
        image: Any = None,
        image_service: Any = None,
        images: Any = None,
        stock: Any = None,
        stock_service: Any = None,
        search: Any = None,
        stock_search: Any = None,
        services: Mapping[Any, Any] | None = None,
        visual_services: Mapping[Any, Any] | None = None,
        router: Any = None,
        media: Any = None,
        config: Any = None,
        budget: Any = None,
        cap_usd: float | None = None,
        cap: float | None = None,
        max_usd: float | None = None,
        budget_usd: float | None = None,
        limit: float | None = None,
        estimates: Any = None,
        costs: Any = None,
        prices: Any = None,
        policy: VisualDowngradePolicy | None = None,
        notifier: Any = None,
        alert_notifier: Any = None,
        emitter: Any = None,
        fallback_on_failure: bool = False,
        default_estimate: float = 0.0,
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
            "video": video,
            "video_service": video_service,
            "videos": videos,
            "image": image,
            "image_service": image_service,
            "images": images,
            "stock": stock,
            "stock_service": stock_service,
            "search": search,
            "stock_search": stock_search,
            "services": services,
            "visual_services": visual_services,
            "router": router,
            "media": media,
            "config": config,
            "budget": budget,
            "cap_usd": cap_usd,
            "cap": cap,
            "max_usd": max_usd,
            "budget_usd": budget_usd,
            "limit": limit,
            "estimates": estimates,
            "costs": costs,
            "prices": prices,
            "policy": policy,
            "notifier": notifier,
            "alert_notifier": alert_notifier,
            "emitter": emitter,
            "fallback_on_failure": fallback_on_failure,
            "default_estimate": default_estimate,
            "filename": filename,
            "manifest_filename": manifest_filename,
        }
        for key, value in supplied.items():
            if value is not None:
                merged[key] = value
        self.filename = _text(
            merged.pop("filename", None)
            or merged.pop("manifest_filename", None)
            or DEFAULT_DOWNGRADE_FILENAME
        )
        if not self.filename:
            self.filename = DEFAULT_DOWNGRADE_FILENAME
        self.artifacts = (self.filename,)

        self._requests_input = merged.pop("requests", None)
        if self._requests_input is None:
            self._requests_input = merged.pop("visuals", None)
        self._script = merged.pop("script", None)
        self._beats = merged.pop("beats", None)
        self._video = merged.pop("video", None)
        if self._video is None:
            self._video = merged.pop("video_service", None)
        if self._video is None:
            self._video = merged.pop("videos", None)
        self._image = merged.pop("image", None)
        if self._image is None:
            self._image = merged.pop("image_service", None)
        if self._image is None:
            self._image = merged.pop("images", None)
        self._stock = merged.pop("stock", None)
        if self._stock is None:
            self._stock = merged.pop("stock_service", None)
        if self._stock is None:
            self._stock = merged.pop("search", None)
        if self._stock is None:
            self._stock = merged.pop("stock_search", None)
        selected_services = merged.pop("services", None)
        if selected_services is None:
            selected_services = merged.pop("visual_services", None)
        if selected_services is not None:
            if not isinstance(selected_services, Mapping):
                raise ConfigError(
                    "visual services must be a mapping of tier to collaborator",
                    details={"services": type(selected_services).__name__},
                )
            for key, value in selected_services.items():
                tier = _tier(key)
                if tier is VisualTier.VIDEO and self._video is None:
                    self._video = value
                elif tier is VisualTier.IMAGE and self._image is None:
                    self._image = value
                elif tier is VisualTier.STOCK and self._stock is None:
                    self._stock = value
        self._router = merged.pop("router", None)
        self._media = merged.pop("media", None)
        self._config = merged.pop("config", None)
        self._budget = merged.pop("budget", None)
        self._cap_usd = merged.pop("cap_usd", None)
        if self._cap_usd is None:
            for alias in ("cap", "max_usd", "budget_usd", "limit"):
                if alias in merged:
                    self._cap_usd = merged.pop(alias)
                    break
        self._estimates = merged.pop("estimates", None)
        if self._estimates is None:
            self._estimates = merged.pop("costs", None)
        if self._estimates is None:
            self._estimates = merged.pop("prices", None)
        self._policy = merged.pop("policy", None)
        self._notifier = merged.pop("notifier", None)
        if self._notifier is None:
            self._notifier = merged.pop("alert_notifier", None)
        if self._notifier is None:
            self._notifier = merged.pop("emitter", None)
        self._fallback_on_failure = bool(merged.pop("fallback_on_failure", False))
        self._default_estimate = merged.pop("default_estimate", 0.0)
        self.results: list[VisualDowngradeResult] = []
        super().__init__(name=name, options=merged, **kwargs)

    def _requests(self) -> list[VisualRequest]:
        value = self._requests_input
        if value is None:
            value = self._script if self._script is not None else self._beats
        if value is None:
            return []
        return _coerce_requests(value)

    def _resolver_for(self, context: StageContext) -> VisualDowngrade:
        config = self._config if self._config is not None else context.config
        policy = self._policy
        if policy is None:
            budget = self._budget
            if budget is None:
                budget = _config_budget(config)
            policy = VisualDowngradePolicy(
                budget,
                cap_usd=self._cap_usd,
                estimates=self._estimates,
                default_estimate=self._default_estimate,
            )
        return VisualDowngrade(
            self._video,
            image=self._image,
            stock=self._stock,
            router=self._router,
            media=self._media,
            config=config,
            policy=policy,
            notifier=self._notifier,
            fallback_on_failure=self._fallback_on_failure,
        )

    async def run(self, context: StageContext) -> StageResult:
        """Resolve every visual beat and write the downgrade manifest."""
        resolver = self._resolver_for(context)
        run_id = context.run.id if context.run is not None else ""
        results = await resolver.resolve_requests(
            self._requests(),
            context=context,
            run_id=run_id,
        )
        self.results = results

        alerts = [alert for result in results for alert in result.alerts]
        resolved_count = sum(1 for result in results if result.resolved)
        downgraded_count = sum(1 for result in results if result.downgraded)
        total_cost = math.fsum(result.cost_usd for result in results)
        providers = [result.provider for result in results if result.provider]
        models = [result.model for result in results if result.model]
        metadata = {
            "beats": len(results),
            "resolved": resolved_count,
            "unresolved": len(results) - resolved_count,
            "downgraded": downgraded_count,
            "alerts": len(alerts),
            "cost_usd": total_cost,
            "cap_usd": resolver.cap_usd,
            "requested_types": [result.requested_type.value for result in results],
            "chosen_types": [
                result.chosen_type.value if result.chosen_type is not None else ""
                for result in results
            ],
            "providers": list(dict.fromkeys(providers)),
            "models": list(dict.fromkeys(models)),
        }
        payload: dict[str, Any] = {
            "version": 1,
            "stage": self.name,
            "beats": [result.to_dict() for result in results],
            "alerts": [alert.model_dump(mode="json") for alert in alerts],
            "metadata": metadata,
        }
        relative = self.filename
        _write_json(context.run_dir / relative, payload)
        if relative == DEFAULT_DOWNGRADE_FILENAME:
            _write_json(
                context.run_dir / DEFAULT_VISUAL_DOWNGRADE_FILENAME,
                payload,
            )
        return VisualDowngradeStageResult(
            run_id=run_id,
            stage=self.name,
            status=StageState.SUCCEEDED,
            artifacts=[relative],
            cost_usd=total_cost,
            beats=results,
            alerts=alerts,
            metadata={**metadata, "manifest": relative},
        )


visual_downgrade_stage = VisualDowngradeStage
DowngradeStage = VisualDowngradeStage


def build_visual_downgrade_stage(
    *,
    options: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> VisualDowngradeStage:
    """Build a :class:`VisualDowngradeStage` from recipe options and overrides."""
    return VisualDowngradeStage(options=options, **kwargs)


build_downgrade_stage = build_visual_downgrade_stage


async def run_visual_downgrade_stage(
    context: StageContext,
    *,
    stage: VisualDowngradeStage | None = None,
    **options: Any,
) -> StageResult:
    """Build a stage unless one is supplied, then run it asynchronously."""
    if stage is not None and options:
        raise ConfigError("pass either a stage or options, not both")
    active = stage if stage is not None else VisualDowngradeStage(**options)
    return await active.run(context)


run_downgrade_stage = run_visual_downgrade_stage
