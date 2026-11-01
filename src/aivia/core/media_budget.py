"""Apply a per-clip spending cap before a media request is submitted.

A single video or image clip can be priced differently by provider, model,
resolution or duration. :class:`PerClipBudgetPolicy` compares the estimate for
the intended clip with the configured per-clip cap (``budget.per_video`` by
default) and never contacts a provider. When the estimate fits, the caller may
submit it unchanged. When it does not, the policy looks through the supplied
candidates and returns a :class:`CheaperHint` for the cheapest option that is
both cheaper than the current one and still inside the cap. If no candidate
qualifies it raises :class:`~aivia.core.errors.BudgetError` before submit, so an
over-budget clip cannot reach a provider.

The policy is deliberately independent of a concrete capability. Callers pass
the current option and a mapping of alternative names to estimates, or richer
:class:`ClipOption` values that name the provider, model, resolution and
duration. A cap of ``0`` means the cap is not configured and every clip is
allowed. :meth:`PerClipBudgetPolicy.evaluate` reports a blocked decision
without raising, while :meth:`PerClipBudgetPolicy.check` enforces the policy.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from aivia.core.config.budget import BudgetConfig
from aivia.core.errors import BudgetError, ConfigError

__all__ = [
    "CheaperHint",
    "ClipBudget",
    "ClipBudgetAction",
    "ClipBudgetCheck",
    "ClipBudgetDecision",
    "ClipBudgetPolicy",
    "ClipOption",
    "MediaBudgetPolicy",
    "PerClipBudget",
    "PerClipBudgetPolicy",
    "apply_clip_budget",
    "check_clip_budget",
    "decide_clip_budget",
    "enforce_clip_budget",
    "ensure_clip_budget",
    "evaluate_clip_budget",
    "resolve_clip_budget",
]


def _exceeds(amount: float, cap: float) -> bool:
    """Compare money without rejecting an amount that is only float noise over."""
    return amount > cap and not math.isclose(
        amount,
        cap,
        rel_tol=1e-9,
        abs_tol=1e-9,
    )


def _coerce_amount(value: object, *, label: str) -> float:
    """Return *value* as a finite, non-negative amount."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(
            f"{label} must be a finite, non-negative number",
            details={label: value},
        )
    amount = float(value)
    if not math.isfinite(amount) or amount < 0:
        raise ConfigError(
            f"{label} must be a finite, non-negative number",
            details={label: value},
        )
    return amount


def _coerce_name(value: object, *, label: str) -> str:
    """Return *value* as a non-empty, stripped name."""
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(
            f"{label} must be a non-empty string",
            details={label: value},
        )
    return value.strip()


def _coerce_optional_name(value: object, *, label: str) -> str:
    """Return *value* as a stripped name, allowing the empty string."""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ConfigError(
            f"{label} must be a string",
            details={label: value},
        )
    return value.strip()


@dataclass(frozen=True, slots=True)
class ClipOption:
    """One way to produce the clip, with its estimate and identifying fields.

    ``name`` is the label used in hints and defaults to ``provider`` or
    ``provider:model``. Two options are the same when their names match, or
    when their provider, model, resolution and duration all match, so a
    cheaper model or resolution from the same provider is still a valid hint.
    """

    provider: str
    estimate_usd: float
    model: str = ""
    resolution: str = ""
    duration_seconds: float | None = None
    name: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        provider = _coerce_name(self.provider, label="provider")
        model = _coerce_optional_name(self.model, label="model")
        resolution = _coerce_optional_name(self.resolution, label="resolution")
        name = _coerce_optional_name(self.name, label="name")
        if not name:
            name = f"{provider}:{model}" if model else provider
        duration = (
            None
            if self.duration_seconds is None
            else _coerce_amount(self.duration_seconds, label="duration_seconds")
        )
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "resolution", resolution)
        object.__setattr__(self, "name", name)
        object.__setattr__(
            self,
            "estimate_usd",
            _coerce_amount(self.estimate_usd, label="estimate_usd"),
        )
        object.__setattr__(self, "duration_seconds", duration)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def label(self) -> str:
        """Alias for :attr:`name`."""
        return self.name

    @property
    def amount_usd(self) -> float:
        """Alias for :attr:`estimate_usd`."""
        return self.estimate_usd

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation for logs and manifests."""
        payload: dict[str, object] = {
            "name": self.name,
            "provider": self.provider,
            "model": self.model,
            "estimate_usd": self.estimate_usd,
        }
        if self.resolution:
            payload["resolution"] = self.resolution
        if self.duration_seconds is not None:
            payload["duration_seconds"] = self.duration_seconds
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


CandidateEntry = tuple[str, float] | ClipOption
Candidates = Mapping[str, float] | Iterable[CandidateEntry] | None


@dataclass(frozen=True, slots=True)
class CheaperHint:
    """A cheaper option that still fits the per-clip cap."""

    option: ClipOption
    current_provider: str = ""
    current_model: str = ""
    current_estimate_usd: float = 0.0
    cap_usd: float = 0.0

    @property
    def provider(self) -> str:
        """The suggested provider."""
        return self.option.provider

    @property
    def model(self) -> str:
        """The suggested model."""
        return self.option.model

    @property
    def name(self) -> str:
        """The suggested option name."""
        return self.option.name

    @property
    def resolution(self) -> str:
        """The suggested resolution, when the option names one."""
        return self.option.resolution

    @property
    def duration_seconds(self) -> float | None:
        """The suggested clip duration, when the option names one."""
        return self.option.duration_seconds

    @property
    def estimate_usd(self) -> float:
        """The cheaper estimate."""
        return self.option.estimate_usd

    @property
    def amount_usd(self) -> float:
        """Alias for :attr:`estimate_usd`."""
        return self.estimate_usd

    @property
    def savings_usd(self) -> float:
        """How much the hint saves against the current estimate."""
        return self.current_estimate_usd - self.option.estimate_usd

    @property
    def message(self) -> str:
        """A short human-readable description of the hint."""
        return (
            f"use {self.option.name} at {self.option.estimate_usd:.6g} USD "
            f"instead of {self.current_provider or '<unknown>'} at "
            f"{self.current_estimate_usd:.6g} USD; saves "
            f"{self.savings_usd:.6g} USD and fits the per-clip cap of "
            f"{self.cap_usd:.6g} USD"
        )

    @property
    def reason(self) -> str:
        """Alias for :attr:`message`."""
        return self.message

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation for logs and manifests."""
        return {
            "name": self.option.name,
            "provider": self.option.provider,
            "model": self.option.model,
            "resolution": self.option.resolution,
            "duration_seconds": self.option.duration_seconds,
            "estimate_usd": self.estimate_usd,
            "savings_usd": self.savings_usd,
            "current_provider": self.current_provider,
            "current_model": self.current_model,
            "current_estimate_usd": self.current_estimate_usd,
            "cap_usd": self.cap_usd,
            "message": self.message,
        }


class ClipBudgetAction(StrEnum):
    """What the caller should do after evaluating one clip."""

    PROCEED = "proceed"
    DOWNGRADE = "downgrade"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class ClipBudgetDecision:
    """The result of comparing one clip estimate with the per-clip cap.

    ``PROCEED`` allows the original option, ``DOWNGRADE`` carries a cheaper
    :class:`CheaperHint` the caller should use instead, and ``BLOCKED`` means
    the clip is over budget and no cheaper option fits. ``evaluate`` returns a
    blocked decision while ``check`` raises instead.
    """

    cap_usd: float
    estimate_usd: float
    provider: str = ""
    model: str = ""
    action: ClipBudgetAction = ClipBudgetAction.PROCEED
    hint: CheaperHint | None = None
    message: str = ""

    @property
    def allowed(self) -> bool:
        """Whether some option may be submitted."""
        return self.action in {
            ClipBudgetAction.PROCEED,
            ClipBudgetAction.DOWNGRADE,
        }

    @property
    def blocked(self) -> bool:
        """Whether the clip must not be submitted."""
        return self.action is ClipBudgetAction.BLOCKED

    @property
    def exceeded(self) -> bool:
        """Whether the original estimate exceeded the cap."""
        return self.action is not ClipBudgetAction.PROCEED

    @property
    def downgraded(self) -> bool:
        """Whether a cheaper option was suggested."""
        return self.action is ClipBudgetAction.DOWNGRADE

    @property
    def cheaper(self) -> bool:
        """Whether a cheaper hint is available."""
        return self.hint is not None

    @property
    def needs_hint(self) -> bool:
        """Alias for :attr:`cheaper`."""
        return self.cheaper

    @property
    def can_submit(self) -> bool:
        """Whether the original option fits the cap unchanged."""
        return self.action is ClipBudgetAction.PROCEED

    @property
    def reason(self) -> str:
        """Alias for :attr:`message`."""
        return self.message

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation for logs and manifests."""
        return {
            "action": self.action.value,
            "allowed": self.allowed,
            "blocked": self.blocked,
            "exceeded": self.exceeded,
            "cap_usd": self.cap_usd,
            "estimate_usd": self.estimate_usd,
            "provider": self.provider,
            "model": self.model,
            "hint": self.hint.to_dict() if self.hint is not None else None,
            "message": self.message,
        }


ClipBudgetCheck = ClipBudgetDecision


def _coerce_candidates(value: Candidates) -> list[ClipOption]:
    """Validate candidate options while preserving their order."""
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [
            ClipOption(
                provider=_coerce_name(name, label="candidate name"),
                estimate_usd=_coerce_amount(
                    estimate,
                    label="candidate estimate",
                ),
                name=_coerce_name(name, label="candidate name"),
            )
            for name, estimate in value.items()
        ]
    try:
        entries = list(value)
    except TypeError as exc:
        raise ConfigError(
            "candidates must be a mapping or an iterable of options",
            details={"candidates": type(value).__name__},
        ) from exc
    options: list[ClipOption] = []
    for entry in entries:
        if isinstance(entry, ClipOption):
            options.append(entry)
            continue
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise ConfigError(
                "candidates must be (name, estimate) pairs or ClipOption values",
                details={"candidate": entry},
            )
        name, estimate = entry
        label = _coerce_name(name, label="candidate name")
        options.append(
            ClipOption(
                provider=label,
                estimate_usd=_coerce_amount(estimate, label="candidate estimate"),
                name=label,
            )
        )
    return options


def _is_same_option(
    option: ClipOption,
    *,
    name: str,
    provider: str,
    model: str,
    resolution: str,
    duration_seconds: float | None,
) -> bool:
    """Return whether *option* describes the current clip."""
    if name and option.name == name:
        return True
    return (
        option.provider == provider
        and option.model == model
        and option.resolution == resolution
        and option.duration_seconds == duration_seconds
    )


class PerClipBudgetPolicy:
    """Enforce the configured per-clip cap and suggest a cheaper option.

    Pass a :class:`~aivia.core.config.budget.BudgetConfig` to read
    ``per_video`` as the cap, or an explicit ``per_clip`` (also called ``cap``
    or ``per_video``) to override it. A cap of ``0`` disables the policy.
    """

    def __init__(
        self,
        config: BudgetConfig | Mapping[str, Any] | float | int | None = None,
        *,
        per_clip: float | None = None,
        cap: float | None = None,
        per_video: float | None = None,
    ) -> None:
        explicit = [value for value in (per_clip, cap, per_video) if value is not None]
        if len(explicit) > 1:
            raise TypeError("pass only one of per_clip, cap or per_video")
        selected = explicit[0] if explicit else None
        if isinstance(config, bool):
            raise TypeError("config must be a BudgetConfig, mapping, number or None")
        if isinstance(config, (int, float)):
            if selected is not None:
                raise TypeError("pass the cap as a config or an argument, not both")
            selected = config
            config = None
        if isinstance(config, Mapping):
            config = BudgetConfig.from_mapping(config)
        if config is not None and not isinstance(config, BudgetConfig):
            raise TypeError("config must be a BudgetConfig, mapping, number or None")
        self._config = config
        base = config.per_video if config is not None else 0.0
        self._cap = _coerce_amount(
            base if selected is None else selected,
            label="per_clip_cap",
        )

    @property
    def config(self) -> BudgetConfig | None:
        """The budget configuration the cap came from, when one was supplied."""
        return self._config

    @property
    def cap(self) -> float:
        """The per-clip cap; ``0`` means the policy is disabled."""
        return self._cap

    @property
    def per_clip(self) -> float:
        """Alias for :attr:`cap`."""
        return self._cap

    @property
    def per_clip_cap(self) -> float:
        """Alias for :attr:`cap`."""
        return self._cap

    @property
    def per_video(self) -> float:
        """Alias for :attr:`cap`."""
        return self._cap

    def evaluate(
        self,
        amount_usd: float,
        *,
        provider: str = "",
        model: str = "",
        name: str = "",
        resolution: str = "",
        duration_seconds: float | None = None,
        candidates: Candidates = None,
        options: Candidates = None,
    ) -> ClipBudgetDecision:
        """Evaluate one clip without raising on an over-budget estimate."""
        amount = _coerce_amount(amount_usd, label="amount_usd")
        current_provider = _coerce_optional_name(provider, label="provider")
        current_model = _coerce_optional_name(model, label="model")
        current_resolution = _coerce_optional_name(resolution, label="resolution")
        current_duration = (
            None
            if duration_seconds is None
            else _coerce_amount(duration_seconds, label="duration_seconds")
        )
        current_name = _coerce_optional_name(name, label="name")
        if not current_name:
            current_name = (
                f"{current_provider}:{current_model}"
                if current_model
                else current_provider
            )
        selected = candidates if candidates is not None else options
        if not self._cap or not _exceeds(amount, self._cap):
            return ClipBudgetDecision(
                cap_usd=self._cap,
                estimate_usd=amount,
                provider=current_provider,
                model=current_model,
                action=ClipBudgetAction.PROCEED,
                message="clip is within the per-clip budget",
            )

        affordable: list[tuple[float, int, ClipOption]] = []
        for index, option in enumerate(_coerce_candidates(selected)):
            if _is_same_option(
                option,
                name=current_name,
                provider=current_provider,
                model=current_model,
                resolution=current_resolution,
                duration_seconds=current_duration,
            ):
                continue
            if not option.estimate_usd < amount or math.isclose(
                option.estimate_usd,
                amount,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                continue
            if _exceeds(option.estimate_usd, self._cap):
                continue
            affordable.append((option.estimate_usd, index, option))

        if not affordable:
            return ClipBudgetDecision(
                cap_usd=self._cap,
                estimate_usd=amount,
                provider=current_provider,
                model=current_model,
                action=ClipBudgetAction.BLOCKED,
                message=(
                    f"estimated clip cost {amount:.6g} USD exceeds the "
                    f"per-clip cap of {self._cap:.6g} USD and no cheaper "
                    "option fits"
                ),
            )

        _, _, option = min(affordable, key=lambda item: (item[0], item[1]))
        hint = CheaperHint(
            option=option,
            current_provider=current_provider,
            current_model=current_model,
            current_estimate_usd=amount,
            cap_usd=self._cap,
        )
        return ClipBudgetDecision(
            cap_usd=self._cap,
            estimate_usd=amount,
            provider=current_provider,
            model=current_model,
            action=ClipBudgetAction.DOWNGRADE,
            hint=hint,
            message=hint.message,
        )

    def check(
        self,
        amount_usd: float,
        *,
        provider: str = "",
        model: str = "",
        name: str = "",
        resolution: str = "",
        duration_seconds: float | None = None,
        candidates: Candidates = None,
        options: Candidates = None,
    ) -> ClipBudgetDecision:
        """Return a cheaper hint or raise before the clip is submitted."""
        decision = self.evaluate(
            amount_usd,
            provider=provider,
            model=model,
            name=name,
            resolution=resolution,
            duration_seconds=duration_seconds,
            candidates=candidates,
            options=options,
        )
        if decision.blocked:
            raise BudgetError(
                decision.message,
                details={
                    "scope": "per_clip",
                    "cap": decision.cap_usd,
                    "per_clip_cap": decision.cap_usd,
                    "amount_usd": decision.estimate_usd,
                    "requested": decision.estimate_usd,
                    "provider": decision.provider,
                    "model": decision.model,
                },
            )
        return decision

    def hint(
        self,
        amount_usd: float,
        **kwargs: Any,
    ) -> CheaperHint | None:
        """Return only the cheaper hint, or ``None`` when the clip fits."""
        return self.check(amount_usd, **kwargs).hint

    def decide(self, amount_usd: float, **kwargs: Any) -> ClipBudgetDecision:
        """Alias for :meth:`check`."""
        return self.check(amount_usd, **kwargs)

    def apply(self, amount_usd: float, **kwargs: Any) -> ClipBudgetDecision:
        """Alias for :meth:`check`."""
        return self.check(amount_usd, **kwargs)

    def enforce(self, amount_usd: float, **kwargs: Any) -> ClipBudgetDecision:
        """Alias for :meth:`check`."""
        return self.check(amount_usd, **kwargs)

    def ensure(self, amount_usd: float, **kwargs: Any) -> ClipBudgetDecision:
        """Alias for :meth:`check`."""
        return self.check(amount_usd, **kwargs)

    def require(self, amount_usd: float, **kwargs: Any) -> ClipBudgetDecision:
        """Alias for :meth:`check`."""
        return self.check(amount_usd, **kwargs)

    def before_submit(self, amount_usd: float, **kwargs: Any) -> ClipBudgetDecision:
        """Alias for :meth:`check`, named for the point it must be called."""
        return self.check(amount_usd, **kwargs)


ClipBudgetPolicy = PerClipBudgetPolicy
ClipBudget = PerClipBudgetPolicy
MediaBudgetPolicy = PerClipBudgetPolicy
PerClipBudget = PerClipBudgetPolicy


def evaluate_clip_budget(
    amount_usd: float,
    *,
    config: BudgetConfig | Mapping[str, Any] | float | int | None = None,
    per_clip: float | None = None,
    cap: float | None = None,
    per_video: float | None = None,
    candidates: Candidates = None,
    options: Candidates = None,
    **kwargs: Any,
) -> ClipBudgetDecision:
    """Evaluate one clip without constructing a long-lived policy."""
    policy = PerClipBudgetPolicy(
        config,
        per_clip=per_clip,
        cap=cap,
        per_video=per_video,
    )
    return policy.evaluate(
        amount_usd,
        candidates=candidates,
        options=options,
        **kwargs,
    )


def check_clip_budget(
    amount_usd: float,
    *,
    config: BudgetConfig | Mapping[str, Any] | float | int | None = None,
    per_clip: float | None = None,
    cap: float | None = None,
    per_video: float | None = None,
    candidates: Candidates = None,
    options: Candidates = None,
    **kwargs: Any,
) -> ClipBudgetDecision:
    """Return a cheaper hint or raise when no cheaper option fits."""
    policy = PerClipBudgetPolicy(
        config,
        per_clip=per_clip,
        cap=cap,
        per_video=per_video,
    )
    return policy.check(
        amount_usd,
        candidates=candidates,
        options=options,
        **kwargs,
    )


def decide_clip_budget(
    amount_usd: float,
    **kwargs: Any,
) -> ClipBudgetDecision:
    """Alias for :func:`check_clip_budget`."""
    return check_clip_budget(amount_usd, **kwargs)


def apply_clip_budget(
    amount_usd: float,
    **kwargs: Any,
) -> ClipBudgetDecision:
    """Alias for :func:`check_clip_budget`."""
    return check_clip_budget(amount_usd, **kwargs)


def resolve_clip_budget(
    amount_usd: float,
    **kwargs: Any,
) -> ClipBudgetDecision:
    """Alias for :func:`check_clip_budget`."""
    return check_clip_budget(amount_usd, **kwargs)


def enforce_clip_budget(
    amount_usd: float,
    **kwargs: Any,
) -> ClipBudgetDecision:
    """Alias for :func:`check_clip_budget`."""
    return check_clip_budget(amount_usd, **kwargs)


def ensure_clip_budget(
    amount_usd: float,
    **kwargs: Any,
) -> ClipBudgetDecision:
    """Alias for :func:`check_clip_budget`."""
    return check_clip_budget(amount_usd, **kwargs)
