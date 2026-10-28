"""Provider selection for a capability chain.

The router owns the policy that decides which provider serves a capability;
adapters only implement the capability itself. A capability names an ordered
``chain`` of provider instances and a ``strategy``. Selection always filters on
the required capabilities first, so a provider that cannot serve the request is
never chosen regardless of its position in the chain.

This module implements five strategies. ``first_available`` walks the chain in
order and returns the first provider that is open and declares every required
capability. ``cheapest`` compares the price of the remaining candidates and
uses chain order as the tie breaker. A price configured in
``pricing_overrides`` wins over the provider's own metadata, and a provider
without any price sorts after every priced candidate so missing data never
looks free. ``best_quality`` picks the highest static ``quality_rank`` until
human ratings exist, warning about any candidate whose rank is missing and
sorting those candidates after every ranked one. ``round_robin`` rotates
through the available providers per capability and is inherently
deterministic. ``weighted_ab`` draws from the configured per-provider weights;
a seed makes the draw reproducible, and the selected weight is recorded on the
selection. Every strategy can be budget aware: when a remaining budget is
given, a provider whose estimated price for the requested units exceeds it is
skipped as ``budget_exceeded`` before the strategy picks a candidate.

:meth:`Router.execute` runs an operation against the selected provider and
falls back to the next candidate when the provider reports a retryable,
timeout or rate limit error. Permanent errors, including authentication and
budget failures, are surfaced immediately because another provider cannot fix
them. When every candidate fails with a retryable error the router raises
:class:`~aivia.core.errors.ProviderExhaustedError` describing each attempt.

Each provider has its own :class:`~aivia.core.circuit.CircuitBreaker`. Retryable
failures are recorded per provider, and a provider whose circuit is open is
skipped until its cooldown elapses. Selection then half-opens the circuit for a
single probe, which either closes it on success or reopens it on failure.

Each provider also has its own :class:`~aivia.core.ratelimit.TokenBucket`, built
from the rate limit declared in its metadata. Execution acquires a token before
calling the provider; a bucket configured to wait sleeps for the next token,
while a fail-fast bucket makes the router skip or fall back to another provider.
When the metadata declares ``max_concurrency``, the same limiter also holds a
slot for the duration of the call, so parallel calls queue or fail instead of
exceeding the provider's cap.
"""

from __future__ import annotations

import math
import random
import warnings
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TypeVar

from aivia.core.circuit import CircuitBreaker
from aivia.core.config.capability import CapabilityChainConfig, CapabilityStrategy
from aivia.core.config.circuit import CircuitBreakerConfig
from aivia.core.config.ratelimit import OnRateLimit, RateLimiterConfig
from aivia.core.errors import (
    AiviaError,
    ConfigError,
    ProviderExhaustedError,
    RateLimitError,
)
from aivia.core.ratelimit import TokenBucket
from aivia.providers.base import Provider

__all__ = [
    "MissingQualityRankWarning",
    "ProviderRouter",
    "RouteAttempt",
    "RouteSelection",
    "RouteSkip",
    "Router",
    "RouterSelection",
    "UnpricedProviderWarning",
]

T = TypeVar("T")


class MissingQualityRankWarning(UserWarning):
    """A provider lacks the static quality rank the strategy needs to compare it."""


class UnpricedProviderWarning(UserWarning):
    """A provider lacks a price, so budget-aware selection cannot estimate it."""


@dataclass(frozen=True, slots=True)
class RouteSkip:
    """One provider the router rejected while walking a chain.

    ``reason`` is a stable short code (``closed`` or ``missing_capabilities``)
    and ``missing`` lists the capabilities the provider did not declare.
    """

    provider: str
    reason: str
    missing: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation for logs and errors."""
        payload: dict[str, object] = {
            "provider": self.provider,
            "reason": self.reason,
        }
        if self.missing:
            payload["missing"] = list(self.missing)
        return payload


@dataclass(frozen=True, slots=True)
class RouteAttempt:
    """One provider that ran the operation and failed with a retryable error.

    ``error`` is the stable :class:`~aivia.core.errors.AiviaError` code and
    ``error_type`` is the concrete exception name, so a report can explain why
    the router moved on without exposing a provider response body.
    """

    provider: str
    error: str
    error_type: str
    message: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation for logs and errors."""
        return {
            "provider": self.provider,
            "error": self.error,
            "error_type": self.error_type,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class RouteSelection:
    """The outcome of one successful selection.

    The selection records the strategy, the chain that was walked and every
    provider that was skipped, so a run can explain why it used a provider.
    """

    provider: Provider
    capability: str
    strategy: CapabilityStrategy
    required: tuple[str, ...]
    chain: tuple[str, ...] = field(default_factory=tuple)
    skipped: tuple[RouteSkip, ...] = field(default_factory=tuple)
    seed: int | None = None
    weight: float | None = None
    quality_rank: int | None = None

    @property
    def name(self) -> str:
        """The selected provider name."""
        return self.provider.name


RouterSelection = RouteSelection


def _provider_name(provider: Provider) -> str:
    name = provider.name.strip()
    if not name:
        raise ConfigError(
            "provider instance must have a non-empty name",
            details={"provider": type(provider).__name__},
        )
    return name


def _coerce_providers(
    providers: Mapping[str, Provider] | Iterable[Provider] | None,
) -> dict[str, Provider]:
    if providers is None:
        return {}
    if isinstance(providers, Mapping):
        entries: Iterable[tuple[str, Provider]] = providers.items()
    else:
        entries = ((_provider_name(provider), provider) for provider in providers)

    pool: dict[str, Provider] = {}
    for key, provider in entries:
        if not isinstance(provider, Provider):
            raise TypeError(
                f"providers must be Provider instances, not {type(provider).__name__}"
            )
        name = key.strip() if isinstance(key, str) else ""
        if not name:
            raise ConfigError("provider names must be non-empty strings")
        if name in pool:
            raise ConfigError(
                f"duplicate provider name: {name}",
                details={"provider": name},
            )
        pool[name] = provider
    return pool


def _coerce_required(
    capability: str, required: Iterable[str] | str | None
) -> tuple[str, ...]:
    if not isinstance(capability, str) or not capability.strip():
        raise ConfigError("capability must be a non-empty string")
    names: list[str] = [capability.strip()]
    if required is None:
        return tuple(names)
    entries = (required,) if isinstance(required, str) else required
    for entry in entries:
        if not isinstance(entry, str) or not entry.strip():
            raise ConfigError("required capabilities must be non-empty strings")
        name = entry.strip()
        if name not in names:
            names.append(name)
    return tuple(names)


def _price_for(
    name: str,
    provider: Provider,
    capability: str,
    overrides: Mapping[str, float],
) -> float | None:
    """Return the per-unit price of *capability* for *provider*, if known.

    *name* is the configured provider name, which may differ from the metadata
    name when the same adapter serves several instances. An explicit override
    wins over metadata because it is how an operator corrects a stale or
    missing upstream price. Metadata prices are normalised to one unit so a
    price expressed per thousand units compares fairly with a price expressed
    per single call.
    """
    override = overrides.get(name)
    if override is not None:
        return override
    price = provider.metadata.pricing_for(capability)
    if price is None:
        return None
    return price.cost_for(1.0)


def _coerce_remaining_budget(value: float | None) -> float | None:
    """Validate the remaining budget, where ``None`` disables the check."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(
            "remaining budget must be a finite number",
            details={"remaining_budget": value},
        )
    budget = float(value)
    if not math.isfinite(budget):
        raise ConfigError(
            "remaining budget must be a finite number",
            details={"remaining_budget": value},
        )
    return budget


def _coerce_units(value: float) -> float:
    """Validate the number of units a request will be charged for."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(
            "units must be a finite, non-negative number",
            details={"units": value},
        )
    units = float(value)
    if not math.isfinite(units) or units < 0:
        raise ConfigError(
            "units must be a finite, non-negative number",
            details={"units": value},
        )
    return units


def _budget_rejection(
    name: str,
    provider: Provider,
    capability: str,
    overrides: Mapping[str, float],
    remaining_budget: float | None,
    units: float,
) -> RouteSkip | None:
    """Return why *provider* exceeds the remaining budget, if it does.

    The estimate is the provider price for *capability* scaled to *units*. A
    provider without a known price cannot be estimated, so it is left to the
    strategy rather than being rejected on missing data, but a
    :class:`UnpricedProviderWarning` is emitted so the gap is visible.
    """
    if remaining_budget is None:
        return None
    price = _price_for(name, provider, capability, overrides)
    if price is None:
        warnings.warn(
            f"provider {name!r} has no price for {capability!r}; "
            "budget-aware selection cannot estimate it",
            UnpricedProviderWarning,
            stacklevel=3,
        )
        return None
    if price * units > remaining_budget:
        return RouteSkip(provider=name, reason="budget_exceeded")
    return None


def _quality_rank_for(name: str, provider: Provider) -> int | None:
    """Return the static quality rank of *provider*, warning when it is absent.

    ``best_quality`` compares ranks until human ratings exist. A provider
    without a rank cannot be compared, so it is reported once per selection
    and treated as worse than every ranked candidate. ``name`` is the
    configured provider name, which may differ from the metadata name.
    """
    rank = provider.metadata.quality_rank
    if rank is None:
        warnings.warn(
            f"provider {name!r} is missing a quality rank; "
            "best_quality treats it as unranked",
            MissingQualityRankWarning,
            stacklevel=3,
        )
    return rank


def _rejection(
    name: str,
    provider: Provider,
    required: tuple[str, ...],
    breaker: CircuitBreaker | None = None,
    limiter: TokenBucket | None = None,
) -> RouteSkip | None:
    """Return why *provider* cannot serve *required*, or ``None`` if it can."""
    if provider.closed:
        return RouteSkip(provider=name, reason="closed")
    if breaker is not None and not breaker.available():
        return RouteSkip(provider=name, reason="circuit_open")
    if (
        limiter is not None
        and limiter.on_limit is OnRateLimit.RAISE
        and not limiter.available()
    ):
        return RouteSkip(provider=name, reason="rate_limited")
    missing = tuple(
        requirement for requirement in required if not provider.supports(requirement)
    )
    if missing:
        return RouteSkip(
            provider=name,
            reason="missing_capabilities",
            missing=missing,
        )
    return None


def _exhausted(
    target: str,
    required: tuple[str, ...],
    strategy: CapabilityStrategy,
    chain: tuple[str, ...],
    skipped: list[RouteSkip],
) -> ProviderExhaustedError:
    return ProviderExhaustedError(
        f"no available provider for capability: {target}",
        details={
            "capability": target,
            "required": list(required),
            "strategy": strategy.value,
            "chain": list(chain),
            "skipped": [entry.to_dict() for entry in skipped],
        },
    )


def _weighted_choice(
    available: list[tuple[str, Provider]],
    weights: Mapping[str, float],
    rng: random.Random,
) -> tuple[str, Provider, float]:
    """Pick one available provider using *weights* and *rng*.

    Providers without a positive weight are not eligible. When every available
    provider has a zero or missing weight the configuration cannot express a
    choice, so fail clearly instead of silently ignoring the strategy.
    """
    weighted = [
        (name, provider, weights.get(name, 0.0))
        for name, provider in available
        if weights.get(name, 0.0) > 0
    ]
    if not weighted:
        raise ConfigError(
            "weighted_ab strategy has no positive weight for an available provider",
            details={
                "available": [name for name, _ in available],
                "weights": dict(weights),
            },
        )

    total = sum(weight for _, _, weight in weighted)
    threshold = rng.random() * total
    cumulative = 0.0
    for name, provider, weight in weighted:
        cumulative += weight
        if threshold < cumulative:
            return name, provider, weight
    name, provider, weight = weighted[-1]
    return name, provider, weight


def _coerce_strategy(value: CapabilityStrategy | str | None) -> CapabilityStrategy:
    if value is None:
        return CapabilityStrategy.FIRST_AVAILABLE
    if isinstance(value, CapabilityStrategy):
        return value
    try:
        return CapabilityStrategy(value.strip())
    except (AttributeError, ValueError):
        allowed = ", ".join(strategy.value for strategy in CapabilityStrategy)
        raise ConfigError(
            f"unknown router strategy: {value!r} (allowed: {allowed})",
            details={
                "strategy": value,
                "allowed": [s.value for s in CapabilityStrategy],
            },
        ) from None


class Router:
    """Select a provider for a capability from an ordered chain.

    ``providers`` is a mapping from configured provider name to instance or an
    iterable of instances whose metadata name is used. ``configs`` maps a
    capability to its :class:`CapabilityChainConfig`; ``config`` is the default
    used for capabilities without their own entry. ``strategy``, ``chain`` and
    ``weights`` override that default for callers that do not build a config
    model. ``seed`` makes weighted draws reproducible. An empty chain means
    every provider in insertion order, which keeps a small configuration usable
    without repeating provider names.

    Each provider gets its own circuit breaker. ``circuit_breakers`` injects
    pre-built breakers by provider name, while ``circuit_breaker_factory``
    builds the remaining ones; the default is a fresh
    :class:`~aivia.core.circuit.CircuitBreaker` using ``circuit_config``.

    Each provider also gets its own token bucket built from its declared
    :class:`~aivia.core.models.provider.RateLimit`. ``rate_limiters`` injects
    pre-built buckets by provider name, while ``rate_limiter_factory`` builds
    the remaining ones; the default bucket uses ``rate_limit_config``.
    """

    def __init__(
        self,
        providers: Mapping[str, Provider] | Iterable[Provider] | None = None,
        config: CapabilityChainConfig
        | Mapping[str, CapabilityChainConfig]
        | None = None,
        *,
        configs: Mapping[str, CapabilityChainConfig] | None = None,
        strategy: CapabilityStrategy | str | None = None,
        chain: Iterable[str] | None = None,
        weights: Mapping[str, float] | None = None,
        seed: int | None = None,
        circuit_breaker_factory: Callable[[str], CircuitBreaker] | None = None,
        circuit_breakers: Mapping[str, CircuitBreaker] | None = None,
        circuit_config: CircuitBreakerConfig | None = None,
        rate_limiter_factory: Callable[..., TokenBucket] | None = None,
        rate_limiters: Mapping[str, TokenBucket] | None = None,
        rate_limit_config: RateLimiterConfig | None = None,
    ) -> None:
        if circuit_config is not None and not isinstance(
            circuit_config, CircuitBreakerConfig
        ):
            raise TypeError("circuit_config must be a CircuitBreakerConfig")
        if rate_limit_config is not None and not isinstance(
            rate_limit_config, RateLimiterConfig
        ):
            raise TypeError("rate_limit_config must be a RateLimiterConfig")
        self._providers = _coerce_providers(providers)
        self._circuits = self._build_circuits(
            circuit_breaker_factory,
            circuit_breakers,
            circuit_config or CircuitBreakerConfig(),
        )
        self._limiters = self._build_limiters(
            rate_limiter_factory,
            rate_limiters,
            rate_limit_config or RateLimiterConfig(),
        )
        default = (
            config
            if isinstance(config, CapabilityChainConfig)
            else CapabilityChainConfig()
        )
        if strategy is not None or chain is not None or weights is not None:
            values = default.model_dump()
            if strategy is not None:
                values["strategy"] = _coerce_strategy(strategy)
            if chain is not None:
                values["chain"] = list(chain)
            if weights is not None:
                values["weights"] = dict(weights)
            default = CapabilityChainConfig.model_validate(values)
        self._default_config = default
        merged: dict[str, CapabilityChainConfig] = {}
        if isinstance(config, Mapping):
            merged.update(config)
        if configs is not None:
            merged.update(configs)
        normalised: dict[str, CapabilityChainConfig] = {}
        for capability, capability_config in merged.items():
            if not isinstance(capability, str) or not capability.strip():
                raise ConfigError("capability names must be non-empty strings")
            if not isinstance(capability_config, CapabilityChainConfig):
                raise TypeError(
                    "capability configs must be CapabilityChainConfig instances"
                )
            normalised[capability.strip()] = capability_config
        self._configs = normalised
        self._last_selection: RouteSelection | None = None
        self._seed = seed
        self._rng = random.Random(seed)
        self._round_robin: dict[str, int] = {}

    def _build_circuits(
        self,
        factory: Callable[[str], CircuitBreaker] | None,
        injected: Mapping[str, CircuitBreaker] | None,
        config: CircuitBreakerConfig,
    ) -> dict[str, CircuitBreaker]:
        if factory is not None and not callable(factory):
            raise TypeError("circuit_breaker_factory must be callable")
        supplied: dict[str, CircuitBreaker] = {}
        if injected is not None:
            for name, breaker in injected.items():
                if not isinstance(name, str) or not name.strip():
                    raise ConfigError("circuit breaker names must be non-empty strings")
                if not isinstance(breaker, CircuitBreaker):
                    raise TypeError(
                        "circuit_breakers values must be CircuitBreaker instances"
                    )
                supplied[name.strip()] = breaker
            unknown = sorted(set(supplied) - set(self._providers))
            if unknown:
                names = ", ".join(unknown)
                raise ConfigError(
                    f"circuit breakers name unknown providers: {names}",
                    details={
                        "unknown": unknown,
                        "available": list(self._providers),
                    },
                )

        circuits: dict[str, CircuitBreaker] = {}
        for name in self._providers:
            if name in supplied:
                circuits[name] = supplied[name]
            elif factory is not None:
                breaker = factory(name)
                if not isinstance(breaker, CircuitBreaker):
                    raise TypeError(
                        "circuit_breaker_factory must return CircuitBreaker instances"
                    )
                circuits[name] = breaker
            else:
                circuits[name] = CircuitBreaker(config=config)
        return circuits

    def _build_limiters(
        self,
        factory: Callable[..., TokenBucket] | None,
        injected: Mapping[str, TokenBucket] | None,
        config: RateLimiterConfig,
    ) -> dict[str, TokenBucket]:
        if factory is not None and not callable(factory):
            raise TypeError("rate_limiter_factory must be callable")
        supplied: dict[str, TokenBucket] = {}
        if injected is not None:
            for name, limiter in injected.items():
                if not isinstance(name, str) or not name.strip():
                    raise ConfigError("rate limiter names must be non-empty strings")
                if not isinstance(limiter, TokenBucket):
                    raise TypeError(
                        "rate_limiters values must be TokenBucket instances"
                    )
                supplied[name.strip()] = limiter
            unknown = sorted(set(supplied) - set(self._providers))
            if unknown:
                names = ", ".join(unknown)
                raise ConfigError(
                    f"rate limiters name unknown providers: {names}",
                    details={
                        "unknown": unknown,
                        "available": list(self._providers),
                    },
                )

        limiters: dict[str, TokenBucket] = {}
        for name, provider in self._providers.items():
            if name in supplied:
                limiters[name] = supplied[name]
            elif factory is not None:
                limiter = self._call_limiter_factory(factory, name, provider)
                if not isinstance(limiter, TokenBucket):
                    raise TypeError(
                        "rate_limiter_factory must return TokenBucket instances"
                    )
                limiters[name] = limiter
            else:
                limiters[name] = TokenBucket(
                    rate_limit=provider.metadata.rate_limit,
                    name=name,
                    config=config,
                )
        return limiters

    @staticmethod
    def _call_limiter_factory(
        factory: Callable[..., TokenBucket],
        name: str,
        provider: Provider,
    ) -> TokenBucket:
        """Call a factory that accepts either ``(name)`` or ``(name, provider)``."""
        try:
            return factory(name, provider)
        except TypeError:
            return factory(name)

    @property
    def providers(self) -> Mapping[str, Provider]:
        """The configured provider instances keyed by name."""
        return dict(self._providers)

    @property
    def circuits(self) -> Mapping[str, CircuitBreaker]:
        """The circuit breaker for every configured provider, keyed by name."""
        return dict(self._circuits)

    @property
    def breakers(self) -> Mapping[str, CircuitBreaker]:
        """Alias for :attr:`circuits`."""
        return self.circuits

    def circuit_for(self, provider: str) -> CircuitBreaker:
        """Return the circuit breaker belonging to *provider*."""
        if not isinstance(provider, str) or not provider.strip():
            raise ConfigError("provider name must be a non-empty string")
        name = provider.strip()
        try:
            return self._circuits[name]
        except KeyError:
            raise ConfigError(
                f"unknown provider for circuit breaker: {name}",
                details={"provider": name, "available": list(self._circuits)},
            ) from None

    def breaker_for(self, provider: str) -> CircuitBreaker:
        """Alias for :meth:`circuit_for`."""
        return self.circuit_for(provider)

    @property
    def rate_limiters(self) -> Mapping[str, TokenBucket]:
        """The token bucket for every configured provider, keyed by name."""
        return dict(self._limiters)

    @property
    def limiters(self) -> Mapping[str, TokenBucket]:
        """Alias for :attr:`rate_limiters`."""
        return self.rate_limiters

    def limiter_for(self, provider: str) -> TokenBucket:
        """Return the token bucket belonging to *provider*."""
        if not isinstance(provider, str) or not provider.strip():
            raise ConfigError("provider name must be a non-empty string")
        name = provider.strip()
        try:
            return self._limiters[name]
        except KeyError:
            raise ConfigError(
                f"unknown provider for rate limiter: {name}",
                details={"provider": name, "available": list(self._limiters)},
            ) from None

    def rate_limiter_for(self, provider: str) -> TokenBucket:
        """Alias for :meth:`limiter_for`."""
        return self.limiter_for(provider)

    @property
    def provider_names(self) -> tuple[str, ...]:
        """Every configured provider name in insertion order."""
        return tuple(self._providers)

    @property
    def last_selection(self) -> RouteSelection | None:
        """The most recent successful selection, if any."""
        return self._last_selection

    @property
    def seed(self) -> int | None:
        """The seed used for weighted selection, if one was configured."""
        return self._seed

    def config_for(self, capability: str) -> CapabilityChainConfig:
        """Return the chain configuration for *capability*."""
        if not isinstance(capability, str) or not capability.strip():
            raise ConfigError("capability must be a non-empty string")
        return self._configs.get(capability.strip(), self._default_config)

    def chain_for(self, capability: str) -> tuple[str, ...]:
        """Return the configured chain for *capability* in order.

        An empty chain expands to every configured provider in insertion order.
        A chain entry that is not a configured provider is a configuration
        error, because a typo must not silently select a different provider.
        """
        config = self.config_for(capability)
        if not config.chain:
            return self.provider_names
        unknown = [name for name in config.chain if name not in self._providers]
        if unknown:
            names = ", ".join(unknown)
            raise ConfigError(
                f"capability {capability!r} chain names unknown providers: {names}",
                details={
                    "capability": capability,
                    "unknown": unknown,
                    "available": list(self.provider_names),
                },
            )
        return tuple(config.chain)

    def candidates(
        self,
        capability: str,
        *,
        required: Iterable[str] | str | None = None,
    ) -> tuple[Provider, ...]:
        """Return every open provider in the chain supporting all requirements."""
        names = _coerce_required(capability, required)
        return tuple(
            self._providers[provider_name]
            for provider_name in self.chain_for(capability)
            if not self._providers[provider_name].closed
            and self._circuits[provider_name].available()
            and not (
                self._limiters[provider_name].on_limit is OnRateLimit.RAISE
                and not self._limiters[provider_name].available()
            )
            and all(
                self._providers[provider_name].supports(requirement)
                for requirement in names
            )
        )

    def available(
        self,
        capability: str,
        *,
        required: Iterable[str] | str | None = None,
    ) -> tuple[Provider, ...]:
        """Alias for :meth:`candidates`."""
        return self.candidates(capability, required=required)

    def select(
        self,
        capability: str,
        *,
        required: Iterable[str] | str | None = None,
        strategy: CapabilityStrategy | str | None = None,
        seed: int | None = None,
        remaining_budget: float | None = None,
        units: float = 1.0,
    ) -> Provider:
        """Return the provider selected for *capability*.

        The capability itself is always required; *required* adds capabilities
        the request additionally needs, for example ``image_edit`` alongside
        ``image_gen``. ``strategy`` overrides the configured strategy for this
        call. ``seed`` overrides the router seed for one weighted draw.

        ``remaining_budget`` is the amount still available for this call. A
        provider whose estimated price for *units* exceeds it is skipped, so
        the strategy picks the best affordable candidate. ``None`` disables the
        budget check. A provider without a known price cannot be estimated, so
        it is never skipped, but :class:`UnpricedProviderWarning` is emitted.
        """
        return self.select_with_record(
            capability,
            required=required,
            strategy=strategy,
            seed=seed,
            remaining_budget=remaining_budget,
            units=units,
        ).provider

    def select_provider(
        self,
        capability: str,
        *,
        required: Iterable[str] | str | None = None,
        strategy: CapabilityStrategy | str | None = None,
        seed: int | None = None,
        remaining_budget: float | None = None,
        units: float = 1.0,
    ) -> Provider:
        """Alias for :meth:`select`."""
        return self.select(
            capability,
            required=required,
            strategy=strategy,
            seed=seed,
            remaining_budget=remaining_budget,
            units=units,
        )

    def _round_robin_pick(
        self,
        capability: str,
        chain: tuple[str, ...],
        available: list[tuple[str, Provider]],
    ) -> Provider:
        """Return the next available provider in the chain and advance it."""
        available_by_name = dict(available)
        cursor = self._round_robin.get(capability, 0) % len(chain)
        for offset in range(len(chain)):
            index = (cursor + offset) % len(chain)
            name = chain[index]
            if name in available_by_name:
                self._round_robin[capability] = (index + 1) % len(chain)
                return available_by_name[name]
        raise AssertionError("available providers must come from the chain")

    def select_with_record(
        self,
        capability: str,
        *,
        required: Iterable[str] | str | None = None,
        strategy: CapabilityStrategy | str | None = None,
        seed: int | None = None,
        exclude: Iterable[str] | None = None,
        remaining_budget: float | None = None,
        units: float = 1.0,
    ) -> RouteSelection:
        """Select a provider and return the full selection record.

        ``exclude`` names providers that must not be selected, which is how
        :meth:`execute` moves on after a retryable failure. Excluded providers
        are reported as skipped with the ``excluded`` reason.

        ``remaining_budget`` and ``units`` make selection budget aware: a
        provider whose estimated price for *units* is greater than the
        remaining budget is reported as skipped with ``budget_exceeded``. An
        unpriced provider cannot be estimated, so it is never skipped, but
        :class:`UnpricedProviderWarning` is emitted.
        """
        names = _coerce_required(capability, required)
        target = names[0]
        excluded = frozenset(exclude or ())
        budget = _coerce_remaining_budget(remaining_budget)
        quantity = _coerce_units(units)
        configured = self.config_for(target)
        overrides = configured.pricing_overrides
        chosen_strategy = (
            configured.strategy if strategy is None else _coerce_strategy(strategy)
        )
        implemented = (
            CapabilityStrategy.FIRST_AVAILABLE,
            CapabilityStrategy.CHEAPEST,
            CapabilityStrategy.BEST_QUALITY,
            CapabilityStrategy.ROUND_ROBIN,
            CapabilityStrategy.WEIGHTED_AB,
        )
        if chosen_strategy not in implemented:
            raise ConfigError(
                f"router strategy not implemented: {chosen_strategy.value}",
                details={
                    "strategy": chosen_strategy.value,
                    "implemented": [entry.value for entry in implemented],
                },
            )

        effective_seed = self._seed if seed is None else seed
        chain = self.chain_for(target)
        skipped: list[RouteSkip] = []
        if chosen_strategy is CapabilityStrategy.FIRST_AVAILABLE:
            for name in chain:
                provider = self._providers[name]
                if name in excluded:
                    skipped.append(RouteSkip(provider=name, reason="excluded"))
                    continue
                rejection = _rejection(
                    name,
                    provider,
                    names,
                    self._circuits[name],
                    self._limiters[name],
                )
                if rejection is None:
                    rejection = _budget_rejection(
                        name,
                        provider,
                        target,
                        overrides,
                        budget,
                        quantity,
                    )
                if rejection is not None:
                    skipped.append(rejection)
                    continue
                selection = RouteSelection(
                    provider=provider,
                    capability=target,
                    strategy=chosen_strategy,
                    required=names,
                    chain=chain,
                    skipped=tuple(skipped),
                    seed=effective_seed,
                )
                self._last_selection = selection
                return selection
            raise _exhausted(target, names, chosen_strategy, chain, skipped)

        available: list[tuple[str, Provider]] = []
        for name in chain:
            provider = self._providers[name]
            if name in excluded:
                skipped.append(RouteSkip(provider=name, reason="excluded"))
                continue
            rejection = _rejection(
                name,
                provider,
                names,
                limiter=self._limiters[name],
            )
            if rejection is None:
                rejection = _budget_rejection(
                    name,
                    provider,
                    target,
                    overrides,
                    budget,
                    quantity,
                )
            if rejection is not None:
                skipped.append(rejection)
                continue
            available.append((name, provider))

        if not available:
            raise _exhausted(target, names, chosen_strategy, chain, skipped)

        weight: float | None = None
        quality_rank: int | None = None
        if chosen_strategy is CapabilityStrategy.CHEAPEST:

            def price_key(entry: tuple[str, Provider]) -> tuple[bool, float]:
                name, provider = entry
                price = _price_for(name, provider, target, overrides)
                return (price is None, price if price is not None else 0.0)

            _, chosen = min(available, key=price_key)
        elif chosen_strategy is CapabilityStrategy.BEST_QUALITY:

            def rank_key(entry: tuple[str, Provider]) -> tuple[bool, int]:
                name, provider = entry
                rank = _quality_rank_for(name, provider)
                return (rank is not None, rank if rank is not None else 0)

            _, chosen = max(available, key=rank_key)
            quality_rank = chosen.metadata.quality_rank
        elif chosen_strategy is CapabilityStrategy.ROUND_ROBIN:
            chosen = self._round_robin_pick(target, chain, available)
        else:
            unknown = sorted(set(configured.weights) - set(chain))
            if unknown:
                names_text = ", ".join(unknown)
                raise ConfigError(
                    "weighted_ab weights reference providers not in the chain: "
                    f"{names_text}",
                    details={
                        "capability": target,
                        "unknown": unknown,
                        "chain": list(chain),
                    },
                )
            rng = random.Random(seed) if seed is not None else self._rng
            _, chosen, weight = _weighted_choice(available, configured.weights, rng)

        selection = RouteSelection(
            provider=chosen,
            capability=target,
            strategy=chosen_strategy,
            required=names,
            chain=chain,
            skipped=tuple(skipped),
            seed=effective_seed,
            weight=weight,
            quality_rank=quality_rank,
        )
        self._last_selection = selection
        return selection

    async def execute(
        self,
        capability: str,
        operation: Callable[[Provider], Awaitable[T]],
        *,
        required: Iterable[str] | str | None = None,
        strategy: CapabilityStrategy | str | None = None,
        seed: int | None = None,
        remaining_budget: float | None = None,
        units: float = 1.0,
    ) -> T:
        """Run *operation* with fallback across the capability chain.

        The router selects a provider with the configured strategy, calls
        ``operation(provider)`` and returns its result. A retryable error
        (including timeout and rate limit failures) excludes that provider and
        selects again, so the next candidate gets a turn. A permanent error is
        raised immediately because retrying the same request on another
        provider cannot fix an invalid request, bad credentials, a budget
        refusal or a content policy rejection.

        ``remaining_budget`` and ``units`` are passed to every selection so a
        provider whose estimate exceeds the budget is never called. Retryable
        failures are recorded on the provider's circuit breaker; a provider
        whose circuit is open is skipped until it half-opens. Before the call
        the provider's token bucket is acquired, which waits for the next token
        or raises a rate limited error according to the bucket policy. The same
        acquisition holds a concurrency slot until the operation finishes, so a
        provider's ``max_concurrency`` is never exceeded. When every candidate
        fails with a retryable error, the router raises
        :class:`~aivia.core.errors.ProviderExhaustedError` with the attempted
        providers and the last error, chained as the cause. The successful
        selection remains available through :attr:`last_selection`.
        """
        if not callable(operation):
            raise TypeError("operation must be callable")

        names = _coerce_required(capability, required)
        target = names[0]
        attempted: list[RouteAttempt] = []
        excluded: set[str] = set()
        last_error: AiviaError | None = None

        while True:
            try:
                selection = self.select_with_record(
                    capability,
                    required=names,
                    strategy=strategy,
                    seed=seed,
                    exclude=excluded,
                    remaining_budget=remaining_budget,
                    units=units,
                )
            except ProviderExhaustedError as exhausted:
                if last_error is None:
                    raise
                details = dict(exhausted.details)
                details["attempted"] = [entry.to_dict() for entry in attempted]
                details["last_error"] = last_error.to_dict()
                raise ProviderExhaustedError(
                    f"no available provider for capability: {target} "
                    f"after {len(attempted)} failed attempt(s)",
                    details=details,
                ) from last_error

            breaker = self._circuits[selection.name]
            if not breaker.acquire():
                excluded.add(selection.name)
                continue

            limiter = self._limiters[selection.name]
            try:
                await limiter.acquire()
            except RateLimitError as exc:
                breaker.release()
                attempted.append(
                    RouteAttempt(
                        provider=selection.name,
                        error=exc.code,
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                )
                excluded.add(selection.name)
                last_error = exc
                continue

            try:
                result = await operation(selection.provider)
            except AiviaError as exc:
                limiter.release()
                if not exc.retryable:
                    breaker.release()
                    raise
                attempted.append(
                    RouteAttempt(
                        provider=selection.name,
                        error=exc.code,
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                )
                excluded.add(selection.name)
                breaker.record_failure()
                last_error = exc
            except BaseException:
                limiter.release()
                breaker.release()
                raise
            else:
                limiter.release()
                breaker.record_success()
                return result


ProviderRouter = Router
