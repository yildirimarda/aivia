"""Provider selection for a capability chain.

The router owns the policy that decides which provider serves a capability;
adapters only implement the capability itself. A capability names an ordered
``chain`` of provider instances and a ``strategy``. Selection always filters on
the required capabilities first, so a provider that cannot serve the request is
never chosen regardless of its position in the chain.

This module implements two strategies. ``first_available`` walks the chain in
order and returns the first provider that is open and declares every required
capability. ``cheapest`` compares the price of the remaining candidates and
uses chain order as the tie breaker. A price configured in
``pricing_overrides`` wins over the provider's own metadata, and a provider
without any price sorts after every priced candidate so missing data never
looks free. The other strategies are added later; asking for one now fails with
a clear configuration error instead of silently falling back to the wrong
policy.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from aivia.core.config.capability import CapabilityChainConfig, CapabilityStrategy
from aivia.core.errors import ConfigError, ProviderExhaustedError
from aivia.providers.base import Provider

__all__ = [
    "ProviderRouter",
    "RouteSelection",
    "RouteSkip",
    "Router",
    "RouterSelection",
]


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


def _rejection(
    name: str,
    provider: Provider,
    required: tuple[str, ...],
) -> RouteSkip | None:
    """Return why *provider* cannot serve *required*, or ``None`` if it can."""
    if provider.closed:
        return RouteSkip(provider=name, reason="closed")
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
    used for capabilities without their own entry. ``strategy`` and ``chain``
    override that default for callers that do not build a config model. An
    empty chain means every provider in insertion order, which keeps a small
    configuration usable without repeating provider names.
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
    ) -> None:
        self._providers = _coerce_providers(providers)
        default = (
            config
            if isinstance(config, CapabilityChainConfig)
            else CapabilityChainConfig()
        )
        if strategy is not None or chain is not None:
            values = default.model_dump()
            if strategy is not None:
                values["strategy"] = _coerce_strategy(strategy)
            if chain is not None:
                values["chain"] = list(chain)
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

    @property
    def providers(self) -> Mapping[str, Provider]:
        """The configured provider instances keyed by name."""
        return dict(self._providers)

    @property
    def provider_names(self) -> tuple[str, ...]:
        """Every configured provider name in insertion order."""
        return tuple(self._providers)

    @property
    def last_selection(self) -> RouteSelection | None:
        """The most recent successful selection, if any."""
        return self._last_selection

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
    ) -> Provider:
        """Return the first available provider for *capability*.

        The capability itself is always required; *required* adds capabilities
        the request additionally needs, for example ``image_edit`` alongside
        ``image_gen``. ``strategy`` overrides the configured strategy for this
        call.
        """
        return self.select_with_record(
            capability,
            required=required,
            strategy=strategy,
        ).provider

    def select_provider(
        self,
        capability: str,
        *,
        required: Iterable[str] | str | None = None,
        strategy: CapabilityStrategy | str | None = None,
    ) -> Provider:
        """Alias for :meth:`select`."""
        return self.select(capability, required=required, strategy=strategy)

    def select_with_record(
        self,
        capability: str,
        *,
        required: Iterable[str] | str | None = None,
        strategy: CapabilityStrategy | str | None = None,
    ) -> RouteSelection:
        """Select a provider and return the full selection record."""
        names = _coerce_required(capability, required)
        target = names[0]
        configured = self.config_for(target)
        chosen_strategy = (
            configured.strategy if strategy is None else _coerce_strategy(strategy)
        )
        if chosen_strategy not in (
            CapabilityStrategy.FIRST_AVAILABLE,
            CapabilityStrategy.CHEAPEST,
        ):
            raise ConfigError(
                f"router strategy not implemented: {chosen_strategy.value}",
                details={
                    "strategy": chosen_strategy.value,
                    "implemented": [
                        CapabilityStrategy.FIRST_AVAILABLE.value,
                        CapabilityStrategy.CHEAPEST.value,
                    ],
                },
            )

        chain = self.chain_for(target)
        skipped: list[RouteSkip] = []
        if chosen_strategy is CapabilityStrategy.FIRST_AVAILABLE:
            for name in chain:
                provider = self._providers[name]
                rejection = _rejection(name, provider, names)
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
                )
                self._last_selection = selection
                return selection
            raise _exhausted(target, names, chosen_strategy, chain, skipped)

        available: list[tuple[str, Provider]] = []
        for name in chain:
            provider = self._providers[name]
            rejection = _rejection(name, provider, names)
            if rejection is not None:
                skipped.append(rejection)
                continue
            available.append((name, provider))

        if not available:
            raise _exhausted(target, names, chosen_strategy, chain, skipped)

        overrides = configured.pricing_overrides

        def price_key(entry: tuple[str, Provider]) -> tuple[bool, float]:
            name, provider = entry
            price = _price_for(name, provider, target, overrides)
            return (price is None, price if price is not None else 0.0)

        _, chosen = min(available, key=price_key)
        selection = RouteSelection(
            provider=chosen,
            capability=target,
            strategy=chosen_strategy,
            required=names,
            chain=chain,
            skipped=tuple(skipped),
        )
        self._last_selection = selection
        return selection


ProviderRouter = Router
