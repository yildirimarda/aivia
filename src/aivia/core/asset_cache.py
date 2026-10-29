"""Cache generated assets by the request that produced them.

An asset is reusable only when the provider, model, prompt, generation
parameters and reference media are all identical. :class:`AssetCacheKey`
combines those values into one stable SHA-256 digest and
:class:`AssetCache` stores that digest in the ``asset_cache`` table next to the
asset it produced. A lookup returns the stored asset without calling a provider
and records a zero-amount cost event, so a cache hit is visible in the ledger
while never counting as spend.

The cache never decides whether a call is safe or affordable; it only answers
whether an identical request already produced an asset. The media service owns
screening, budgets, routing and storage.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import Connection, Engine
from sqlalchemy.exc import IntegrityError

from aivia.core.cost import CostTracker
from aivia.core.db.repos import AssetCacheRepository, AssetRepository
from aivia.core.models.asset import Asset, AssetCacheEntry, compute_prompt_hash
from aivia.core.models.base import canonical_json
from aivia.core.models.cost import CostEvent

__all__ = [
    "AssetCache",
    "AssetCacheKey",
    "asset_cache_key",
]

CACHE_HIT_OPERATION = "asset_gen"
"""The default operation recorded for a cache hit."""


@contextmanager
def _write_transaction(engine: Engine) -> Iterator[Connection]:
    """Open a transaction that takes SQLite's write lock before reading.

    SQLite starts a deferred transaction by default, so two stores could both
    see a miss and only then try to insert the same cache key. ``BEGIN
    IMMEDIATE`` serialises stores before the lookup, so the loser re-reads the
    winner's entry instead of colliding on the primary key.
    """
    with engine.begin() as connection:
        if engine.dialect.name == "sqlite":
            connection.exec_driver_sql("BEGIN IMMEDIATE")
        yield connection


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _reference_hashes(values: Sequence[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    normalised = {
        value.strip().lower()
        for value in values
        if isinstance(value, str) and value.strip()
    }
    return tuple(sorted(normalised))


@dataclass(frozen=True, slots=True)
class AssetCacheKey:
    """Everything that makes two asset requests the same request.

    ``prompt`` whitespace is ignored, parameter key order does not matter and
    reference hashes are deduplicated and sorted, so the digest is stable
    across equivalent inputs. Provider and model names are stripped but remain
    case sensitive because provider model identifiers may be case sensitive.
    """

    provider: str
    model: str
    prompt: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    reference_hashes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider", _text(self.provider))
        object.__setattr__(self, "model", _text(self.model))
        object.__setattr__(self, "prompt", _text(self.prompt))
        object.__setattr__(self, "parameters", dict(self.parameters or {}))
        object.__setattr__(
            self,
            "reference_hashes",
            _reference_hashes(self.reference_hashes),
        )

    @property
    def digest(self) -> str:
        """The stable SHA-256 digest used as the stored cache key."""
        payload = {
            "provider": self.provider,
            "model": self.model,
            "prompt": self.prompt,
            "parameters": dict(self.parameters),
            "reference_hashes": list(self.reference_hashes),
        }
        encoded = canonical_json(payload).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def cache_key(self) -> str:
        """Alias for :attr:`digest`."""
        return self.digest

    @property
    def prompt_hash(self) -> str:
        """The prompt-only hash, kept separate from the full cache digest."""
        return compute_prompt_hash(self.prompt, parameters=self.parameters)

    def as_mapping(self) -> dict[str, Any]:
        """Return the key as a JSON-compatible mapping for sidecars and logs."""
        return {
            "provider": self.provider,
            "model": self.model,
            "prompt": self.prompt,
            "parameters": dict(self.parameters),
            "reference_hashes": list(self.reference_hashes),
            "cache_key": self.digest,
        }


def asset_cache_key(
    provider: str,
    model: str,
    prompt: str,
    *,
    parameters: Mapping[str, Any] | None = None,
    reference_hashes: Sequence[str] | None = None,
) -> AssetCacheKey:
    """Build an :class:`AssetCacheKey` from the request values."""
    return AssetCacheKey(
        provider=provider,
        model=model,
        prompt=prompt,
        parameters=parameters or {},
        reference_hashes=tuple(reference_hashes or ()),
    )


class AssetCache:
    """Look up and store assets by their full request digest.

    Pass an SQLAlchemy :class:`~sqlalchemy.Engine` or ready asset and cache
    repositories. The engine form also builds the
    :class:`~aivia.core.cost.CostTracker` used to record zero cost hits. With
    ``no_cache`` set, every lookup misses and no hit is recorded, which lets a
    caller honour a ``--no-cache`` flag without changing its flow.
    """

    def __init__(
        self,
        engine: Engine | None = None,
        *,
        repository: AssetCacheRepository | None = None,
        assets: AssetRepository | None = None,
        cost: CostTracker | None = None,
        no_cache: bool = False,
    ) -> None:
        if engine is not None:
            if repository is not None or assets is not None:
                raise TypeError("pass either an engine or repositories, not both")
            self._cache = AssetCacheRepository(engine)
            self._assets = AssetRepository(engine)
            self._cost = cost if cost is not None else CostTracker(engine)
        else:
            if repository is None or assets is None:
                raise TypeError("AssetCache needs an engine or both asset repositories")
            if cost is None:
                raise TypeError(
                    "AssetCache needs a cost tracker when given repositories"
                )
            self._cache = repository
            self._assets = assets
            self._cost = cost
        self.no_cache = bool(no_cache)

    @property
    def repository(self) -> AssetCacheRepository:
        """The repository holding request digests and their asset ids."""
        return self._cache

    @property
    def assets(self) -> AssetRepository:
        """The repository holding the cached asset records."""
        return self._assets

    @property
    def cost(self) -> CostTracker:
        """The tracker that records zero cost hits."""
        return self._cost

    def key(
        self,
        provider: str,
        model: str,
        prompt: str,
        *,
        parameters: Mapping[str, Any] | None = None,
        reference_hashes: Sequence[str] | None = None,
    ) -> AssetCacheKey:
        """Build the cache key for one request."""
        return asset_cache_key(
            provider,
            model,
            prompt,
            parameters=parameters,
            reference_hashes=reference_hashes,
        )

    def entry(self, key: AssetCacheKey) -> AssetCacheEntry | None:
        """Return the cache entry for *key*, or ``None`` on a miss."""
        if self.no_cache:
            return None
        return self._cache.get(key.digest)

    def get(self, key: AssetCacheKey) -> Asset | None:
        """Return the cached asset for *key*, or ``None`` on a miss."""
        entry = self.entry(key)
        if entry is None:
            return None
        return self._assets.get(entry.asset_id)

    def lookup(
        self,
        key: AssetCacheKey,
        *,
        run_id: str = "",
        operation: str = CACHE_HIT_OPERATION,
        record_hit: bool = True,
    ) -> Asset | None:
        """Return the cached asset and record a zero cost hit when found."""
        entry = self.entry(key)
        if entry is None:
            return None
        asset = self._assets.get(entry.asset_id)
        if asset is None:
            return None
        if record_hit and not self.no_cache:
            self.record_hit(key, asset, run_id=run_id, operation=operation)
        return asset

    def record_hit(
        self,
        key: AssetCacheKey,
        asset: Asset,
        *,
        run_id: str = "",
        operation: str = CACHE_HIT_OPERATION,
    ) -> CostEvent:
        """Record a cache hit and increment the entry's hit counter.

        The cost event is an actual event with a zero amount, so reports show
        the hit without treating it as spend.
        """
        entry = self._cache.get(key.digest)
        if entry is not None:
            self._cache.update(key.digest, hit_count=entry.hit_count + 1)
        return self._cost.record(
            provider=key.provider or asset.provider,
            model=key.model or asset.model,
            operation=operation,
            units=0.0,
            unit_kind="call",
            amount_usd=0.0,
            estimated=False,
            run_id=run_id or asset.run_id,
            metadata={
                "cache": "hit",
                "cache_key": key.digest,
                "asset_id": asset.id,
            },
        )

    def store(
        self,
        asset: Asset,
        *,
        key: AssetCacheKey | None = None,
    ) -> Asset:
        """Store *asset* under *key*, returning an existing record on a hit.

        When *key* is omitted the key is derived from the asset's own provider,
        model and prompt, with no parameters or references. An existing asset
        with the same digest wins, so storing an equivalent result twice does
        not create a duplicate. The asset and its cache entry are written in
        one transaction, so concurrent identical stores create one entry and
        the loser returns the winner's asset without leaving an orphan behind.
        """
        cache_key = key or self.key(asset.provider, asset.model, asset.prompt)
        try:
            with _write_transaction(self._cache.engine) as connection:
                if not self.no_cache:
                    entry = self._cache.get_in(connection, cache_key.digest)
                    if entry is not None:
                        existing = self._assets.get_in(connection, entry.asset_id)
                        if existing is not None:
                            return existing
                stored = self._assets.create_in(connection, asset)
                self._cache.create_in(
                    connection,
                    AssetCacheEntry(
                        cache_key=cache_key.digest,
                        asset_id=stored.id,
                        provider=cache_key.provider,
                        model=cache_key.model,
                        prompt_hash=stored.prompt_hash,
                        parameters=dict(cache_key.parameters),
                        reference_hashes=list(cache_key.reference_hashes),
                    ),
                )
                return stored
        except IntegrityError:
            if self.no_cache:
                raise
            existing = self.get(cache_key)
            if existing is None:
                raise
            return existing

    def put(
        self,
        asset: Asset,
        *,
        key: AssetCacheKey | None = None,
    ) -> Asset:
        """Alias for :meth:`store`."""
        return self.store(asset, key=key)
