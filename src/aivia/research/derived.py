"""Persist brief derivations so they are only regenerated when a brief changes.

The derivation itself lives in :mod:`aivia.research.brief`; this module adds a
small JSON store keyed by profile name. Each stored entry carries the digest of
the brief it was derived from, so a profile whose brief is unchanged reuses its
settings without another model call. Explicit keywords are applied after the
lookup, which means changing a keyword updates the effective settings without
invalidating the derivation.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from aivia.core import paths
from aivia.core.config.profile import ContentProfileConfig
from aivia.core.errors import ConfigError
from aivia.core.models.base import canonical_json
from aivia.research.brief import (
    BriefDerivation,
    BriefDerivationCache,
    _as_profile,
    brief_hash,
    derive_brief,
)

__all__ = [
    "BriefDerivationStore",
    "PersistentBriefDerivationCache",
    "ProfileDerivationStore",
    "derivation_store_path",
    "ensure_derivation",
    "ensure_derivation_sync",
    "load_derivations",
    "save_derivations",
]

_STORE_VERSION = 1
_STORE_FILENAME = "derived.json"


def derivation_store_path(
    data_dir: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Return the default persistent derivation store path."""
    root = paths.data_dir(environ) if data_dir is None else Path(data_dir)
    return Path(root).expanduser() / "research" / _STORE_FILENAME


def _profile_name(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError("profile name must be a non-empty string")
    return value.strip()


def _read_store(path: Path) -> dict[str, BriefDerivation]:
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read derivation store: {path}") from exc
    if not text.strip():
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid derivation store JSON: {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ConfigError(f"derivation store root must be a mapping: {path}")
    entries: object = payload.get("profiles", payload)
    if not isinstance(entries, Mapping):
        raise ConfigError(f"derivation store profiles must be a mapping: {path}")
    derivations: dict[str, BriefDerivation] = {}
    for name, value in entries.items():
        if not isinstance(name, str) or not isinstance(value, Mapping):
            raise ConfigError(f"invalid derivation entry in {path}: {name!r}")
        try:
            derivations[name] = BriefDerivation.model_validate(dict(value))
        except Exception as exc:
            raise ConfigError(
                f"invalid derivation for profile {name!r} in {path}: {exc}"
            ) from exc
    return derivations


def _write_store(path: Path, entries: Mapping[str, BriefDerivation]) -> None:
    payload = {
        "version": _STORE_VERSION,
        "profiles": {
            name: value.model_dump(mode="json")
            for name, value in sorted(entries.items())
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(canonical_json(payload, indent=2))
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def load_derivations(path: str | Path) -> dict[str, BriefDerivation]:
    """Read every persisted derivation from *path*."""
    return _read_store(Path(path).expanduser())


def save_derivations(
    path: str | Path,
    entries: Mapping[str, BriefDerivation],
) -> None:
    """Write *entries* to *path* deterministically and atomically."""
    validated = {_profile_name(name): value for name, value in entries.items()}
    _write_store(Path(path).expanduser(), validated)


class BriefDerivationStore:
    """A persistent mapping of profile name to its last brief derivation."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        data_dir: str | Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self._path = (
            derivation_store_path(data_dir, environ=environ)
            if path is None
            else Path(path).expanduser()
        )

    @property
    def path(self) -> Path:
        """The JSON file backing this store."""
        return self._path

    def load(self) -> dict[str, BriefDerivation]:
        """Return every stored derivation keyed by profile name."""
        return _read_store(self._path)

    def get(self, profile: str) -> BriefDerivation | None:
        """Return the derivation stored for *profile*, or ``None``."""
        value = _read_store(self._path).get(_profile_name(profile))
        return None if value is None else value.model_copy(deep=True)

    def get_for_brief(self, profile: str, brief: str) -> BriefDerivation | None:
        """Return the derivation for *profile* only when *brief* is unchanged."""
        value = self.get(profile)
        if value is None or value.brief_hash != brief_hash(brief):
            return None
        return value

    def store(self, profile: str, value: BriefDerivation) -> BriefDerivation:
        """Persist *value* under *profile* and return it unchanged."""
        if not isinstance(value, BriefDerivation):
            raise TypeError("stored value must be a BriefDerivation")
        entries = _read_store(self._path)
        entries[_profile_name(profile)] = value.model_copy(deep=True)
        _write_store(self._path, entries)
        return value

    def put(self, profile: str, value: BriefDerivation) -> BriefDerivation:
        """Alias for :meth:`store`."""
        return self.store(profile, value)

    def remove(self, profile: str) -> bool:
        """Remove *profile* from the store, returning whether it existed."""
        entries = _read_store(self._path)
        removed = entries.pop(_profile_name(profile), None)
        if removed is not None:
            _write_store(self._path, entries)
        return removed is not None

    def clear(self) -> None:
        """Remove every persisted derivation."""
        _write_store(self._path, {})

    def __contains__(self, profile: object) -> bool:
        return isinstance(profile, str) and profile.strip() in self.load()

    def __len__(self) -> int:
        return len(self.load())


PersistentBriefDerivationCache = BriefDerivationStore
ProfileDerivationStore = BriefDerivationStore


def _without_keywords(
    derivation: BriefDerivation,
    keywords: Sequence[str],
) -> BriefDerivation:
    keys = {keyword.casefold() for keyword in keywords}
    if not keys:
        return derivation
    return derivation.model_copy(
        update={
            "queries": [
                term for term in derivation.queries if term.casefold() not in keys
            ],
            "include_terms": [
                term for term in derivation.include_terms if term.casefold() not in keys
            ],
            "exclude_terms": [
                term for term in derivation.exclude_terms if term.casefold() not in keys
            ],
            "keywords": [],
        }
    )


def _apply_keywords(
    derivation: BriefDerivation,
    keywords: Sequence[str],
) -> BriefDerivation:
    base = _without_keywords(derivation, derivation.keywords)
    return base.with_explicit_keywords(keywords)


async def ensure_derivation(
    profile_name: str,
    profile: ContentProfileConfig | Mapping[str, Any] | str,
    *,
    store: BriefDerivationStore | None = None,
    cache: BriefDerivationCache | None = None,
    deriver: Any | None = None,
    force: bool = False,
    no_cache: bool = False,
    **values: Any,
) -> BriefDerivation:
    """Return a persisted derivation, regenerating it only when needed.

    A stored entry is reused when its ``brief_hash`` matches the current brief.
    Explicit keywords are reapplied on every call, so keyword changes never
    trigger a model call. Pass ``force=True`` (or ``--no-cache`` at the CLI) to
    regenerate regardless of the stored entry.
    """
    config = _as_profile(profile)
    active_store = store if store is not None else BriefDerivationStore()
    digest = brief_hash(config.brief)

    if not force:
        stored = active_store.get(profile_name)
        if stored is not None and stored.brief_hash == digest:
            effective = _apply_keywords(stored, config.keywords)
            return effective.model_copy(
                update={
                    "brief": config.brief,
                    "brief_hash": digest,
                    "cache_hit": True,
                }
            )

    active_cache = (
        cache if cache is not None else BriefDerivationCache(no_cache=no_cache)
    )
    if deriver is not None:
        result = await deriver.derive(
            config,
            cache=active_cache,
            no_cache=no_cache,
            **values,
        )
    else:
        result = await derive_brief(
            config,
            cache=active_cache,
            no_cache=no_cache,
            **values,
        )
    raw = None if no_cache else active_cache.get_for_brief(config.brief)
    if raw is None:
        raw = _without_keywords(result, result.keywords)
    active_store.store(profile_name, raw)
    return _apply_keywords(raw, config.keywords).model_copy(
        update={
            "brief": config.brief,
            "brief_hash": digest,
            "cache_hit": result.cache_hit,
        }
    )


def ensure_derivation_sync(
    profile_name: str,
    profile: ContentProfileConfig | Mapping[str, Any] | str,
    **values: Any,
) -> BriefDerivation:
    """Blocking wrapper around :func:`ensure_derivation`."""
    import asyncio

    return asyncio.run(ensure_derivation(profile_name, profile, **values))
