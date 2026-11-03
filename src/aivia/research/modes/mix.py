"""Combine topic modes by weight into one deterministic candidate stream.

A profile may name several topic modes with a share for each. The mix runs the
modes concurrently, then interleaves their candidates by the configured
weights. Every candidate keeps the name of the mode that found it, so later
stages can explain where a topic came from.

The interleaving is deterministic: the same modes, weights and seed always
produce the same order. The seed only breaks ties between modes that are
equally due, which keeps the weighted shares stable while still allowing two
runs to vary their starting mode. Candidates keep the order their own mode
reported, and an optional limit stops the stream early.
"""

from __future__ import annotations

import asyncio
import math
import random
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from aivia.core.errors import ConfigError
from aivia.core.models.topic import TopicCandidate
from aivia.providers.protocols.source import SourceRequest, SourceResponse
from aivia.research.modes.base import (
    TopicModeBase,
    topic_mode_registry,
)
from aivia.research.modes.otd import normalize_title

__all__ = [
    "DEFAULT_CONTINUE_ON_ERROR",
    "DEFAULT_DEDUPE",
    "DEFAULT_LIMIT",
    "DEFAULT_MIX_SEED",
    "DEFAULT_MIX_WEIGHT",
    "DEFAULT_MODE_MIX_SEED",
    "DEFAULT_MODE_WEIGHT",
    "DEFAULT_SEED",
    "DEFAULT_WEIGHT",
    "Combine",
    "CombinedMode",
    "CombiningMode",
    "Mix",
    "MixMode",
    "MixModeProvider",
    "MixResult",
    "MixSourceMode",
    "MixTopicMode",
    "MixedMode",
    "MixedModeProvider",
    "MixedTopicMode",
    "ModeEntry",
    "ModeMix",
    "ModeMixEntry",
    "ModeMixMode",
    "ModeMixProvider",
    "ModeMixResult",
    "ModeMixSourceMode",
    "ModeMixTopicMode",
    "ModeMixer",
    "ModeShare",
    "ModesMix",
    "TopicModeMix",
    "WeightedMix",
    "WeightedMixResult",
    "WeightedMode",
    "WeightedModeMix",
    "WeightedTopicMode",
    "combine",
    "combine_candidates",
    "combine_modes",
    "interleave_modes",
    "mix",
    "mix_candidates",
    "mix_modes",
    "normalise_mode_weights",
    "normalise_weights",
    "normalize_mode_weights",
    "normalize_weights",
    "seeded_order",
    "weighted_combine",
    "weighted_interleave",
    "weighted_mix",
    "weighted_order",
]

DEFAULT_SEED = 0
"""The seed used when a mix is created without one, keeping runs repeatable."""

DEFAULT_MIX_SEED = DEFAULT_SEED
"""Alias for :data:`DEFAULT_SEED`."""

DEFAULT_MODE_MIX_SEED = DEFAULT_SEED
"""Alias for :data:`DEFAULT_SEED`."""

DEFAULT_MODE_SHARE = 1.0
"""The relative share of a mode named without one."""

DEFAULT_WEIGHT = DEFAULT_MODE_SHARE
"""The relative weight of a mode named without one."""

DEFAULT_MODE_WEIGHT = DEFAULT_MODE_SHARE
"""Alias for :data:`DEFAULT_WEIGHT`."""

DEFAULT_MIX_WEIGHT = DEFAULT_MODE_SHARE
"""Alias for :data:`DEFAULT_WEIGHT`."""

DEFAULT_REQUEST_SIZE = 20
"""The request size used by the one-call helpers when none is supplied."""

DEFAULT_LIMIT: int | None = None
"""No limit is applied unless the caller sets one."""

DEFAULT_DEDUPE = False
"""Whether duplicate titles across modes are dropped by default."""

DEFAULT_CONTINUE_ON_ERROR = False
"""Whether a failing mode is skipped by default."""


@dataclass(frozen=True, slots=True)
class ModeEntry:
    """One mode in a mix with the relative weight it receives."""

    name: str
    mode: Any
    weight: float = DEFAULT_WEIGHT

    @property
    def provider(self) -> Any:
        """Alias for :attr:`mode`."""
        return self.mode

    @property
    def instance(self) -> Any:
        """Alias for :attr:`mode`."""
        return self.mode

    @property
    def topic_mode(self) -> Any:
        """Alias for :attr:`mode`."""
        return self.mode


ModeShare = ModeEntry
ModeMixEntry = ModeEntry
WeightedMode = ModeEntry


class MixResult(SourceResponse):
    """A source response that also reads like the candidate list it carries."""

    @property
    def mode_counts(self) -> dict[str, int]:
        """The number of served candidates from each mode."""
        counts: dict[str, int] = {}
        for candidate in self.candidates:
            counts[candidate.mode] = counts.get(candidate.mode, 0) + 1
        return counts

    def __iter__(self) -> Iterator[TopicCandidate]:  # type: ignore[override]
        return iter(self.candidates)

    def __len__(self) -> int:
        return len(self.candidates)

    def __getitem__(self, index: int | slice) -> TopicCandidate | list[TopicCandidate]:
        return self.candidates[index]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (list, tuple)):
            return list(self.candidates) == list(other)
        return super().__eq__(other)


def _validate_mode_name(name: object) -> str:
    if not isinstance(name, str):
        raise TypeError("topic mode name must be a string")
    stripped = name.strip()
    if not stripped:
        raise ConfigError("topic mode name must not be empty")
    if any(character.isspace() for character in stripped):
        raise ConfigError(f"invalid topic mode name: {name!r}")
    return stripped


def _validate_weight(name: str, weight: object) -> float:
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        raise ConfigError(f"topic mode weight for {name!r} must be a number")
    number = float(weight)
    if not math.isfinite(number):
        raise ConfigError(f"topic mode weight for {name!r} must be finite")
    if number < 0:
        raise ConfigError(f"topic mode weight for {name!r} must not be negative")
    return number


def _validate_seed(seed: object) -> int:
    if seed is None:
        return DEFAULT_SEED
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigError("topic mode mix seed must be an integer")
    return seed


def _validate_limit(limit: object) -> int | None:
    if limit is None:
        return None
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ConfigError("topic mode mix limit must be an integer")
    maximum = limit
    if maximum < 0:
        raise ConfigError("topic mode mix limit must not be negative")
    return maximum


def _is_mode_object(value: object) -> bool:
    return not isinstance(value, type) and callable(getattr(value, "discover", None))


def _mode_name(value: object) -> str:
    name = getattr(value, "name", None)
    if not isinstance(name, str) or not name.strip():
        raise ConfigError("a topic mode must declare a name")
    return _validate_mode_name(name)


def _instantiate(mode: type[Any]) -> Any:
    try:
        return mode()
    except TypeError as exc:
        raise ConfigError(
            f"cannot instantiate topic mode {mode.__name__}: {exc}",
            details={"mode": mode.__name__},
        ) from exc


def _resolve_mode(value: Any, registry: Any) -> Any:
    if isinstance(value, TopicModeBase):
        return value
    if isinstance(value, str):
        active = registry if registry is not None else topic_mode_registry
        return _instantiate(active.resolve(value))
    if isinstance(value, type):
        return _instantiate(value)
    if _is_mode_object(value):
        return value
    if callable(value):
        return _resolve_mode(value(), registry)
    raise ConfigError(
        "topic mode must be a name, class, instance or factory, "
        f"not {type(value).__name__}"
    )


def _profile_weights(profile: Any) -> dict[str, float] | None:
    if profile is None:
        return None
    for key in (
        "topic_modes",
        "modes",
        "topic_mode",
        "mode_weights",
        "topic_weights",
    ):
        if isinstance(profile, Mapping):
            value = profile.get(key)
        else:
            value = getattr(profile, key, None)
        if value is not None:
            if not isinstance(value, Mapping):
                raise ConfigError("topic mode weights must be a mapping")
            return {str(name): float(weight) for name, weight in value.items()}
    return None


def _parse_mapping(
    modes: Mapping[Any, Any],
) -> list[tuple[str, Any, float | None]]:
    entries: list[tuple[str, Any, float | None]] = []
    for raw_name, value in modes.items():
        name = _validate_mode_name(raw_name)
        weight: float | None = None
        target = value
        if isinstance(value, Mapping) and not _is_mode_object(value):
            target = value.get(
                "mode",
                value.get(
                    "provider",
                    value.get("instance", value.get("class", value.get("name", name))),
                ),
            )
            weight_value = value.get("weight")
            if weight_value is not None:
                weight = _validate_weight(name, weight_value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            target = name
            weight = _validate_weight(name, value)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            target, raw_weight = value
            weight = _validate_weight(name, raw_weight)
        elif value is None:
            target = name
        entries.append((name, target, weight))
    return entries


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _parse_pair(
    item: Sequence[Any],
) -> tuple[str, Any, float | None]:
    if len(item) == 2:
        first, second = item
        if isinstance(first, str):
            name = _validate_mode_name(first)
            if _is_number(second):
                return name, name, _validate_weight(name, second)
            return name, second, None
        name = _mode_name(first)
        return name, first, _validate_weight(name, second)
    if len(item) == 3:
        first, second, third = item
        if isinstance(first, str):
            name = _validate_mode_name(first)
            if _is_number(second) and not _is_number(third):
                return name, third, _validate_weight(name, second)
            return name, second, _validate_weight(name, third)
        name = _mode_name(first)
        if isinstance(second, str):
            return name, first, _validate_weight(name, third)
        return _validate_mode_name(third), first, _validate_weight(name, second)
    raise ConfigError(
        "a topic mode entry must have two or three values (name, mode, weight)"
    )


def _parse_sequence(modes: Sequence[Any]) -> list[tuple[str, Any, float | None]]:
    entries: list[tuple[str, Any, float | None]] = []
    for item in modes:
        if isinstance(item, ModeEntry):
            entries.append((_validate_mode_name(item.name), item.mode, item.weight))
        elif isinstance(item, Mapping):
            name = item.get("name", item.get("mode_name", item.get("id")))
            target = item.get(
                "mode",
                item.get(
                    "provider",
                    item.get("instance", item.get("class", name)),
                ),
            )
            if name is None:
                name = _mode_name(target)
            weight_value = item.get("weight")
            weight = (
                None
                if weight_value is None
                else _validate_weight(str(name), weight_value)
            )
            entries.append((_validate_mode_name(name), target, weight))
        elif isinstance(item, (tuple, list)):
            entries.append(_parse_pair(item))
        else:
            name = _mode_name(item)
            entries.append((name, item, None))
    return entries


def _parse_modes(modes: Any) -> list[tuple[str, Any, float | None]]:
    if isinstance(modes, Mapping):
        return _parse_mapping(modes)
    if isinstance(modes, (str, bytes, bytearray)):
        name = _validate_mode_name(
            modes.decode() if isinstance(modes, (bytes, bytearray)) else modes
        )
        return [(name, name, None)]
    if isinstance(modes, Sequence):
        return _parse_sequence(modes)
    name = _mode_name(modes)
    return [(name, modes, None)]


def _weight_map(weights: Any) -> dict[str, float] | None:
    if weights is None:
        return None
    if isinstance(weights, Mapping):
        return {
            _validate_mode_name(name): _validate_weight(str(name), weight)
            for name, weight in weights.items()
        }
    raise ConfigError("topic mode weights must be a mapping or a sequence")


def _resolve_weights(
    entries: list[tuple[str, Any, float | None]],
    weights: Any,
    *,
    profile: Any = None,
) -> dict[str, float]:
    if isinstance(weights, Sequence) and not isinstance(
        weights, (str, bytes, bytearray, Mapping)
    ):
        values = [
            _validate_weight(str(index), weight) for index, weight in enumerate(weights)
        ]
        if len(values) != len(entries):
            raise ConfigError(
                "topic mode weights must match the number of modes",
                details={"modes": len(entries), "weights": len(values)},
            )
        return {
            name: (values[index] if entry_weight is None else entry_weight)
            for index, (name, _mode, entry_weight) in enumerate(entries)
        }

    supplied = _weight_map(weights)
    profile_weights = _profile_weights(profile)
    if supplied is None and profile_weights is not None:
        supplied = {
            _validate_mode_name(name): _validate_weight(str(name), weight)
            for name, weight in profile_weights.items()
        }

    known = {name for name, _mode, _weight in entries}
    if supplied is not None:
        unknown = sorted(set(supplied) - known)
        if unknown:
            raise ConfigError(
                "topic mode weights reference unknown modes: " + ", ".join(unknown),
                details={"unknown": unknown, "modes": sorted(known)},
            )

    resolved: dict[str, float] = {}
    for name, _mode, entry_weight in entries:
        if supplied is not None and name in supplied:
            resolved[name] = supplied[name]
        elif entry_weight is not None:
            resolved[name] = entry_weight
        else:
            resolved[name] = DEFAULT_WEIGHT

    if resolved and all(not weight for weight in resolved.values()):
        raise ConfigError("at least one topic mode must have a positive weight")
    return resolved


def normalize_weights(
    weights: Mapping[str, float],
    *,
    require_sum: bool = False,
) -> dict[str, float]:
    """Validate *weights* and return them as finite, non-negative floats.

    Weights are relative by default, so they need not sum to one. Set
    ``require_sum`` when a caller wants profile style shares instead.
    """

    if not isinstance(weights, Mapping):
        raise ConfigError("topic mode weights must be a mapping")
    normalised: dict[str, float] = {}
    for name, weight in weights.items():
        mode = _validate_mode_name(name)
        if mode in normalised:
            raise ConfigError(f"duplicate topic mode weight: {mode}")
        normalised[mode] = _validate_weight(mode, weight)
    if require_sum and normalised:
        total = math.fsum(normalised.values())
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ConfigError(
                f"topic mode weights must sum to 1, got {total:g}",
                details={"total": total},
            )
    return normalised


normalise_weights = normalize_weights
normalize_mode_weights = normalize_weights
normalise_mode_weights = normalize_weights


def _as_candidate(value: Any) -> TopicCandidate:
    if isinstance(value, TopicCandidate):
        return value
    if isinstance(value, str):
        return TopicCandidate(title=value)
    if isinstance(value, Mapping):
        return TopicCandidate.model_validate(value)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return TopicCandidate.model_validate(dump())
    raise ConfigError(
        f"a topic mode returned an unsupported candidate: {type(value).__name__}"
    )


def _response_candidates(value: Any) -> list[TopicCandidate]:
    if value is None:
        return []
    if isinstance(value, SourceResponse):
        return [_as_candidate(candidate) for candidate in value.candidates]
    if isinstance(value, TopicCandidate):
        return [value]
    if isinstance(value, Mapping):
        for key in ("candidates", "topics", "results", "items"):
            if key in value:
                return _response_candidates(value[key])
        return []
    candidates = getattr(value, "candidates", None)
    if candidates is not None:
        return _response_candidates(candidates)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        return [_as_candidate(candidate) for candidate in value]
    raise ConfigError(
        f"a topic mode returned an unsupported response: {type(value).__name__}"
    )


def _interleave(
    streams: Mapping[str, Sequence[TopicCandidate]],
    weights: Mapping[str, float],
    *,
    seed: int,
    limit: int | None = None,
) -> list[TopicCandidate]:
    names = [
        name for name in streams if streams[name] and weights.get(name, DEFAULT_WEIGHT)
    ]
    if not names:
        return []

    rng = random.Random(seed)
    tie_order = list(names)
    rng.shuffle(tie_order)
    tie_rank = {name: index for index, name in enumerate(tie_order)}

    positions = dict.fromkeys(names, 0)
    current = dict.fromkeys(names, 0.0)
    total_weight = math.fsum(weights[name] for name in names)
    combined: list[TopicCandidate] = []

    while names:
        if limit is not None and len(combined) >= limit:
            break
        active = [name for name in names if positions[name] < len(streams[name])]
        if not active:
            break
        for name in active:
            current[name] += weights[name]
        chosen = max(
            active,
            key=lambda name: (round(current[name], 12), tie_rank[name]),
        )
        current[chosen] -= total_weight
        combined.append(streams[chosen][positions[chosen]])
        positions[chosen] += 1
        if positions[chosen] >= len(streams[chosen]):
            names = [name for name in names if name != chosen]
    return combined


def weighted_interleave(
    streams: Mapping[str, Sequence[Any]],
    weights: Mapping[str, float] | None = None,
    *,
    seed: int | None = DEFAULT_SEED,
    limit: int | None = None,
    tag: bool = True,
    dedupe: bool = DEFAULT_DEDUPE,
) -> list[TopicCandidate]:
    """Interleave candidate *streams* by mode *weights*.

    Streams keep their own candidate order. The mode sequence follows the
    weights, while *seed* chooses between modes that are equally due. When
    ``tag`` is true each returned candidate is tagged with its mode name and
    given its final rank.
    """

    if not isinstance(streams, Mapping):
        raise ConfigError("candidate streams must be a mapping")
    normalised_streams: dict[str, list[TopicCandidate]] = {}
    for raw_name, candidates in streams.items():
        name = _validate_mode_name(raw_name)
        if name in normalised_streams:
            raise ConfigError(f"duplicate topic mode stream: {name}")
        normalised_streams[name] = [
            _as_candidate(candidate).model_copy(update={"mode": name})
            for candidate in candidates
        ]

    if weights is None:
        normalised_weights = dict.fromkeys(normalised_streams, DEFAULT_WEIGHT)
    else:
        normalised_weights = normalize_weights(weights)
        unknown = sorted(set(normalised_weights) - set(normalised_streams))
        if unknown:
            raise ConfigError(
                "topic mode weights reference unknown modes: " + ", ".join(unknown),
                details={"unknown": unknown},
            )
        for name in normalised_streams:
            normalised_weights.setdefault(name, DEFAULT_WEIGHT)

    effective_seed = _validate_seed(seed)
    effective_limit = _validate_limit(limit)
    ordered = _interleave(
        normalised_streams,
        normalised_weights,
        seed=effective_seed,
        limit=effective_limit,
    )

    combined: list[TopicCandidate] = []
    seen: set[str] = set()
    for candidate in ordered:
        mode = candidate.mode
        if dedupe:
            key = normalize_title(candidate.title)
            if key in seen:
                continue
            seen.add(key)
        combined.append(
            candidate.model_copy(update={"mode": mode, "rank": len(combined)})
        )
    return combined


interleave_modes = weighted_interleave
weighted_combine = weighted_interleave
combine_candidates = weighted_interleave
mix_candidates = weighted_interleave
weighted_order = weighted_interleave
seeded_order = weighted_interleave


def _build_request(
    request: SourceRequest | None,
    *,
    query: str = "",
    language: str | None = None,
    limit: int | None = None,
    mode: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> SourceRequest:
    if request is not None:
        return request
    return SourceRequest(
        query=query,
        language=language,
        limit=DEFAULT_REQUEST_SIZE if limit is None else max(limit, 1),
        mode=mode,
        metadata=dict(metadata or {}),
    )


class ModeMix(TopicModeBase):
    """Combine several topic modes into one weighted, seeded stream.

    ``modes`` accepts a mapping from mode name to a mode instance, class,
    factory, registered name or dotted path. A sequence of ``(name, mode)`` or
    ``(name, mode, weight)`` entries is accepted too, as is a mapping from name
    to ``{"mode": ..., "weight": ...}``.

    ``weights`` may be a mapping or a sequence aligned with the entries. When
    omitted, each entry keeps its own weight or defaults to ``1.0``. Weights are
    relative, so they need not sum to one. A zero weight mode is not run.
    """

    name = "mix"

    def __init__(
        self,
        modes: Any = None,
        weights: Any = None,
        *,
        entries: Any = None,
        mode_weights: Any = None,
        shares: Any = None,
        mix: Any = None,
        profile: Any = None,
        registry: Any = None,
        seed: int | None = DEFAULT_SEED,
        limit: int | None = DEFAULT_LIMIT,
        dedupe: bool = DEFAULT_DEDUPE,
        continue_on_error: bool = DEFAULT_CONTINUE_ON_ERROR,
        skip_errors: bool | None = None,
        name: str | None = None,
        **options: Any,
    ) -> None:
        super().__init__(name)
        supplied_weights = [
            ("weights", weights),
            ("mode_weights", mode_weights),
            ("shares", shares),
        ]
        provided = [
            (label, value) for label, value in supplied_weights if value is not None
        ]
        if len(provided) > 1:
            labels = ", ".join(label for label, _ in provided)
            raise ConfigError(
                f"use only one of weights, mode_weights or shares (got {labels})"
            )
        effective_weights = provided[0][1] if provided else None
        if modes is None and entries is not None:
            modes = entries
        if modes is None and mix is not None:
            modes = mix
        if modes is None and profile is not None:
            profile_weights = _profile_weights(profile)
            if profile_weights is None:
                raise ConfigError("the profile does not define topic modes")
            modes = {mode_name: mode_name for mode_name in profile_weights}
            if effective_weights is None:
                effective_weights = profile_weights
        if modes is None:
            modes = {}

        if skip_errors is not None:
            if not isinstance(skip_errors, bool):
                raise ConfigError("skip_errors must be a boolean")
            continue_on_error = skip_errors
        if not isinstance(dedupe, bool):
            raise ConfigError("dedupe must be a boolean")
        if not isinstance(continue_on_error, bool):
            raise ConfigError("continue_on_error must be a boolean")

        parsed = _parse_modes(modes)
        names: set[str] = set()
        for entry_name, _target, _entry_weight in parsed:
            if entry_name in names:
                raise ConfigError(f"duplicate topic mode: {entry_name}")
            names.add(entry_name)
        resolved_weights = _resolve_weights(
            parsed,
            effective_weights,
            profile=profile,
        )

        self._entries = tuple(
            ModeEntry(
                name=entry_name,
                mode=_resolve_mode(target, registry),
                weight=resolved_weights[entry_name],
            )
            for entry_name, target, _entry_weight in parsed
        )
        self._seed = _validate_seed(seed)
        self._limit = _validate_limit(limit)
        self._dedupe = dedupe
        self._continue_on_error = continue_on_error
        self._registry = registry
        self._options = dict(options)

    @property
    def entries(self) -> tuple[ModeEntry, ...]:
        """The configured modes in declaration order."""
        return self._entries

    @property
    def modes(self) -> tuple[Any, ...]:
        """The mode instances in declaration order."""
        return tuple(entry.mode for entry in self._entries)

    @property
    def names(self) -> tuple[str, ...]:
        """The mode names in declaration order."""
        return tuple(entry.name for entry in self._entries)

    @property
    def mode_names(self) -> tuple[str, ...]:
        """Alias for :attr:`names`."""
        return self.names

    @property
    def keys(self) -> tuple[str, ...]:
        """Alias for :attr:`names`."""
        return self.names

    @property
    def weights(self) -> dict[str, float]:
        """The relative weight of each mode."""
        return {entry.name: entry.weight for entry in self._entries}

    @property
    def mode_weights(self) -> dict[str, float]:
        """Alias for :attr:`weights`."""
        return self.weights

    @property
    def shares(self) -> dict[str, float]:
        """Return each weight as a share of the total."""
        total = math.fsum(self.weights.values())
        if total <= 0:
            return dict.fromkeys(self.weights, 0.0)
        return {name: weight / total for name, weight in self.weights.items()}

    @property
    def normalized_weights(self) -> dict[str, float]:
        """Alias for :attr:`shares`."""
        return self.shares

    @property
    def seed(self) -> int:
        """The seed used to break ties between equally due modes."""
        return self._seed

    @property
    def limit(self) -> int | None:
        """The maximum number of candidates to serve, if any."""
        return self._limit

    @property
    def dedupe(self) -> bool:
        """Whether duplicate titles across modes are dropped."""
        return self._dedupe

    @property
    def continue_on_error(self) -> bool:
        """Whether a failing mode is skipped instead of failing the mix."""
        return self._continue_on_error

    async def _run_mode(
        self,
        entry: ModeEntry,
        request: SourceRequest,
    ) -> tuple[list[TopicCandidate], Exception | None]:
        mode_request = request.model_copy(update={"mode": entry.name})
        try:
            response = await entry.mode.discover(mode_request)
        except Exception as exc:
            if self._continue_on_error:
                return [], exc
            raise
        return _response_candidates(response), None

    async def discover(self, request: SourceRequest) -> MixResult:
        """Run every active mode and return their weighted, tagged candidates."""
        active = [entry for entry in self._entries if entry.weight]
        if not active:
            raise ConfigError("at least one topic mode must have a positive weight")

        results = await asyncio.gather(
            *(self._run_mode(entry, request) for entry in active)
        )
        streams: dict[str, list[TopicCandidate]] = {}
        errors: list[dict[str, Any]] = []
        for entry, (candidates, error) in zip(active, results, strict=True):
            if error is not None:
                errors.append(
                    {
                        "mode": entry.name,
                        "error": str(error),
                        "type": type(error).__name__,
                    }
                )
                continue
            streams[entry.name] = [
                candidate.model_copy(update={"mode": entry.name})
                for candidate in candidates
            ]

        weights = {
            entry.name: entry.weight for entry in active if entry.name in streams
        }
        combined = weighted_interleave(
            streams,
            weights,
            seed=self._seed,
            limit=self._limit,
            dedupe=self._dedupe,
        )
        found = sum(len(candidates) for candidates in streams.values())
        return MixResult(
            provider=self.name,
            query=request.query,
            candidates=combined,
            metadata={
                "mode": self.name,
                "modes": list(self.names),
                "mode_weights": self.weights,
                "shares": self.shares,
                "seed": self._seed,
                "limit": self._limit,
                "dedupe": self._dedupe,
                "mode_counts": {
                    name: sum(1 for candidate in combined if candidate.mode == name)
                    for name in streams
                },
                "found_count": found,
                "served_count": len(combined),
                "skipped_duplicates": found - len(combined),
                "failed_modes": [error["mode"] for error in errors],
                "errors": errors,
            },
        )

    async def combine(self, request: SourceRequest) -> MixResult:
        """Alias for :meth:`discover`."""
        return await self.discover(request)

    async def mix(self, request: SourceRequest) -> MixResult:
        """Alias for :meth:`discover`."""
        return await self.discover(request)

    @classmethod
    def from_profile(
        cls,
        profile: Any,
        *,
        registry: Any = None,
        seed: int | None = DEFAULT_SEED,
        limit: int | None = DEFAULT_LIMIT,
        dedupe: bool = DEFAULT_DEDUPE,
        continue_on_error: bool = DEFAULT_CONTINUE_ON_ERROR,
        name: str | None = None,
    ) -> ModeMix:
        """Build a mix from a profile's ``topic_modes`` mapping."""
        return cls(
            profile=profile,
            registry=registry,
            seed=seed,
            limit=limit,
            dedupe=dedupe,
            continue_on_error=continue_on_error,
            name=name,
        )


MixMode = ModeMix
MixedMode = ModeMix
ModeMixer = ModeMix
WeightedModeMix = ModeMix
ModesMix = ModeMix
Mix = ModeMix
Combine = ModeMix
TopicModeMix = ModeMix
MixedTopicMode = ModeMix
ModeMixMode = ModeMix
CombinedMode = ModeMix
CombiningMode = ModeMix
WeightedMix = ModeMix
WeightedTopicMode = ModeMix
ModeMixProvider = ModeMix
ModeMixTopicMode = ModeMix
ModeMixSourceMode = ModeMix
MixModeProvider = ModeMix
MixTopicMode = ModeMix
MixSourceMode = ModeMix
MixedModeProvider = ModeMix
ModeMixResult = MixResult
WeightedMixResult = MixResult

for _name in ("mix", "mixed", "mode_mix", "mix_mode", "weighted_mix"):
    topic_mode_registry.register(_name, ModeMix)
del _name


async def mix_modes(
    modes: Any = None,
    weights: Any = None,
    *,
    entries: Any = None,
    mode_weights: Any = None,
    shares: Any = None,
    mix: Any = None,
    profile: Any = None,
    registry: Any = None,
    seed: int | None = DEFAULT_SEED,
    limit: int | None = DEFAULT_LIMIT,
    dedupe: bool = DEFAULT_DEDUPE,
    continue_on_error: bool = DEFAULT_CONTINUE_ON_ERROR,
    request: SourceRequest | None = None,
    query: str = "",
    language: str | None = None,
    name: str | None = None,
    **options: Any,
) -> MixResult:
    """Build and run a :class:`ModeMix` in one call."""
    configured = ModeMix(
        modes,
        weights,
        entries=entries,
        mode_weights=mode_weights,
        shares=shares,
        mix=mix,
        profile=profile,
        registry=registry,
        seed=seed,
        limit=limit,
        dedupe=dedupe,
        continue_on_error=continue_on_error,
        name=name,
        **options,
    )
    return await configured.discover(
        _build_request(
            request,
            query=query,
            language=language,
            limit=limit,
            mode=configured.name,
        )
    )


async def combine_modes(
    modes: Any = None,
    weights: Any = None,
    **options: Any,
) -> MixResult:
    """Alias for :func:`mix_modes`."""
    return await mix_modes(modes, weights, **options)


async def mix(*args: Any, **options: Any) -> MixResult:
    """Alias for :func:`mix_modes`."""
    return await mix_modes(*args, **options)


async def combine(*args: Any, **options: Any) -> MixResult:
    """Alias for :func:`mix_modes`."""
    return await mix_modes(*args, **options)


async def weighted_mix(*args: Any, **options: Any) -> MixResult:
    """Alias for :func:`mix_modes`."""
    return await mix_modes(*args, **options)
