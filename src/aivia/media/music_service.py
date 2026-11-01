"""Run one music generation request through the single paid path.

The service is the only place a music call is executed. It screens the prompt
and any lyrics, answers an identical request from the asset cache when
possible, applies the remaining budget, asks the router for a provider,
validates and stores every returned track, records one actual cost event and
writes a cost sidecar next to each output. Callers never choose a vendor,
handle a vendor payload or write a cost record themselves.

The order is deliberate and matches the project invariant: screen -> cache ->
budget -> router -> provider -> validate -> store -> cost -> sidecar. A cache
hit short-circuits the paid path but still records a zero-amount cost event, so
a reused track is visible in the ledger without counting as spend.

Every track keeps the license recorded by its provider. A track whose license
does not explicitly allow commercial use is never considered cleared, so
:class:`MusicLicenseGate` can block publishing or delivery without guessing.
Missing license metadata is treated the same as a license that forbids
commercial use.
"""

from __future__ import annotations

import hashlib
import secrets
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote, urlsplit

import httpx
from sqlalchemy import Engine

from aivia.core.asset_cache import AssetCache, AssetCacheKey
from aivia.core.budget import BudgetGuard
from aivia.core.config.budget import BudgetConfig
from aivia.core.cost import CostTracker
from aivia.core.errors import ValidationError
from aivia.core.media_probe import (
    MediaDownloader,
    MediaFormat,
    MediaLimits,
    MediaProbe,
    validate_media,
)
from aivia.core.models.asset import (
    Asset,
    AssetAnalysis,
    AssetCostLink,
    AssetKind,
    AssetLicense,
)
from aivia.core.models.base import canonical_json
from aivia.core.models.cost import CostEvent
from aivia.core.models.run import ToolchainFingerprint
from aivia.core.pricing import ResolvedPrice
from aivia.core.router import Router, RouteSelection
from aivia.core.sidecar import SIDECAR_SUFFIX, build_cost_sidecar, write_cost_sidecar
from aivia.media.accessibility import (
    store_accessibility_sidecars,
    write_accessibility_sidecars,
)
from aivia.media.adaptation import (
    adapt_for_cache,
    adapt_for_provider,
    cache_request_variants,
)
from aivia.media.screening import (
    DEFAULT_PROMPT_KIND,
    ScreeningToken,
    require_screened,
    screen_prompt,
)
from aivia.providers.base import Provider
from aivia.providers.media.adapt import (
    AdaptationError,
    AdaptationResult,
    AdaptationWarning,
)
from aivia.providers.media.dto import MediaKind
from aivia.providers.music.cost import (
    MUSIC_OPERATION,
    MUSIC_UNIT,
    TRACK_UNIT,
    MusicCostCalculator,
)
from aivia.providers.protocols.audio import AudioFormat
from aivia.providers.protocols.music import (
    CAPABILITY_MUSIC_GEN,
    MusicProvider,
    MusicRequest,
    MusicResponse,
    MusicTrack,
)
from aivia.providers.protocols.storage import (
    StorageObject,
    StorageProvider,
    StoragePutRequest,
)

__all__ = [
    "MUSIC_OPERATION",
    "MUSIC_UNIT",
    "TRACK_UNIT",
    "LicenseDecision",
    "MusicGeneration",
    "MusicLicenseError",
    "MusicLicenseGate",
    "MusicOutcome",
    "MusicService",
    "MusicServiceResult",
    "check_commercial_license",
    "generate_music",
    "license_from_track",
    "music_cache_key",
    "music_service_from_config",
    "require_commercial_license",
]

ToolchainLike = ToolchainFingerprint | Mapping[str, Any] | str | None
BudgetLike = BudgetGuard | BudgetConfig | Mapping[str, Any] | None

_LICENSE_KEYS: dict[str, tuple[str, ...]] = {
    "name": ("license", "license_name", "licence", "spdx", "type", "name"),
    "url": ("license_url", "terms_url", "url", "link"),
    "attribution": (
        "attribution",
        "credit",
        "attribution_text",
        "required_attribution",
    ),
    "commercial_use": (
        "commercial_use",
        "commercial",
        "allow_commercial",
        "commercial_allowed",
    ),
    "attribution_required": (
        "attribution_required",
        "requires_attribution",
        "require_attribution",
    ),
    "share_alike": ("share_alike", "copyleft"),
}

_AUDIO_FORMATS = {item.value: item for item in AudioFormat}
_TIME_UNITS = frozenset(
    {
        "millisecond",
        "milliseconds",
        "ms",
        "second",
        "seconds",
        "sec",
        "secs",
        "s",
        "minute",
        "minutes",
        "min",
        "mins",
        "hour",
        "hours",
        "hr",
        "hrs",
        "h",
    }
)


def _file_path(url: str) -> Path | None:
    parsed = urlsplit(url)
    if parsed.scheme.lower() != "file":
        return None
    return Path(unquote(parsed.path))


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _audio_format(
    value: str | AudioFormat | None,
    fallback: AudioFormat,
) -> AudioFormat:
    if isinstance(value, AudioFormat):
        return value
    if value is None:
        return fallback
    return _AUDIO_FORMATS.get(str(value).lower(), fallback)


def _media_format(value: AudioFormat | str | None) -> MediaFormat | None:
    if value is None:
        return None
    text = value.value if isinstance(value, AudioFormat) else str(value).lower()
    aliases = {"opus": "ogg", "pcm": "wav"}
    text = aliases.get(text, text)
    try:
        return MediaFormat(text)
    except ValueError:
        return None


def _measured_audio(
    probe: MediaProbe,
    *,
    fallback_format: AudioFormat,
    fallback_sample_rate: int | None = None,
    fallback_channels: int | None = None,
) -> tuple[AudioFormat, int | None, int | None]:
    """Return the container, sample rate and channel count measured by ffprobe."""
    fmt = _audio_format(probe.format.value, fallback_format)
    sample_rate = fallback_sample_rate
    channels = fallback_channels
    info = probe.ffprobe
    if info is not None:
        streams = info.audio_streams
        if streams:
            stream = streams[0]
            if stream.sample_rate is not None:
                sample_rate = stream.sample_rate
            if stream.channels is not None:
                channels = stream.channels
    return fmt, sample_rate, channels


def _stored_audio_format(
    analysis: AssetAnalysis,
    fallback: AudioFormat,
) -> AudioFormat:
    """Return the format recorded on *analysis*, or *fallback* when unknown."""
    value = analysis.format.strip().lower()
    if value:
        found = _AUDIO_FORMATS.get(value)
        if found is not None:
            return found
    return fallback


def _license_values(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        return {}
    values: dict[str, Any] = {}
    for name, keys in _LICENSE_KEYS.items():
        for key in keys:
            if key in metadata:
                values[name] = metadata[key]
                break
    return values


def _license_from_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    base: AssetLicense | None = None,
) -> AssetLicense:
    values = (base or AssetLicense()).model_dump()
    values.update(_license_values(metadata))
    return AssetLicense.model_validate(values)


def license_from_track(
    track: MusicTrack,
    *,
    provider: Provider | None = None,
    response: MusicResponse | None = None,
) -> AssetLicense:
    """Return the license for *track*, falling back to response and provider.

    A track's own metadata wins, then the response metadata, then the provider
    metadata. A value that is absent from all three stays ``None``, which the
    gate treats as not cleared.
    """
    if not isinstance(track, MusicTrack):
        raise TypeError("track must be a MusicTrack")
    base = provider.metadata.license if provider is not None else None
    values = (base or AssetLicense()).model_dump()
    if response is not None:
        values.update(_license_values(response.metadata))
    values.update(_license_values(track.metadata))
    return AssetLicense.model_validate(values)


class MusicLicenseError(ValidationError):
    """A music track is not cleared for the requested commercial action."""

    code = "music_license"

    def __init__(
        self,
        message: str,
        *,
        action: str = "publish",
        blockers: Sequence[str] = (),
        licenses: Sequence[AssetLicense] = (),
    ) -> None:
        self.action = action
        self.blockers = tuple(blockers)
        self.licenses = tuple(licenses)
        super().__init__(
            message,
            details={
                "action": action,
                "blockers": list(self.blockers),
                "licenses": [
                    license.model_dump(mode="json") for license in self.licenses
                ],
            },
        )


@dataclass(frozen=True, slots=True)
class LicenseDecision:
    """The verdict of a commercial license check for one action."""

    allowed: bool
    action: str = "publish"
    licenses: tuple[AssetLicense, ...] = ()
    blockers: tuple[str, ...] = ()
    attribution: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Whether the action is allowed."""
        return self.allowed

    @property
    def blocked(self) -> bool:
        """Whether at least one license blocked the action."""
        return not self.allowed

    @property
    def reasons(self) -> tuple[str, ...]:
        """Alias for :attr:`blockers`."""
        return self.blockers

    @property
    def required_attribution(self) -> tuple[str, ...]:
        """Alias for :attr:`attribution`."""
        return self.attribution

    def require(self) -> LicenseDecision:
        """Return this decision, raising when the action is blocked."""
        if not self.allowed:
            raise MusicLicenseError(
                f"music is not cleared for {self.action}",
                action=self.action,
                blockers=self.blockers,
                licenses=self.licenses,
            )
        return self

    def ensure(self) -> LicenseDecision:
        """Alias for :meth:`require`."""
        return self.require()


def _collect_licenses(value: Any) -> tuple[AssetLicense, ...]:
    if value is None:
        return ()
    if isinstance(value, MusicOutcome):
        return value.licenses
    if isinstance(value, AssetLicense):
        return (value,)
    if isinstance(value, Asset):
        return (value.license,)
    if isinstance(value, MusicTrack):
        return (_license_from_metadata(value.metadata),)
    if isinstance(value, MusicResponse):
        result: list[AssetLicense] = []
        for track in value.tracks:
            metadata = dict(value.metadata)
            metadata.update(track.metadata)
            result.append(_license_from_metadata(metadata))
        return tuple(result)
    if isinstance(value, Mapping):
        return (_license_from_metadata(value),)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        collected: list[AssetLicense] = []
        for item in value:
            collected.extend(_collect_licenses(item))
        return tuple(collected)
    raise TypeError("license source must be music, assets, tracks or licenses")


class MusicLicenseGate:
    """Block publishing or delivery unless every track is commercially cleared.

    A license is cleared only when it names or links to its terms and states
    ``commercial_use`` as ``True``. A missing license and an unstated
    commercial right are both blockers. Set ``require_attribution`` to also
    reject a license whose required credit line is empty.
    """

    def __init__(
        self,
        *,
        require_commercial: bool = True,
        require_attribution: bool = False,
    ) -> None:
        self._require_commercial = bool(require_commercial)
        self._require_attribution = bool(require_attribution)

    @property
    def require_commercial(self) -> bool:
        """Whether a commercial-use grant is required."""
        return self._require_commercial

    @property
    def require_attribution(self) -> bool:
        """Whether a non-empty attribution line is required."""
        return self._require_attribution

    def check(self, *sources: Any, action: str = "publish") -> LicenseDecision:
        """Return the commercial verdict for *sources* without raising."""
        licenses = _collect_licenses(sources)
        blockers: list[str] = []
        if self._require_commercial:
            if not licenses:
                blockers.append("license_missing")
            for index, license in enumerate(licenses):
                if not license.is_known:
                    blockers.append(f"license_missing:{index}")
                if license.commercial_use is None:
                    blockers.append(f"commercial_use_unconfirmed:{index}")
                elif not license.is_commercial:
                    blockers.append(f"commercial_use_not_allowed:{index}")
        if self._require_attribution:
            for index, license in enumerate(licenses):
                if license.requires_attribution and not license.attribution:
                    blockers.append(f"attribution_missing:{index}")
        attribution = tuple(
            license.attribution
            for license in licenses
            if license.requires_attribution and license.attribution
        )
        return LicenseDecision(
            allowed=not blockers,
            action=action,
            licenses=licenses,
            blockers=tuple(dict.fromkeys(blockers)),
            attribution=attribution,
        )

    def require(self, *sources: Any, action: str = "publish") -> LicenseDecision:
        """Return the verdict, raising :class:`MusicLicenseError` when blocked."""
        return self.check(*sources, action=action).require()

    def check_publish(self, *sources: Any) -> LicenseDecision:
        """Check the commercial license for publishing."""
        return self.check(*sources, action="publish")

    def check_delivery(self, *sources: Any) -> LicenseDecision:
        """Check the commercial license for delivery."""
        return self.check(*sources, action="deliver")

    check_deliver = check_delivery

    def require_publish(self, *sources: Any) -> LicenseDecision:
        """Require a commercial license for publishing."""
        return self.require(*sources, action="publish")

    def require_delivery(self, *sources: Any) -> LicenseDecision:
        """Require a commercial license for delivery."""
        return self.require(*sources, action="deliver")

    require_deliver = require_delivery


DEFAULT_LICENSE_GATE = MusicLicenseGate()
"""The default gate requiring a known, commercially usable license."""


def check_commercial_license(
    value: Any,
    *,
    action: str = "publish",
    gate: MusicLicenseGate | None = None,
) -> LicenseDecision:
    """Return the commercial license verdict for *value*."""
    return (gate or DEFAULT_LICENSE_GATE).check(value, action=action)


def require_commercial_license(
    value: Any,
    *,
    action: str = "publish",
    gate: MusicLicenseGate | None = None,
) -> LicenseDecision:
    """Raise :class:`MusicLicenseError` unless *value* is commercially cleared."""
    return (gate or DEFAULT_LICENSE_GATE).require(value, action=action)


@dataclass(frozen=True, slots=True)
class MusicOutcome:
    """The response together with its stored assets, price, events and licenses."""

    request: MusicRequest
    response: MusicResponse
    assets: tuple[Asset, ...] = ()
    probes: tuple[MediaProbe, ...] = ()
    storage_objects: tuple[StorageObject, ...] = ()
    events: tuple[CostEvent, ...] = ()
    price: ResolvedPrice | None = None
    selection: RouteSelection | None = None
    sidecars: tuple[Path, ...] = ()
    accessibility_sidecars: tuple[Any, ...] = ()
    cache_hit: bool = False
    licenses: tuple[AssetLicense, ...] = ()
    attribution: tuple[str, ...] = field(default_factory=tuple)
    adaptation_warnings: tuple[AdaptationWarning, ...] = ()

    @property
    def warnings(self) -> tuple[AdaptationWarning, ...]:
        """Every change the adaptation layer made before submission."""
        return self.adaptation_warnings

    @property
    def adaptation_messages(self) -> tuple[str, ...]:
        """The human-readable adaptation warnings in order."""
        return tuple(warning.message for warning in self.adaptation_warnings)

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
    def tracks(self) -> list[MusicTrack]:
        """Every track in the provider response."""
        return list(self.response.tracks)

    @property
    def track(self) -> MusicTrack | None:
        """The first track, or ``None``."""
        return self.tracks[0] if self.tracks else None

    @property
    def first(self) -> MusicTrack | None:
        """Alias for :attr:`track`."""
        return self.track

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
        """The first sidecar written, or ``None``."""
        return self.sidecars[0] if self.sidecars else None

    @property
    def sidecar_paths(self) -> tuple[Path, ...]:
        """Alias for :attr:`sidecars`."""
        return self.sidecars

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
        """Whether this outcome came from the asset cache."""
        return self.cache_hit

    @property
    def amount_usd(self) -> float:
        """The recorded cost, or zero when no cost was recorded."""
        if self.events:
            return sum(event.amount_usd for event in self.events)
        if self.price is not None:
            return self.price.amount_usd
        return 0.0

    @property
    def duration_seconds(self) -> float:
        """The measured duration of every track added together."""
        measured = [probe.duration_seconds for probe in self.probes]
        if any(value is not None for value in measured):
            return sum(value or 0.0 for value in measured)
        return sum(track.duration_seconds or 0.0 for track in self.response.tracks)

    @property
    def license(self) -> AssetLicense | None:
        """The first track's license, or ``None``."""
        return self.licenses[0] if self.licenses else None

    @property
    def license_names(self) -> tuple[str, ...]:
        """The name of every track license."""
        return tuple(license.name for license in self.licenses)

    @property
    def is_commercial(self) -> bool:
        """Whether every track explicitly allows commercial use."""
        return bool(self.licenses) and all(
            license.is_commercial for license in self.licenses
        )

    @property
    def is_commercially_cleared(self) -> bool:
        """Whether every track has a known commercial license."""
        return bool(self.licenses) and all(
            license.is_commercial and license.is_known for license in self.licenses
        )

    @property
    def requires_attribution(self) -> bool:
        """Whether any track license requires a credit line."""
        return bool(self.attribution) or any(
            license.requires_attribution for license in self.licenses
        )

    def check_commercial_license(
        self,
        *,
        action: str = "publish",
        gate: MusicLicenseGate | None = None,
    ) -> LicenseDecision:
        """Return the commercial license verdict for these tracks."""
        return check_commercial_license(self, action=action, gate=gate)

    def require_commercial_license(
        self,
        action: str = "publish",
        *,
        gate: MusicLicenseGate | None = None,
    ) -> LicenseDecision:
        """Raise unless every track is cleared for *action*."""
        return require_commercial_license(self, action=action, gate=gate)

    def require_for_publish(
        self,
        gate: MusicLicenseGate | None = None,
    ) -> LicenseDecision:
        """Raise unless every track is cleared for publishing."""
        return self.require_commercial_license("publish", gate=gate)

    def require_for_delivery(
        self,
        gate: MusicLicenseGate | None = None,
    ) -> LicenseDecision:
        """Raise unless every track is cleared for delivery."""
        return self.require_commercial_license("deliver", gate=gate)

    ensure_publishable = require_for_publish
    ensure_deliverable = require_for_delivery

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary for logs and manifests."""
        return {
            "provider": self.provider,
            "model": self.model,
            "cache_hit": self.cache_hit,
            "amount_usd": self.amount_usd,
            "duration_seconds": self.duration_seconds,
            "paths": list(self.paths),
            "licenses": [license.model_dump(mode="json") for license in self.licenses],
            "commercial_cleared": self.is_commercially_cleared,
            "adaptation_warnings": [
                warning.to_dict() for warning in self.adaptation_warnings
            ],
        }


MusicGeneration = MusicOutcome
"""Alias for :class:`MusicOutcome`."""

MusicServiceResult = MusicOutcome
"""Alias for :class:`MusicOutcome`."""


def _seed_for(request: MusicRequest, index: int) -> int | None:
    if request.seed is None:
        return None
    return request.seed + index


def _response_seeds(
    request: MusicRequest,
    response: MusicResponse,
    base_seed: int,
) -> tuple[int, ...]:
    if request.seed is None:
        reported = [track.seed for track in response.tracks]
        if all(seed is not None for seed in reported) and len(set(reported)) == len(
            reported
        ):
            return tuple(cast(int, seed) for seed in reported)
    return tuple(base_seed + index for index in range(response.count))


def _license_fingerprint(
    value: AssetLicense | Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, AssetLicense):
        license = value
    elif isinstance(value, Mapping):
        license = AssetLicense.model_validate(value)
    else:
        raise TypeError("license must be an AssetLicense, mapping or None")
    return license.model_dump(mode="json")


def music_cache_key(
    request: MusicRequest,
    *,
    index: int = 0,
    model: str | None = None,
    license: AssetLicense | Mapping[str, Any] | None = None,
) -> AssetCacheKey:
    """Return the cache key for one output of *request*.

    The provider is deliberately left empty so an identical request can be
    answered before the router has selected a provider. The stored asset still
    records who actually served the call.

    The selected provider's *license* is part of the key, so changing the
    rights attached to a provider invalidates its cached tracks instead of
    serving a clearance that no longer applies.
    """
    if not isinstance(request, MusicRequest):
        raise TypeError("request must be a MusicRequest")
    parameters: dict[str, Any] = {
        "lyrics": request.lyrics,
        "instrumental": request.instrumental,
        "model": request.model,
        "duration_seconds": request.duration_seconds,
        "format": request.format.value,
        "sample_rate": request.sample_rate,
        "tempo_bpm": request.tempo_bpm,
        "style": request.style,
        "seed": _seed_for(request, index),
        "count": request.count,
        "index": index,
        "license": _license_fingerprint(license),
    }
    return AssetCacheKey(
        provider="",
        model=(request.model if model is None else model) or "",
        prompt=request.prompt,
        parameters=parameters,
    )


class MusicService:
    """Execute music generation through a :class:`~aivia.core.router.Router`.

    ``cache`` answers an identical request without a provider call and persists
    the tracks of a miss. When it is omitted but a tracker or engine is given,
    the service builds an :class:`~aivia.core.asset_cache.AssetCache` on that
    ledger so caching is never silently disabled. ``storage`` stores validated
    audio bytes; without one the asset record is still built but no file is
    written.
    """

    def __init__(
        self,
        router: Router,
        *,
        tracker: CostTracker | None = None,
        cost_tracker: CostTracker | None = None,
        cache: AssetCache | None = None,
        engine: Engine | None = None,
        storage: StorageProvider | None = None,
        budget: BudgetLike = None,
        budget_guard: BudgetGuard | None = None,
        safety: Any = None,
        safety_config: Any = None,
        screening: Any = None,
        screening_config: Any = None,
        screener: Any = None,
        limits: MediaLimits | None = None,
        media_limits: MediaLimits | None = None,
        downloader: MediaDownloader | None = None,
        http: Any = None,
        transport: httpx.AsyncBaseTransport | None = None,
        pricing_overrides: Mapping[str, float] | None = None,
        pricing_version: str = "",
        operation: str = MUSIC_OPERATION,
        toolchain: ToolchainLike = None,
        toolchain_fingerprint: ToolchainLike = None,
        key_prefix: str | None = None,
        record_cost: bool = True,
        record_sidecar: bool = True,
        accessibility: Any = None,
        accessibility_config: Any = None,
        no_cache: bool = False,
        ffprobe_executable: str = "ffprobe",
        ffprobe_timeout_seconds: float | None = None,
        ffprobe_runner: Any = None,
    ) -> None:
        if not isinstance(router, Router):
            raise TypeError("router must be a Router")
        if tracker is not None and cost_tracker is not None:
            raise TypeError("pass either tracker or cost_tracker, not both")
        if budget is not None and budget_guard is not None:
            raise TypeError("pass either budget or budget_guard, not both")
        if limits is not None and media_limits is not None:
            raise TypeError("pass either limits or media_limits, not both")
        if toolchain is not None and toolchain_fingerprint is not None:
            raise TypeError("pass either toolchain or toolchain_fingerprint, not both")
        if downloader is not None and (http is not None or transport is not None):
            raise TypeError(
                "pass either a downloader or http/transport options, not both"
            )
        active_tracker = tracker if tracker is not None else cost_tracker
        if active_tracker is not None and not isinstance(active_tracker, CostTracker):
            raise TypeError("tracker must be a CostTracker")
        if cache is not None and not isinstance(cache, AssetCache):
            raise TypeError("cache must be an AssetCache")
        if engine is not None and not isinstance(engine, Engine):
            raise TypeError("engine must be a SQLAlchemy Engine")
        if downloader is not None and not isinstance(downloader, MediaDownloader):
            raise TypeError("downloader must be a MediaDownloader")

        supplied_safety = [
            value
            for value in (safety, safety_config, screening, screening_config)
            if value is not None
        ]
        if len(supplied_safety) > 1:
            raise TypeError("pass only one safety or screening configuration")
        selected_safety = supplied_safety[0] if supplied_safety else None
        if (
            screener is None
            and selected_safety is not None
            and hasattr(selected_safety, "require")
        ):
            screener = selected_safety
            selected_safety = None
        if not isinstance(operation, str) or not operation.strip():
            raise ValueError("operation must be a non-empty string")
        if key_prefix is not None:
            if not isinstance(key_prefix, str) or not key_prefix.strip():
                raise ValueError("key_prefix must be a non-empty string")
            key_prefix = key_prefix.strip().strip("/")

        selected_cache = cache
        if selected_cache is None and engine is not None:
            selected_cache = AssetCache(engine, no_cache=no_cache)
        elif (
            selected_cache is None
            and active_tracker is not None
            and active_tracker.repository.engine is not None
        ):
            selected_cache = AssetCache(
                active_tracker.repository.engine,
                cost=active_tracker,
                no_cache=no_cache,
            )
        if selected_cache is not None and no_cache:
            selected_cache.no_cache = True
        if active_tracker is None and selected_cache is not None:
            active_tracker = selected_cache.cost

        selected_limits = limits if limits is not None else media_limits
        if selected_limits is not None and not isinstance(selected_limits, MediaLimits):
            raise TypeError("limits must be a MediaLimits")

        active_budget = budget_guard if budget_guard is not None else budget
        if isinstance(active_budget, (BudgetConfig, Mapping)):
            active_budget = BudgetGuard(active_budget, active_tracker)
        if active_budget is not None and not isinstance(active_budget, BudgetGuard):
            raise TypeError("budget must be a BudgetGuard, BudgetConfig or mapping")

        self._router = router
        self._tracker = active_tracker
        self._cache = selected_cache
        self._storage = storage
        self._budget = active_budget
        self._safety = selected_safety
        self._screener = screener
        self._limits = selected_limits or MediaLimits()
        self._downloader = downloader
        self._http = http
        self._transport = transport
        self._operation = operation.strip()
        self._toolchain = toolchain if toolchain is not None else toolchain_fingerprint
        self._key_prefix = key_prefix
        self._record_cost = bool(record_cost)
        self._record_sidecar = bool(record_sidecar)
        self._accessibility = (
            accessibility if accessibility is not None else accessibility_config
        )
        self._no_cache = bool(no_cache)
        self._ffprobe_executable = ffprobe_executable
        self._ffprobe_timeout = ffprobe_timeout_seconds
        self._ffprobe_runner = ffprobe_runner
        self._calculator = MusicCostCalculator(
            pricing_overrides,
            pricing_version=pricing_version,
        )
        self._last_outcome: MusicOutcome | None = None
        self._last_selection: RouteSelection | None = None
        self._last_event: CostEvent | None = None
        self._last_price: ResolvedPrice | None = None

    @property
    def router(self) -> Router:
        """The router every call goes through."""
        return self._router

    @property
    def tracker(self) -> CostTracker | None:
        """The default cost tracker, when one was configured."""
        return self._tracker

    @property
    def cache(self) -> AssetCache | None:
        """The asset cache, when one was configured."""
        return self._cache

    @property
    def storage(self) -> StorageProvider | None:
        """The storage provider, when one was configured."""
        return self._storage

    @property
    def budget(self) -> BudgetGuard | None:
        """The budget guard, when one was configured."""
        return self._budget

    @property
    def limits(self) -> MediaLimits:
        """The caps applied to every downloaded or returned track."""
        return self._limits

    @property
    def calculator(self) -> MusicCostCalculator:
        """The calculator that prices and records music calls."""
        return self._calculator

    @property
    def last_outcome(self) -> MusicOutcome | None:
        """The most recent outcome."""
        return self._last_outcome

    @property
    def last_selection(self) -> RouteSelection | None:
        """The most recent successful router selection."""
        return self._last_selection

    @property
    def last_event(self) -> CostEvent | None:
        """The most recent recorded cost event."""
        return self._last_event

    @property
    def last_price(self) -> ResolvedPrice | None:
        """The most recent resolved price."""
        return self._last_price

    def screen(self, text: str) -> ScreeningToken:
        """Screen *text* with this service's configuration and return a token."""
        if self._screener is not None:
            return cast(
                ScreeningToken,
                cast(Any, self._screener).screen_prompt(text),
            )
        return screen_prompt(text, self._safety)

    def _require_screened(self, token: Any, text: str) -> ScreeningToken:
        if self._screener is not None:
            return cast(
                ScreeningToken,
                cast(Any, self._screener).require(
                    token,
                    text,
                    kind=DEFAULT_PROMPT_KIND,
                ),
            )
        return require_screened(
            token,
            text,
            kind=DEFAULT_PROMPT_KIND,
            config=self._safety,
        )

    def _screen_request(self, token: Any, request: MusicRequest) -> None:
        self._require_screened(token, request.prompt)
        if request.lyrics:
            self._require_screened(self.screen(request.lyrics), request.lyrics)

    def _provider_licenses(self) -> tuple[AssetLicense, ...]:
        """Return the distinct licenses of providers that may serve music."""
        providers = self._router.providers
        licenses: list[AssetLicense] = []
        seen: set[str] = set()
        for name in self._router.chain_for(CAPABILITY_MUSIC_GEN):
            provider = providers.get(name)
            if provider is None or not provider.supports(CAPABILITY_MUSIC_GEN):
                continue
            license = provider.metadata.license
            fingerprint = canonical_json(_license_fingerprint(license))
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            licenses.append(license)
        return tuple(licenses)

    def _default_cache_license(self) -> AssetLicense | None:
        licenses = self._provider_licenses()
        return licenses[0] if len(licenses) == 1 else None

    def _cache_key_for(
        self,
        request: MusicRequest,
        *,
        index: int = 0,
        model: str | None = None,
        license: AssetLicense | Mapping[str, Any] | None = None,
    ) -> AssetCacheKey:
        return music_cache_key(
            request,
            index=index,
            model=model,
            license=license,
        )

    def _adapted_cache_request(
        self,
        request: MusicRequest,
        provider: Provider | None,
    ) -> MusicRequest:
        if provider is None:
            return request
        try:
            return cast(
                MusicRequest,
                adapt_for_cache(request, provider, MediaKind.MUSIC),
            )
        except AdaptationError:
            return request

    def cache_key(
        self,
        request: MusicRequest,
        *,
        index: int = 0,
        model: str | None = None,
        license: AssetLicense | Mapping[str, Any] | None = None,
        provider: Provider | None = None,
    ) -> AssetCacheKey:
        """Return the cache key for one output of *request*.

        The key is built from the request the provider would actually submit,
        so a clamped duration or count is reused on a repeat.
        """
        if not isinstance(request, MusicRequest):
            raise TypeError("request must be a MusicRequest")
        if provider is not None:
            request = self._adapted_cache_request(request, provider)
            if license is None:
                license = provider.metadata.license
        else:
            variants = cache_request_variants(
                request,
                self._router,
                MediaKind.MUSIC,
            )
            if variants:
                request = cast(MusicRequest, variants[0])
            if license is None:
                license = self._default_cache_license()
        return self._cache_key_for(
            request,
            index=index,
            model=model,
            license=license,
        )

    @staticmethod
    def _router_units(request: MusicRequest) -> tuple[float, str]:
        if request.duration_seconds is not None:
            return request.duration_seconds * request.count, MUSIC_UNIT
        return float(request.count), TRACK_UNIT

    @staticmethod
    def _billing_units(
        request: MusicRequest,
        tracks: Sequence[MusicTrack],
        provider: Provider,
        operation: str,
    ) -> tuple[float, str]:
        price = provider.metadata.pricing_for(operation)
        unit = price.unit.strip() if price is not None and price.unit else ""
        if unit.lower() in _TIME_UNITS:
            total = sum(
                track.duration_seconds or request.duration_seconds or 0.0
                for track in tracks
            )
            return total, unit
        if unit:
            return float(len(tracks)), unit
        return float(len(tracks)), TRACK_UNIT

    def _validate(
        self,
        payload: bytes,
        *,
        expected_format: MediaFormat | None,
    ) -> MediaProbe:
        return validate_media(
            payload,
            limits=self._limits,
            expected_format=expected_format,
            ffprobe_executable=self._ffprobe_executable,
            ffprobe_timeout_seconds=(
                self._ffprobe_timeout if self._ffprobe_timeout is not None else 10.0
            ),
            ffprobe_runner=self._ffprobe_runner,
        )

    async def generate(
        self,
        request: MusicRequest,
        *,
        token: Any = None,
        screening_token: Any = None,
        screen_token: Any = None,
        run_id: str = "",
        tracker: CostTracker | None = None,
        operation: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        record_cost: bool | None = None,
        record_sidecar: bool | None = None,
        remaining_budget: float | None = None,
        no_cache: bool | None = None,
        storage: StorageProvider | None = None,
        key: str | None = None,
        key_prefix: str | None = None,
        pinned: bool = True,
        estimated: bool = False,
        exclude: Iterable[str] | None = None,
    ) -> MusicOutcome:
        """Generate *request* through the screened paid path."""
        if not isinstance(request, MusicRequest):
            raise TypeError("request must be a MusicRequest")
        supplied_tokens = [
            value
            for value in (token, screening_token, screen_token)
            if value is not None
        ]
        if len(supplied_tokens) > 1 and any(
            value != supplied_tokens[0] for value in supplied_tokens[1:]
        ):
            raise ValueError("token, screening_token and screen_token disagree")
        active_token = supplied_tokens[0] if supplied_tokens else None
        self._screen_request(active_token, request)

        if tracker is not None and not isinstance(tracker, CostTracker):
            raise TypeError("tracker must be a CostTracker")
        active_tracker = tracker if tracker is not None else self._tracker
        operation_name = self._operation if operation is None else operation.strip()
        if not operation_name:
            raise ValueError("operation must be a non-empty string")
        active_storage = storage if storage is not None else self._storage
        active_prefix = (
            key_prefix.strip().strip("/")
            if key_prefix is not None
            else self._key_prefix
        )
        if key_prefix is not None and not active_prefix:
            raise ValueError("key_prefix must be a non-empty string")
        use_cache = (
            self._cache is not None and not self._no_cache and not bool(no_cache)
        )
        write_sidecar = (
            self._record_sidecar if record_sidecar is None else bool(record_sidecar)
        )
        write_cost = self._record_cost if record_cost is None else bool(record_cost)

        if use_cache:
            cached = await self._cached_outcome(
                cache_request_variants(
                    request,
                    self._router,
                    MediaKind.MUSIC,
                )
                or (request,),
                run_id=run_id,
                operation=operation_name,
                storage=active_storage,
                write_sidecar=write_sidecar,
            )
            if cached is not None:
                self._remember(cached)
                return cached

        budget = remaining_budget
        if budget is None and self._budget is not None:
            budget = self._budget.remaining(run_id, operation=operation_name)

        base_seed = request.seed if request.seed is not None else secrets.randbits(32)
        adaptation: AdaptationResult[Any] | None = None

        async def run(provider: Provider) -> MusicResponse:
            nonlocal adaptation
            adaptation = adapt_for_provider(
                request,
                provider,
                MediaKind.MUSIC,
                seed=base_seed,
            )
            adapted = cast(MusicRequest, adaptation.request)
            return await cast(MusicProvider, provider).generate(adapted)

        units, unit_kind = self._router_units(request)
        response = await self._router.execute(
            CAPABILITY_MUSIC_GEN,
            run,
            remaining_budget=budget,
            units=units,
            unit_kind=unit_kind,
            exclude=exclude,
        )
        if not isinstance(response, MusicResponse):
            raise TypeError("music provider must return a MusicResponse")
        selection = self._router.last_selection
        if selection is None:
            raise RuntimeError("router completed a music call without a selection")
        provider_name = response.provider or selection.name
        model_name = response.model or request.model or ""

        seeds = _response_seeds(request, response, base_seed)
        materialised: list[
            tuple[MediaProbe, StorageObject | None, MusicTrack, AssetLicense]
        ] = []
        for index, track in enumerate(response.tracks):
            probe, stored, updated = await self._materialise(
                track,
                request,
                index=index,
                seed=seeds[index],
                storage=active_storage,
                key=key,
                key_prefix=active_prefix,
                run_id=run_id,
            )
            license = license_from_track(
                updated,
                provider=selection.provider,
                response=response,
            )
            materialised.append((probe, stored, updated, license))
        if not materialised:
            raise ValueError("music provider returned no tracks")

        validated_tracks = [track for _, _, track, _ in materialised]
        response = response.model_copy(
            update={
                "tracks": validated_tracks,
                "provider": provider_name,
                "model": model_name,
            }
        )
        licenses = tuple(license for _, _, _, license in materialised)
        attribution = tuple(
            dict.fromkeys(
                license.attribution
                for license in licenses
                if license.requires_attribution and license.attribution
            )
        )

        billing_units, billing_kind = self._billing_units(
            request,
            validated_tracks,
            selection.provider,
            operation_name,
        )
        price = self._calculator.resolve(
            selection.provider,
            billing_units,
            provider_name=selection.name,
            unit_kind=billing_kind,
            operation=operation_name,
        )
        event = self._calculator.cost_event(
            selection.provider,
            units=billing_units,
            unit_kind=billing_kind,
            response=response,
            track_count=len(validated_tracks),
            provider_name=selection.name,
            run_id=run_id,
            operation=operation_name,
            estimated=estimated,
            metadata=metadata,
            price=price,
        )
        recorded: CostEvent | None = None
        if write_cost and active_tracker is not None:
            recorded = (
                active_tracker.record_estimate(event)
                if estimated
                else active_tracker.record_actual(event)
            )
            event = recorded

        key_request = self._adapted_cache_request(request, selection.provider)
        assets: list[Asset] = []
        stored_objects: list[StorageObject | None] = []
        for index, (probe, stored, track, license) in enumerate(materialised):
            stored_objects.append(stored)
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
                kind=AssetKind.MUSIC,
                label=track.title or "",
                provider=provider_name,
                model=model_name,
                prompt=request.prompt,
                path=(stored.url or stored.key) if stored is not None else "",
                content_hash=probe.sha256,
                run_id=run_id,
                pinned=pinned,
                seed=track.seed,
                license=license,
                analysis=AssetAnalysis(
                    duration_seconds=probe.duration_seconds,
                    format=track.format.value,
                    sample_rate=track.sample_rate,
                    channels=track.channels,
                ),
                cost_links=cost_links,
            )
            if use_cache and self._cache is not None:
                key_for_asset = self._cache_key_for(
                    key_request,
                    index=index,
                    license=selection.provider.metadata.license,
                )
                asset = self._cache.store(asset, key=key_for_asset)
            assets.append(asset)

        sidecars: tuple[Path, ...] = ()
        if write_sidecar:
            sidecars = await self._write_sidecars(
                assets,
                [event] * len(assets),
                run_id=run_id,
                storage=active_storage,
                stored_objects=stored_objects,
            )

        accessibility = await self._write_accessibility_sidecars(
            assets,
            request,
            storage=active_storage,
            stored_objects=stored_objects,
        )
        outcome = MusicOutcome(
            request=request,
            response=response,
            assets=tuple(assets),
            probes=tuple(probe for probe, _, _, _ in materialised),
            storage_objects=tuple(
                stored for stored in stored_objects if stored is not None
            ),
            events=(event,),
            price=price,
            selection=selection,
            sidecars=sidecars,
            accessibility_sidecars=accessibility,
            cache_hit=False,
            licenses=licenses,
            attribution=attribution,
            adaptation_warnings=(adaptation.warnings if adaptation is not None else ()),
        )
        self._remember(outcome)
        return outcome

    async def generate_music(
        self,
        request: MusicRequest,
        **values: Any,
    ) -> MusicOutcome:
        """Alias for :meth:`generate`."""
        return await self.generate(request, **values)

    async def generate_with_cost(
        self,
        request: MusicRequest,
        **values: Any,
    ) -> MusicOutcome:
        """Alias for :meth:`generate` returning the outcome with cost data."""
        return await self.generate(request, **values)

    def _remember(self, outcome: MusicOutcome) -> None:
        self._last_outcome = outcome
        self._last_selection = outcome.selection
        self._last_event = outcome.event
        self._last_price = outcome.price

    async def _materialise(
        self,
        track: MusicTrack,
        request: MusicRequest,
        *,
        index: int,
        seed: int | None,
        storage: StorageProvider | None,
        key: str | None,
        key_prefix: str | None,
        run_id: str,
    ) -> tuple[MediaProbe, StorageObject | None, MusicTrack]:
        if track.data is not None:
            payload = track.data
            probe = self._validate(
                payload,
                expected_format=_media_format(track.format),
            )
        else:
            if track.url is None:
                raise ValueError("music track has neither data nor url")
            probe = await self._downloader_for().download(
                track.url,
                limits=self._limits,
                expected_format=_media_format(track.format),
            )
            payload = probe.data

        stored: StorageObject | None = None
        if storage is not None:
            storage_key = self._storage_key(
                probe,
                index=index,
                key=key,
                key_prefix=key_prefix,
                count=request.count,
            )
            stored = await storage.put(
                StoragePutRequest(
                    data=payload,
                    key=storage_key,
                    content_type=probe.mime_type,
                    metadata={
                        "kind": "music",
                        "format": probe.format.value,
                        "provider": track.metadata.get("provider", ""),
                        "run_id": run_id,
                    },
                )
            )
        fmt, sample_rate, channels = _measured_audio(
            probe,
            fallback_format=track.format,
            fallback_sample_rate=track.sample_rate,
            fallback_channels=track.channels,
        )
        metadata = dict(track.metadata)
        metadata["duration_seconds"] = probe.duration_seconds
        metadata["format"] = fmt.value
        updated = track.model_copy(
            update={
                "data": payload,
                "format": fmt,
                "duration_seconds": probe.duration_seconds,
                "sample_rate": sample_rate,
                "channels": channels,
                "seed": seed,
                "metadata": metadata,
            }
        )
        return probe, stored, updated

    def _downloader_for(self) -> MediaDownloader:
        if self._downloader is None:
            self._downloader = MediaDownloader(
                http=self._http,
                transport=self._transport,
                limits=self._limits,
                ffprobe_executable=self._ffprobe_executable,
                ffprobe_timeout_seconds=(
                    self._ffprobe_timeout if self._ffprobe_timeout is not None else 10.0
                ),
                ffprobe_runner=self._ffprobe_runner,
            )
        return self._downloader

    @staticmethod
    def _storage_key(
        probe: MediaProbe,
        *,
        index: int,
        key: str | None,
        key_prefix: str | None,
        count: int,
    ) -> str | None:
        if key:
            base = key.strip()
            if not base:
                raise ValueError("storage key must be non-empty")
            if count > 1:
                return f"{base}-{index}.{probe.extension}"
            return base
        if key_prefix:
            digest = probe.sha256
            return f"{key_prefix}/{digest[:2]}/{digest}.{probe.extension}"
        return None

    async def _cached_outcome(
        self,
        requests: Iterable[MusicRequest],
        *,
        run_id: str,
        operation: str,
        storage: StorageProvider | None,
        write_sidecar: bool,
    ) -> MusicOutcome | None:
        cache = self._cache
        if cache is None:
            return None
        keys: list[AssetCacheKey] = []
        assets: list[Asset] = []
        payloads: list[bytes] = []
        matched: MusicRequest | None = None
        for request in requests:
            for license in self._provider_licenses():
                candidate_keys = [
                    self._cache_key_for(request, index=index, license=license)
                    for index in range(request.count)
                ]
                candidate_assets: list[Asset] = []
                for key in candidate_keys:
                    entry = cache.entry(key)
                    if entry is None:
                        break
                    asset = cache.assets.get(entry.asset_id)
                    if asset is None or asset.kind is not AssetKind.MUSIC:
                        break
                    candidate_assets.append(asset)
                else:
                    candidate_payloads: list[bytes] = []
                    for asset in candidate_assets:
                        payload = await self._cached_payload(asset, storage)
                        if payload is None:
                            break
                        candidate_payloads.append(payload)
                    else:
                        keys = candidate_keys
                        assets = candidate_assets
                        payloads = candidate_payloads
                        matched = request
                        break
            if matched is not None:
                break
        else:
            return None
        if matched is None:  # pragma: no cover - the loop sets it on a hit
            return None

        events: list[CostEvent] = [
            cache.record_hit(key, asset, run_id=run_id, operation=operation)
            for key, asset in zip(keys, assets, strict=True)
        ]
        tracks: list[MusicTrack] = []
        probes: list[MediaProbe] = []
        for payload, asset in zip(payloads, assets, strict=True):
            probe = self._validate(
                payload,
                expected_format=None,
            )
            probes.append(probe)
            metadata = asset.license.model_dump(mode="json")
            metadata["cache"] = "hit"
            fmt = _stored_audio_format(
                asset.analysis,
                _audio_format(probe.format.value, request.format),
            )
            tracks.append(
                MusicTrack(
                    data=payload,
                    format=fmt,
                    duration_seconds=probe.duration_seconds,
                    sample_rate=(
                        asset.analysis.sample_rate
                        if asset.analysis.sample_rate is not None
                        else request.sample_rate
                    ),
                    channels=asset.analysis.channels,
                    seed=asset.seed,
                    title=asset.label or request.prompt,
                    lyrics=None if request.instrumental else request.lyrics,
                    metadata=metadata,
                )
            )
        response = MusicResponse(
            tracks=tracks,
            model=assets[0].model,
            provider=assets[0].provider,
            metadata={"cache": "hit"},
        )
        licenses = tuple(asset.license for asset in assets)
        attribution = tuple(
            dict.fromkeys(
                license.attribution
                for license in licenses
                if license.requires_attribution and license.attribution
            )
        )
        sidecars: tuple[Path, ...] = ()
        if write_sidecar:
            sidecars = await self._write_sidecars(
                assets,
                events,
                run_id=run_id,
                storage=storage,
                stored_objects=[None] * len(assets),
            )
        accessibility = await self._write_accessibility_sidecars(
            assets,
            request,
            storage=storage,
            stored_objects=[None] * len(assets),
        )
        return MusicOutcome(
            request=request,
            response=response,
            assets=tuple(assets),
            probes=tuple(probes),
            events=tuple(events),
            sidecars=sidecars,
            accessibility_sidecars=accessibility,
            cache_hit=True,
            licenses=licenses,
            attribution=attribution,
        )

    async def _cached_payload(
        self,
        asset: Asset,
        storage: StorageProvider | None,
    ) -> bytes | None:
        if asset.path:
            path = _file_path(asset.path)
            if path is not None and path.is_file():
                return path.read_bytes()
        if storage is not None and asset.content_hash:
            digest = asset.content_hash
            key = f"objects/{digest[:2]}/{digest}"
            try:
                return await storage.get(key)
            except KeyError:
                return None
        return None

    async def _write_accessibility_sidecars(
        self,
        assets: Sequence[Asset],
        request: MusicRequest,
        *,
        storage: StorageProvider | None,
        stored_objects: Sequence[StorageObject | None],
    ) -> tuple[Any, ...]:
        if self._accessibility is None:
            return ()
        metadata = request.metadata or {}
        written: list[Any] = []
        for index, asset in enumerate(assets):
            if not asset.path:
                continue
            values: dict[str, Any] = {
                "alt_text": (
                    metadata.get("alt_text")
                    or metadata.get("alt")
                    or metadata.get("title")
                    or asset.label
                    or request.prompt
                ),
                "subtitle": metadata.get("subtitle"),
                "words": metadata.get("words"),
                "transcript": metadata.get("transcript"),
                "text": metadata.get("text") or request.lyrics or request.prompt,
                "duration_seconds": asset.analysis.duration_seconds,
            }
            path = _file_path(asset.path)
            result: Any
            if path is not None:
                result = write_accessibility_sidecars(
                    path,
                    self._accessibility,
                    **values,
                )
            else:
                if storage is None:
                    continue
                stored = stored_objects[index] if index < len(stored_objects) else None
                result = await store_accessibility_sidecars(
                    asset.path,
                    storage,
                    self._accessibility,
                    key=stored.key if stored is not None else None,
                    **values,
                )
            if result:
                written.append(result)
        return tuple(written)

    async def _write_sidecars(
        self,
        assets: Sequence[Asset],
        events: Sequence[CostEvent],
        *,
        run_id: str,
        storage: StorageProvider | None,
        stored_objects: Sequence[StorageObject | None],
    ) -> tuple[Path, ...]:
        written: list[Path] = []
        for index, asset in enumerate(assets):
            asset_events = [events[index]] if index < len(events) else list(events)
            if not asset_events:
                continue
            path = _file_path(asset.path) if asset.path else None
            if path is not None:
                written.append(
                    write_cost_sidecar(
                        path,
                        asset_events,
                        toolchain_fingerprint=self._toolchain,
                        run_id=run_id or None,
                    )
                )
                continue
            stored = stored_objects[index] if index < len(stored_objects) else None
            if storage is None or stored is None:
                continue
            sidecar = build_cost_sidecar(
                stored.url or stored.key,
                asset_events,
                toolchain_fingerprint=self._toolchain,
                run_id=run_id or None,
            )
            payload = (canonical_json(sidecar.to_dict(), indent=2) + "\n").encode(
                "utf-8"
            )
            await storage.put(
                StoragePutRequest(
                    data=payload,
                    key=f"{stored.key}{SIDECAR_SUFFIX}",
                    content_type="application/json",
                    overwrite=True,
                )
            )
        return tuple(written)


async def generate_music(
    request: MusicRequest,
    router: Router,
    *,
    tracker: CostTracker | None = None,
    cache: AssetCache | None = None,
    storage: StorageProvider | None = None,
    budget: BudgetLike = None,
    safety: Any = None,
    limits: MediaLimits | None = None,
    pricing_overrides: Mapping[str, float] | None = None,
    pricing_version: str = "",
    run_id: str = "",
    metadata: Mapping[str, Any] | None = None,
    remaining_budget: float | None = None,
    **values: Any,
) -> MusicOutcome:
    """Execute one music request without constructing a long-lived service."""
    service = MusicService(
        router,
        tracker=tracker,
        cache=cache,
        storage=storage,
        budget=budget,
        safety=safety,
        limits=limits,
        pricing_overrides=pricing_overrides,
        pricing_version=pricing_version,
    )
    return await service.generate(
        request,
        run_id=run_id,
        metadata=metadata,
        remaining_budget=remaining_budget,
        **values,
    )


def music_service_from_config(
    router: Router,
    config: Any = None,
    **values: Any,
) -> MusicService:
    """Build a :class:`MusicService` for *router* and an optional config.

    A mapping with ``chain`` or ``strategy`` keys is applied as the
    ``music_gen`` capability chain, mirroring the image, video and speech
    service helpers.
    """
    if isinstance(config, Mapping):
        section = config.get("music_gen", config)
        if isinstance(section, Mapping) and (
            "chain" in section or "strategy" in section
        ):
            base = router.config_for(CAPABILITY_MUSIC_GEN)
            merged = base.model_copy(
                update={
                    key: section[key]
                    for key in ("chain", "strategy", "weights", "pricing_overrides")
                    if key in section
                }
            )
            router = Router(
                router.providers,
                configs={CAPABILITY_MUSIC_GEN: merged},
                seed=router.seed,
                circuit_breakers=router.circuits,
                rate_limiters=router.limiters,
            )
    return MusicService(router, **values)
