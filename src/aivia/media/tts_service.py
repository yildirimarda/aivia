"""Run one speech synthesis request through the single paid path.

The service is the only place a text to speech call is executed. It screens the
text, answers an identical request from the asset cache when possible, applies
the remaining budget, asks the router for a provider, measures and processes the
returned audio, stores it, records one actual cost event and writes a cost
sidecar next to the output.

The order matches the project invariant: screen -> cache -> budget -> router ->
provider -> validate -> store -> cost -> sidecar. A cache hit short-circuits the
paid path but still records a zero-amount event, so reused speech is visible in
the ledger without counting as spend.

Post-processing is deterministic and AI-free: the requested trim range and
fades are applied with ffmpeg and the result is optionally normalized to a
target loudness. Duration always comes from ffprobe on the encoded payload, not
from provider metadata, because providers disagree about what a "duration"
means for a compressed container.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast
from urllib.parse import unquote, urlsplit

from sqlalchemy import Engine

from aivia.core.asset_cache import AssetCache, AssetCacheKey
from aivia.core.budget import BudgetGuard
from aivia.core.config.budget import BudgetConfig
from aivia.core.cost import CostTracker
from aivia.core.errors import ValidationError
from aivia.core.media_probe import MediaLimits, MediaProbe, validate_media
from aivia.core.models.asset import Asset, AssetAnalysis, AssetCostLink, AssetKind
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
from aivia.media.adaptation import adapt_for_provider
from aivia.media.audio_utils import (
    AudioProcessing,
    AudioSettings,
    SpeechSettings,
    _positive,
    audio_duration,
    fade_audio,
    measure_audio_duration,
    normalize_audio,
    normalize_loudness,
    probe_audio_duration,
    process_audio,
    trim_audio,
)
from aivia.media.screening import (
    DEFAULT_PROMPT_KIND,
    ScreeningToken,
    require_screened,
    screen_prompt,
)
from aivia.media.voices import prepare_speech_request
from aivia.providers.base import Provider
from aivia.providers.media.adapt import AdaptationResult, AdaptationWarning
from aivia.providers.media.dto import MediaKind
from aivia.providers.protocols.audio import AudioFormat
from aivia.providers.protocols.storage import (
    StorageObject,
    StorageProvider,
    StoragePutRequest,
)
from aivia.providers.protocols.tts import (
    CAPABILITY_TTS,
    TTSProvider,
    TTSRequest,
    TTSResponse,
)
from aivia.providers.tts.cost import (
    CHARACTER_UNIT,
    TTS_OPERATION,
    TTSCostCalculator,
)

__all__ = [
    "TTS_OPERATION",
    "AudioProcessing",
    "AudioSettings",
    "SpeechGeneration",
    "SpeechOutcome",
    "SpeechService",
    "SpeechServiceResult",
    "SpeechSettings",
    "TTSGeneration",
    "TTSOutcome",
    "TTSResult",
    "TTSService",
    "audio_duration",
    "fade_audio",
    "generate_speech",
    "measure_audio_duration",
    "normalize_audio",
    "normalize_loudness",
    "probe_audio_duration",
    "process_audio",
    "speech_cache_key",
    "speech_service_from_config",
    "synthesize_speech",
    "trim_audio",
    "tts_service_from_config",
]

ToolchainLike = ToolchainFingerprint | Mapping[str, Any] | str | None
BudgetLike = BudgetGuard | BudgetConfig | Mapping[str, Any] | None


def _file_path(url: str) -> Path | None:
    parsed = urlsplit(url)
    if parsed.scheme.lower() != "file":
        return None
    return Path(unquote(parsed.path))


def _audio_format(
    value: str | AudioFormat | None,
    fallback: AudioFormat,
) -> AudioFormat:
    if isinstance(value, AudioFormat):
        return value
    if value is None:
        return fallback
    try:
        return AudioFormat(str(value).lower())
    except ValueError:
        return fallback


def _payload_format(payload: bytes, fallback: AudioFormat) -> AudioFormat:
    from aivia.core.media_probe import sniff_media

    signature = sniff_media(payload)
    if signature is None:
        return fallback
    return _audio_format(signature.format.value, fallback)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _redacted_message(error: BaseException) -> str:
    text = str(error).strip()
    return text or type(error).__name__


@dataclass(frozen=True, slots=True)
class SpeechOutcome:
    """The response, stored asset, price, events and sidecars for one call.

    ``events`` holds one zero-amount event on a cache hit, or the one actual
    event for the provider call on a miss. ``duration_seconds`` is the measured
    length of the processed audio, which may differ from the provider's claim.
    """

    request: TTSRequest
    response: TTSResponse
    settings: AudioSettings
    assets: tuple[Asset, ...] = ()
    probes: tuple[MediaProbe, ...] = ()
    storage_objects: tuple[StorageObject, ...] = ()
    events: tuple[CostEvent, ...] = ()
    price: ResolvedPrice | None = None
    selection: RouteSelection | None = None
    sidecars: tuple[Path, ...] = ()
    accessibility_sidecars: tuple[Any, ...] = ()
    cache_hit: bool = False
    processed: bool = False
    failures: tuple[Mapping[str, Any], ...] = ()
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
    def audio(self) -> bytes:
        """The processed audio payload."""
        return self.response.audio

    @property
    def data(self) -> bytes:
        """Alias for :attr:`audio`."""
        return self.audio

    @property
    def audio_bytes(self) -> bytes:
        """Alias for :attr:`audio`."""
        return self.audio

    @property
    def format(self) -> AudioFormat:
        """The processed container format."""
        return self.response.format

    @property
    def duration_seconds(self) -> float | None:
        """The measured duration of the processed audio."""
        return self.response.duration_seconds

    @property
    def duration(self) -> float | None:
        """Alias for :attr:`duration_seconds`."""
        return self.duration_seconds

    @property
    def voice(self) -> str:
        """The voice that served or cached the call."""
        return self.response.voice

    @property
    def provider(self) -> str:
        """The provider that served or cached the call."""
        return self.response.provider

    @property
    def model(self) -> str:
        """The model that served or cached the call."""
        return self.response.model

    @property
    def asset(self) -> Asset | None:
        """The stored speech asset, or ``None``."""
        return self.assets[0] if self.assets else None

    @property
    def first(self) -> Asset | None:
        """Alias for :attr:`asset`."""
        return self.asset

    @property
    def output(self) -> Asset | None:
        """Alias for :attr:`asset`."""
        return self.asset

    @property
    def path(self) -> str | None:
        """The stored asset path, or ``None``."""
        return self.asset.path if self.asset is not None else None

    @property
    def url(self) -> str | None:
        """Alias for :attr:`path`."""
        return self.path

    @property
    def sidecar(self) -> Path | None:
        """The first written sidecar, or ``None``."""
        return self.sidecars[0] if self.sidecars else None

    @property
    def sidecar_path(self) -> Path | None:
        """Alias for :attr:`sidecar`."""
        return self.sidecar

    @property
    def cached(self) -> bool:
        """Whether this outcome came from the asset cache."""
        return self.cache_hit

    @property
    def amount_usd(self) -> float:
        """The sum of every recorded event amount."""
        return math.fsum(event.amount_usd for event in self.events)

    @property
    def cost_usd(self) -> float:
        """Alias for :attr:`amount_usd`."""
        return self.amount_usd

    @property
    def total_cost_usd(self) -> float:
        """Alias for :attr:`amount_usd`."""
        return self.amount_usd

    @property
    def failure(self) -> Mapping[str, Any] | None:
        """The first recorded failure, or ``None``."""
        return self.failures[0] if self.failures else None

    @property
    def failed(self) -> bool:
        """Whether any attempt failed."""
        return bool(self.failures)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary for logs and manifests."""
        return {
            "provider": self.provider,
            "model": self.model,
            "voice": self.voice,
            "format": self.format.value,
            "duration_seconds": self.duration_seconds,
            "cache_hit": self.cache_hit,
            "amount_usd": self.amount_usd,
            "path": self.path,
            "adaptation_warnings": [
                warning.to_dict() for warning in self.adaptation_warnings
            ],
        }


SpeechServiceResult = SpeechOutcome
"""Alias for :class:`SpeechOutcome`."""

TTSOutcome = SpeechOutcome
"""Alias for :class:`SpeechOutcome`."""

TTSResult = SpeechOutcome
"""Alias for :class:`SpeechOutcome`."""

SpeechGeneration = SpeechOutcome
"""Alias for :class:`SpeechOutcome`."""

TTSGeneration = SpeechOutcome
"""Alias for :class:`SpeechOutcome`."""


def speech_cache_key(
    request: TTSRequest,
    settings: AudioSettings | Mapping[str, Any] | None = None,
    *,
    model: str | None = None,
) -> AssetCacheKey:
    """Return the cache key for *request* and its processing settings.

    The provider is deliberately left empty so an identical text, voice and
    settings request is answered before the router has selected a provider. The
    stored asset still records who actually served it.
    """
    if not isinstance(request, TTSRequest):
        raise TypeError("request must be a TTSRequest")
    active = AudioSettings.from_value(settings)
    parameters: dict[str, Any] = {
        "voice": request.voice,
        "model": request.model,
        "language": request.language,
        "format": request.format.value,
        "sample_rate": request.sample_rate,
        "speed": request.speed,
        "instructions": request.instructions,
        "pronunciation": dict(sorted(request.pronunciation.items())),
        "seed": request.seed,
        "word_timestamps": request.word_timestamps,
        "settings": active.cache_parameters(),
    }
    return AssetCacheKey(
        provider="",
        model=(request.model if model is None else model) or "",
        prompt=request.text,
        parameters=parameters,
    )


class SpeechService:
    """Execute speech synthesis through a :class:`~aivia.core.router.Router`.

    ``cache`` answers an identical text, voice and settings request without a
    provider call and persists the processed audio of a miss. When it is
    omitted but a tracker or engine is given, the service builds an
    :class:`~aivia.core.asset_cache.AssetCache` on that ledger so caching is
    never silently disabled. ``storage`` stores the processed bytes; without
    one the asset record is still built but no file is written.

    ``settings`` supplies the default loudness, trim and fade options. A call
    may override them, and the cache key includes whichever settings were used.
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
        settings: AudioSettings | Mapping[str, Any] | None = None,
        pricing_overrides: Mapping[str, float] | None = None,
        pricing_version: str = "",
        operation: str = TTS_OPERATION,
        toolchain: ToolchainLike = None,
        toolchain_fingerprint: ToolchainLike = None,
        key_prefix: str | None = None,
        record_cost: bool = True,
        record_sidecar: bool = True,
        accessibility: Any = None,
        accessibility_config: Any = None,
        no_cache: bool = False,
        process: bool = True,
        ffmpeg_executable: str = "ffmpeg",
        ffmpeg_runner: Any = None,
        ffmpeg_timeout_seconds: float | None = None,
        ffprobe_executable: str = "ffprobe",
        ffprobe_runner: Any = None,
        ffprobe_timeout_seconds: float | None = None,
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
        if cache is not None and not isinstance(cache, AssetCache):
            raise TypeError("cache must be an AssetCache")
        if engine is not None and not isinstance(engine, Engine):
            raise TypeError("engine must be a SQLAlchemy Engine")
        if settings is not None and not isinstance(settings, (AudioSettings, Mapping)):
            raise TypeError("settings must be an AudioSettings, mapping or None")

        active_tracker = tracker if tracker is not None else cost_tracker
        if active_tracker is not None and not isinstance(active_tracker, CostTracker):
            raise TypeError("tracker must be a CostTracker")

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
        self._settings = AudioSettings.from_value(settings)
        self._operation = operation.strip()
        self._toolchain = toolchain if toolchain is not None else toolchain_fingerprint
        self._key_prefix = key_prefix
        self._record_cost = bool(record_cost)
        self._record_sidecar = bool(record_sidecar)
        self._accessibility = (
            accessibility if accessibility is not None else accessibility_config
        )
        self._no_cache = bool(no_cache)
        self._process = bool(process)
        self._ffmpeg_executable = ffmpeg_executable
        self._ffmpeg_runner = ffmpeg_runner
        self._ffmpeg_timeout = ffmpeg_timeout_seconds
        self._ffprobe_executable = ffprobe_executable
        self._ffprobe_runner = ffprobe_runner
        self._ffprobe_timeout = ffprobe_timeout_seconds
        self._calculator = TTSCostCalculator(
            pricing_overrides,
            pricing_version=pricing_version,
        )
        self._last_outcome: SpeechOutcome | None = None
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
        """The caps applied to every stored payload."""
        return self._limits

    @property
    def settings(self) -> AudioSettings:
        """The default post-processing settings."""
        return self._settings

    @property
    def calculator(self) -> TTSCostCalculator:
        """The calculator that prices and records speech calls."""
        return self._calculator

    @property
    def last_outcome(self) -> SpeechOutcome | None:
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

    def _screen(self, token: Any, text: str) -> ScreeningToken:
        if token is not None:
            return self._require_screened(token, text)
        return self.screen(text)

    def cache_key(
        self,
        request: TTSRequest,
        settings: AudioSettings | Mapping[str, Any] | None = None,
    ) -> AssetCacheKey:
        """Return the cache key for one speech request."""
        if not isinstance(request, TTSRequest):
            raise TypeError("request must be a TTSRequest")
        active = AudioSettings.from_value(
            settings if settings is not None else self._settings
        )
        return speech_cache_key(request, active)

    def _measure_duration(
        self,
        payload: bytes,
        *,
        fmt: AudioFormat,
        fallback: float | None = None,
    ) -> float | None:
        try:
            return measure_audio_duration(
                payload,
                format=fmt,
                ffprobe_executable=self._ffprobe_executable,
                ffprobe_runner=self._ffprobe_runner,
                timeout_seconds=(
                    self._ffprobe_timeout if self._ffprobe_timeout is not None else 10.0
                ),
            )
        except ValidationError:
            if fallback is not None:
                return fallback
            raise

    def _process_audio(
        self,
        payload: bytes,
        response: TTSResponse,
        settings: AudioSettings,
    ) -> tuple[bytes, AudioFormat, bool]:
        source_format = response.format
        output_format = _audio_format(settings.format, source_format)
        input_duration = response.duration_seconds
        if settings.has_processing and self._process:
            if (
                input_duration is None
                and _positive(settings.fade_out_seconds)
                and settings.trim_end_seconds is None
            ):
                input_duration = self._measure_duration(
                    payload,
                    fmt=source_format,
                )
            with TemporaryDirectory(prefix="aivia-tts-") as directory:
                destination = Path(directory) / f"speech.{output_format.value}"
                process_audio(
                    payload,
                    destination,
                    settings,
                    duration_seconds=input_duration,
                    output_format=output_format,
                    ffmpeg_executable=self._ffmpeg_executable,
                    ffmpeg_runner=self._ffmpeg_runner,
                    timeout_seconds=(
                        self._ffmpeg_timeout
                        if self._ffmpeg_timeout is not None
                        else 300.0
                    ),
                )
                processed = destination.read_bytes()
            return processed, output_format, True
        return payload, source_format, False

    async def synthesize(
        self,
        request: TTSRequest,
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
        settings: AudioSettings | Mapping[str, Any] | None = None,
        process: bool | None = None,
    ) -> SpeechOutcome:
        """Synthesize *request* through the screened paid path."""
        if not isinstance(request, TTSRequest):
            raise TypeError("request must be a TTSRequest")
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
        self._screen(active_token, request.text)

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
        active_settings = AudioSettings.from_value(
            settings if settings is not None else self._settings
        )
        should_process = self._process if process is None else bool(process)
        use_cache = (
            self._cache is not None and not self._no_cache and not bool(no_cache)
        )
        write_sidecar = (
            self._record_sidecar if record_sidecar is None else bool(record_sidecar)
        )
        write_cost = self._record_cost if record_cost is None else bool(record_cost)

        if use_cache:
            cached = await self._cached_outcome(
                request,
                settings=active_settings,
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

        units = float(len(request.text))
        adaptation: AdaptationResult[Any] | None = None

        async def run(provider: Provider) -> TTSResponse:
            nonlocal adaptation
            prepared = prepare_speech_request(request, provider=provider)
            adaptation = adapt_for_provider(
                prepared,
                provider,
                MediaKind.SPEECH,
            )
            adapted = cast(TTSRequest, adaptation.request)
            return await cast(TTSProvider, provider).synthesize(adapted)

        response = await self._router.execute(
            CAPABILITY_TTS,
            run,
            remaining_budget=budget,
            units=units,
            unit_kind=CHARACTER_UNIT,
            exclude=exclude,
        )
        if not isinstance(response, TTSResponse):
            raise TypeError("speech provider must return a TTSResponse")
        selection = self._router.last_selection
        if selection is None:
            raise RuntimeError("router completed a speech call without a selection")
        provider_name = response.provider or selection.name
        model_name = response.model or request.model or ""

        if should_process:
            payload, output_format, processed = self._process_audio(
                response.audio,
                response,
                active_settings,
            )
        else:
            payload = response.audio
            output_format = response.format
            processed = False

        probe = validate_media(
            payload,
            limits=self._limits,
            ffprobe_executable=self._ffprobe_executable,
            ffprobe_timeout_seconds=(
                self._ffprobe_timeout if self._ffprobe_timeout is not None else 10.0
            ),
            ffprobe_runner=self._ffprobe_runner,
        )
        duration = probe.duration_seconds
        if duration is None:
            duration = response.duration_seconds

        response_metadata = dict(response.metadata)
        response_metadata.setdefault("characters", len(request.text))
        response_metadata["processed"] = processed
        if duration is not None:
            response_metadata["duration_seconds"] = duration
        response = response.model_copy(
            update={
                "audio": payload,
                "format": output_format,
                "duration_seconds": duration,
                "provider": provider_name,
                "model": model_name,
                "metadata": response_metadata,
            }
        )

        stored = await self._store_payload(
            payload,
            request,
            output_format,
            provider=provider_name,
            model=model_name,
            run_id=run_id,
            storage=active_storage,
            key=key,
            key_prefix=active_prefix,
        )

        resolved = self._calculator.resolve(
            selection.provider,
            units,
            provider_name=selection.name,
            operation=operation_name,
        )
        event = self._calculator.cost_event(
            selection.provider,
            characters=units,
            response=response,
            provider_name=selection.name,
            run_id=run_id,
            operation=operation_name,
            estimated=estimated,
            metadata=metadata,
            price=resolved,
        )
        recorded: CostEvent | None = None
        if write_cost and active_tracker is not None:
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
        asset = Asset(
            kind=AssetKind.VOICE,
            provider=provider_name,
            model=model_name,
            prompt=request.text,
            path=(stored.url or stored.key) if stored is not None else "",
            content_hash=_sha256(payload),
            run_id=run_id,
            pinned=pinned,
            seed=request.seed,
            analysis=AssetAnalysis(duration_seconds=duration),
            cost_links=cost_links,
        )
        if use_cache and self._cache is not None:
            key_for_asset = self.cache_key(request, active_settings)
            asset = self._cache.store(asset, key=key_for_asset)
        sidecars: tuple[Path, ...] = ()
        if write_sidecar:
            sidecars = await self._write_sidecars(
                asset,
                event,
                run_id=run_id,
                storage=active_storage,
                stored=stored,
            )

        accessibility = await self._write_accessibility_sidecars(
            asset,
            request,
            response,
            storage=active_storage,
            stored=stored,
        )
        outcome = SpeechOutcome(
            request=request,
            response=response,
            settings=active_settings,
            assets=(asset,),
            probes=(probe,),
            storage_objects=(stored,) if stored is not None else (),
            events=(event,),
            price=resolved,
            selection=selection,
            sidecars=sidecars,
            accessibility_sidecars=accessibility,
            cache_hit=False,
            processed=processed,
            adaptation_warnings=(adaptation.warnings if adaptation is not None else ()),
        )
        self._remember(outcome)
        return outcome

    async def generate(self, request: TTSRequest, **values: Any) -> SpeechOutcome:
        """Alias for :meth:`synthesize`."""
        return await self.synthesize(request, **values)

    async def generate_speech(
        self,
        request: TTSRequest,
        **values: Any,
    ) -> SpeechOutcome:
        """Alias for :meth:`synthesize`."""
        return await self.synthesize(request, **values)

    async def generate_with_cost(
        self,
        request: TTSRequest,
        **values: Any,
    ) -> SpeechOutcome:
        """Alias for :meth:`synthesize` returning the outcome with cost data."""
        return await self.synthesize(request, **values)

    def _remember(self, outcome: SpeechOutcome) -> None:
        self._last_outcome = outcome
        self._last_selection = outcome.selection
        self._last_event = outcome.event
        self._last_price = outcome.price

    async def _store_payload(
        self,
        payload: bytes,
        request: TTSRequest,
        fmt: AudioFormat,
        *,
        provider: str,
        model: str,
        run_id: str,
        storage: StorageProvider | None,
        key: str | None,
        key_prefix: str | None,
    ) -> StorageObject | None:
        if storage is None:
            return None
        storage_key = key.strip() if isinstance(key, str) else None
        if storage_key == "":
            raise ValueError("storage key must be non-empty")
        if storage_key is None and key_prefix:
            digest = _sha256(payload)
            storage_key = f"{key_prefix}/{digest[:2]}/{digest}.{fmt.value}"
        return await storage.put(
            StoragePutRequest(
                data=payload,
                key=storage_key,
                content_type=f"audio/{fmt.value}",
                metadata={
                    "kind": "speech",
                    "format": fmt.value,
                    "provider": provider,
                    "model": model,
                    "voice": request.voice,
                    "run_id": run_id,
                },
            )
        )

    async def _cached_outcome(
        self,
        request: TTSRequest,
        *,
        settings: AudioSettings,
        run_id: str,
        operation: str,
        storage: StorageProvider | None,
        write_sidecar: bool,
    ) -> SpeechOutcome | None:
        cache = self._cache
        if cache is None:
            return None
        key = self.cache_key(request, settings)
        entry = cache.entry(key)
        if entry is None:
            return None
        asset = cache.assets.get(entry.asset_id)
        if asset is None or asset.kind is not AssetKind.VOICE:
            return None
        payload = await self._cached_payload(asset, storage)
        if payload is None:
            return None
        fmt = _payload_format(payload, request.format)
        duration = asset.analysis.duration_seconds
        if duration is None:
            duration = self._measure_duration(payload, fmt=fmt)
        event = cache.record_hit(key, asset, run_id=run_id, operation=operation)
        response = TTSResponse(
            audio=payload,
            format=fmt,
            sample_rate=request.sample_rate,
            duration_seconds=duration,
            voice=request.voice,
            model=asset.model,
            provider=asset.provider,
            metadata={
                "characters": len(request.text),
                "cache": "hit",
                "duration_seconds": duration,
            },
        )
        sidecars: tuple[Path, ...] = ()
        if write_sidecar:
            sidecars = await self._write_sidecars(
                asset,
                event,
                run_id=run_id,
                storage=storage,
                stored=None,
            )
        accessibility = await self._write_accessibility_sidecars(
            asset,
            request,
            response,
            storage=storage,
            stored=None,
        )
        outcome = SpeechOutcome(
            request=request,
            response=response,
            settings=settings,
            assets=(asset,),
            events=(event,),
            sidecars=sidecars,
            accessibility_sidecars=accessibility,
            cache_hit=True,
        )
        return outcome

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
            try:
                return await storage.get(f"objects/{digest[:2]}/{digest}")
            except KeyError:
                return None
        return None

    async def _write_accessibility_sidecars(
        self,
        asset: Asset,
        request: TTSRequest,
        response: TTSResponse,
        *,
        storage: StorageProvider | None,
        stored: StorageObject | None,
    ) -> tuple[Any, ...]:
        if self._accessibility is None or not asset.path:
            return ()
        metadata = request.metadata or {}
        values: dict[str, Any] = {
            "alt_text": metadata.get("alt_text") or metadata.get("alt") or request.text,
            "subtitle": metadata.get("subtitle"),
            "words": response.word_timestamps or metadata.get("words"),
            "transcript": metadata.get("transcript"),
            "text": request.text,
            "language": request.language or "",
            "duration_seconds": (
                response.duration_seconds or asset.analysis.duration_seconds
            ),
        }
        path = _file_path(asset.path)
        result: Any
        if path is not None:
            result = write_accessibility_sidecars(
                path,
                self._accessibility,
                **values,
            )
            return (result,) if result else ()
        if storage is None:
            return ()
        result = await store_accessibility_sidecars(
            asset.path,
            storage,
            self._accessibility,
            key=stored.key if stored is not None else None,
            **values,
        )
        return (result,) if result else ()

    async def _write_sidecars(
        self,
        asset: Asset,
        event: CostEvent,
        *,
        run_id: str,
        storage: StorageProvider | None,
        stored: StorageObject | None,
    ) -> tuple[Path, ...]:
        path = _file_path(asset.path) if asset.path else None
        if path is not None:
            return (
                write_cost_sidecar(
                    path,
                    [event],
                    toolchain_fingerprint=self._toolchain,
                    run_id=run_id or None,
                ),
            )
        if storage is None or stored is None:
            return ()
        sidecar = build_cost_sidecar(
            stored.url or stored.key,
            [event],
            toolchain_fingerprint=self._toolchain,
            run_id=run_id or None,
        )
        payload = (canonical_json(sidecar.to_dict(), indent=2) + "\n").encode("utf-8")
        await storage.put(
            StoragePutRequest(
                data=payload,
                key=f"{stored.key}{SIDECAR_SUFFIX}",
                content_type="application/json",
                overwrite=True,
            )
        )
        return ()


async def synthesize_speech(
    request: TTSRequest,
    router: Router,
    *,
    tracker: CostTracker | None = None,
    cache: AssetCache | None = None,
    storage: StorageProvider | None = None,
    settings: AudioSettings | Mapping[str, Any] | None = None,
    **values: Any,
) -> SpeechOutcome:
    """Execute one speech request without constructing a long-lived service."""
    service = SpeechService(
        router,
        tracker=tracker,
        cache=cache,
        storage=storage,
        settings=settings,
    )
    return await service.synthesize(request, **values)


generate_speech = synthesize_speech
"""Alias for :func:`synthesize_speech`."""


def speech_service_from_config(
    router: Router,
    config: Any = None,
    **values: Any,
) -> SpeechService:
    """Build a :class:`SpeechService` for *router* and an optional config.

    A mapping with ``chain`` or ``strategy`` keys is applied as the ``tts``
    capability chain, mirroring the image and video service helpers.
    """
    if isinstance(config, Mapping):
        section = config.get("tts", config)
        if isinstance(section, Mapping) and (
            "chain" in section or "strategy" in section
        ):
            base = router.config_for(CAPABILITY_TTS)
            merged = base.model_copy(
                update={
                    key: section[key]
                    for key in ("chain", "strategy", "weights", "pricing_overrides")
                    if key in section
                }
            )
            router = Router(
                router.providers,
                configs={CAPABILITY_TTS: merged},
                seed=router.seed,
                circuit_breakers=router.circuits,
                rate_limiters=router.limiters,
            )
    return SpeechService(router, **values)


tts_service_from_config = speech_service_from_config
"""Alias for :func:`speech_service_from_config`."""

TTSService = SpeechService
"""Alias for :class:`SpeechService`."""
