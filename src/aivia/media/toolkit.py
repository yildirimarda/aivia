"""A single Python entry point for every paid media capability.

``MediaToolkit`` wraps the image, video, speech, transcription, music and image
edit services behind one small object. Each call runs the same screened, cached,
budgeted and cost tracked path as the CLI: the toolkit only chooses the service,
builds the typed request and returns a :class:`MediaResult` that carries the
outcome plus a uniform cost summary.

Callers may pass a typed request model, a prompt string or keyword options. A
prompt based call is screened automatically, while an explicit token can still
be supplied when the caller has already screened the text. Every method is
asynchronous; a ``*_sync`` alias is provided for scripts that are not already
inside an event loop.
"""

from __future__ import annotations

import asyncio
import mimetypes
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self, cast
from urllib.parse import unquote, urlsplit

from sqlalchemy import Engine

from aivia.core.config.app import RootConfig, parse_app_config
from aivia.core.config.budget import BudgetConfig
from aivia.core.config.database import DatabaseConfig
from aivia.core.config.resolve import resolve_config
from aivia.core.cost import CostTracker
from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.models.asset import Asset
from aivia.core.paths import resolve_config_file, resolve_paths
from aivia.core.router import Router
from aivia.media.image_edit_service import ImageEditOutcome, ImageEditService
from aivia.media.image_service import ImageOutcome, ImageService
from aivia.media.music_service import MusicOutcome, MusicService
from aivia.media.stt_service import STTOutcome, STTService
from aivia.media.tts_service import SpeechOutcome, SpeechService
from aivia.media.video_service import VideoOutcome, VideoService
from aivia.providers.protocols.image import ImageRequest
from aivia.providers.protocols.image_edit import (
    ImageEditOperation,
    ImageEditRequest,
)
from aivia.providers.protocols.media import MediaInput
from aivia.providers.protocols.music import MusicRequest
from aivia.providers.protocols.stt import STTRequest
from aivia.providers.protocols.tts import TTSRequest
from aivia.providers.protocols.video import VideoRequest
from aivia.providers.provider_factory import build_providers
from aivia.providers.storage_factory import build_storage_provider

__all__ = [
    "MediaResult",
    "MediaToolkit",
    "MediaToolkitResult",
    "ToolkitResult",
]

MediaOutcome = (
    ImageOutcome
    | VideoOutcome
    | SpeechOutcome
    | STTOutcome
    | MusicOutcome
    | ImageEditOutcome
)

_CALL_OPTIONS = frozenset(
    {
        "estimated",
        "exclude",
        "key",
        "key_prefix",
        "metadata",
        "no_cache",
        "operation",
        "pinned",
        "record_cost",
        "record_sidecar",
        "remaining_budget",
        "run_id",
        "screen_token",
        "screening_token",
        "settings",
        "storage",
        "token",
        "tracker",
    }
)

_EDIT_ALIASES = {
    "background-removal": ImageEditOperation.REMOVE_BACKGROUND,
    "background_removal": ImageEditOperation.REMOVE_BACKGROUND,
    "cutout": ImageEditOperation.REMOVE_BACKGROUND,
    "expand": ImageEditOperation.OUTPAINT,
    "instruction": ImageEditOperation.EDIT,
    "instruction-edit": ImageEditOperation.EDIT,
    "instruction_edit": ImageEditOperation.EDIT,
    "outpainting": ImageEditOperation.OUTPAINT,
    "remove-bg": ImageEditOperation.REMOVE_BACKGROUND,
    "remove-background": ImageEditOperation.REMOVE_BACKGROUND,
    "remove_bg": ImageEditOperation.REMOVE_BACKGROUND,
    "remove_background": ImageEditOperation.REMOVE_BACKGROUND,
    "replace-background": ImageEditOperation.REPLACE_BACKGROUND,
    "replace_background": ImageEditOperation.REPLACE_BACKGROUND,
    "upscaling": ImageEditOperation.UPSCALE,
}


def _local_path(value: str) -> Path | None:
    parsed = urlsplit(value)
    if parsed.scheme.lower() == "file":
        return Path(unquote(parsed.path))
    if not parsed.scheme:
        return Path(value)
    return None


def _media_input(value: Any) -> MediaInput:
    if isinstance(value, MediaInput):
        return value
    if isinstance(value, Mapping):
        return MediaInput.model_validate(value)
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser()
        media_type = mimetypes.guess_type(path.name)[0] or "image/png"
        return MediaInput(
            data=path.read_bytes(),
            media_type=media_type,
            filename=path.name,
        )
    if isinstance(value, (bytes, bytearray, memoryview)):
        return MediaInput(data=bytes(value), media_type="image/png")
    raise TypeError("image must be a path, bytes, MediaInput or mapping")


def _edit_operation(value: Any) -> ImageEditOperation:
    if isinstance(value, ImageEditOperation):
        return value
    text = str(value).strip().lower().replace(" ", "-")
    if text in _EDIT_ALIASES:
        return _EDIT_ALIASES[text]
    try:
        return ImageEditOperation(text)
    except ValueError:
        raise ValueError(f"unknown image edit operation: {value!r}") from None


def _split_options(options: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    request_values: dict[str, Any] = {}
    call_values: dict[str, Any] = {}
    for name, value in options.items():
        if name in _CALL_OPTIONS:
            call_values[name] = value
        else:
            request_values[name] = value
    return request_values, call_values


def _request_values(
    request: Any,
    values: Mapping[str, Any],
    *,
    prompt_field: str | None,
    prompt: Any = None,
) -> dict[str, Any]:
    if isinstance(request, Mapping):
        merged = dict(request)
    elif isinstance(request, str):
        if prompt_field is None:
            raise TypeError("pass a typed request, not a string")
        merged = {prompt_field: request}
    elif request is None:
        merged = {}
    elif hasattr(request, "model_dump"):
        merged = request.model_dump(mode="python")
    else:
        raise TypeError(f"request must be a typed model or mapping, got {request!r}")
    merged.update(values)
    if prompt is not None:
        if prompt_field is None:
            raise TypeError("this request does not take a prompt")
        merged[prompt_field] = prompt
    return merged


def _updated(model: Any, updates: Mapping[str, Any]) -> Any:
    if not updates:
        return model
    dumped = model.model_dump(mode="python")
    dumped.update(updates)
    return type(model).model_validate(dumped)


@dataclass(frozen=True, slots=True)
class MediaResult:
    """One media outcome plus a uniform provider, path and cost summary.

    Unknown attributes are delegated to :attr:`outcome`, so callers can still
    reach the rich service result (``response``, ``events``, ``tracks`` and so
    on) while every capability shares ``cost_usd``, ``paths`` and ``to_dict``.
    """

    kind: str
    outcome: MediaOutcome

    @property
    def result(self) -> MediaOutcome:
        """Alias for :attr:`outcome`."""
        return self.outcome

    @property
    def assets(self) -> tuple[Asset, ...]:
        """Every stored asset in response order."""
        return tuple(getattr(self.outcome, "assets", ()) or ())

    @property
    def outputs(self) -> tuple[Asset, ...]:
        """Alias for :attr:`assets`."""
        return self.assets

    @property
    def asset(self) -> Asset | None:
        """The first stored asset, or ``None``."""
        return self.assets[0] if self.assets else None

    @property
    def paths(self) -> tuple[str, ...]:
        """Every stored asset path in response order."""
        paths = getattr(self.outcome, "paths", None)
        if paths is not None:
            return tuple(str(path) for path in paths)
        return tuple(asset.path for asset in self.assets if asset.path)

    @property
    def urls(self) -> tuple[str, ...]:
        """Alias for :attr:`paths`."""
        urls = getattr(self.outcome, "urls", None)
        if urls is not None:
            return tuple(str(url) for url in urls)
        return self.paths

    @property
    def local_paths(self) -> tuple[Path, ...]:
        """Every stored path that resolves to a local file."""
        paths: list[Path] = []
        for value in self.paths:
            path = _local_path(value)
            if path is not None:
                paths.append(path)
        return tuple(paths)

    @property
    def files(self) -> tuple[Path, ...]:
        """Alias for :attr:`local_paths`."""
        return self.local_paths

    @property
    def sidecars(self) -> tuple[Path, ...]:
        """Every cost sidecar written next to the outputs."""
        return tuple(getattr(self.outcome, "sidecars", ()) or ())

    @property
    def provider(self) -> str:
        """The provider that served or cached the call."""
        return str(getattr(self.outcome, "provider", "") or "")

    @property
    def model(self) -> str:
        """The model that served or cached the call."""
        return str(getattr(self.outcome, "model", "") or "")

    @property
    def cached(self) -> bool:
        """Whether this result came from the asset cache."""
        return bool(getattr(self.outcome, "cached", False))

    @property
    def events(self) -> tuple[Any, ...]:
        """Every cost event recorded for this result."""
        return tuple(getattr(self.outcome, "events", ()) or ())

    @property
    def event(self) -> Any | None:
        """The first recorded cost event, or ``None``."""
        events = self.events
        return events[0] if events else None

    @property
    def amount_usd(self) -> float:
        """The recorded cost in US dollars."""
        return float(getattr(self.outcome, "amount_usd", 0.0) or 0.0)

    @property
    def cost_usd(self) -> float:
        """Alias for :attr:`amount_usd`."""
        return self.amount_usd

    @property
    def cost(self) -> float:
        """Alias for :attr:`amount_usd`."""
        return self.amount_usd

    @property
    def total_cost_usd(self) -> float:
        """Alias for :attr:`amount_usd`."""
        return self.amount_usd

    @property
    def text(self) -> str | None:
        """The transcript text for a transcription result, else ``None``."""
        value = getattr(self.outcome, "text", None)
        return value if isinstance(value, str) else None

    @property
    def audio(self) -> bytes | None:
        """The generated audio payload, when the outcome carries one."""
        value = getattr(self.outcome, "audio", None)
        return value if isinstance(value, bytes) else None

    @property
    def data(self) -> bytes | None:
        """Alias for :attr:`audio`."""
        return self.audio

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary for logs and manifests."""
        summary: dict[str, Any] = {
            "kind": self.kind,
            "provider": self.provider,
            "model": self.model,
            "cost_usd": self.cost_usd,
            "cached": self.cached,
            "paths": list(self.paths),
            "urls": list(self.urls),
            "sidecars": [str(path) for path in self.sidecars],
        }
        text = self.text
        if text is not None:
            summary["text"] = text
        return summary

    def __getattr__(self, name: str) -> Any:
        return getattr(self.outcome, name)


MediaToolkitResult = MediaResult
"""Alias for :class:`MediaResult`."""

ToolkitResult = MediaResult
"""Alias for :class:`MediaResult`."""


class MediaToolkit:
    """Generate and edit media through one configured router and ledger.

    ``router`` chooses the provider; ``tracker``, ``engine``, ``storage`` and
    ``budget`` are the same collaborators the individual services accept. The
    first call of each kind builds and remembers that service, so repeated
    calls share the cache and the router's circuit breakers.

    Use :meth:`from_config` to build the router, database, tracker and storage
    from a resolved application config, or pass an explicit ``router`` for a
    test or an embedding application.
    """

    def __init__(
        self,
        router: Router,
        *,
        tracker: CostTracker | None = None,
        engine: Engine | None = None,
        storage: Any = None,
        budget: Any = None,
        safety: Any = None,
        cache: Any = None,
        limits: Any = None,
        downloader: Any = None,
        http: Any = None,
        transport: Any = None,
        pricing_overrides: Mapping[str, float] | None = None,
        pricing_version: str = "",
        toolchain: Any = None,
        key_prefix: str | None = None,
        no_cache: bool = False,
        record_cost: bool = True,
        record_sidecar: bool = True,
        ffmpeg_executable: str = "ffmpeg",
        ffmpeg_runner: Any = None,
        ffmpeg_timeout_seconds: float | None = None,
        ffprobe_executable: str = "ffprobe",
        ffprobe_runner: Any = None,
        ffprobe_timeout_seconds: float | None = None,
    ) -> None:
        if not isinstance(router, Router):
            raise TypeError("router must be a Router")
        if tracker is not None and not isinstance(tracker, CostTracker):
            raise TypeError("tracker must be a CostTracker")
        if engine is not None and not isinstance(engine, Engine):
            raise TypeError("engine must be a SQLAlchemy Engine")
        if key_prefix is not None and (
            not isinstance(key_prefix, str) or not key_prefix.strip()
        ):
            raise ValueError("key_prefix must be a non-empty string")

        self._router = router
        self._tracker = tracker
        self._engine = engine
        self._storage = storage
        self._budget = budget
        self._safety = safety
        self._cache = cache
        self._limits = limits
        self._downloader = downloader
        self._http = http
        self._transport = transport
        self._pricing_overrides = pricing_overrides
        self._pricing_version = pricing_version
        self._toolchain = toolchain
        self._key_prefix = key_prefix.strip().strip("/") if key_prefix else None
        self._no_cache = bool(no_cache)
        self._record_cost = bool(record_cost)
        self._record_sidecar = bool(record_sidecar)
        self._ffmpeg_executable = ffmpeg_executable
        self._ffmpeg_runner = ffmpeg_runner
        self._ffmpeg_timeout = ffmpeg_timeout_seconds
        self._ffprobe_executable = ffprobe_executable
        self._ffprobe_runner = ffprobe_runner
        self._ffprobe_timeout = ffprobe_timeout_seconds
        self._owns_engine = False
        self._image_service: ImageService | None = None
        self._video_service: VideoService | None = None
        self._speech_service: SpeechService | None = None
        self._stt_service: STTService | None = None
        self._music_service: MusicService | None = None
        self._edit_service: ImageEditService | None = None

    @classmethod
    def from_config(
        cls,
        config: Any = None,
        *,
        config_file: str | Path | None = None,
        environment: str | None = None,
        profile: str | None = None,
        preset: str | None = None,
        set_values: Any = None,
        data_dir: str | Path | None = None,
        router: Router | None = None,
        engine: Engine | None = None,
        tracker: CostTracker | None = None,
        storage: Any = None,
        budget: Any = None,
        safety: Any = None,
        **values: Any,
    ) -> Self:
        """Build a toolkit from an application config, file, mapping or defaults.

        A missing *engine* is opened and migrated under the resolved data
        directory; :meth:`close` disposes it. Supplying an engine leaves its
        lifecycle to the caller.
        """
        selected = config
        resolved_config_file: Path | None
        if selected is None or isinstance(selected, (str, Path)):
            resolved_config_file = resolve_config_file(
                selected if selected is not None else config_file
            )
            resolved = resolve_config(
                config_file=resolved_config_file,
                environment=environment,
                profile=profile,
                preset=preset,
                set_values=set_values,
            )
            selected = resolved.to_app_config()
        elif isinstance(selected, Mapping):
            selected = parse_app_config(selected)
            resolved_config_file = Path(config_file) if config_file else None
        elif isinstance(selected, RootConfig):
            resolved_config_file = Path(config_file) if config_file else None
        else:
            resolved_config_file = Path(config_file) if config_file else None

        root = selected
        active_data_dir = (
            Path(data_dir).expanduser()
            if data_dir is not None
            else resolve_paths(resolved_config_file).data_dir
        )
        owns_engine = False
        if router is None:
            router = Router(build_providers(getattr(root, "providers", None)))
        if engine is None:
            database = getattr(root, "database", None)
            filename = (
                database.filename
                if isinstance(database, DatabaseConfig)
                else DatabaseConfig().filename
            )
            database_path = active_data_dir / filename
            database_path.parent.mkdir(parents=True, exist_ok=True)
            engine = create_database_engine(path=database_path)
            upgrade_database(engine=engine)
            owns_engine = True
        if tracker is None:
            tracker = CostTracker(engine)
        if storage is None:
            storage = build_storage_provider(
                getattr(root, "storage", None),
                data_dir=active_data_dir,
            )
        if budget is None:
            budget_config = getattr(root, "budget", None)
            if isinstance(budget_config, BudgetConfig):
                budget = budget_config
        if safety is None:
            safety = getattr(root, "safety", None)

        toolkit = cls(
            router,
            tracker=tracker,
            engine=engine,
            storage=storage,
            budget=budget,
            safety=safety,
            **values,
        )
        toolkit._owns_engine = owns_engine
        return toolkit

    @property
    def router(self) -> Router:
        """The router every call goes through."""
        return self._router

    @property
    def tracker(self) -> CostTracker | None:
        """The cost ledger shared by every service."""
        return self._tracker

    @property
    def image_service(self) -> ImageService:
        """The lazily built image generation service."""
        if self._image_service is None:
            self._image_service = ImageService(
                self._router, **self._image_values("image")
            )
        return self._image_service

    @property
    def video_service(self) -> VideoService:
        """The lazily built video generation service."""
        if self._video_service is None:
            self._video_service = VideoService(self._router, **self._video_values())
        return self._video_service

    @property
    def speech_service(self) -> SpeechService:
        """The lazily built text to speech service."""
        if self._speech_service is None:
            self._speech_service = SpeechService(self._router, **self._speech_values())
        return self._speech_service

    @property
    def stt_service(self) -> STTService:
        """The lazily built speech to text service."""
        if self._stt_service is None:
            self._stt_service = STTService(
                self._router, **self._base_values("transcription")
            )
        return self._stt_service

    @property
    def music_service(self) -> MusicService:
        """The lazily built music generation service."""
        if self._music_service is None:
            self._music_service = MusicService(self._router, **self._music_values())
        return self._music_service

    @property
    def edit_service(self) -> ImageEditService:
        """The lazily built image edit service."""
        if self._edit_service is None:
            self._edit_service = ImageEditService(
                self._router, **self._image_values("edit")
            )
        return self._edit_service

    def _base_values(self, kind: str) -> dict[str, Any]:
        prefix = kind if self._key_prefix is None else f"{self._key_prefix}/{kind}"
        return {
            "tracker": self._tracker,
            "engine": self._engine,
            "storage": self._storage,
            "budget": self._budget,
            "safety": self._safety,
            "pricing_overrides": self._pricing_overrides,
            "pricing_version": self._pricing_version,
            "toolchain": self._toolchain,
            "key_prefix": prefix,
            "record_cost": self._record_cost,
            "record_sidecar": self._record_sidecar,
        }

    def _image_values(self, kind: str) -> dict[str, Any]:
        values = self._base_values(kind)
        values.update(
            {
                "cache": self._cache,
                "limits": self._limits,
                "downloader": self._downloader,
                "http": self._http,
                "transport": self._transport,
                "no_cache": self._no_cache,
            }
        )
        return values

    def _video_values(self) -> dict[str, Any]:
        values = self._base_values("video")
        values.update(
            {
                "cache": self._cache,
                "limits": self._limits,
                "no_cache": self._no_cache,
                "ffprobe_executable": self._ffprobe_executable,
                "ffprobe_runner": self._ffprobe_runner,
                "ffprobe_timeout_seconds": self._ffprobe_timeout,
            }
        )
        return values

    def _speech_values(self) -> dict[str, Any]:
        values = self._base_values("speech")
        values.update(
            {
                "cache": self._cache,
                "limits": self._limits,
                "no_cache": self._no_cache,
                "ffmpeg_executable": self._ffmpeg_executable,
                "ffmpeg_runner": self._ffmpeg_runner,
                "ffmpeg_timeout_seconds": self._ffmpeg_timeout,
                "ffprobe_executable": self._ffprobe_executable,
                "ffprobe_runner": self._ffprobe_runner,
                "ffprobe_timeout_seconds": self._ffprobe_timeout,
            }
        )
        return values

    def _music_values(self) -> dict[str, Any]:
        values = self._base_values("music")
        values.update(
            {
                "cache": self._cache,
                "limits": self._limits,
                "no_cache": self._no_cache,
                "ffprobe_executable": self._ffprobe_executable,
                "ffprobe_runner": self._ffprobe_runner,
                "ffprobe_timeout_seconds": self._ffprobe_timeout,
            }
        )
        return values

    async def image(
        self,
        request: ImageRequest | str | Mapping[str, Any] | None = None,
        *,
        prompt: str | None = None,
        token: Any = None,
        **options: Any,
    ) -> MediaResult:
        """Generate one or more images and return them with their cost."""
        request_values, call_values = _split_options(options)
        values = _request_values(
            request, request_values, prompt_field="prompt", prompt=prompt
        )
        image_request = ImageRequest(**values)
        service = self.image_service
        active_token = (
            token if token is not None else service.screen(image_request.prompt)
        )
        outcome = await service.generate(
            image_request, token=active_token, **call_values
        )
        return MediaResult("image", outcome)

    async def video(
        self,
        request: VideoRequest | str | Mapping[str, Any] | None = None,
        *,
        prompt: str | None = None,
        token: Any = None,
        **options: Any,
    ) -> MediaResult:
        """Generate one or more video clips and return them with their cost."""
        request_values, call_values = _split_options(options)
        values = _request_values(
            request, request_values, prompt_field="prompt", prompt=prompt
        )
        video_request = VideoRequest(**values)
        service = self.video_service
        active_token = (
            token if token is not None else service.screen(video_request.prompt)
        )
        outcome = await service.generate(
            video_request, token=active_token, **call_values
        )
        return MediaResult("video", outcome)

    async def speech(
        self,
        request: TTSRequest | str | Mapping[str, Any] | None = None,
        *,
        prompt: str | None = None,
        text: str | None = None,
        token: Any = None,
        **options: Any,
    ) -> MediaResult:
        """Synthesize speech from text and return the audio with its cost."""
        request_values, call_values = _split_options(options)
        values = _request_values(
            request,
            request_values,
            prompt_field="text",
            prompt=prompt if prompt is not None else text,
        )
        values.setdefault("voice", "default")
        speech_request = TTSRequest(**values)
        service = self.speech_service
        active_token = (
            token if token is not None else service.screen(speech_request.text)
        )
        outcome = await service.synthesize(
            speech_request, token=active_token, **call_values
        )
        return MediaResult("speech", outcome)

    async def transcribe(
        self,
        audio: STTRequest | str | Path | bytes | Mapping[str, Any] | None = None,
        *,
        request: STTRequest | Mapping[str, Any] | None = None,
        path: str | Path | None = None,
        file: str | Path | None = None,
        prompt: str | None = None,
        token: Any = None,
        **options: Any,
    ) -> MediaResult:
        """Transcribe an audio source and return the text with its cost."""
        request_values, call_values = _split_options(options)
        selected = request if request is not None else audio
        if isinstance(selected, STTRequest):
            updates = dict(request_values)
            if prompt is not None:
                updates["prompt"] = prompt
            stt_request = _updated(selected, updates)
        else:
            if isinstance(selected, Mapping):
                values = _request_values(selected, request_values, prompt_field=None)
            else:
                values = dict(request_values)
            source: str | Path | bytes | None = path if path is not None else file
            if source is None and not isinstance(selected, Mapping):
                source = selected
            if source is not None:
                if isinstance(source, (str, Path)):
                    audio_path = Path(source).expanduser()
                    values["audio"] = audio_path.read_bytes()
                    values["filename"] = audio_path.name
                    values["media_type"] = (
                        mimetypes.guess_type(audio_path.name)[0] or "audio/mpeg"
                    )
                elif isinstance(source, (bytes, bytearray, memoryview)):
                    values["audio"] = bytes(source)
                elif not isinstance(source, Mapping):
                    raise TypeError("audio must be a path, bytes or STTRequest")
            if prompt is not None:
                values["prompt"] = prompt
            stt_request = STTRequest(**values)
        service = self.stt_service
        active_token = token
        if active_token is None and stt_request.prompt:
            active_token = service.screen(stt_request.prompt)
        outcome = await service.generate(stt_request, token=active_token, **call_values)
        return MediaResult("transcription", outcome)

    async def music(
        self,
        request: MusicRequest | str | Mapping[str, Any] | None = None,
        *,
        prompt: str | None = None,
        token: Any = None,
        **options: Any,
    ) -> MediaResult:
        """Generate music and return the tracks with their cost and license."""
        request_values, call_values = _split_options(options)
        values = _request_values(
            request, request_values, prompt_field="prompt", prompt=prompt
        )
        music_request = MusicRequest(**values)
        service = self.music_service
        active_token = (
            token if token is not None else service.screen(music_request.prompt)
        )
        outcome = await service.generate(
            music_request, token=active_token, **call_values
        )
        return MediaResult("music", outcome)

    async def edit(
        self,
        request: ImageEditRequest | str | Path | bytes | MediaInput | None = None,
        *,
        image: str | Path | bytes | MediaInput | Mapping[str, Any] | None = None,
        prompt: str | None = None,
        operation: ImageEditOperation | str | None = None,
        mask: str | Path | bytes | MediaInput | Mapping[str, Any] | None = None,
        token: Any = None,
        **options: Any,
    ) -> MediaResult:
        """Edit an image and return the results with their cost."""
        request_values, call_values = _split_options(options)
        updates = dict(request_values)
        if image is not None:
            updates["image"] = _media_input(image)
        elif isinstance(request, (str, Path, bytes, bytearray, memoryview, MediaInput)):
            updates["image"] = _media_input(request)
        if prompt is not None:
            updates["prompt"] = prompt
        if operation is not None:
            updates["operation"] = _edit_operation(operation)
        if mask is not None:
            updates["mask"] = _media_input(mask)
        if isinstance(request, ImageEditRequest):
            edit_request = _updated(request, updates)
        elif isinstance(request, Mapping):
            merged = dict(request)
            merged.update(updates)
            edit_request = ImageEditRequest.model_validate(merged)
        elif request is None or "image" in updates:
            edit_request = ImageEditRequest.model_validate(updates)
        else:
            raise TypeError("edit requires an image path, bytes or request")
        service = self.edit_service
        active_token = token
        if active_token is None and edit_request.prompt:
            active_token = service.screen(edit_request.prompt)
        outcome = await service.edit(edit_request, token=active_token, **call_values)
        return MediaResult("edit", outcome)

    def image_sync(
        self,
        request: ImageRequest | str | Mapping[str, Any] | None = None,
        **options: Any,
    ) -> MediaResult:
        """Blocking alias for :meth:`image`."""
        return asyncio.run(self.image(request, **options))

    def video_sync(
        self,
        request: VideoRequest | str | Mapping[str, Any] | None = None,
        **options: Any,
    ) -> MediaResult:
        """Blocking alias for :meth:`video`."""
        return asyncio.run(self.video(request, **options))

    def speech_sync(
        self,
        request: TTSRequest | str | Mapping[str, Any] | None = None,
        **options: Any,
    ) -> MediaResult:
        """Blocking alias for :meth:`speech`."""
        return asyncio.run(self.speech(request, **options))

    def transcribe_sync(
        self,
        audio: STTRequest | str | Path | bytes | Mapping[str, Any] | None = None,
        **options: Any,
    ) -> MediaResult:
        """Blocking alias for :meth:`transcribe`."""
        return asyncio.run(self.transcribe(audio, **options))

    def music_sync(
        self,
        request: MusicRequest | str | Mapping[str, Any] | None = None,
        **options: Any,
    ) -> MediaResult:
        """Blocking alias for :meth:`music`."""
        return asyncio.run(self.music(request, **options))

    def edit_sync(
        self,
        request: ImageEditRequest | str | Path | bytes | MediaInput | None = None,
        **options: Any,
    ) -> MediaResult:
        """Blocking alias for :meth:`edit`."""
        return asyncio.run(self.edit(request, **options))

    generate_image = image
    generate_video = video
    generate_speech = speech
    synthesize_speech = speech
    tts = speech
    generate_transcription = transcribe
    transcription = transcribe
    stt = transcribe
    generate_music = music
    edit_image = edit
    image_edit = edit

    async def generate(
        self,
        kind: str,
        request: Any = None,
        **options: Any,
    ) -> MediaResult:
        """Dispatch one call by image, video, speech, transcription, music or edit."""
        key = kind.strip().lower()
        methods: dict[str, Callable[..., Any]] = {
            "image": self.image,
            "video": self.video,
            "speech": self.speech,
            "tts": self.speech,
            "transcription": self.transcribe,
            "transcribe": self.transcribe,
            "stt": self.transcribe,
            "music": self.music,
            "edit": self.edit,
        }
        method = methods.get(key)
        if method is None:
            supported = ", ".join(sorted(methods))
            raise ValueError(f"unknown media kind {kind!r} (supported: {supported})")
        return cast(MediaResult, await method(request, **options))

    def generate_sync(
        self,
        kind: str,
        request: Any = None,
        **options: Any,
    ) -> MediaResult:
        """Blocking alias for :meth:`generate`."""
        return asyncio.run(self.generate(kind, request, **options))

    def close(self) -> None:
        """Dispose the database engine when this toolkit opened it."""
        if self._owns_engine and self._engine is not None:
            self._engine.dispose()
            self._owns_engine = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
