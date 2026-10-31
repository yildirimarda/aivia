"""Route video requests by frame and native-audio capabilities.

A provider must declare the optional inputs it can consume. A request with a
first frame, a last frame or native audio therefore has to be routed only to
providers that declare the matching capability; otherwise the option would be
silently dropped. This module answers that policy question before a call,
filters candidates and offers a guard that fails loudly instead of losing an
input.

It also removes audio a provider returned even though the request did not ask
for it. The stripped payload is inspected with ffprobe before it is returned,
so a failed or no-op ffmpeg invocation is never mistaken for a silent clip.
"""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aivia.core.config.media import (
    DEFAULT_MEDIA_FFPROBE_TIMEOUT_SECONDS as DEFAULT_FFPROBE_TIMEOUT_SECONDS,
)
from aivia.core.errors import InvalidRequestError, ValidationError
from aivia.core.media_probe import (
    FFprobeInfo,
    FFprobeRunner,
    MediaProbe,
    run_ffprobe,
    sniff_media,
)
from aivia.core.models.provider import ProviderMeta
from aivia.providers.base import Provider
from aivia.providers.protocols.video import (
    CAPABILITY_VIDEO_AUDIO,
    CAPABILITY_VIDEO_AUDIO_NATIVE,
    CAPABILITY_VIDEO_FIRST_FRAME,
    CAPABILITY_VIDEO_GEN,
    CAPABILITY_VIDEO_IMAGE_INPUT,
    CAPABILITY_VIDEO_LAST_FRAME,
    CAPABILITY_VIDEO_NATIVE_AUDIO,
    VideoRequest,
    VideoResult,
)

__all__ = [
    "CAPABILITY_VIDEO_AUDIO",
    "CAPABILITY_VIDEO_AUDIO_NATIVE",
    "CAPABILITY_VIDEO_FIRST_FRAME",
    "CAPABILITY_VIDEO_GEN",
    "CAPABILITY_VIDEO_IMAGE_INPUT",
    "CAPABILITY_VIDEO_LAST_FRAME",
    "CAPABILITY_VIDEO_NATIVE_AUDIO",
    "FFmpegRunner",
    "UnsupportedVideoFlagsError",
    "VideoCapabilityNotSupportedError",
    "VideoFlagNotSupportedError",
    "VideoFlags",
    "VideoFlagsNotSupportedError",
    "VideoInputNotSupportedError",
    "apply_audio_policy",
    "declared_video_flags",
    "enforce_audio_policy",
    "ensure_video_flags",
    "ensure_video_support",
    "filter_providers_for_flags",
    "filter_providers_for_frames",
    "filter_providers_for_video_capabilities",
    "filter_providers_for_video_flags",
    "filter_providers_for_video_input",
    "prepare_video_request",
    "remove_unwanted_audio",
    "required_video_capabilities",
    "should_strip_audio",
    "strip_audio",
    "strip_audio_file",
    "strip_audio_if_unwanted",
    "strip_unwanted_audio",
    "strip_video_audio",
    "supports_audio",
    "supports_first_frame",
    "supports_last_frame",
    "supports_native_audio",
    "video_flags",
    "wants_native_audio",
]

FFmpegRunner = Callable[[Sequence[str]], Any]
"""A callable that runs an ffmpeg argv and returns bytes or a completed process."""

_FIRST_FRAME_OPTIONS = frozenset(
    {"first_frame", "first_frame_image", "image_url", "image_input"}
)
_LAST_FRAME_OPTIONS = frozenset({"last_frame", "last_frame_image", "end_image"})
_NATIVE_AUDIO_OPTIONS = frozenset(
    {"generate_audio", "audio", "native_audio", "audio_prompt"}
)
_AUDIO_ALIASES = frozenset({"video_native_audio", "video_audio", "native_audio"})


def _metadata(
    source: Provider | ProviderMeta | Mapping[str, Any],
) -> ProviderMeta:
    if isinstance(source, ProviderMeta):
        return source
    if isinstance(source, Provider):
        return source.metadata
    if isinstance(source, Mapping):
        return ProviderMeta.model_validate(source)
    raise TypeError("source must be a Provider, ProviderMeta or mapping")


def _declares(
    metadata: ProviderMeta,
    capability: str,
    options: frozenset[str],
) -> bool:
    if metadata.supports(capability):
        return True
    if capability in _AUDIO_ALIASES and any(
        metadata.supports(alias) for alias in _AUDIO_ALIASES
    ):
        return True
    supported = metadata.supported_options.get(CAPABILITY_VIDEO_GEN, ())
    return any(option in supported for option in options)


def supports_first_frame(
    source: Provider | ProviderMeta | Mapping[str, Any],
) -> bool:
    """Whether *source* accepts a first frame image."""
    return _declares(
        _metadata(source), CAPABILITY_VIDEO_FIRST_FRAME, _FIRST_FRAME_OPTIONS
    )


def supports_last_frame(
    source: Provider | ProviderMeta | Mapping[str, Any],
) -> bool:
    """Whether *source* accepts a last frame image."""
    return _declares(
        _metadata(source), CAPABILITY_VIDEO_LAST_FRAME, _LAST_FRAME_OPTIONS
    )


def supports_native_audio(
    source: Provider | ProviderMeta | Mapping[str, Any],
) -> bool:
    """Whether *source* can generate a native soundtrack."""
    return _declares(
        _metadata(source), CAPABILITY_VIDEO_NATIVE_AUDIO, _NATIVE_AUDIO_OPTIONS
    )


supports_audio = supports_native_audio
"""Alias for :func:`supports_native_audio`."""


@dataclass(frozen=True, slots=True)
class VideoFlags:
    """The optional video inputs one provider declares it can consume."""

    first_frame: bool = False
    last_frame: bool = False
    native_audio: bool = False

    @property
    def audio(self) -> bool:
        """Alias for :attr:`native_audio`."""
        return self.native_audio

    @property
    def first(self) -> bool:
        """Alias for :attr:`first_frame`."""
        return self.first_frame

    @property
    def last(self) -> bool:
        """Alias for :attr:`last_frame`."""
        return self.last_frame

    @property
    def capabilities(self) -> tuple[str, ...]:
        """The capability names implied by these flags, in stable order."""
        names: list[str] = []
        if self.first_frame:
            names.append(CAPABILITY_VIDEO_FIRST_FRAME)
        if self.last_frame:
            names.append(CAPABILITY_VIDEO_LAST_FRAME)
        if self.native_audio:
            names.append(CAPABILITY_VIDEO_NATIVE_AUDIO)
        return tuple(names)


def declared_video_flags(
    source: Provider | ProviderMeta | Mapping[str, Any],
) -> VideoFlags:
    """Return the optional video inputs *source* declares."""
    return VideoFlags(
        first_frame=supports_first_frame(source),
        last_frame=supports_last_frame(source),
        native_audio=supports_native_audio(source),
    )


video_flags = declared_video_flags
"""Alias for :func:`declared_video_flags`."""


def wants_native_audio(request: VideoRequest) -> bool:
    """Whether *request* asks for a provider-generated soundtrack."""
    if not isinstance(request, VideoRequest):
        raise TypeError("request must be a VideoRequest")
    return request.generate_audio or request.audio_prompt is not None


def required_video_capabilities(request: VideoRequest) -> tuple[str, ...]:
    """Return the extra capabilities *request* needs from its provider.

    A plain text-to-video request needs nothing beyond ``video_gen``. A request
    that pins a frame requires the matching frame capability, and a request
    that asks for native audio requires the audio capability, so the router can
    skip providers that would silently ignore those inputs.
    """
    if not isinstance(request, VideoRequest):
        raise TypeError("request must be a VideoRequest")
    required: list[str] = []
    if request.first_frame is not None:
        required.append(CAPABILITY_VIDEO_FIRST_FRAME)
    if request.last_frame is not None:
        required.append(CAPABILITY_VIDEO_LAST_FRAME)
    if wants_native_audio(request):
        required.append(CAPABILITY_VIDEO_NATIVE_AUDIO)
    return tuple(required)


def _supports_capability(
    source: Provider | ProviderMeta | Mapping[str, Any],
    capability: str,
) -> bool:
    if capability == CAPABILITY_VIDEO_FIRST_FRAME:
        return supports_first_frame(source)
    if capability == CAPABILITY_VIDEO_LAST_FRAME:
        return supports_last_frame(source)
    if capability in _AUDIO_ALIASES:
        return supports_native_audio(source)
    return _metadata(source).supports(capability)


def filter_providers_for_video_flags(
    providers: Iterable[Provider],
    request: VideoRequest,
) -> tuple[Provider, ...]:
    """Return only the providers that may receive *request* with its flags."""
    if not isinstance(request, VideoRequest):
        raise TypeError("request must be a VideoRequest")
    candidates = tuple(providers)
    required = required_video_capabilities(request)
    if not required:
        return candidates
    return tuple(
        provider
        for provider in candidates
        if all(_supports_capability(provider, item) for item in required)
    )


filter_providers_for_flags = filter_providers_for_video_flags
"""Alias for :func:`filter_providers_for_video_flags`."""

filter_providers_for_video_capabilities = filter_providers_for_video_flags
"""Alias for :func:`filter_providers_for_video_flags`."""

filter_providers_for_frames = filter_providers_for_video_flags
"""Alias for :func:`filter_providers_for_video_flags`."""

filter_providers_for_video_input = filter_providers_for_video_flags
"""Alias for :func:`filter_providers_for_video_flags`."""


class VideoCapabilityNotSupportedError(InvalidRequestError):
    """A video request was sent to a provider missing a required capability."""

    code = "video_capability_not_supported"

    def __init__(
        self,
        provider: str,
        capability: str,
        *,
        field: str | None = None,
        required: Sequence[str] = (),
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.provider = provider
        self.capability = capability
        self.field = field
        merged: dict[str, Any] = {
            "provider": provider,
            "required_capability": capability,
            "required": list(required) or [capability],
        }
        if field is not None:
            merged["field"] = field
        if details:
            merged.update(details)
        super().__init__(
            f"provider {provider!r} does not declare {capability} support",
            details=merged,
        )


VideoFlagNotSupportedError = VideoCapabilityNotSupportedError
"""Alias for :class:`VideoCapabilityNotSupportedError`."""

VideoFlagsNotSupportedError = VideoCapabilityNotSupportedError
"""Alias for :class:`VideoCapabilityNotSupportedError`."""

VideoInputNotSupportedError = VideoCapabilityNotSupportedError
"""Alias for :class:`VideoCapabilityNotSupportedError`."""

UnsupportedVideoFlagsError = VideoCapabilityNotSupportedError
"""Alias for :class:`VideoCapabilityNotSupportedError`."""


def _missing_capability(
    source: Provider | ProviderMeta | Mapping[str, Any],
    request: VideoRequest,
) -> tuple[str, str | None] | None:
    if request.first_frame is not None and not supports_first_frame(source):
        return CAPABILITY_VIDEO_FIRST_FRAME, "first_frame"
    if request.last_frame is not None and not supports_last_frame(source):
        return CAPABILITY_VIDEO_LAST_FRAME, "last_frame"
    if wants_native_audio(request) and not supports_native_audio(source):
        return CAPABILITY_VIDEO_NATIVE_AUDIO, "generate_audio"
    return None


def ensure_video_support(
    source: Provider | ProviderMeta | Mapping[str, Any],
    request: VideoRequest,
) -> None:
    """Raise when *request* needs a capability *source* does not declare."""
    if not isinstance(request, VideoRequest):
        raise TypeError("request must be a VideoRequest")
    missing = _missing_capability(source, request)
    if missing is None:
        return
    capability, field = missing
    raise VideoCapabilityNotSupportedError(
        _metadata(source).name or "provider",
        capability,
        field=field,
        required=required_video_capabilities(request),
    )


ensure_video_flags = ensure_video_support
"""Alias for :func:`ensure_video_support`."""


def prepare_video_request(
    source: Provider | ProviderMeta | Mapping[str, Any],
    request: VideoRequest,
) -> VideoRequest:
    """Return *request* after proving *source* can receive all its flags."""
    ensure_video_support(source, request)
    return request


def _suffix_for(payload: bytes, path: Path | None) -> str:
    if path is not None and path.suffix:
        return path.suffix
    signature = sniff_media(payload)
    return f".{signature.extension}" if signature is not None else ".mp4"


def _read_source(
    source: bytes | bytearray | memoryview | str | Path,
) -> tuple[bytes, Path | None]:
    if isinstance(source, (bytes, bytearray, memoryview)):
        return bytes(source), None
    path = Path(source)
    if not path.is_file():
        raise ValidationError(
            f"video source does not exist: {path}",
            details={"path": str(path)},
        )
    return path.read_bytes(), path


def _ffmpeg_argv(executable: str, source: Path, target: Path) -> list[str]:
    return [
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-an",
        "-c",
        "copy",
        str(target),
    ]


def _run_ffmpeg(
    argv: Sequence[str],
    target: Path,
    runner: FFmpegRunner | None,
) -> bytes:
    if runner is None:
        try:
            completed = subprocess.run(
                list(argv),
                capture_output=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise ValidationError(
                f"ffmpeg is not available: {argv[0]}",
                details={"executable": argv[0]},
            ) from exc
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValidationError(
                "ffmpeg could not strip the audio",
                details={"executable": argv[0], "error": type(exc).__name__},
            ) from exc
        if completed.returncode != 0:
            raise ValidationError(
                "ffmpeg rejected the audio stripping command",
                details={
                    "returncode": completed.returncode,
                    "stderr": (completed.stderr or b"")[:500].decode("utf-8", "replace")
                    if isinstance(completed.stderr, bytes)
                    else str(completed.stderr or "")[:500],
                },
            )
    else:
        result = runner(argv)
        if isinstance(result, (bytes, bytearray, memoryview)):
            payload = bytes(result)
            target.write_bytes(payload)
            return payload
        if isinstance(result, subprocess.CompletedProcess):
            if result.returncode != 0:
                raise ValidationError(
                    "ffmpeg rejected the audio stripping command",
                    details={"returncode": result.returncode},
                )
            if not target.is_file():
                stdout = result.stdout
                if isinstance(stdout, bytes) and stdout:
                    target.write_bytes(stdout)
                elif isinstance(stdout, str) and stdout:
                    target.write_bytes(stdout.encode("utf-8"))
        elif isinstance(result, (str, Path)):
            candidate = Path(result)
            if candidate.is_file() and candidate != target:
                target.write_bytes(candidate.read_bytes())
    if not target.is_file():
        raise ValidationError(
            "ffmpeg did not produce a stripped video",
            details={"target": str(target)},
        )
    return target.read_bytes()


def _verify_no_audio(
    target: Path,
    *,
    ffprobe_executable: str,
    timeout_seconds: float,
    runner: FFprobeRunner | None,
) -> FFprobeInfo:
    info = run_ffprobe(
        target,
        executable=ffprobe_executable,
        timeout_seconds=timeout_seconds,
        runner=runner,
    )
    if info.audio_streams:
        raise ValidationError(
            "audio stream remains after stripping",
            details={
                "audio_streams": len(info.audio_streams),
                "path": str(target),
            },
        )
    return info


def strip_audio(
    source: bytes | bytearray | memoryview | str | Path,
    *,
    destination: str | Path | None = None,
    ffmpeg_executable: str = "ffmpeg",
    ffprobe_executable: str = "ffprobe",
    timeout_seconds: float = DEFAULT_FFPROBE_TIMEOUT_SECONDS,
    ffmpeg_runner: FFmpegRunner | None = None,
    ffprobe_runner: FFprobeRunner | None = None,
    verify: bool = True,
    **options: Any,
) -> bytes:
    """Return *source* without an audio stream, verified by ffprobe.

    *source* is either inline bytes or a readable file. The stripped payload is
    written to *destination* when one is given; the bytes are always returned.
    Both runners are injectable so the offline suite never needs the binaries.
    """
    ffmpeg_executable = options.pop("ffmpeg", ffmpeg_executable)
    ffprobe_executable = options.pop("ffprobe", ffprobe_executable)
    timeout_seconds = options.pop(
        "ffprobe_timeout_seconds",
        options.pop("timeout", timeout_seconds),
    )
    ffmpeg_runner = options.pop("runner", ffmpeg_runner)
    payload, source_path = _read_source(source)
    suffix = _suffix_for(payload, source_path)
    with tempfile.TemporaryDirectory(prefix="aivia-strip-audio-") as directory:
        workdir = Path(directory)
        source_file = source_path
        if source_file is None:
            source_file = workdir / f"source{suffix}"
            source_file.write_bytes(payload)
        requested = Path(destination) if destination is not None else None
        in_place = (
            source_file is not None
            and requested is not None
            and requested.resolve() == source_file.resolve()
        )
        target = (
            requested
            if requested is not None and not in_place
            else workdir / f"stripped{suffix}"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        stripped = _run_ffmpeg(
            _ffmpeg_argv(ffmpeg_executable, source_file, target),
            target,
            ffmpeg_runner,
        )
        if verify:
            _verify_no_audio(
                target,
                ffprobe_executable=ffprobe_executable,
                timeout_seconds=timeout_seconds,
                runner=ffprobe_runner,
            )
        if in_place and source_file is not None:
            source_file.write_bytes(stripped)
            return stripped
    return stripped


def strip_audio_file(
    source: str | Path,
    destination: str | Path | None = None,
    **values: Any,
) -> Path:
    """Strip audio from a file and return the written path.

    Without *destination* the source file is replaced in place, which keeps the
    common "make this clip silent" call short.
    """
    path = Path(source)
    target = Path(destination) if destination is not None else path
    strip_audio(path, destination=target, **values)
    return target


def _strip_result(
    result: VideoResult,
    request: VideoRequest | None,
    *,
    destination: str | Path | None = None,
    generate_audio: bool | None = None,
    **values: Any,
) -> VideoResult:
    if generate_audio is None and request is not None:
        generate_audio = wants_native_audio(request)
    if generate_audio:
        return result
    if result.has_audio is False and request is not None:
        return result
    if result.data is None:
        raise ValidationError(
            "cannot strip audio from a remote video; download it first",
            details={"url": result.url},
        )
    stripped = strip_audio(result.data, destination=destination, **values)
    return result.model_copy(update={"data": stripped, "has_audio": False})


def strip_unwanted_audio(
    source: VideoResult | bytes | bytearray | memoryview | str | Path,
    request: VideoRequest | bool | None = None,
    *,
    destination: str | Path | None = None,
    generate_audio: bool | None = None,
    probe: MediaProbe | FFprobeInfo | None = None,
    has_audio: bool | None = None,
    **values: Any,
) -> VideoResult | bytes | Path:
    """Remove audio that the request did not ask for.

    A :class:`VideoResult` is returned in kind, inline bytes stay bytes and a
    file path stays a path. *request* decides whether audio was wanted; passing
    *generate_audio* directly is accepted for callers that do not have a
    request. A result that already reports no audio is left untouched.
    """
    if isinstance(request, bool):
        generate_audio = request
        request = None
    if request is not None and not isinstance(request, VideoRequest):
        raise TypeError("request must be a VideoRequest or None")
    if isinstance(source, VideoResult):
        if has_audio is not None:
            source = source.model_copy(update={"has_audio": has_audio})
        return _strip_result(
            source,
            request,
            destination=destination,
            generate_audio=generate_audio,
            **values,
        )
    if generate_audio:
        if isinstance(source, (bytes, bytearray, memoryview)):
            return bytes(source)
        return Path(source)
    if isinstance(source, (bytes, bytearray, memoryview)):
        return strip_audio(source, destination=destination, **values)
    return strip_audio_file(source, destination, **values)


def should_strip_audio(
    result: VideoResult,
    request: VideoRequest | None = None,
    *,
    generate_audio: bool | None = None,
) -> bool:
    """Whether *result* carries audio that the request did not ask for."""
    if not isinstance(result, VideoResult):
        raise TypeError("result must be a VideoResult")
    if generate_audio is None and request is not None:
        generate_audio = wants_native_audio(request)
    return bool(result.has_audio) and not bool(generate_audio)


remove_unwanted_audio = strip_unwanted_audio
"""Alias for :func:`strip_unwanted_audio`."""

strip_video_audio = strip_unwanted_audio
"""Alias for :func:`strip_unwanted_audio`."""

strip_audio_if_unwanted = strip_unwanted_audio
"""Alias for :func:`strip_unwanted_audio`."""

apply_audio_policy = strip_unwanted_audio
"""Alias for :func:`strip_unwanted_audio`."""

enforce_audio_policy = strip_unwanted_audio
"""Alias for :func:`strip_unwanted_audio`."""
