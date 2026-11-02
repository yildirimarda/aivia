"""Validate a rendered output against the platform publishing limits.

Rendering and publishing have different jobs. The renderer makes a file with
the requested duration, canvas, audio and loudness; this module checks what was
actually produced against the limits in ``platforms.yaml``. A file can be
perfectly valid media and still be too long for X, the wrong shape for a
YouTube Short, too large for Instagram or too quiet for TikTok.

The checks are best effort: a value that cannot be measured is not guessed at.
Images, audio, video and web cards each have their own applicable checks, and
every issue names the media type so one report can cover several outputs.
"""

from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aivia.core.config.media import (
    DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS,
    DEFAULT_MEDIA_FFPROBE_TIMEOUT_SECONDS,
)
from aivia.core.errors import ValidationError
from aivia.core.platforms import (
    MediaLimits,
    Platform,
    PlatformCatalog,
    load_platforms,
)

from .errors import RenderError

if TYPE_CHECKING:
    from aivia.core.media_probe import FFprobeInfo, FFprobeRunner

__all__ = [
    "OutputIssue",
    "OutputIssues",
    "OutputValidationError",
    "OutputValidationIssue",
    "OutputValidationIssues",
    "measure_loudness",
    "validate_output",
    "validate_output_or_raise",
    "validate_platform_output",
]

OutputLoudnessRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]
"""A callable that runs an ffmpeg argv and returns the completed process."""

_EPSILON = 1e-6

_MEDIA_TYPE_ALIASES: dict[str, str] = {
    "animated": "image",
    "animation": "image",
    "audio_only": "audio",
    "card": "web_card",
    "carousel": "carousel",
    "gif": "image",
    "graphic": "image",
    "html": "web_card",
    "movie": "video",
    "photo": "image",
    "picture": "image",
    "reel": "video",
    "reels": "video",
    "short": "video",
    "shorts": "video",
    "web-card": "web_card",
    "webcard": "web_card",
}

_SUFFIX_MEDIA_TYPES: dict[str, str] = {
    "aac": "audio",
    "avif": "image",
    "avi": "video",
    "bmp": "image",
    "flac": "audio",
    "gif": "image",
    "heic": "image",
    "htm": "web_card",
    "html": "web_card",
    "jpeg": "image",
    "jpg": "image",
    "m4a": "audio",
    "m4v": "video",
    "mkv": "video",
    "mov": "video",
    "mp3": "audio",
    "mp4": "video",
    "mpeg": "video",
    "mpg": "video",
    "oga": "audio",
    "ogg": "audio",
    "opus": "audio",
    "png": "image",
    "tif": "image",
    "tiff": "image",
    "wav": "audio",
    "webm": "video",
    "webp": "image",
    "wma": "audio",
}

_AUDIO_TYPES = frozenset({"audio"})
_VISUAL_TYPES = frozenset({"video", "image", "carousel"})


@dataclass(frozen=True, slots=True)
class OutputIssue:
    """One platform limit an output fails."""

    path: str
    message: str
    code: str = "invalid"
    media_type: str = ""

    @property
    def field(self) -> str:
        """Alias for :attr:`path`."""
        return self.path

    def __str__(self) -> str:
        return f"{self.path}: {self.message}" if self.path else self.message

    def to_dict(self) -> dict[str, str]:
        """Return the issue as a JSON-compatible mapping."""
        return {
            "path": self.path,
            "message": self.message,
            "code": self.code,
            "media_type": self.media_type,
        }


OutputValidationIssue = OutputIssue


class OutputIssues(list[OutputIssue]):
    """Every platform limit an output fails in one pass.

    The object behaves like a plain list and also carries the context needed to
    explain the result: the output path, platform and media type that were
    checked.
    """

    def __init__(
        self,
        iterable: Sequence[OutputIssue] = (),
        *,
        output: str | Path | None = None,
        platform: Platform | None = None,
        media_type: str = "",
        probe: FFprobeInfo | None = None,
    ) -> None:
        super().__init__(iterable)
        self.output = Path(output) if output is not None else None
        self.platform = platform
        self.media_type = media_type
        self.probe = probe

    @property
    def ok(self) -> bool:
        """Whether the output satisfies every applicable platform limit."""
        return not self

    @property
    def errors(self) -> list[OutputIssue]:
        """The issues as a plain list."""
        return list(self)

    @property
    def issues(self) -> OutputIssues:
        """Alias for the collection itself."""
        return self

    @property
    def codes(self) -> list[str]:
        """The issue codes in order."""
        return [issue.code for issue in self]

    def raise_for_errors(self) -> None:
        """Raise :class:`OutputValidationError` when any issue exists."""
        if self:
            raise OutputValidationError(self)


OutputValidationIssues = OutputIssues


class OutputValidationError(RenderError):
    """A rendered output does not satisfy the selected platform's limits."""

    code = "output_validation"

    def __init__(self, issues: Sequence[OutputIssue]) -> None:
        self.issues = OutputIssues(issues)
        message = "; ".join(str(issue) for issue in self.issues)
        super().__init__(
            message or "output failed platform validation",
            details={"issues": [issue.to_dict() for issue in self.issues]},
        )


@dataclass(slots=True)
class _Facts:
    path: Path | None = None
    probe: Any | None = None
    size_bytes: int | None = None
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    has_audio: bool | None = None
    loudness_lufs: float | None = None
    platform: str | None = None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _path(value: Any) -> Path | None:
    if isinstance(value, (str, Path)):
        text = str(value).strip()
        return Path(text) if text else None
    return None


def _mapping_facts(values: Mapping[str, Any]) -> _Facts:
    facts = _Facts()
    facts.path = _path(values.get("output") or values.get("path"))
    facts.probe = values.get("probe") or values.get("ffprobe")
    facts.size_bytes = _integer(
        values.get("size_bytes", values.get("size", values.get("bytes")))
    )
    facts.duration_seconds = _number(
        values.get("duration_seconds", values.get("duration"))
    )
    facts.width = _integer(values.get("width"))
    facts.height = _integer(values.get("height"))
    if values.get("has_audio") is not None:
        facts.has_audio = bool(values.get("has_audio"))
    elif values.get("audio_present") is not None:
        facts.has_audio = bool(values.get("audio_present"))
    facts.loudness_lufs = _number(values.get("loudness_lufs", values.get("loudness")))
    platform = values.get("platform")
    facts.platform = platform if isinstance(platform, str) else None
    return facts


def _object_facts(value: Any) -> _Facts:
    facts = _Facts()
    facts.path = _path(getattr(value, "output", None) or getattr(value, "path", None))
    facts.probe = getattr(value, "probe", None)
    facts.size_bytes = _integer(
        getattr(value, "size_bytes", None) or getattr(value, "size", None)
    )
    facts.duration_seconds = _number(getattr(value, "duration_seconds", None))
    facts.width = _integer(getattr(value, "width", None))
    facts.height = _integer(getattr(value, "height", None))
    if getattr(value, "has_audio", None) is not None:
        facts.has_audio = bool(value.has_audio)
    elif getattr(value, "audio_present", None) is not None:
        facts.has_audio = bool(value.audio_present)
    facts.loudness_lufs = _number(getattr(value, "loudness_lufs", None))
    platform = getattr(value, "platform", None)
    facts.platform = platform if isinstance(platform, str) else None
    if facts.size_bytes is None:
        html = getattr(value, "html", None)
        if isinstance(html, str):
            facts.size_bytes = len(html.encode("utf-8"))
    return facts


def _facts(output: Any) -> _Facts:
    if isinstance(output, Mapping):
        return _mapping_facts(output)
    if isinstance(output, (str, Path)):
        return _Facts(path=_path(output))
    return _object_facts(output)


def _normalise_media_type(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return _MEDIA_TYPE_ALIASES.get(text, text)


def _suffix_media_type(path: Path | None) -> str:
    if path is None or not path.suffix:
        return ""
    return _SUFFIX_MEDIA_TYPES.get(path.suffix.lower().lstrip("."), "")


def _resolve_media_type(
    platform: Platform,
    requested: Any,
    path: Path | None,
) -> str:
    explicit = _normalise_media_type(requested)
    if explicit:
        return explicit
    from_suffix = _suffix_media_type(path)
    if from_suffix:
        return from_suffix
    if len(platform.media_types) == 1:
        return platform.media_types[0]
    return platform.media_types[0] if platform.media_types else ""


def _resolve_platform(
    platform: Any,
    catalog: PlatformCatalog | None,
    facts: _Facts,
) -> Platform:
    if isinstance(platform, Platform):
        return platform
    if isinstance(platform, PlatformCatalog):
        if len(platform) == 1:
            return next(iter(platform.entries().values()))
        raise OutputValidationError(
            [
                OutputIssue(
                    path="platform",
                    message=(
                        "a platform id is required when the catalog contains "
                        "more than one platform"
                    ),
                    code="platform",
                )
            ]
        )
    name = platform if isinstance(platform, str) else facts.platform
    if not name:
        raise OutputValidationError(
            [
                OutputIssue(
                    path="platform",
                    message="a platform id is required",
                    code="platform",
                )
            ]
        )
    active = catalog if catalog is not None else load_platforms()
    return active.require(name)


def _probe_image_dimensions(path: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.width, image.height
    except (OSError, ValueError):
        return None


def _probe_facts(
    path: Path | None,
    *,
    media_type: str,
    probe: Any | None,
    ffprobe_executable: str,
    ffprobe_timeout_seconds: float,
    ffprobe_runner: FFprobeRunner | None,
    issues: OutputIssues,
) -> Any | None:
    if probe is not None:
        return probe
    if path is None or not path.is_file():
        return None
    if media_type in {"web_card"}:
        return None
    if media_type in {"image", "carousel"}:
        return None
    from aivia.core.media_probe import run_ffprobe

    try:
        return run_ffprobe(
            path,
            executable=ffprobe_executable,
            timeout_seconds=ffprobe_timeout_seconds,
            runner=ffprobe_runner,
        )
    except ValidationError as error:
        issues.append(
            OutputIssue(
                path="probe",
                message=f"output could not be inspected: {error}",
                code="probe",
                media_type=media_type,
            )
        )
        return None


def _probe_duration(probe: Any | None) -> float | None:
    return _number(getattr(probe, "duration_seconds", None))


def _probe_size(probe: Any | None) -> int | None:
    return _integer(getattr(probe, "size_bytes", None))


def _probe_dimensions(probe: Any | None) -> tuple[int | None, int | None]:
    stream = getattr(probe, "primary_video", None)
    if stream is None:
        streams = getattr(probe, "video_streams", None)
        if streams:
            stream = streams[0]
    if stream is None:
        return None, None
    return _integer(getattr(stream, "width", None)), _integer(
        getattr(stream, "height", None)
    )


def _probe_has_audio(probe: Any | None) -> bool | None:
    if probe is None:
        return None
    streams = getattr(probe, "audio_streams", None)
    if streams is None:
        return None
    return bool(streams)


def _media_label(platform: Platform, media_type: str) -> str:
    name = platform.display_name or platform.id
    return f"{name} {media_type}"


def _check_duration(
    limits: MediaLimits,
    duration: float | None,
    *,
    platform: Platform,
    media_type: str,
    issues: OutputIssues,
) -> None:
    if duration is None:
        return
    minimum = limits.min_duration_seconds
    maximum = limits.max_duration_seconds
    label = _media_label(platform, media_type)
    if minimum is not None and duration < minimum - _EPSILON:
        issues.append(
            OutputIssue(
                path="duration",
                message=(
                    f"{label} duration {duration:g}s is below the minimum "
                    f"of {minimum:g}s"
                ),
                code="duration",
                media_type=media_type,
            )
        )
    if maximum is not None and duration > maximum + _EPSILON:
        issues.append(
            OutputIssue(
                path="duration",
                message=(
                    f"{label} duration {duration:g}s is above the maximum "
                    f"of {maximum:g}s"
                ),
                code="duration",
                media_type=media_type,
            )
        )


def _check_resolution(
    limits: MediaLimits,
    width: int | None,
    height: int | None,
    *,
    platform: Platform,
    media_type: str,
    issues: OutputIssues,
) -> None:
    if not limits.resolutions or width is None or height is None:
        return
    resolution = f"{width}x{height}"
    if resolution in limits.resolutions:
        return
    allowed = ", ".join(limits.resolutions)
    issues.append(
        OutputIssue(
            path="resolution",
            message=(
                f"{_media_label(platform, media_type)} resolution "
                f"{resolution} is not allowed (allowed: {allowed})"
            ),
            code="resolution",
            media_type=media_type,
        )
    )


def _check_size(
    limits: MediaLimits,
    size_bytes: int | None,
    *,
    platform: Platform,
    media_type: str,
    issues: OutputIssues,
) -> None:
    maximum = limits.max_file_size_bytes
    if maximum is None or size_bytes is None:
        return
    if size_bytes <= maximum:
        return
    issues.append(
        OutputIssue(
            path="size",
            message=(
                f"{_media_label(platform, media_type)} file size "
                f"{size_bytes} bytes is above the maximum of {maximum} bytes"
            ),
            code="size",
            media_type=media_type,
        )
    )


def _check_audio(
    limits: MediaLimits,
    has_audio: bool | None,
    *,
    platform: Platform,
    media_type: str,
    issues: OutputIssues,
) -> None:
    if has_audio is not False:
        return
    label = _media_label(platform, media_type)
    if media_type in _AUDIO_TYPES:
        message = f"{label} has no audio stream"
    elif limits.audio_required:
        message = f"{label} has no audio stream but audio is required"
    else:
        return
    issues.append(
        OutputIssue(
            path="audio",
            message=message,
            code="audio",
            media_type=media_type,
        )
    )


def _check_loudness(
    limits: MediaLimits,
    loudness_lufs: float | None,
    *,
    tolerance: float | None,
    platform: Platform,
    media_type: str,
    issues: OutputIssues,
) -> None:
    target = limits.loudness_lufs
    if target is None or loudness_lufs is None:
        return
    difference = abs(loudness_lufs - target)
    allowed = tolerance if tolerance is not None else _EPSILON
    if difference <= allowed + _EPSILON:
        return
    if tolerance is None:
        detail = f"target of {target:g} LUFS"
    else:
        detail = f"target of {target:g} LUFS (tolerance {tolerance:g} LU)"
    issues.append(
        OutputIssue(
            path="loudness",
            message=(
                f"{_media_label(platform, media_type)} loudness "
                f"{loudness_lufs:g} LUFS is {difference:g} LU from the {detail}"
            ),
            code="loudness",
            media_type=media_type,
        )
    )


def _validate_loudness_argument(value: Any) -> float | None:
    number = _number(value)
    if value is not None and number is None:
        raise ValueError("loudness_lufs must be a finite number or None")
    return number


def _validate_tolerance(value: Any) -> float | None:
    if value is None:
        return None
    number = _number(value)
    if number is None or number < 0:
        raise ValueError("loudness_tolerance_lufs must be a finite non-negative number")
    return number


def _measure_loudness(
    path: Path,
    *,
    executable: str,
    timeout_seconds: float,
    runner: OutputLoudnessRunner | None,
) -> float:
    return measure_loudness(
        path,
        executable=executable,
        timeout_seconds=timeout_seconds,
        runner=runner,
    )


def measure_loudness(
    path: str | Path,
    *,
    executable: str = "ffmpeg",
    timeout_seconds: float = DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS,
    runner: OutputLoudnessRunner | None = None,
) -> float:
    """Measure the integrated loudness of *path* in LUFS with ffmpeg.

    The ``loudnorm`` filter prints its analysis as JSON on stderr. Only the
    measured ``input_i`` is used; no gain is applied to the file. Tests inject
    *runner* so the default suite never needs a real encoder.
    """
    target = Path(path)
    if not target.is_file():
        raise OutputValidationError(
            [
                OutputIssue(
                    path="loudness",
                    message=f"cannot measure loudness: {target} does not exist",
                    code="loudness",
                )
            ]
        )
    if not isinstance(executable, str) or not executable.strip():
        raise ValueError("ffmpeg executable must be a non-empty string")
    argv = [
        executable.strip(),
        "-hide_banner",
        "-nostats",
        "-i",
        str(target),
        "-af",
        "loudnorm=print_format=json",
        "-f",
        "null",
        "-",
    ]
    try:
        completed = (
            runner(argv)
            if runner is not None
            else subprocess.run(
                argv,
                capture_output=True,
                text=True,
                check=False,
                timeout=float(timeout_seconds),
            )
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise OutputValidationError(
            [
                OutputIssue(
                    path="loudness",
                    message=f"loudness could not be measured: {type(error).__name__}",
                    code="loudness",
                )
            ]
        ) from error
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        raise OutputValidationError(
            [
                OutputIssue(
                    path="loudness",
                    message=(
                        "loudness could not be measured: ffmpeg exited with "
                        f"code {completed.returncode}"
                        + (f": {stderr[:200]}" if stderr else "")
                    ),
                    code="loudness",
                )
            ]
        )
    payload = _last_json_object(completed.stderr or "")
    if payload is None:
        raise OutputValidationError(
            [
                OutputIssue(
                    path="loudness",
                    message="loudness could not be measured: ffmpeg returned no JSON",
                    code="loudness",
                )
            ]
        )
    measured = _number(payload.get("input_i"))
    if measured is None:
        raise OutputValidationError(
            [
                OutputIssue(
                    path="loudness",
                    message="loudness could not be measured: input_i is missing",
                    code="loudness",
                )
            ]
        )
    return measured


def _last_json_object(text: str) -> Mapping[str, Any] | None:
    end = text.rfind("}")
    while end >= 0:
        start = text.rfind("{", 0, end + 1)
        if start < 0:
            return None
        try:
            payload = json.loads(text[start : end + 1])
        except (TypeError, ValueError):
            end = text.rfind("}", 0, start)
            continue
        if isinstance(payload, Mapping):
            return payload
        end = text.rfind("}", 0, start)
    return None


def validate_output(
    output: Any,
    platform: str | Platform | PlatformCatalog | None = None,
    media_type: str | None = None,
    *,
    catalog: PlatformCatalog | None = None,
    probe: FFprobeInfo | None = None,
    duration_seconds: float | None = None,
    width: int | None = None,
    height: int | None = None,
    size_bytes: int | None = None,
    has_audio: bool | None = None,
    audio_present: bool | None = None,
    loudness_lufs: float | None = None,
    loudness_tolerance_lufs: float | None = None,
    measure: bool = False,
    ffprobe_executable: str = "ffprobe",
    ffprobe_timeout_seconds: float = DEFAULT_MEDIA_FFPROBE_TIMEOUT_SECONDS,
    ffprobe_runner: FFprobeRunner | None = None,
    ffmpeg_executable: str = "ffmpeg",
    ffmpeg_timeout_seconds: float = DEFAULT_MEDIA_FFMPEG_TIMEOUT_SECONDS,
    loudness_runner: OutputLoudnessRunner | None = None,
) -> OutputIssues:
    """Return every platform limit *output* violates.

    *output* may be a path, a render result (video, audio, animated, loop or web
    card) or a mapping with the same fields. Missing measurements are skipped
    rather than guessed; pass *probe*, *size_bytes* or *loudness_lufs* when the
    caller already measured them. Set *measure* (or pass *loudness_runner*) to
    run ffmpeg's ``loudnorm`` analysis when a measured loudness is absent.
    """
    facts = _facts(output)
    active_platform = _resolve_platform(platform, catalog, facts)
    active_type = _resolve_media_type(active_platform, media_type, facts.path)
    issues = OutputIssues(platform=active_platform, media_type=active_type)
    issues.output = facts.path

    limits = active_platform.limit(active_type) if active_type else None
    if limits is None:
        available = ", ".join(active_platform.media_types) or "none"
        issues.append(
            OutputIssue(
                path="media_type",
                message=(
                    f"{active_platform.display_name} does not support "
                    f"{active_type or 'an unknown media type'} "
                    f"(supported: {available})"
                ),
                code="media_type",
                media_type=active_type,
            )
        )
        return issues

    active_probe = _probe_facts(
        facts.path,
        media_type=active_type,
        probe=probe if probe is not None else facts.probe,
        ffprobe_executable=ffprobe_executable,
        ffprobe_timeout_seconds=ffprobe_timeout_seconds,
        ffprobe_runner=ffprobe_runner,
        issues=issues,
    )
    issues.probe = active_probe

    resolved_duration = duration_seconds
    if resolved_duration is None:
        resolved_duration = facts.duration_seconds
    if resolved_duration is None:
        resolved_duration = _probe_duration(active_probe)

    resolved_width = width if width is not None else facts.width
    resolved_height = height if height is not None else facts.height
    if resolved_width is None or resolved_height is None:
        probe_width, probe_height = _probe_dimensions(active_probe)
        if resolved_width is None:
            resolved_width = probe_width
        if resolved_height is None:
            resolved_height = probe_height
    if (
        (resolved_width is None or resolved_height is None)
        and facts.path is not None
        and active_type in _VISUAL_TYPES
    ):
        dimensions = _probe_image_dimensions(facts.path)
        if dimensions is not None:
            if resolved_width is None:
                resolved_width = dimensions[0]
            if resolved_height is None:
                resolved_height = dimensions[1]

    resolved_audio = has_audio if has_audio is not None else audio_present
    if resolved_audio is None:
        resolved_audio = facts.has_audio
    if resolved_audio is None:
        resolved_audio = _probe_has_audio(active_probe)

    resolved_size = size_bytes
    if resolved_size is None:
        resolved_size = facts.size_bytes
    if resolved_size is None and facts.path is not None and facts.path.is_file():
        try:
            resolved_size = facts.path.stat().st_size
        except OSError:
            resolved_size = None
    if resolved_size is None:
        resolved_size = _probe_size(active_probe)

    resolved_loudness = _validate_loudness_argument(
        loudness_lufs if loudness_lufs is not None else facts.loudness_lufs
    )
    tolerance = _validate_tolerance(loudness_tolerance_lufs)
    if tolerance is None:
        tolerance = limits.loudness_tolerance_lufs
    target = limits.loudness_lufs
    should_measure = (
        resolved_loudness is None
        and target is not None
        and (measure or loudness_runner is not None)
    )
    if (
        should_measure
        and facts.path is not None
        and facts.path.is_file()
        and active_type not in {"web_card"}
        and resolved_audio is not False
    ):
        try:
            resolved_loudness = _measure_loudness(
                facts.path,
                executable=ffmpeg_executable,
                timeout_seconds=ffmpeg_timeout_seconds,
                runner=loudness_runner,
            )
        except OutputValidationError as error:
            issues.extend(error.issues)

    _check_duration(
        limits,
        resolved_duration,
        platform=active_platform,
        media_type=active_type,
        issues=issues,
    )
    _check_resolution(
        limits,
        resolved_width,
        resolved_height,
        platform=active_platform,
        media_type=active_type,
        issues=issues,
    )
    _check_size(
        limits,
        resolved_size,
        platform=active_platform,
        media_type=active_type,
        issues=issues,
    )
    _check_audio(
        limits,
        resolved_audio,
        platform=active_platform,
        media_type=active_type,
        issues=issues,
    )
    _check_loudness(
        limits,
        resolved_loudness,
        tolerance=tolerance,
        platform=active_platform,
        media_type=active_type,
        issues=issues,
    )
    return issues


def validate_output_or_raise(
    output: Any,
    platform: str | Platform | PlatformCatalog | None = None,
    media_type: str | None = None,
    **kwargs: Any,
) -> OutputIssues:
    """Validate *output* and return the report, raising every issue at once."""
    issues = validate_output(output, platform, media_type, **kwargs)
    issues.raise_for_errors()
    return issues


validate_platform_output = validate_output
