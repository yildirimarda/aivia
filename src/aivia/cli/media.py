"""``aivia media``: generate one artifact or edit an existing image.

Every command builds the same screened, budgeted, cached and cost tracked path
the library services use. ``--estimate`` prints the price table without a paid
call, ``--dry-run`` swaps every provider for the deterministic mock and
``--provider``/``--strategy`` choose the router chain for one invocation.
"""

from __future__ import annotations

import argparse
import asyncio
import mimetypes
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from pydantic import ValidationError
from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError

from aivia.cli.registry import register_command
from aivia.core import paths
from aivia.core.budget import BudgetGuard
from aivia.core.config.budget import BudgetConfig
from aivia.core.config.capability import CapabilityChainConfig, CapabilityStrategy
from aivia.core.config.database import DatabaseConfig
from aivia.core.config.formats import FormatRecipeConfig, StageConfig
from aivia.core.config.providers import ProvidersConfig
from aivia.core.config.resolve import resolve_config
from aivia.core.cost import CostTracker
from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.dry_run import apply_dry_run
from aivia.core.errors import AiviaError, ConfigError
from aivia.core.estimate import Estimator, render_estimate
from aivia.core.router import Router
from aivia.media.image_edit_service import ImageEditService
from aivia.media.image_service import ImageService
from aivia.media.music_service import MusicService
from aivia.media.sfx_service import SFXService
from aivia.media.stt_service import STTService
from aivia.media.tts_service import SpeechService
from aivia.media.video_service import VideoService
from aivia.providers import image_edit as _image_edit  # noqa: F401
from aivia.providers.protocols.audio import AudioFormat
from aivia.providers.protocols.image import ImageRequest
from aivia.providers.protocols.image_edit import ImageEditOperation, ImageEditRequest
from aivia.providers.protocols.media import ImageFormat, MediaInput, VideoFormat
from aivia.providers.protocols.music import MusicRequest
from aivia.providers.protocols.sfx import SFXRequest
from aivia.providers.protocols.stt import STTRequest
from aivia.providers.protocols.tts import TTSRequest
from aivia.providers.protocols.video import VideoRequest
from aivia.providers.provider_factory import build_providers
from aivia.providers.storage_factory import build_storage_provider

__all__ = [
    "CAPABILITIES",
    "KINDS",
    "configure",
    "run_media",
]

KINDS = ("image", "video", "speech", "transcription", "music", "sfx")
"""The capabilities this command can generate."""

CAPABILITIES: dict[str, str] = {
    "image": "image_gen",
    "video": "video_gen",
    "speech": "tts",
    "transcription": "stt",
    "music": "music_gen",
    "sfx": "sfx",
    "edit": "image_edit",
}
"""The router capability behind each command name."""

_EDIT_ALIASES: dict[str, ImageEditOperation] = {
    "edit": ImageEditOperation.EDIT,
    "instruction": ImageEditOperation.EDIT,
    "instruction-edit": ImageEditOperation.EDIT,
    "instruction_edit": ImageEditOperation.EDIT,
    "inpaint": ImageEditOperation.INPAINT,
    "outpaint": ImageEditOperation.OUTPAINT,
    "outpainting": ImageEditOperation.OUTPAINT,
    "expand": ImageEditOperation.OUTPAINT,
    "upscale": ImageEditOperation.UPSCALE,
    "upscaling": ImageEditOperation.UPSCALE,
    "upscale-image": ImageEditOperation.UPSCALE,
    "remove-background": ImageEditOperation.REMOVE_BACKGROUND,
    "remove_background": ImageEditOperation.REMOVE_BACKGROUND,
    "background-removal": ImageEditOperation.REMOVE_BACKGROUND,
    "background_removal": ImageEditOperation.REMOVE_BACKGROUND,
    "remove-bg": ImageEditOperation.REMOVE_BACKGROUND,
    "remove_bg": ImageEditOperation.REMOVE_BACKGROUND,
    "rembg": ImageEditOperation.REMOVE_BACKGROUND,
    "cutout": ImageEditOperation.REMOVE_BACKGROUND,
}
"""Every accepted operation name for ``aivia media edit``."""

_STRATEGIES = tuple(strategy.value for strategy in CapabilityStrategy)


def _configure_resolution(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        metavar="PATH",
        help="config file to merge instead of the default location",
    )
    parser.add_argument(
        "--env",
        "--environment",
        dest="environment",
        metavar="NAME",
        help="merge config.<NAME>.yaml beside the user config",
    )
    parser.add_argument(
        "--profile",
        metavar="NAME",
        help="merge the overrides of a configured profile",
    )
    parser.add_argument(
        "--set",
        dest="set_values",
        action="append",
        metavar="KEY=VALUE",
        help="override a dotted key; repeatable and applied after the profile",
    )


def _configure_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "prompt",
        nargs="?",
        metavar="PROMPT",
        help="prompt text; omit it when using --prompt or --prompt-file",
    )
    parser.add_argument(
        "--prompt",
        dest="prompt_option",
        metavar="TEXT",
        help="prompt text instead of the positional argument",
    )
    parser.add_argument(
        "--prompt-file",
        metavar="PATH",
        help="read the prompt from a UTF-8 file (use - for standard input)",
    )
    parser.add_argument(
        "--provider",
        action="append",
        default=None,
        metavar="NAME",
        help="provider name to use; repeat to build an ordered fallback chain",
    )
    parser.add_argument(
        "--strategy",
        choices=_STRATEGIES,
        metavar="NAME",
        help=f"provider selection strategy ({', '.join(_STRATEGIES)})",
    )
    parser.add_argument("--model", metavar="NAME", help="provider model id")
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        metavar="N",
        help="number of outputs to request (default: 1)",
    )
    parser.add_argument("--seed", type=int, metavar="N", help="reproducible seed")
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="copy the first output (or write the transcript) to PATH",
    )
    parser.add_argument(
        "--database",
        "--db",
        dest="database",
        metavar="PATH",
        help="database file to use instead of the one in the user data directory",
    )
    _configure_resolution(parser)


def _configure_image(parser: argparse.ArgumentParser) -> None:
    _configure_common(parser)
    parser.add_argument("--negative-prompt", metavar="TEXT")
    parser.add_argument("--width", type=int, metavar="PX")
    parser.add_argument("--height", type=int, metavar="PX")
    parser.add_argument("--aspect-ratio", metavar="W:H")
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=tuple(item.value for item in ImageFormat),
        help="image container format",
    )
    parser.add_argument("--quality", metavar="NAME")
    parser.add_argument("--style", metavar="NAME")
    parser.set_defaults(_media_kind="image")


def _configure_video(parser: argparse.ArgumentParser) -> None:
    _configure_common(parser)
    parser.add_argument("--negative-prompt", metavar="TEXT")
    parser.add_argument("--width", type=int, metavar="PX")
    parser.add_argument("--height", type=int, metavar="PX")
    parser.add_argument("--aspect-ratio", metavar="W:H")
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=tuple(item.value for item in VideoFormat),
        help="video container format",
    )
    parser.add_argument("--duration", type=float, metavar="SECONDS")
    parser.add_argument("--fps", type=int, metavar="FPS")
    parser.add_argument(
        "--audio",
        dest="generate_audio",
        action="store_true",
        help="ask for a native soundtrack",
    )
    parser.add_argument("--audio-prompt", metavar="TEXT")
    parser.set_defaults(_media_kind="video")


def _configure_speech(parser: argparse.ArgumentParser) -> None:
    _configure_common(parser)
    parser.add_argument(
        "--voice",
        default="default",
        metavar="NAME",
        help="provider voice id (default: default)",
    )
    parser.add_argument("--language", metavar="CODE")
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=tuple(item.value for item in AudioFormat),
        help="audio container format",
    )
    parser.add_argument("--speed", type=float, metavar="FACTOR")
    parser.add_argument("--instructions", metavar="TEXT")
    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        help="ask the provider for word level timestamps",
    )
    parser.set_defaults(_media_kind="speech")


def _configure_transcription(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "audio",
        nargs="?",
        metavar="AUDIO",
        help="audio or video file to transcribe",
    )
    parser.add_argument(
        "--audio-file",
        "--input-file",
        "--file",
        dest="audio_file",
        metavar="PATH",
        help="audio or video file to transcribe instead of the positional argument",
    )
    parser.add_argument(
        "--prompt",
        dest="prompt_option",
        metavar="TEXT",
        help="context prompt for the transcriber",
    )
    parser.add_argument(
        "--prompt-file",
        metavar="PATH",
        help="read the context prompt from a UTF-8 file",
    )
    parser.add_argument(
        "--provider",
        action="append",
        default=None,
        metavar="NAME",
        help="provider name to use; repeat to build an ordered fallback chain",
    )
    parser.add_argument(
        "--strategy",
        choices=_STRATEGIES,
        metavar="NAME",
        help=f"provider selection strategy ({', '.join(_STRATEGIES)})",
    )
    parser.add_argument("--model", metavar="NAME", help="provider model id")
    parser.add_argument("--language", metavar="CODE")
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="write the transcript text to PATH instead of standard output",
    )
    parser.add_argument(
        "--database",
        "--db",
        dest="database",
        metavar="PATH",
        help="database file to use instead of the one in the user data directory",
    )
    _configure_resolution(parser)
    parser.set_defaults(_media_kind="transcription", count=1, seed=None)


def _configure_music(parser: argparse.ArgumentParser) -> None:
    _configure_common(parser)
    parser.add_argument("--lyrics", metavar="TEXT")
    parser.add_argument("--lyrics-file", metavar="PATH")
    parser.add_argument(
        "--instrumental",
        action="store_true",
        help="generate an instrumental track without lyrics",
    )
    parser.add_argument("--duration", type=float, metavar="SECONDS")
    parser.add_argument("--tempo", type=float, metavar="BPM")
    parser.add_argument("--style", metavar="NAME")
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=tuple(item.value for item in AudioFormat),
        help="audio container format",
    )
    parser.set_defaults(_media_kind="music")


def _configure_sfx(parser: argparse.ArgumentParser) -> None:
    _configure_common(parser)
    parser.add_argument("--duration", type=float, metavar="SECONDS")
    parser.add_argument(
        "--loop",
        action="store_true",
        help="request a seamless loop",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=tuple(item.value for item in AudioFormat),
        help="audio container format",
    )
    parser.set_defaults(_media_kind="sfx")


def _configure_edit(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "operation",
        nargs="?",
        metavar="OPERATION_OR_IMAGE",
        help=(
            "edit operation (instruction, remove-background, upscale, outpaint, "
            "inpaint) or the source image path"
        ),
    )
    parser.add_argument(
        "image",
        nargs="?",
        metavar="IMAGE",
        help="source image to edit when OPERATION_OR_IMAGE names an operation",
    )
    parser.add_argument(
        "--operation",
        dest="operation_option",
        metavar="NAME",
        help="edit operation instead of the positional name",
    )
    parser.add_argument(
        "--image",
        "--input",
        "--input-file",
        "--file",
        dest="image_option",
        metavar="PATH",
        help="source image to edit instead of the positional path",
    )
    parser.add_argument("--mask", metavar="PATH", help="mask image for an inpaint")
    parser.add_argument(
        "--prompt",
        dest="prompt_option",
        metavar="TEXT",
        help="instruction text for an instruction edit, inpaint or outpaint",
    )
    parser.add_argument(
        "--prompt-file",
        metavar="PATH",
        help="read the instruction from a UTF-8 file (use - for standard input)",
    )
    parser.add_argument(
        "--provider",
        action="append",
        default=None,
        metavar="NAME",
        help="provider name to use; repeat to build an ordered fallback chain",
    )
    parser.add_argument(
        "--strategy",
        choices=_STRATEGIES,
        metavar="NAME",
        help=f"provider selection strategy ({', '.join(_STRATEGIES)})",
    )
    parser.add_argument("--model", metavar="NAME", help="provider model id")
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        metavar="N",
        help="number of outputs to request (default: 1)",
    )
    parser.add_argument("--seed", type=int, metavar="N", help="reproducible seed")
    parser.add_argument("--width", type=int, metavar="PX", help="target width")
    parser.add_argument("--height", type=int, metavar="PX", help="target height")
    parser.add_argument(
        "--scale",
        "--factor",
        dest="scale",
        type=float,
        metavar="FACTOR",
        help="upscale factor greater than 1 (default: 2)",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=tuple(item.value for item in ImageFormat),
        help="image container format",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        metavar="SECONDS",
        help="maximum time to wait for a remote edit",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="copy the first output to PATH",
    )
    parser.add_argument(
        "--database",
        "--db",
        dest="database",
        metavar="PATH",
        help="database file to use instead of the one in the user data directory",
    )
    _configure_resolution(parser)
    parser.set_defaults(_media_handler=_run_edit, _media_kind="edit")


def configure(parser: argparse.ArgumentParser) -> None:
    """Add the media group with its ``generate`` and ``edit`` commands."""
    subparsers = parser.add_subparsers(dest="media_command", metavar="COMMAND")
    generate = subparsers.add_parser(
        "generate",
        help="Generate an image, video, voice, transcript, song or sound effect",
        description="Generate one media artifact through the configured providers",
    )
    kinds = generate.add_subparsers(dest="media_kind", metavar="KIND")
    builders = {
        "image": _configure_image,
        "video": _configure_video,
        "speech": _configure_speech,
        "transcription": _configure_transcription,
        "music": _configure_music,
        "sfx": _configure_sfx,
    }
    for name, builder in builders.items():
        kind = kinds.add_parser(
            name,
            help=f"Generate {name}",
            description=f"Generate {name} through the configured provider chain",
        )
        builder(kind)
    generate.set_defaults(_media_kind="image")
    edit = subparsers.add_parser(
        "edit",
        help="Edit an image: instruction, background removal, upscale or outpaint",
        description="Edit one image through the configured provider chain",
    )
    _configure_edit(edit)
    parser.set_defaults(_media_handler=run_media)


def run_media(args: argparse.Namespace) -> int:
    """Dispatch to the selected media command."""
    handler = getattr(args, "_media_handler", None)
    if handler is not None and handler is not run_media:
        return int(handler(args))
    kind = getattr(args, "_media_kind", None)
    if handler is None or kind not in CAPABILITIES:
        print("error: missing media command", file=sys.stderr)
        print(
            "usage: aivia media {generate,edit}",
            file=sys.stderr,
        )
        return 2
    return int(_run_generate(args, kind))


def _read_text_file(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    return Path(path).expanduser().read_text(encoding="utf-8")


def _prompt(args: argparse.Namespace) -> str:
    positional = getattr(args, "prompt", None)
    option = getattr(args, "prompt_option", None)
    path = getattr(args, "prompt_file", None)
    sources: list[str] = [
        value for value in (positional, option) if isinstance(value, str) and value
    ]
    if path:
        sources.append(_read_text_file(path))
    if not sources:
        raise ConfigError(
            "a prompt is required (use PROMPT, --prompt or --prompt-file)"
        )
    if len(sources) > 1:
        raise ConfigError("pass only one of PROMPT, --prompt or --prompt-file")
    prompt = sources[0].strip()
    if not prompt:
        raise ConfigError("the prompt must not be empty")
    return prompt


def _prompt_or_empty(args: argparse.Namespace) -> str:
    sources = (
        getattr(args, "prompt", None),
        getattr(args, "prompt_option", None),
        getattr(args, "prompt_file", None),
    )
    if not any(sources):
        return ""
    return _prompt(args)


def _lyrics(args: argparse.Namespace) -> str | None:
    option = getattr(args, "lyrics", None)
    path = getattr(args, "lyrics_file", None)
    if option and path:
        raise ConfigError("pass only one of --lyrics or --lyrics-file")
    if path:
        text = _read_text_file(path).strip()
        return text or None
    text = option.strip() if isinstance(option, str) else ""
    return text or None


def _audio_path(args: argparse.Namespace) -> Path:
    positional = getattr(args, "audio", None)
    option = getattr(args, "audio_file", None)
    if positional and option:
        raise ConfigError("pass the audio file once, positionally or with --audio-file")
    selected = option or positional
    if not selected:
        raise ConfigError("an audio or video file is required")
    path = Path(selected).expanduser()
    if not path.is_file():
        raise ConfigError(f"audio file not found: {path}")
    return path


def _edit_operation(args: argparse.Namespace) -> tuple[ImageEditOperation, str]:
    """Resolve the operation and the source image from the flexible arguments."""
    explicit = getattr(args, "operation_option", None)
    first = getattr(args, "operation", None)
    second = getattr(args, "image", None)
    image_option = getattr(args, "image_option", None)

    first_operation = _EDIT_ALIASES.get(first.strip().lower()) if first else None
    explicit_operation = (
        _EDIT_ALIASES.get(explicit.strip().lower()) if explicit else None
    )
    if explicit and explicit_operation is None:
        raise ConfigError(f"unknown edit operation: {explicit}")

    if image_option:
        if second is not None:
            raise ConfigError("pass the source image once")
        if first is not None and first_operation is None:
            raise ConfigError("pass the source image once")
        operation = explicit_operation or first_operation or ImageEditOperation.EDIT
        return operation, image_option

    if first_operation is not None:
        operation = explicit_operation or first_operation
        if second is None:
            raise ConfigError("a source image is required")
        return operation, second

    if first is None:
        raise ConfigError("a source image is required")
    if second is not None:
        raise ConfigError("pass the source image once")
    return explicit_operation or ImageEditOperation.EDIT, first


def _edit_image_path(path: str) -> Path:
    selected = Path(path).expanduser()
    if not selected.is_file():
        raise ConfigError(f"image file not found: {selected}")
    return selected


def _media_input(path: Path) -> MediaInput:
    media_type = mimetypes.guess_type(path.name)[0] or "image/png"
    return MediaInput(
        data=path.read_bytes(),
        media_type=media_type,
        filename=path.name,
    )


def _edit_request(
    args: argparse.Namespace,
    operation: ImageEditOperation,
    image: Path,
    prompt: str,
) -> ImageEditRequest:
    values: dict[str, Any] = {
        "image": _media_input(image),
        "operation": operation,
        "prompt": prompt or None,
        "model": getattr(args, "model", None),
        "width": getattr(args, "width", None),
        "height": getattr(args, "height", None),
        "scale": None,
        "seed": getattr(args, "seed", None),
        "count": max(1, getattr(args, "count", 1)),
        "timeout_seconds": getattr(args, "timeout", None),
    }
    mask = getattr(args, "mask", None)
    if mask:
        values["mask"] = _media_input(_edit_image_path(mask))
    if operation is ImageEditOperation.UPSCALE:
        values["scale"] = (
            getattr(args, "scale", None) if getattr(args, "scale", None) else 2.0
        )
    output_format = getattr(args, "output_format", None)
    if operation is ImageEditOperation.REMOVE_BACKGROUND:
        if output_format and output_format != ImageFormat.PNG.value:
            raise ConfigError("background removal writes a PNG with alpha")
        values["format"] = ImageFormat.PNG
    elif output_format:
        values["format"] = output_format
    return ImageEditRequest(**values)


def _validate_edit_arguments(
    args: argparse.Namespace,
    operation: ImageEditOperation,
) -> None:
    if operation is ImageEditOperation.OUTPAINT:
        width = getattr(args, "width", None)
        height = getattr(args, "height", None)
        if width is None or height is None:
            raise ConfigError("outpaint requires both --width and --height")
    if operation is ImageEditOperation.INPAINT and not getattr(args, "mask", None):
        raise ConfigError("inpaint requires --mask")


def _provider_supports_edit(provider: Any, operation: ImageEditOperation) -> bool:
    checker = getattr(provider, "supports_operation", None)
    if callable(checker):
        try:
            return bool(checker(operation))
        except (TypeError, ValueError):
            return False
    operations = getattr(provider, "supported_operations", None)
    if operations is not None:
        return operation in operations
    return True


def _edit_units(args: argparse.Namespace) -> tuple[float, str]:
    return float(max(1, getattr(args, "count", 1))), "image"


def _print_edit_estimate(args: argparse.Namespace) -> int:
    config = _load_config(args)
    capability = CAPABILITIES["edit"]
    metadata = _provider_metadata(config)
    chain = _chain(args, capability, metadata)
    if not chain:
        print(f"error: no provider serves {capability}", file=sys.stderr)
        return 1
    units, unit_kind = _edit_units(args)
    stage = StageConfig(
        stage=capability,
        options={"units": units, "unit": unit_kind},
    )
    recipe = FormatRecipeConfig(stages=[stage], capabilities=[capability])
    estimate = Estimator(
        metadata,
        chain=chain,
        strategy=_strategy(args),
    ).estimate("edit", recipe)
    print(render_estimate(estimate), end="")
    return 0 if estimate.complete else 1


async def _edit(
    runtime: _Runtime,
    args: argparse.Namespace,
    operation: ImageEditOperation,
    image: Path,
) -> Any:
    prompt = _prompt_or_empty(args)
    service = ImageEditService(
        runtime.router,
        **_common_service_values(runtime, args, "edit"),
    )
    request = _edit_request(args, operation, image, prompt)
    token = service.screen(prompt) if prompt else None
    return await service.edit(request, token=token)


def _run_edit(args: argparse.Namespace) -> int:
    try:
        if getattr(args, "estimate", False):
            return _print_edit_estimate(args)
        operation, image_value = _edit_operation(args)
        _validate_edit_arguments(args, operation)
        image = _edit_image_path(image_value)
        runtime = _runtime(args, "edit", edit_operation=operation)
        try:
            outcome = asyncio.run(_edit(runtime, args, operation, image))
        finally:
            runtime.engine.dispose()
        return _print_outcome(outcome, args, "edit")
    except (AiviaError, ConfigError, OSError, SQLAlchemyError, ValidationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _load_config(args: argparse.Namespace) -> Any:
    resolved = resolve_config(
        config_file=args.config,
        environment=args.environment,
        profile=args.profile,
        set_values=args.set_values,
    )
    config = resolved.to_app_config()
    if getattr(args, "dry_run", False):
        config = apply_dry_run(config)
    return config


def _providers_config(config: Any) -> ProvidersConfig:
    section = getattr(config, "providers", None)
    return section if isinstance(section, ProvidersConfig) else ProvidersConfig()


def _provider_metadata(config: Any) -> dict[str, Any]:
    from aivia.cli.run import provider_metadata

    return provider_metadata(config)


def _chain(
    args: argparse.Namespace,
    capability: str,
    available: Mapping[str, Any],
) -> list[str]:
    explicit = list(getattr(args, "provider", None) or ())
    if explicit:
        missing = sorted(set(explicit) - set(available))
        if missing:
            names = ", ".join(sorted(available)) or "none"
            raise ConfigError(
                f"unknown provider: {', '.join(missing)} (available: {names})"
            )
        return explicit
    return [
        name
        for name, metadata in available.items()
        if capability in getattr(metadata, "capabilities", ())
    ]


def _strategy(args: argparse.Namespace) -> CapabilityStrategy:
    value = getattr(args, "strategy", None)
    return CapabilityStrategy(value) if value else CapabilityStrategy.FIRST_AVAILABLE


def _estimate_units(
    kind: str, args: argparse.Namespace, prompt: str
) -> tuple[float, str]:
    if kind == "speech":
        return float(max(1, len(prompt))), "character"
    if kind == "image":
        return float(max(1, getattr(args, "count", 1))), "image"
    if kind == "sfx":
        return float(max(1, getattr(args, "count", 1))), "clip"
    if kind == "music":
        duration = getattr(args, "duration", None)
        if duration:
            return float(duration) * max(1, getattr(args, "count", 1)), "second"
        return float(max(1, getattr(args, "count", 1))), "track"
    if kind == "video":
        duration = getattr(args, "duration", None)
        if duration:
            return float(duration) * max(1, getattr(args, "count", 1)), "second"
        return float(max(1, getattr(args, "count", 1))), "clip"
    return 1.0, "minute"


def _print_estimate(args: argparse.Namespace, kind: str) -> int:
    config = _load_config(args)
    capability = CAPABILITIES[kind]
    metadata = _provider_metadata(config)
    chain = _chain(args, capability, metadata)
    if not chain:
        print(f"error: no provider serves {capability}", file=sys.stderr)
        return 1
    prompt = _prompt_or_empty(args) if kind != "transcription" else ""
    units, unit_kind = _estimate_units(kind, args, prompt)
    stage = StageConfig(
        stage=capability,
        options={"units": units, "unit": unit_kind},
    )
    recipe = FormatRecipeConfig(stages=[stage], capabilities=[capability])
    estimate = Estimator(
        metadata,
        chain=chain,
        strategy=_strategy(args),
    ).estimate(kind, recipe)
    print(render_estimate(estimate), end="")
    return 0 if estimate.complete else 1


def _database_path(args: argparse.Namespace, config: Any) -> Path:
    explicit = getattr(args, "database", None)
    if explicit:
        return Path(explicit).expanduser()
    database = getattr(config, "database", None)
    if not isinstance(database, DatabaseConfig):
        database = None
    filename = database.filename if database is not None else DatabaseConfig().filename
    return paths.resolve_paths(args.config).data_dir / filename


def _open_database(path: Path) -> Engine:
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_database_engine(path=path)
    try:
        upgrade_database(engine=engine)
    except BaseException:
        engine.dispose()
        raise
    return engine


@dataclass(slots=True)
class _Runtime:
    config: Any
    router: Router
    engine: Engine
    tracker: CostTracker
    storage: Any
    budget: BudgetGuard | None
    safety: Any


def _runtime(
    args: argparse.Namespace,
    kind: str,
    *,
    edit_operation: ImageEditOperation | None = None,
) -> _Runtime:
    config = _load_config(args)
    capability = CAPABILITIES[kind]
    section = _providers_config(config)
    providers = build_providers(section)
    if not providers:
        raise ConfigError(
            "no providers are configured; add a providers section or use --dry-run"
        )
    chain = _chain(args, capability, providers)
    if edit_operation is not None:
        chain = [
            name
            for name in chain
            if _provider_supports_edit(providers[name], edit_operation)
        ]
        if not chain:
            raise ConfigError(f"no configured provider supports {edit_operation.value}")
    if not chain:
        raise ConfigError(f"no configured provider serves {capability}")
    router = Router(
        providers,
        configs={
            capability: CapabilityChainConfig(
                strategy=_strategy(args),
                chain=chain,
            )
        },
    )
    engine = _open_database(_database_path(args, config))
    tracker = CostTracker(engine)
    storage_config = getattr(config, "storage", None)
    storage = build_storage_provider(
        storage_config,
        data_dir=paths.resolve_paths(args.config).data_dir,
    )
    budget_config = getattr(config, "budget", None)
    budget = (
        BudgetGuard(budget_config, tracker)
        if isinstance(budget_config, BudgetConfig)
        else None
    )
    return _Runtime(
        config=config,
        router=router,
        engine=engine,
        tracker=tracker,
        storage=storage,
        budget=budget,
        safety=getattr(config, "safety", None),
    )


def _common_service_values(
    runtime: _Runtime,
    args: argparse.Namespace,
    kind: str,
) -> dict[str, Any]:
    return {
        "tracker": runtime.tracker,
        "engine": runtime.engine,
        "storage": runtime.storage,
        "budget": runtime.budget,
        "safety": runtime.safety,
        "key_prefix": kind,
        "no_cache": bool(getattr(args, "no_cache", False)),
    }


def _image_request(args: argparse.Namespace, prompt: str) -> ImageRequest:
    values: dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": getattr(args, "negative_prompt", None),
        "model": getattr(args, "model", None),
        "width": getattr(args, "width", None),
        "height": getattr(args, "height", None),
        "aspect_ratio": getattr(args, "aspect_ratio", None),
        "quality": getattr(args, "quality", None),
        "style": getattr(args, "style", None),
        "seed": getattr(args, "seed", None),
        "count": max(1, getattr(args, "count", 1)),
    }
    output_format = getattr(args, "output_format", None)
    if output_format:
        values["format"] = output_format
    return ImageRequest(**values)


def _video_request(args: argparse.Namespace, prompt: str) -> VideoRequest:
    values: dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": getattr(args, "negative_prompt", None),
        "model": getattr(args, "model", None),
        "width": getattr(args, "width", None),
        "height": getattr(args, "height", None),
        "aspect_ratio": getattr(args, "aspect_ratio", None),
        "duration_seconds": getattr(args, "duration", None),
        "fps": getattr(args, "fps", None),
        "seed": getattr(args, "seed", None),
        "count": max(1, getattr(args, "count", 1)),
        "generate_audio": bool(getattr(args, "generate_audio", False)),
        "audio_prompt": getattr(args, "audio_prompt", None),
    }
    output_format = getattr(args, "output_format", None)
    if output_format:
        values["format"] = output_format
    return VideoRequest(**values)


def _speech_request(args: argparse.Namespace, prompt: str) -> TTSRequest:
    values: dict[str, Any] = {
        "text": prompt,
        "voice": getattr(args, "voice", None) or "default",
        "model": getattr(args, "model", None),
        "language": getattr(args, "language", None),
        "speed": getattr(args, "speed", None) or 1.0,
        "instructions": getattr(args, "instructions", None),
        "seed": getattr(args, "seed", None),
        "word_timestamps": bool(getattr(args, "word_timestamps", False)),
    }
    output_format = getattr(args, "output_format", None)
    if output_format:
        values["format"] = output_format
    return TTSRequest(**values)


def _music_request(args: argparse.Namespace, prompt: str) -> MusicRequest:
    values: dict[str, Any] = {
        "prompt": prompt,
        "lyrics": _lyrics(args),
        "instrumental": bool(getattr(args, "instrumental", False)),
        "model": getattr(args, "model", None),
        "duration_seconds": getattr(args, "duration", None),
        "tempo_bpm": getattr(args, "tempo", None),
        "style": getattr(args, "style", None),
        "seed": getattr(args, "seed", None),
        "count": max(1, getattr(args, "count", 1)),
    }
    output_format = getattr(args, "output_format", None)
    if output_format:
        values["format"] = output_format
    return MusicRequest(**values)


def _sfx_request(args: argparse.Namespace, prompt: str) -> SFXRequest:
    values: dict[str, Any] = {
        "prompt": prompt,
        "model": getattr(args, "model", None),
        "duration_seconds": getattr(args, "duration", None),
        "loop": bool(getattr(args, "loop", False)),
        "seed": getattr(args, "seed", None),
        "count": max(1, getattr(args, "count", 1)),
    }
    output_format = getattr(args, "output_format", None)
    if output_format:
        values["format"] = output_format
    return SFXRequest(**values)


def _transcription_request(args: argparse.Namespace) -> STTRequest:
    path = _audio_path(args)
    return STTRequest(
        audio=path.read_bytes(),
        filename=path.name,
        model=getattr(args, "model", None),
        language=getattr(args, "language", None),
        prompt=getattr(args, "prompt_option", None)
        or (
            _read_text_file(args.prompt_file)
            if getattr(args, "prompt_file", None)
            else None
        ),
    )


async def _generate(runtime: _Runtime, args: argparse.Namespace, kind: str) -> Any:
    values = _common_service_values(runtime, args, kind)
    if kind == "image":
        image_service = ImageService(runtime.router, **values)
        prompt = _prompt(args)
        return await image_service.generate(
            _image_request(args, prompt),
            token=image_service.screen(prompt),
        )
    if kind == "video":
        video_service = VideoService(runtime.router, **values)
        prompt = _prompt(args)
        return await video_service.generate(
            _video_request(args, prompt),
            token=video_service.screen(prompt),
        )
    if kind == "speech":
        speech_service = SpeechService(runtime.router, **values)
        prompt = _prompt(args)
        return await speech_service.synthesize(
            _speech_request(args, prompt),
            token=speech_service.screen(prompt),
        )
    if kind == "music":
        music_service = MusicService(runtime.router, **values)
        prompt = _prompt(args)
        return await music_service.generate(
            _music_request(args, prompt),
            token=music_service.screen(prompt),
        )
    if kind == "sfx":
        sfx_service = SFXService(runtime.router, **values)
        prompt = _prompt(args)
        return await sfx_service.generate(
            _sfx_request(args, prompt),
            token=sfx_service.screen(prompt),
        )
    stt_service = STTService(runtime.router, **values)
    request = _transcription_request(args)
    token = stt_service.screen(request.prompt) if request.prompt else None
    return await stt_service.generate(request, token=token)


def _asset_paths(outcome: Any) -> list[Path]:
    paths: list[Path] = []
    for asset in getattr(outcome, "assets", ()):
        value = getattr(asset, "path", "")
        if not value:
            continue
        parsed = urlsplit(value)
        if parsed.scheme.lower() != "file":
            continue
        paths.append(Path(unquote(parsed.path)))
    return paths


def _copy_output(paths: Sequence[Path], output: str) -> list[Path]:
    if not paths:
        return []
    target = Path(output).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for index, source in enumerate(paths):
        destination = target
        if len(paths) > 1:
            destination = target.with_name(
                f"{target.stem}-{index}{target.suffix or source.suffix}"
            )
        shutil.copyfile(source, destination)
        written.append(destination)
    return written


def _print_outcome(outcome: Any, args: argparse.Namespace, kind: str) -> int:
    print(f"Provider: {getattr(outcome, 'provider', '') or '-'}")
    print(f"Model: {getattr(outcome, 'model', '') or '-'}")
    print(f"Cost: ${getattr(outcome, 'amount_usd', 0.0):.6f}")
    print(f"Cached: {'yes' if getattr(outcome, 'cached', False) else 'no'}")

    paths = _asset_paths(outcome)
    output = getattr(args, "output", None)
    if output and kind == "transcription":
        target = Path(output).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(getattr(outcome, "text", ""), encoding="utf-8")
        print(f"Output: {target}")
        return 0
    if output:
        written = _copy_output(paths, output)
        for path in written:
            print(f"Output: {path}")
        return 0
    for path in paths:
        print(f"Output: {path}")
    if kind == "transcription":
        text = getattr(outcome, "text", "")
        if text:
            print(text)
    return 0


def _run_generate(args: argparse.Namespace, kind: str) -> int:
    try:
        if getattr(args, "estimate", False):
            return _print_estimate(args, kind)
        runtime = _runtime(args, kind)
        try:
            outcome = asyncio.run(_generate(runtime, args, kind))
        finally:
            runtime.engine.dispose()
        return _print_outcome(outcome, args, kind)
    except (AiviaError, ConfigError, OSError, SQLAlchemyError, ValidationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


register_command(
    "media",
    run_media,
    help="Generate an image, video, voice, transcript, song or sound effect",
    configure=configure,
)
