"""``aivia compose``: validate, inspect and render composition documents.

Rendering is deterministic and performs no AI call or network request. The
output container is selected from ``--format`` or the destination suffix, and
``--preview`` renders a smaller, faster version through the shared preview
pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from aivia.cli.registry import register_command
from aivia.compose import (
    Composition,
    CompositionIOError,
    dump_composition_schema,
    load_composition,
    preview_settings,
    render_preview,
    render_still,
    render_video,
    render_web_card,
    validate_composition,
)
from aivia.core.errors import AiviaError

__all__ = [
    "configure",
    "render_audio",
    "render_gif",
    "render_webp",
    "run_compose",
    "run_render",
    "run_schema",
    "run_validate",
]

_VIDEO_SUFFIXES = {".mp4", ".m4v"}
_AUDIO_SUFFIXES = {".mp3", ".m4a", ".wav", ".flac", ".ogg", ".oga", ".opus", ".aac"}
_STILL_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
_WEB_CARD_SUFFIXES = {".html", ".htm"}


def render_gif(composition: Any, output: str | Path, **options: Any) -> Any:
    """Render an animated GIF through the shared video pipeline."""
    return render_video(composition, output, **options)


def render_webp(composition: Any, output: str | Path, **options: Any) -> Any:
    """Render an animated WebP through the shared video pipeline."""
    return render_video(composition, output, **options)


def render_audio(composition: Any, output: str | Path, **options: Any) -> Any:
    """Render an audio-only output through the shared video pipeline."""
    return render_video(composition, output, **options)


_FORMAT_ALIASES: dict[str, tuple[str, str]] = {
    "auto": ("auto", ""),
    "video": ("video", ".mp4"),
    "mp4": ("video", ".mp4"),
    "m4v": ("video", ".m4v"),
    "gif": ("gif", ".gif"),
    "webp": ("webp", ".webp"),
    "animated-webp": ("webp", ".webp"),
    "anim-webp": ("webp", ".webp"),
    "audio": ("audio", ".mp3"),
    "mp3": ("audio", ".mp3"),
    "m4a": ("audio", ".m4a"),
    "wav": ("audio", ".wav"),
    "flac": ("audio", ".flac"),
    "ogg": ("audio", ".ogg"),
    "oga": ("audio", ".oga"),
    "opus": ("audio", ".opus"),
    "aac": ("audio", ".aac"),
    "still": ("still", ".png"),
    "image": ("still", ".png"),
    "png": ("still", ".png"),
    "jpg": ("still", ".jpg"),
    "jpeg": ("still", ".jpeg"),
    "web-card": ("web-card", ".html"),
    "web_card": ("web-card", ".html"),
    "webcard": ("web-card", ".html"),
    "html": ("web-card", ".html"),
}


def _configure_validate(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "composition", metavar="PATH", help="composition file to validate"
    )
    parser.add_argument(
        "--assets",
        metavar="DIR",
        help=(
            "resolve relative asset paths against DIR and report missing files "
            "(default: the composition's directory)"
        ),
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="print every issue as JSON instead of text",
    )
    parser.set_defaults(_compose_handler=run_validate)


def _configure_schema(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output",
        "-o",
        metavar="PATH",
        help="write the schema to PATH instead of standard output",
    )
    parser.set_defaults(_compose_handler=run_schema)


def _configure_render(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "composition", metavar="PATH", help="composition file to render"
    )
    parser.add_argument(
        "destination",
        nargs="?",
        metavar="OUTPUT",
        help="output path (the same as --output)",
    )
    parser.add_argument(
        "--output",
        "-o",
        metavar="PATH",
        help="output path; its suffix selects the container unless --format is set",
    )
    parser.add_argument(
        "--format",
        "--output-format",
        "--type",
        dest="output_format",
        metavar="NAME",
        help="output selection: video, gif, webp, audio, still or web-card",
    )
    parser.add_argument(
        "--preview",
        "-p",
        action="store_true",
        help="render a reduced resolution and frame rate preview",
    )
    parser.add_argument(
        "--scale",
        type=float,
        metavar="FACTOR",
        help="preview scale between zero and one",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        metavar="PX",
        help="preview width cap",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        metavar="PX",
        help="preview height cap",
    )
    parser.add_argument(
        "--max-fps",
        type=int,
        metavar="FPS",
        help="preview frame rate cap",
    )
    parser.add_argument(
        "--time",
        type=float,
        metavar="SECONDS",
        help="still frame timestamp (default: 0)",
    )
    parser.add_argument(
        "--platform",
        metavar="NAME",
        help="apply a configured platform output profile",
    )
    parser.add_argument(
        "--profile",
        metavar="NAME",
        help="apply a configured output profile",
    )
    parser.set_defaults(_compose_handler=run_render)


def configure(parser: argparse.ArgumentParser) -> None:
    """Add the compose group and its subcommands."""
    subparsers = parser.add_subparsers(dest="compose_command", metavar="COMMAND")
    validate = subparsers.add_parser(
        "validate",
        help="Validate a composition and print every problem",
        description="Validate a composition and print every problem",
    )
    _configure_validate(validate)
    schema = subparsers.add_parser(
        "schema",
        help="Export the composition JSON Schema",
        description="Export the composition JSON Schema",
    )
    _configure_schema(schema)
    render = subparsers.add_parser(
        "render",
        help="Render a composition to the selected output",
        description="Render a composition to the selected output",
    )
    _configure_render(render)


def run_compose(args: argparse.Namespace) -> int:
    """Dispatch to the selected compose subcommand."""
    handler = getattr(args, "_compose_handler", None)
    if handler is None:
        print("error: missing compose command", file=sys.stderr)
        print("usage: aivia compose {validate, schema, render}", file=sys.stderr)
        return 2
    return int(handler(args))


def _read_document(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"composition file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise CompositionIOError("composition document is empty")
    try:
        if path.suffix.lower() == ".json" or text.lstrip().startswith(("{", "[")):
            payload = json.loads(text)
        else:
            payload = yaml.safe_load(text)
    except json.JSONDecodeError as exc:
        raise CompositionIOError(f"invalid JSON composition: {exc}") from exc
    except yaml.YAMLError as exc:
        raise CompositionIOError(f"invalid YAML composition: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise CompositionIOError("composition document must be a mapping")
    return payload


def _print_issue(path: str, message: str) -> None:
    if path:
        print(f"error: {path}: {message}", file=sys.stderr)
    else:
        print(f"error: {message}", file=sys.stderr)


def _asset_root(source: Path, assets: str | None = None) -> Path:
    """Return the directory that relative asset references resolve against."""
    return Path(assets).expanduser() if assets is not None else source.parent


def _resolve_asset_path(value: str, root: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return value
    return str((root / path).resolve())


def _resolve_composition_assets(composition: Composition, root: Path) -> Composition:
    """Resolve every relative asset reference in *composition* against *root*."""
    for layer in composition.layers:
        target: Any = layer
        source = getattr(target, "source", None)
        if isinstance(source, str) and source.strip():
            target.source = _resolve_asset_path(source, root)
        mask = getattr(target, "mask", None)
        if mask is not None and isinstance(mask.source, str) and mask.source.strip():
            mask.source = _resolve_asset_path(mask.source, root)
    for track in composition.audio_tracks:
        if isinstance(track.source, str) and track.source.strip():
            track.source = _resolve_asset_path(track.source, root)
    if composition.subtitle_file.strip():
        composition.subtitle_file = _resolve_asset_path(composition.subtitle_file, root)
    return composition


def run_validate(args: argparse.Namespace) -> int:
    """Validate a composition and report every schema and cross-object issue."""
    source = Path(args.composition).expanduser()
    try:
        document = _read_document(source)
        issues = validate_composition(
            document,
            assets=_asset_root(source, args.assets),
        )
    except (AiviaError, OSError, ValueError) as exc:
        if getattr(args, "json_output", False):
            print(
                json.dumps(
                    [{"path": "", "message": str(exc), "code": "io"}],
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1

    if getattr(args, "json_output", False):
        print(
            json.dumps(
                [issue.to_dict() for issue in issues],
                indent=2,
                sort_keys=True,
            )
        )
    else:
        for issue in issues:
            _print_issue(issue.path, issue.message)
    if issues:
        return 1
    if not getattr(args, "json_output", False):
        print("composition is valid")
    return 0


def run_schema(args: argparse.Namespace) -> int:
    """Print or write the deterministic composition JSON Schema."""
    text = dump_composition_schema()
    if args.output is None:
        print(text, end="")
        return 0
    target = Path(args.output).expanduser()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    except OSError as exc:
        print(f"error: cannot write {target}: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {target}")
    return 0


def _normalise_format(value: str) -> tuple[str, str]:
    key = value.strip().lower().lstrip(".")
    try:
        return _FORMAT_ALIASES[key]
    except KeyError:
        allowed = ", ".join(sorted(_FORMAT_ALIASES))
        raise ValueError(
            f"unsupported output format {value!r}; expected one of {allowed}"
        ) from None


def _kind_for_suffix(suffix: str) -> str | None:
    suffix = suffix.lower()
    if suffix in _VIDEO_SUFFIXES:
        return "video"
    if suffix == ".gif":
        return "gif"
    if suffix == ".webp":
        return "webp"
    if suffix in _AUDIO_SUFFIXES:
        return "audio"
    if suffix in _STILL_SUFFIXES:
        return "still"
    if suffix in _WEB_CARD_SUFFIXES:
        return "web-card"
    return None


def _allowed_suffixes(kind: str) -> set[str]:
    if kind == "video":
        return _VIDEO_SUFFIXES
    if kind == "gif":
        return {".gif"}
    if kind == "webp":
        return {".webp"}
    if kind == "audio":
        return _AUDIO_SUFFIXES
    if kind == "still":
        return _STILL_SUFFIXES
    if kind == "web-card":
        return _WEB_CARD_SUFFIXES
    return set()


def _resolve_output(
    source: Path,
    destination: str | None,
    output_format: str | None,
) -> tuple[Path, str]:
    requested = Path(destination).expanduser() if destination else None
    explicit = (
        _normalise_format(output_format)
        if output_format is not None and output_format.strip().lower() != "auto"
        else None
    )

    if explicit is not None:
        selected_kind, default_suffix = explicit
        if requested is None:
            name = source.stem or "composition"
            return Path(f"{name}{default_suffix}"), selected_kind
        if not requested.suffix:
            return requested.with_suffix(default_suffix), selected_kind
        if requested.suffix.lower() in _allowed_suffixes(selected_kind):
            return requested, selected_kind
        raise ValueError(
            f"output suffix {requested.suffix!r} does not match "
            f"format {output_format!r}"
        )

    if requested is None:
        return Path(f"{source.stem or 'composition'}.mp4"), "video"
    if not requested.suffix:
        return requested.with_suffix(".mp4"), "video"
    kind = _kind_for_suffix(requested.suffix)
    if kind is None:
        raise ValueError(
            f"unsupported output extension {requested.suffix!r}; "
            "use --format to select the output"
        )
    return requested, kind


def _profile_options(args: argparse.Namespace) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if getattr(args, "platform", None) is not None:
        options["platform"] = args.platform
    if getattr(args, "profile", None) is not None:
        options["profile"] = args.profile
    return options


def _preview_options(args: argparse.Namespace) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if getattr(args, "scale", None) is not None:
        options["scale"] = args.scale
    if getattr(args, "max_width", None) is not None:
        options["max_width"] = args.max_width
    if getattr(args, "max_height", None) is not None:
        options["max_height"] = args.max_height
    if getattr(args, "max_fps", None) is not None:
        options["max_fps"] = args.max_fps
    return options


def _save_still(
    composition: Any,
    output: Path,
    *,
    time_seconds: float,
    preview: bool,
    preview_options: Mapping[str, Any],
) -> Path:
    image = render_still(composition, time_seconds)
    if preview:
        settings = preview_settings(composition, **dict(preview_options))
        image = image.resize(
            (settings.width, settings.height),
            Image.Resampling.LANCZOS,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return output


def run_render(args: argparse.Namespace) -> int:
    """Render a composition to the selected output container."""
    source = Path(args.composition).expanduser()
    destination = args.output if args.output is not None else args.destination
    try:
        composition = load_composition(source)
        _resolve_composition_assets(composition, source.parent)
        issues = validate_composition(composition)
        if issues:
            for issue in issues:
                _print_issue(issue.path, issue.message)
            return 1
        output, kind = _resolve_output(source, destination, args.output_format)
        preview = bool(getattr(args, "preview", False))
        profile_options = _profile_options(args)
        preview_options = _preview_options(args)

        if kind == "web-card":
            if preview:
                raise ValueError("a preview renders to a video or a still image")
            render_web_card(composition, output)
        elif kind == "still":
            _save_still(
                composition,
                output,
                time_seconds=getattr(args, "time", None) or 0.0,
                preview=preview,
                preview_options=preview_options,
            )
        elif kind == "video" and preview:
            render_preview(composition, output, **preview_options, **profile_options)
        elif preview:
            raise ValueError("a preview renders to a video or a still image")
        elif kind == "gif":
            render_gif(composition, output, **profile_options)
        elif kind == "webp":
            render_webp(composition, output, **profile_options)
        elif kind == "audio":
            render_audio(composition, output, **profile_options)
        else:
            render_video(composition, output, **profile_options)
    except (AiviaError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Output: {output}")
    return 0


register_command(
    "compose",
    run_compose,
    help="Validate, export the schema of and render compositions",
    configure=configure,
)
