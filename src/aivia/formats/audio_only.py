"""Audio-only format: a song with cover art and embedded metadata.

The format writes lyrics, generates a song, renders one cover image and turns
the resulting timeline into an audio file. Rendering is deterministic and free:
it mixes the existing audio tracks, embeds the cover as an attached picture and
writes the requested tags, but it never creates or charges for video. The
cover stage is the only paid visual step, and it uses the shared image service
so screening, caching, budgeting and cost recording stay in one place.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from aivia.compose.audio_outputs import (
    AudioMetadata,
    AudioRenderResult,
    render_audio,
)
from aivia.compose.io import composition_to_yaml
from aivia.compose.timeline import Timeline, timeline_to_compositions
from aivia.core.errors import ConfigError
from aivia.core.models.run import StageResult, StageState
from aivia.core.models.script import MediaKind, Script, VisualType
from aivia.core.stages.base import Stage, StageContext
from aivia.visuals.image_stage import ImageBeatResult, resolve_image_request
from aivia.visuals.strategy import VisualRequest

__all__ = [
    "AUDIO_ONLY_STAGE_NAME",
    "COVER_ART_STAGE_NAME",
    "DEFAULT_AUDIO_ONLY_COMPOSITION_FILENAME",
    "DEFAULT_AUDIO_ONLY_FILENAME",
    "DEFAULT_AUDIO_ONLY_MANIFEST_FILENAME",
    "DEFAULT_AUDIO_ONLY_PATTERN",
    "DEFAULT_COVER_ASPECT_RATIO",
    "DEFAULT_COVER_FILENAME",
    "DEFAULT_COVER_MANIFEST_FILENAME",
    "DEFAULT_MUSIC_MANIFEST_FILENAME",
    "DEFAULT_TIMELINE_FILENAME",
    "DEFAULT_VISUAL_MANIFEST_FILENAME",
    "AudioOnlyFormat",
    "AudioOnlyRenderer",
    "AudioOnlyStage",
    "CoverArtFormat",
    "CoverArtRenderer",
    "CoverArtStage",
    "audio_only",
    "audio_only_metadata",
    "audio_only_stage",
    "build_audio_metadata",
    "build_audio_only_stage",
    "build_cover_art_stage",
    "cover_art_prompt",
    "cover_art_source",
    "cover_art_stage",
    "render_audio_only",
]

AUDIO_ONLY_STAGE_NAME = "render"
"""The recipe stage name the audio renderer serves."""

COVER_ART_STAGE_NAME = "cover"
"""The recipe stage name the cover art generator serves."""

DEFAULT_TIMELINE_FILENAME = "timeline.json"
DEFAULT_MUSIC_MANIFEST_FILENAME = "music.json"
DEFAULT_VISUAL_MANIFEST_FILENAME = "visual-reuse.json"
DEFAULT_COVER_FILENAME = "cover.png"
DEFAULT_COVER_MANIFEST_FILENAME = "cover.json"
DEFAULT_COVER_ASPECT_RATIO = "1:1"
DEFAULT_AUDIO_ONLY_COMPOSITION_FILENAME = "audio-only.yaml"
DEFAULT_AUDIO_ONLY_FILENAME = "audio-only.mp3"
DEFAULT_AUDIO_ONLY_PATTERN = "audio-only.{language}.mp3"
DEFAULT_AUDIO_ONLY_MANIFEST_FILENAME = "audio-only.json"


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(
            f"could not read {path.name}",
            details={"path": str(path), "error": str(exc)},
        ) from exc


def _script_from(value: Any) -> Script | None:
    if isinstance(value, Script):
        return value
    if isinstance(value, Mapping):
        try:
            return Script.model_validate(dict(value))
        except ValueError:
            return None
    return None


def _read_script(directory: Path, language: str) -> Script | None:
    names: list[str] = []
    code = _text(language).replace("_", "-").lower()
    if code:
        names.extend((f"script.{code}.json", f"script_{code}.json"))
    names.append("script.json")
    for name in names:
        path = directory / name
        if not path.is_file():
            continue
        payload = _read_json(path)
        script = _script_from(payload)
        if script is not None:
            return script
    for path in sorted(directory.glob("script.*.json")):
        payload = _read_json(path)
        script = _script_from(payload)
        if script is not None:
            return script
    return None


def cover_art_prompt(
    script: Script | Mapping[str, Any] | None = None,
    *,
    prompt: str = "",
    title: str = "",
    style: str = "",
) -> str:
    """Build a deterministic cover prompt from an explicit prompt or a script."""
    explicit = _text(prompt)
    if explicit:
        return explicit
    value = script if isinstance(script, Script) else _script_from(script)
    name = _text(title) or (_text(value.title) if value is not None else "")
    lines: list[str] = []
    if value is not None:
        for beat in value.beats:
            line = _text(beat.text)
            if line and line not in lines:
                lines.append(line)
            if len(lines) >= 4:
                break
    subject = name or "the song"
    parts = [f"Album cover art for {subject}."]
    if lines:
        parts.append(" ".join(lines) + ".")
    chosen_style = _text(style)
    if chosen_style:
        parts.append(chosen_style.rstrip(".") + ".")
    parts.append("Square artwork with no typography.")
    return " ".join(parts)


def _music_values(music: Any) -> Mapping[str, Any]:
    if not isinstance(music, Mapping):
        return {}
    selected = music.get("selected")
    if isinstance(selected, Mapping):
        return selected
    track = music.get("track")
    if isinstance(track, Mapping):
        return track
    return music


def audio_only_metadata(
    script: Script | Mapping[str, Any] | None = None,
    music: Mapping[str, Any] | None = None,
    *,
    title: str = "",
    artist: str = "",
    album: str = "",
    genre: str = "",
    comment: str = "",
    date: str = "",
    track: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> AudioMetadata:
    """Merge explicit tags with the written script and the selected song.

    Explicit *metadata* wins over the derived values, so a recipe or caller can
    always override a title or add a custom tag without losing the defaults.
    """
    if isinstance(script, Mapping):
        script = _script_from(script)
    values = dict(metadata or {})
    selected = _music_values(music)
    resolved_title = (
        _text(title)
        or (_text(script.title) if script is not None else "")
        or _text(selected.get("title"))
    )
    resolved_genre = _text(genre) or _text(selected.get("genre"))
    resolved_comment = _text(comment) or _text(
        (music or {}).get("prompt") if isinstance(music, Mapping) else ""
    )
    derived = {
        "title": resolved_title,
        "artist": _text(artist),
        "album": _text(album) or resolved_title,
        "genre": resolved_genre,
        "comment": resolved_comment,
        "date": _text(date),
        "track": _text(track),
    }
    for key, value in derived.items():
        if value and key not in values:
            values[key] = value
    return AudioMetadata.from_value(values)


build_audio_metadata = audio_only_metadata
audio_only = audio_only_metadata


def cover_art_source(value: Any, *, run_dir: Path | None = None) -> Path | None:
    """Return the cover image path from a manifest, asset or plain path."""
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        text = str(value).strip()
        if not text:
            return None
        path = Path(text)
        if not path.is_absolute() and run_dir is not None:
            path = run_dir / path
        return path
    if not isinstance(value, Mapping):
        return None
    for key in ("cover_art", "cover", "image", "path", "source", "file"):
        found = cover_art_source(value.get(key), run_dir=run_dir)
        if found is not None:
            return found
    asset = value.get("asset")
    found = cover_art_source(asset, run_dir=run_dir)
    if found is not None:
        return found
    beats = value.get("beats")
    if isinstance(beats, list):
        for beat in beats:
            found = cover_art_source(beat, run_dir=run_dir)
            if found is not None:
                return found
    return None


def _fallback_cover(path: Path, title: str) -> None:
    """Write a simple local cover when a generated asset has no local file."""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (512, 512), (24, 24, 32))
    draw = ImageDraw.Draw(image)
    font: Any
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 36)
    except OSError:
        font = ImageFont.load_default()
    text = _text(title) or "AIVIA"
    draw.multiline_text((36, 220), text, fill=(255, 255, 255), font=font)
    image.save(path)


class CoverArtStage(Stage):
    """Generate one square cover image and write its manifest.

    ``scripts`` may carry an already loaded script; otherwise the stage reads
    the written script from the run directory. An injected ``image`` service
    keeps tests offline. The stage uses the shared image resolver so caching,
    budgeting and cost recording are not duplicated here.
    """

    name = COVER_ART_STAGE_NAME
    artifacts: tuple[str, ...] = (
        DEFAULT_COVER_FILENAME,
        DEFAULT_COVER_MANIFEST_FILENAME,
    )

    def __init__(
        self,
        name: str | None = None,
        *,
        options: Mapping[str, Any] | None = None,
        scripts: Mapping[str, Script] | None = None,
        script: Script | Mapping[str, Any] | None = None,
        source_language: str | None = None,
        prompt: str | None = None,
        title: str | None = None,
        style: str | None = None,
        negative_prompt: str | None = None,
        aspect_ratio: str | None = None,
        seed: int | None = None,
        model: str | None = None,
        width: int | None = None,
        height: int | None = None,
        image: Any = None,
        service: Any = None,
        generator: Any = None,
        cover_filename: str | None = None,
        manifest_filename: str | None = None,
        render: bool | None = None,
        **kwargs: Any,
    ) -> None:
        merged = dict(options or {})
        supplied = {
            "scripts": scripts,
            "script": script,
            "source_language": source_language,
            "prompt": prompt,
            "title": title,
            "style": style,
            "negative_prompt": negative_prompt,
            "aspect_ratio": aspect_ratio,
            "seed": seed,
            "model": model,
            "width": width,
            "height": height,
            "image": image,
            "service": service,
            "generator": generator,
            "cover_filename": cover_filename,
            "manifest_filename": manifest_filename,
            "render": render,
        }
        for key, value in supplied.items():
            if value is not None:
                merged[key] = value
        super().__init__(
            name=name or COVER_ART_STAGE_NAME,
            options=merged,
        )
        self.scripts = dict(merged.get("scripts") or {})
        self._script = merged.get("script")
        self.source_language = _text(merged.get("source_language")) or "en"
        self.prompt = _text(merged.get("prompt"))
        self.title = _text(merged.get("title"))
        self.style = _text(merged.get("style"))
        self.negative_prompt = _text(merged.get("negative_prompt"))
        self.aspect_ratio = (
            _text(merged.get("aspect_ratio")) or DEFAULT_COVER_ASPECT_RATIO
        )
        self.seed = merged.get("seed")
        self.model = _text(merged.get("model"))
        self.width = merged.get("width")
        self.height = merged.get("height")
        self._service = (
            merged.get("image") or merged.get("service") or merged.get("generator")
        )
        self.cover_filename = (
            _text(merged.get("cover_filename")) or DEFAULT_COVER_FILENAME
        )
        self.manifest_filename = (
            _text(merged.get("manifest_filename")) or DEFAULT_COVER_MANIFEST_FILENAME
        )
        self.render = bool(merged.get("render", True))
        self.artifacts = (
            (self.cover_filename, self.manifest_filename)
            if self.render
            else (self.manifest_filename,)
        )

    def _script_value(self, context: StageContext) -> Script | None:
        value = self._script
        if value is None:
            value = self.scripts.get(self.source_language)
        if value is not None:
            return value if isinstance(value, Script) else _script_from(value)
        return _read_script(context.run_dir, self.source_language)

    async def run(self, context: StageContext) -> StageResult:
        script = self._script_value(context)
        prompt = cover_art_prompt(
            script,
            prompt=self.prompt,
            title=self.title,
            style=self.style,
        )
        request = VisualRequest(
            visual_type=VisualType.IMAGE,
            kind=MediaKind.IMAGE,
            prompt=prompt,
            text=prompt,
            aspect_ratio=self.aspect_ratio,
            negative_prompt=self.negative_prompt,
            seed=self.seed if isinstance(self.seed, int) else None,
        )
        run_id = context.run.id if context.run is not None else ""
        result: ImageBeatResult = await resolve_image_request(
            request,
            service=self._service,
            context=context,
            config=context.config,
            model=self.model or None,
            aspect_ratio=self.aspect_ratio,
            negative_prompt=self.negative_prompt or None,
            seed=self.seed if isinstance(self.seed, int) else None,
            width=self.width,
            height=self.height,
            run_id=run_id,
            metadata={"stage": self.name, "cover": True},
        )
        if not result.resolved or result.asset is None:
            raise ConfigError(
                "cover art stage could not generate a cover",
                details={"stage": self.name, "error": result.error},
            )
        asset = result.asset
        source = cover_art_source(asset.model_dump(mode="json"))
        artifacts: list[str] = []
        cover_path: Path | None = None
        if self.render:
            destination = context.run_dir / self.cover_filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source is not None and source.is_file():
                shutil.copyfile(source, destination)
            else:
                _fallback_cover(destination, script.title if script else self.title)
            cover_path = destination
            artifacts.append(self.cover_filename)
        manifest = {
            "version": 1,
            "stage": self.name,
            "prompt": prompt,
            "aspect_ratio": self.aspect_ratio,
            "cover": (
                cover_path.relative_to(context.run_dir).as_posix()
                if cover_path is not None
                else ""
            ),
            "source": str(source) if source is not None else "",
            "asset": asset.model_dump(mode="json"),
            "provider": result.provider,
            "model": result.model,
            "cost_usd": result.cost_usd,
            "cached": result.cached,
            "title": _text(script.title) if script is not None else self.title,
        }
        manifest_path = context.run_dir / self.manifest_filename
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifacts.append(self.manifest_filename)
        return StageResult(
            run_id=run_id,
            stage=self.name,
            status=StageState.SUCCEEDED,
            artifacts=artifacts,
            cost_usd=result.cost_usd,
            metadata={
                "cover": manifest["cover"],
                "source": manifest["source"],
                "provider": result.provider,
                "model": result.model,
                "cost_usd": result.cost_usd,
                "cached": result.cached,
            },
        )


class AudioOnlyStage(Stage):
    """Render a timeline to audio with cover art and metadata, never video.

    ``options`` may carry ``timeline``, ``timeline_filename``, ``scripts``,
    ``source_language``, ``cover_art``, ``cover_manifest``, ``metadata``,
    ``title``, ``artist``, ``album``, ``genre``, ``audio_filename``,
    ``output_format`` and the usual encoder settings. It reads the timeline,
    cover manifest and music manifest from the run directory when they are not
    injected. The stage is deterministic apart from the final ffmpeg render.
    """

    name = AUDIO_ONLY_STAGE_NAME
    artifacts: tuple[str, ...] = (
        DEFAULT_AUDIO_ONLY_COMPOSITION_FILENAME,
        DEFAULT_AUDIO_ONLY_FILENAME,
        DEFAULT_AUDIO_ONLY_MANIFEST_FILENAME,
    )

    def __init__(
        self,
        name: str | None = None,
        *,
        options: Mapping[str, Any] | None = None,
        timeline: Any = None,
        scripts: Mapping[str, Script] | None = None,
        script: Script | Mapping[str, Any] | None = None,
        source_language: str | None = None,
        cover_art: Any = None,
        cover: Any = None,
        cover_manifest: str | None = None,
        visual_manifest: str | None = None,
        music_manifest: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        title: str | None = None,
        artist: str | None = None,
        album: str | None = None,
        genre: str | None = None,
        comment: str | None = None,
        date: str | None = None,
        track: str | None = None,
        render: bool | None = None,
        embed_cover_art: bool | None = None,
        timeline_filename: str | None = None,
        composition_filename: str | None = None,
        audio_filename: str | None = None,
        manifest_filename: str | None = None,
        output_format: str | None = None,
        audio_codec: str | None = None,
        audio_bitrate: int | None = None,
        sample_rate: int | None = None,
        channels: int | None = None,
        video: Any = None,
        renderer: Any = None,
        audio_renderer: Any = None,
        ffmpeg_runner: Any = None,
        ffprobe_runner: Any = None,
        **kwargs: Any,
    ) -> None:
        merged = dict(options or {})
        supplied = {
            "timeline": timeline,
            "scripts": scripts,
            "script": script,
            "source_language": source_language,
            "cover_art": cover_art if cover_art is not None else cover,
            "cover_manifest": cover_manifest,
            "visual_manifest": visual_manifest,
            "music_manifest": music_manifest,
            "metadata": metadata,
            "title": title,
            "artist": artist,
            "album": album,
            "genre": genre,
            "comment": comment,
            "date": date,
            "track": track,
            "render": render,
            "embed_cover_art": embed_cover_art,
            "timeline_filename": timeline_filename,
            "composition_filename": composition_filename,
            "audio_filename": audio_filename,
            "manifest_filename": manifest_filename,
            "output_format": output_format,
            "audio_codec": audio_codec,
            "audio_bitrate": audio_bitrate,
            "sample_rate": sample_rate,
            "channels": channels,
            "video": video,
            "renderer": renderer if renderer is not None else audio_renderer,
            "ffmpeg_runner": ffmpeg_runner,
            "ffprobe_runner": ffprobe_runner,
        }
        for key, value in supplied.items():
            if value is not None:
                merged[key] = value
        super().__init__(
            name=name or AUDIO_ONLY_STAGE_NAME,
            options=merged,
        )
        self._timeline = merged.get("timeline")
        self.scripts = dict(merged.get("scripts") or {})
        self._script = merged.get("script")
        self.source_language = _text(merged.get("source_language"))
        self._cover = merged.get("cover_art")
        self.cover_manifest = (
            _text(merged.get("cover_manifest")) or DEFAULT_COVER_MANIFEST_FILENAME
        )
        self.visual_manifest = (
            _text(merged.get("visual_manifest")) or DEFAULT_VISUAL_MANIFEST_FILENAME
        )
        self.music_manifest = (
            _text(merged.get("music_manifest")) or DEFAULT_MUSIC_MANIFEST_FILENAME
        )
        self.metadata = dict(merged.get("metadata") or {})
        self.title = _text(merged.get("title"))
        self.artist = _text(merged.get("artist"))
        self.album = _text(merged.get("album"))
        self.genre = _text(merged.get("genre"))
        self.comment = _text(merged.get("comment"))
        self.date = _text(merged.get("date"))
        self.track = _text(merged.get("track"))
        self.render = bool(merged.get("render", True))
        self.embed_cover_art = bool(merged.get("embed_cover_art", True))
        self.timeline_filename = (
            _text(merged.get("timeline_filename")) or DEFAULT_TIMELINE_FILENAME
        )
        self.composition_filename = (
            _text(merged.get("composition_filename"))
            or DEFAULT_AUDIO_ONLY_COMPOSITION_FILENAME
        )
        self.audio_filename = (
            _text(merged.get("audio_filename")) or DEFAULT_AUDIO_ONLY_FILENAME
        )
        self.manifest_filename = (
            _text(merged.get("manifest_filename"))
            or DEFAULT_AUDIO_ONLY_MANIFEST_FILENAME
        )
        self.output_format = _text(merged.get("output_format")) or None
        self.audio_codec = _text(merged.get("audio_codec")) or None
        self.audio_bitrate = merged.get("audio_bitrate")
        self.sample_rate = merged.get("sample_rate")
        self.channels = merged.get("channels")
        self.video = merged.get("video")
        self._renderer = merged.get("renderer")
        self.ffmpeg_runner = merged.get("ffmpeg_runner")
        self.ffprobe_runner = merged.get("ffprobe_runner")
        if not self.render:
            self.artifacts = (self.composition_filename,)

    def _load_timeline(self, context: StageContext) -> Timeline:
        value = self._timeline
        if value is None:
            path = context.run_dir / self.timeline_filename
            if not path.is_file():
                raise ConfigError(
                    "audio-only stage needs a timeline",
                    details={"timeline": self.timeline_filename},
                )
            return Timeline.model_validate(_read_json(path))
        if isinstance(value, Timeline):
            return value.model_copy(deep=True)
        if isinstance(value, Mapping):
            return Timeline.model_validate(dict(value))
        if isinstance(value, (str, Path)):
            path = Path(value)
            if not path.is_absolute():
                path = context.run_dir / path
            if not path.is_file():
                raise ConfigError(
                    "audio-only stage needs a timeline",
                    details={"timeline": str(path)},
                )
            return Timeline.model_validate(_read_json(path))
        raise ConfigError(
            "timeline must be a Timeline, mapping or file path",
            details={"value": type(value).__name__},
        )

    def _load_json_if_present(self, context: StageContext, name: str) -> Any:
        path = context.run_dir / name
        if not path.is_file():
            return None
        return _read_json(path)

    def _music(self, context: StageContext) -> Mapping[str, Any]:
        value = self._load_json_if_present(context, self.music_manifest)
        return value if isinstance(value, Mapping) else {}

    def _cover_path(self, context: StageContext) -> Path | None:
        value = self._cover
        if value is None:
            value = self._load_json_if_present(context, self.cover_manifest)
        if value is None:
            value = self._load_json_if_present(context, self.visual_manifest)
        return cover_art_source(value, run_dir=context.run_dir)

    def _script_value(self, context: StageContext, language: str) -> Script | None:
        value = self._script
        if value is not None:
            return value if isinstance(value, Script) else _script_from(value)
        selected = self.scripts.get(language)
        if isinstance(selected, Script):
            return selected
        selected = self.scripts.get(self.source_language)
        if isinstance(selected, Script):
            return selected
        return _read_script(context.run_dir, language or self.source_language)

    def _video_config(self, context: StageContext) -> Any:
        if self.video is not None:
            return self.video
        config = context.config
        if isinstance(config, Mapping):
            return config.get("video")
        return getattr(config, "video", None)

    def _output_name(self, language: str, *, multiple: bool) -> str:
        if not multiple:
            return self.audio_filename
        suffix = Path(self.audio_filename).suffix or ".mp3"
        return DEFAULT_AUDIO_ONLY_PATTERN.format(language=language).replace(
            ".mp3", suffix
        )

    def _render_one(
        self,
        composition: Any,
        output: Path,
        metadata: AudioMetadata,
        cover: Path | None,
    ) -> AudioRenderResult:
        options: dict[str, Any] = {
            "metadata": metadata,
            "cover_art": cover if self.embed_cover_art else None,
            "output_format": self.output_format,
            "audio_codec": self.audio_codec,
            "audio_bitrate": self.audio_bitrate,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
        }
        options = {key: value for key, value in options.items() if value is not None}
        renderer = self._renderer or render_audio
        result = renderer(
            composition,
            output,
            ffmpeg_runner=self.ffmpeg_runner,
            ffprobe_runner=self.ffprobe_runner,
            **options,
        )
        if not isinstance(result, AudioRenderResult):
            raise ConfigError(
                "audio renderer returned an unexpected result",
                details={"value": type(result).__name__},
            )
        return result

    async def run(self, context: StageContext) -> StageResult:
        timeline = self._load_timeline(context)
        compositions = timeline_to_compositions(
            timeline,
            video=self._video_config(context),
        )
        if not compositions:
            raise ConfigError(
                "audio-only stage has no composition to render",
                details={"timeline": self.timeline_filename},
            )
        compositions = [
            composition.model_copy(update={"layers": [], "subtitles": None})
            for composition in compositions
        ]
        primary = compositions[0]
        composition_path = context.run_dir / self.composition_filename
        composition_path.parent.mkdir(parents=True, exist_ok=True)
        composition_path.write_text(
            composition_to_yaml(primary),
            encoding="utf-8",
        )
        artifacts = [self.composition_filename]
        music = self._music(context)
        cover = self._cover_path(context) if self.embed_cover_art else None
        rendered: list[dict[str, Any]] = []
        if self.render:
            multiple = len(compositions) > 1
            for composition in compositions:
                script = self._script_value(context, composition.language)
                metadata = audio_only_metadata(
                    script,
                    music,
                    title=self.title,
                    artist=self.artist,
                    album=self.album,
                    genre=self.genre,
                    comment=self.comment,
                    date=self.date,
                    track=self.track,
                    metadata=self.metadata,
                )
                name = self._output_name(composition.language, multiple=multiple)
                output = context.run_dir / name
                output.parent.mkdir(parents=True, exist_ok=True)
                result = self._render_one(composition, output, metadata, cover)
                artifacts.append(name)
                rendered.append(
                    {
                        "language": composition.language,
                        "platform": composition.platform,
                        "path": name,
                        "duration_seconds": result.duration_seconds,
                        "format": result.format,
                        "codec": result.codec,
                        "metadata": metadata.as_tags(),
                        "cover_art": (
                            str(result.cover_art) if result.cover_art else ""
                        ),
                        "has_cover_art": bool(result.probe.video_streams),
                        "has_video": bool(result.probe.video_streams)
                        and result.cover_art is None,
                        "video_cost_usd": 0.0,
                    }
                )
        manifest = {
            "version": 1,
            "stage": self.name,
            "output_type": "audio",
            "timeline": self.timeline_filename,
            "composition": self.composition_filename,
            "cover_art": str(cover) if cover is not None else "",
            "music": self.music_manifest if music else "",
            "outputs": rendered,
            "cost_usd": 0.0,
            "video_cost_usd": 0.0,
        }
        manifest_path = context.run_dir / self.manifest_filename
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifacts.append(self.manifest_filename)
        run_id = context.run.id if context.run is not None else ""
        return StageResult(
            run_id=run_id,
            stage=self.name,
            status=StageState.SUCCEEDED,
            artifacts=artifacts,
            cost_usd=0.0,
            metadata={
                "outputs": len(rendered),
                "languages": [item.language for item in compositions],
                "timeline": self.timeline_filename,
                "cover_art": str(cover) if cover is not None else "",
                "cost_usd": 0.0,
                "video_cost_usd": 0.0,
                "has_video": any(item["has_video"] for item in rendered),
            },
        )


AudioOnlyFormat = AudioOnlyStage
AudioOnlyRenderer = AudioOnlyStage
audio_only_stage = AudioOnlyStage
CoverArtFormat = CoverArtStage
CoverArtRenderer = CoverArtStage
cover_art_stage = CoverArtStage


def build_audio_only_stage(
    *,
    options: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> AudioOnlyStage:
    """Build an :class:`AudioOnlyStage` from recipe options and overrides."""
    return AudioOnlyStage(options=options, **kwargs)


def build_cover_art_stage(
    *,
    options: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> CoverArtStage:
    """Build a :class:`CoverArtStage` from recipe options and overrides."""
    return CoverArtStage(options=options, **kwargs)


def render_audio_only(
    composition: Any,
    output: str | Path,
    **options: Any,
) -> AudioRenderResult:
    """Alias for the deterministic audio renderer used by this format."""
    return render_audio(composition, output, **options)
