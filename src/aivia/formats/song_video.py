"""Song video format: lyrics, music, karaoke subtitles and visual backgrounds.

The song video recipe writes lyrics, generates a song, aligns the song to the
written lines, generates a background for every line and renders the result.
This module owns the deterministic glue between those stages: it attaches the
shared visual manifest to the aligned timeline beats and renders the resulting
composition. The renderer already knows how to mix the song track and burn the
karaoke subtitle track, so this stage never calls a provider.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from aivia.compose.io import composition_to_yaml
from aivia.compose.timeline import Timeline, timeline_to_compositions
from aivia.compose.video import render_video
from aivia.core.errors import ConfigError
from aivia.core.models.run import StageResult, StageState
from aivia.core.stages.base import Stage, StageContext

__all__ = [
    "DEFAULT_SONG_VIDEO_COMPOSITION_FILENAME",
    "DEFAULT_SONG_VIDEO_FILENAME",
    "DEFAULT_SONG_VIDEO_PATTERN",
    "DEFAULT_TIMELINE_FILENAME",
    "DEFAULT_VISUAL_MANIFEST_FILENAME",
    "SONG_VIDEO_STAGE_NAME",
    "SongVideoFormat",
    "SongVideoRenderer",
    "SongVideoStage",
    "attach_visual_assets",
    "build_song_video_stage",
    "song_video",
    "song_video_stage",
    "song_video_timeline",
]

SONG_VIDEO_STAGE_NAME = "render"
"""The recipe stage name this implementation serves."""

DEFAULT_TIMELINE_FILENAME = "timeline.json"
DEFAULT_VISUAL_MANIFEST_FILENAME = "visual-reuse.json"
DEFAULT_SONG_VIDEO_COMPOSITION_FILENAME = "song-video.yaml"
DEFAULT_SONG_VIDEO_FILENAME = "song-video.mp4"
DEFAULT_SONG_VIDEO_PATTERN = "song-video.{language}.mp4"


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


def _visual_asset(entry: Any) -> Mapping[str, Any] | None:
    if not isinstance(entry, Mapping):
        return None
    asset = entry.get("asset")
    if not isinstance(asset, Mapping):
        result = entry.get("result")
        if isinstance(result, Mapping):
            asset = result.get("asset")
            if not isinstance(asset, Mapping) and _text(result.get("path")):
                asset = result
    if not isinstance(asset, Mapping):
        return None
    return asset


def attach_visual_assets(
    timeline: Timeline | Mapping[str, Any],
    visuals: Mapping[str, Any] | Sequence[Any] | None,
) -> Timeline:
    """Attach a shared visual manifest to a timeline's beats.

    Visual entries are matched to beats by position, which is how the writing,
    align and visual stages order lyric lines. Existing assets not owned by
    this helper are kept; previously attached ``visual_*`` assets are replaced
    so a re-run stays idempotent.
    """
    value = (
        timeline
        if isinstance(timeline, Timeline)
        else Timeline.model_validate(dict(timeline))
    )
    if visuals is None:
        return value
    entries: Sequence[Any]
    if isinstance(visuals, Mapping):
        raw = visuals.get("beats", visuals.get("results", visuals.get("visuals", ())))
        entries = raw if isinstance(raw, Sequence) else ()
    else:
        entries = visuals

    payload = value.model_dump(mode="python")
    beats: list[dict[str, Any]] = list(payload.get("beats", []))
    assets: list[dict[str, Any]] = [
        asset
        for asset in payload.get("assets", [])
        if not _text(asset.get("id")).startswith("visual_")
    ]
    for position, entry in enumerate(entries):
        if position >= len(beats):
            break
        asset = _visual_asset(entry)
        if asset is None:
            continue
        source = _text(asset.get("path") or asset.get("source"))
        if not source:
            continue
        kind = _text(asset.get("kind")) or "image"
        asset_id = f"visual_{position}"
        beat = beats[position]
        beat["asset"] = asset_id
        beat["asset_kind"] = kind
        assets.append(
            {
                "id": asset_id,
                "source": source,
                "kind": kind,
                "beat": beat.get("id", ""),
            }
        )
    payload["beats"] = beats
    payload["assets"] = assets
    return Timeline.model_validate(payload)


def song_video_timeline(
    timeline: Timeline | Mapping[str, Any],
    visuals: Mapping[str, Any] | Sequence[Any] | None = None,
) -> Timeline:
    """Return a timeline with visual backgrounds attached to its beats."""
    return attach_visual_assets(timeline, visuals)


class SongVideoStage(Stage):
    """Render an aligned song timeline with karaoke subtitles and backgrounds.

    ``options`` may carry ``timeline``, ``timeline_filename``, ``visuals``,
    ``visual_manifest``, ``source_language``, ``render``, ``highlight``,
    ``video``, ``composition_filename`` and ``video_filename``. The stage reads the
    timeline and visual manifest from the run directory when they are not
    injected. It is deterministic apart from the final ffmpeg render.
    """

    name = SONG_VIDEO_STAGE_NAME
    artifacts: tuple[str, ...] = (
        DEFAULT_SONG_VIDEO_COMPOSITION_FILENAME,
        DEFAULT_SONG_VIDEO_FILENAME,
    )

    def __init__(
        self,
        name: str | None = None,
        *,
        options: Mapping[str, Any] | None = None,
        timeline: Any = None,
        visuals: Any = None,
        visual_manifest: str | None = None,
        source_language: str | None = None,
        render: bool | None = None,
        highlight: bool | str | None = None,
        video: Any = None,
        timeline_filename: str | None = None,
        composition_filename: str | None = None,
        video_filename: str | None = None,
        **kwargs: Any,
    ) -> None:
        merged = dict(options or {})
        supplied = {
            "timeline": timeline,
            "visuals": visuals,
            "visual_manifest": visual_manifest,
            "source_language": source_language,
            "render": render,
            "highlight": highlight,
            "video": video,
            "timeline_filename": timeline_filename,
            "composition_filename": composition_filename,
            "video_filename": video_filename,
        }
        for key, value in supplied.items():
            if value is not None:
                merged[key] = value
        timeline_name = str(
            merged.get("timeline_filename") or DEFAULT_TIMELINE_FILENAME
        )
        composition_name = str(
            merged.get("composition_filename")
            or DEFAULT_SONG_VIDEO_COMPOSITION_FILENAME
        )
        video_name = str(merged.get("video_filename") or DEFAULT_SONG_VIDEO_FILENAME)
        super().__init__(
            name=name or SONG_VIDEO_STAGE_NAME,
            options=merged,
            artifacts=(composition_name, video_name),
        )
        self._timeline = merged.get("timeline")
        self._visuals = merged.get("visuals")
        self.visual_manifest = str(
            merged.get("visual_manifest") or DEFAULT_VISUAL_MANIFEST_FILENAME
        )
        self.source_language = _text(merged.get("source_language"))
        self.render = bool(merged.get("render", True))
        self.highlight = merged.get("highlight", "karaoke")
        self.video = merged.get("video")
        self.timeline_filename = timeline_name
        self.composition_filename = composition_name
        self.video_filename = video_name
        if not self.render:
            self.artifacts = (self.composition_filename,)

    def _load_timeline(self, context: StageContext) -> Timeline:
        value = self._timeline
        if value is None:
            path = context.run_dir / self.timeline_filename
            if not path.is_file():
                raise ConfigError(
                    "song video stage needs a timeline",
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
                    "song video stage needs a timeline",
                    details={"timeline": str(path)},
                )
            return Timeline.model_validate(_read_json(path))
        raise ConfigError(
            "timeline must be a Timeline, mapping or file path",
            details={"value": type(value).__name__},
        )

    def _load_visuals(self, context: StageContext) -> Any:
        if self._visuals is not None:
            return self._visuals
        path = context.run_dir / self.visual_manifest
        if not path.is_file():
            return None
        return _read_json(path)

    def _video_config(self, context: StageContext) -> Any:
        if self.video is not None:
            return self.video
        config = context.config
        if isinstance(config, Mapping):
            return config.get("video")
        return getattr(config, "video", None)

    def _output_name(self, language: str, *, multiple: bool) -> str:
        if not multiple:
            return self.video_filename
        return DEFAULT_SONG_VIDEO_PATTERN.format(language=language)

    async def run(self, context: StageContext) -> StageResult:
        timeline = self._load_timeline(context)
        visuals = self._load_visuals(context)
        if visuals is not None:
            timeline = attach_visual_assets(timeline, visuals)
        compositions = timeline_to_compositions(
            timeline,
            video=self._video_config(context),
        )
        if not compositions:
            raise ConfigError(
                "song video stage has no composition to render",
                details={"timeline": self.timeline_filename},
            )
        primary = compositions[0]
        composition_path = context.run_dir / self.composition_filename
        composition_path.parent.mkdir(parents=True, exist_ok=True)
        composition_path.write_text(
            composition_to_yaml(primary),
            encoding="utf-8",
        )
        artifacts = [self.composition_filename]
        if self.render:
            multiple = len(compositions) > 1
            for composition in compositions:
                name = self._output_name(composition.language, multiple=multiple)
                output = context.run_dir / name
                output.parent.mkdir(parents=True, exist_ok=True)
                render_video(
                    composition,
                    output,
                    config=context.config,
                    highlight=self.highlight,
                )
                artifacts.append(name)
        run_id = context.run.id if context.run is not None else ""
        return StageResult(
            run_id=run_id,
            stage=self.name,
            status=StageState.SUCCEEDED,
            artifacts=artifacts,
            cost_usd=0.0,
            metadata={
                "compositions": len(compositions),
                "languages": [item.language for item in compositions],
                "timeline": self.timeline_filename,
                "visual_manifest": self.visual_manifest if visuals else "",
                "cost_usd": 0.0,
            },
        )


SongVideoFormat = SongVideoStage
SongVideoRenderer = SongVideoStage
song_video = attach_visual_assets
song_video_stage = SongVideoStage


def build_song_video_stage(
    *,
    options: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> SongVideoStage:
    """Build a :class:`SongVideoStage` from recipe options and overrides."""
    return SongVideoStage(options=options, **kwargs)
