"""The subtitle stage: a timeline in, one styled ASS file per language out.

The voice stage writes a timeline with measured word timings. This stage turns
the subtitles or voice words for every language into a burn-ready ASS document,
applying a style preset and the safe area of the target platform. It is a pure
deterministic step: it reads no provider, spends no money and writes only under
the run directory.

An injected timeline or subtitle track keeps the stage offline in tests. By
default it reads ``timeline.json`` from the run directory, the hand-off the
voice stage already writes.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from aivia.compose.timeline import Timeline, TimelineLanguage
from aivia.core.config.video import Resolution
from aivia.core.errors import AiviaError, ConfigError
from aivia.core.models.run import StageResult, StageState
from aivia.core.platforms import Platform, SafeArea, get_platform
from aivia.core.stages.base import Stage, StageContext
from aivia.media.subtitles import (
    Subtitle as MediaSubtitle,
)
from aivia.media.subtitles import (
    SubtitleStyle,
    resolve_style,
)

__all__ = [
    "DEFAULT_SUBTITLE_FILENAME_PATTERN",
    "DEFAULT_TIMELINE_FILENAME",
    "SUBTITLE_STAGE_NAME",
    "Subtitle",
    "SubtitleStage",
    "SubtitleStageResult",
    "SubtitlesStage",
    "build_subtitle_stage",
    "run_subtitle_stage",
    "subtitle_stage",
]

SUBTITLE_STAGE_NAME = "subtitle"
"""The recipe stage name this implementation serves."""

DEFAULT_TIMELINE_FILENAME = "timeline.json"
"""The timeline file the voice stage writes under the run directory."""

DEFAULT_SUBTITLE_FILENAME_PATTERN = "subtitle.{language}.ass"
"""The per-language ASS file name pattern the stage writes."""

_DEFAULT_WIDTH = 1920
_DEFAULT_HEIGHT = 1080
_SAFE_AREA_KEYS = frozenset({"top", "bottom", "left", "right"})
_STYLE_KEYS = frozenset({"style", "preset", "style_preset"})


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _language_key(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ConfigError("language must be a string", details={"value": repr(value)})
    return value.strip().replace("_", "-").casefold()


def _base_language(value: str) -> str:
    return value.split("-", 1)[0]


def _section(config: Any, name: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name, None)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _normalise_languages(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw: Iterable[Any] = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        raw = value
    else:
        raise ConfigError(
            "languages must be a language code or a list of language codes",
            details={"value": repr(value)},
        )
    codes: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            raise ConfigError(
                "language codes must be strings",
                details={"value": repr(item)},
            )
        for part in item.split(","):
            code = _language_key(part)
            if code and code not in codes:
                codes.append(code)
    return codes


def _language_entries(config: Any) -> Mapping[Any, Any]:
    section = _section(config, "languages")
    if section is None:
        return {}
    nested = _field(section, "languages")
    if isinstance(nested, Mapping):
        return nested
    if isinstance(section, Mapping):
        return {
            key: value
            for key, value in section.items()
            if key not in {"default", "schema_version", "languages"}
        }
    return {}


def _language_config(config: Any, language: str) -> Any:
    entries = _language_entries(config)
    for key in (language, _base_language(language), "default"):
        if key in entries:
            return entries[key]
    return None


def _language_font(config: Any, language: str) -> str:
    entry = _language_config(config, language)
    return _text(_field(entry, "subtitle_font"))


def _accessibility_style(config: Any) -> SubtitleStyle | None:
    section = _section(config, "accessibility")
    if section is None:
        return None
    values = {
        "font": _field(section, "subtitle_font"),
        "font_size": _field(section, "subtitle_font_size"),
        "outline": _field(section, "subtitle_outline"),
        "background": _field(section, "subtitle_background"),
    }
    if not any(value is not None for value in values.values()):
        return None
    font = _text(values["font"]) or "Lato"
    size = values["font_size"]
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        size = 32
    outline = values["outline"]
    return SubtitleStyle(
        font=font,
        font_size=size,
        outline=2.0 if outline is not False else 0.0,
        shadow=1.0 if outline is not False else 0.0,
        background=bool(values["background"]),
    )


def _is_safe_area_mapping(value: Mapping[Any, Any]) -> bool:
    return bool(value) and set(value) <= _SAFE_AREA_KEYS


def _safe_area(value: Any) -> SafeArea:
    if isinstance(value, SafeArea):
        return value
    if isinstance(value, Mapping):
        try:
            return SafeArea(
                top=float(value.get("top", 0.0)),
                bottom=float(value.get("bottom", 0.0)),
                left=float(value.get("left", 0.0)),
                right=float(value.get("right", 0.0)),
            )
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                "safe area margins must be numbers",
                details={"field": "safe_area"},
            ) from exc
    raise ConfigError(
        "safe_area must be a SafeArea or a margin mapping",
        details={"value": type(value).__name__},
    )


def _resolution(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, Resolution):
        return value.width, value.height
    try:
        parsed = Resolution(str(value))
    except (TypeError, ValueError):
        return None
    return parsed.width, parsed.height


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


class SubtitleStageResult(StageResult):
    """A stage result whose metadata maps each language to its ASS file."""

    @property
    def files(self) -> dict[str, str]:
        raw = self.metadata.get("files")
        if not isinstance(raw, Mapping):
            return {}
        return {
            str(language): str(path)
            for language, path in raw.items()
            if isinstance(path, str)
        }

    @property
    def subtitles(self) -> dict[str, str]:
        """Alias for :attr:`files`."""
        return self.files

    @property
    def timeline(self) -> str:
        raw = self.metadata.get("timeline")
        return raw if isinstance(raw, str) else ""


class SubtitleStage(Stage):
    """Render the timeline's subtitles as one styled ASS file per language.

    ``options`` may carry ``languages``/``language``, ``timeline_filename``,
    ``filename_pattern``, ``preset``/``style_preset``, ``style``, ``styles``,
    ``style_presets``, ``safe_area``, ``safe_areas``, ``platform``, ``width``,
    ``height`` and ``highlight``. A caller may inject a ``timeline`` (model,
    mapping or path) or ``subtitles`` instead of reading ``timeline.json``.
    """

    name = SUBTITLE_STAGE_NAME
    artifacts: tuple[str, ...] = ()

    def __init__(
        self,
        name: str | None = None,
        *,
        options: Mapping[str, Any] | None = None,
        timeline: Any = None,
        subtitles: Any = None,
        languages: Any = None,
        preset: str | None = None,
        style_preset: str | None = None,
        style: Any = None,
        styles: Mapping[str, Any] | None = None,
        style_presets: Mapping[str, Any] | None = None,
        safe_area: Any = None,
        safe_areas: Mapping[str, Any] | None = None,
        platform: Platform | str | None = None,
        width: int | None = None,
        height: int | None = None,
        highlight: bool | str | None = None,
        timeline_filename: str | None = None,
        filename_pattern: str | None = None,
        subtitle_filename_pattern: str | None = None,
        **kwargs: Any,
    ) -> None:
        merged = dict(options or {})
        if timeline is not None:
            merged["timeline"] = timeline
        if subtitles is not None:
            merged["subtitles"] = subtitles
        if languages is not None:
            merged["languages"] = languages
        if preset is not None:
            merged["preset"] = preset
        if style_preset is not None:
            merged["style_preset"] = style_preset
        if style is not None:
            merged["style"] = style
        if styles is not None:
            merged["styles"] = styles
        if style_presets is not None:
            merged["style_presets"] = style_presets
        if safe_area is not None:
            merged["safe_area"] = safe_area
        if safe_areas is not None:
            merged["safe_areas"] = safe_areas
        if platform is not None:
            merged["platform"] = platform
        if width is not None:
            merged["width"] = width
        if height is not None:
            merged["height"] = height
        if highlight is not None:
            merged["highlight"] = highlight
        if timeline_filename is not None:
            merged["timeline_filename"] = timeline_filename
        if filename_pattern is not None:
            merged["filename_pattern"] = filename_pattern
        if subtitle_filename_pattern is not None:
            merged["subtitle_filename_pattern"] = subtitle_filename_pattern
        merged.update(kwargs)
        super().__init__(name=name or SUBTITLE_STAGE_NAME, options=merged)
        self.timeline = merged.get("timeline")
        self.subtitles = merged.get("subtitles")
        self.languages = merged.get("languages")
        self.preset = merged.get("preset")
        self.style_preset = merged.get("style_preset")
        self.style = merged.get("style")
        self.styles = merged.get("styles")
        self.style_presets = merged.get("style_presets")
        self.safe_area = merged.get("safe_area")
        self.safe_areas = merged.get("safe_areas")
        self.platform = merged.get("platform")
        self.width = merged.get("width")
        self.height = merged.get("height")
        self.highlight = merged.get("highlight", True)
        self.timeline_filename = str(
            merged.get("timeline_filename") or DEFAULT_TIMELINE_FILENAME
        )
        self.filename_pattern = str(
            merged.get("filename_pattern")
            or merged.get("subtitle_filename_pattern")
            or DEFAULT_SUBTITLE_FILENAME_PATTERN
        )
        if self.timeline is None:
            self.inputs = (self.timeline_filename,)

    def _load_timeline(self, context: StageContext) -> tuple[Timeline, str]:
        value = self.timeline
        if value is None:
            relative = self.timeline_filename
            path = context.run_dir / relative
            if not path.is_file():
                raise ConfigError(
                    "subtitle stage needs a timeline",
                    details={"timeline": relative, "run_dir": str(context.run_dir)},
                )
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ConfigError(
                    f"invalid timeline JSON: {path}",
                    details={"path": str(path)},
                ) from exc
            try:
                return Timeline.model_validate(payload), relative
            except ValueError as exc:
                raise ConfigError(
                    f"invalid timeline: {path}",
                    details={"path": str(path)},
                ) from exc
        if isinstance(value, (str, Path)):
            path = Path(value)
            if not path.is_absolute():
                path = context.run_dir / path
            if not path.is_file():
                raise ConfigError(
                    "subtitle stage needs a timeline",
                    details={"timeline": str(path)},
                )
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ConfigError(
                    f"invalid timeline JSON: {path}",
                    details={"path": str(path)},
                ) from exc
            try:
                return Timeline.model_validate(payload), str(path)
            except ValueError as exc:
                raise ConfigError(
                    f"invalid timeline: {path}",
                    details={"path": str(path)},
                ) from exc
        if isinstance(value, Timeline):
            return value, ""
        try:
            return Timeline.model_validate(value), ""
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                "timeline must be a Timeline, mapping or file path",
                details={"value": type(value).__name__},
            ) from exc

    def _target_languages(self, timeline: Timeline) -> list[TimelineLanguage]:
        entries = timeline.resolved_languages()
        by_code: dict[str, TimelineLanguage] = {}
        for entry in entries:
            code = _language_key(entry.language)
            if code:
                by_code.setdefault(code, entry)
        explicit = _normalise_languages(
            self.languages
            if self.languages is not None
            else self.options.get("language")
        )
        if not explicit:
            if by_code:
                return list(by_code.values())
            if entries:
                return list(entries)
            raise ConfigError(
                "timeline has no languages to subtitle",
                details={"timeline": timeline.id},
            )
        missing = [code for code in explicit if code not in by_code]
        if missing:
            raise ConfigError(
                "timeline is missing some languages",
                details={"missing": missing, "available": list(by_code)},
            )
        return [by_code[code] for code in explicit]

    def _injected_track(self, language: str) -> tuple[Any, bool]:
        value = self.subtitles
        if value is None:
            return None, False
        if isinstance(value, Mapping) and not (
            _STYLE_KEYS & set(value) or {"cues", "words"} & set(value)
        ):
            for key in (language, _base_language(language)):
                if key in value:
                    return value[key], True
            return None, False
        return value, True

    def _track(
        self,
        entry: TimelineLanguage,
        language: str,
    ) -> tuple[MediaSubtitle, bool]:
        injected, injected_has_style = self._injected_track(language)
        source: Any = injected
        if source is None and entry.subtitles is not None:
            source = entry.subtitles
        had_style = injected_has_style or (
            injected is None
            and entry.subtitles is not None
            and entry.subtitles.style is not None
        )
        if source is not None:
            track = self._to_media_subtitle(source)
        else:
            words = [
                word for voice in entry.voices if voice.enabled for word in voice.words
            ]
            if not words:
                raise ConfigError(
                    "timeline has no subtitles or timed words for a language",
                    details={"language": language},
                )
            track = MediaSubtitle.from_words(
                [word.model_dump(mode="python") for word in words],
                language=language,
            )
        if not track.language:
            track = track.model_copy(update={"language": language})
        return track, had_style

    @staticmethod
    def _to_media_subtitle(value: Any) -> MediaSubtitle:
        if isinstance(value, MediaSubtitle):
            return value.model_copy(deep=True)
        if isinstance(value, (str, Path)):
            return MediaSubtitle.from_file(value)
        if isinstance(value, Mapping):
            try:
                return MediaSubtitle.model_validate(value)
            except ValueError as exc:
                raise ConfigError(
                    "subtitle track is invalid",
                    details={"field": "subtitles"},
                ) from exc
        to_media = getattr(value, "to_media", None)
        if callable(to_media):
            return cast(MediaSubtitle, to_media())
        if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
            try:
                return MediaSubtitle.model_validate({"cues": list(value)})
            except (TypeError, ValueError) as exc:
                raise ConfigError(
                    "subtitle track is invalid",
                    details={"field": "subtitles"},
                ) from exc
        raise ConfigError(
            "subtitles must be a track, cues or a mapping",
            details={"value": type(value).__name__},
        )

    def _mapped(self, mapping: Any, language: str) -> Any:
        if not isinstance(mapping, Mapping):
            return None
        for key in (language, _base_language(language)):
            if key in mapping:
                return mapping[key]
        return None

    @staticmethod
    def _names_font(value: Any) -> bool:
        if isinstance(value, Mapping):
            return bool({"font", "fontname", "font_name", "family"} & set(value))
        return isinstance(value, SubtitleStyle)

    def _style_for(
        self,
        language: str,
        track: MediaSubtitle,
        had_style: bool,
        config: Any,
    ) -> SubtitleStyle:
        explicit = self._mapped(self.styles, language)
        if explicit is None:
            explicit = self.style
        if explicit is not None:
            chosen = self._resolve_style(explicit, language)
            explicit_font = self._names_font(explicit)
        else:
            explicit_font = False
            preset = self._mapped(self.style_presets, language)
            if preset is None:
                preset = self.preset if self.preset is not None else self.style_preset
            if preset is not None:
                chosen = self._resolve_style(preset, language)
            elif had_style:
                chosen = track.style.model_copy(deep=True)
                explicit_font = True
            else:
                chosen = _accessibility_style(config) or track.style.model_copy(
                    deep=True
                )

        font = _language_font(config, language)
        if font and not explicit_font:
            chosen = chosen.model_copy(update={"font": font})
        return chosen

    @staticmethod
    def _resolve_style(value: Any, language: str) -> SubtitleStyle:
        try:
            return resolve_style(value)
        except (TypeError, ValueError, AiviaError) as exc:
            raise ConfigError(
                "unknown subtitle style",
                details={"language": language, "value": repr(value)},
            ) from exc

    def _platform_for(self, timeline: Timeline) -> Platform | None:
        value = self.platform
        if value is None and len(timeline.platforms) == 1:
            value = timeline.platforms[0]
        if value is None:
            return None
        if isinstance(value, Platform):
            return value
        try:
            return get_platform(str(value))
        except (KeyError, ValueError) as exc:
            raise ConfigError(
                f"unknown platform: {value!r}",
                details={"platform": str(value)},
            ) from exc

    def _safe_area_for(
        self,
        language: str,
        platform: Platform | None,
    ) -> SafeArea:
        mapped = self._mapped(self.safe_areas, language)
        if mapped is not None:
            return _safe_area(mapped)
        value = self.safe_area
        if value is not None:
            if isinstance(value, Mapping) and not _is_safe_area_mapping(value):
                per_language = self._mapped(value, language)
                if per_language is not None:
                    return _safe_area(per_language)
            return _safe_area(value)
        if platform is not None:
            return platform.safe_area
        return SafeArea()

    def _canvas(
        self,
        config: Any,
        platform: Platform | None,
    ) -> tuple[int, int]:
        width = self.width
        height = self.height
        if width is not None and height is not None:
            return int(width), int(height)

        video = _section(config, "video")
        if platform is not None:
            merge = getattr(video, "for_platform", None)
            if callable(merge):
                with contextlib.suppress(KeyError, ValueError, AiviaError):
                    video = merge(platform.id)
        resolution = _resolution(_field(video, "resolution"))
        if resolution is None and platform is not None:
            limit = platform.limit("video")
            resolutions = getattr(limit, "resolutions", ()) if limit else ()
            if resolutions:
                resolution = _resolution(resolutions[0])
        if resolution is None:
            resolution = (_DEFAULT_WIDTH, _DEFAULT_HEIGHT)

        resolved_width = int(width) if width is not None else resolution[0]
        resolved_height = int(height) if height is not None else resolution[1]
        for value, label in ((resolved_width, "width"), (resolved_height, "height")):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConfigError(
                    f"subtitle canvas {label} must be a positive integer",
                    details={"field": label},
                )
        return resolved_width, resolved_height

    async def run(self, context: StageContext) -> StageResult:
        timeline, timeline_source = self._load_timeline(context)
        languages = self._target_languages(timeline)
        platform = self._platform_for(timeline)
        width, height = self._canvas(context.config, platform)
        run_id = context.run.id if context.run is not None else ""

        artifacts: list[str] = []
        files: dict[str, str] = {}
        presets: dict[str, str] = {}
        safe_areas: dict[str, dict[str, float]] = {}
        for entry in languages:
            language = _language_key(entry.language)
            track, had_style = self._track(entry, language)
            style = self._style_for(language, track, had_style, context.config)
            area = self._safe_area_for(language, platform)
            try:
                document = track.to_ass(
                    width=width,
                    height=height,
                    style=style,
                    safe_area=area,
                    platform=platform,
                    highlight=self.highlight,
                )
            except ValueError as exc:
                raise ConfigError(
                    "subtitle track cannot be rendered as ASS",
                    details={"language": language},
                ) from exc
            relative = self.filename_pattern.format(
                language=language,
                code=language,
            )
            _write_text(context.run_dir / relative, document)
            artifacts.append(relative)
            files[language] = relative
            presets[language] = style.ass_name
            safe_areas[language] = area.as_dict()

        self.artifacts = tuple(artifacts)
        preset = self.preset if self.preset is not None else self.style_preset
        metadata = {
            "languages": list(files),
            "files": files,
            "subtitles": files,
            "styles": presets,
            "presets": presets,
            "preset": str(preset) if preset is not None else "",
            "safe_areas": safe_areas,
            "width": width,
            "height": height,
            "platform": platform.id if platform is not None else "",
            "timeline": timeline_source,
            "cost_usd": 0.0,
        }
        return SubtitleStageResult(
            run_id=run_id,
            stage=self.name,
            status=StageState.SUCCEEDED,
            artifacts=artifacts,
            cost_usd=0.0,
            metadata=metadata,
        )


Subtitle = SubtitleStage
SubtitlesStage = SubtitleStage
subtitle_stage = SubtitleStage


def build_subtitle_stage(
    *,
    options: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> SubtitleStage:
    """Build a :class:`SubtitleStage` from recipe options and overrides."""
    return SubtitleStage(options=options, **kwargs)


async def run_subtitle_stage(
    context: StageContext,
    *,
    stage: SubtitleStage | None = None,
    **options: Any,
) -> StageResult:
    """Build a stage unless one is supplied, then run it asynchronously."""
    if stage is not None and options:
        raise ConfigError("pass either a stage or options, not both")
    active = stage if stage is not None else SubtitleStage(**options)
    return await active.run(context)
