"""Song card format: a still image card for a song, with optional character support.

The stage renders a deterministic still card that names the song. When a
character is selected it is loaded from the user directory and its name and art
style drive the card; when no character is selected the stage still proceeds,
rendering a plain song card. Character support is optional: a missing selected
character fails clearly, but an absent selection is not an error. Rendering is
deterministic and free: it uses the Pillow still renderer and makes no provider
or network call.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from aivia.characters.errors import CharacterNotFoundError
from aivia.characters.loader import CharacterLibrary
from aivia.compose.io import composition_to_yaml
from aivia.compose.render import render_still
from aivia.compose.schema import (
    CURRENT_COMPOSITION_SCHEMA_VERSION,
    Composition,
    TextAlign,
    TextLayer,
    TextWeight,
    VerticalAlign,
)
from aivia.core.errors import ConfigError
from aivia.core.models.run import StageResult, StageState
from aivia.core.stages.base import Stage, StageContext

__all__ = [
    "DEFAULT_SONG_CARD_COMPOSITION_FILENAME",
    "DEFAULT_SONG_CARD_FILENAME",
    "SONG_CARD_STAGE_NAME",
    "SongCard",
    "SongCardFormat",
    "SongCardRenderer",
    "SongCardStage",
    "build_song_card_stage",
    "song_card",
    "song_card_composition",
    "song_card_stage",
]

SONG_CARD_STAGE_NAME = "render"
"""The recipe stage name this implementation serves."""

DEFAULT_SONG_CARD_COMPOSITION_FILENAME = "song-card.yaml"
DEFAULT_SONG_CARD_FILENAME = "song-card.png"

_DEFAULT_BACKGROUND = "#0b0c10"
_DEFAULT_TEXT_COLOR = "#ffffff"


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def song_card_composition(
    *,
    song_title: str,
    character_name: str | None = None,
    art_style: str | None = None,
    width: int = 1080,
    height: int = 1080,
    background: str = _DEFAULT_BACKGROUND,
    duration_seconds: float = 6.0,
    composition_id: str = "",
    language: str = "",
    platform: str = "",
) -> Composition:
    """Build one still card composition that names the song."""
    margin = max(1, round(min(width, height) * 0.06))
    content_width = max(1, width - margin * 2)
    layers: list[dict[str, Any]] = []
    subject = song_title or "Untitled song"
    lines: list[str] = [subject]
    if character_name:
        lines.append(character_name)
    if art_style:
        lines.append(art_style)
    text = "\n".join(lines)
    layers.append(
        TextLayer(
            id="song",
            text=text,
            x=margin,
            y=round(height * 0.42),
            width=content_width,
            height=max(1, round(height * 0.16)),
            font_family="Lato",
            font_size=max(24.0, round(min(width, height) * 0.1)),
            font_weight=TextWeight.BOLD,
            color=_DEFAULT_TEXT_COLOR,
            align=TextAlign.CENTER,
            vertical_align=VerticalAlign.MIDDLE,
            shrink_to_fit=True,
            wrap=True,
            max_lines=3,
        ).model_dump(mode="python", by_alias=False)
    )
    slug = _text(song_title).replace(" ", "-") or "song"
    payload = {
        "id": composition_id or f"song-card-{slug}",
        "schema_version": CURRENT_COMPOSITION_SCHEMA_VERSION,
        "canvas": {
            "width": width,
            "height": height,
            "background": background,
        },
        "duration_seconds": duration_seconds,
        "fps": 30,
        "layers": layers,
        "language": language,
        "platform": platform,
    }
    try:
        return Composition.model_validate(payload)
    except ValueError as exc:
        raise ConfigError(f"invalid song card composition: {exc}") from exc


class SongCardStage(Stage):
    """Render a written song as a still card, optionally driven by a character.

    The stage never requires a character. When one is selected it loads the
    character from the user directory and its name and art style drive the
    card. With no character it renders a plain song card naming the song. A
    selected character that cannot be loaded fails clearly before any media
    work starts.
    """

    name = SONG_CARD_STAGE_NAME
    artifacts: tuple[str, ...] = (
        DEFAULT_SONG_CARD_COMPOSITION_FILENAME,
        DEFAULT_SONG_CARD_FILENAME,
    )

    def __init__(
        self,
        *,
        character: str | None = None,
        library: CharacterLibrary | None = None,
        data_dir: str | Path | None = None,
        scripts: Mapping[str, Any] | None = None,
        source_language: str = "en",
        background: str | None = None,
        render: bool = True,
        composition_filename: str = DEFAULT_SONG_CARD_COMPOSITION_FILENAME,
        image_filename: str = DEFAULT_SONG_CARD_FILENAME,
        options: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        merged = dict(options or {})
        supplied = {
            "character": character,
            "library": library,
            "data_dir": data_dir,
            "scripts": scripts,
            "source_language": source_language,
            "background": background,
            "render": render,
            "composition_filename": composition_filename,
            "image_filename": image_filename,
        }
        for key, value in supplied.items():
            if value is not None:
                merged[key] = value
        super().__init__(options=merged, **kwargs)
        self.character = _text(merged.get("character"))
        self.library: CharacterLibrary | None = merged.get("library")
        self.data_dir: Path | None = (
            Path(merged["data_dir"]).expanduser().resolve()
            if merged.get("data_dir") is not None
            else None
        )
        self.scripts = dict(merged.get("scripts") or {})
        self.source_language = _text(merged.get("source_language")) or "en"
        self.background = _text(merged.get("background")) or _DEFAULT_BACKGROUND
        self.render = bool(merged.get("render", True))
        self.composition_filename = (
            _text(merged.get("composition_filename"))
            or DEFAULT_SONG_CARD_COMPOSITION_FILENAME
        )
        self.image_filename = (
            _text(merged.get("image_filename")) or DEFAULT_SONG_CARD_FILENAME
        )
        if not self.render:
            self.artifacts = (self.composition_filename,)

    def _library_for(self, context: StageContext) -> CharacterLibrary:
        library = self.library
        if library is not None:
            return library
        data_dir = self.data_dir
        if data_dir is not None:
            return CharacterLibrary(data_dir=data_dir)
        return CharacterLibrary(data_dir=context.run_dir.parent.parent)

    def _song_title(self, context: StageContext) -> str:
        script = self.scripts.get(self.source_language)
        if script is None:
            return ""
        if isinstance(script, Mapping):
            return _text(script.get("title"))
        return _text(getattr(script, "title", ""))

    async def run(self, context: StageContext) -> StageResult:
        library = self._library_for(context)
        character = None
        art_style = None
        if self.character:
            try:
                character = library.load(self.character)
            except CharacterNotFoundError as exc:
                raise ConfigError(
                    f"unknown character: {self.character}",
                    details={
                        "stage": self.name,
                        "format": "song_card",
                        "error": "character_not_found",
                        "name": self.character,
                    },
                ) from exc
            art_style = character.art_style
        composition = song_card_composition(
            song_title=self._song_title(context),
            character_name=character.name if character else None,
            art_style=art_style,
            language=self.source_language,
            background=self.background,
        )
        composition_path = context.run_dir / self.composition_filename
        composition_path.parent.mkdir(parents=True, exist_ok=True)
        composition_path.write_text(composition_to_yaml(composition), encoding="utf-8")
        artifacts = [self.composition_filename]
        if self.render:
            image_path = context.run_dir / self.image_filename
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image = render_still(composition, 0.0, source_root=context.run_dir)
            image.save(image_path)
            artifacts.append(self.image_filename)
        return StageResult(
            run_id=context.run.id if context.run is not None else "",
            stage=self.name,
            status=StageState.SUCCEEDED,
            artifacts=artifacts,
            cost_usd=0.0,
            metadata={
                "character": self.character,
                "song_title": self._song_title(context),
            },
        )


SongCardFormat = SongCardStage
"""Alias for :class:`SongCardStage`."""

SongCardRenderer = SongCardStage
"""Alias for :class:`SongCardStage`."""

SongCard = SongCardStage
"""Alias for :class:`SongCardStage`."""

song_card = SongCardStage
"""Alias for :class:`SongCardStage`."""

song_card_stage = SongCardStage
"""Alias for :class:`SongCardStage`."""


def build_song_card_stage(
    *,
    options: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> SongCardStage:
    """Build a :class:`SongCardStage` from recipe options and overrides."""
    return SongCardStage(options=options, **kwargs)
