"""Character short format: a short video built around a consistent character.

The format requires a selected character. When none is selected -- no ``--character``
flag, no order or instruction character and no profile character -- it fails
clearly instead of silently dropping character consistency. A selected character
is loaded from the user directory and its look, voice and pose library drive the
visuals. Rendering itself is deterministic and free: this module only builds the
composition manifest a renderer turns into a video, so no provider or network
call is made here.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from aivia.characters.errors import CharacterNotFoundError
from aivia.characters.loader import CharacterLibrary
from aivia.compose.io import composition_to_yaml
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
    "CHARACTER_SHORT_STAGE_NAME",
    "DEFAULT_CHARACTER_SHORT_COMPOSITION_FILENAME",
    "CharacterShort",
    "CharacterShortFormat",
    "CharacterShortRenderer",
    "CharacterShortStage",
    "build_character_short_stage",
    "character_short",
    "character_short_stage",
]

CHARACTER_SHORT_STAGE_NAME = "render"
"""The recipe stage name this implementation serves."""

DEFAULT_CHARACTER_SHORT_COMPOSITION_FILENAME = "character-short.yaml"
"""The composition manifest written into the run directory."""

_DEFAULT_BACKGROUND = "#0b0c10"
_DEFAULT_TEXT_COLOR = "#ffffff"


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _character_composition(
    *,
    character_name: str,
    art_style: str,
    width: int = 1080,
    height: int = 1920,
    background: str = _DEFAULT_BACKGROUND,
    duration_seconds: float = 6.0,
    composition_id: str = "",
    language: str = "",
    platform: str = "",
) -> Composition:
    """Build one still composition that names the selected character."""
    margin = max(1, round(min(width, height) * 0.06))
    content_width = max(1, width - margin * 2)
    layers: list[dict[str, Any]] = []
    subject = art_style or character_name
    layers.append(
        TextLayer(
            id="character",
            text=subject,
            x=margin,
            y=round(height * 0.45),
            width=content_width,
            height=max(1, round(height * 0.12)),
            font_family="Lato",
            font_size=max(24.0, round(min(width, height) * 0.09)),
            font_weight=TextWeight.BOLD,
            color=_DEFAULT_TEXT_COLOR,
            align=TextAlign.CENTER,
            vertical_align=VerticalAlign.MIDDLE,
            shrink_to_fit=True,
            wrap=True,
            max_lines=2,
        ).model_dump(mode="python", by_alias=False)
    )
    payload = {
        "id": composition_id or f"character-short-{character_name}",
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
        raise ConfigError(f"invalid character short composition: {exc}") from exc


class CharacterShortStage(Stage):
    """Render a written script as a short video driven by a selected character.

    The stage requires a selected character. With none selected it raises a clear
    error before any media work starts. With a character selected it loads the
    character from the user directory and writes the composition manifest that
    the render stage turns into a video.
    """

    name = CHARACTER_SHORT_STAGE_NAME
    artifacts: tuple[str, ...] = (DEFAULT_CHARACTER_SHORT_COMPOSITION_FILENAME,)

    def __init__(
        self,
        *,
        character: str | None = None,
        library: CharacterLibrary | None = None,
        data_dir: str | Path | None = None,
        scripts: Mapping[str, Any] | None = None,
        source_language: str = "en",
        aspect: Any = None,
        background: str | None = None,
        render: bool = True,
        composition_filename: str = DEFAULT_CHARACTER_SHORT_COMPOSITION_FILENAME,
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
            "aspect": aspect,
            "background": background,
            "render": render,
            "composition_filename": composition_filename,
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
        self.aspect = merged.get("aspect")
        self.background = _text(merged.get("background")) or _DEFAULT_BACKGROUND
        self.render = bool(merged.get("render", True))
        self.composition_filename = (
            _text(merged.get("composition_filename"))
            or DEFAULT_CHARACTER_SHORT_COMPOSITION_FILENAME
        )

    def _library_for(self, context: StageContext) -> CharacterLibrary:
        library = self.library
        if library is not None:
            return library
        data_dir = self.data_dir
        if data_dir is not None:
            return CharacterLibrary(data_dir=data_dir)
        return CharacterLibrary(data_dir=context.run_dir.parent.parent)

    async def run(self, context: StageContext) -> StageResult:
        if not self.character:
            raise ConfigError(
                "character short requires a selected character; none was selected. "
                "Pass --character <name> or set the profile's character.",
                details={
                    "stage": self.name,
                    "format": "character_short",
                    "error": "character_not_selected",
                },
            )
        library = self._library_for(context)
        try:
            character = library.load(self.character)
        except CharacterNotFoundError:
            raise
        composition = _character_composition(
            character_name=character.name,
            art_style=character.art_style,
            language=self.source_language,
            background=self.background,
        )
        composition_path = context.run_dir / self.composition_filename
        composition_path.parent.mkdir(parents=True, exist_ok=True)
        composition_path.write_text(composition_to_yaml(composition), encoding="utf-8")
        artifacts = [self.composition_filename]
        return StageResult(
            run_id=context.run.id if context.run is not None else "",
            stage=self.name,
            status=StageState.SUCCEEDED,
            artifacts=artifacts,
            cost_usd=0.0,
            metadata={"character": self.character},
        )


CharacterShortFormat = CharacterShortStage
"""Alias for :class:`CharacterShortStage`."""

CharacterShort = CharacterShortStage
"""Alias for :class:`CharacterShortStage`."""

CharacterShortRenderer = CharacterShortStage
"""Alias for :class:`CharacterShortStage`."""

character_short = CharacterShortStage
"""Alias for :class:`CharacterShortStage`."""


def build_character_short_stage(
    *,
    options: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> CharacterShortStage:
    """Build a :class:`CharacterShortStage` from recipe options and overrides."""
    return CharacterShortStage(options=options, **kwargs)


character_short_stage = build_character_short_stage
"""Alias for :func:`build_character_short_stage`."""
