"""Character card format: a still image built around a consistent character.

The format requires a selected character. When none is selected -- no ``--character``
flag, no order or instruction character and no profile character -- it fails
clearly instead of silently dropping character consistency. A selected character
is loaded from the user directory and its look, voice and pose library drive the
card. Rendering is deterministic and free: it uses the Pillow still renderer and
makes no provider or network call.
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
    "CHARACTER_CARD_STAGE_NAME",
    "DEFAULT_CHARACTER_CARD_COMPOSITION_FILENAME",
    "DEFAULT_CHARACTER_CARD_FILENAME",
    "CharacterCard",
    "CharacterCardFormat",
    "CharacterCardRenderer",
    "CharacterCardStage",
    "build_character_card_stage",
    "character_card",
    "character_card_stage",
]

CHARACTER_CARD_STAGE_NAME = "render"
"""The recipe stage name this implementation serves."""

DEFAULT_CHARACTER_CARD_COMPOSITION_FILENAME = "character-card.yaml"
DEFAULT_CHARACTER_CARD_FILENAME = "character-card.png"

_DEFAULT_BACKGROUND = "#0b0c10"
_DEFAULT_TEXT_COLOR = "#ffffff"


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _character_card_composition(
    *,
    character_name: str,
    art_style: str,
    width: int = 1080,
    height: int = 1080,
    background: str = _DEFAULT_BACKGROUND,
    duration_seconds: float = 6.0,
    composition_id: str = "",
    language: str = "",
    platform: str = "",
) -> Composition:
    """Build one still card composition that names the selected character."""
    margin = max(1, round(min(width, height) * 0.06))
    content_width = max(1, width - margin * 2)
    layers: list[dict[str, Any]] = []
    subject = art_style or character_name
    layers.append(
        TextLayer(
            id="character",
            text=subject,
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
            max_lines=2,
        ).model_dump(mode="python", by_alias=False)
    )
    payload = {
        "id": composition_id or f"character-card-{character_name}",
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
        raise ConfigError(f"invalid character card composition: {exc}") from exc


class CharacterCardStage(Stage):
    """Render a written script as a still card driven by a selected character.

    The stage requires a selected character. With none selected it raises a clear
    error before any media work starts. With a character selected it loads the
    character from the user directory, writes the composition manifest and renders
    a still PNG.
    """

    name = CHARACTER_CARD_STAGE_NAME
    artifacts: tuple[str, ...] = (
        DEFAULT_CHARACTER_CARD_COMPOSITION_FILENAME,
        DEFAULT_CHARACTER_CARD_FILENAME,
    )

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
        composition_filename: str = DEFAULT_CHARACTER_CARD_COMPOSITION_FILENAME,
        image_filename: str = DEFAULT_CHARACTER_CARD_FILENAME,
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
        self.aspect = merged.get("aspect")
        self.background = _text(merged.get("background")) or _DEFAULT_BACKGROUND
        self.render = bool(merged.get("render", True))
        self.composition_filename = (
            _text(merged.get("composition_filename"))
            or DEFAULT_CHARACTER_CARD_COMPOSITION_FILENAME
        )
        self.image_filename = (
            _text(merged.get("image_filename")) or DEFAULT_CHARACTER_CARD_FILENAME
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

    async def run(self, context: StageContext) -> StageResult:
        if not self.character:
            raise ConfigError(
                "character card requires a selected character; none was selected. "
                "Pass --character <name> or set the profile's character.",
                details={
                    "stage": self.name,
                    "format": "character_card",
                    "error": "character_not_selected",
                },
            )
        library = self._library_for(context)
        try:
            character = library.load(self.character)
        except CharacterNotFoundError:
            raise
        composition = _character_card_composition(
            character_name=character.name,
            art_style=character.art_style,
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
            metadata={"character": self.character},
        )


CharacterCardFormat = CharacterCardStage
"""Alias for :class:`CharacterCardStage`."""

CharacterCard = CharacterCardStage
"""Alias for :class:`CharacterCardStage`."""

CharacterCardRenderer = CharacterCardStage
"""Alias for :class:`CharacterCardStage`."""

character_card = CharacterCardStage
"""Alias for :class:`CharacterCardStage`."""


def build_character_card_stage(
    *,
    options: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> CharacterCardStage:
    """Build a :class:`CharacterCardStage` from recipe options and overrides."""
    return CharacterCardStage(options=options, **kwargs)


character_card_stage = build_character_card_stage
"""Alias for :func:`build_character_card_stage`."""
