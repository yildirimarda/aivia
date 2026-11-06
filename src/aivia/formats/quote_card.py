"""Quote card format helpers.

The typography and composition logic lives in :mod:`aivia.compose.quote_card`;
this module is the format-facing import path so a recipe, CLI command or test
can reach the quote card builder from :mod:`aivia.formats`.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from aivia.compose.io import composition_to_yaml
from aivia.compose.quote_card import *  # noqa: F403
from aivia.compose.quote_card import __all__ as _quote_card_all
from aivia.compose.quote_card import quote_card_composition, render_quote_card
from aivia.core.models.run import StageResult
from aivia.core.stages import Stage, StageContext
from aivia.writing.schema import Script

__all__ = [
    *_quote_card_all,
    "QuoteCardRenderStage",
    "QuoteCardStage",
    "build_quote_card_stage",
    "quote_card_stage",
]


def _read_script(directory: Path, language: str) -> Script:
    names = (
        f"script.{language}.json",
        f"script_{language}.json",
        "script.json",
    )
    for name in names:
        path = directory / name
        if path.is_file():
            return Script.model_validate_json(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        f"no written script for language {language!r} in {directory}"
    )


class QuoteCardStage(Stage):
    """Render a written quote script into a still card.

    The stage is deterministic and free: it reads a script from the run
    directory, fills the selected typography template and writes both the
    composition and a rendered PNG. It never calls a provider or the network.
    """

    name = "quote_card"
    artifacts: tuple[str, ...] = ("quote-card.yaml", "quote-card.png")

    def __init__(
        self,
        *,
        scripts: Mapping[str, Script] | None = None,
        template: Any = None,
        source_language: str = "en",
        render: bool = True,
        composition_filename: str = "quote-card.yaml",
        image_filename: str = "quote-card.png",
        options: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(options=options, **kwargs)
        self.scripts = dict(scripts or {})
        self.template = (
            template
            if template is not None
            else self.options.get("template", self.options.get("typography"))
        )
        self.source_language = source_language or self.options.get(
            "source_language", "en"
        )
        self.render = render and bool(self.options.get("render", True))
        self.composition_filename = composition_filename or self.options.get(
            "composition_filename", "quote-card.yaml"
        )
        self.image_filename = image_filename or self.options.get(
            "image_filename", "quote-card.png"
        )
        self.artifacts = (
            (self.composition_filename, self.image_filename)
            if self.render
            else (self.composition_filename,)
        )

    async def run(self, context: StageContext) -> StageResult:
        script = self.scripts.get(self.source_language)
        if script is None:
            script = _read_script(context.run_dir, self.source_language)
        composition = quote_card_composition(
            script,
            self.template,
            language=self.source_language,
        )
        composition_path = context.run_dir / self.composition_filename
        composition_path.parent.mkdir(parents=True, exist_ok=True)
        composition_path.write_text(
            composition_to_yaml(composition),
            encoding="utf-8",
        )
        artifacts = [self.composition_filename]
        if self.render:
            image_path = context.run_dir / self.image_filename
            image_path.parent.mkdir(parents=True, exist_ok=True)
            render_quote_card(
                script,
                self.template,
                output=image_path,
                language=self.source_language,
            )
            artifacts.append(self.image_filename)
        return StageResult(
            run_id=context.run.id if context.run is not None else "",
            stage=self.name,
            artifacts=artifacts,
            cost_usd=0.0,
        )


class QuoteCardRenderStage(QuoteCardStage):
    """The render-named quote card stage used by a format recipe."""

    name = "render"


def build_quote_card_stage(**kwargs: Any) -> QuoteCardStage:
    """Build a :class:`QuoteCardStage` from keyword options."""
    return QuoteCardStage(**kwargs)


def quote_card_stage(**kwargs: Any) -> QuoteCardStage:
    """Alias for :func:`build_quote_card_stage`."""
    return QuoteCardStage(**kwargs)
