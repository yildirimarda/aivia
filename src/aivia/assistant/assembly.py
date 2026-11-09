"""Assemble tool outputs and patches into a single renderable composition.

Generation tool outputs (images, video, audio) are registered as composition
assets and turned into placement layers or audio tracks. Composition tool
patches are applied on top in recording order. The assembled composition is
rendered with the shared Pillow still renderer.

No media is generated or rendered here: assembly only edits the declarative
composition and hands it to :func:`aivia.compose.render_still`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from PIL import Image

from aivia.assistant.tool_framework import ToolResult
from aivia.compose import Canvas, Composition, render_still
from aivia.compose.patch import CompositionPatch, apply_patch
from aivia.media import MediaResult

_LAYER_KINDS = frozenset({"image", "video"})
_TRACK_KINDS = frozenset({"voice", "music", "audio", "background", "sfx", "effects"})


class Assembly:
    """Collect tool outputs and patches into one renderable composition.

    ``add_output`` records the result of one tool call. Generation outputs
    (``MediaResult``) become registered assets and placement layers or audio
    tracks; composition outputs (a ``ToolResult`` carrying a ``patch`` and/or a
    full ``composition``) are merged in. ``add_patch`` records a
    :class:`~aivia.compose.patch.CompositionPatch` to apply before rendering.

    ``assemble`` builds the final
    :class:`~aivia.compose.schema.Composition` and ``render`` renders it
    through :func:`aivia.compose.render_still`, the shared Pillow still
    renderer.
    """

    def __init__(
        self,
        composition: Composition | dict[str, Any] | None = None,
        *,
        assets: dict[str, str] | None = None,
    ) -> None:
        if composition is None:
            self._composition = Composition(
                canvas=Canvas(width=1280, height=720, background="#000000"),
                duration_seconds=5.0,
                layers=[],
                audio_tracks=[],
                effects=[],
            )
        elif isinstance(composition, Composition):
            self._composition = composition
        else:
            self._composition = Composition.model_validate(composition)
        self._assets: dict[str, str] = dict(assets or {})
        self._outputs: list[tuple[str, Any]] = []
        self._patches: list[CompositionPatch] = []
        self._asset_counter = 0

    @property
    def composition(self) -> Composition:
        """The base composition before outputs and patches are applied."""
        return self._composition

    @property
    def assets(self) -> dict[str, str]:
        """The registered asset source -> path mapping."""
        return dict(self._assets)

    @property
    def outputs(self) -> list[tuple[str, Any]]:
        """The recorded ``(tool, output)`` pairs in recording order."""
        return list(self._outputs)

    @property
    def patches(self) -> list[CompositionPatch]:
        """The recorded patches in recording order."""
        return list(self._patches)

    def add_output(self, tool: str, output: Any) -> None:
        """Record one tool output for later assembly."""
        self._outputs.append((tool, output))

    def add_patch(self, patch: CompositionPatch | Mapping[str, Any]) -> None:
        """Record one composition patch to apply before rendering."""
        self._patches.append(
            CompositionPatch.from_dict(patch) if isinstance(patch, Mapping) else patch
        )

    # --- assembly -------------------------------------------------------

    def assemble(self) -> Composition:
        """Merge recorded outputs and patches into a single composition."""
        data = self._composition.model_dump(mode="python")
        assets = dict(self._assets)

        for tool, output in self._outputs:
            if isinstance(output, ToolResult):
                raw = output.output
                if isinstance(raw, Mapping):
                    if "composition" in raw:
                        data = {
                            **data,
                            **raw["composition"],
                        }
                    if "patch" in raw:
                        data = apply_patch(
                            data, raw["patch"], validate=True
                        ).model_dump(mode="python")
            elif isinstance(output, MediaResult):
                assets, data = self._merge_media_result(
                    tool or "output", output, assets, data
                )
            elif isinstance(output, Mapping):
                if "composition" in output:
                    data = {**data, **output["composition"]}
                if "patch" in output:
                    data = apply_patch(data, output["patch"], validate=True).model_dump(
                        mode="python"
                    )

        composition = Composition.model_validate(data)
        for patch in self._patches:
            composition = apply_patch(composition, patch, validate=True)
        self._assets = assets
        return composition

    def render(
        self,
        *,
        time_seconds: float = 0.0,
        assets: dict[str, str] | None = None,
        **render_kwargs: Any,
    ) -> Image.Image:
        """Assemble and render one still frame with the shared renderer."""
        composition = self.assemble()
        resolver = assets if assets is not None else self._assets
        return render_still(
            composition,
            time_seconds=time_seconds,
            assets=resolver,
            **render_kwargs,
        )

    # --- internal -------------------------------------------------------

    def _merge_media_result(
        self,
        tool: str,
        result: MediaResult,
        assets: dict[str, str],
        data: dict[str, Any],
    ) -> tuple[dict[str, str], dict[str, Any]]:
        kind = getattr(result, "kind", "") or ""
        paths = tuple(result.paths)
        if not paths:
            return assets, data
        prefix = f"{tool}-{kind}"
        layers = list(data.get("layers", []))
        tracks = list(data.get("audio_tracks", []))
        for index, path in enumerate(paths):
            source = f"{prefix}-{index}"
            assets[source] = str(path)
            layer_id = f"{prefix}-{index}"
            if kind in _LAYER_KINDS:
                layers.append(
                    {
                        "id": layer_id,
                        "type": kind,
                        "source": source,
                    }
                )
            elif kind in _TRACK_KINDS:
                tracks.append(
                    {
                        "id": layer_id,
                        "source": str(path),
                        "role": kind,
                    }
                )
        data["layers"] = layers
        data["audio_tracks"] = tracks
        return assets, data
