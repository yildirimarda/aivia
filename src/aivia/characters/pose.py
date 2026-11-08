"""Pose library: named cutouts with alpha, anchors and tags.

A pose cutout is a named image stored next to a character that carries an alpha
channel, named anchor points for compositing and tags for lookup. Edge cleanup
removes the background so the cutout sits cleanly on any canvas. Loading and
storing are pure file operations: no AI call and no network request happen here.

Poses live at ``<data dir>/characters/poses/<character>/<name>.yaml`` next to
their cutout images. A pose without a character is stored under the shared
``poses`` directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from aivia.characters.errors import PoseError, PoseNotFoundError, PoseSchemaError
from aivia.core.errors import ValidationError
from aivia.providers.image_edit.rembg import remove_background

__all__ = [
    "POSE_LIBRARY_SCHEMA_VERSION",
    "Pose",
    "PoseLibrary",
    "cleanup_edges",
    "load_pose_library",
]

POSE_LIBRARY_SCHEMA_VERSION = 1
"""The pose library schema version written by this release."""

POSES_DIR = "poses"
"""Directory under the characters directory that holds pose libraries."""


class Pose(BaseModel):
    """One named pose cutout in a pose library.

    ``image`` is the path to the cutout image relative to the pose library's
    directory. ``alpha`` records whether the image carries an alpha channel.
    ``anchors`` maps anchor names to normalized (x, y) points on the cutout used
    for compositing. ``tags`` categorise the pose for lookup. ``edge_cleanup``
    records the method used to clean the cutout's edges, empty when none was
    applied.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    name: str = Field(min_length=1)
    image: str = Field(min_length=1)
    alpha: bool = False
    anchors: dict[str, tuple[float, float]] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    edge_cleanup: str = ""

    @field_validator("name", "image", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator("anchors", mode="before")
    @classmethod
    def _normalise_anchors(cls, value: Any) -> dict[str, tuple[float, float]]:
        anchors: dict[str, tuple[float, float]] = {}
        for key, point in (value or {}).items():
            anchor_name = str(key).strip()
            if not anchor_name:
                raise ValueError("anchor names must be non-empty")
            if anchor_name in anchors:
                raise ValueError(f"duplicate anchor: {anchor_name}")
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError(f"anchor {anchor_name!r} must be an (x, y) pair")
            x, y = point
            try:
                x = float(x)
                y = float(y)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"anchor {anchor_name!r} point must be numeric"
                ) from exc
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError(f"anchor {anchor_name!r} point must be within 0-1")
            anchors[anchor_name] = (x, y)
        return anchors

    @field_validator("tags", mode="before")
    @classmethod
    def _normalise_tags(cls, value: Any) -> list[str]:
        tags: list[str] = []
        for tag in value or []:
            text = str(tag).strip()
            if not text:
                raise ValueError("tags must not contain blank entries")
            tags.append(text)
        return tags

    @field_validator("edge_cleanup")
    @classmethod
    def _normalise_edge_cleanup(cls, value: str) -> str:
        text = value.strip()
        if text and text not in {"rembg"}:
            raise ValueError(f"unsupported edge cleanup method: {text!r}")
        return text


def _safe_image_name(image: str) -> str:
    """Return *image* when it is a safe, non-escaping relative file name."""
    text = image.strip()
    if not text:
        raise ValidationError("pose image must be a non-empty path")
    if Path(text).is_absolute() or Path(text).name != text:
        raise ValidationError(
            f"pose image must be a simple file name, not a path: {image!r}",
            details={"image": image},
        )
    return text


def cleanup_edges(data: bytes, *, tolerance: int = 24) -> bytes:
    """Return *data* as an RGBA PNG with the border-connected background clear.

    The background is every region reachable from the image border whose pixels
    stay within ``tolerance`` of the border pixel that seeded that region.
    Foreground pixels keep their colour and existing transparency is preserved.
    This is a deterministic, no-cost local fallback for cutout edge cleanup.
    """
    return remove_background(data, tolerance=tolerance)


class PoseLibrary:
    """Named pose cutouts stored next to a character.

    Each pose is a YAML file beside its cutout image. Loading a pose never
    calls a model or the network. Storing a pose optionally runs edge cleanup
    so the stored cutout carries a clean alpha channel.
    """

    def __init__(
        self,
        *,
        data_dir: str | Path | None = None,
        character: str | None = None,
    ) -> None:
        from aivia.core import paths

        if data_dir is None:
            self.data_dir = paths.data_dir()
        else:
            self.data_dir = Path(data_dir).expanduser().resolve()
        self.character = character.strip() if isinstance(character, str) else None
        if self.character and any(
            part in {"", ".", ".."} for part in self.character.split("/")
        ):
            raise PoseError(
                f"character name may not escape the poses directory: {character}",
                details={"character": character},
            )
        self.poses_dir = self.data_dir / "characters" / POSES_DIR
        if self.character:
            self.poses_dir = self.poses_dir / self.character

    def pose_path(self, name: str) -> Path:
        """Return the pose file for *name*."""
        if not isinstance(name, str) or not name.strip():
            raise PoseError("pose name must be a non-empty string")
        text = name.strip().replace("\\", "/")
        if any(part in {"", ".", ".."} for part in text.split("/")):
            raise PoseError(
                f"pose name may not escape the poses directory: {name}",
                details={"name": text},
            )
        return self.poses_dir / f"{text}.yaml"

    def names(self) -> tuple[str, ...]:
        """Return every pose name with a file in the poses directory."""
        if not self.poses_dir.is_dir():
            return ()
        names: list[str] = []
        for path in sorted(self.poses_dir.glob("*.yaml")):
            names.append(path.stem)
        return tuple(names)

    def exists(self, name: str) -> bool:
        """Whether a pose file exists for *name*."""
        try:
            return self.pose_path(name).is_file()
        except PoseError:
            return False

    def store(
        self,
        pose: Pose,
        *,
        image_bytes: bytes,
        edge_cleanup: bool = False,
    ) -> Pose:
        """Store *pose* and its cutout image, optionally cleaning its edges.

        When ``edge_cleanup`` is true the background is removed from the image
        before it is stored, the stored image is written as a PNG and the pose
        records ``alpha=True`` and ``edge_cleanup="rembg"``.
        """
        image_name = _safe_image_name(pose.image)
        stored = Pose.model_validate(pose)
        stored.image = image_name

        image_data = image_bytes
        if edge_cleanup:
            image_data = cleanup_edges(image_data)
            stored.alpha = True
            stored.edge_cleanup = "rembg"

        self.poses_dir.mkdir(parents=True, exist_ok=True)
        path = self.pose_path(stored.name)
        if path.is_file():
            raise PoseError(
                f"pose already exists at {path}",
                details={
                    "character": self.character,
                    "pose": stored.name,
                    "file": str(path),
                },
            )
        image_target = self.poses_dir / image_name
        try:
            image_target.write_bytes(image_data)
        except OSError as exc:
            raise PoseError(
                f"cannot store pose cutout {image_target}: {exc}",
                details={
                    "character": self.character,
                    "pose": stored.name,
                    "file": str(image_target),
                },
            ) from exc
        try:
            path.write_text(
                yaml.safe_dump(stored.model_dump(mode="json"), sort_keys=False),
                encoding="utf-8",
            )
        except OSError as exc:
            raise PoseError(
                f"cannot store pose {path}: {exc}",
                details={
                    "character": self.character,
                    "pose": stored.name,
                    "file": str(path),
                },
            ) from exc
        return stored

    def load(self, name: str) -> Pose:
        """Load and validate *name*, confirming its cutout image exists on disk.

        Raises :class:`PoseNotFoundError` when no pose exists and
        :class:`PoseSchemaError` when the document is invalid or references a
        missing asset.
        """
        path = self.pose_path(name)
        if not path.is_file():
            raise PoseNotFoundError(
                f"unknown pose: {name}",
                details={
                    "character": self.character,
                    "name": name,
                    "file": str(path),
                },
            )
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise PoseError(
                f"cannot read pose {path}: {exc}",
                details={
                    "character": self.character,
                    "name": name,
                    "file": str(path),
                },
            ) from exc
        try:
            payload: Any = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise PoseError(
                f"invalid YAML in pose {path}: {exc}",
                details={
                    "character": self.character,
                    "name": name,
                    "file": str(path),
                },
            ) from exc
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            raise PoseSchemaError(
                f"pose root must be a mapping: {path}",
                details={
                    "character": self.character,
                    "name": name,
                    "file": str(path),
                    "field": "",
                },
            )
        try:
            pose = Pose.model_validate(dict(payload))
        except Exception as exc:
            raise PoseSchemaError(
                f"pose {path} is invalid: {exc}",
                details={
                    "character": self.character,
                    "name": name,
                    "file": str(path),
                    "field": "",
                },
            ) from exc
        image_target = self.poses_dir / pose.image
        if not image_target.is_file():
            raise PoseSchemaError(
                f"pose {path} references missing asset {pose.image!r}",
                details={
                    "character": self.character,
                    "name": name,
                    "file": str(path),
                    "field": "image",
                    "asset": pose.image,
                },
            )
        return pose

    def find_by_tag(self, tag: str) -> tuple[Pose, ...]:
        """Return every pose tagged with *tag*, sorted by name."""
        key = tag.strip()
        if not key:
            return ()
        matches: list[Pose] = []
        for name in self.names():
            try:
                pose = self.load(name)
            except PoseError:
                continue
            if key in pose.tags:
                matches.append(pose)
        matches.sort(key=lambda pose: pose.name)
        return tuple(matches)


def load_pose_library(
    data_dir: str | Path | None = None,
    *,
    character: str | None = None,
) -> PoseLibrary:
    """Return a :class:`PoseLibrary` for *data_dir* and optional *character*."""
    return PoseLibrary(data_dir=data_dir, character=character)
