"""Brand kits: reusable branding stored in the user data directory.

A brand kit bundles colours, fonts, a logo, a watermark, intro and outro
clips and a caption style. The kit is stored as a YAML file next to the
assets it references. Loading is plain validation only: no AI call and no
network request happens here. The loader confirms every referenced asset
exists on disk.

A character references a brand kit by name through ``Character.brand_kit``;
binding a kit to a run means resolving that name to a loaded :class:`BrandKit`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)
from pydantic import ValidationError as PydanticValidationError

from aivia.characters.errors import (
    BrandKitError,
    BrandKitNotFoundError,
    BrandKitSchemaError,
)
from aivia.core import paths

__all__ = [
    "CURRENT_BRAND_KIT_SCHEMA_VERSION",
    "BrandKit",
    "BrandKitError",
    "BrandKitLibrary",
    "BrandKitNotFoundError",
    "BrandKitSchemaError",
    "load_brand_kit_file",
]

CURRENT_BRAND_KIT_SCHEMA_VERSION = 1
"""The brand kit schema version written by this release."""

_BRAND_KITS_DIR = "brand_kits"


def _normalise_brand_names(value: Any, label: str) -> Any:
    """Coerce colours/fonts to a list of unique names or a name->value map.

    Blank entries are rejected, as are duplicate list entries. Dict keys
    (names) and values must both be non-empty.
    """
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if isinstance(value, Mapping):
        names: dict[str, str] = {}
        for key, entry in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"{label} names must be non-empty")
            if not isinstance(entry, str) or not entry.strip():
                raise ValueError(f"{label} values must be non-empty")
            names[key.strip()] = entry.strip()
        return names
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        entries: list[str] = []
        seen: set[str] = set()
        for entry in value:
            if not isinstance(entry, str) or not entry.strip():
                raise ValueError(f"{label} entries must be non-empty")
            name = entry.strip()
            if name in seen:
                raise ValueError(f"duplicate {label}: {name}")
            seen.add(name)
            entries.append(name)
        return entries
    return value


class BrandKit(BaseModel):
    """One reusable brand kit: colours, fonts, logo, watermark, clips, style.

    ``logo``, ``watermark``, ``intro`` and ``outro`` hold paths relative to
    the brand kit file's directory; the loader confirms they exist on disk.
    An empty kit is valid; binding it to a character or run is separate.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    name: str = Field(min_length=1)
    version: int = Field(default=CURRENT_BRAND_KIT_SCHEMA_VERSION, ge=1)
    colors: list[str] | dict[str, str] = Field(default_factory=list)
    fonts: list[str] | dict[str, str] = Field(default_factory=list)
    logo: str = ""
    watermark: str = ""
    intro: str = ""
    outro: str = ""
    caption_style: str = ""

    @field_validator("name")
    @classmethod
    def _strip_name(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("name must be non-empty")
        return text

    @field_validator(
        "logo", "watermark", "intro", "outro", "caption_style", mode="before"
    )
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("colors", "fonts")
    @classmethod
    def _normalise_branding(cls, value: Any, info: Any) -> Any:
        return _normalise_brand_names(value, info.field_name or "brand")

    @property
    def schema_version(self) -> int:
        """The schema version this kit was written against."""
        return self.version


class BrandKitLibrary:
    """Brand kit files in the user data directory with on-disk assets."""

    def __init__(
        self,
        *,
        data_dir: str | Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        if data_dir is None:
            self.data_dir = paths.data_dir(environ)
        else:
            self.data_dir = Path(data_dir).expanduser().resolve()
        self.brand_kits_dir = self.data_dir / _BRAND_KITS_DIR

    def brand_kit_path(self, name: str) -> Path:
        """Return the file for *name*: ``<data dir>/brand_kits/<name>.yaml``."""
        return self.brand_kits_dir / f"{self._normalise_name(name)}.yaml"

    @staticmethod
    def _normalise_name(name: str) -> str:
        if not isinstance(name, str) or not name.strip():
            raise BrandKitError("brand kit name must be a non-empty string")
        text = name.strip().replace("\\", "/")
        if any(part in {"", ".", ".."} for part in text.split("/")):
            raise BrandKitError(
                f"brand kit name may not escape the brand kits directory: {name}",
                details={"name": text},
            )
        return text

    def names(self) -> tuple[str, ...]:
        """Return every brand kit name with a file in the brand kits directory."""
        if not self.brand_kits_dir.is_dir():
            return ()
        names: list[str] = []
        for path in sorted(self.brand_kits_dir.glob("*.yaml")):
            names.append(path.stem)
        return tuple(names)

    def resolve(self, name: str) -> Path:
        """Return the brand kit file for *name*.

        Raises :class:`BrandKitNotFoundError`, naming the file, when it does
        not exist.
        """
        path = self.brand_kit_path(name)
        if not path.is_file():
            raise BrandKitNotFoundError(
                f"unknown brand kit: {name}",
                details={"name": name, "file": str(path)},
            )
        return path

    def load(self, name: str) -> BrandKit:
        """Load and validate *name*, confirming its assets exist on disk."""
        path = self.resolve(name)
        return load_brand_kit_file(path)

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        try:
            self.resolve(name)
        except BrandKitError:
            return False
        return True


def load_brand_kit_file(path: str | Path) -> BrandKit:
    """Read *path*, validate it and confirm its assets exist on disk.

    Raises :class:`~aivia.characters.errors.BrandKitSchemaError`, naming the
    file and field, when the document is invalid or references a missing asset.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BrandKitError(
            f"cannot read brand kit file {path}: {exc}",
            details={"file": str(path)},
        ) from exc
    try:
        payload: Any = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise BrandKitError(
            f"invalid YAML in brand kit file {path}: {exc}",
            details={"file": str(path)},
        ) from exc
    if payload is None:
        payload = {}
    if not isinstance(payload, Mapping):
        raise BrandKitError(
            f"brand kit root must be a mapping: {path}",
            details={"file": str(path)},
        )
    try:
        kit = BrandKit.model_validate(dict(payload))
    except PydanticValidationError as exc:
        raise _schema_error(exc, path) from exc
    _check_assets(kit, path)
    return kit


def _schema_error(exc: PydanticValidationError, path: Path) -> BrandKitSchemaError:
    errors = [dict(error) for error in exc.errors()]
    first = errors[0] if errors else {}
    loc = cast(tuple[object, ...], first.get("loc", ()) or ())
    field = ".".join(str(part) for part in loc)
    message = str(first.get("msg", "invalid value"))
    raise BrandKitSchemaError(
        f"brand kit file {path} has invalid {field or 'root'}: {message}",
        details={
            "file": str(path),
            "field": field,
            "errors": errors,
        },
    )


def _check_assets(kit: BrandKit, path: Path) -> None:
    base = path.parent
    for field, asset in (
        ("logo", kit.logo),
        ("watermark", kit.watermark),
        ("intro", kit.intro),
        ("outro", kit.outro),
    ):
        if not asset:
            continue
        if not (base / asset).is_file():
            raise BrandKitSchemaError(
                f"brand kit file {path} references missing asset {asset!r}",
                details={
                    "file": str(path),
                    "field": field,
                    "asset": asset,
                },
            )
