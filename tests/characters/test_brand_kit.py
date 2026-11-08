"""Brand kits: colours, fonts, logo, watermark, intro, outro and caption style.

A brand kit is a YAML file in ``<data dir>/brand_kits/<name>.yaml`` next to the
assets it references. Loading validates the kit and confirms every referenced
logo, watermark, intro and outro asset exists on disk, naming the file and
field in every error.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError as PydanticValidationError

from aivia.characters import (
    CURRENT_BRAND_KIT_SCHEMA_VERSION,
    BrandKit,
    BrandKitError,
    BrandKitLibrary,
    BrandKitNotFoundError,
    BrandKitSchemaError,
    load_brand_kit_file,
)
from aivia.core.errors import ValidationError

DEFAULT = {
    "name": "studio",
    "colors": ["#112233", "#abcdef"],
    "fonts": ["Lato", "Noto Sans"],
    "logo": "logo.png",
    "watermark": "watermark.png",
    "intro": "intro.mp4",
    "outro": "outro.mp4",
    "caption_style": "bold-outline",
}


def write_brand_kit(root: Path, name: str, payload: dict[str, Any]) -> Path:
    path = root / "brand_kits" / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def write_asset(root: Path, name: str, content: bytes = b"fake-asset") -> Path:
    path = root / "brand_kits" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


def test_brand_kit_schema_parses_every_field() -> None:
    kit = BrandKit.model_validate(DEFAULT)

    assert kit.name == "studio"
    assert kit.version == CURRENT_BRAND_KIT_SCHEMA_VERSION
    assert kit.schema_version == 1
    assert kit.colors == ["#112233", "#abcdef"]
    assert kit.fonts == ["Lato", "Noto Sans"]
    assert kit.logo == "logo.png"
    assert kit.watermark == "watermark.png"
    assert kit.intro == "intro.mp4"
    assert kit.outro == "outro.mp4"
    assert kit.caption_style == "bold-outline"


def test_brand_kit_schema_defaults_are_empty_and_versioned() -> None:
    kit = BrandKit.model_validate({"name": "studio"})

    assert kit.name == "studio"
    assert kit.version == CURRENT_BRAND_KIT_SCHEMA_VERSION
    assert kit.schema_version == 1
    assert kit.colors == []
    assert kit.fonts == []
    assert kit.logo == ""
    assert kit.watermark == ""
    assert kit.intro == ""
    assert kit.outro == ""
    assert kit.caption_style == ""


def test_brand_kit_schema_round_trips() -> None:
    kit = BrandKit.model_validate(DEFAULT)
    restored = BrandKit.model_validate(kit.model_dump(mode="json"))

    assert restored == kit


def test_brand_kit_schema_rejects_unknown_fields() -> None:
    with pytest.raises(PydanticValidationError) as caught:
        BrandKit.model_validate({**DEFAULT, "bogus_key": True})

    assert any(part == "bogus_key" for part in caught.value.errors()[0].get("loc", ()))


def test_brand_kit_schema_requires_a_name() -> None:
    with pytest.raises(PydanticValidationError):
        BrandKit.model_validate({})

    with pytest.raises(PydanticValidationError):
        BrandKit.model_validate({"name": "   "})


def test_brand_kit_schema_strips_text_fields() -> None:
    kit = BrandKit.model_validate(
        {
            "name": "  Studio  ",
            "logo": "  logo.png  ",
            "watermark": "  mark.png  ",
            "intro": "  intro.mp4  ",
            "outro": "  outro.mp4  ",
            "caption_style": "  bold-outline  ",
        }
    )

    assert kit.name == "Studio"
    assert kit.logo == "logo.png"
    assert kit.watermark == "mark.png"
    assert kit.intro == "intro.mp4"
    assert kit.outro == "outro.mp4"
    assert kit.caption_style == "bold-outline"


def test_brand_kit_schema_accepts_named_colors_and_fonts() -> None:
    kit = BrandKit.model_validate(
        {
            "name": "studio",
            "colors": {"primary": "#112233", "accent": "#abcdef"},
            "fonts": {"heading": "Lato", "body": "Noto Sans"},
        }
    )

    assert kit.colors == {"primary": "#112233", "accent": "#abcdef"}
    assert kit.fonts == {"heading": "Lato", "body": "Noto Sans"}


def test_brand_kit_schema_accepts_list_colors_and_fonts() -> None:
    kit = BrandKit.model_validate(
        {
            "name": "studio",
            "colors": ["#112233", "#abcdef"],
            "fonts": ["Lato", "Noto Sans"],
        }
    )

    assert kit.colors == ["#112233", "#abcdef"]
    assert kit.fonts == ["Lato", "Noto Sans"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("colors", [""], "colors entries must be non-empty"),
        ("fonts", ["Lato", " "], "fonts entries must be non-empty"),
        ("colors", ["#111111", "#111111"], "duplicate colors"),
        ("fonts", {"": "Lato"}, "fonts names must be non-empty"),
        ("colors", {"primary": ""}, "colors values must be non-empty"),
    ],
)
def test_brand_kit_schema_rejects_blank_or_duplicate_branding_values(
    field: str, value: object, message: str
) -> None:
    with pytest.raises(PydanticValidationError) as caught:
        BrandKit.model_validate({**DEFAULT, field: value})

    assert message in str(caught.value)


def test_brand_kit_schema_rejects_version_below_one() -> None:
    with pytest.raises(PydanticValidationError):
        BrandKit.model_validate({**DEFAULT, "version": 0})

    with pytest.raises(PydanticValidationError):
        BrandKit.model_validate({**DEFAULT, "version": -1})


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------


def test_loader_loads_a_brand_kit_from_the_user_directory(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    write_brand_kit(data_dir, "studio", DEFAULT)
    for asset in ("logo.png", "watermark.png", "intro.mp4", "outro.mp4"):
        write_asset(data_dir, asset)

    library = BrandKitLibrary(data_dir=data_dir)
    kit = library.load("studio")

    assert kit.name == "studio"
    assert kit.colors == ["#112233", "#abcdef"]
    assert kit.fonts == ["Lato", "Noto Sans"]
    assert kit.logo == "logo.png"
    assert kit.watermark == "watermark.png"
    assert kit.intro == "intro.mp4"
    assert kit.outro == "outro.mp4"
    assert kit.caption_style == "bold-outline"
    assert "studio" in library
    assert library.names() == ("studio",)


def test_loader_load_brand_kit_file_directly(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    path = write_brand_kit(data_dir, "studio", DEFAULT)
    for asset in ("logo.png", "watermark.png", "intro.mp4", "outro.mp4"):
        write_asset(data_dir, asset)

    kit = load_brand_kit_file(path)

    assert kit.name == "studio"
    assert kit.colors == ["#112233", "#abcdef"]
    assert kit.fonts == ["Lato", "Noto Sans"]


def test_loader_unknown_brand_kit_names_the_file(tmp_path: Path) -> None:
    library = BrandKitLibrary(data_dir=tmp_path / "data")

    with pytest.raises(BrandKitNotFoundError) as caught:
        library.load("missing")

    assert caught.value.code == "brand_kit_not_found"
    assert caught.value.details["name"] == "missing"
    assert "brand_kits" in caught.value.details["file"]
    assert caught.value.details["file"].endswith("brand_kits/missing.yaml")
    assert "unknown brand kit: missing" in str(caught.value)


def test_loader_invalid_yaml_names_the_file(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    path = write_brand_kit(data_dir, "studio", DEFAULT)
    path.write_text("name: studio\ncolors: [unclosed\n", encoding="utf-8")

    library = BrandKitLibrary(data_dir=data_dir)

    with pytest.raises(BrandKitError) as caught:
        library.load("studio")

    assert caught.value.code == "brand_kit"
    assert caught.value.details["file"] == str(path)
    assert "invalid YAML" in str(caught.value)


def test_loader_missing_required_field_names_file_and_field(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    path = write_brand_kit(data_dir, "studio", {"colors": ["#112233"]})

    library = BrandKitLibrary(data_dir=data_dir)

    with pytest.raises(BrandKitSchemaError) as caught:
        library.load("studio")

    assert caught.value.code == "brand_kit_schema"
    assert isinstance(caught.value, ValidationError)
    assert caught.value.details["file"] == str(path)
    assert caught.value.details["field"] == "name"
    assert "name" in str(caught.value)
    assert str(path) in str(caught.value)


def test_loader_unknown_key_names_file_and_field(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    payload: dict[str, Any] = dict(DEFAULT)
    payload["bogus_key"] = True
    path = write_brand_kit(data_dir, "studio", payload)

    library = BrandKitLibrary(data_dir=data_dir)

    with pytest.raises(BrandKitSchemaError) as caught:
        library.load("studio")

    assert caught.value.details["file"] == str(path)
    assert caught.value.details["field"] == "bogus_key"
    assert "bogus_key" in str(caught.value)


@pytest.mark.parametrize(
    ("field", "asset"),
    [
        ("logo", "logo.png"),
        ("watermark", "watermark.png"),
        ("intro", "intro.mp4"),
        ("outro", "outro.mp4"),
    ],
)
def test_loader_missing_asset_names_file_and_field(
    tmp_path: Path, field: str, asset: str
) -> None:
    data_dir = tmp_path / "data"
    payload = {
        "name": "studio",
        "logo": "",
        "watermark": "",
        "intro": "",
        "outro": "",
    }
    payload[field] = asset
    path = write_brand_kit(data_dir, "studio", payload)

    library = BrandKitLibrary(data_dir=data_dir)

    with pytest.raises(BrandKitSchemaError) as caught:
        library.load("studio")

    assert caught.value.details["file"] == str(path)
    assert caught.value.details["field"] == field
    assert caught.value.details["asset"] == asset
    assert asset in str(caught.value)


def test_loader_assets_are_resolved_relative_to_the_brand_kit_file(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    sub = data_dir / "brand_kits" / "series"
    sub.mkdir(parents=True)
    kit_file = sub / "studio.yaml"
    kit_file.write_text(
        yaml.safe_dump(
            {
                **DEFAULT,
                "logo": "logo.png",
                "watermark": "",
                "intro": "",
                "outro": "",
            }
        ),
        encoding="utf-8",
    )
    (sub / "logo.png").write_bytes(b"asset")

    library = BrandKitLibrary(data_dir=data_dir)
    kit = library.load("series/studio")

    assert kit.name == "studio"
    assert kit.logo == "logo.png"


def test_loader_brand_kit_name_cannot_escape_the_directory(tmp_path: Path) -> None:
    library = BrandKitLibrary(data_dir=tmp_path / "data")

    with pytest.raises(BrandKitError):
        library.load("../secret")
    with pytest.raises(BrandKitError):
        library.load("/etc/passwd")


def test_loader_default_uses_the_user_data_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    library = BrandKitLibrary()

    assert library.data_dir == tmp_path / "aivia" / "data"
    assert library.brand_kits_dir == tmp_path / "aivia" / "data" / "brand_kits"
