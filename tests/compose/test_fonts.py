"""Font resolution: bundled families, faces and language fallback chains."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageChops

from aivia.compose import (
    Composition,
    FontError,
    FontManager,
    FontStyle,
    PillowStillRenderer,
    RenderError,
    TextStyle,
    TextWeight,
    available_fonts,
    font_manager,
    resolve_font,
)
from aivia.core.config import LanguagesConfig, RenderConfig
from aivia.core.paths import package_fonts_dir

_SUFFIXES = {
    "regular": "Regular",
    "bold": "Bold",
    "italic": "Italic",
    "bolditalic": "BoldItalic",
}


def _family(
    root: Path,
    name: str,
    styles: tuple[str, ...] = ("regular",),
) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    for style in styles:
        (directory / f"{name}-{_SUFFIXES[style]}.ttf").write_bytes(b"font")
    return directory


def _copy_lato(root: Path, name: str = "Custom") -> Path:
    source = package_fonts_dir() / "lato"
    directory = root / name
    directory.mkdir(parents=True)
    for style in _SUFFIXES:
        original = source / f"Lato-{_SUFFIXES[style]}.ttf"
        (directory / f"{name}-{_SUFFIXES[style]}.ttf").write_bytes(
            original.read_bytes()
        )
    return directory


def test_bundled_families_and_faces_resolve_case_insensitively() -> None:
    manager = FontManager()

    assert manager.available_families == ("Lato",)
    assert manager.resolve("Lato").name == "Lato-Regular.ttf"
    assert manager.resolve("lato", weight=TextWeight.BOLD).name == "Lato-Bold.ttf"
    assert (
        manager.resolve("LATO", style=TextStyle.ITALIC).name == "Lato-Italic.ttf"
    )
    assert (
        manager.resolve("Lato", weight="semibold", style="italic").name
        == "Lato-BoldItalic.ttf"
    )
    assert manager.resolve("Lato-Bold").name == "Lato-Bold.ttf"
    assert (
        manager.select("Lato-BoldItalic").style is FontStyle.BOLDITALIC
    )


def test_explicit_font_file_wins_over_the_bundled_directory(tmp_path: Path) -> None:
    manager = FontManager()
    explicit = package_fonts_dir() / "lato" / "Lato-Bold.ttf"

    face = manager.select(str(explicit))

    assert manager.resolve(str(explicit)) == explicit
    assert face.path == explicit
    assert face.bundled is False


def test_a_missing_bundled_family_never_falls_back_to_a_system_font() -> None:
    manager = FontManager()

    with pytest.raises(FontError, match="system fallback is disabled") as caught:
        manager.resolve("Missing")

    assert isinstance(caught.value, RenderError)
    assert caught.value.details["family"] == "Missing"
    assert caught.value.details["require_bundled"] is True
    assert "Lato" in caught.value.details["available"]


def test_a_missing_font_directory_reports_bundled_fonts_unavailable(
    tmp_path: Path,
) -> None:
    manager = FontManager(tmp_path / "absent")

    with pytest.raises(FontError, match="bundled fonts are not available"):
        manager.resolve("Lato")


def test_missing_face_falls_through_the_language_chain(tmp_path: Path) -> None:
    _family(tmp_path, "Primary", ("regular",))
    _family(tmp_path, "Secondary", ("bold",))
    manager = FontManager(
        tmp_path,
        default_family="Primary",
        fallback_families=["Secondary"],
        language_fallbacks={"tr": ["Primary", "Secondary"]},
    )

    assert manager.resolve(language="tr").name == "Primary-Regular.ttf"
    assert (
        manager.resolve(language="tr", weight="bold").name == "Secondary-Bold.ttf"
    )
    assert (
        manager.resolve(language="tr-TR", weight="bold").name == "Secondary-Bold.ttf"
    )


def test_per_language_chain_precedes_the_global_and_default_fallbacks(
    tmp_path: Path,
) -> None:
    for name in ("Turkish", "Global", "Default"):
        _family(tmp_path, name)
    manager = FontManager(
        tmp_path,
        default_family="Default",
        fallback_families=["Global"],
        language_fallbacks={"tr": ["Turkish"]},
    )

    assert manager.resolve(language="tr").name == "Turkish-Regular.ttf"
    assert manager.resolve(language="tr-TR").name == "Turkish-Regular.ttf"
    assert manager.resolve(language="de").name == "Global-Regular.ttf"
    assert manager.resolve().name == "Default-Regular.ttf"


def test_language_config_contributes_the_subtitle_font_to_each_chain(
    tmp_path: Path,
) -> None:
    _family(tmp_path, "Turkish")
    _family(tmp_path, "Lato")
    render = RenderConfig.from_mapping(
        {
            "default_font": "Lato",
            "fallback_fonts": ["Lato"],
            "require_bundled_fonts": True,
        }
    )
    languages = LanguagesConfig.from_mapping(
        {
            "default": "en",
            "languages": {
                "en": {"subtitle_font": "Lato"},
                "tr": {"subtitle_font": "Turkish"},
            },
        }
    )

    manager = FontManager.from_config(render, languages, fonts_dir=tmp_path)

    assert manager.language_fallbacks["tr"] == ("Turkish",)
    assert manager.resolve(language="tr").name == "Turkish-Regular.ttf"
    assert manager.resolve(language="en").name == "Lato-Regular.ttf"


def test_render_config_controls_defaults_and_system_fallback(tmp_path: Path) -> None:
    system = tmp_path / "System-Regular.ttf"
    system.write_bytes(b"font")
    render = RenderConfig.from_mapping(
        {
            "default_font": "Missing",
            "fallback_fonts": ["Missing"],
            "require_bundled_fonts": False,
        }
    )

    manager = FontManager.from_config(render, system_fallback=system)

    assert manager.require_bundled is False
    assert manager.resolve() == system
    assert manager.select().bundled is False


def test_requiring_bundled_fonts_rejects_a_configured_system_fallback(
    tmp_path: Path,
) -> None:
    system = tmp_path / "System-Regular.ttf"
    system.write_bytes(b"font")
    manager = FontManager(
        tmp_path / "absent",
        system_fallback=system,
        require_bundled_fonts=True,
    )

    with pytest.raises(FontError, match="system fallback is disabled"):
        manager.resolve("Missing")


@pytest.mark.parametrize("weight", ["ultra", 0, 1001, True])
def test_an_invalid_weight_is_a_font_error(weight: object) -> None:
    with pytest.raises(FontError, match="font weight"):
        FontManager().resolve("Lato", weight=weight)


@pytest.mark.parametrize("style", ["slanted", 5000])
def test_an_invalid_style_is_a_font_error(style: object) -> None:
    with pytest.raises(FontError, match="font style"):
        FontManager().resolve("Lato", style=style)


def test_module_helpers_resolve_and_list_families() -> None:
    manager = FontManager()

    assert resolve_font("Lato").name == "Lato-Regular.ttf"
    assert resolve_font(manager, "Lato").name == "Lato-Regular.ttf"
    assert font_manager(manager) is manager
    assert available_fonts(manager) == ("Lato",)


def test_the_renderer_accepts_a_custom_font_manager(tmp_path: Path) -> None:
    _copy_lato(tmp_path)
    manager = FontManager(tmp_path)
    composition = {
        "canvas": {"width": 120, "height": 50, "background": "#000000"},
        "duration_seconds": 1.0,
        "layers": [
            {
                "type": "text",
                "text": "Hello",
                "font_family": "Custom",
                "font_size": 32,
                "color": "#ff0000",
            }
        ],
    }

    image = PillowStillRenderer(fonts=manager).render_still(
        Composition.model_validate(composition),
        0.0,
    )

    difference = ImageChops.difference(
        image.convert("RGB"),
        Image.new("RGB", image.size, (0, 0, 0)),
    )
    assert difference.getbbox() is not None
