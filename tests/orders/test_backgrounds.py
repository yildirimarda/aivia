"""A tagged background library with static, animated and AI loop modes.

The library indexes local backgrounds by mode and tag, selects them
deterministically and verifies loop seamlessness by comparing the first and
last frames of animated containers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from PIL import Image

from aivia.compose.loop_outputs import measure_loop_seam
from aivia.core.errors import ConfigError
from aivia.media.background_library import (
    BACKGROUND_LIBRARY_SCHEMA_VERSION,
    BackgroundLibrary,
    BackgroundLibraryEntry,
    BackgroundLibraryError,
    BackgroundLibraryMatch,
    BackgroundMode,
    background_library,
    build_background_library,
    load_background_library,
    select_background,
)

CC0 = {
    "name": "CC0-1.0",
    "url": "https://creativecommons.org/publicdomain/zero/1.0/",
    "commercial_use": True,
}
CC_BY = {
    "name": "CC BY 4.0",
    "url": "https://creativecommons.org/licenses/by/4.0/",
    "commercial_use": True,
    "attribution_required": True,
    "attribution": "Background by Example",
}
NON_COMMERCIAL = {
    "name": "Test Non Commercial",
    "url": "https://example.com/non-commercial",
    "commercial_use": False,
}


def background(
    entry_id: str,
    *,
    mode: str | BackgroundMode = BackgroundMode.STATIC,
    tags: tuple[str, ...] = (),
    duration_seconds: float | None = None,
    seam_similarity: float | None = None,
    seam_verified: bool = False,
    license: dict[str, Any] | str = CC0,
    path: str | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": entry_id,
        "title": entry_id.replace("-", " ").title(),
        "path": path or f"{entry_id}.png",
        "mode": mode,
        "tags": list(tags),
        "license": license,
    }
    if duration_seconds is not None:
        data["duration_seconds"] = duration_seconds
    if seam_similarity is not None:
        data["seam_similarity"] = seam_similarity
    if seam_verified:
        data["seam_verified"] = True
    return data


BACKGROUNDS = [
    background(
        "calm-sky",
        mode=BackgroundMode.STATIC,
        tags=("calm", "sky"),
        duration_seconds=None,
    ),
    background(
        "calm-mountain",
        mode=BackgroundMode.STATIC,
        tags=("calm", "nature"),
        license=CC_BY,
    ),
    background(
        "energetic-loop",
        mode=BackgroundMode.ANIMATED,
        tags=("energetic", "loop"),
        duration_seconds=5.0,
    ),
    background(
        "calm-loop",
        mode=BackgroundMode.ANIMATED,
        tags=("calm", "loop"),
        duration_seconds=6.0,
    ),
    background(
        "dark-cinematic",
        mode=BackgroundMode.STATIC,
        tags=("dark", "cinematic"),
        duration_seconds=None,
        license=NON_COMMERCIAL,
    ),
    background(
        "ai-sunrise",
        mode=BackgroundMode.AI_LOOP,
        tags=("ai", "sunrise", "loop"),
        duration_seconds=8.0,
        seam_similarity=0.99,
        seam_verified=True,
    ),
    background(
        "ai-unverified",
        mode=BackgroundMode.AI_LOOP,
        tags=("ai", "loop"),
        duration_seconds=8.0,
        seam_verified=False,
    ),
]


@pytest.fixture
def library() -> BackgroundLibrary:
    return BackgroundLibrary.from_mapping(
        {
            "schema_version": BACKGROUND_LIBRARY_SCHEMA_VERSION,
            "backgrounds": BACKGROUNDS,
        }
    )


def test_entries_are_indexed_by_mode_and_tag(library: BackgroundLibrary) -> None:
    assert library.ids == (
        "ai-sunrise",
        "ai-unverified",
        "calm-loop",
        "calm-mountain",
        "calm-sky",
        "dark-cinematic",
        "energetic-loop",
    )
    assert library.modes == (
        BackgroundMode.AI_LOOP,
        BackgroundMode.ANIMATED,
        BackgroundMode.STATIC,
    )
    assert library.tags == (
        "ai",
        "calm",
        "cinematic",
        "dark",
        "energetic",
        "loop",
        "nature",
        "sky",
        "sunrise",
    )

    assert [entry.id for entry in library.by_mode[BackgroundMode.STATIC]] == [
        "calm-mountain",
        "calm-sky",
        "dark-cinematic",
    ]
    assert [entry.id for entry in library.by_mode[BackgroundMode.ANIMATED]] == [
        "calm-loop",
        "energetic-loop",
    ]
    assert [entry.id for entry in library.by_mode[BackgroundMode.AI_LOOP]] == [
        "ai-sunrise",
        "ai-unverified",
    ]
    assert [entry.id for entry in library.by_tag["loop"]] == [
        "ai-sunrise",
        "ai-unverified",
        "calm-loop",
        "energetic-loop",
    ]
    assert [entry.id for entry in library.by_tag["calm"]] == [
        "calm-loop",
        "calm-mountain",
        "calm-sky",
    ]
    assert len(library) == 7
    assert library.get("calm-sky") is not None
    assert library.get("missing") is None
    assert "calm-sky" in library
    assert library.entries[0].id == "ai-sunrise"
    assert [entry.id for entry in library] == list(library.ids)


def test_entry_labels_tags_and_mode_are_normalised() -> None:
    entry = BackgroundLibraryEntry.model_validate(
        {
            "id": "  one  ",
            "title": "  One  ",
            "path": "one.png",
            "mode": " Animated ",
            "tags": [" Loop ", "CALM", "loop"],
            "license": CC0,
        }
    )

    assert entry.id == "one"
    assert entry.title == "One"
    assert entry.mode is BackgroundMode.ANIMATED
    assert entry.tags == ("calm", "loop")
    assert entry.is_loop is True
    assert entry.license.name == "CC0-1.0"
    assert entry.is_commercial is True
    assert entry.is_cleared is True
    image_format = entry.image_format
    assert image_format is not None
    assert image_format.value == "png"


def test_static_has_no_loop_and_ai_loop_is_a_loop() -> None:
    static = BackgroundLibraryEntry.model_validate(
        {"id": "s", "title": "S", "path": "s.png", "mode": "static", "license": CC0}
    )
    animated = BackgroundLibraryEntry.model_validate(
        {"id": "a", "title": "A", "path": "a.gif", "mode": "animated", "license": CC0}
    )
    ai_loop = BackgroundLibraryEntry.model_validate(
        {"id": "l", "title": "L", "path": "l.gif", "mode": "ai_loop", "license": CC0}
    )

    assert static.is_loop is False
    assert animated.is_loop is True
    assert ai_loop.is_loop is True


def test_selection_by_tags_and_duration_picks_a_matching_background(
    library: BackgroundLibrary,
) -> None:
    entry = library.select(
        ("loop", "calm"),
        mode=BackgroundMode.ANIMATED,
        duration_seconds=6.0,
        duration_tolerance_seconds=0.5,
    )

    assert entry.id == "calm-loop"
    assert entry.tags == ("calm", "loop")

    with pytest.raises(BackgroundLibraryError) as missing:
        library.select(["loop"], duration_seconds=60.0, duration_tolerance_seconds=1)
    assert missing.value.code == "background_library"
    assert missing.value.details["tags"] == ["loop"]

    assert library.select("calm,loop").id == "calm-loop"


def test_tags_match_any_by_default_and_all_on_request(
    library: BackgroundLibrary,
) -> None:
    any_match = library.find(["nature", "energetic"])
    assert [match.id for match in any_match] == ["calm-mountain", "energetic-loop"]

    assert library.find(["nature", "energetic"], match_all_tags=True) == ()


def test_selection_filters_mode_tags_license_and_commercial_use(
    library: BackgroundLibrary,
) -> None:
    assert library.select(mode="static", tags="calm").id == "calm-mountain"
    assert library.select(mode="animated", tags="loop", license="cc0-1.0").id in {
        "calm-loop",
        "energetic-loop",
    }
    assert library.select(mode="ai_loop", tags="sunrise").id == "ai-sunrise"

    commercial = library.find(commercial_only=True)
    assert "dark-cinematic" not in [match.id for match in commercial]

    with pytest.raises(BackgroundLibraryError):
        library.select(license="unknown")


def test_the_closest_duration_wins_a_tie_on_tags() -> None:
    entries = [
        background("short", duration_seconds=30.0, tags=("loop",)),
        background("long", duration_seconds=31.0, tags=("loop",)),
    ]
    library = BackgroundLibrary(entries)

    assert (
        library.select(
            ["loop"], duration_seconds=30.1, duration_tolerance_seconds=1.0
        ).id
        == "short"
    )
    assert (
        library.select(
            ["loop"], duration_seconds=30.9, duration_tolerance_seconds=1.0
        ).id
        == "long"
    )


def test_matches_record_score_tags_and_reasons(library: BackgroundLibrary) -> None:
    match = library.select_match(
        ["calm"],
        mode="animated",
        duration_seconds=6.0,
    )

    assert isinstance(match, BackgroundLibraryMatch)
    assert match.id == "calm-loop"
    assert match.matched_tags == ("calm",)
    assert "mode:animated" in match.reasons
    assert "duration" in match.reasons
    assert match.score > 0
    assert match.duration_distance_seconds == 0
    assert match.license.name == "CC0-1.0"
    assert match.to_dict()["entry"]["id"] == "calm-loop"


def test_select_many_for_duration_and_aliases(library: BackgroundLibrary) -> None:
    assert [entry.id for entry in library.select_many(["loop"])] == [
        "ai-sunrise",
        "ai-unverified",
        "calm-loop",
        "energetic-loop",
    ]
    assert library.for_duration(6.0, ["loop"]).id == "calm-loop"
    assert library.choose(["loop"]).id == "ai-sunrise"
    assert library.matches(["loop"])[0].id == "ai-sunrise"


def test_limit_and_invalid_filter_values(library: BackgroundLibrary) -> None:
    assert len(library.find(limit=1)) == 1

    with pytest.raises(ValueError):
        library.find(limit=0)
    with pytest.raises(TypeError):
        library.find(limit=True)
    with pytest.raises(ValueError):
        library.find(min_duration_seconds=10, max_duration_seconds=5)
    with pytest.raises(ValueError):
        library.find(min_seam_similarity=1.1)
    with pytest.raises(TypeError):
        library.find(duration_seconds="long")  # type: ignore[arg-type]


def test_an_empty_library_reports_that_nothing_matched() -> None:
    library = BackgroundLibrary()

    assert len(library) == 0
    with pytest.raises(BackgroundLibraryError) as missing:
        library.select(["calm"])
    assert missing.value.details["count"] == 0


def test_duplicate_ids_and_remote_paths_are_rejected() -> None:
    with pytest.raises(ConfigError):
        BackgroundLibrary([background("same"), background("same")])

    with pytest.raises(ValueError):
        BackgroundLibraryEntry.model_validate(
            background("remote", path="https://x.test/a.png")
        )


def test_unsupported_mode_is_rejected() -> None:
    with pytest.raises(ValueError):
        BackgroundLibraryEntry.model_validate(background("bad", mode="video"))


def _animated(tmp_path: Path, name: str, *colors: tuple[int, int, int]) -> Path:
    path = tmp_path / name
    frames = [Image.new("RGB", (40, 30), color) for color in colors]
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
    )
    return path


def test_animated_palmindrome_closes_the_seam(tmp_path: Path) -> None:
    palindrome = _animated(
        tmp_path, "palindrome.gif", (255, 0, 0), (0, 0, 255), (255, 0, 0)
    )
    jumping = _animated(tmp_path, "jumping.gif", (255, 0, 0), (0, 0, 255), (0, 255, 0))

    assert measure_loop_seam(palindrome) == pytest.approx(1.0)
    assert measure_loop_seam(jumping) < 0.5


def test_verify_seam_marks_a_palindrome_verified_and_rejects_a_jump(
    tmp_path: Path,
) -> None:
    palindrome = _animated(
        tmp_path, "palindrome.gif", (255, 0, 0), (0, 0, 255), (255, 0, 0)
    )
    jumping = _animated(tmp_path, "jumping.gif", (255, 0, 0), (0, 0, 255), (0, 255, 0))

    library = BackgroundLibrary(
        [
            background("pal", mode="animated", path=str(palindrome)),
            background("jump", mode="animated", path=str(jumping)),
        ]
    )

    pal_match = library.verify_seam("pal")
    assert pal_match.seam_verified is True
    assert pal_match.seam_similarity == pytest.approx(1.0)

    jump_match = library.verify_seam("jump")
    assert jump_match.seam_verified is False
    assert jump_match.seam_similarity is not None
    assert jump_match.seam_similarity < 0.5


def test_verify_seam_on_static_is_trivially_verified() -> None:
    library = BackgroundLibrary([background("still", mode="static")])

    match = library.verify_seam("still")
    assert match.seam_verified is True
    assert match.seam_similarity is None


def test_verified_only_excludes_unverified_loops(tmp_path: Path) -> None:
    palindrome = _animated(
        tmp_path, "palindrome.gif", (255, 0, 0), (0, 0, 255), (255, 0, 0)
    )
    library = BackgroundLibrary(
        [
            background("pal", mode="animated", path=str(palindrome)),
            background("jump", mode="animated", path=str(palindrome)),
        ]
    )

    library.verify_seam("pal")

    verified = library.find(verified_only=True)
    assert [match.id for match in verified] == ["pal"]

    all_loops = library.find(mode="animated")
    assert [match.id for match in all_loops] == ["jump", "pal"]


def test_min_seam_similarity_filters_loops(tmp_path: Path) -> None:
    palindrome = _animated(
        tmp_path, "palindrome.gif", (255, 0, 0), (0, 0, 255), (255, 0, 0)
    )
    jumping = _animated(tmp_path, "jumping.gif", (255, 0, 0), (0, 0, 255), (0, 255, 0))
    library = BackgroundLibrary(
        [
            background("pal", mode="animated", path=str(palindrome)),
            background("jump", mode="animated", path=str(jumping)),
        ]
    )
    library.verify_seam("pal")
    library.verify_seam("jump")

    strong = library.find(mode="animated", min_seam_similarity=0.9)
    assert [match.id for match in strong] == ["pal"]

    weak = library.find(mode="animated", min_seam_similarity=0.0)
    assert [match.id for match in weak] == ["jump", "pal"]


def test_ai_loop_selection_requires_a_verified_seam(tmp_path: Path) -> None:
    palindrome = _animated(
        tmp_path, "palindrome.gif", (255, 0, 0), (0, 0, 255), (255, 0, 0)
    )
    library = BackgroundLibrary(
        [
            background("ai-verified", mode="ai_loop", path=str(palindrome)),
            background("ai-unverified", mode="ai_loop", path=str(palindrome)),
        ]
    )
    library.verify_seam("ai-verified")

    assert library.select(mode="ai_loop", verified_only=True).id == "ai-verified"

    with pytest.raises(BackgroundLibraryError):
        library.select(mode="ai_loop", tags="missing")


def test_a_yaml_manifest_resolves_relative_paths(tmp_path: Path) -> None:
    manifest = tmp_path / "library.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": BACKGROUND_LIBRARY_SCHEMA_VERSION,
                "backgrounds": [background("one", mode="static", path="bg/one.png")],
            }
        ),
        encoding="utf-8",
    )

    library = BackgroundLibrary.from_file(manifest)

    assert library.ids == ("one",)
    assert Path(library.entries[0].path) == (tmp_path / "bg" / "one.png").resolve()
    assert load_background_library(manifest).ids == ("one",)


def test_a_json_manifest_and_an_id_keyed_mapping_load(tmp_path: Path) -> None:
    manifest = tmp_path / "library.json"
    manifest.write_text(json.dumps({"entries": [background("one")]}), encoding="utf-8")
    assert BackgroundLibrary.from_file(manifest).ids == ("one",)

    library = BackgroundLibrary.from_mapping(
        {"one": {key: value for key, value in background("one").items() if key != "id"}}
    )
    assert library.ids == ("one",)

    single = BackgroundLibrary.from_mapping(background("single"))
    assert single.ids == ("single",)


def test_manifest_errors_are_clear(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        BackgroundLibrary.from_file(tmp_path / "missing.yaml")

    unsupported = tmp_path / "library.txt"
    unsupported.write_text("not a manifest", encoding="utf-8")
    with pytest.raises(ConfigError):
        BackgroundLibrary.from_file(unsupported)

    invalid = tmp_path / "library.yaml"
    invalid.write_text("backgrounds: not-a-list\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        BackgroundLibrary.from_file(invalid)

    with pytest.raises(ConfigError):
        BackgroundLibrary.from_mapping({"backgrounds": "not-a-list"})


def test_module_level_helpers_load_and_select(library: BackgroundLibrary) -> None:
    assert load_background_library(library) is library
    assert build_background_library is load_background_library
    assert background_library is load_background_library

    entry = select_background(
        {"backgrounds": BACKGROUNDS}, ["loop"], mode="animated", duration_seconds=6.0
    )
    assert entry.id == "calm-loop"
