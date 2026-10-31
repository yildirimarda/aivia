"""The local music library indexes tracks and selects by tags and duration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from aivia.core.errors import ConfigError
from aivia.media.music_library import (
    DEFAULT_TEMPO_BUCKET_BPM,
    DEFAULT_TEMPO_TOLERANCE_BPM,
    LIBRARY_SCHEMA_VERSION,
    MusicLibrary,
    MusicLibraryEntry,
    MusicLibraryError,
    MusicLibraryMatch,
    MusicLibraryTrack,
    MusicSelection,
    build_music_library,
    load_music_library,
    music_library,
    select_music,
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
    "attribution": "Music by Example",
}
NON_COMMERCIAL = {
    "name": "Test Non Commercial",
    "url": "https://example.com/non-commercial",
    "commercial_use": False,
}


def track(
    entry_id: str,
    *,
    mood: str = "calm",
    genre: str = "ambient",
    tempo_bpm: float = 90.0,
    duration_seconds: float = 30.0,
    tags: tuple[str, ...] = (),
    license: dict[str, Any] | str = CC0,
    path: str | None = None,
) -> dict[str, Any]:
    return {
        "id": entry_id,
        "title": entry_id.replace("-", " ").title(),
        "path": path or f"{entry_id}.wav",
        "duration_seconds": duration_seconds,
        "mood": mood,
        "genre": genre,
        "tempo_bpm": tempo_bpm,
        "license": license,
        "tags": list(tags),
    }


TRACKS = [
    track(
        "calm-ambient",
        mood="calm",
        genre="ambient",
        tempo_bpm=90.0,
        duration_seconds=30.0,
        tags=("calm", "loop"),
    ),
    track(
        "calm-piano",
        mood="calm",
        genre="piano",
        tempo_bpm=80.0,
        duration_seconds=60.0,
        tags=("calm", "piano"),
        license=CC_BY,
    ),
    track(
        "upbeat-pop",
        mood="upbeat",
        genre="pop",
        tempo_bpm=120.0,
        duration_seconds=30.0,
        tags=("energetic", "loop"),
    ),
    track(
        "dark-cinematic",
        mood="dark",
        genre="cinematic",
        tempo_bpm=70.0,
        duration_seconds=45.0,
        tags=("dark",),
        license=NON_COMMERCIAL,
    ),
]


@pytest.fixture
def library() -> MusicLibrary:
    return MusicLibrary.from_mapping(
        {"schema_version": LIBRARY_SCHEMA_VERSION, "tracks": TRACKS}
    )


def test_entries_are_indexed_by_mood_tempo_genre_and_license(
    library: MusicLibrary,
) -> None:
    assert library.ids == (
        "calm-ambient",
        "calm-piano",
        "dark-cinematic",
        "upbeat-pop",
    )
    assert library.moods == ("calm", "dark", "upbeat")
    assert library.genres == ("ambient", "cinematic", "piano", "pop")
    assert library.licenses == ("cc by 4.0", "cc0-1.0", "test non commercial")
    assert library.tags == ("calm", "dark", "energetic", "loop", "piano")
    assert library.tempos == (70.0, 80.0, 90.0, 120.0)

    assert [entry.id for entry in library.by_mood["calm"]] == [
        "calm-ambient",
        "calm-piano",
    ]
    assert [entry.id for entry in library.by_genre["pop"]] == ["upbeat-pop"]
    assert [entry.id for entry in library.by_tag["loop"]] == [
        "calm-ambient",
        "upbeat-pop",
    ]
    assert [entry.id for entry in library.by_license["cc0-1.0"]] == [
        "calm-ambient",
        "upbeat-pop",
    ]
    assert [entry.id for entry in library.by_tempo[90]] == ["calm-ambient"]
    assert [entry.id for entry in library.by_tempo_bucket[90.0]] == ["calm-ambient"]
    assert library.tempo_bucket_bpm == DEFAULT_TEMPO_BUCKET_BPM

    assert len(library) == 4
    assert library.get("calm-piano") is not None
    assert library.get("missing") is None
    assert "calm-ambient" in library
    assert library.entries[0].id == "calm-ambient"
    assert [entry.id for entry in library] == list(library.ids)
    assert MusicLibraryTrack is MusicLibraryEntry
    assert MusicSelection is MusicLibraryMatch


def test_entry_labels_and_tags_are_normalised() -> None:
    entry = MusicLibraryEntry.model_validate(
        {
            "id": "  one  ",
            "title": "  One  ",
            "path": "one.wav",
            "duration_seconds": 12,
            "mood": " Calm ",
            "genre": " AMBIENT ",
            "tempo_bpm": 90,
            "license": "CC0-1.0",
            "tags": [" Loop ", "CALM", "loop"],
            "artist": "  Artist  ",
        }
    )

    assert entry.id == "one"
    assert entry.title == "One"
    assert entry.mood == "calm"
    assert entry.genre == "ambient"
    assert entry.tags == ("calm", "loop")
    assert entry.artist == "Artist"
    assert entry.license.name == "CC0-1.0"
    assert entry.license.is_commercial is False
    assert entry.is_commercial is False
    assert entry.is_cleared is False
    audio_format = entry.audio_format
    assert audio_format is not None
    assert audio_format.value == "wav"


def test_selection_by_tags_and_duration_picks_a_matching_track(
    library: MusicLibrary,
) -> None:
    entry = library.select(
        ("loop", "calm"),
        duration_seconds=30.0,
        duration_tolerance_seconds=0.5,
    )

    assert entry.id == "calm-ambient"
    assert entry.tags == ("calm", "loop")

    with pytest.raises(MusicLibraryError) as missing:
        library.select(["loop"], duration_seconds=60.0, duration_tolerance_seconds=1)
    assert missing.value.code == "music_library"
    assert missing.value.details["tags"] == ["loop"]

    assert library.select("calm,loop").id == "calm-ambient"


def test_tags_match_any_by_default_and_all_on_request(
    library: MusicLibrary,
) -> None:
    any_match = library.find(["piano", "energetic"])
    assert [match.id for match in any_match] == ["calm-piano", "upbeat-pop"]

    assert library.find(["piano", "energetic"], match_all_tags=True) == ()


def test_selection_filters_mood_genre_license_and_commercial_use(
    library: MusicLibrary,
) -> None:
    assert library.select(mood="calm", genre="ambient").id == "calm-ambient"
    assert library.select(genre="pop", license="cc0-1.0").id == "upbeat-pop"
    assert library.select(license="cc by 4.0").id == "calm-piano"

    commercial = library.find(commercial_only=True)
    assert "dark-cinematic" not in [match.id for match in commercial]

    with pytest.raises(MusicLibraryError):
        library.select(license="unknown")


def test_selection_matches_tempo_within_tolerance_and_range(
    library: MusicLibrary,
) -> None:
    assert library.select(tempo_bpm=91, tempo_tolerance_bpm=2).id == "calm-ambient"

    with pytest.raises(MusicLibraryError):
        library.select(tempo_bpm=100, tempo_tolerance_bpm=1)

    ranged = library.find(min_tempo_bpm=80, max_tempo_bpm=100)
    assert [match.id for match in ranged] == ["calm-ambient", "calm-piano"]
    assert DEFAULT_TEMPO_TOLERANCE_BPM == 5.0


def test_the_closest_duration_wins_a_tie_on_tags() -> None:
    entries = [
        track("short", duration_seconds=30.0, tags=("loop",)),
        track("long", duration_seconds=31.0, tags=("loop",)),
    ]
    library = MusicLibrary(entries)

    assert library.select(["loop"], duration_seconds=30.1).id == "short"
    assert library.select(["loop"], duration_seconds=30.9).id == "long"


def test_matches_record_score_tags_and_reasons(library: MusicLibrary) -> None:
    match = library.select_match(
        ["calm"],
        duration_seconds=30.0,
        mood="calm",
    )

    assert isinstance(match, MusicLibraryMatch)
    assert match.id == "calm-ambient"
    assert match.matched_tags == ("calm",)
    assert "mood" in match.reasons
    assert "duration" in match.reasons
    assert match.score > 0
    assert match.duration_distance_seconds == 0
    assert match.license.name == "CC0-1.0"
    assert match.to_dict()["entry"]["id"] == "calm-ambient"


def test_select_many_for_duration_and_aliases(library: MusicLibrary) -> None:
    assert [entry.id for entry in library.select_many(["loop"])] == [
        "calm-ambient",
        "upbeat-pop",
    ]
    assert library.for_duration(30.0, ["loop"]).id == "calm-ambient"
    assert library.choose(["loop"]).id == "calm-ambient"
    assert library.matches(["loop"])[0].id == "calm-ambient"


def test_limit_and_invalid_filter_values(library: MusicLibrary) -> None:
    assert len(library.find(limit=1)) == 1

    with pytest.raises(ValueError):
        library.find(limit=0)
    with pytest.raises(TypeError):
        library.find(limit=True)
    with pytest.raises(ValueError):
        library.find(min_duration_seconds=10, max_duration_seconds=5)
    with pytest.raises(ValueError):
        library.find(min_tempo_bpm=120, max_tempo_bpm=80)
    with pytest.raises(TypeError):
        library.find(tempo_bpm="fast")  # type: ignore[arg-type]


def test_an_empty_library_reports_that_nothing_matched() -> None:
    library = MusicLibrary()

    assert len(library) == 0
    with pytest.raises(MusicLibraryError) as missing:
        library.select(["calm"])
    assert missing.value.details["count"] == 0


def test_duplicate_ids_and_remote_paths_are_rejected() -> None:
    with pytest.raises(ConfigError):
        MusicLibrary([track("same"), track("same")])

    with pytest.raises(ValueError):
        MusicLibraryEntry.model_validate(track("remote", path="https://x.test/a.wav"))


def test_a_yaml_manifest_resolves_relative_paths(tmp_path: Path) -> None:
    manifest = tmp_path / "library.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {"schema_version": 1, "tracks": [track("one", path="audio/one.wav")]}
        ),
        encoding="utf-8",
    )

    library = MusicLibrary.from_file(manifest)

    assert library.ids == ("one",)
    assert Path(library.entries[0].path) == (tmp_path / "audio" / "one.wav").resolve()
    audio_format = library.entries[0].audio_format
    assert audio_format is not None
    assert audio_format.value == "wav"
    assert load_music_library(manifest).ids == ("one",)


def test_a_json_manifest_and_an_id_keyed_mapping_load(tmp_path: Path) -> None:
    manifest = tmp_path / "library.json"
    manifest.write_text(
        json.dumps({"entries": [track("one")]}),
        encoding="utf-8",
    )
    assert MusicLibrary.from_file(manifest).ids == ("one",)

    library = MusicLibrary.from_mapping(
        {"one": {key: value for key, value in track("one").items() if key != "id"}}
    )
    assert library.ids == ("one",)

    single = MusicLibrary.from_mapping(track("single"))
    assert single.ids == ("single",)


def test_manifest_errors_are_clear(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        MusicLibrary.from_file(tmp_path / "missing.yaml")

    unsupported = tmp_path / "library.txt"
    unsupported.write_text("not a manifest", encoding="utf-8")
    with pytest.raises(ConfigError):
        MusicLibrary.from_file(unsupported)

    invalid = tmp_path / "library.yaml"
    invalid.write_text("tracks: not-a-list\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        MusicLibrary.from_file(invalid)

    with pytest.raises(ConfigError):
        MusicLibrary.from_mapping({"tracks": "not-a-list"})


def test_module_level_helpers_load_and_select(library: MusicLibrary) -> None:
    assert load_music_library(library) is library
    assert build_music_library is load_music_library
    assert music_library is load_music_library

    entry = select_music({"tracks": TRACKS}, ["loop"], duration_seconds=30.0)
    assert entry.id == "calm-ambient"
