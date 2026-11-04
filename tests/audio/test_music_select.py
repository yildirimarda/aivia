"""Background music is selected, licensed and marked for voice ducking."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from aivia.audio import (
    BackgroundMusic,
    MusicStage,
    select_library_music,
)
from aivia.compose import Timeline
from aivia.core.errors import ConfigError
from aivia.core.stages import StageContext
from aivia.providers.protocols.music import MusicRequest

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


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine)


def _track(
    entry_id: str,
    *,
    mood: str,
    genre: str = "ambient",
    tempo_bpm: float = 90.0,
    duration_seconds: float = 30.0,
    tags: tuple[str, ...] = (),
    license: dict[str, Any] = CC0,
) -> dict[str, Any]:
    return {
        "id": entry_id,
        "title": entry_id.replace("-", " ").title(),
        "path": f"music/{entry_id}.wav",
        "duration_seconds": duration_seconds,
        "mood": mood,
        "genre": genre,
        "tempo_bpm": tempo_bpm,
        "license": license,
        "tags": list(tags),
    }


LIBRARY = {
    "tracks": [
        _track("calm-ambient", mood="calm", tags=("loop",)),
        _track(
            "calm-piano",
            mood="calm",
            genre="piano",
            tempo_bpm=80.0,
            duration_seconds=12.0,
            tags=("piano",),
            license=CC_BY,
        ),
        _track(
            "dark-cinematic",
            mood="dark",
            genre="cinematic",
            tempo_bpm=70.0,
            license=NON_COMMERCIAL,
        ),
    ]
}


def _timeline(*, duration_seconds: float = 6.0) -> dict[str, Any]:
    return {
        "id": "run",
        "duration_seconds": duration_seconds,
        "beats": [{"id": "beat_0", "text": "Hello there."}],
        "languages": [
            {
                "language": "en",
                "voices": [
                    {
                        "id": "voice_0",
                        "beat": "beat_0",
                        "source": "audio/voice.wav",
                        "start_seconds": 0.0,
                        "end_seconds": duration_seconds,
                    }
                ],
            }
        ],
        "metadata": {"existing": True},
    }


def _write_timeline(run_dir: Path, value: dict[str, Any]) -> Path:
    path = run_dir / "timeline.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _context(run_dir: Path, *, config: Any = None) -> StageContext:
    return StageContext(run_dir=run_dir, config=config or {})


class FakeMusic:
    """Return a provider-like music outcome and record every request."""

    def __init__(
        self,
        *,
        path: str = "music/generated.mp3",
        duration_seconds: float = 5.0,
        license: dict[str, Any] | None = None,
        data: bytes | None = None,
    ) -> None:
        self.path = path
        self.duration_seconds = duration_seconds
        self.license = license or CC0
        self.data = data
        self.requests: list[MusicRequest] = []

    async def __call__(self, request: MusicRequest) -> SimpleNamespace:
        self.requests.append(request)
        return SimpleNamespace(
            path=None if self.data is not None else self.path,
            data=self.data,
            format="mp3",
            duration_seconds=self.duration_seconds,
            license=self.license,
            licenses=(self.license,),
            cost_usd=0.02,
            provider="fake",
            model="fake-music",
            title="Generated Bed",
        )


def test_library_selection_by_mood_records_the_license_and_ducks(
    tmp_path: Path,
) -> None:
    _write_timeline(tmp_path, _timeline(duration_seconds=6.0))
    music = FakeMusic()
    stage = MusicStage(
        options={
            "source": "auto",
            "library": LIBRARY,
            "mood": "calm",
            "duration_seconds": 6.0,
        },
        music=music,
    )

    result = _run(stage.run(_context(tmp_path)))

    assert result.stage == "music"
    assert result.artifacts == ["timeline.json", "music.json"]
    assert result.metadata["source"] == "library"
    assert result.metadata["license_name"] == "CC0-1.0"
    assert result.metadata["commercial_cleared"] is True
    assert result.metadata["ducking_applied"] is True
    assert result.metadata["ducking_verified"] is True
    assert result.metadata["calls"] == 0
    assert result.cost_usd == pytest.approx(0.0)
    assert music.requests == []

    payload = json.loads((tmp_path / "timeline.json").read_text(encoding="utf-8"))
    timeline = Timeline.model_validate(payload)
    assert timeline.duration_seconds == pytest.approx(6.0)
    assert timeline.metadata["existing"] is True
    assert [voice.id for voice in timeline.languages[0].voices] == ["voice_0"]
    assert len(timeline.music) == 1
    track = timeline.music[0]
    assert track.source == "music/calm-ambient.wav"
    assert track.role.value == "music"
    assert track.duck is True
    assert track.duration_seconds == pytest.approx(6.0)
    assert track.fade_in_seconds == pytest.approx(1.0)
    assert track.fade_out_seconds == pytest.approx(2.0)
    assert track.license is not None
    assert track.license.name == "CC0-1.0"
    assert track.metadata["origin"] == "library"
    assert track.metadata["mood"] == "calm"

    manifest = json.loads((tmp_path / "music.json").read_text(encoding="utf-8"))
    assert manifest["source"] == "library"
    assert manifest["license"]["name"] == "CC0-1.0"
    assert manifest["commercial_cleared"] is True
    assert manifest["selected"]["id"] == "calm-ambient"
    assert manifest["ducking"]["enabled"] is True
    assert manifest["ducking"]["applied"] is True
    assert manifest["ducking"]["verified"] is True


def test_auto_falls_back_to_the_provider_by_mood_and_records_its_license(
    tmp_path: Path,
) -> None:
    _write_timeline(tmp_path, _timeline(duration_seconds=5.0))
    music = FakeMusic(
        path="music/epic.mp3",
        duration_seconds=5.0,
        license={
            "name": "Provider License",
            "url": "https://example.com/provider",
            "commercial_use": True,
            "attribution_required": True,
            "attribution": "Bed by Provider",
        },
    )
    stage = MusicStage(
        options={
            "source": "auto",
            "library": LIBRARY,
            "mood": "epic",
            "genre": "orchestral",
            "duration_seconds": 5.0,
        },
        music=music,
    )

    result = _run(stage.run(_context(tmp_path)))

    assert len(music.requests) == 1
    request = music.requests[0]
    assert request.instrumental is True
    assert request.duration_seconds == pytest.approx(5.0)
    assert "epic mood" in request.prompt
    assert "orchestral genre" in request.prompt
    assert request.metadata["mood"] == "epic"

    assert result.metadata["source"] == "provider"
    assert result.metadata["calls"] == 1
    assert result.cost_usd == pytest.approx(0.02)
    assert result.metadata["license_name"] == "Provider License"
    assert result.metadata["attribution"] == "Bed by Provider"
    assert result.metadata["requires_attribution"] is True

    timeline = Timeline.model_validate(
        json.loads((tmp_path / "timeline.json").read_text(encoding="utf-8"))
    )
    track = timeline.music[0]
    assert track.source == "music/epic.mp3"
    assert track.license is not None
    assert track.license.name == "Provider License"
    assert track.metadata["provider"] == "fake"
    assert track.metadata["cost_usd"] == pytest.approx(0.02)
    assert track.duck is True


def test_library_only_fails_without_calling_a_provider(tmp_path: Path) -> None:
    _write_timeline(tmp_path, _timeline())
    music = FakeMusic()
    stage = MusicStage(
        options={"source": "library", "library": LIBRARY, "mood": "epic"},
        music=music,
    )

    with pytest.raises(ConfigError, match="no local music track"):
        _run(stage.run(_context(tmp_path)))

    assert music.requests == []
    assert not (tmp_path / "music.json").exists()


def test_inline_provider_bytes_are_written_under_the_run_directory(
    tmp_path: Path,
) -> None:
    _write_timeline(tmp_path, _timeline(duration_seconds=4.0))
    stage = MusicStage(
        options={
            "source": "provider",
            "prompt": "warm lofi bed",
            "duration_seconds": 4.0,
        },
        music=FakeMusic(data=b"fake-mp3", duration_seconds=4.0),
    )

    result = _run(stage.run(_context(tmp_path)))

    path = tmp_path / "music" / "music.mp3"
    assert path.read_bytes() == b"fake-mp3"
    assert result.metadata["path"] == str(path)
    timeline = Timeline.model_validate(
        json.loads((tmp_path / "timeline.json").read_text(encoding="utf-8"))
    )
    assert timeline.music[0].source == str(path)


def test_custom_ducking_settings_are_recorded_and_verified(tmp_path: Path) -> None:
    _write_timeline(tmp_path, _timeline(duration_seconds=8.0))
    stage = MusicStage(
        options={
            "source": "library",
            "library": LIBRARY,
            "mood": "calm",
            "duration_seconds": 8.0,
            "volume_db": -12.0,
            "ducking": {
                "threshold_db": -18.0,
                "ratio": 4.0,
                "attack_ms": 15.0,
                "release_ms": 250.0,
            },
        }
    )

    result = _run(stage.run(_context(tmp_path)))

    assert result.metadata["ducking"]["threshold_db"] == pytest.approx(-18.0)
    assert result.metadata["ducking"]["ratio"] == pytest.approx(4.0)
    assert result.metadata["ducking_verified"] is True

    timeline = Timeline.model_validate(
        json.loads((tmp_path / "timeline.json").read_text(encoding="utf-8"))
    )
    track = timeline.music[0]
    assert track.volume_db == pytest.approx(-12.0)
    assert track.duck is True
    assert timeline.metadata["music"]["ducking"]["threshold_db"] == pytest.approx(-18.0)


def test_ducking_can_be_disabled_for_a_music_only_timeline(tmp_path: Path) -> None:
    _write_timeline(
        tmp_path,
        {"id": "run", "duration_seconds": 3.0, "beats": [{"id": "b"}]},
    )
    stage = MusicStage(
        options={
            "source": "library",
            "library": LIBRARY,
            "mood": "calm",
            "duration_seconds": 3.0,
            "duck": False,
        }
    )

    result = _run(stage.run(_context(tmp_path)))

    assert result.metadata["ducking_applied"] is False
    assert result.metadata["ducking_verified"] is False
    timeline = Timeline.model_validate(
        json.loads((tmp_path / "timeline.json").read_text(encoding="utf-8"))
    )
    assert timeline.music[0].duck is False


def test_select_library_music_returns_a_licensed_choice() -> None:
    choice = select_library_music(LIBRARY, mood="calm", tags="piano")

    assert isinstance(choice, BackgroundMusic)
    assert choice.id == "calm-piano"
    assert choice.origin == "library"
    assert choice.license_name == "CC BY 4.0"
    assert choice.attribution == "Music by Example"
    assert choice.requires_attribution is True
    assert choice.to_dict()["license"]["name"] == "CC BY 4.0"


def test_unknown_source_and_negative_volume_are_rejected() -> None:
    with pytest.raises(ConfigError, match="unknown background music source"):
        MusicStage(options={"source": "radio"})

    with pytest.raises(ConfigError, match="volume_db"):
        MusicStage(options={"volume_db": "loud"})
