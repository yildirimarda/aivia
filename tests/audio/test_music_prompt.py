"""A music prompt is generated from profile settings and stored with the request."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from aivia.audio import (
    MusicPromptBuilder,
    MusicStage,
    build_music_prompt,
    build_profile_music_prompt,
    music_prompt_for_profile,
)
from aivia.core.config.profile import ContentProfileConfig
from aivia.core.models.run import Run
from aivia.core.stages import StageContext
from aivia.providers.protocols.music import MusicRequest

CC0 = {
    "name": "CC0-1.0",
    "url": "https://creativecommons.org/publicdomain/zero/1.0/",
    "commercial_use": True,
}


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine)


def _timeline(*, duration_seconds: float = 5.0) -> dict[str, Any]:
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
    }


def _write_timeline(run_dir: Path, value: dict[str, Any]) -> None:
    (run_dir / "timeline.json").write_text(json.dumps(value), encoding="utf-8")


class FakeMusic:
    def __init__(self) -> None:
        self.requests: list[MusicRequest] = []

    async def __call__(self, request: MusicRequest) -> SimpleNamespace:
        self.requests.append(request)
        return SimpleNamespace(
            path="music/generated.mp3",
            data=None,
            format="mp3",
            duration_seconds=5.0,
            license=CC0,
            licenses=(CC0,),
            cost_usd=0.02,
            provider="fake",
            model="fake-music",
            title="Generated Bed",
        )


def test_profile_settings_build_a_deterministic_music_prompt() -> None:
    profile = ContentProfileConfig.from_mapping(
        {
            "brief": "A calm explainer about the night sky.",
            "niche": "astronomy",
            "tone": "warm and curious",
            "keywords": ["ambient", "piano"],
            "style_bible": "gentle-documentary",
        }
    )

    prompt = build_music_prompt(profile)

    assert "Instrumental background music" in prompt
    assert "warm and curious mood" in prompt
    assert "style: ambient, piano" in prompt
    assert "theme: astronomy" in prompt
    assert "brief: A calm explainer about the night sky." in prompt
    assert "inspired by gentle-documentary" in prompt


def test_profile_prompt_helpers_agree_and_explicit_values_win() -> None:
    profile = ContentProfileConfig.from_mapping(
        {
            "tone": "warm",
            "keywords": ["ambient"],
            "brief": "Profile brief",
        }
    )

    assert build_profile_music_prompt(profile) == build_music_prompt(profile)
    assert music_prompt_for_profile(profile) == build_music_prompt(profile)

    prompt = build_music_prompt(
        profile,
        mood="epic",
        genre="orchestral",
        tags=["drums"],
    )

    assert "epic mood" in prompt
    assert "orchestral genre" in prompt
    assert "style: drums" in prompt
    assert "warm mood" not in prompt
    assert "ambient" not in prompt
    assert build_music_prompt(profile, prompt="explicit bed") == "explicit bed"


def test_the_prompt_builder_uses_the_profile_and_explicit_values() -> None:
    profile = ContentProfileConfig.from_mapping(
        {"tone": "warm", "keywords": ["ambient"]}
    )
    builder = MusicPromptBuilder(profile)

    assert "warm mood" in builder()
    assert "warm mood" in builder.prompt()
    assert "epic mood" in builder.build(mood="epic")
    assert builder.to_dict(mood="epic")["prompt"] == builder.build(mood="epic")


def test_profile_overrides_music_settings_take_precedence() -> None:
    profile = {
        "tone": "warm",
        "keywords": ["ambient"],
        "overrides": {
            "music": {
                "prompt": "slow cinematic strings",
                "mood": "hopeful",
                "genre": "cinematic",
                "tags": ["strings"],
            }
        },
    }

    assert build_music_prompt(profile) == "slow cinematic strings"

    prompt = build_music_prompt(profile, prompt=None, mood=None, genre=None, tags=None)
    assert prompt == "slow cinematic strings"

    overridden = build_music_prompt(profile, mood="calm")
    assert "calm mood" in overridden
    assert "cinematic genre" in overridden
    assert "style: strings" in overridden


def test_stage_generates_the_prompt_from_the_run_profile_and_stores_it(
    tmp_path: Path,
) -> None:
    _write_timeline(tmp_path, _timeline())
    music = FakeMusic()
    config = {
        "profiles": {
            "song": {
                "brief": "A short song about the sea.",
                "niche": "ocean",
                "tone": "dreamy",
                "keywords": ["synth", "wave"],
            }
        }
    }
    stage = MusicStage(
        options={"source": "provider", "duration_seconds": 5.0},
        music=music,
    )
    context = StageContext(
        run_dir=tmp_path,
        config=config,
        run=Run(id="run", profile="song"),
    )

    result = _run(stage.run(context))

    assert len(music.requests) == 1
    request = music.requests[0]
    assert "dreamy mood" in request.prompt
    assert "style: synth, wave" in request.prompt
    assert "theme: ocean" in request.prompt
    assert "brief: A short song about the sea." in request.prompt
    assert request.metadata["prompt"] == request.prompt
    assert request.metadata["profile"] == "song"
    assert request.metadata["mood"] == "dreamy"
    assert request.metadata["tags"] == ["synth", "wave"]

    assert result.metadata["prompt"] == request.prompt
    assert result.metadata["profile"] == "song"
    manifest = json.loads((tmp_path / "music.json").read_text(encoding="utf-8"))
    assert manifest["prompt"] == request.prompt
    assert manifest["profile"] == "song"
    assert manifest["selected"]["prompt"] == request.prompt


def test_a_profile_music_prompt_wins_over_blank_music_config(tmp_path: Path) -> None:
    _write_timeline(tmp_path, _timeline())
    music = FakeMusic()
    config = {
        "music": {"mood": ""},
        "profiles": {
            "song": {"overrides": {"music": {"prompt": "slow cinematic strings"}}}
        },
    }
    stage = MusicStage(
        options={"source": "provider", "duration_seconds": 5.0},
        music=music,
    )
    context = StageContext(
        run_dir=tmp_path,
        config=config,
        run=Run(id="run", profile="song"),
    )

    _run(stage.run(context))

    assert music.requests[0].prompt == "slow cinematic strings"
    assert music.requests[0].metadata["prompt"] == "slow cinematic strings"


def test_an_explicit_stage_profile_overrides_the_run_profile(tmp_path: Path) -> None:
    _write_timeline(tmp_path, _timeline())
    music = FakeMusic()
    config = {
        "profiles": {
            "run": {"tone": "sleepy"},
            "override": {"tone": "playful"},
        }
    }
    stage = MusicStage(
        options={
            "source": "provider",
            "duration_seconds": 5.0,
            "profile": "override",
        },
        music=music,
    )
    context = StageContext(
        run_dir=tmp_path,
        config=config,
        run=Run(id="run", profile="run"),
    )

    _run(stage.run(context))

    request = music.requests[0]
    assert "playful mood" in request.prompt
    assert "sleepy" not in request.prompt
    assert request.metadata["profile"] == "override"


def test_explicit_music_options_still_win_over_profile_settings(
    tmp_path: Path,
) -> None:
    _write_timeline(tmp_path, _timeline())
    music = FakeMusic()
    config = {
        "profiles": {
            "song": {
                "tone": "dreamy",
                "keywords": ["synth"],
            }
        }
    }
    stage = MusicStage(
        options={
            "source": "provider",
            "duration_seconds": 5.0,
            "mood": "epic",
            "genre": "orchestral",
            "tags": ["drums"],
        },
        music=music,
    )
    context = StageContext(
        run_dir=tmp_path,
        config=config,
        run=Run(id="run", profile="song"),
    )

    _run(stage.run(context))

    request = music.requests[0]
    assert "epic mood" in request.prompt
    assert "orchestral genre" in request.prompt
    assert "style: drums" in request.prompt
    assert "dreamy" not in request.prompt
    assert request.metadata["mood"] == "epic"
    assert request.metadata["tags"] == ["drums"]
