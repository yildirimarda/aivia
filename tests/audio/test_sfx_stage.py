"""Rule matched sound effects are placed with a gain and never extend a timeline."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from sqlalchemy import Engine

from aivia.audio import (
    SFXRule,
    SfxStage,
    load_sfx_rules,
    match_sfx_rules,
    place_sound_effects,
)
from aivia.compose import Timeline, compositions_from_timeline
from aivia.core.cost import CostTracker
from aivia.core.db import (
    RunRepository,
    StageRunRepository,
    create_database_engine,
    upgrade_database,
)
from aivia.core.errors import ConfigError
from aivia.core.models import Run, StageState
from aivia.core.orchestrator import Orchestrator
from aivia.core.resume import compute_input_hash
from aivia.core.run_state import RunStateMachine
from aivia.core.stages import StageContext
from aivia.providers.protocols.sfx import SFXRequest


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine)


class FakeSFX:
    """Return a measured duration and stored path for each sound effect."""

    def __init__(
        self,
        *,
        durations: dict[str, float] | None = None,
        data: bytes | None = None,
    ) -> None:
        self.durations = durations or {}
        self.data = data
        self.requests: list[SFXRequest] = []

    async def __call__(self, request: SFXRequest) -> SimpleNamespace:
        self.requests.append(request)
        duration = self.durations.get(request.prompt, 0.5)
        if self.data is not None:
            return SimpleNamespace(
                path=None,
                data=self.data,
                format="mp3",
                duration_seconds=duration,
                cost_usd=0.01,
                provider="fake",
                model="fake-sfx",
            )
        slug = request.prompt.replace(" ", "-").casefold()
        return SimpleNamespace(
            path=f"sfx/{slug}.mp3",
            duration_seconds=duration,
            cost_usd=0.01,
            provider="fake",
            model="fake-sfx",
        )


def _timeline(
    *,
    duration_seconds: float | None = 6.0,
    words: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": "run",
        "beats": [
            {
                "id": "beat_0",
                "text": "A door slams shut.",
                "start_seconds": 0.0,
                "end_seconds": 2.0,
            },
            {
                "id": "beat_1",
                "text": "The camera clicks twice.",
                "start_seconds": 2.0,
                "end_seconds": 4.0,
            },
        ],
        "languages": [
            {
                "language": "en",
                "voices": [
                    {
                        "id": "voice_0",
                        "beat": "beat_0",
                        "source": "audio/voice.wav",
                        "start_seconds": 0.0,
                        "end_seconds": 2.0,
                        "words": words or [],
                    }
                ],
            }
        ],
        "metadata": {"existing": True},
    }
    if duration_seconds is not None:
        payload["duration_seconds"] = duration_seconds
    return payload


def _write_timeline(run_dir: Path, value: dict[str, Any]) -> Path:
    path = run_dir / "timeline.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _write_rules(path: Path, rules: list[dict[str, Any]]) -> Path:
    path.write_text(yaml.safe_dump({"rules": rules}, sort_keys=False), encoding="utf-8")
    return path


def _context(run_dir: Path, *, config: Any = None) -> StageContext:
    return StageContext(run_dir=run_dir, config=config or {})


@pytest.fixture
def engine() -> Iterator[Engine]:
    created = create_database_engine(url="sqlite+pysqlite:///:memory:")
    upgrade_database(engine=created)
    try:
        yield created
    finally:
        created.dispose()


def test_rules_place_effects_on_matching_beats_with_gain_and_keep_duration(
    tmp_path: Path,
) -> None:
    _write_timeline(tmp_path, _timeline(duration_seconds=6.0))
    sfx = FakeSFX(durations={"heavy door creaks": 0.75, "camera shutter": 0.5})
    stage = SfxStage(
        options={
            "rules": [
                {
                    "id": "door",
                    "match": "door",
                    "prompt": "heavy door creaks",
                    "position": "start",
                    "volume_db": -12.0,
                },
                {
                    "id": "click",
                    "beat": "beat_1",
                    "prompt": "camera shutter",
                    "position": "end",
                    "duration_seconds": 0.5,
                    "volume_db": -18.0,
                },
            ]
        },
        sfx=sfx,
    )

    result = _run(stage.run(_context(tmp_path)))

    assert result.stage == "sfx"
    assert result.artifacts == ["timeline.json", "sfx.json"]
    assert result.metadata["placements"] == 2
    assert result.metadata["calls"] == 2
    assert result.cost_usd == pytest.approx(0.02)
    assert result.metadata["duration_seconds"] == pytest.approx(6.0)
    assert result.metadata["duration_unchanged"] is True
    assert [request.prompt for request in sfx.requests] == [
        "heavy door creaks",
        "camera shutter",
    ]
    assert sfx.requests[0].metadata["rule"] == "door"
    assert sfx.requests[0].metadata["beat"] == "beat_0"

    payload = json.loads((tmp_path / "timeline.json").read_text(encoding="utf-8"))
    timeline = Timeline.model_validate(payload)
    assert timeline.duration_seconds == pytest.approx(6.0)
    assert timeline.metadata["existing"] is True
    assert len(timeline.music) == 2
    door, click = timeline.music
    assert door.role.value == "sfx"
    assert door.start_seconds == pytest.approx(0.0)
    assert door.end_seconds == pytest.approx(0.75)
    assert door.volume_db == pytest.approx(-12.0)
    assert door.duck is False
    assert door.metadata["rule_id"] == "door"
    assert click.role.value == "sfx"
    assert click.start_seconds == pytest.approx(3.5)
    assert click.end_seconds == pytest.approx(4.0)
    assert click.volume_db == pytest.approx(-18.0)

    compositions = compositions_from_timeline(timeline)
    assert compositions[0].duration_seconds == pytest.approx(6.0)
    sfx_tracks = [
        track for track in compositions[0].audio_tracks if track.role.value == "sfx"
    ]
    assert [(track.start_seconds, track.duration_seconds) for track in sfx_tracks] == (
        pytest.approx([(0.0, 0.75), (3.5, 0.5)])
    )

    manifest = json.loads((tmp_path / "sfx.json").read_text(encoding="utf-8"))
    assert manifest["placements"] == 2
    assert manifest["duration_unchanged"] is True
    assert [track["rule"] for track in manifest["tracks"]] == ["door", "click"]


def test_an_effect_that_would_run_past_the_end_is_shortened(tmp_path: Path) -> None:
    _write_timeline(
        tmp_path,
        {
            "id": "run",
            "duration_seconds": 5.0,
            "beats": [
                {
                    "id": "beat_0",
                    "text": "A door slams.",
                    "start_seconds": 4.0,
                    "end_seconds": 5.0,
                }
            ],
        },
    )
    stage = SfxStage(
        options={
            "rules": [
                {
                    "id": "door",
                    "match": "door",
                    "prompt": "heavy door creaks",
                    "position": "start",
                }
            ]
        },
        sfx=FakeSFX(durations={"heavy door creaks": 3.0}),
    )

    result = _run(stage.run(_context(tmp_path)))

    assert result.metadata["duration_unchanged"] is True
    timeline = Timeline.model_validate(
        json.loads((tmp_path / "timeline.json").read_text(encoding="utf-8"))
    )
    track = timeline.music[0]
    assert timeline.duration_seconds == pytest.approx(5.0)
    assert track.start_seconds == pytest.approx(4.0)
    assert track.duration_seconds == pytest.approx(1.0)
    assert track.end_seconds == pytest.approx(5.0)


def test_an_offset_past_the_end_is_moved_back_without_extending(tmp_path: Path) -> None:
    _write_timeline(tmp_path, _timeline(duration_seconds=6.0))
    stage = SfxStage(
        options={
            "rules": [
                {
                    "id": "late",
                    "beat": "beat_1",
                    "prompt": "whoosh",
                    "position": "offset",
                    "start_seconds": 20.0,
                    "duration_seconds": 1.0,
                }
            ]
        },
        sfx=FakeSFX(durations={"whoosh": 1.0}),
    )

    _run(stage.run(_context(tmp_path)))

    timeline = Timeline.model_validate(
        json.loads((tmp_path / "timeline.json").read_text(encoding="utf-8"))
    )
    track = timeline.music[0]
    assert track.start_seconds == pytest.approx(5.0)
    assert track.end_seconds == pytest.approx(6.0)
    assert timeline.duration_seconds == pytest.approx(6.0)


def test_a_timeline_without_a_duration_is_pinned_to_its_original_end(
    tmp_path: Path,
) -> None:
    _write_timeline(tmp_path, _timeline(duration_seconds=None))
    sfx = FakeSFX()
    stage = SfxStage(
        options={
            "rules": [
                {
                    "id": "existing",
                    "beat": "beat_1",
                    "source": "assets/clip.mp3",
                    "position": "end",
                    "duration_seconds": 1.0,
                }
            ]
        },
        sfx=sfx,
    )

    result = _run(stage.run(_context(tmp_path)))

    assert result.metadata["pinned_duration"] is True
    assert result.metadata["duration_unchanged"] is True
    assert sfx.requests == []
    timeline = Timeline.model_validate(
        json.loads((tmp_path / "timeline.json").read_text(encoding="utf-8"))
    )
    assert timeline.duration_seconds == pytest.approx(4.0)
    assert timeline.music[0].source == "assets/clip.mp3"
    assert timeline.music[0].start_seconds == pytest.approx(3.0)
    assert timeline.music[0].end_seconds == pytest.approx(4.0)


def test_matched_words_anchor_the_effect_and_matching_is_case_insensitive(
    tmp_path: Path,
) -> None:
    _write_timeline(
        tmp_path,
        _timeline(
            duration_seconds=6.0,
            words=[
                {"word": "A", "start_seconds": 0.0, "end_seconds": 0.1},
                {"word": "door", "start_seconds": 0.5, "end_seconds": 0.8},
                {"word": "slams", "start_seconds": 0.9, "end_seconds": 1.2},
            ],
        ),
    )
    stage = SfxStage(
        options={
            "rules": [
                {
                    "id": "door",
                    "match": "DOOR",
                    "prompt": "door impact",
                    "position": "word",
                    "duration_seconds": 0.25,
                }
            ]
        },
        sfx=FakeSFX(durations={"door impact": 0.25}),
    )

    _run(stage.run(_context(tmp_path)))

    timeline = Timeline.model_validate(
        json.loads((tmp_path / "timeline.json").read_text(encoding="utf-8"))
    )
    assert timeline.music[0].start_seconds == pytest.approx(0.5)
    assert timeline.music[0].end_seconds == pytest.approx(0.75)


def test_rules_replace_existing_effects_but_keep_music(tmp_path: Path) -> None:
    payload = _timeline(duration_seconds=6.0)
    payload["music"] = [
        {
            "id": "bed",
            "source": "music/bed.mp3",
            "role": "music",
            "start_seconds": 0.0,
            "end_seconds": 6.0,
        },
        {
            "id": "old-sfx",
            "source": "sfx/old.mp3",
            "role": "sfx",
            "start_seconds": 1.0,
            "end_seconds": 2.0,
        },
    ]
    _write_timeline(tmp_path, payload)
    stage = SfxStage(
        options={
            "rules": [
                {
                    "id": "door",
                    "match": "door",
                    "prompt": "door impact",
                    "duration_seconds": 0.5,
                }
            ]
        },
        sfx=FakeSFX(durations={"door impact": 0.5}),
    )

    _run(stage.run(_context(tmp_path)))
    _run(stage.run(_context(tmp_path)))

    timeline = Timeline.model_validate(
        json.loads((tmp_path / "timeline.json").read_text(encoding="utf-8"))
    )
    assert [track.id for track in timeline.music if track.role.value == "music"] == [
        "bed"
    ]
    effects = [track for track in timeline.music if track.role.value == "sfx"]
    assert len(effects) == 1
    assert effects[0].metadata["rule_id"] == "door"
    assert timeline.duration_seconds == pytest.approx(6.0)


def test_inline_provider_bytes_are_written_under_the_sfx_directory(
    tmp_path: Path,
) -> None:
    _write_timeline(tmp_path, _timeline(duration_seconds=6.0))
    stage = SfxStage(
        options={
            "rules": [
                {
                    "id": "door",
                    "beat": "beat_0",
                    "prompt": "door impact",
                    "duration_seconds": 0.5,
                }
            ]
        },
        sfx=FakeSFX(data=b"ID3-sound-effect", durations={"door impact": 0.5}),
    )

    result = _run(stage.run(_context(tmp_path)))

    path = tmp_path / "sfx" / "door_beat_0_0.mp3"
    assert path.read_bytes() == b"ID3-sound-effect"
    assert result.metadata["tracks"][0]["source"] == str(path)
    timeline = Timeline.model_validate(
        json.loads((tmp_path / "timeline.json").read_text(encoding="utf-8"))
    )
    assert timeline.music[0].source == str(path)


def test_rules_are_loaded_from_mappings_files_and_reject_bad_placements(
    tmp_path: Path,
) -> None:
    rules = load_sfx_rules(
        {
            "door": {"match": "door", "prompt": "door impact", "volume_db": -9.0},
            "click": "camera click",
        }
    )
    assert [rule.id for rule in rules] == ["door", "click"]
    assert isinstance(rules[0], SFXRule)
    assert rules[1].prompt == "camera click"

    path = tmp_path / "rules.yaml"
    path.write_text(
        "rules:\n  - id: door\n    match: door\n    prompt: door impact\n",
        encoding="utf-8",
    )
    assert [rule.id for rule in load_sfx_rules(path)] == ["door"]

    with pytest.raises(ConfigError, match="unknown sfx placement"):
        SfxStage(
            options={
                "rules": [{"prompt": "whoosh", "position": "sideways"}],
            }
        )

    with pytest.raises(ConfigError, match="volume_db"):
        SfxStage(options={"rules": [], "volume_db": "loud"})


def test_matching_uses_word_boundaries_so_outdoor_is_not_a_door(tmp_path: Path) -> None:
    rules = load_sfx_rules(
        [
            {"id": "door", "match": "door", "prompt": "door impact"},
            {"id": "outdoor", "match": "outdoor", "prompt": "birds"},
        ]
    )

    matched = match_sfx_rules("The outdoor market is loud.", rules)

    assert [rule.id for rule in matched] == ["outdoor"]
    assert match_sfx_rules("A DOOR slams.", rules)[0].id == "door"


def test_preplaced_effects_are_clamped_by_the_pure_helper() -> None:
    timeline = Timeline.model_validate(_timeline(duration_seconds=4.0))

    tracks = place_sound_effects(
        timeline,
        [
            {
                "id": "late",
                "source": "sfx/late.mp3",
                "start_seconds": 3.5,
                "duration_seconds": 2.0,
                "volume_db": -10.0,
            }
        ],
    )

    assert timeline.duration_seconds == pytest.approx(4.0)
    assert tracks[0].start_seconds == pytest.approx(3.5)
    assert tracks[0].end_seconds == pytest.approx(4.0)
    assert tracks[0].volume_db == pytest.approx(-10.0)


def test_stage_declares_the_rule_file_and_timeline_as_inputs(tmp_path: Path) -> None:
    rules = _write_rules(
        tmp_path / "rules.yaml",
        [{"id": "door", "match": "door", "prompt": "door impact"}],
    )
    stage = SfxStage(options={"rules": str(rules)})

    assert stage.inputs == ("timeline.json", str(rules))


def test_stage_declares_no_rule_file_input_for_inline_rules() -> None:
    stage = SfxStage(
        options={"rules": [{"id": "door", "prompt": "door impact"}]},
    )

    assert stage.inputs == ("timeline.json",)


def test_editing_a_rule_file_changes_the_stage_input_hash(tmp_path: Path) -> None:
    rules = _write_rules(
        tmp_path / "rules.yaml",
        [{"id": "door", "match": "door", "prompt": "door impact"}],
    )
    stage = SfxStage(options={"rules": str(rules)})
    original = compute_input_hash(stage, base_dir=tmp_path)

    _write_rules(
        rules,
        [{"id": "door", "match": "door", "prompt": "heavy door creaks"}],
    )
    assert compute_input_hash(stage, base_dir=tmp_path) != original

    _write_rules(
        rules,
        [{"id": "door", "match": "door", "prompt": "door impact"}],
    )
    assert compute_input_hash(stage, base_dir=tmp_path) == original


def test_config_rule_files_volume_and_placement_are_stage_defaults(
    tmp_path: Path,
) -> None:
    _write_timeline(tmp_path, _timeline(duration_seconds=6.0))
    doors = _write_rules(
        tmp_path / "doors.yaml",
        [{"id": "door", "match": "door", "prompt": "door impact"}],
    )
    clicks = _write_rules(
        tmp_path / "clicks.yaml",
        [{"id": "click", "match": "camera", "prompt": "camera shutter"}],
    )
    config = {
        "sfx": {
            "rule_files": [str(doors), str(clicks)],
            "volume_db": -21.0,
            "placement": "end",
        }
    }
    sfx = FakeSFX(durations={"door impact": 0.5, "camera shutter": 0.5})
    stage = SfxStage(sfx=sfx)

    assert stage.resolve_inputs(config=config) == (
        "timeline.json",
        str(doors),
        str(clicks),
    )

    result = _run(stage.run(_context(tmp_path, config=config)))

    assert result.metadata["placements"] == 2
    assert [request.prompt for request in sfx.requests] == [
        "door impact",
        "camera shutter",
    ]
    timeline = Timeline.model_validate(
        json.loads((tmp_path / "timeline.json").read_text(encoding="utf-8"))
    )
    door, click = timeline.music
    assert (door.start_seconds, door.end_seconds) == pytest.approx((1.5, 2.0))
    assert (click.start_seconds, click.end_seconds) == pytest.approx((3.5, 4.0))
    assert door.volume_db == pytest.approx(-21.0)
    assert click.volume_db == pytest.approx(-21.0)


def test_stage_options_and_rule_values_override_sfx_config_defaults(
    tmp_path: Path,
) -> None:
    _write_timeline(tmp_path, _timeline(duration_seconds=6.0))
    rules = _write_rules(
        tmp_path / "rules.yaml",
        [
            {"id": "door", "match": "door", "prompt": "door impact"},
            {
                "id": "click",
                "match": "camera",
                "prompt": "camera shutter",
                "position": "start",
                "volume_db": -3.0,
            },
        ],
    )
    config = {
        "sfx": {
            "rule_files": [str(rules)],
            "volume_db": -21.0,
            "placement": "end",
        }
    }
    sfx = FakeSFX(durations={"door impact": 0.5, "camera shutter": 0.5})
    stage = SfxStage(options={"volume_db": -9.0, "position": "center"}, sfx=sfx)

    _run(stage.run(_context(tmp_path, config=config)))

    timeline = Timeline.model_validate(
        json.loads((tmp_path / "timeline.json").read_text(encoding="utf-8"))
    )
    door, click = timeline.music
    assert (door.start_seconds, door.end_seconds) == pytest.approx((0.75, 1.25))
    assert door.volume_db == pytest.approx(-9.0)
    assert (click.start_seconds, click.end_seconds) == pytest.approx((2.0, 2.5))
    assert click.volume_db == pytest.approx(-3.0)


def test_editing_a_rule_file_reruns_the_sfx_stage_on_resume(
    tmp_path: Path,
    engine: Engine,
) -> None:
    rules = _write_rules(
        tmp_path / "rules.yaml",
        [{"id": "door", "match": "door", "prompt": "door impact"}],
    )
    timeline = Timeline.model_validate(_timeline(duration_seconds=6.0))
    sfx = FakeSFX(durations={"door impact": 0.5, "heavy door creaks": 0.5})
    stage = SfxStage(
        options={"rules": str(rules)},
        timeline=timeline,
        sfx=sfx,
    )
    machine = RunStateMachine(RunRepository(engine))
    run = machine.create(Run())

    def execute() -> Any:
        return Orchestrator(
            stages=[stage],
            run=run.id,
            repository=StageRunRepository(engine),
            run_dir=tmp_path,
            config={},
            cost=CostTracker(engine),
            raise_on_failure=False,
        ).execute_sync()

    first = execute()
    stored = StageRunRepository(engine).get(run.id, "sfx")
    assert first.ok is True
    assert len(sfx.requests) == 1
    assert stored is not None

    second = execute()

    assert second.ok is True
    assert len(sfx.requests) == 1
    assert second.stages[0].status is StageState.SKIPPED

    _write_rules(
        rules,
        [{"id": "door", "match": "door", "prompt": "heavy door creaks"}],
    )
    third = execute()

    assert third.ok is True
    assert len(sfx.requests) == 2
    assert third.stages[0].status is StageState.SUCCEEDED
    assert third.stages[0].input_hash != stored.input_hash

    fourth = execute()

    assert fourth.ok is True
    assert len(sfx.requests) == 2
    assert fourth.stages[0].status is StageState.SKIPPED


def test_an_unchanged_sfx_stage_is_skipped_when_it_reads_the_timeline_from_disk(
    tmp_path: Path,
    engine: Engine,
) -> None:
    _write_timeline(tmp_path, _timeline(duration_seconds=6.0))
    sfx = FakeSFX(durations={"door impact": 0.5})
    stage = SfxStage(
        options={
            "rules": [
                {"id": "door", "match": "door", "prompt": "door impact"},
            ]
        },
        sfx=sfx,
    )
    machine = RunStateMachine(RunRepository(engine))
    run = machine.create(Run())

    def execute() -> Any:
        return Orchestrator(
            stages=[stage],
            run=run.id,
            repository=StageRunRepository(engine),
            run_dir=tmp_path,
            config={},
            cost=CostTracker(engine),
            raise_on_failure=False,
        ).execute_sync()

    first = execute()
    stored = StageRunRepository(engine).get(run.id, "sfx")
    assert first.ok is True
    assert stored is not None
    assert len(sfx.requests) == 1

    second = execute()

    assert second.ok is True
    assert second.stages[0].status is StageState.SKIPPED
    assert second.stages[0].input_hash == stored.input_hash
    assert len(sfx.requests) == 1

    _write_timeline(tmp_path, _timeline(duration_seconds=7.0))
    third = execute()

    assert third.ok is True
    assert third.stages[0].status is StageState.SUCCEEDED
    assert third.stages[0].input_hash != stored.input_hash
    assert len(sfx.requests) == 2
