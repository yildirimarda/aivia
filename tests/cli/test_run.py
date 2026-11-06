"""``aivia run``: create, execute or enqueue a run and inspect its outcome."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from sqlalchemy import Connection

from aivia.cli import main
from aivia.core.cost import CostTracker
from aivia.core.db import (
    JobRepository,
    RunRepository,
    StageRunRepository,
    create_database_engine,
    upgrade_database,
)
from aivia.core.errors import QuotaError
from aivia.core.jobs import JobQueue
from aivia.core.models import Job, Run, RunState, StageResult, StageState
from aivia.core.run_dirs import RunDirectory
from aivia.core.stages import Stage, StageContext
from aivia.providers.protocols import StoragePutRequest


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.lstrip(), encoding="utf-8")
    return path


@pytest.fixture
def config(tmp_path: Path) -> Path:
    return write_config(
        tmp_path,
        """
providers:
  text:
    type: mock-llm
formats:
  sample:
    description: A small priced recipe.
    stages:
      - stage: write
    capabilities: [llm]
    output_type: video
    handler: save
""",
    )


@pytest.fixture
def database(tmp_path: Path) -> Path:
    path = tmp_path / "data" / "aivia.db"
    upgrade_database(path)
    return path


def run_command(
    capsys: pytest.CaptureFixture[str],
    *args: str,
) -> tuple[int, str, str]:
    code = main(["run", *args])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def run_id_from(output: str) -> str:
    line = next(line for line in output.splitlines() if line.startswith("Run: "))
    return line.removeprefix("Run: ").strip()


def test_run_creates_and_executes_a_run_by_default(
    config: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Write(Stage):
        name = "write"
        artifacts = ("script.md",)

        async def run(self, context: StageContext) -> StageResult:
            (context.run_dir / "script.md").write_text("script", encoding="utf-8")
            return StageResult(stage=self.name, artifacts=["script.md"], cost_usd=0.1)

    monkeypatch.setattr(
        "aivia.cli.run.build_stages",
        lambda recipe: [Write() for _stage in recipe.stages],
    )

    code, out, err = run_command(
        capsys,
        "--format",
        "sample",
        "--config",
        str(config),
        "--database",
        str(database),
    )

    assert code == 0
    assert err == ""
    run_id = run_id_from(out)
    assert "Estimate: $0.000500" in out
    assert "State: succeeded" in out

    engine = create_database_engine(database)
    try:
        run = RunRepository(engine).get(run_id)
        assert run is not None
        assert run.state is RunState.SUCCEEDED
        assert run.format == "sample"
        assert run.estimated_cost_usd == pytest.approx(0.0005)
        assert run.cost_usd == pytest.approx(0.1)
        stage = StageRunRepository(engine).get(run_id, "write")
        assert stage is not None
        assert stage.status is StageState.SUCCEEDED
        assert stage.artifacts == ["script.md"]
    finally:
        engine.dispose()

    assert Path(run.run_dir, "script.md").is_file()


def test_run_stores_generated_media_through_the_configured_storage(
    tmp_path: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = write_config(
        tmp_path,
        """
providers:
  text:
    type: mock-llm
storage:
  backend: local
  local_dir: generated-media
formats:
  sample:
    description: A recipe whose stage stores generated media.
    stages:
      - stage: write
    capabilities: [llm]
    output_type: video
    handler: save
""",
    )

    class Store(Stage):
        name = "write"
        artifacts = ("script.md",)

        async def run(self, context: StageContext) -> StageResult:
            assert context.storage is not None
            (context.run_dir / "script.md").write_text("script", encoding="utf-8")
            await context.storage.put(
                StoragePutRequest(
                    data=b"generated-media",
                    key="outputs/generated.bin",
                )
            )
            return StageResult(stage=self.name, artifacts=["script.md"])

    monkeypatch.setattr(
        "aivia.cli.run.build_stages",
        lambda recipe: [Store() for _stage in recipe.stages],
    )

    code, out, err = run_command(
        capsys,
        "--format",
        "sample",
        "--config",
        str(config),
        "--database",
        str(database),
    )

    assert code == 0, err
    assert err == ""
    assert "State: succeeded" in out

    stored = tmp_path / "data" / "generated-media" / "outputs" / "generated.bin"
    assert stored.read_bytes() == b"generated-media"


def test_run_enqueues_a_job_without_executing(
    config: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, out, err = run_command(
        capsys,
        "--format",
        "sample",
        "--enqueue",
        "--config",
        str(config),
        "--database",
        str(database),
    )

    assert code == 0
    assert err == ""
    run_id = run_id_from(out)
    assert "Estimate: $0.000500" in out
    assert "State: queued" in out
    assert "Job: job_" in out

    engine = create_database_engine(database)
    try:
        run = RunRepository(engine).get(run_id)
        assert run is not None
        assert run.state is RunState.QUEUED
        assert run.format == "sample"
        assert run.estimated_cost_usd == pytest.approx(0.0005)
        assert run.run_dir

        jobs = JobRepository(engine).list(run_id=run_id)
        assert len(jobs) == 1
        assert jobs[0].state.value == "pending"
        assert jobs[0].kind == "run"
        assert jobs[0].payload["format"] == "sample"
        assert StageRunRepository(engine).list(run_id=run_id) == []
    finally:
        engine.dispose()

    assert Path(run.run_dir).is_dir()


def test_run_enqueues_when_no_stage_implementation_is_registered(
    tmp_path: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = write_config(
        tmp_path,
        """
providers:
  text:
    type: mock-llm
formats:
  sample:
    description: A recipe whose only stage is not implemented yet.
    stages:
      - stage: publish
    capabilities: [llm]
    output_type: video
    handler: save
""",
    )

    code, out, err = run_command(
        capsys,
        "--format",
        "sample",
        "--config",
        str(config),
        "--database",
        str(database),
    )

    assert code == 0
    assert err == ""
    run_id = run_id_from(out)
    assert "State: queued" in out

    engine = create_database_engine(database)
    try:
        stored = RunRepository(engine).get(run_id)
        assert stored is not None
        assert stored.state is RunState.QUEUED
        assert len(JobRepository(engine).list(run_id=run_id)) == 1
    finally:
        engine.dispose()


def test_build_stages_uses_the_concrete_write_stage() -> None:
    from aivia.cli.run import build_stages
    from aivia.core.config.formats import FormatRecipeConfig
    from aivia.writing import WriteStage

    recipe = FormatRecipeConfig.model_validate(
        {"stages": [{"stage": "write", "options": {"languages": ["en"]}}]}
    )

    stages = build_stages(recipe)

    assert isinstance(stages[0], WriteStage)
    assert stages[0].options["languages"] == ["en"]


def test_build_stages_uses_the_concrete_visuals_stage() -> None:
    from aivia.cli.run import build_stages
    from aivia.core.config.formats import FormatRecipeConfig
    from aivia.visuals import VisualReuseStage

    recipe = FormatRecipeConfig.model_validate(
        {
            "stages": [{"stage": "visuals", "options": {"strategy": "image"}}],
            "output_type": "video",
        }
    )

    stages = build_stages(recipe)

    assert isinstance(stages[0], VisualReuseStage)
    assert stages[0].name == "visuals"
    assert stages[0].strategy == "image"


def test_build_stages_uses_the_concrete_music_stage() -> None:
    from aivia.audio import MusicSelectStage
    from aivia.cli.run import build_stages
    from aivia.core.config.formats import FormatRecipeConfig

    recipe = FormatRecipeConfig.model_validate(
        {
            "stages": [
                {
                    "stage": "music",
                    "options": {"lyrics_from_script": True, "duration_seconds": 6},
                }
            ],
            "output_type": "audio",
        }
    )

    stages = build_stages(recipe)

    assert isinstance(stages[0], MusicSelectStage)
    assert stages[0].name == "music"
    assert stages[0].lyrics_from_script is True
    assert stages[0].duration_seconds == 6


def test_build_stages_uses_the_concrete_align_stage() -> None:
    from aivia.audio import LyricAlignStage
    from aivia.cli.run import build_stages
    from aivia.core.config.formats import FormatRecipeConfig

    recipe = FormatRecipeConfig.model_validate(
        {
            "stages": [{"stage": "align", "options": {"require_timings": False}}],
            "output_type": "audio",
        }
    )

    stages = build_stages(recipe)

    assert isinstance(stages[0], LyricAlignStage)
    assert stages[0].name == "align"
    assert stages[0].require_timings is False


def test_build_stages_uses_the_concrete_cover_art_stage() -> None:
    from aivia.cli.run import build_stages
    from aivia.core.config.formats import FormatRecipeConfig
    from aivia.formats.audio_only import CoverArtStage

    recipe = FormatRecipeConfig.model_validate(
        {
            "stages": [{"stage": "cover", "options": {"aspect_ratio": "1:1"}}],
            "output_type": "audio",
        }
    )

    stages = build_stages(recipe)

    assert isinstance(stages[0], CoverArtStage)
    assert stages[0].name == "cover"
    assert stages[0].aspect_ratio == "1:1"


def test_build_stages_uses_the_concrete_image_render_stage() -> None:
    from aivia.cli.run import build_stages
    from aivia.core.config.formats import FormatRecipeConfig
    from aivia.formats.image_post import ImagePostStage

    recipe = FormatRecipeConfig.model_validate(
        {
            "stages": [{"stage": "render", "options": {"aspect": "square"}}],
            "output_type": "image",
        }
    )

    stages = build_stages(recipe)

    assert isinstance(stages[0], ImagePostStage)
    assert stages[0].options["aspect"] == "square"


def test_build_stages_uses_the_concrete_quote_card_render_stage() -> None:
    from aivia.cli.run import build_stages
    from aivia.core.config.formats import FormatRecipeConfig
    from aivia.formats.quote_card import QuoteCardRenderStage

    recipe = FormatRecipeConfig.model_validate(
        {
            "stages": [{"stage": "render", "options": {"template": "classic"}}],
            "output_type": "image",
        }
    )

    stages = build_stages(recipe)

    assert isinstance(stages[0], QuoteCardRenderStage)
    assert stages[0].options["template"] == "classic"


def test_build_stages_uses_the_concrete_carousel_render_stage() -> None:
    from aivia.cli.run import build_stages
    from aivia.core.config.formats import FormatRecipeConfig
    from aivia.formats.image_post import CarouselStage

    recipe = FormatRecipeConfig.model_validate(
        {
            "stages": [{"stage": "render", "options": {"carousel": True}}],
            "output_type": "carousel",
        }
    )

    stages = build_stages(recipe)

    assert isinstance(stages[0], CarouselStage)


def test_build_stages_uses_the_concrete_video_render_stage() -> None:
    from aivia.cli.run import build_stages
    from aivia.core.config.formats import FormatRecipeConfig
    from aivia.formats.song_video import SongVideoStage

    recipe = FormatRecipeConfig.model_validate(
        {"stages": [{"stage": "render"}], "output_type": "video"}
    )

    stages = build_stages(recipe)

    assert isinstance(stages[0], SongVideoStage)


def test_build_stages_pins_a_named_formats_renderer() -> None:
    from aivia.cli.run import build_stages
    from aivia.core.config.formats import FormatRecipeConfig
    from aivia.formats.quote_card import QuoteCardRenderStage

    recipe = FormatRecipeConfig.model_validate(
        {"stages": [{"stage": "render"}], "output_type": "video"}
    )

    stages = build_stages(recipe, "quote_card")

    assert isinstance(stages[0], QuoteCardRenderStage)


_SCRIPTED_PROVIDER = """
from aivia.providers.mock import MockLLMProvider
from aivia.providers.protocols.llm import LLMResponse


class ScriptedLLM(MockLLMProvider):
    async def complete(self, request):
        return LLMResponse(
            content=(
                '{"title": "Honey", "hook": "Honey keeps forever.", '
                '"beats": [{"text": "Honey never spoils."}]}'
            ),
            provider=self.name,
            model=request.model or "mock-llm",
        )
"""


_LYRICS_PROVIDER = """
import json

from aivia.providers.mock import MockLLMProvider, MockSTTProvider
from aivia.providers.protocols.audio import WordTimestamp
from aivia.providers.protocols.llm import LLMResponse
from aivia.providers.protocols.stt import STTResponse

LYRICS = [
    "City lights are calling me",
    "We are running wild and free",
]


class LyricsLLM(MockLLMProvider):
    async def complete(self, request):
        return LLMResponse(
            content=json.dumps(
                {
                    "title": "Neon sky",
                    "payload": {
                        "sections": [{"name": "verse", "lines": LYRICS}]
                    },
                    "beats": [{"text": line} for line in LYRICS],
                }
            ),
            provider=self.name,
            model=request.model or "mock-llm",
        )


class LyricsSTT(MockSTTProvider):
    async def transcribe(self, request):
        words = []
        cursor = 0.0
        for line in LYRICS:
            for word in line.split():
                start = cursor
                end = start + 0.28
                words.append(
                    WordTimestamp(
                        word=word,
                        start_seconds=start,
                        end_seconds=end,
                    )
                )
                cursor = end + 0.04
            cursor += 0.12
        return STTResponse(
            text=" ".join(word.word for word in words),
            language=request.language or "en",
            duration_seconds=cursor,
            words=words,
            provider=self.name,
            model="lyrics-stt",
        )
"""


def test_a_song_run_generates_and_times_audio(
    tmp_path: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = f"fake_song_provider_{tmp_path.name.replace('-', '_')}"
    (tmp_path / f"{module_name}.py").write_text(_LYRICS_PROVIDER, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ffprobe = bin_dir / "ffprobe"
    ffprobe.write_text(
        """#!/bin/sh
cat <<'JSON'
{"streams":[{"index":0,"codec_type":"audio","codec_name":"mp3","duration":"6.000","sample_rate":"44100","channels":1}],"format":{"format_name":"mp3","format_long_name":"MP3","duration":"6.000","size":"88200"}}
JSON
""",
        encoding="utf-8",
    )
    ffprobe.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    source = tmp_path / "facts.txt"
    source.write_text(
        "Honey never spoils.\nBees recognise human faces.\n",
        encoding="utf-8",
    )
    config = write_config(
        tmp_path,
        f"""
providers:
  text:
    type: {module_name}:LyricsLLM
  songs:
    type: mock-music
  speech:
    type: {module_name}:LyricsSTT
music:
  source: provider
formats:
  sample:
    description: A song recipe that generates and times audio.
    stages:
      - stage: input_topic
      - stage: write
        options:
          mode: lyrics
      - stage: music
        options:
          lyrics_from_script: true
          duration_seconds: 6.0
      - stage: align
    capabilities: [llm, music_gen, stt]
    output_type: audio
    handler: save
""",
    )

    code, out, err = run_command(
        capsys,
        "--format",
        "sample",
        "--text-file",
        str(source),
        "--config",
        str(config),
        "--database",
        str(database),
    )

    assert code == 0, err
    assert err == ""
    assert "State: succeeded" in out
    assert "Job:" not in out
    run_id = run_id_from(out)

    engine = create_database_engine(database)
    try:
        run = RunRepository(engine).get(run_id)
        assert run is not None
        assert run.state is RunState.SUCCEEDED
        assert JobRepository(engine).list(run_id=run_id) == []
        music = StageRunRepository(engine).get(run_id, "music")
        assert music is not None
        assert music.status is StageState.SUCCEEDED
        assert music.artifacts == ["timeline.json", "music.json"]
        align = StageRunRepository(engine).get(run_id, "align")
        assert align is not None
        assert align.status is StageState.SUCCEEDED
        assert align.artifacts == ["timeline.json", "lyrics-alignment.json"]
    finally:
        engine.dispose()

    assert run.run_dir
    run_dir = Path(run.run_dir)
    music_manifest = json.loads((run_dir / "music.json").read_text(encoding="utf-8"))
    assert music_manifest["source"] == "provider"
    selected = Path(music_manifest["selected"]["source"])
    assert selected.is_file()

    timeline = json.loads((run_dir / "timeline.json").read_text(encoding="utf-8"))
    assert timeline["duration_seconds"] > 0
    assert len(timeline["music"]) == 1
    assert timeline["music"][0]["source"] == str(selected)
    assert [beat["text"] for beat in timeline["beats"]] == [
        "City lights are calling me",
        "We are running wild and free",
    ]
    assert all(
        beat["start_seconds"] < beat["end_seconds"] for beat in timeline["beats"]
    )
    subtitles = timeline["languages"][0]["subtitles"]
    assert subtitles["metadata"]["karaoke"] is True
    assert [cue["text"] for cue in subtitles["cues"]] == [
        beat["text"] for beat in timeline["beats"]
    ]

    alignment = json.loads(
        (run_dir / "lyrics-alignment.json").read_text(encoding="utf-8")
    )
    assert alignment["coverage"] == pytest.approx(1.0)
    assert alignment["timeline"] == "timeline.json"


def test_an_audio_run_generates_cover_art_instead_of_enqueueing(
    tmp_path: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = f"fake_cover_writer_{tmp_path.name.replace('-', '_')}"
    (tmp_path / f"{module_name}.py").write_text(_SCRIPTED_PROVIDER, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    source = tmp_path / "facts.txt"
    source.write_text(
        "Honey never spoils.\nBees recognise human faces.\n",
        encoding="utf-8",
    )
    config = write_config(
        tmp_path,
        f"""
providers:
  text:
    type: {module_name}:ScriptedLLM
  pictures:
    type: mock-image
storage:
  backend: local
  local_dir: generated-media
formats:
  sample:
    description: An audio recipe that generates a cover.
    stages:
      - stage: input_topic
      - stage: write
      - stage: cover
        options:
          aspect_ratio: "1:1"
    capabilities: [llm, image_gen]
    output_type: audio
    handler: save
""",
    )

    code, out, err = run_command(
        capsys,
        "--format",
        "sample",
        "--text-file",
        str(source),
        "--config",
        str(config),
        "--database",
        str(database),
    )

    assert code == 0, err
    assert err == ""
    assert "State: succeeded" in out
    assert "Job:" not in out
    run_id = run_id_from(out)

    engine = create_database_engine(database)
    try:
        run = RunRepository(engine).get(run_id)
        assert run is not None
        assert run.state is RunState.SUCCEEDED
        assert JobRepository(engine).list(run_id=run_id) == []
        stage = StageRunRepository(engine).get(run_id, "cover")
        assert stage is not None
        assert stage.status is StageState.SUCCEEDED
        assert stage.artifacts == ["cover.png", "cover.json"]
        assert stage.cost_usd > 0.0
    finally:
        engine.dispose()

    assert run.run_dir
    run_dir = Path(run.run_dir)
    assert (run_dir / "cover.png").is_file()
    manifest = json.loads((run_dir / "cover.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "cover"
    assert manifest["cover"] == "cover.png"
    assert manifest["provider"] == "pictures"
    assert manifest["title"] == "Honey"
    assert manifest["aspect_ratio"] == "1:1"
    assert manifest["cost_usd"] > 0.0

    digest = hashlib.sha256((run_dir / "cover.png").read_bytes()).hexdigest()
    stored = (
        tmp_path
        / "data"
        / "generated-media"
        / "image"
        / digest[:2]
        / f"{digest}.png"
    )
    assert stored.read_bytes() == (run_dir / "cover.png").read_bytes()


def test_a_text_file_run_executes_the_concrete_write_stage(
    tmp_path: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = f"fake_writer_{tmp_path.name.replace('-', '_')}"
    (tmp_path / f"{module_name}.py").write_text(_SCRIPTED_PROVIDER, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    source = tmp_path / "facts.txt"
    source.write_text(
        "Honey never spoils.\nBees recognise human faces.\n",
        encoding="utf-8",
    )
    config = write_config(
        tmp_path,
        f"""
providers:
  text:
    type: {module_name}:ScriptedLLM
formats:
  sample:
    description: A text file feeding the concrete write stage.
    stages:
      - stage: input_topic
      - stage: write
    capabilities: [llm]
    output_type: video
    handler: save
""",
    )

    code, out, err = run_command(
        capsys,
        "--format",
        "sample",
        "--text-file",
        str(source),
        "--config",
        str(config),
        "--database",
        str(database),
    )

    assert code == 0
    assert err == ""
    assert "State: succeeded" in out
    assert "Job:" not in out
    run_id = run_id_from(out)

    engine = create_database_engine(database)
    try:
        run = RunRepository(engine).get(run_id)
        assert run is not None
        assert run.state is RunState.SUCCEEDED
        assert JobRepository(engine).list(run_id=run_id) == []
        stage = StageRunRepository(engine).get(run_id, "write")
        assert stage is not None
        assert stage.status is StageState.SUCCEEDED
    finally:
        engine.dispose()

    assert run.run_dir
    script = json.loads(
        (Path(run.run_dir) / "script.en.json").read_text(encoding="utf-8")
    )
    assert script["hook"] == "Honey keeps forever."


def test_an_image_run_executes_the_concrete_render_stage(
    tmp_path: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = f"fake_image_writer_{tmp_path.name.replace('-', '_')}"
    (tmp_path / f"{module_name}.py").write_text(_SCRIPTED_PROVIDER, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    source = tmp_path / "facts.txt"
    source.write_text(
        "Honey never spoils.\nBees recognise human faces.\n",
        encoding="utf-8",
    )
    config = write_config(
        tmp_path,
        f"""
providers:
  text:
    type: {module_name}:ScriptedLLM
formats:
  sample:
    description: A text file feeding an image render.
    stages:
      - stage: input_topic
      - stage: write
      - stage: render
        options:
          aspect: square
    capabilities: [llm]
    output_type: image
    handler: save
""",
    )

    code, out, err = run_command(
        capsys,
        "--format",
        "sample",
        "--text-file",
        str(source),
        "--config",
        str(config),
        "--database",
        str(database),
    )

    assert code == 0
    assert err == ""
    assert "State: succeeded" in out
    assert "Job:" not in out
    run_id = run_id_from(out)

    engine = create_database_engine(database)
    try:
        run = RunRepository(engine).get(run_id)
        assert run is not None
        assert run.state is RunState.SUCCEEDED
        assert JobRepository(engine).list(run_id=run_id) == []
        stage = StageRunRepository(engine).get(run_id, "render")
        assert stage is not None
        assert stage.status is StageState.SUCCEEDED
        assert stage.artifacts == ["image-post.yaml", "image-post.png"]
    finally:
        engine.dispose()

    assert run.run_dir
    assert (Path(run.run_dir) / "image-post.png").is_file()


def test_a_video_run_executes_the_concrete_render_stage(
    tmp_path: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = f"fake_video_writer_{tmp_path.name.replace('-', '_')}"
    (tmp_path / f"{module_name}.py").write_text(_SCRIPTED_PROVIDER, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    source = tmp_path / "facts.txt"
    source.write_text(
        "Honey never spoils.\nBees recognise human faces.\n",
        encoding="utf-8",
    )
    timeline = tmp_path / "timeline.json"
    timeline.write_text(
        json.dumps(
            {
                "beats": [
                    {
                        "id": "beat-1",
                        "start_seconds": 0.0,
                        "duration_seconds": 1.0,
                        "text": "Honey never spoils.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    config = write_config(
        tmp_path,
        f"""
providers:
  text:
    type: {module_name}:ScriptedLLM
formats:
  sample:
    description: A text file feeding a video render.
    stages:
      - stage: input_topic
      - stage: write
      - stage: render
        options:
          timeline: {timeline}
          render: false
    capabilities: [llm]
    output_type: video
    handler: save
""",
    )

    code, out, err = run_command(
        capsys,
        "--format",
        "sample",
        "--text-file",
        str(source),
        "--config",
        str(config),
        "--database",
        str(database),
    )

    assert code == 0
    assert err == ""
    assert "State: succeeded" in out
    assert "Job:" not in out
    run_id = run_id_from(out)

    engine = create_database_engine(database)
    try:
        run = RunRepository(engine).get(run_id)
        assert run is not None
        assert run.state is RunState.SUCCEEDED
        assert JobRepository(engine).list(run_id=run_id) == []
        stage = StageRunRepository(engine).get(run_id, "render")
        assert stage is not None
        assert stage.status is StageState.SUCCEEDED
        assert stage.artifacts == ["song-video.yaml"]
    finally:
        engine.dispose()

    assert run.run_dir
    assert (Path(run.run_dir) / "song-video.yaml").is_file()


def test_a_video_run_generates_visuals_instead_of_enqueueing(
    tmp_path: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = f"fake_visuals_writer_{tmp_path.name.replace('-', '_')}"
    (tmp_path / f"{module_name}.py").write_text(_SCRIPTED_PROVIDER, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    source = tmp_path / "facts.txt"
    source.write_text(
        "Honey never spoils.\nBees recognise human faces.\n",
        encoding="utf-8",
    )
    config = write_config(
        tmp_path,
        f"""
providers:
  text:
    type: {module_name}:ScriptedLLM
  pictures:
    type: mock-image
formats:
  sample:
    description: A text file feeding a video visuals stage.
    stages:
      - stage: input_topic
      - stage: write
      - stage: visuals
        options:
          strategy: image
    capabilities: [llm, image_gen]
    output_type: video
    handler: save
""",
    )

    code, out, err = run_command(
        capsys,
        "--format",
        "sample",
        "--text-file",
        str(source),
        "--config",
        str(config),
        "--database",
        str(database),
    )

    assert code == 0
    assert err == ""
    assert "State: succeeded" in out
    assert "Job:" not in out
    run_id = run_id_from(out)

    engine = create_database_engine(database)
    try:
        run = RunRepository(engine).get(run_id)
        assert run is not None
        assert run.state is RunState.SUCCEEDED
        assert JobRepository(engine).list(run_id=run_id) == []
        stage = StageRunRepository(engine).get(run_id, "visuals")
        assert stage is not None
        assert stage.status is StageState.SUCCEEDED
        assert stage.artifacts == ["visual-reuse.json", "shared-visuals.json"]
    finally:
        engine.dispose()

    assert run.run_dir
    manifest = json.loads(
        (Path(run.run_dir) / "visual-reuse.json").read_text(encoding="utf-8")
    )
    assert manifest["stage"] == "visuals"
    assert len(manifest["beats"]) == 1
    assert manifest["beats"][0]["request"]["visual_type"] == "image"
    assert manifest["beats"][0]["provider"] == "pictures"
    assert manifest["beats"][0]["asset"] is not None


def test_run_requires_a_format(
    config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = run_command(capsys, "--config", str(config))

    assert code == 2
    assert out == ""
    assert "--format is required" in err


def seed_run(database: Path) -> str:
    engine = create_database_engine(database)
    try:
        run = RunRepository(engine).create(
            Run(
                id="run_show",
                state=RunState.SUCCEEDED,
                profile="daily",
                format="sample",
                cost_usd=0.25,
                estimated_cost_usd=0.5,
            )
        )
        run_dir = RunDirectory(database.parent / "runs" / run.id, run_id=run.id)
        run_dir.create()
        (run_dir.path / "script.md").write_text("script", encoding="utf-8")
        run_dir.register("write", ["script.md"])
        RunRepository(engine).update(run.id, run_dir=str(run_dir.path))

        StageRunRepository(engine).create(
            StageResult(
                run_id=run.id,
                stage="write",
                status=StageState.SUCCEEDED,
                index=0,
                artifacts=["script.md"],
                cost_usd=0.25,
                estimated_cost_usd=0.5,
            )
        )
        return run.id
    finally:
        engine.dispose()


def test_run_show_prints_stages_cost_and_artifacts(
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_id = seed_run(database)

    code, out, err = run_command(capsys, "show", run_id, "--database", str(database))

    assert code == 0
    assert err == ""
    assert f"Run: {run_id}" in out
    assert "Format: sample" in out
    assert "State: succeeded" in out
    assert "Cost: $0.250000" in out
    assert "Estimate: $0.500000" in out
    assert "STAGES" in out
    assert "write\tsucceeded\t$0.250000\t$0.500000" in out
    assert "ARTIFACTS" in out
    assert "write\tscript.md\t6" in out


def test_run_show_uses_the_cost_ledger_when_the_run_has_no_total(
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_id = seed_run(database)
    engine = create_database_engine(database)
    try:
        RunRepository(engine).update(run_id, cost_usd=0.0)
        CostTracker(engine).record(
            provider="acme",
            model="llm-1",
            operation="write",
            units=10.0,
            unit_kind="token",
            amount_usd=0.65,
            run_id=run_id,
            estimated=False,
        )
    finally:
        engine.dispose()

    code, out, err = run_command(capsys, "show", run_id, "--database", str(database))

    assert code == 0
    assert err == ""
    assert "Cost: $0.650000" in out


def test_run_show_reports_a_missing_run(
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, out, err = run_command(
        capsys,
        "show",
        "run_missing",
        "--database",
        str(database),
    )

    assert code == 1
    assert out == ""
    assert "run not found: run_missing" in err


def test_run_show_requires_a_run_id(
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, out, err = run_command(capsys, "show", "--database", str(database))

    assert code == 2
    assert out == ""
    assert "a run id is required" in err


def test_run_show_reports_a_missing_database(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing.db"

    code, out, err = run_command(
        capsys,
        "show",
        "run_1",
        "--database",
        str(missing),
    )

    assert code == 1
    assert out == ""
    assert f"database not found: {missing}" in err


def test_run_enqueue_rejects_a_full_queue_without_creating_a_run(
    tmp_path: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = write_config(
        tmp_path,
        """
jobs:
  max_queue_depth: 1
providers:
  text:
    type: mock-llm
formats:
  sample:
    description: A small priced recipe.
    stages:
      - stage: write
    capabilities: [llm]
    output_type: video
    handler: save
""",
    )
    setup = create_database_engine(database)
    try:
        JobQueue(setup).enqueue(kind="seed")
    finally:
        setup.dispose()

    code, out, err = run_command(
        capsys,
        "--format",
        "sample",
        "--enqueue",
        "--config",
        str(config),
        "--database",
        str(database),
    )

    assert code == 1
    assert out == ""
    assert "queue is full" in err

    engine = create_database_engine(database)
    try:
        assert RunRepository(engine).list() == []
        assert len(JobRepository(engine).list()) == 1
    finally:
        engine.dispose()


def test_a_rejected_enqueue_rolls_back_the_run_created_for_the_submission(
    tmp_path: Path,
    config: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(self: JobQueue, *args: object, **kwargs: object) -> None:
        raise QuotaError(
            "queue is full: depth 1 reached limit 1",
            details={"depth": 1, "max_queue_depth": 1},
        )

    monkeypatch.setattr(JobQueue, "enqueue", reject)

    code, out, err = run_command(
        capsys,
        "--format",
        "sample",
        "--enqueue",
        "--config",
        str(config),
        "--database",
        str(database),
    )

    assert code == 1
    assert out == ""
    assert "queue is full" in err

    runs = tmp_path / "data" / "runs"
    assert not runs.exists() or list(runs.iterdir()) == []

    engine = create_database_engine(database)
    try:
        assert RunRepository(engine).list() == []
        assert JobRepository(engine).list() == []
    finally:
        engine.dispose()


def test_a_queue_filled_during_submission_leaves_no_orphan_run(
    tmp_path: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = write_config(
        tmp_path,
        """
jobs:
  max_queue_depth: 1
providers:
  text:
    type: mock-llm
formats:
  sample:
    description: A small priced recipe.
    stages:
      - stage: write
    capabilities: [llm]
    output_type: video
    handler: save
""",
    )
    original = JobQueue.enqueue

    def fill_then_enqueue(
        self: JobQueue,
        job: Job | None = None,
        *,
        max_queue_depth: int | None = None,
        connection: Connection | None = None,
    ) -> Job:
        if connection is not None:
            self.repository.create_in(connection, Job(kind="seed"))
        else:
            self.repository.create(Job(kind="seed"))
        return original(
            self,
            job,
            max_queue_depth=max_queue_depth,
            connection=connection,
        )

    monkeypatch.setattr(JobQueue, "enqueue", fill_then_enqueue)

    code, out, err = run_command(
        capsys,
        "--format",
        "sample",
        "--enqueue",
        "--config",
        str(config),
        "--database",
        str(database),
    )

    assert code == 1
    assert out == ""
    assert "queue is full" in err

    runs = tmp_path / "data" / "runs"
    assert not runs.exists() or list(runs.iterdir()) == []

    engine = create_database_engine(database)
    try:
        assert RunRepository(engine).list() == []
        assert JobRepository(engine).list() == []
    finally:
        engine.dispose()
