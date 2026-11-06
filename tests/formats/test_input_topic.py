"""The input topic stage builds the fact list every format starts from."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from aivia.core.errors import ConfigError
from aivia.core.models import Run
from aivia.core.models.topic import TopicCandidate
from aivia.core.stages import StageContext
from aivia.formats import (
    DEFAULT_TOPIC_FILENAME,
    INPUT_TOPIC_STAGE_NAME,
    InputTopicStage,
    build_input_topic_stage,
    resolve_input_topic,
    run_input_topic_stage,
)
from aivia.writing import WriteStage


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine)


def _context(run_dir: Path) -> StageContext:
    return StageContext(run_dir=run_dir, config={}, run=Run(id="run_topic"))


def _payload(run_dir: Path) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads((run_dir / DEFAULT_TOPIC_FILENAME).read_text(encoding="utf-8")),
    )


def test_cli_arguments_build_a_topic_without_research(tmp_path: Path) -> None:
    stage = InputTopicStage(topic="Honey never spoils.", title="Honey")

    result = _run(stage.run(_context(tmp_path)))

    assert result.stage == INPUT_TOPIC_STAGE_NAME
    assert result.status.value == "succeeded"
    assert result.cost_usd == 0.0
    assert result.artifacts == [DEFAULT_TOPIC_FILENAME]
    assert result.metadata["origin"] == "cli"
    assert result.metadata["title"] == "Honey"
    assert result.metadata["fact_count"] == 1

    payload = _payload(tmp_path)
    assert payload["title"] == "Honey"
    assert payload["facts"] == [
        {
            "text": "Honey never spoils.",
            "source_urls": [],
            "confidence": 1.0,
            "verified": False,
        }
    ]
    assert payload["source_urls"] == []
    assert "claims" not in payload
    assert stage.topic is not None
    assert stage.topic.title == "Honey"


def test_a_topic_file_supplies_title_claims_and_source_urls(tmp_path: Path) -> None:
    source = tmp_path / "topic.json"
    source.write_text(
        json.dumps(
            {
                "title": "Bridge history",
                "summary": "How the bridge was built.",
                "claims": [
                    {
                        "text": "The bridge opened in 1937.",
                        "source_urls": ["https://example.com/bridge"],
                        "confidence": 0.8,
                        "verified": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    stage = InputTopicStage(file=source)

    result = _run(stage.run(_context(tmp_path)))

    assert result.metadata["origin"] == "file"
    assert result.metadata["source"] == str(source)
    assert result.metadata["source_urls"] == ["https://example.com/bridge"]
    payload = _payload(tmp_path)
    assert payload["title"] == "Bridge history"
    assert payload["summary"] == "How the bridge was built."
    assert payload["facts"][0]["text"] == "The bridge opened in 1937."
    assert payload["claims"][0]["source_urls"] == ["https://example.com/bridge"]
    assert stage.topic is not None
    assert stage.topic.claims[0].verified is True


def test_a_plain_text_file_is_split_into_facts(tmp_path: Path) -> None:
    source = tmp_path / "topic.txt"
    source.write_text(
        "Honey never spoils.\nBees recognise human faces.\n",
        encoding="utf-8",
    )
    stage = InputTopicStage(topic_file=source)

    result = _run(stage.run(_context(tmp_path)))

    payload = _payload(tmp_path)
    assert payload["title"] == "Honey never spoils."
    assert [fact["text"] for fact in payload["facts"]] == [
        "Honey never spoils.",
        "Bees recognise human faces.",
    ]
    assert result.metadata["fact_count"] == 2
    assert payload["source_urls"] == []


def test_research_output_supplies_the_topic_without_importing_research(
    tmp_path: Path,
) -> None:
    research = SimpleNamespace(
        selected=[
            TopicCandidate(
                title="Solar panel costs",
                summary="Solar panels became much cheaper.",
                source_urls=["https://example.com/solar"],
            )
        ]
    )
    stage = InputTopicStage(research=research)

    result = _run(stage.run(_context(tmp_path)))

    payload = _payload(tmp_path)
    assert result.metadata["origin"] == "research"
    assert payload["title"] == "Solar panel costs"
    assert payload["facts"][0]["text"] == "Solar panels became much cheaper."
    assert payload["facts"][0]["source_urls"] == ["https://example.com/solar"]
    assert payload["source_urls"] == ["https://example.com/solar"]


def test_a_research_artifact_in_the_run_directory_is_used(tmp_path: Path) -> None:
    research = tmp_path / "research" / "facts.json"
    research.parent.mkdir()
    research.write_text(
        json.dumps(
            {
                "title": "Battery storage",
                "claims": [
                    {
                        "text": "Batteries store solar power.",
                        "source_urls": ["https://example.com/battery"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    stage = InputTopicStage()

    result = _run(stage.run(_context(tmp_path)))

    payload = _payload(tmp_path)
    assert result.metadata["origin"] == "research"
    assert result.metadata["source"] == str(research)
    assert payload["title"] == "Battery storage"
    assert payload["facts"][0]["text"] == "Batteries store solar power."


def test_a_missing_source_names_the_stage(tmp_path: Path) -> None:
    stage = InputTopicStage()

    with pytest.raises(ConfigError) as caught:
        _run(stage.run(_context(tmp_path)))

    assert "input topic" in str(caught.value)
    assert caught.value.details["stage"] == INPUT_TOPIC_STAGE_NAME


def test_a_missing_file_names_the_path(tmp_path: Path) -> None:
    stage = InputTopicStage(file=tmp_path / "missing.json")

    with pytest.raises(ConfigError) as caught:
        _run(stage.run(_context(tmp_path)))

    assert "missing.json" in str(caught.value)
    assert caught.value.details["path"] == str(tmp_path / "missing.json")


def test_an_invalid_source_url_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "topic.json"
    source.write_text(
        json.dumps(
            {
                "title": "Bad source",
                "claims": [{"text": "A fact.", "source_urls": ["not-a-url"]}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="absolute http"):
        _run(InputTopicStage(file=source).run(_context(tmp_path)))


def test_the_written_topic_is_consumed_by_the_write_stage(tmp_path: Path) -> None:
    class Writer:
        def write(self, facts: list[Any], languages: list[str], **_: Any) -> Any:
            return {
                code: {"hook": facts[0].text, "beats": [{"text": facts[0].text}]}
                for code in languages
            }

    _run(InputTopicStage(topic="Honey never spoils.").run(_context(tmp_path)))
    write = WriteStage(options={"languages": ["en"]}, writer=Writer())

    result = _run(write.run(_context(tmp_path)))

    assert result.artifacts == ["script.en.json"]
    script = json.loads((tmp_path / "script.en.json").read_text(encoding="utf-8"))
    assert script["hook"] == "Honey never spoils."


def test_helpers_build_resolve_and_run_the_same_stage(tmp_path: Path) -> None:
    resolved = resolve_input_topic("A topic from the CLI")

    assert resolved.title == "A topic from the CLI"
    assert resolved.facts[0]["text"] == "A topic from the CLI"

    stage = build_input_topic_stage(topic="Another topic")
    result = _run(run_input_topic_stage(_context(tmp_path), stage=stage))

    assert result.artifacts == [DEFAULT_TOPIC_FILENAME]
    assert _payload(tmp_path)["title"] == "Another topic"


def test_build_stages_uses_the_concrete_input_topic_stage() -> None:
    from aivia.cli.run import build_stages
    from aivia.core.config.formats import FormatRecipeConfig

    recipe = FormatRecipeConfig.model_validate(
        {"stages": [{"stage": "input_topic", "options": {"topic": "A topic"}}]}
    )

    stages = build_stages(recipe)

    assert isinstance(stages[0], InputTopicStage)
    assert stages[0].options["topic"] == "A topic"
