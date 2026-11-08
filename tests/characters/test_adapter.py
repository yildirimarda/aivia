"""Trained character adapters store model id, trigger words and training cost,
and unusable adapters are skipped with a reason.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from aivia.characters.adapter import AdapterUnusableError, TrainerAdapter
from aivia.providers.protocols.character_trainer import (
    CharacterTrainingJob,
    TrainingState,
)


def _job(
    *,
    model_id: str | None = "adapter-1",
    state: TrainingState = TrainingState.READY,
    cost_usd: float = 1.25,
) -> CharacterTrainingJob:
    return CharacterTrainingJob(
        job_id="train-1",
        character_id="ada",
        state=state,
        model_id=model_id,
        trigger_words=["ada"],
        cost_usd=cost_usd,
        created_at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        completed_at=datetime(2026, 1, 2, 4, 5, 6, tzinfo=UTC),
    )


def test_adapter_stores_model_id_trigger_words_and_cost() -> None:
    adapter = TrainerAdapter(
        model_id="  adapter-1  ",
        trigger_words=["  ada  ", "ada", "  narrator  "],
        cost_usd=1.25,
    )

    assert adapter.model_id == "adapter-1"
    assert adapter.trigger_words == ["ada", "narrator"]
    assert adapter.cost_usd == 1.25


def test_adapter_exposes_model_triggers_and_cost_aliases() -> None:
    adapter = TrainerAdapter(model_id="adapter-1", trigger_words=["ada"], cost_usd=2.5)

    assert adapter.model == "adapter-1"
    assert adapter.triggers == ["ada"]
    assert adapter.cost == 2.5


def test_adapter_from_job_builds_from_a_ready_training_job() -> None:
    adapter = TrainerAdapter.from_job(_job())

    assert adapter.model_id == "adapter-1"
    assert adapter.trigger_words == ["ada"]
    assert adapter.cost_usd == 1.25
    assert adapter.is_usable is True
    assert adapter.reason == ""


def test_adapter_from_job_with_no_model_id_is_unusable() -> None:
    adapter = TrainerAdapter.from_job(_job(model_id=None, state=TrainingState.PENDING))

    assert adapter.model_id == ""
    assert adapter.is_usable is False
    assert adapter.reason == "adapter has no model id"


def test_adapter_from_a_failed_job_is_unusable() -> None:
    adapter = TrainerAdapter.from_job(
        _job(model_id=None, state=TrainingState.FAILED, cost_usd=0.0)
    )

    assert adapter.is_usable is False
    assert adapter.cost_usd == 0.0


def test_an_adapter_without_a_model_id_is_unusable() -> None:
    adapter = TrainerAdapter(trigger_words=["ada"], cost_usd=1.0)

    assert adapter.is_usable is False
    assert adapter.reason == "adapter has no model id"


def test_require_usable_returns_the_adapter_when_usable() -> None:
    adapter = TrainerAdapter(model_id="adapter-1", trigger_words=["ada"], cost_usd=1.0)

    assert adapter.require_usable() is adapter


def test_require_usable_raises_with_a_reason_when_unusable() -> None:
    adapter = TrainerAdapter(trigger_words=["ada"], cost_usd=1.0)

    with pytest.raises(AdapterUnusableError) as caught:
        adapter.require_usable()

    assert caught.value.reason == "adapter has no model id"
    assert "adapter has no model id" in str(caught.value)
    assert isinstance(caught.value, ValueError)


def test_adapter_rejects_blank_trigger_words() -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        TrainerAdapter(model_id="adapter-1", trigger_words=["  "])

    with pytest.raises(ValidationError, match="non-empty"):
        TrainerAdapter(model_id="adapter-1", trigger_words=["ada", ""])


def test_adapter_rejects_negative_training_cost() -> None:
    with pytest.raises(ValidationError):
        TrainerAdapter(model_id="adapter-1", cost_usd=-1.0)


def test_adapter_is_immutable_and_rejects_unknown_fields() -> None:
    adapter = TrainerAdapter(model_id="adapter-1", trigger_words=["ada"], cost_usd=1.0)

    with pytest.raises(ValidationError):
        adapter.model_id = "changed"

    with pytest.raises(ValidationError):
        TrainerAdapter.model_validate(
            {
                "model_id": "adapter-1",
                "trigger_words": ["ada"],
                "unknown": True,
            }
        )
