"""Prompt-only fallback when no provider supports character references."""

from __future__ import annotations

import logging

import pytest

from aivia.characters.capabilities import (
    CAPABILITY_CHARACTER_REFERENCE,
    prepare_character_request,
)
from aivia.characters.schema import Character
from aivia.core.models import ProviderMeta
from aivia.providers import Provider


class FakeProvider(Provider):
    def __init__(
        self,
        name: str,
        capabilities: list[str],
    ) -> None:
        super().__init__(
            ProviderMeta(
                name=name,
                capabilities=capabilities,
            )
        )


def _character(*, references: int = 2, name: str = "Mascot") -> Character:
    images = [f"ref-{i}.png" for i in range(references)]
    return Character.model_validate({"name": name, "reference_images": images})


def test_no_character_capable_provider_falls_back_to_prompt_only_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    without = FakeProvider("without", [])

    with caplog.at_level(logging.WARNING):
        providers, warning = prepare_character_request(
            _character(), [without], max_reference_images=4
        )

    assert warning is not None
    assert CAPABILITY_CHARACTER_REFERENCE in warning
    assert "prompt-only generation" in warning
    assert providers == (without,)
    assert any(
        record.levelname == "WARNING"
        and "no provider declares character_reference" in record.getMessage()
        for record in caplog.records
    )


def test_fallback_emits_a_user_warning() -> None:
    without = FakeProvider("without", [])

    with pytest.warns(UserWarning, match="prompt-only generation"):
        prepare_character_request(_character(), [without], max_reference_images=4)


def test_empty_provider_list_with_references_falls_back(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        providers, warning = prepare_character_request(
            _character(), [], max_reference_images=4
        )

    assert warning is not None
    assert CAPABILITY_CHARACTER_REFERENCE in warning
    assert providers == ()
    assert any(record.levelname == "WARNING" for record in caplog.records)


def test_no_references_returns_all_providers_without_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    without = FakeProvider("without", [])
    with_flag = FakeProvider("with-flag", [CAPABILITY_CHARACTER_REFERENCE])

    with caplog.at_level(logging.WARNING):
        providers, warning = prepare_character_request(
            _character(references=0), [without, with_flag], max_reference_images=4
        )

    assert warning is None
    assert providers == (without, with_flag)
    assert not any(record.levelname == "WARNING" for record in caplog.records)
