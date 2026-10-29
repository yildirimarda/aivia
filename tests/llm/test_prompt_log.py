"""Prompt audit logging: name, version and privacy-aware content."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Coroutine, Iterator
from typing import Any

import pytest
from pydantic import BaseModel, Field
from sqlalchemy import Engine

from aivia.core.config.sections import PrivacyConfig
from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.redact import REDACTED, clear_secrets, register_secret
from aivia.llm import (
    LLMCache,
    PromptLog,
    PromptLogEntry,
    StructuredOutputHelper,
    prompt_content_for_log,
)
from aivia.providers.protocols.llm import (
    ChatMessage,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    MessageRole,
)

MODEL = "model-a"
PROVIDER = "acme"


class Item(BaseModel):
    title: str = Field(min_length=1)


class FakeProvider:
    """An async LLM provider that returns one canned response."""

    name = PROVIDER

    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return LLMResponse(
            content='{"title": "hello"}',
            model=MODEL,
            provider=PROVIDER,
            usage=LLMUsage(input_tokens=10, output_tokens=5),
        )


@pytest.fixture
def engine() -> Iterator[Engine]:
    created = create_database_engine(url="sqlite+pysqlite:///:memory:")
    upgrade_database(engine=created)
    try:
        yield created
    finally:
        created.dispose()


def _request(text: str = "make an item") -> LLMRequest:
    return LLMRequest(messages=[ChatMessage(role=MessageRole.USER, content=text)])


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine)


def test_a_call_records_the_prompt_name_version_and_content() -> None:
    entries: list[PromptLogEntry] = []
    log = PromptLog(sink=entries.append)

    entry = log.record(
        "image.md",
        "user",
        "Draw a fox.",
        provider=PROVIDER,
        model=MODEL,
        run_id="run_1",
    )

    assert entries == [entry]
    assert entry.name == "image.md"
    assert entry.version == "user"
    assert entry.content == "Draw a fox."
    assert entry.provider == PROVIDER
    assert entry.model == MODEL
    assert entry.run_id == "run_1"
    assert entry.operation == "llm"
    assert entry.cache_hit is False
    assert entry.to_dict() == {
        "prompt_name": "image.md",
        "prompt_version": "user",
        "content": "Draw a fox.",
        "provider": PROVIDER,
        "model": MODEL,
        "run_id": "run_1",
        "operation": "llm",
        "cache_hit": False,
    }


def test_the_default_sink_writes_one_json_log_record(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="aivia.llm.prompt_log"):
        PromptLog().record("image.md", "user", "Draw a fox.")

    payload = json.loads(caplog.records[0].getMessage())
    assert payload["prompt_name"] == "image.md"
    assert payload["prompt_version"] == "user"
    assert payload["content"] == "Draw a fox."


def test_redacted_prompts_omit_content_entirely() -> None:
    entries: list[PromptLogEntry] = []
    log = PromptLog(
        privacy=PrivacyConfig(redact_prompts=True),
        sink=entries.append,
    )

    entry = log.record("image.md", "user", "Draw a secret fox.")

    assert entry.content is None
    assert "content" not in entry.to_dict()
    assert entries[0].content is None


def test_disabled_prompt_storage_omits_content_entirely() -> None:
    entry = PromptLog(
        privacy=PrivacyConfig(store_prompt_text=False),
        sink=lambda recorded: None,
    ).record("image.md", "user", "Draw a fox.")

    assert entry.content is None
    assert "content" not in entry.to_dict()


def test_retained_content_masks_registered_secrets() -> None:
    register_secret("super-secret-token")
    try:
        entry = PromptLog(sink=lambda recorded: None).record(
            "image.md",
            "user",
            "Use super-secret-token for the request.",
        )
    finally:
        clear_secrets()

    assert entry.content == f"Use {REDACTED} for the request."
    assert "super-secret-token" not in str(entry.to_dict())


def test_content_helper_returns_none_when_privacy_disables_prompts() -> None:
    assert prompt_content_for_log("hello") == "hello"
    assert (
        prompt_content_for_log(
            "hello",
            PrivacyConfig(redact_prompts=True),
        )
        is None
    )
    assert (
        prompt_content_for_log(
            "hello",
            PrivacyConfig(store_prompt_text=False),
        )
        is None
    )


def test_the_structured_helper_records_every_call() -> None:
    entries: list[PromptLogEntry] = []
    provider = FakeProvider()
    helper = StructuredOutputHelper(
        provider,
        Item,
        prompt_name="image.md",
        prompt_version="v1",
        prompt_log=PromptLog(sink=entries.append),
        run_id="run_1",
    )

    _run(helper.complete(_request("first prompt")))
    _run(helper.complete(_request("second prompt")))

    assert len(provider.requests) == 2
    assert [(entry.name, entry.version) for entry in entries] == [
        ("image.md", "v1"),
        ("image.md", "v1"),
    ]
    assert [entry.content for entry in entries] == ["first prompt", "second prompt"]
    assert [entry.provider for entry in entries] == [PROVIDER, PROVIDER]
    assert [entry.model for entry in entries] == [MODEL, MODEL]
    assert [entry.run_id for entry in entries] == ["run_1", "run_1"]
    assert [entry.cache_hit for entry in entries] == [False, False]


def test_a_request_can_supply_the_prompt_identity() -> None:
    entries: list[PromptLogEntry] = []
    helper = StructuredOutputHelper(
        FakeProvider(),
        Item,
        prompt_log=PromptLog(sink=entries.append),
    )
    request = _request().model_copy(
        update={"prompt_name": "image.md", "prompt_version": "user"}
    )

    _run(helper.complete(request))

    assert (entries[0].name, entries[0].version) == ("image.md", "user")


def test_the_helper_prompt_identity_wins_over_the_request() -> None:
    entries: list[PromptLogEntry] = []
    helper = StructuredOutputHelper(
        FakeProvider(),
        Item,
        prompt_name="image.md",
        prompt_version="v1",
        prompt_log=PromptLog(sink=entries.append),
    )
    request = _request().model_copy(
        update={"prompt_name": "other.md", "prompt_version": "user"}
    )

    _run(helper.complete(request))

    assert (entries[0].name, entries[0].version) == ("image.md", "v1")


def test_the_helper_logs_a_cache_hit_without_calling_the_provider(
    engine: Engine,
) -> None:
    entries: list[PromptLogEntry] = []
    provider = FakeProvider()
    helper = StructuredOutputHelper(
        provider,
        Item,
        cache=LLMCache(engine),
        prompt_name="image.md",
        prompt_version="v1",
        prompt_log=PromptLog(sink=entries.append),
        run_id="run_1",
    )

    first = _run(helper.complete(_request()))
    second = _run(helper.complete(_request()))

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert len(provider.requests) == 1
    assert [entry.cache_hit for entry in entries] == [False, True]
    assert [(entry.name, entry.version) for entry in entries] == [
        ("image.md", "v1"),
        ("image.md", "v1"),
    ]


def test_the_helper_omits_content_for_redacted_prompts() -> None:
    entries: list[PromptLogEntry] = []
    helper = StructuredOutputHelper(
        FakeProvider(),
        Item,
        prompt_name="image.md",
        prompt_version="v1",
        prompt_log=PromptLog(
            privacy=PrivacyConfig(redact_prompts=True),
            sink=entries.append,
        ),
    )

    _run(helper.complete(_request("a private prompt")))

    assert entries[0].name == "image.md"
    assert entries[0].version == "v1"
    assert entries[0].content is None
    assert "content" not in entries[0].to_dict()


def test_a_failed_call_is_still_recorded() -> None:
    entries: list[PromptLogEntry] = []
    helper = StructuredOutputHelper(
        FakeProvider(error=RuntimeError("provider down")),
        Item,
        prompt_name="image.md",
        prompt_version="v1",
        prompt_log=PromptLog(sink=entries.append),
    )

    with pytest.raises(RuntimeError, match="provider down"):
        _run(helper.complete(_request("a failing prompt")))

    assert len(entries) == 1
    assert entries[0].content == "a failing prompt"
    assert entries[0].cache_hit is False
