"""Structured JSON logging contract."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta
from io import StringIO
from typing import Any, cast

import pytest

from aivia.core.logging import (
    CONTEXT_VARS,
    JsonFormatter,
    configure_logging,
    current_context,
    log_context,
    provider,
    run_id,
    stage,
)


def _record(
    *,
    name: str = "aivia.test",
    level: int = logging.INFO,
    msg: str = "hello",
    args: tuple[object, ...] | None = None,
) -> logging.LogRecord:
    return logging.LogRecord(name, level, __file__, 1, msg, args, None)


def _payload(record: logging.LogRecord) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(JsonFormatter().format(record)))


def test_json_log_has_the_required_fields() -> None:
    payload = _payload(_record())

    assert {"ts", "level", "msg", "logger"} <= set(payload)
    assert payload["level"] == "INFO"
    assert payload["msg"] == "hello"
    assert payload["logger"] == "aivia.test"

    timestamp = datetime.fromisoformat(str(payload["ts"]).replace("Z", "+00:00"))
    assert timestamp.utcoffset() == timedelta(0)


def test_message_arguments_are_rendered() -> None:
    payload = _payload(_record(msg="processed %s of %d", args=("item", 3)))

    assert payload["msg"] == "processed item of 3"


def test_context_variables_are_added_to_every_log() -> None:
    tokens = (run_id.set("run_01"), stage.set("write"), provider.set("mock"))
    try:
        payload = _payload(_record())
    finally:
        provider.reset(tokens[2])
        stage.reset(tokens[1])
        run_id.reset(tokens[0])

    assert payload["run_id"] == "run_01"
    assert payload["stage"] == "write"
    assert payload["provider"] == "mock"


def test_unset_context_variables_are_omitted() -> None:
    payload = _payload(_record())

    for name, _ in CONTEXT_VARS:
        assert name not in payload


def test_log_context_sets_values_and_restores_them() -> None:
    with log_context(run_id="run_01", stage="write"):
        assert current_context() == {"run_id": "run_01", "stage": "write"}
        payload = _payload(_record())

    assert payload["run_id"] == "run_01"
    assert payload["stage"] == "write"
    assert current_context() == {}


def test_log_context_nests_and_restores_outer_values() -> None:
    with log_context(run_id="run_01", stage="write"):
        with log_context(stage="render", provider="mock"):
            assert current_context() == {
                "run_id": "run_01",
                "stage": "render",
                "provider": "mock",
            }
        assert current_context() == {"run_id": "run_01", "stage": "write"}
    assert current_context() == {}


def test_log_context_can_clear_a_value_explicitly() -> None:
    with log_context(stage="write"):
        with log_context(stage=None):
            assert current_context() == {}
        assert current_context() == {"stage": "write"}


def test_log_context_restores_outer_values_after_an_exception() -> None:
    with log_context(run_id="run_01"):
        with pytest.raises(ValueError, match="boom"), log_context(stage="write"):
            raise ValueError("boom")
        assert current_context() == {"run_id": "run_01"}
    assert current_context() == {}


def test_log_context_is_isolated_between_async_tasks() -> None:
    async def observe(run: str) -> list[dict[str, str]]:
        with log_context(run_id=run):
            await asyncio.sleep(0)
            first = current_context()
            await asyncio.sleep(0)
            second = current_context()
        return [first, second]

    async def main() -> list[list[dict[str, str]]]:
        results = await asyncio.gather(observe("run_a"), observe("run_b"))
        return list(results)

    assert asyncio.run(main()) == [
        [{"run_id": "run_a"}, {"run_id": "run_a"}],
        [{"run_id": "run_b"}, {"run_id": "run_b"}],
    ]
    assert current_context() == {}


def test_log_context_supports_async_with() -> None:
    async def main() -> dict[str, str]:
        async with log_context(run_id="run_01", stage="write"):
            await asyncio.sleep(0)
            inside = current_context()
        assert current_context() == {}
        return inside

    assert asyncio.run(main()) == {"run_id": "run_01", "stage": "write"}
    assert current_context() == {}


def test_log_context_cannot_be_entered_twice() -> None:
    context = log_context(run_id="run_01")
    with context, pytest.raises(RuntimeError, match="already active"):
        context.__enter__()
    assert current_context() == {}


def test_configure_logging_writes_json_to_the_stream() -> None:
    stream = StringIO()
    logger = logging.getLogger("aivia.test.configure")
    previous_level = logger.level
    previous_propagate = logger.propagate
    previous_handlers = list(logger.handlers)

    try:
        configure_logging(logging.DEBUG, stream=stream, logger=logger)
        token = run_id.set("run_42")
        try:
            logger.debug("processed %s", "item")
        finally:
            run_id.reset(token)

        payload = json.loads(stream.getvalue())
    finally:
        logger.handlers[:] = previous_handlers
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate

    assert payload["level"] == "DEBUG"
    assert payload["msg"] == "processed item"
    assert payload["logger"] == "aivia.test.configure"
    assert payload["run_id"] == "run_42"


def test_configure_logging_replaces_existing_handlers() -> None:
    stream = StringIO()
    logger = logging.getLogger("aivia.test.replace")
    previous_level = logger.level
    previous_propagate = logger.propagate
    previous_handlers = list(logger.handlers)
    logger.addHandler(logging.NullHandler())

    try:
        handler = configure_logging(logging.INFO, stream=stream, logger=logger)

        assert logger.handlers == [handler]
        assert logger.level == logging.INFO
        assert logger.propagate is False
    finally:
        logger.handlers[:] = previous_handlers
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate
