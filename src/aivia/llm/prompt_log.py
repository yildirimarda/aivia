"""Audit log for LLM calls that records the prompt template that produced them.

Every call records the prompt name and version so a run can be audited and
replayed. Prompt content is included only when privacy allows it: it is omitted
when ``store_prompt_text`` is false or ``redact_prompts`` is true, and otherwise
written with known secrets masked.

The log is a data object plus a sink. By default it writes one JSON object per
call through the ``aivia.llm.prompt_log`` logger; tests and alternate exporters
can pass a callable sink instead. Nothing here contacts a provider or knows a
model id.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from aivia.core.config.sections import PrivacyConfig
from aivia.core.logging import get_logger
from aivia.core.redact import redact

__all__ = [
    "PromptLog",
    "PromptLogEntry",
    "PromptLogSink",
    "prompt_content_for_log",
]

PromptLogSink = Callable[["PromptLogEntry"], None]
"""Receives each recorded entry when a :class:`PromptLog` uses a custom sink."""

_DEFAULT_LOGGER = "aivia.llm.prompt_log"


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def prompt_content_for_log(
    content: str | None,
    privacy: PrivacyConfig | None = None,
) -> str | None:
    """Return the prompt text that may be logged, or ``None`` to omit it.

    Content is omitted entirely when prompt storage is disabled or prompts are
    redacted. Otherwise known secrets and token-like strings are masked so a
    prompt can never leak a credential into a log.
    """
    if content is None:
        return None
    if not isinstance(content, str):
        raise TypeError("prompt content must be a string or None")
    policy = privacy or PrivacyConfig()
    if not policy.store_prompt_text or policy.redact_prompts:
        return None
    return redact(content)


@dataclass(frozen=True, slots=True)
class PromptLogEntry:
    """One LLM call with the prompt identity and optionally its content."""

    name: str
    version: str
    content: str | None = None
    provider: str = ""
    model: str = ""
    run_id: str = ""
    operation: str = "llm"
    cache_hit: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping, omitting content when redacted."""
        payload: dict[str, Any] = {
            "prompt_name": self.name,
            "prompt_version": self.version,
            "provider": self.provider,
            "model": self.model,
            "run_id": self.run_id,
            "operation": self.operation,
            "cache_hit": self.cache_hit,
        }
        if self.content is not None:
            payload["content"] = self.content
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


class PromptLog:
    """Record the prompt name, version and content of every LLM call."""

    def __init__(
        self,
        *,
        privacy: PrivacyConfig | None = None,
        logger: logging.Logger | None = None,
        sink: PromptLogSink | None = None,
    ) -> None:
        if privacy is not None and not isinstance(privacy, PrivacyConfig):
            raise TypeError("privacy must be a PrivacyConfig")
        if sink is not None and not callable(sink):
            raise TypeError("sink must be callable")
        self.privacy = privacy or PrivacyConfig()
        self._logger = logger or get_logger(_DEFAULT_LOGGER)
        self._sink = sink

    @property
    def sink(self) -> PromptLogSink | None:
        """The custom sink, when one was configured."""
        return self._sink

    def record(
        self,
        name: str,
        version: str = "",
        content: str | None = None,
        *,
        provider: str = "",
        model: str = "",
        run_id: str = "",
        operation: str = "llm",
        cache_hit: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> PromptLogEntry:
        """Record one call and return the entry that was emitted."""
        if not isinstance(name, str):
            raise TypeError("prompt name must be a string")
        if not isinstance(version, str):
            raise TypeError("prompt version must be a string")
        if not isinstance(cache_hit, bool):
            raise TypeError("cache_hit must be a boolean")
        entry = PromptLogEntry(
            name=name.strip(),
            version=version.strip(),
            content=prompt_content_for_log(content, self.privacy),
            provider=_text(provider),
            model=_text(model),
            run_id=_text(run_id),
            operation=_text(operation) or "llm",
            cache_hit=cache_hit,
            metadata=dict(metadata or {}),
        )
        if self._sink is not None:
            self._sink(entry)
        else:
            self._logger.info(
                "%s",
                json.dumps(
                    entry.to_dict(),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                    default=str,
                ),
            )
        return entry
