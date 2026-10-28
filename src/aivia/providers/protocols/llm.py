"""Typed LLM protocol, request and response models.

An LLM provider turns a validated :class:`LLMRequest` into an
:class:`LLMResponse`. The request carries messages, optional multimodal
content, sampling controls, tools and a JSON schema for structured output.
The response carries the text, tool calls, token usage and a finish reason.

The protocol deliberately says nothing about vendors, model ids or prices:
those belong to adapter modules, ``models.yaml`` and configuration. Role and
model resolution happen before the call, so a provider only executes it.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal, Protocol, Self, runtime_checkable

from pydantic import Field, field_validator, model_validator

from aivia.providers.protocols.base import (
    CapabilityProvider,
    ProtocolModel,
    validate_http_url,
)

__all__ = [
    "CAPABILITY_LLM",
    "ChatMessage",
    "ContentPart",
    "FinishReason",
    "ImageDetail",
    "ImagePart",
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "LLMUsage",
    "MessageRole",
    "MessageRoleEnum",
    "TextPart",
    "ToolCall",
    "ToolChoice",
    "ToolDefinition",
]

CAPABILITY_LLM = "llm"
"""The capability name an LLM provider declares."""


class MessageRole(StrEnum):
    """The role of one chat message."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


MessageRoleEnum = MessageRole
"""Alias kept for callers that prefer the explicit enum suffix."""


class ImageDetail(StrEnum):
    """How much detail to request for an image input."""

    AUTO = "auto"
    LOW = "low"
    HIGH = "high"


class FinishReason(StrEnum):
    """Why the model stopped generating."""

    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"


class ToolChoice(StrEnum):
    """Whether the model must, may or must not call a tool."""

    AUTO = "auto"
    NONE = "none"
    REQUIRED = "required"


class TextPart(ProtocolModel):
    """A text segment inside a multimodal message."""

    type: Literal["text"] = "text"
    text: str = Field(min_length=1)

    @field_validator("text", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value


class ImagePart(ProtocolModel):
    """An image input supplied either by URL or as base64 data.

    Exactly one of ``url`` and ``data`` must be set. ``media_type`` is required
    with inline data so an adapter never has to guess the encoding.
    """

    type: Literal["image"] = "image"
    url: str | None = None
    data: str | None = None
    media_type: str | None = None
    detail: ImageDetail = ImageDetail.AUTO

    @field_validator("url")
    @classmethod
    def _validate_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_http_url(value, field="image url")

    @field_validator("data", "media_type", mode="before")
    @classmethod
    def _strip_optional_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip()
            return text or None
        return value

    @model_validator(mode="after")
    def _validate_source(self) -> Self:
        sources = (self.url is not None, self.data is not None)
        if sum(sources) != 1:
            raise ValueError("image part needs exactly one of url or data")
        if self.data is not None and self.media_type is None:
            raise ValueError("inline image data requires media_type")
        if self.media_type is not None and not self.media_type.startswith("image/"):
            raise ValueError("media_type must start with 'image/'")
        return self


ContentPart = Annotated[TextPart | ImagePart, Field(discriminator="type")]
"""One typed segment of a multimodal message."""


class ToolDefinition(ProtocolModel):
    """A tool the model may call, described by a JSON schema."""

    name: str = Field(min_length=1)
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name", "description", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value


class ToolCall(ProtocolModel):
    """One tool invocation requested by the model."""

    id: str = ""
    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "name", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value


class ChatMessage(ProtocolModel):
    """One message in a conversation.

    ``content`` is either a plain string or a list of typed text and image
    parts. An assistant message may be empty when it only carries tool calls,
    and a tool message must name the call it answers.
    """

    role: MessageRole
    content: str | list[ContentPart]
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)

    @field_validator("role", mode="before")
    @classmethod
    def _coerce_role(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("content", mode="before")
    @classmethod
    def _strip_content(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("name", "tool_call_id", mode="before")
    @classmethod
    def _strip_optional_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip()
            return text or None
        return value

    @model_validator(mode="after")
    def _validate_message(self) -> Self:
        has_parts = isinstance(self.content, list) and bool(self.content)
        if not self.text and not has_parts and not self.tool_calls:
            raise ValueError("message must carry text, content parts or tool calls")
        if self.role is MessageRole.TOOL and self.tool_call_id is None:
            raise ValueError("a tool message requires tool_call_id")
        return self

    @property
    def text(self) -> str:
        """The text of the message, joining text parts and ignoring images."""
        if isinstance(self.content, str):
            return self.content
        return "\n".join(
            part.text for part in self.content if isinstance(part, TextPart)
        )

    @property
    def images(self) -> list[ImagePart]:
        """Every image part in the message, in order."""
        if isinstance(self.content, str):
            return []
        return [part for part in self.content if isinstance(part, ImagePart)]


class LLMUsage(ProtocolModel):
    """Token usage reported by a provider."""

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        """Input plus output tokens."""
        return self.input_tokens + self.output_tokens

    @property
    def prompt_tokens(self) -> int:
        """Alias for :attr:`input_tokens`."""
        return self.input_tokens

    @property
    def completion_tokens(self) -> int:
        """Alias for :attr:`output_tokens`."""
        return self.output_tokens


class LLMRequest(ProtocolModel):
    """A validated request to one LLM provider.

    ``role`` is resolved to a model before the call; ``model`` may be set
    explicitly when a caller already knows it. ``response_schema`` asks for
    structured JSON and is validated by the caller, not the adapter.
    """

    messages: list[ChatMessage] = Field(min_length=1)
    role: str = "default"
    model: str | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    stop: list[str] = Field(default_factory=list)
    seed: int | None = None
    response_schema: dict[str, Any] | None = None
    tools: list[ToolDefinition] = Field(default_factory=list)
    tool_choice: ToolChoice = ToolChoice.AUTO
    timeout_seconds: float | None = Field(default=None, gt=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("role", mode="before")
    @classmethod
    def _validate_role(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip()
            if not text:
                raise ValueError("role must be non-empty")
            return text
        return value

    @field_validator("model", mode="before")
    @classmethod
    def _strip_model(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip()
            return text or None
        return value

    @field_validator("stop", mode="before")
    @classmethod
    def _coerce_stop(cls, value: Any) -> Any:
        if value is None:
            return []
        return value

    @field_validator("stop")
    @classmethod
    def _validate_stop(cls, value: list[str]) -> list[str]:
        sequences: list[str] = []
        for entry in value:
            if not isinstance(entry, str) or not entry:
                raise ValueError("stop sequences must be non-empty strings")
            sequences.append(entry)
        return sequences

    @model_validator(mode="after")
    def _validate_tools(self) -> Self:
        names = [tool.name for tool in self.tools]
        if len(names) != len(set(names)):
            raise ValueError("tool names must be unique")
        if self.tool_choice is ToolChoice.REQUIRED and not self.tools:
            raise ValueError("tool_choice 'required' needs at least one tool")
        return self

    @property
    def last_user_text(self) -> str:
        """The text of the last user message, or an empty string."""
        for message in reversed(self.messages):
            if message.role is MessageRole.USER:
                return message.text
        return ""


class LLMResponse(ProtocolModel):
    """A validated response from one LLM provider."""

    content: str = ""
    model: str = ""
    provider: str = ""
    usage: LLMUsage = Field(default_factory=LLMUsage)
    finish_reason: FinishReason = FinishReason.STOP
    tool_calls: list[ToolCall] = Field(default_factory=list)
    request_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("model", "provider", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("request_id", mode="before")
    @classmethod
    def _strip_request_id(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip()
            return text or None
        return value

    @property
    def text(self) -> str:
        """Alias for :attr:`content`."""
        return self.content

    @property
    def total_tokens(self) -> int:
        """Alias for :attr:`LLMUsage.total_tokens`."""
        return self.usage.total_tokens

    @property
    def has_tool_calls(self) -> bool:
        """Whether the model requested at least one tool call."""
        return bool(self.tool_calls)


@runtime_checkable
class LLMProvider(CapabilityProvider, Protocol):
    """A provider that serves the ``llm`` capability."""

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Execute *request* and return a validated response."""
