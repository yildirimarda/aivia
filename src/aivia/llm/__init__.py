"""LLM helpers built on the provider protocol.

The package holds validation, prompt and repair utilities that are independent
of any vendor adapter. Provider calls still go through
:class:`~aivia.providers.protocols.llm.LLMProvider`; nothing here knows a model
id or a price.
"""

from aivia.llm.structured import (
    StructuredCompletion,
    StructuredLLM,
    StructuredOutput,
    StructuredOutputError,
    StructuredOutputHelper,
    StructuredOutputResult,
    StructuredResult,
    StructuredValue,
    complete_structured,
    complete_with_schema,
    generate_structured,
    parse_structured,
    parse_structured_response,
    structured_complete,
    validate_structured,
)

__all__ = [
    "StructuredCompletion",
    "StructuredLLM",
    "StructuredOutput",
    "StructuredOutputError",
    "StructuredOutputHelper",
    "StructuredOutputResult",
    "StructuredResult",
    "StructuredValue",
    "complete_structured",
    "complete_with_schema",
    "generate_structured",
    "parse_structured",
    "parse_structured_response",
    "structured_complete",
    "validate_structured",
]
