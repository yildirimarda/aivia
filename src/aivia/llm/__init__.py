"""LLM helpers built on the provider protocol.

The package holds validation, prompt and repair utilities that are independent
of any vendor adapter. Provider calls still go through
:class:`~aivia.providers.protocols.llm.LLMProvider`; nothing here knows a model
id or a price.
"""

from aivia.llm.repair import (
    RepairHelper,
    RepairingStructuredOutput,
    RepairLoop,
    StructuredOutputRepair,
    StructuredRepair,
    StructuredRepairLoop,
    build_repair_prompt,
    complete_structured_with_repair,
    complete_with_repair,
    repair_loop,
    repair_messages,
    repair_structured,
    repair_structured_output,
    structured_repair,
)
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
    "RepairHelper",
    "RepairLoop",
    "RepairingStructuredOutput",
    "StructuredCompletion",
    "StructuredLLM",
    "StructuredOutput",
    "StructuredOutputError",
    "StructuredOutputHelper",
    "StructuredOutputRepair",
    "StructuredOutputResult",
    "StructuredRepair",
    "StructuredRepairLoop",
    "StructuredResult",
    "StructuredValue",
    "build_repair_prompt",
    "complete_structured",
    "complete_structured_with_repair",
    "complete_with_repair",
    "complete_with_schema",
    "generate_structured",
    "parse_structured",
    "parse_structured_response",
    "repair_loop",
    "repair_messages",
    "repair_structured",
    "repair_structured_output",
    "structured_complete",
    "structured_repair",
    "validate_structured",
]
