"""Visual self check: verify a rendered preview matches the instruction.

Renders a composition to a still preview, asks a vision-capable LLM provider
whether the preview matches the instruction, and when it does not, asks the
provider for a composition patch and re-renders. The loop is bounded by the
planner's ``max_iterations``; when the limit is reached the last failure is
raised as :class:`SelfCheckError`.

config-coverage: allow-file
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Any, cast

from PIL import Image
from pydantic import BaseModel, Field

from aivia.compose import render_still
from aivia.compose.patch import CompositionPatch, PatchOperation, apply_patch
from aivia.compose.schema import Composition
from aivia.core.config.planner import PlannerConfig
from aivia.core.errors import AiviaError
from aivia.llm.structured import parse_structured
from aivia.providers.protocols.llm import (
    ChatMessage,
    ImageDetail,
    ImagePart,
    LLMRequest,
    LLMResponse,
    MessageRole,
    TextPart,
)


class SelfCheckError(AiviaError):
    """The preview could not be brought in line with the instruction."""

    code = "self_check_error"

    def __init__(
        self,
        message: str,
        *,
        iterations: int = 0,
        max_iterations: int = 0,
        reason: str = "",
    ) -> None:
        super().__init__(
            message,
            details={
                "iterations": iterations,
                "max_iterations": max_iterations,
                "reason": reason,
            },
        )
        self.iterations = iterations
        self.max_iterations = max_iterations
        self.reason = reason


class _Verdict(BaseModel):
    """The provider's judgement of one preview against the instruction."""

    passed: bool = Field(..., description="Whether the preview matches the instruction")
    reason: str = Field(default="", description="Why the preview passed or failed")
    patch: list[dict[str, Any]] | None = Field(
        default=None,
        description="JSON patch operations that correct the composition",
    )


@dataclass(frozen=True, slots=True)
class SelfCheckResult:
    """The outcome of one self-check run."""

    composition: Composition
    preview: Image.Image
    iterations: int
    passed: bool
    reason: str = ""


class SelfCheck:
    """Verify a rendered preview matches the instruction, repairing up to the limit.

    ``max_iterations`` comes from the planner configuration and bounds the number
    of fix attempts. Each attempt renders the composition, asks the provider for
    a verdict, and applies the provider's patch when the preview does not match.
    """

    def __init__(
        self,
        *,
        planner_config: PlannerConfig | None = None,
        provider: Any = None,
    ) -> None:
        self._planner_config = planner_config or PlannerConfig()
        self._provider = provider

    @property
    def planner_config(self) -> PlannerConfig:
        """The planner configuration supplying the iteration limit."""
        return self._planner_config

    @property
    def max_iterations(self) -> int:
        """How many fix attempts may be made before giving up."""
        return self._planner_config.max_iterations

    @property
    def provider(self) -> Any:
        """The vision-capable LLM provider used to judge each preview."""
        return self._provider

    async def check(
        self,
        instruction: str,
        composition: Composition,
        *,
        assets: dict[str, str] | None = None,
    ) -> SelfCheckResult:
        """Render *composition*, check it against *instruction*, and auto-fix.

        The preview is rendered once per iteration. When the provider reports a
        mismatch, its suggested patch is applied and the preview is re-rendered,
        up to :attr:`max_iterations` fixes. Returns the final composition and
        preview, or raises :class:`SelfCheckError` when the limit is exhausted.
        """
        if self._provider is None:
            raise SelfCheckError(
                "self check has no LLM provider",
                iterations=0,
                max_iterations=self.max_iterations,
            )

        current = composition
        last_reason = ""
        for iteration in range(self.max_iterations + 1):
            preview = render_still(current, time_seconds=0.0, assets=assets or {})
            verdict = await self._judge(instruction, preview)
            last_reason = verdict.reason
            if verdict.passed:
                return SelfCheckResult(
                    composition=current,
                    preview=preview,
                    iterations=iteration,
                    passed=True,
                    reason=verdict.reason,
                )
            if iteration >= self.max_iterations:
                break
            current = apply_patch(
                current, self._patch_from(verdict.patch), validate=True
            )

        raise SelfCheckError(
            "preview did not match the instruction after "
            f"{self.max_iterations} fix(es)",
            iterations=self.max_iterations,
            max_iterations=self.max_iterations,
            reason=last_reason or "no further fixes available",
        )

    # --- internal -----------------------------------------------------------

    async def _judge(self, instruction: str, preview: Image.Image) -> _Verdict:
        buffer = io.BytesIO()
        preview.save(buffer, format="PNG")
        data = base64.b64encode(buffer.getvalue()).decode("ascii")
        image = ImagePart(
            type="image",
            data=data,
            media_type="image/png",
            detail=ImageDetail.LOW,
        )
        request = LLMRequest(
            messages=[
                ChatMessage(
                    role=MessageRole.USER,
                    content=[
                        TextPart(type="text", text=self._prompt(instruction)),
                        image,
                    ],
                )
            ],
            response_schema=_Verdict.model_json_schema(),
        )
        response: LLMResponse = await self._provider.complete(request)
        return cast(_Verdict, parse_structured(response, _Verdict))

    @staticmethod
    def _prompt(instruction: str) -> str:
        return (
            "You are a visual reviewer. Compare the rendered preview image with "
            "the instruction below. Reply only with the JSON verdict.\n"
            f"Instruction: {instruction}\n"
            "Rules:\n"
            '- "passed": true when the preview faithfully shows the instruction.\n'
            '- "passed": false when anything is missing, wrong or misplaced.\n'
            '- "reason": one sentence explaining the verdict.\n'
            '- "patch": JSON Patch operations (op, path, value) that correct the '
            "composition, or null when nothing can be fixed.\n"
            "Return only valid JSON, no prose or markdown fences."
        )

    @staticmethod
    def _patch_from(
        raw: list[dict[str, Any]] | None,
    ) -> CompositionPatch:
        if not raw:
            return CompositionPatch.from_operations([])
        operations = [
            PatchOperation(
                op=str(entry.get("op", "replace")),
                path=str(entry.get("path", "")),
                value=entry.get("value"),
            )
            for entry in raw
        ]
        return CompositionPatch.from_operations(operations)
