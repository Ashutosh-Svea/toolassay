"""Optional LLM judge for free-text answers.

Kept apart from the deterministic scorers on purpose: a run only touches this module when
the judge is switched on, so a judge-free run stays reproducible and costs nothing extra.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from pydantic import BaseModel, Field

from toolassay.adapters.base import ModelAdapter
from toolassay.cases import Case
from toolassay.core import ToolCall, Usage

JUDGE_SYSTEM_PROMPT = (
    "You grade whether an assistant's final answer satisfies a rubric. "
    "Judge only the final answer against the rubric. "
    'Reply with JSON only, in the form {"pass": true or false, "reason": "one sentence"}.'
)


class JudgeVerdict(BaseModel):
    passed: bool
    reason: str
    model: str
    usage: Usage = Field(default_factory=Usage)


def build_judge_prompt(case: Case, final_answer: str, calls: Sequence[ToolCall]) -> str:
    call_lines = [f"- {call.name}({json.dumps(call.arguments, sort_keys=True)})" for call in calls]
    calls_block = "\n".join(call_lines) if call_lines else "- (no tool calls)"
    return (
        f"Rubric:\n{case.judge}\n\n"
        f"User request:\n{case.prompt}\n\n"
        f"Tools the assistant called:\n{calls_block}\n\n"
        f"Final answer to grade:\n{final_answer or '(empty answer)'}"
    )


def parse_verdict(text: str) -> tuple[bool, str]:
    """Pull the pass flag and reason out of the judge's reply; unparseable replies fail."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return False, f"judge reply was not JSON: {text[:200]!r}"
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        return False, f"judge reply was not valid JSON ({exc}): {text[:200]!r}"
    if not isinstance(data, dict) or "pass" not in data:
        return False, f"judge reply lacked a 'pass' field: {text[:200]!r}"
    flag = data["pass"]
    reason = data.get("reason")
    reason_text = str(reason) if reason is not None else ""
    if not isinstance(flag, bool):
        return False, f"judge 'pass' field was {flag!r}, not a JSON boolean ({reason_text})"
    return flag, reason_text


class LlmJudge:
    """Grades a final answer against a case's rubric using a model adapter."""

    def __init__(self, adapter: ModelAdapter) -> None:
        self._adapter = adapter

    @property
    def model(self) -> str:
        return self._adapter.model

    async def evaluate(
        self, case: Case, final_answer: str, calls: Sequence[ToolCall]
    ) -> JudgeVerdict:
        conversation = self._adapter.start(system=JUDGE_SYSTEM_PROMPT, tools=[])
        turn = await conversation.send_user(build_judge_prompt(case, final_answer, calls))
        passed, reason = parse_verdict(turn.text)
        return JudgeVerdict(
            passed=passed, reason=reason, model=self._adapter.model, usage=turn.usage
        )
