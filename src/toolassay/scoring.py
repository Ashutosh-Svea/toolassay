"""Deterministic assertions: tool selection, argument values, and answer substrings.

Nothing in this module calls a model. A run that uses only these scorers is reproducible
given the same model responses.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

from toolassay.cases import Case, ToolMatch
from toolassay.core import ToolCall


class Score(BaseModel):
    """Outcome of the deterministic checks. ``None`` means the check was not configured."""

    tool_selection_correct: bool | None = None
    args_correct: bool | None = None
    answer_correct: bool | None = None
    failures: list[str] = Field(default_factory=list)


def is_subsequence(expected: Sequence[str], actual: Sequence[str]) -> bool:
    position = 0
    for item in actual:
        if position < len(expected) and item == expected[position]:
            position += 1
    return position == len(expected)


def match_tools(expected: Sequence[str], actual: Sequence[str], mode: ToolMatch) -> bool:
    if mode is ToolMatch.EXACT:
        return list(expected) == list(actual)
    return is_subsequence(expected, actual)


def _normalise(value: Any) -> Any:
    """Round-trip through JSON so tuples, ints, and floats compare the way they serialise."""
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def check_args(
    expected_args: dict[str, dict[str, Any]], calls: Sequence[ToolCall]
) -> tuple[bool | None, list[str]]:
    """Compare expected argument values against the first call made to each tool."""
    if not expected_args:
        return None, []
    failures: list[str] = []
    for tool_name, expected in expected_args.items():
        call = next((c for c in calls if c.name == tool_name), None)
        if call is None:
            failures.append(f"expected a call to {tool_name} but it was never called")
            continue
        for key, value in expected.items():
            if key not in call.arguments:
                failures.append(f"{tool_name}: argument {key!r} missing (expected {value!r})")
            elif _normalise(call.arguments[key]) != _normalise(value):
                failures.append(
                    f"{tool_name}: argument {key!r} was {call.arguments[key]!r}, expected {value!r}"
                )
    return not failures, failures


def check_substrings(expected: Sequence[str], answer: str) -> tuple[bool | None, list[str]]:
    if not expected:
        return None, []
    lowered = answer.lower()
    missing = [needle for needle in expected if needle.lower() not in lowered]
    return not missing, [f"final answer does not contain {needle!r}" for needle in missing]


def score_case(case: Case, calls: Sequence[ToolCall], final_answer: str) -> Score:
    score = Score()
    called = [call.name for call in calls]
    if case.expected_tools is not None:
        score.tool_selection_correct = match_tools(case.expected_tools, called, case.tool_match)
        if not score.tool_selection_correct:
            score.failures.append(
                f"expected tools {case.expected_tools} ({case.tool_match.value}), "
                f"model called {called}"
            )
    score.args_correct, arg_failures = check_args(case.expected_args, calls)
    score.failures.extend(arg_failures)
    score.answer_correct, text_failures = check_substrings(case.expected_substrings, final_answer)
    score.failures.extend(text_failures)
    return score
