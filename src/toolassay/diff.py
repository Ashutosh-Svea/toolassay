"""Comparing two run artifacts, and the gates that turn a comparison into a CI verdict."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from toolassay.artifact import CaseResult, RunArtifact
from toolassay.core import ConfigError, ToolDefinition

CaseStatus = Literal["pass", "fail", "fixed", "regressed", "added", "removed"]


class Threshold(BaseModel):
    """A cost tolerance, either relative (``10%``) or absolute in USD (``0.05``)."""

    model_config = ConfigDict(frozen=True)

    percent: float | None = None
    absolute: float | None = None

    @classmethod
    def parse(cls, text: str) -> Threshold:
        value = text.strip()
        try:
            if value.endswith("%"):
                percent = float(value[:-1])
                if percent < 0:
                    raise ValueError
                return cls(percent=percent)
            absolute = float(value)
            if absolute < 0:
                raise ValueError
            return cls(absolute=absolute)
        except ValueError:
            raise ConfigError(
                f"threshold {text!r} must be a percentage like 10% or a USD amount like 0.05"
            ) from None

    def allows(self, before: float, after: float) -> bool:
        increase = after - before
        if increase <= 0:
            return True
        if self.absolute is not None:
            return increase <= self.absolute + 1e-12
        if before <= 0:
            return False
        return (increase / before) * 100 <= (self.percent or 0) + 1e-9

    def describe(self) -> str:
        if self.absolute is not None:
            return f"${self.absolute:.4f}"
        return f"{self.percent:g}%"


class CaseDelta(BaseModel):
    id: str
    status: CaseStatus
    baseline_passed: bool | None = None
    candidate_passed: bool | None = None
    baseline_turns: int | None = None
    candidate_turns: int | None = None
    baseline_tokens: int | None = None
    candidate_tokens: int | None = None
    baseline_cost: float | None = None
    candidate_cost: float | None = None
    candidate_failures: list[str] = Field(default_factory=list)

    @property
    def turns_delta(self) -> int | None:
        return _delta_int(self.baseline_turns, self.candidate_turns)

    @property
    def tokens_delta(self) -> int | None:
        return _delta_int(self.baseline_tokens, self.candidate_tokens)

    @property
    def cost_delta(self) -> float | None:
        if self.baseline_cost is None or self.candidate_cost is None:
            return None
        return self.candidate_cost - self.baseline_cost


class ToolChange(BaseModel):
    name: str
    kind: Literal["added", "removed", "description", "schema"]


class Totals(BaseModel):
    """Aggregates over the cases both runs share, so the comparison is like for like."""

    cases: int
    passed: int
    turns: int
    tokens: int
    cost: float | None

    @property
    def pass_rate(self) -> float:
        return self.passed / self.cases if self.cases else 0.0


class DiffReport(BaseModel):
    baseline_model: str
    candidate_model: str
    baseline_label: str | None
    candidate_label: str | None
    cases: list[CaseDelta]
    baseline: Totals
    candidate: Totals
    tool_changes: list[ToolChange]
    only_in_baseline: list[str]
    only_in_candidate: list[str]

    @property
    def pass_rate_drop(self) -> float:
        return self.baseline.pass_rate - self.candidate.pass_rate

    @property
    def cost_change_percent(self) -> float | None:
        if self.baseline.cost is None or self.candidate.cost is None or self.baseline.cost == 0:
            return None
        return (self.candidate.cost - self.baseline.cost) / self.baseline.cost * 100


def _delta_int(before: int | None, after: int | None) -> int | None:
    if before is None or after is None:
        return None
    return after - before


def _tokens(case: CaseResult) -> int:
    return case.usage.total_tokens


def _status(before: CaseResult | None, after: CaseResult | None) -> CaseStatus:
    if before is None:
        return "added"
    if after is None:
        return "removed"
    if before.passed and after.passed:
        return "pass"
    if before.passed and not after.passed:
        return "regressed"
    if not before.passed and after.passed:
        return "fixed"
    return "fail"


def _totals(cases: list[CaseResult]) -> Totals:
    cost: float | None = 0.0
    for case in cases:
        if case.cost_usd is None:
            cost = None
            break
        assert cost is not None
        cost += case.cost_usd
    return Totals(
        cases=len(cases),
        passed=sum(1 for c in cases if c.passed),
        turns=sum(c.turns for c in cases),
        tokens=sum(_tokens(c) for c in cases),
        cost=cost,
    )


def tool_changes(before: list[ToolDefinition], after: list[ToolDefinition]) -> list[ToolChange]:
    before_by_name = {tool.name: tool for tool in before}
    after_by_name = {tool.name: tool for tool in after}
    changes: list[ToolChange] = []
    for name in sorted(set(before_by_name) | set(after_by_name)):
        old = before_by_name.get(name)
        new = after_by_name.get(name)
        if old is None:
            changes.append(ToolChange(name=name, kind="added"))
        elif new is None:
            changes.append(ToolChange(name=name, kind="removed"))
        else:
            if old.description != new.description:
                changes.append(ToolChange(name=name, kind="description"))
            if old.input_schema != new.input_schema:
                changes.append(ToolChange(name=name, kind="schema"))
    return changes


def compute_diff(baseline: RunArtifact, candidate: RunArtifact) -> DiffReport:
    before_by_id = {case.id: case for case in baseline.cases}
    after_by_id = {case.id: case for case in candidate.cases}
    ordered_ids = list(before_by_id) + [i for i in after_by_id if i not in before_by_id]

    deltas: list[CaseDelta] = []
    for case_id in ordered_ids:
        before = before_by_id.get(case_id)
        after = after_by_id.get(case_id)
        deltas.append(
            CaseDelta(
                id=case_id,
                status=_status(before, after),
                baseline_passed=before.passed if before else None,
                candidate_passed=after.passed if after else None,
                baseline_turns=before.turns if before else None,
                candidate_turns=after.turns if after else None,
                baseline_tokens=_tokens(before) if before else None,
                candidate_tokens=_tokens(after) if after else None,
                baseline_cost=before.cost_usd if before else None,
                candidate_cost=after.cost_usd if after else None,
                candidate_failures=list(after.failures) if after else [],
            )
        )

    shared = [i for i in ordered_ids if i in before_by_id and i in after_by_id]
    return DiffReport(
        baseline_model=baseline.model,
        candidate_model=candidate.model,
        baseline_label=baseline.label,
        candidate_label=candidate.label,
        cases=deltas,
        baseline=_totals([before_by_id[i] for i in shared]),
        candidate=_totals([after_by_id[i] for i in shared]),
        tool_changes=tool_changes(baseline.tools, candidate.tools),
        only_in_baseline=[i for i in before_by_id if i not in after_by_id],
        only_in_candidate=[i for i in after_by_id if i not in before_by_id],
    )


def check_gates(
    report: DiffReport,
    *,
    max_cost_increase: Threshold | None,
    max_pass_rate_drop: float,
) -> list[str]:
    """Return the gate violations; an empty list means the candidate is acceptable."""
    violations: list[str] = []
    if report.baseline.cases == 0:
        violations.append("the two runs share no case ids, so there is nothing to compare")
        return violations
    drop = report.pass_rate_drop * 100
    if drop > max_pass_rate_drop + 1e-9:
        violations.append(
            f"pass rate dropped by {drop:.1f} points "
            f"({report.baseline.pass_rate:.0%} to {report.candidate.pass_rate:.0%}); "
            f"allowed drop is {max_pass_rate_drop:g} points"
        )
    if max_cost_increase is not None:
        if report.baseline.cost is None or report.candidate.cost is None:
            violations.append(
                "cost gate cannot be evaluated: at least one run has no cost estimate "
                "(unknown model price; pass --prices)"
            )
        elif not max_cost_increase.allows(report.baseline.cost, report.candidate.cost):
            change = report.cost_change_percent
            change_text = f"{change:+.1f}%" if change is not None else "from zero"
            violations.append(
                f"cost rose {change_text} (${report.baseline.cost:.4f} to "
                f"${report.candidate.cost:.4f}); allowed increase is "
                f"{max_cost_increase.describe()}"
            )
    return violations
