from __future__ import annotations

from datetime import UTC, datetime

import pytest

from toolassay.artifact import CaseResult, RunArtifact, summarize
from toolassay.core import ConfigError, ToolDefinition, Usage
from toolassay.diff import Threshold, check_gates, compute_diff, tool_changes


def _case(
    case_id: str, passed: bool, *, turns: int = 2, tokens: int = 1000, cost: float | None = 0.01
) -> CaseResult:
    return CaseResult(
        id=case_id,
        prompt="p",
        passed=passed,
        completed=passed,
        turns=turns,
        usage=Usage(input_tokens=tokens - 100, output_tokens=100),
        cost_usd=cost,
        latency_ms=10.0,
        failures=[] if passed else ["nope"],
    )


def _artifact(
    cases: list[CaseResult], tools: list[ToolDefinition] | None = None, label: str = "x"
) -> RunArtifact:
    return RunArtifact(
        created_at=datetime(2026, 9, 4, tzinfo=UTC),
        label=label,
        provider="scripted",
        model="m",
        max_tokens=1,
        strict_tools=True,
        server={},
        cases_path="cases.yaml",
        tools=tools or [],
        summary=summarize(cases, None),
        cases=cases,
    )


def test_threshold_parsing() -> None:
    assert Threshold.parse("10%") == Threshold(percent=10)
    assert Threshold.parse(" 0.05 ") == Threshold(absolute=0.05)
    assert Threshold.parse("0%").describe() == "0%"
    for bad in ("abc", "-5%", "-1"):
        with pytest.raises(ConfigError, match="threshold"):
            Threshold.parse(bad)


def test_threshold_allows() -> None:
    assert Threshold(percent=10).allows(1.0, 1.1)
    assert not Threshold(percent=10).allows(1.0, 1.2)
    assert Threshold(percent=0).allows(1.0, 0.5)
    assert not Threshold(percent=50).allows(0.0, 0.01)
    assert Threshold(absolute=0.05).allows(1.0, 1.05)
    assert not Threshold(absolute=0.05).allows(1.0, 1.06)


def test_compute_diff_statuses_and_totals() -> None:
    baseline = _artifact(
        [_case("a", True), _case("b", False, turns=4), _case("c", True), _case("gone", True)]
    )
    candidate = _artifact(
        [
            _case("a", True, turns=1, tokens=500, cost=0.005),
            _case("b", True),
            _case("c", False),
            _case("new", True),
        ]
    )
    report = compute_diff(baseline, candidate)
    statuses = {d.id: d.status for d in report.cases}
    assert statuses == {
        "a": "pass",
        "b": "fixed",
        "c": "regressed",
        "gone": "removed",
        "new": "added",
    }
    a = next(d for d in report.cases if d.id == "a")
    assert a.turns_delta == -1 and a.tokens_delta == -500 and a.cost_delta == pytest.approx(-0.005)
    assert report.only_in_baseline == ["gone"] and report.only_in_candidate == ["new"]
    assert report.baseline.cases == 3 and report.candidate.cases == 3
    assert report.baseline.passed == 2 and report.candidate.passed == 2
    assert report.baseline.turns == 8 and report.candidate.turns == 5
    assert report.baseline.cost == pytest.approx(0.03) and report.candidate.cost == pytest.approx(
        0.025
    )
    assert report.cost_change_percent == pytest.approx(-16.666, abs=0.01)


def test_tool_changes_detected() -> None:
    before = [
        ToolDefinition(name="a", description="old"),
        ToolDefinition(
            name="b", description="same", input_schema={"type": "object", "properties": {}}
        ),
        ToolDefinition(name="gone"),
    ]
    after = [
        ToolDefinition(name="a", description="new"),
        ToolDefinition(
            name="b", description="same", input_schema={"type": "object", "properties": {"x": {}}}
        ),
        ToolDefinition(name="added"),
    ]
    changes = [(c.name, c.kind) for c in tool_changes(before, after)]
    assert changes == [
        ("a", "description"),
        ("added", "added"),
        ("b", "schema"),
        ("gone", "removed"),
    ]


def test_gates() -> None:
    baseline = _artifact([_case("a", True), _case("b", True)])
    worse = _artifact([_case("a", True), _case("b", False, cost=0.02)])
    report = compute_diff(baseline, worse)
    violations = check_gates(report, max_cost_increase=Threshold.parse("10%"), max_pass_rate_drop=0)
    assert len(violations) == 3
    assert "1 case(s) regressed from pass to fail: b" in violations[0]
    assert "pass rate dropped by 50.0 points" in violations[1]
    assert "cost rose +50.0%" in violations[2]
    relaxed = check_gates(report, max_cost_increase=None, max_pass_rate_drop=50, max_regressions=1)
    assert relaxed == []
    absolute = check_gates(
        report,
        max_cost_increase=Threshold.parse("0.01"),
        max_pass_rate_drop=50,
        max_regressions=1,
    )
    assert absolute == []


def test_a_fix_cannot_hide_a_regression() -> None:
    baseline = _artifact([_case("a", True), _case("b", False)])
    candidate = _artifact([_case("a", False), _case("b", True)])
    report = compute_diff(baseline, candidate)
    assert report.pass_rate_drop == 0
    violations = check_gates(report, max_cost_increase=None, max_pass_rate_drop=0)
    assert violations == ["1 case(s) regressed from pass to fail: a; allowed is 0"]


def test_missing_baseline_cases_are_a_violation_unless_allowed() -> None:
    baseline = _artifact([_case("a", True), _case("b", True)])
    candidate = _artifact([_case("a", True)])
    report = compute_diff(baseline, candidate)
    violations = check_gates(report, max_cost_increase=None, max_pass_rate_drop=0)
    assert len(violations) == 1
    assert violations[0].startswith("1 baseline case(s) missing from the candidate: b")
    allowed = check_gates(
        report, max_cost_increase=None, max_pass_rate_drop=0, allow_missing_cases=True
    )
    assert allowed == []


def test_gate_with_missing_cost_and_no_shared_cases() -> None:
    baseline = _artifact([_case("a", True, cost=None)])
    candidate = _artifact([_case("a", True)])
    violations = check_gates(
        compute_diff(baseline, candidate),
        max_cost_increase=Threshold.parse("0%"),
        max_pass_rate_drop=0,
    )
    assert violations == [
        "cost gate cannot be evaluated: at least one run has no cost estimate "
        "(unknown model price; pass --prices)"
    ]
    disjoint = compute_diff(_artifact([_case("a", True)]), _artifact([_case("b", True)]))
    assert (
        "nothing to compare"
        in check_gates(disjoint, max_cost_increase=None, max_pass_rate_drop=0)[0]
    )


def test_zero_baseline_cost_with_percent_threshold_is_a_violation() -> None:
    report = compute_diff(
        _artifact([_case("a", True, cost=0.0)]), _artifact([_case("a", True, cost=0.01)])
    )
    violations = check_gates(report, max_cost_increase=Threshold.parse("10%"), max_pass_rate_drop=0)
    assert violations and "from zero" in violations[0]
