"""Terminal rendering for run results and diffs."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from toolassay.artifact import CaseResult, RunArtifact
from toolassay.diff import DiffReport


def _cost(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.4f}"


def _signed(value: int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+d}"


def _signed_cost(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.4f}"


def case_line(result: CaseResult) -> str:
    """One-line progress summary printed as each case finishes."""
    if result.error is not None:
        verdict = "[red]ERROR[/]"
    elif result.passed:
        verdict = "[green]PASS[/] "
    else:
        verdict = "[red]FAIL[/] "
    tools = ", ".join(result.called_tools) or "(no tools)"
    return (
        f"{verdict} {result.id}  tools={tools}  turns={result.turns}  "
        f"tokens={result.usage.total_tokens}  cost={_cost(result.cost_usd)}"
    )


def print_run(artifact: RunArtifact, console: Console) -> None:
    table = Table(title=f"toolassay run: {artifact.model}", show_lines=False)
    table.add_column("case")
    table.add_column("result")
    table.add_column("tools called")
    table.add_column("turns", justify="right")
    table.add_column("in tok", justify="right")
    table.add_column("out tok", justify="right")
    table.add_column("cost", justify="right")
    table.add_column("ms", justify="right")
    for case in artifact.cases:
        if case.error is not None:
            result = "[red]ERROR[/]"
        elif case.passed:
            result = "[green]PASS[/]"
        else:
            result = "[red]FAIL[/]"
        table.add_row(
            case.id,
            result,
            ", ".join(case.called_tools) or "-",
            str(case.turns),
            str(case.usage.input_tokens),
            str(case.usage.output_tokens),
            _cost(case.cost_usd),
            f"{case.latency_ms:.0f}",
        )
    console.print(table)
    summary = artifact.summary
    console.print(
        f"{summary.passed}/{summary.cases} passed ({summary.pass_rate:.0%}), "
        f"{summary.turns} turns, {summary.input_tokens} in + {summary.output_tokens} out tokens, "
        f"cost {_cost(summary.cost_usd)}"
        + (f", judge cost {_cost(summary.judge_cost_usd)}" if artifact.judge_model else "")
    )
    for case in artifact.cases:
        for failure in case.failures:
            console.print(f"  [yellow]{case.id}[/]: {failure}")


def print_diff(report: DiffReport, violations: list[str], console: Console) -> None:
    baseline = report.baseline_label or "baseline"
    candidate = report.candidate_label or "candidate"
    table = Table(title=f"toolassay diff: {baseline} vs {candidate}")
    table.add_column("case")
    table.add_column("status")
    table.add_column("turns", justify="right")
    table.add_column("tokens", justify="right")
    table.add_column("cost", justify="right")
    for delta in report.cases:
        colour = {
            "pass": "green",
            "fixed": "green",
            "fail": "red",
            "regressed": "red",
            "added": "cyan",
            "removed": "cyan",
        }[delta.status]
        table.add_row(
            delta.id,
            f"[{colour}]{delta.status}[/]",
            f"{_fmt_pair(delta.baseline_turns, delta.candidate_turns)} "
            f"({_signed(delta.turns_delta)})",
            f"{_fmt_pair(delta.baseline_tokens, delta.candidate_tokens)} "
            f"({_signed(delta.tokens_delta)})",
            f"{_cost(delta.baseline_cost)} to {_cost(delta.candidate_cost)} "
            f"({_signed_cost(delta.cost_delta)})",
        )
    console.print(table)

    if report.tool_changes:
        changes = ", ".join(f"{c.name} ({c.kind})" for c in report.tool_changes)
        console.print(f"tool contract changes: {changes}")
    if report.only_in_baseline or report.only_in_candidate:
        console.print(
            "[yellow]note:[/] case sets differ; totals below cover only the shared cases "
            f"(only in baseline: {report.only_in_baseline or '-'}, "
            f"only in candidate: {report.only_in_candidate or '-'})"
        )

    b, c = report.baseline, report.candidate
    change = report.cost_change_percent
    change_text = f" ({change:+.1f}%)" if change is not None else ""
    console.print(
        f"pass rate {b.pass_rate:.0%} to {c.pass_rate:.0%}, "
        f"turns {b.turns} to {c.turns}, tokens {b.tokens} to {c.tokens}, "
        f"cost {_cost(b.cost)} to {_cost(c.cost)}{change_text}"
    )
    if violations:
        for violation in violations:
            console.print(f"[red]gate failed:[/] {violation}")
    else:
        console.print("[green]gates passed[/]")


def _fmt_pair(before: int | None, after: int | None) -> str:
    left = "-" if before is None else str(before)
    right = "-" if after is None else str(after)
    return f"{left} to {right}"
