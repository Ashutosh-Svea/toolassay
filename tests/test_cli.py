from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from toolassay.adapters import register
from toolassay.artifact import CaseResult, RunArtifact, read_artifact, summarize, write_artifact
from toolassay.cli import app
from toolassay.core import Usage

from .support import FatalAdapter, bookshop_scripts, scripted_factory, turn

runner = CliRunner()


def _artifact(passed: list[bool], cost: float) -> RunArtifact:
    cases = [
        CaseResult(
            id=f"case-{i}",
            prompt="p",
            passed=ok,
            completed=ok,
            turns=2,
            usage=Usage(input_tokens=100, output_tokens=10),
            cost_usd=cost / len(passed),
            latency_ms=1.0,
        )
        for i, ok in enumerate(passed)
    ]
    return RunArtifact(
        created_at=datetime(2026, 9, 4, tzinfo=UTC),
        provider="scripted",
        model="m",
        max_tokens=1,
        strict_tools=True,
        server={},
        cases_path="c.yaml",
        tools=[],
        summary=summarize(cases, None),
        cases=cases,
    )


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0 and "toolassay" in result.output


def test_validate_example_cases(examples_dir: Path) -> None:
    result = runner.invoke(app, ["validate", "--cases", str(examples_dir / "cases.yaml")])
    assert result.exit_code == 0, result.output
    assert "7 cases valid" in result.output


def test_validate_does_not_wrap_long_paths(examples_dir: Path, tmp_path: Path) -> None:
    """A long case-file path must not push the summary across a line break.

    Regression test. Rich hard-wraps at the console width (80 when stdout is not a tty),
    which split the summary mid-phrase and mangled the path. It only showed up in CI,
    because a runner path is a few characters longer than a laptop one, so the wrap
    landed inside "7 cases valid" there and after it locally. Building the path here
    makes the failure deterministic on any machine.
    """
    deep = tmp_path / ("d" * 120)
    deep.mkdir()
    target = deep / "cases.yaml"
    target.write_text((examples_dir / "cases.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    assert len(str(target)) > 80, "path must exceed the console width for this to test anything"

    result = runner.invoke(app, ["validate", "--cases", str(target)])

    assert result.exit_code == 0, result.output
    assert "7 cases valid" in result.output
    assert str(target) in result.output, "the path itself must survive intact"
    assert result.output.count("\n") == 1, f"expected a single unwrapped line: {result.output!r}"


def test_validate_reports_errors(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("cases:\n- id: a\n", encoding="utf-8")
    result = runner.invoke(app, ["validate", "--cases", str(path)])
    assert result.exit_code == 2
    assert "prompt" in result.output


def test_diff_passes_and_fails_gates(tmp_path: Path) -> None:
    base = tmp_path / "base.json"
    cand = tmp_path / "cand.json"
    write_artifact(_artifact([True, True], 0.10), base)
    write_artifact(_artifact([True, False], 0.15), cand)

    result = runner.invoke(app, ["diff", str(base), str(cand), "--max-cost-increase", "10%"])
    assert result.exit_code == 1, result.output
    assert "regressed from pass to fail" in result.output
    assert "pass rate dropped" in result.output
    assert "cost rose" in result.output

    lenient = ["--max-cost-increase", "100%", "--max-pass-rate-drop", "50"]
    result = runner.invoke(app, ["diff", str(base), str(cand), *lenient])
    assert result.exit_code == 1, result.output
    assert "regressed from pass to fail" in result.output

    result = runner.invoke(app, ["diff", str(base), str(cand), *lenient, "--max-regressions", "1"])
    assert result.exit_code == 0, result.output
    assert "gates passed" in result.output

    result = runner.invoke(
        app,
        [
            "diff",
            str(base),
            str(cand),
            "--no-cost-gate",
            "--max-pass-rate-drop",
            "50",
            "--max-regressions",
            "1",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["violations"] == []
    assert payload["candidate"]["passed"] == 1


def test_diff_rejects_bad_threshold_and_missing_file(tmp_path: Path) -> None:
    base = tmp_path / "base.json"
    write_artifact(_artifact([True], 0.1), base)
    result = runner.invoke(app, ["diff", str(base), str(base), "--max-cost-increase", "lots"])
    assert result.exit_code == 2 and "threshold" in result.output
    result = runner.invoke(app, ["diff", str(base), str(tmp_path / "missing.json")])
    assert result.exit_code == 2 and "cannot read" in result.output


def test_tools_command_lists_demo_tools(examples_dir: Path) -> None:
    result = runner.invoke(app, ["tools", "--server", str(examples_dir / "server-v2.yaml")])
    assert result.exit_code == 0, result.output
    assert "find_book" in result.output and "reserve_book" in result.output
    result = runner.invoke(
        app, ["tools", "--server", str(examples_dir / "server-v2.yaml"), "--json"]
    )
    assert result.exit_code == 0
    names = [t["name"] for t in json.loads(result.output)]
    assert names == ["find_book", "get_book", "list_inventory", "reserve_book"]


@pytest.fixture
def scripted_provider() -> None:
    register("scripted", scripted_factory(bookshop_scripts(), default=[turn("I do not know.")]))


def test_run_end_to_end_with_scripted_provider(
    examples_dir: Path, tmp_path: Path, scripted_provider: None
) -> None:
    out = tmp_path / "run.json"
    result = runner.invoke(
        app,
        [
            "run",
            "--server",
            str(examples_dir / "server-v2.yaml"),
            "--cases",
            str(examples_dir / "cases.yaml"),
            "--model",
            "scripted-model",
            "--provider",
            "scripted",
            "--out",
            str(out),
            "--otel",
            "none",
            "--label",
            "cli-test",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "7/7 passed" in result.output
    artifact = read_artifact(out)
    assert artifact.label == "cli-test" and artifact.summary.passed == 7
    assert artifact.server["transport"] == "stdio"
    assert artifact.cases_path.endswith("cases.yaml")


def test_run_fail_under_and_case_filter(
    examples_dir: Path, tmp_path: Path, scripted_provider: None
) -> None:
    out = tmp_path / "run.json"
    result = runner.invoke(
        app,
        [
            "run",
            "--server",
            str(examples_dir / "server-v2.yaml"),
            "--cases",
            str(examples_dir / "cases.yaml"),
            "--model",
            "scripted-model",
            "--provider",
            "scripted",
            "--out",
            str(out),
            "--otel",
            "none",
            "--case",
            "price-by-id",
            "--case",
            "unknown-title",
            "--fail-under",
            "100",
            "--quiet",
        ],
    )
    assert result.exit_code == 0, result.output
    assert read_artifact(out).summary.cases == 2


def test_run_reports_config_errors(tmp_path: Path, examples_dir: Path) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "--server",
            str(tmp_path / "nope.yaml"),
            "--cases",
            str(examples_dir / "cases.yaml"),
            "--otel",
            "none",
        ],
    )
    assert result.exit_code == 2 and "cannot read" in result.output
    result = runner.invoke(
        app,
        [
            "run",
            "--server",
            str(examples_dir / "server-v2.yaml"),
            "--cases",
            str(examples_dir / "cases.yaml"),
            "--model",
            "mystery-9000",
            "--otel",
            "none",
        ],
    )
    assert result.exit_code == 2 and "cannot infer a provider" in result.output


def test_run_stops_on_fatal_adapter_error(examples_dir: Path, tmp_path: Path) -> None:
    register("fatal", lambda settings: FatalAdapter())
    result = runner.invoke(
        app,
        [
            "run",
            "--server",
            str(examples_dir / "server-v2.yaml"),
            "--cases",
            str(examples_dir / "cases.yaml"),
            "--model",
            "fatal-model",
            "--provider",
            "fatal",
            "--out",
            str(tmp_path / "run.json"),
            "--otel",
            "none",
        ],
    )
    assert result.exit_code == 2, result.output
    assert "credentials rejected" in result.output
    assert not (tmp_path / "run.json").exists()
