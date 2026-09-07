"""Command-line entry points: run, diff, tools, validate."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from toolassay import __version__
from toolassay.adapters import AdapterSettings, create_adapter
from toolassay.adapters.base import Effort
from toolassay.artifact import CaseResult, RunArtifact, read_artifact, write_artifact
from toolassay.cases import load_cases
from toolassay.core import ConfigError, ToolassayError
from toolassay.diff import Threshold, check_gates, compute_diff
from toolassay.judge import LlmJudge
from toolassay.pricing import PriceBook
from toolassay.report import case_line, print_diff, print_run
from toolassay.runner import RunOptions, run_suite
from toolassay.server import McpToolServer, load_server_config, redacted_config
from toolassay.telemetry import (
    Telemetry,
    TelemetrySettings,
    otlp_endpoint_reachable,
    resolve_otlp_endpoint,
)

app = typer.Typer(
    help="Evaluate how well a model uses an MCP server's tools, and diff two runs.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()
errors = Console(stderr=True)

EXIT_OK = 0
EXIT_GATE_FAILED = 1
EXIT_USAGE = 2


class OtelMode(StrEnum):
    otlp = "otlp"
    console = "console"
    none = "none"


class EffortLevel(StrEnum):
    low = "low"
    medium = "medium"
    high = "high"
    xhigh = "xhigh"
    max = "max"


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.WARNING, stream=sys.stderr)
    for noisy in ("httpx2", "httpx", "mcp", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.DEBUG if verbose else logging.WARNING)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"toolassay {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Print version."),
    ] = False,
) -> None:
    """toolassay: an evaluation harness for MCP tool contracts."""


def _fail(message: str, code: int = EXIT_USAGE) -> typer.Exit:
    errors.print(f"[red]error:[/] {message}")
    return typer.Exit(code)


def _build_telemetry(mode: OtelMode, endpoint: str | None, capture_content: bool) -> Telemetry:
    if mode is OtelMode.otlp:
        resolved = resolve_otlp_endpoint(endpoint)
        if not otlp_endpoint_reachable(resolved):
            errors.print(
                f"[yellow]note:[/] no OTLP collector reachable at {resolved}; spans will not be "
                "exported this run. Set OTEL_EXPORTER_OTLP_ENDPOINT or pass --otel none."
            )
            return Telemetry.disabled()
    settings = TelemetrySettings(
        exporter=mode.value, endpoint=endpoint, capture_content=capture_content
    )
    return Telemetry.configure(settings)


@app.command()
def run(
    server: Annotated[Path, typer.Option("--server", help="Server config (YAML or JSON).")],
    cases: Annotated[Path, typer.Option("--cases", help="Case file (YAML).")],
    model: Annotated[str, typer.Option("--model", help="Model id.")] = "claude-opus-5",
    provider: Annotated[
        str | None,
        typer.Option("--provider", help="Adapter to use; inferred from the model id when omitted."),
    ] = None,
    out: Annotated[Path, typer.Option("--out", help="Where to write the run artifact.")] = Path(
        "run.json"
    ),
    effort: Annotated[
        EffortLevel | None, typer.Option("--effort", help="Reasoning effort for the model.")
    ] = EffortLevel.low,
    max_tokens: Annotated[int, typer.Option("--max-tokens", min=1)] = 4096,
    max_turns: Annotated[
        int | None,
        typer.Option("--max-turns", min=1, help="Model requests allowed per case."),
    ] = None,
    strict_tools: Annotated[
        bool,
        typer.Option("--strict/--no-strict", help="Send tools with strict schema validation."),
    ] = True,
    judge: Annotated[
        bool, typer.Option("--judge/--no-judge", help="Grade cases that have a rubric with an LLM.")
    ] = False,
    judge_model: Annotated[
        str | None, typer.Option("--judge-model", help="Model for the judge (default: --model).")
    ] = None,
    otel: Annotated[OtelMode, typer.Option("--otel", help="Span exporter.")] = OtelMode.otlp,
    otel_endpoint: Annotated[
        str | None, typer.Option("--otel-endpoint", help="OTLP/HTTP base URL.")
    ] = None,
    otel_content: Annotated[
        bool,
        typer.Option("--otel-content", help="Attach tool arguments and results to spans."),
    ] = False,
    prices: Annotated[
        Path | None, typer.Option("--prices", help="YAML price table overriding the built-in one.")
    ] = None,
    case_ids: Annotated[
        list[str] | None, typer.Option("--case", help="Run only these case ids (repeatable).")
    ] = None,
    label: Annotated[
        str | None, typer.Option("--label", help="Free-text label stored in the artifact.")
    ] = None,
    fail_under: Annotated[
        float | None,
        typer.Option("--fail-under", min=0, max=100, help="Exit 1 if pass rate is below this."),
    ] = None,
    timeout: Annotated[
        float, typer.Option("--timeout", min=1, help="Per-request model timeout in seconds.")
    ] = 120.0,
    quiet: Annotated[bool, typer.Option("--quiet", help="Only print the final summary.")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", help="Debug logging.")] = False,
) -> None:
    """Run every case against the server and write a JSON artifact."""
    _configure_logging(verbose)
    try:
        case_file = load_cases(cases).select(case_ids or [])
        server_config = load_server_config(server)
        price_book = PriceBook.from_file(prices) if prices else PriceBook()
        effort_value: Effort | None = effort.value if effort is not None else None
        settings = AdapterSettings(
            model=model,
            effort=effort_value,
            max_tokens=max_tokens,
            strict_tools=strict_tools,
            timeout_seconds=timeout,
        )
        adapter = create_adapter(settings, provider)
        judge_adapter = None
        if judge:
            judge_settings = settings.model_copy(update={"model": judge_model or model})
            judge_adapter = create_adapter(judge_settings, provider)
    except ToolassayError as exc:
        raise _fail(str(exc)) from None

    telemetry = _build_telemetry(otel, otel_endpoint, otel_content)
    options = RunOptions(
        label=label,
        max_turns=max_turns,
        cases_path=str(cases),
        server=redacted_config(server_config),
    )

    def on_case(result: CaseResult) -> None:
        if not quiet:
            console.print(case_line(result))

    async def _run() -> RunArtifact:
        try:
            async with McpToolServer(server_config) as mcp_server:
                if not quiet:
                    console.print(f"connected to MCP server: {mcp_server.address}")
                return await run_suite(
                    case_file,
                    adapter=adapter,
                    server=mcp_server,
                    telemetry=telemetry,
                    prices=price_book,
                    options=options,
                    judge=LlmJudge(judge_adapter) if judge_adapter is not None else None,
                    on_case=on_case,
                )
        finally:
            await adapter.aclose()
            if judge_adapter is not None:
                await judge_adapter.aclose()

    try:
        artifact = asyncio.run(_run())
    except ToolassayError as exc:
        telemetry.shutdown()
        raise _fail(str(exc)) from None
    except Exception as exc:
        telemetry.shutdown()
        raise _fail(f"{type(exc).__name__}: {exc}") from None
    telemetry.shutdown()

    try:
        write_artifact(artifact, out)
    except OSError as exc:
        raise _fail(f"cannot write {out}: {exc}") from None
    print_run(artifact, console)
    if artifact.relaxed_tools:
        errors.print(
            "[yellow]note:[/] sent without strict validation because their schema allows "
            f"additional properties: {', '.join(artifact.relaxed_tools)}"
        )
    console.print(f"wrote {out}", soft_wrap=True)
    if fail_under is not None and artifact.summary.pass_rate * 100 < fail_under:
        errors.print(
            f"[red]pass rate {artifact.summary.pass_rate:.0%} is below --fail-under "
            f"{fail_under:g}%[/]"
        )
        raise typer.Exit(EXIT_GATE_FAILED)


@app.command()
def diff(
    baseline: Annotated[Path, typer.Argument(help="Baseline run artifact.")],
    candidate: Annotated[Path, typer.Argument(help="Candidate run artifact.")],
    max_cost_increase: Annotated[
        str,
        typer.Option(
            "--max-cost-increase",
            help="Allowed cost increase over the baseline: a percentage (10%) or USD (0.05).",
        ),
    ] = "0%",
    max_pass_rate_drop: Annotated[
        float,
        typer.Option(
            "--max-pass-rate-drop", min=0, max=100, help="Allowed pass-rate drop in points."
        ),
    ] = 0.0,
    no_cost_gate: Annotated[
        bool, typer.Option("--no-cost-gate", help="Skip the cost gate entirely.")
    ] = False,
    max_regressions: Annotated[
        int,
        typer.Option("--max-regressions", min=0, help="Cases allowed to go from pass to fail."),
    ] = 0,
    allow_missing_cases: Annotated[
        bool,
        typer.Option(
            "--allow-missing-cases",
            help="Do not fail when a baseline case is absent from the candidate.",
        ),
    ] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
) -> None:
    """Compare two runs and exit 1 when the candidate regresses past the thresholds."""
    try:
        threshold = None if no_cost_gate else Threshold.parse(max_cost_increase)
        report = compute_diff(read_artifact(baseline), read_artifact(candidate))
    except ToolassayError as exc:
        raise _fail(str(exc)) from None
    violations = check_gates(
        report,
        max_cost_increase=threshold,
        max_pass_rate_drop=max_pass_rate_drop,
        max_regressions=max_regressions,
        allow_missing_cases=allow_missing_cases,
    )
    if as_json:
        payload = report.model_dump(mode="json")
        payload["violations"] = violations
        payload["cost_change_percent"] = report.cost_change_percent
        console.print_json(json.dumps(payload))
    else:
        print_diff(report, violations, console)
    if violations:
        raise typer.Exit(EXIT_GATE_FAILED)


@app.command()
def tools(
    server: Annotated[Path, typer.Option("--server", help="Server config (YAML or JSON).")],
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the tool definitions as JSON.")
    ] = False,
) -> None:
    """Connect to the server and list the tools it advertises."""
    _configure_logging(False)
    try:
        server_config = load_server_config(server)
    except ToolassayError as exc:
        raise _fail(str(exc)) from None

    async def _list() -> list[dict[str, object]]:
        async with McpToolServer(server_config) as mcp_server:
            return [tool.model_dump() for tool in await mcp_server.list_tools()]

    try:
        discovered = asyncio.run(_list())
    except Exception as exc:
        raise _fail(f"{type(exc).__name__}: {exc}") from None
    if as_json:
        console.print_json(json.dumps(discovered))
        return
    table = Table(title=f"tools advertised by {server}")
    table.add_column("name")
    table.add_column("parameters")
    table.add_column("description")
    for tool in discovered:
        schema = tool["input_schema"]
        assert isinstance(schema, dict)
        required = set(schema.get("required", []))
        properties = schema.get("properties", {})
        params = ", ".join(f"{name}{'' if name in required else '?'}" for name in properties) or "-"
        table.add_row(str(tool["name"]), params, str(tool["description"]))
    console.print(table)


@app.command()
def validate(
    cases: Annotated[Path, typer.Option("--cases", help="Case file (YAML).")],
) -> None:
    """Check a case file for schema errors without running anything."""
    try:
        case_file = load_cases(cases)
    except ConfigError as exc:
        raise _fail(str(exc)) from None
    with_judge = sum(1 for c in case_file.cases if c.judge)
    # soft_wrap because this line carries a user-supplied path. Rich otherwise hard-wraps
    # at the console width (80 when stdout is not a tty), splitting mid-token: it turns
    # "cases.yaml" into "cases.\nyaml" and breaks copy-paste. Where the wrap lands depends
    # on how long the path is, which is why CI saw a different line break than a laptop.
    console.print(
        f"{cases}: {len(case_file.cases)} cases valid ({with_judge} with a judge rubric)",
        soft_wrap=True,
    )


if __name__ == "__main__":
    app()
