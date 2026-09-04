"""The evaluation loop: prompt the model, execute the tools it asks for, score the result."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from time import perf_counter
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from toolassay.adapters.base import ModelAdapter
from toolassay.artifact import CaseResult, RunArtifact, ToolCallRecord, now, summarize
from toolassay.cases import Case, CaseFile
from toolassay.core import ModelTurn, ToolCall, ToolDefinition, ToolResult, Usage
from toolassay.judge import JudgeVerdict, LlmJudge
from toolassay.pricing import PriceBook
from toolassay.scoring import score_case
from toolassay.server import McpToolServer
from toolassay.telemetry import Telemetry

NORMAL_STOP_REASONS = frozenset({"end_turn", "stop_sequence"})

CaseCallback = Callable[[CaseResult], None]


class RunOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str | None = None
    max_turns: int | None = None
    """Overrides the case file's max_turns when set."""
    result_max_chars: int = Field(default=10_000, ge=100)
    """Tool results longer than this are cut in the artifact, never in what the model sees."""
    cases_path: str = ""
    server: dict[str, Any] = Field(default_factory=dict)
    """Redacted server config, stored in the artifact for the record."""


class RunContext(BaseModel):
    """Everything a case run needs besides the case itself."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    adapter: ModelAdapter
    server: McpToolServer
    tools: Sequence[ToolDefinition]
    telemetry: Telemetry
    prices: PriceBook
    judge: LlmJudge | None = None
    run_id: str
    system: str | None = None
    max_turns: int = 10
    result_max_chars: int = 10_000


async def run_case(case: Case, ctx: RunContext) -> CaseResult:
    """Run one case to completion and score it. Never raises for model or tool failures."""
    max_turns = case.max_turns or ctx.max_turns
    tools_by_name = {tool.name: tool for tool in ctx.tools}
    provider = ctx.adapter.provider_name
    model = ctx.adapter.model
    max_tokens = getattr(ctx.adapter, "max_tokens", None)

    calls: list[ToolCall] = []
    records: list[ToolCallRecord] = []
    usage = Usage()
    turns = 0
    failures: list[str] = []
    error: str | None = None
    stop_reason: str | None = None
    final_answer = ""
    pending_tool_calls = False

    started = perf_counter()
    with ctx.telemetry.case_span(
        run_id=ctx.run_id, case_id=case.id, model=model, provider_name=provider
    ) as case_span:
        try:
            conversation = ctx.adapter.start(system=case.system or ctx.system, tools=ctx.tools)
            with ctx.telemetry.chat_span(
                model=model, provider_name=provider, max_tokens=max_tokens
            ) as chat:
                turn = await conversation.send_user(case.prompt)
                ctx.telemetry.record_turn(chat, turn)
            turns += 1
            usage = usage + turn.usage

            while turn.tool_calls:
                if turns >= max_turns:
                    pending_tool_calls = True
                    failures.append(
                        f"stopped after {turns} turns with tool calls still pending "
                        f"(max_turns={max_turns})"
                    )
                    break
                results = await _execute_tool_calls(turn, ctx, tools_by_name, turns, calls, records)
                with ctx.telemetry.chat_span(
                    model=model, provider_name=provider, max_tokens=max_tokens
                ) as chat:
                    turn = await conversation.send_tool_results(results)
                    ctx.telemetry.record_turn(chat, turn)
                turns += 1
                usage = usage + turn.usage

            final_answer = turn.text
            stop_reason = turn.stop_reason
            if stop_reason == "refusal":
                failures.append("model refused the request")
            elif not pending_tool_calls and stop_reason not in NORMAL_STOP_REASONS:
                failures.append(f"model stopped with reason {stop_reason!r}")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            failures.append(f"run error: {error}")

        latency_ms = (perf_counter() - started) * 1000
        score = score_case(case, calls, final_answer)
        failures.extend(score.failures)

        verdict: JudgeVerdict | None = None
        if ctx.judge is not None and case.judge and error is None:
            verdict = await ctx.judge.evaluate(case, final_answer, calls)
            if not verdict.passed:
                failures.append(f"judge: {verdict.reason}")

        completed = (
            error is None
            and not pending_tool_calls
            and stop_reason in NORMAL_STOP_REASONS
            and score.answer_correct is not False
            and (verdict is None or verdict.passed)
        )
        passed = (
            completed
            and score.tool_selection_correct is not False
            and score.args_correct is not False
        )
        ctx.telemetry.record_case_outcome(
            case_span,
            passed=passed,
            turns=turns,
            usage=usage,
            stop_reason=stop_reason,
            error=error,
        )

    return CaseResult(
        id=case.id,
        prompt=case.prompt,
        tags=case.tags,
        expected_tools=case.expected_tools,
        called_tools=[call.name for call in calls],
        passed=passed,
        completed=completed,
        tool_selection_correct=score.tool_selection_correct,
        args_correct=score.args_correct,
        answer_correct=score.answer_correct,
        turns=turns,
        usage=usage,
        cost_usd=ctx.prices.estimate(model, usage),
        latency_ms=latency_ms,
        stop_reason=stop_reason,
        final_answer=final_answer,
        tool_calls=records,
        failures=failures,
        judge=verdict,
        error=error,
    )


async def _execute_tool_calls(
    turn: ModelTurn,
    ctx: RunContext,
    tools_by_name: dict[str, ToolDefinition],
    turn_index: int,
    calls: list[ToolCall],
    records: list[ToolCallRecord],
) -> list[ToolResult]:
    results: list[ToolResult] = []
    for call in turn.tool_calls:
        calls.append(call)
        with ctx.telemetry.tool_span(
            call=call,
            tool=tools_by_name.get(call.name),
            transport=ctx.server.transport_kind,
            server_address=ctx.server.address,
        ) as span:
            tool_started = perf_counter()
            result = await ctx.server.call_tool(call)
            tool_latency_ms = (perf_counter() - tool_started) * 1000
            ctx.telemetry.record_tool_result(span, result)
        results.append(result)
        truncated = len(result.text) > ctx.result_max_chars
        records.append(
            ToolCallRecord(
                turn=turn_index,
                id=call.id,
                name=call.name,
                arguments=call.arguments,
                result=result.text[: ctx.result_max_chars],
                is_error=result.is_error,
                latency_ms=tool_latency_ms,
                result_truncated=truncated,
            )
        )
    return results


async def run_suite(
    case_file: CaseFile,
    *,
    adapter: ModelAdapter,
    server: McpToolServer,
    telemetry: Telemetry,
    prices: PriceBook,
    options: RunOptions,
    judge: LlmJudge | None = None,
    on_case: CaseCallback | None = None,
) -> RunArtifact:
    """Discover the server's tools, run every case, and assemble the artifact."""
    run_id = uuid.uuid4().hex[:12]
    tools = await server.list_tools()
    ctx = RunContext(
        adapter=adapter,
        server=server,
        tools=tools,
        telemetry=telemetry,
        prices=prices,
        judge=judge,
        run_id=run_id,
        system=case_file.system,
        max_turns=options.max_turns or case_file.max_turns,
        result_max_chars=options.result_max_chars,
    )
    results: list[CaseResult] = []
    judge_usage = Usage()
    with telemetry.run_span(
        run_id=run_id,
        model=adapter.model,
        provider_name=adapter.provider_name,
        case_count=len(case_file.cases),
    ) as span:
        for case in case_file.cases:
            result = await run_case(case, ctx)
            results.append(result)
            if result.judge is not None:
                judge_usage = judge_usage + result.judge.usage
            if on_case is not None:
                on_case(result)
        total_usage = Usage()
        for result in results:
            total_usage = total_usage + result.usage
        telemetry.record_run_outcome(
            span, passed=sum(1 for r in results if r.passed), usage=total_usage
        )

    judge_cost = prices.estimate(judge.model, judge_usage) if judge is not None else None
    return RunArtifact(
        created_at=now(),
        label=options.label,
        provider=adapter.provider_name,
        model=adapter.model,
        effort=getattr(adapter, "effort", None),
        max_tokens=getattr(adapter, "max_tokens", 0),
        strict_tools=bool(getattr(adapter, "strict_tools", False)),
        server=options.server,
        cases_path=options.cases_path,
        judge_model=judge.model if judge is not None else None,
        tools=list(tools),
        summary=summarize(results, judge_cost),
        cases=results,
    )
