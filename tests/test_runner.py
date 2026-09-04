from __future__ import annotations

from pathlib import Path

import pytest

from toolassay.cases import Case, CaseFile, load_cases
from toolassay.core import AdapterFatalError
from toolassay.judge import LlmJudge
from toolassay.pricing import ModelPrice, PriceBook
from toolassay.runner import RunContext, RunOptions, run_case, run_suite
from toolassay.server import McpToolServer
from toolassay.telemetry import Telemetry

from .support import (
    ExplodingAdapter,
    FatalAdapter,
    ScriptedAdapter,
    bookshop_scripts,
    call,
    connect,
    turn,
)

PRICES = PriceBook({"scripted-model": ModelPrice(input=1.0, output=10.0)})


async def _context(
    adapter: ScriptedAdapter | ExplodingAdapter,
    server: McpToolServer,
    *,
    system: str | None = None,
    judge: LlmJudge | None = None,
    result_max_chars: int = 10_000,
) -> RunContext:
    tools = await server.list_tools()
    return RunContext(
        adapter=adapter,
        server=server,
        tools=tools,
        telemetry=Telemetry.disabled(),
        prices=PRICES,
        run_id="run1",
        system=system,
        judge=judge,
        result_max_chars=result_max_chars,
    )


async def test_single_case_passes_and_records_trace() -> None:
    async with connect("v2") as v2_connection:
        adapter = ScriptedAdapter(bookshop_scripts())
        case = Case(
            id="find-by-author",
            prompt="Do you have any books by Mara Quill?",
            expected_tools=["find_book"],
            expected_args={"find_book": {"query": "Mara Quill"}},
            expected_substrings=["The Quiet Lantern"],
        )
        result = await run_case(case, await _context(adapter, v2_connection, system="be brief"))

        assert result.passed and result.completed
        assert result.tool_selection_correct is True
        assert result.args_correct is True
        assert result.answer_correct is True
        assert result.turns == 2
        assert result.called_tools == ["find_book"]
        assert result.usage.input_tokens == 200 and result.usage.output_tokens == 40
        assert result.cost_usd == pytest.approx(200 / 1e6 * 1.0 + 40 / 1e6 * 10.0)
        assert result.stop_reason == "end_turn"
        assert result.failures == []
        assert len(result.tool_calls) == 1
        record = result.tool_calls[0]
        assert record.name == "find_book" and record.turn == 1 and not record.is_error
        assert "BK-0042" in record.result
        assert result.latency_ms > 0

        system, tools = adapter.started[0]
        assert system == "be brief"
        assert [t.name for t in tools] == [
            "find_book",
            "get_book",
            "list_inventory",
            "reserve_book",
        ]
        sent = adapter.conversations[0].sent
        assert sent[0] == ("user", case.prompt)
        assert sent[1][0] == "tool_results"
        assert sent[1][1][0].call_id == "call-1"


async def test_wrong_tool_fails_selection_but_can_complete() -> None:
    async with connect("v2") as v2_connection:
        script = [
            turn(tool_calls=[call("list_inventory", section="Poetry")]),
            turn("Two books by Mara Quill: The Quiet Lantern and Salt and Ember."),
        ]
        adapter = ScriptedAdapter(default=script)
        case = Case(
            id="c",
            prompt="any",
            expected_tools=["find_book"],
            expected_substrings=["Quiet Lantern"],
        )
        result = await run_case(case, await _context(adapter, v2_connection))
        assert result.completed and not result.passed
        assert result.tool_selection_correct is False
        assert result.failures == [
            "expected tools ['find_book'] (exact), model called ['list_inventory']"
        ]


async def test_max_turns_stops_the_loop() -> None:
    async with connect("v2") as v2_connection:
        looping = [turn(tool_calls=[call("get_book", book_id="BK-0007")]) for _ in range(5)]
        adapter = ScriptedAdapter(default=looping)
        case = Case(id="loop", prompt="x", max_turns=3)
        result = await run_case(case, await _context(adapter, v2_connection))
        assert result.turns == 3
        assert not result.completed and not result.passed
        assert result.failures[0].startswith("stopped after 3 turns")
        assert len(result.tool_calls) == 2


async def test_tool_error_results_are_recorded() -> None:
    async with connect("v2") as v2_connection:
        script = [
            turn(tool_calls=[call("get_book", book_id="BK-9999")]),
            turn("No such book."),
        ]
        adapter = ScriptedAdapter(default=script)
        result = await run_case(Case(id="c", prompt="x"), await _context(adapter, v2_connection))
        assert result.completed
        assert result.tool_calls[0].is_error is False
        assert "No book has book_id" in result.tool_calls[0].result


async def test_refusal_and_max_tokens_do_not_complete() -> None:
    async with connect("v2") as v2_connection:
        adapter = ScriptedAdapter(default=[turn("", stop_reason="refusal")])
        result = await run_case(Case(id="c", prompt="x"), await _context(adapter, v2_connection))
        assert not result.completed and "refused" in result.failures[0]
        adapter = ScriptedAdapter(default=[turn("partial", stop_reason="max_tokens")])
        result = await run_case(Case(id="c", prompt="x"), await _context(adapter, v2_connection))
        assert not result.completed and "max_tokens" in result.failures[0]


async def test_provider_errors_become_case_errors() -> None:
    async with connect("v2") as v2_connection:
        result = await run_case(
            Case(id="c", prompt="x"), await _context(ExplodingAdapter(), v2_connection)
        )
        assert result.error == "ConnectionError: simulated provider outage"
        assert not result.passed and result.turns == 0
        assert result.cost_usd is None


async def test_long_results_are_truncated_in_the_artifact_only() -> None:
    async with connect("v1") as v1_connection:
        script = [turn(tool_calls=[call("list_inventory")]), turn("done")]
        adapter = ScriptedAdapter(default=script)
        ctx = await _context(adapter, v1_connection, result_max_chars=500)
        result = await run_case(Case(id="c", prompt="x"), ctx)
        record = result.tool_calls[0]
        assert record.result_truncated and len(record.result) == 500
        sent_results = adapter.conversations[0].sent[1][1]
        assert len(sent_results[0].text) > 10_000


async def test_judge_is_only_used_when_configured() -> None:
    async with connect("v2") as v2_connection:
        case = Case(id="c", prompt="x", judge="Says no such book exists.")
        answer = [turn("No book by that name.")]
        result = await run_case(
            case, await _context(ScriptedAdapter(default=answer), v2_connection)
        )
        assert result.judge is None and result.passed

        judge_adapter = ScriptedAdapter(
            default=[turn('{"pass": false, "reason": "too vague"}')], model="judge-model"
        )
        ctx = await _context(
            ScriptedAdapter(default=answer), v2_connection, judge=LlmJudge(judge_adapter)
        )
        result = await run_case(case, ctx)
        assert result.judge is not None and result.judge.passed is False
        assert result.judge.model == "judge-model"
        assert not result.completed and result.failures == ["judge: too vague"]
        judge_system, judge_tools = judge_adapter.started[0]
        assert judge_system is not None and "rubric" in judge_system
        assert judge_tools == []
        judge_prompt = judge_adapter.conversations[0].sent[0][1]
        assert (
            "Says no such book exists." in judge_prompt and "No book by that name." in judge_prompt
        )


async def test_unparseable_judge_reply_fails_closed() -> None:
    async with connect("v2") as v2_connection:
        case = Case(id="c", prompt="x", judge="rubric")
        judge_adapter = ScriptedAdapter(default=[turn("I think it passes.")])
        ctx = await _context(
            ScriptedAdapter(default=[turn("answer")]), v2_connection, judge=LlmJudge(judge_adapter)
        )
        result = await run_case(case, ctx)
        assert result.judge is not None and not result.judge.passed
        assert "not JSON" in result.judge.reason


async def test_run_suite_over_example_cases(examples_dir: Path) -> None:
    async with connect("v2") as v2_connection:
        case_file = load_cases(examples_dir / "cases.yaml")
        adapter = ScriptedAdapter(bookshop_scripts())
        seen: list[str] = []
        artifact = await run_suite(
            case_file,
            adapter=adapter,
            server=v2_connection,
            telemetry=Telemetry.disabled(),
            prices=PRICES,
            options=RunOptions(label="v2", cases_path="cases.yaml", server={"transport": "inproc"}),
            on_case=lambda result: seen.append(result.id),
        )
        assert seen == [case.id for case in case_file.cases]
        assert artifact.summary.cases == 7
        assert artifact.summary.passed == 7, [c.failures for c in artifact.cases if not c.passed]
        assert artifact.summary.pass_rate == 1.0
        assert artifact.summary.turns == 15
        assert artifact.summary.cost_usd is not None and artifact.summary.cost_usd > 0
        assert artifact.summary.judge_cost_usd is None
        assert (
            artifact.label == "v2"
            and artifact.provider == "scripted"
            and artifact.model == "scripted-model"
        )
        assert [t.name for t in artifact.tools] == [
            "find_book",
            "get_book",
            "list_inventory",
            "reserve_book",
        ]
        assert (
            artifact.strict_tools is True
            and artifact.effort == "low"
            and artifact.max_tokens == 512
        )
    assert artifact.relaxed_tools == []


async def test_run_suite_continues_after_a_case_error() -> None:
    async with connect("v2") as v2_connection:
        case_file = CaseFile.model_validate(
            {"cases": [{"id": "a", "prompt": "x"}, {"id": "b", "prompt": "y"}]}
        )
        artifact = await run_suite(
            case_file,
            adapter=ExplodingAdapter(),
            server=v2_connection,
            telemetry=Telemetry.disabled(),
            prices=PRICES,
            options=RunOptions(),
        )
        assert artifact.summary.errored == 2 and artifact.summary.passed == 0
        assert artifact.summary.cost_usd is None


async def test_fatal_adapter_errors_abort_the_run() -> None:
    case_file = CaseFile.model_validate({"cases": [{"id": "a", "prompt": "x"}]})
    async with connect("v2") as v2_connection:
        with pytest.raises(AdapterFatalError, match="credentials rejected"):
            await run_suite(
                case_file,
                adapter=FatalAdapter(),
                server=v2_connection,
                telemetry=Telemetry.disabled(),
                prices=PRICES,
                options=RunOptions(),
            )


async def test_judge_errors_fail_the_case_without_aborting() -> None:
    case = Case(id="c", prompt="x", judge="rubric")
    async with connect("v2") as v2_connection:
        ctx = await _context(
            ScriptedAdapter(default=[turn("answer")]),
            v2_connection,
            judge=LlmJudge(ExplodingAdapter()),
        )
        result = await run_case(case, ctx)
    assert result.judge is not None and not result.judge.passed
    assert result.judge.reason.startswith("judge error: ConnectionError")
    assert not result.completed and result.error is None
