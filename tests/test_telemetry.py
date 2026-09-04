from __future__ import annotations

import socket

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import SpanKind, StatusCode

from toolassay.cases import Case
from toolassay.core import ToolCall, ToolResult, Usage
from toolassay.pricing import PriceBook
from toolassay.runner import RunContext, run_case
from toolassay.server import McpToolServer
from toolassay.telemetry import Telemetry, otlp_endpoint_reachable, resolve_otlp_endpoint

from .support import ScriptedAdapter, call, connect, turn


def _by_name(spans: list[ReadableSpan], name: str) -> ReadableSpan:
    matches = [s for s in spans if s.name == name]
    assert matches, f"no span named {name!r}; have {[s.name for s in spans]}"
    return matches[0]


async def _run(connection: McpToolServer, telemetry: Telemetry) -> None:
    script = [
        turn(
            tool_calls=[
                call("get_book", book_id="BK-0007"),
                call("get_book", "call-2", book_id="BK-9999"),
            ]
        ),
        turn("Salt Roads costs 12.50."),
    ]
    ctx = RunContext(
        adapter=ScriptedAdapter(default=script),
        server=connection,
        tools=await connection.list_tools(),
        telemetry=telemetry,
        prices=PriceBook(),
        run_id="run-xyz",
    )
    case = Case(
        id="price", prompt="What does BK-0007 cost?", expected_tools=["get_book", "get_book"]
    )
    with telemetry.run_span(
        run_id="run-xyz", model="scripted-model", provider_name="scripted", case_count=1
    ) as span:
        result = await run_case(case, ctx)
        telemetry.record_run_outcome(span, passed=int(result.passed), usage=result.usage)


async def test_span_hierarchy_and_genai_attributes() -> None:
    async with connect("v2") as v2_connection:
        telemetry, exporter = Telemetry.in_memory()
        await _run(v2_connection, telemetry)
        telemetry.shutdown()
        spans = list(exporter.get_finished_spans())

        run = _by_name(spans, "toolassay.run")
        case = _by_name(spans, "invoke_agent toolassay")
        chats = [s for s in spans if s.name == "chat scripted-model"]
        tools = [s for s in spans if s.name.startswith("execute_tool ")]
        assert len(chats) == 2 and len(tools) == 2
        assert run.parent is None
        assert case.parent is not None and case.parent.span_id == run.context.span_id
        assert all(
            s.parent is not None and s.parent.span_id == case.context.span_id for s in chats + tools
        )

        assert case.kind is SpanKind.INTERNAL
        assert case.attributes is not None
        assert case.attributes["gen_ai.operation.name"] == "invoke_agent"
        assert case.attributes["gen_ai.agent.name"] == "toolassay"
        assert case.attributes["gen_ai.provider.name"] == "scripted"
        assert case.attributes["gen_ai.conversation.id"] == "run-xyz/price"
        assert case.attributes["toolassay.case.passed"] is True
        assert case.attributes["gen_ai.usage.input_tokens"] == 200

        chat = chats[0]
        assert chat.kind is SpanKind.CLIENT
        assert chat.attributes is not None
        assert chat.attributes["gen_ai.operation.name"] == "chat"
        assert chat.attributes["gen_ai.request.model"] == "scripted-model"
        assert chat.attributes["gen_ai.request.max_tokens"] == 512
        assert chat.attributes["gen_ai.usage.output_tokens"] == 20
        assert chat.attributes["gen_ai.response.finish_reasons"] == ("tool_use",)
        assert chat.attributes["gen_ai.response.model"] == "scripted-model"

        ok, failed = tools
        assert ok.name == "execute_tool get_book" and ok.kind is SpanKind.CLIENT
        assert ok.attributes is not None
        assert ok.attributes["gen_ai.operation.name"] == "execute_tool"
        assert ok.attributes["gen_ai.tool.name"] == "get_book"
        assert ok.attributes["gen_ai.tool.call.id"] == "call-1"
        assert ok.attributes["gen_ai.tool.type"] == "function"
        assert ok.attributes["mcp.method.name"] == "tools/call"
        assert ok.attributes["network.transport"] == "inproc"
        assert "book_id" in str(ok.attributes["gen_ai.tool.description"])
        assert "gen_ai.tool.call.arguments" not in ok.attributes
        assert "gen_ai.tool.call.result" not in ok.attributes
        assert ok.status.status_code is StatusCode.UNSET
        assert failed.status.status_code is StatusCode.UNSET

        assert run.attributes is not None
        assert run.attributes["toolassay.cases.total"] == 1
        assert run.attributes["toolassay.cases.passed"] == 1


async def test_content_capture_is_opt_in() -> None:
    async with connect("v2") as v2_connection:
        telemetry, exporter = Telemetry.in_memory(capture_content=True)
        await _run(v2_connection, telemetry)
        spans = [s for s in exporter.get_finished_spans() if s.name.startswith("execute_tool")]
        assert spans[0].attributes is not None
        assert spans[0].attributes["gen_ai.tool.call.arguments"] == '{"book_id": "BK-0007"}'
        assert "Salt Roads" in str(spans[0].attributes["gen_ai.tool.call.result"])


def test_tool_error_marks_span() -> None:
    telemetry, exporter = Telemetry.in_memory()
    tool_call = ToolCall(id="c", name="t", arguments={})
    with telemetry.tool_span(call=tool_call, tool=None, transport="pipe") as span:
        telemetry.record_tool_result(
            span, ToolResult(call_id="c", name="t", text="boom", is_error=True)
        )
    span_data = exporter.get_finished_spans()[0]
    assert span_data.status.status_code is StatusCode.ERROR
    assert span_data.attributes is not None
    assert span_data.attributes["error.type"] == "tool_error"


def test_disabled_telemetry_is_a_no_op() -> None:
    telemetry = Telemetry.disabled()
    with telemetry.run_span(run_id="r", model="m", provider_name="p", case_count=0) as span:
        telemetry.record_run_outcome(span, passed=0, usage=Usage())
    telemetry.shutdown()


def test_resolve_otlp_endpoint_precedence() -> None:
    assert resolve_otlp_endpoint(None, {}) == "http://localhost:4318/v1/traces"
    assert resolve_otlp_endpoint("http://col:4318", {}) == "http://col:4318/v1/traces"
    assert resolve_otlp_endpoint("http://col:4318/custom", {}) == "http://col:4318/custom"
    env = {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://env:4318/"}
    assert resolve_otlp_endpoint(None, env) == "http://env:4318/v1/traces"
    env["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = "http://env:9999/v1/traces"
    assert resolve_otlp_endpoint(None, env) == "http://env:9999/v1/traces"
    assert resolve_otlp_endpoint("http://flag:1", env) == "http://flag:1/v1/traces"


def test_otlp_endpoint_reachable_probe() -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        assert otlp_endpoint_reachable(f"http://127.0.0.1:{port}/v1/traces")
    assert not otlp_endpoint_reachable(f"http://127.0.0.1:{port}/v1/traces")
    assert not otlp_endpoint_reachable("not a url")
