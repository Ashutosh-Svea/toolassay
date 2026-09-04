"""OpenTelemetry spans for runs, cases, model requests, and tool calls.

Span names and attributes follow the OpenTelemetry GenAI semantic conventions (development
status; attribute names checked against opentelemetry-semantic-conventions 0.65b0) and the
MCP conventions from the same repository. Hierarchy per run:

    toolassay.run                         one span per run
      invoke_agent toolassay              one per case
        chat {model}                      one per model request
        execute_tool {tool}               one per tool call

Message and tool content is only attached when ``capture_content`` is on, matching the
conventions' opt-in rule for sensitive content.
"""

from __future__ import annotations

import json
import os
import socket
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any, Literal
from urllib.parse import urlsplit

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import NoOpTracer, Span, SpanKind, StatusCode, Tracer
from pydantic import BaseModel, ConfigDict, Field

from toolassay import __version__
from toolassay.core import ModelTurn, ToolCall, ToolDefinition, ToolResult, Usage

# GenAI semantic convention attribute names.
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_RESPONSE_ID = "gen_ai.response.id"
GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS = "gen_ai.usage.cache_creation.input_tokens"
GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS = "gen_ai.usage.cache_read.input_tokens"
GEN_AI_AGENT_NAME = "gen_ai.agent.name"
GEN_AI_CONVERSATION_ID = "gen_ai.conversation.id"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"
GEN_AI_TOOL_CALL_ID = "gen_ai.tool.call.id"
GEN_AI_TOOL_TYPE = "gen_ai.tool.type"
GEN_AI_TOOL_DESCRIPTION = "gen_ai.tool.description"
GEN_AI_TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments"
GEN_AI_TOOL_CALL_RESULT = "gen_ai.tool.call.result"
ERROR_TYPE = "error.type"
MCP_METHOD_NAME = "mcp.method.name"
NETWORK_TRANSPORT = "network.transport"

# toolassay's own attributes, namespaced to keep clear of the conventions.
TOOLASSAY_RUN_ID = "toolassay.run.id"
TOOLASSAY_CASE_ID = "toolassay.case.id"
TOOLASSAY_CASE_PASSED = "toolassay.case.passed"
TOOLASSAY_CASE_TURNS = "toolassay.case.turns"
TOOLASSAY_CASES_TOTAL = "toolassay.cases.total"
TOOLASSAY_CASES_PASSED = "toolassay.cases.passed"

AGENT_NAME = "toolassay"
ExporterKind = Literal["otlp", "console", "none"]
DEFAULT_OTLP_TRACES_ENDPOINT = "http://localhost:4318/v1/traces"
TRACES_PATH = "/v1/traces"


class TelemetrySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exporter: ExporterKind = "otlp"
    endpoint: str | None = None
    """OTLP/HTTP base URL. When unset the standard OTEL_EXPORTER_OTLP_* variables apply."""
    capture_content: bool = False
    service_name: str = "toolassay"
    export_timeout_seconds: float = Field(default=5.0, gt=0)


def resolve_otlp_endpoint(endpoint: str | None, environ: Mapping[str, str] | None = None) -> str:
    """Work out the full traces URL the OTLP/HTTP exporter will post to."""
    env = os.environ if environ is None else environ
    if endpoint:
        return _with_traces_path(endpoint)
    traces = env.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    if traces:
        return traces
    base = env.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if base:
        return _with_traces_path(base)
    return DEFAULT_OTLP_TRACES_ENDPOINT


def _with_traces_path(url: str) -> str:
    parts = urlsplit(url)
    if parts.path in ("", "/"):
        return url.rstrip("/") + TRACES_PATH
    return url


def otlp_endpoint_reachable(endpoint: str, timeout: float = 0.3) -> bool:
    """True when something accepts TCP connections at the endpoint's host and port."""
    parts = urlsplit(endpoint)
    host = parts.hostname
    if host is None:
        return False
    port = parts.port or (443 if parts.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class Telemetry:
    """Creates the spans; owns the tracer provider when it built one."""

    def __init__(
        self,
        tracer: Tracer,
        *,
        provider: TracerProvider | None = None,
        capture_content: bool = False,
    ) -> None:
        self._tracer = tracer
        self._provider = provider
        self.capture_content = capture_content

    @classmethod
    def disabled(cls) -> Telemetry:
        return cls(NoOpTracer())

    @classmethod
    def configure(cls, settings: TelemetrySettings) -> Telemetry:
        if settings.exporter == "none":
            return cls.disabled()
        exporter: SpanExporter
        if settings.exporter == "console":
            exporter = ConsoleSpanExporter()
        else:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            exporter = OTLPSpanExporter(
                endpoint=resolve_otlp_endpoint(settings.endpoint),
                timeout=settings.export_timeout_seconds,
            )
        return cls.with_exporter(
            exporter,
            capture_content=settings.capture_content,
            service_name=settings.service_name,
        )

    @classmethod
    def with_exporter(
        cls,
        exporter: SpanExporter,
        *,
        capture_content: bool = False,
        service_name: str = "toolassay",
        batch: bool = True,
    ) -> Telemetry:
        resource = Resource.create({"service.name": service_name, "service.version": __version__})
        provider = TracerProvider(resource=resource)
        processor = BatchSpanProcessor(exporter) if batch else SimpleSpanProcessor(exporter)
        provider.add_span_processor(processor)
        tracer = provider.get_tracer("toolassay", __version__)
        return cls(tracer, provider=provider, capture_content=capture_content)

    @classmethod
    def in_memory(cls, *, capture_content: bool = False) -> tuple[Telemetry, InMemorySpanExporter]:
        """A telemetry instance whose spans land in memory. Meant for tests."""
        exporter = InMemorySpanExporter()
        telemetry = cls.with_exporter(exporter, capture_content=capture_content, batch=False)
        return telemetry, exporter

    def shutdown(self) -> None:
        if self._provider is not None:
            self._provider.shutdown()

    @contextmanager
    def run_span(
        self, *, run_id: str, model: str, provider_name: str, case_count: int
    ) -> Iterator[Span]:
        with self._tracer.start_as_current_span("toolassay.run", kind=SpanKind.INTERNAL) as span:
            span.set_attribute(TOOLASSAY_RUN_ID, run_id)
            span.set_attribute(GEN_AI_PROVIDER_NAME, provider_name)
            span.set_attribute(GEN_AI_REQUEST_MODEL, model)
            span.set_attribute(TOOLASSAY_CASES_TOTAL, case_count)
            yield span

    @contextmanager
    def case_span(
        self, *, run_id: str, case_id: str, model: str, provider_name: str
    ) -> Iterator[Span]:
        name = f"invoke_agent {AGENT_NAME}"
        with self._tracer.start_as_current_span(name, kind=SpanKind.INTERNAL) as span:
            span.set_attribute(GEN_AI_OPERATION_NAME, "invoke_agent")
            span.set_attribute(GEN_AI_AGENT_NAME, AGENT_NAME)
            span.set_attribute(GEN_AI_PROVIDER_NAME, provider_name)
            span.set_attribute(GEN_AI_REQUEST_MODEL, model)
            span.set_attribute(GEN_AI_CONVERSATION_ID, f"{run_id}/{case_id}")
            span.set_attribute(TOOLASSAY_RUN_ID, run_id)
            span.set_attribute(TOOLASSAY_CASE_ID, case_id)
            yield span

    @contextmanager
    def chat_span(
        self, *, model: str, provider_name: str, max_tokens: int | None
    ) -> Iterator[Span]:
        with self._tracer.start_as_current_span(f"chat {model}", kind=SpanKind.CLIENT) as span:
            span.set_attribute(GEN_AI_OPERATION_NAME, "chat")
            span.set_attribute(GEN_AI_PROVIDER_NAME, provider_name)
            span.set_attribute(GEN_AI_REQUEST_MODEL, model)
            if max_tokens is not None:
                span.set_attribute(GEN_AI_REQUEST_MAX_TOKENS, max_tokens)
            yield span

    @contextmanager
    def tool_span(
        self,
        *,
        call: ToolCall,
        tool: ToolDefinition | None,
        transport: str,
        server_address: str | None = None,
    ) -> Iterator[Span]:
        name = f"execute_tool {call.name}"
        with self._tracer.start_as_current_span(name, kind=SpanKind.CLIENT) as span:
            span.set_attribute(GEN_AI_OPERATION_NAME, "execute_tool")
            span.set_attribute(GEN_AI_TOOL_NAME, call.name)
            span.set_attribute(GEN_AI_TOOL_CALL_ID, call.id)
            span.set_attribute(GEN_AI_TOOL_TYPE, "function")
            span.set_attribute(MCP_METHOD_NAME, "tools/call")
            span.set_attribute(NETWORK_TRANSPORT, transport)
            if server_address:
                span.set_attribute("server.address", server_address)
            if tool is not None and tool.description:
                span.set_attribute(GEN_AI_TOOL_DESCRIPTION, tool.description)
            if self.capture_content:
                span.set_attribute(GEN_AI_TOOL_CALL_ARGUMENTS, _json(call.arguments))
            yield span

    def record_turn(self, span: Span, turn: ModelTurn) -> None:
        self._record_usage(span, turn.usage)
        span.set_attribute(GEN_AI_RESPONSE_FINISH_REASONS, [turn.stop_reason])
        if turn.response_id:
            span.set_attribute(GEN_AI_RESPONSE_ID, turn.response_id)
        if turn.response_model:
            span.set_attribute(GEN_AI_RESPONSE_MODEL, turn.response_model)

    def record_tool_result(self, span: Span, result: ToolResult) -> None:
        if self.capture_content:
            span.set_attribute(GEN_AI_TOOL_CALL_RESULT, result.text)
        if result.is_error:
            span.set_attribute(ERROR_TYPE, "tool_error")
            span.set_status(StatusCode.ERROR, "tool returned an error result")

    def record_case_outcome(
        self,
        span: Span,
        *,
        passed: bool,
        turns: int,
        usage: Usage,
        stop_reason: str | None,
        error: str | None,
    ) -> None:
        span.set_attribute(TOOLASSAY_CASE_PASSED, passed)
        span.set_attribute(TOOLASSAY_CASE_TURNS, turns)
        self._record_usage(span, usage)
        if stop_reason:
            span.set_attribute(GEN_AI_RESPONSE_FINISH_REASONS, [stop_reason])
        if error:
            span.set_attribute(ERROR_TYPE, error.split(":", 1)[0])
            span.set_status(StatusCode.ERROR, error)

    def record_run_outcome(self, span: Span, *, passed: int, usage: Usage) -> None:
        span.set_attribute(TOOLASSAY_CASES_PASSED, passed)
        self._record_usage(span, usage)

    @staticmethod
    def _record_usage(span: Span, usage: Usage) -> None:
        span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, usage.input_tokens)
        span.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, usage.output_tokens)
        if usage.cache_creation_input_tokens:
            span.set_attribute(
                GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS, usage.cache_creation_input_tokens
            )
        if usage.cache_read_input_tokens:
            span.set_attribute(GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS, usage.cache_read_input_tokens)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


__all__ = [
    "Telemetry",
    "TelemetrySettings",
    "otlp_endpoint_reachable",
    "resolve_otlp_endpoint",
    "trace",
]
