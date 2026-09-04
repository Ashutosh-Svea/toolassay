"""Test doubles: a scripted model adapter and canned turns for the bookshop cases."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from typing import Any, Literal

from toolassay.adapters.base import AdapterSettings, Conversation, ModelAdapter
from toolassay.core import ModelTurn, ToolCall, ToolDefinition, ToolResult, Usage
from toolassay.demo.server import build_server
from toolassay.server import McpToolServer

Script = list[ModelTurn]


@asynccontextmanager
async def connect(contracts: Literal["v1", "v2"]) -> AsyncIterator[McpToolServer]:
    """An in-process connection to the demo server.

    Entered inside the test body rather than in an async fixture: the MCP client runs anyio
    task groups, and those must be entered and exited from the same task.
    """
    async with McpToolServer(build_server(contracts)) as connection:
        yield connection


def call(name: str, call_id: str = "call-1", **arguments: Any) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments)


def turn(
    text: str = "",
    *,
    tool_calls: Sequence[ToolCall] = (),
    stop_reason: str | None = None,
    input_tokens: int = 100,
    output_tokens: int = 20,
) -> ModelTurn:
    reason = stop_reason or ("tool_use" if tool_calls else "end_turn")
    return ModelTurn(
        text=text,
        tool_calls=list(tool_calls),
        stop_reason=reason,
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
        response_id=f"resp-{id(text) % 10_000}",
        response_model="scripted-model",
    )


class ScriptedConversation(Conversation):
    """Replays a script chosen by the first user message; records what was sent."""

    def __init__(self, scripts: dict[str, Script], default: Script | None) -> None:
        self._scripts = scripts
        self._default = default
        self._pending: Script = []
        self.sent: list[tuple[str, Any]] = []

    async def send_user(self, text: str) -> ModelTurn:
        self.sent.append(("user", text))
        script = self._scripts.get(text, self._default)
        if script is None:
            raise AssertionError(f"no script for prompt {text!r}")
        self._pending = list(script)
        return self._next()

    async def send_tool_results(self, results: Sequence[ToolResult]) -> ModelTurn:
        self.sent.append(("tool_results", list(results)))
        return self._next()

    def _next(self) -> ModelTurn:
        if not self._pending:
            raise AssertionError("script exhausted: the runner asked for one turn too many")
        return self._pending.pop(0)


class ScriptedAdapter(ModelAdapter):
    provider_name = "scripted"

    def __init__(
        self,
        scripts: dict[str, Script] | None = None,
        *,
        default: Script | None = None,
        model: str = "scripted-model",
        max_tokens: int = 512,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.effort = "low"
        self.strict_tools = True
        self._scripts = scripts or {}
        self._default = default
        self.started: list[tuple[str | None, list[ToolDefinition]]] = []
        self.conversations: list[ScriptedConversation] = []
        self.closed = False

    def start(self, *, system: str | None, tools: Sequence[ToolDefinition]) -> Conversation:
        self.started.append((system, list(tools)))
        conversation = ScriptedConversation(self._scripts, self._default)
        self.conversations.append(conversation)
        return conversation

    async def aclose(self) -> None:
        self.closed = True


class ExplodingAdapter(ModelAdapter):
    """Raises on the first request, to exercise the runner's error handling."""

    provider_name = "scripted"

    def __init__(self) -> None:
        self.model = "exploding-model"

    def start(self, *, system: str | None, tools: Sequence[ToolDefinition]) -> Conversation:
        return _ExplodingConversation()


class _ExplodingConversation(Conversation):
    async def send_user(self, text: str) -> ModelTurn:
        raise ConnectionError("simulated provider outage")

    async def send_tool_results(self, results: Sequence[ToolResult]) -> ModelTurn:
        raise AssertionError("unreachable")


def bookshop_scripts() -> dict[str, Script]:
    """Turns that complete every case in examples/bookshop/cases.yaml against the v2 server."""
    return {
        "Do you have any books by Mara Quill?": [
            turn(tool_calls=[call("find_book", query="Mara Quill")]),
            turn("Yes: The Quiet Lantern (11.00) and Salt and Ember (6.50), both by Mara Quill."),
        ],
        "What does book BK-0007 cost?": [
            turn(tool_calls=[call("get_book", book_id="BK-0007")]),
            turn("Salt Roads (BK-0007) costs 12.50."),
        ],
        'Where is "The Quiet Lantern" shelved?': [
            turn(tool_calls=[call("find_book", query="The Quiet Lantern")]),
            turn(tool_calls=[call("get_book", "call-2", book_id="BK-0042")]),
            turn("The Quiet Lantern is on shelf P3."),
        ],
        'How many copies of "The Quiet Lantern" are in stock?': [
            turn(tool_calls=[call("find_book", query="The Quiet Lantern")]),
            turn("There are 7 copies of The Quiet Lantern in stock."),
        ],
        "What is the cheapest book in the Poetry section?": [
            turn(tool_calls=[call("list_inventory", section="Poetry", limit=5)]),
            turn("The cheapest Poetry title is Salt and Ember at 6.50."),
        ],
        "Reserve BK-0088 for pat@example.com, for pickup on 12 September 2026.": [
            turn(
                tool_calls=[
                    call(
                        "reserve_book",
                        book_id="BK-0088",
                        customer_email="pat@example.com",
                        pickup_date="2026-09-12",
                    )
                ]
            ),
            turn("Reserved Nine Winters for pickup on 2026-09-12. Reservation id RSV-E1C1E9."),
        ],
        'Is "The Copper Almanac of Nowhere" in stock?': [
            turn(tool_calls=[call("find_book", query="The Copper Almanac of Nowhere")]),
            turn("No, the catalogue has no book called The Copper Almanac of Nowhere."),
        ],
    }


def scripted_factory(
    scripts: dict[str, Script] | None = None, default: Script | None = None
) -> Callable[[AdapterSettings], ModelAdapter]:
    def factory(settings: AdapterSettings) -> ModelAdapter:
        return ScriptedAdapter(scripts, default=default, model=settings.model)

    return factory
