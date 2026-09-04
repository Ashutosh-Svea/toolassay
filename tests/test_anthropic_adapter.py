from __future__ import annotations

from typing import Any, cast

import anthropic
import httpx2
import pytest
from anthropic import AsyncAnthropic, omit
from anthropic.types import Message

from toolassay.adapters import AdapterSettings, create_adapter, detect_provider
from toolassay.adapters.anthropic import AnthropicAdapter, strict_schema, to_tool_param
from toolassay.core import AdapterFatalError, ConfigError, ToolDefinition, ToolResult

NESTED_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "minLength": 1},
        "filter": {
            "type": "object",
            "properties": {"section": {"type": "string"}},
        },
        "tags": {
            "type": "array",
            "items": {"type": "object", "properties": {"k": {"type": "string"}}},
        },
        "choice": {"anyOf": [{"type": "object", "properties": {}}, {"type": "null"}]},
    },
    "required": ["query"],
    "$defs": {"Inner": {"type": "object", "properties": {"x": {"type": "integer"}}}},
}


def test_strict_schema_closes_every_object_and_keeps_the_rest() -> None:
    closed = strict_schema(NESTED_SCHEMA)
    assert closed["additionalProperties"] is False
    assert closed["properties"]["filter"]["additionalProperties"] is False
    assert closed["properties"]["tags"]["items"]["additionalProperties"] is False
    assert closed["properties"]["choice"]["anyOf"][0]["additionalProperties"] is False
    assert closed["$defs"]["Inner"]["additionalProperties"] is False
    assert closed["properties"]["query"] == {"type": "string", "minLength": 1}
    assert closed["required"] == ["query"]
    assert "additionalProperties" not in NESTED_SCHEMA


def test_to_tool_param_strict_and_plain() -> None:
    tool = ToolDefinition(name="find_book", description="Find.", input_schema=NESTED_SCHEMA)
    strict = to_tool_param(tool, strict=True)
    assert strict["name"] == "find_book" and strict["description"] == "Find."
    assert strict.get("strict") is True
    assert cast(dict[str, Any], strict["input_schema"])["additionalProperties"] is False
    plain = to_tool_param(tool, strict=False)
    assert "strict" not in plain
    assert "additionalProperties" not in cast(dict[str, Any], plain["input_schema"])


def _message(content: list[dict[str, Any]], stop_reason: str, **usage: int) -> Message:
    return Message.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5",
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0, **usage},
        }
    )


class FakeMessages:
    def __init__(self, responses: list[Message]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Message:
        # Snapshot the history: the adapter keeps appending to the same list afterwards.
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> tuple[AsyncAnthropic, FakeMessages]:
    fake = FakeMessages(
        [
            _message(
                [
                    {"type": "text", "text": "Looking that up."},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "find_book",
                        "input": {"query": "Quill"},
                    },
                ],
                "tool_use",
                input_tokens=120,
                output_tokens=30,
                cache_read_input_tokens=50,
            ),
            _message(
                [{"type": "text", "text": "Found two books."}],
                "end_turn",
                input_tokens=300,
                output_tokens=12,
            ),
        ]
    )
    sdk_client = AsyncAnthropic(api_key="test-key-not-real")
    monkeypatch.setattr(sdk_client.messages, "create", fake.create)
    return sdk_client, fake


async def test_conversation_loop_builds_the_expected_requests(
    client: tuple[AsyncAnthropic, FakeMessages],
) -> None:
    sdk_client, fake = client
    adapter = AnthropicAdapter("claude-opus-5", effort="low", max_tokens=777, client=sdk_client)
    tool = ToolDefinition(name="find_book", description="Find.", input_schema=NESTED_SCHEMA)
    conversation = adapter.start(system="be brief", tools=[tool])

    first = await conversation.send_user("Books by Quill?")
    assert first.text == "Looking that up."
    assert first.stop_reason == "tool_use"
    assert [c.name for c in first.tool_calls] == ["find_book"]
    assert first.tool_calls[0].id == "toolu_1"
    assert first.tool_calls[0].arguments == {"query": "Quill"}
    assert first.usage.input_tokens == 120 and first.usage.output_tokens == 30
    assert first.usage.cache_read_input_tokens == 50
    assert first.response_id == "msg_1" and first.response_model == "claude-opus-5"

    request = fake.calls[0]
    assert request["model"] == "claude-opus-5"
    assert request["max_tokens"] == 777
    assert request["system"] == "be brief"
    assert request["output_config"] == {"effort": "low"}
    assert request["tools"][0]["strict"] is True
    assert request["tools"][0]["input_schema"]["additionalProperties"] is False
    assert request["messages"] == [{"role": "user", "content": "Books by Quill?"}]

    second = await conversation.send_tool_results(
        [ToolResult(call_id="toolu_1", name="find_book", text="[...]", is_error=False)]
    )
    assert second.text == "Found two books." and second.tool_calls == []
    assert second.stop_reason == "end_turn"

    messages = fake.calls[1]["messages"]
    assert len(messages) == 3
    assert messages[1]["role"] == "assistant"
    assert [b.type for b in messages[1]["content"]] == ["text", "tool_use"]
    assert messages[2] == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "[...]", "is_error": False}
        ],
    }


async def test_optional_parameters_are_omitted_when_unset(
    client: tuple[AsyncAnthropic, FakeMessages],
) -> None:
    sdk_client, fake = client
    adapter = AnthropicAdapter(effort=None, strict_tools=False, client=sdk_client)
    conversation = adapter.start(system=None, tools=[])
    await conversation.send_user("hi")
    request = fake.calls[0]
    assert request["system"] is omit
    assert request["tools"] is omit
    assert request["output_config"] is omit
    assert adapter.model == "claude-opus-5"


async def test_aclose_only_closes_owned_clients(
    client: tuple[AsyncAnthropic, FakeMessages],
) -> None:
    sdk_client, _ = client
    adapter = AnthropicAdapter(client=sdk_client)
    await adapter.aclose()
    assert not sdk_client.is_closed()
    owned = AnthropicAdapter.from_settings(AdapterSettings(model="claude-opus-5"))
    await owned.aclose()


def test_registry_detects_anthropic_models_and_rejects_unknown() -> None:
    assert detect_provider("claude-opus-5") == "anthropic"
    assert detect_provider("gpt-5") is None
    adapter = create_adapter(AdapterSettings(model="claude-sonnet-5", effort="high"))
    assert isinstance(adapter, AnthropicAdapter)
    assert adapter.effort == "high" and adapter.model == "claude-sonnet-5"
    with pytest.raises(ConfigError, match="cannot infer a provider"):
        create_adapter(AdapterSettings(model="gpt-5"))
    with pytest.raises(ConfigError, match="unknown provider"):
        create_adapter(AdapterSettings(model="gpt-5"), provider="openai")


def _status_error(cls: type[anthropic.APIStatusError], status: int) -> anthropic.APIStatusError:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(status, request=request, json={"error": {"message": "denied"}})
    return cls("denied", response=response, body=None)


@pytest.mark.parametrize(
    ("raised", "match"),
    [
        (_status_error(anthropic.AuthenticationError, 401), "rejected the credentials"),
        (_status_error(anthropic.PermissionDeniedError, 403), "rejected the credentials"),
        (_status_error(anthropic.NotFoundError, 404), "was not found"),
        (TypeError("Could not resolve authentication method."), "no Anthropic credentials"),
    ],
)
async def test_fatal_sdk_errors_are_translated(
    monkeypatch: pytest.MonkeyPatch, raised: Exception, match: str
) -> None:
    sdk_client = AsyncAnthropic(api_key="test-key-not-real")

    async def create(**kwargs: Any) -> Message:
        raise raised

    monkeypatch.setattr(sdk_client.messages, "create", create)
    conversation = AnthropicAdapter(client=sdk_client).start(system=None, tools=[])
    with pytest.raises(AdapterFatalError, match=match):
        await conversation.send_user("hi")


async def test_other_sdk_errors_pass_through(monkeypatch: pytest.MonkeyPatch) -> None:
    sdk_client = AsyncAnthropic(api_key="test-key-not-real")

    async def create(**kwargs: Any) -> Message:
        raise _status_error(anthropic.RateLimitError, 429)

    monkeypatch.setattr(sdk_client.messages, "create", create)
    conversation = AnthropicAdapter(client=sdk_client).start(system=None, tools=[])
    with pytest.raises(anthropic.RateLimitError):
        await conversation.send_user("hi")
