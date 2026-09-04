"""Anthropic adapter built on the official ``anthropic`` SDK."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any, cast

import anthropic
from anthropic import AsyncAnthropic, omit
from anthropic.types import (
    MessageParam,
    OutputConfigParam,
    ToolParam,
    ToolResultBlockParam,
)

from toolassay.adapters.base import AdapterSettings, Conversation, Effort, ModelAdapter
from toolassay.core import (
    AdapterFatalError,
    ModelTurn,
    ToolCall,
    ToolDefinition,
    ToolResult,
    Usage,
)

DEFAULT_MODEL = "claude-opus-5"

_SCHEMA_CHILD_LISTS = ("anyOf", "oneOf", "allOf", "prefixItems")
_SCHEMA_CHILD_MAPS = ("properties", "$defs", "definitions", "patternProperties")
_SCHEMA_CHILD_NODES = ("items", "not", "if", "then", "else")


def strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with ``additionalProperties: false`` on every object node.

    Strict tool use requires it on every object, nested ones and ``$defs`` included. Nothing
    else is changed, so the model sees the contract exactly as the server published it.
    """
    result = copy.deepcopy(schema)
    _close_objects(result)
    return result


def _is_object_node(node: dict[str, Any]) -> bool:
    node_type = node.get("type")
    if node_type == "object" or "properties" in node:
        return True
    return isinstance(node_type, list) and "object" in node_type


def _close_objects(node: Any) -> None:
    if isinstance(node, list):
        for item in node:
            _close_objects(item)
        return
    if not isinstance(node, dict):
        return
    if _is_object_node(node):
        node["additionalProperties"] = False
    for key in _SCHEMA_CHILD_MAPS:
        children = node.get(key)
        if isinstance(children, dict):
            for child in children.values():
                _close_objects(child)
    for key in _SCHEMA_CHILD_LISTS:
        _close_objects(node.get(key))
    for key in _SCHEMA_CHILD_NODES:
        _close_objects(node.get(key))


def to_tool_param(tool: ToolDefinition, *, strict: bool) -> ToolParam:
    """Translate a discovered tool into the SDK's tool definition shape."""
    schema = strict_schema(tool.input_schema) if strict else copy.deepcopy(tool.input_schema)
    schema.setdefault("type", "object")
    param: ToolParam = {
        "name": tool.name,
        "description": tool.description,
        "input_schema": cast(Any, schema),
    }
    if strict:
        param["strict"] = True
    return param


def _arguments_as_dict(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    return {"value": value}


class AnthropicAdapter(ModelAdapter):
    """Runs conversations through the Messages API."""

    provider_name = "anthropic"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        effort: Effort | None = "low",
        max_tokens: int = 4096,
        strict_tools: bool = True,
        timeout_seconds: float = 120.0,
        client: AsyncAnthropic | None = None,
    ) -> None:
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.strict_tools = strict_tools
        # The SDK reads ANTHROPIC_API_KEY (or another supported credential) from the environment.
        self._client = client or AsyncAnthropic(timeout=timeout_seconds)
        self._owns_client = client is None

    @classmethod
    def from_settings(cls, settings: AdapterSettings) -> AnthropicAdapter:
        return cls(
            settings.model,
            effort=settings.effort,
            max_tokens=settings.max_tokens,
            strict_tools=settings.strict_tools,
            timeout_seconds=settings.timeout_seconds,
        )

    def start(self, *, system: str | None, tools: Sequence[ToolDefinition]) -> Conversation:
        return AnthropicConversation(
            self._client,
            model=self.model,
            system=system,
            tools=[to_tool_param(tool, strict=self.strict_tools) for tool in tools],
            effort=self.effort,
            max_tokens=self.max_tokens,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.close()


class AnthropicConversation(Conversation):
    """Message history plus the request loop for one case."""

    def __init__(
        self,
        client: AsyncAnthropic,
        *,
        model: str,
        system: str | None,
        tools: list[ToolParam],
        effort: Effort | None,
        max_tokens: int,
    ) -> None:
        self._client = client
        self._model = model
        self._system = system
        self._tools = tools
        self._effort = effort
        self._max_tokens = max_tokens
        self.messages: list[MessageParam] = []

    async def send_user(self, text: str) -> ModelTurn:
        self.messages.append({"role": "user", "content": text})
        return await self._request()

    async def send_tool_results(self, results: Sequence[ToolResult]) -> ModelTurn:
        blocks: list[ToolResultBlockParam] = [
            {
                "type": "tool_result",
                "tool_use_id": result.call_id,
                "content": result.text,
                "is_error": result.is_error,
            }
            for result in results
        ]
        # All results for one assistant turn go back in a single user message.
        self.messages.append({"role": "user", "content": blocks})
        return await self._request()

    async def _request(self) -> ModelTurn:
        output_config: OutputConfigParam | None = (
            {"effort": self._effort} if self._effort is not None else None
        )
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=self._system if self._system is not None else omit,
                tools=self._tools if self._tools else omit,
                output_config=output_config if output_config is not None else omit,
                messages=self.messages,
            )
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
            raise AdapterFatalError(f"Anthropic rejected the credentials: {exc.message}") from exc
        except anthropic.NotFoundError as exc:
            raise AdapterFatalError(f"model {self._model!r} was not found: {exc.message}") from exc
        except TypeError as exc:
            # The SDK raises TypeError when it cannot find any credential at request time.
            if "authentication" not in str(exc).lower():
                raise
            raise AdapterFatalError(
                "no Anthropic credentials found: set ANTHROPIC_API_KEY (or another credential "
                "the anthropic SDK supports) and run again"
            ) from exc
        # Echo the full content back (thinking blocks included) so the next request is valid.
        self.messages.append({"role": "assistant", "content": response.content})

        texts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                texts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id, name=block.name, arguments=_arguments_as_dict(block.input)
                    )
                )
        usage = response.usage
        return ModelTurn(
            text="\n".join(texts).strip(),
            tool_calls=tool_calls,
            stop_reason=response.stop_reason or "end_turn",
            usage=Usage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_creation_input_tokens=usage.cache_creation_input_tokens or 0,
                cache_read_input_tokens=usage.cache_read_input_tokens or 0,
            ),
            response_id=response.id,
            response_model=response.model,
        )
