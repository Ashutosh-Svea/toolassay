"""Provider-neutral value types shared by adapters, the runner, and run artifacts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolDefinition(BaseModel):
    """A tool as discovered from an MCP server, in the shape every adapter consumes."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )


class ToolCall(BaseModel):
    """One tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """The text handed back to the model for one tool call."""

    call_id: str
    name: str
    text: str
    is_error: bool = False


class Usage(BaseModel):
    """Token counts for one model request, or the sum over several."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_input_tokens=(
                self.cache_creation_input_tokens + other.cache_creation_input_tokens
            ),
            cache_read_input_tokens=self.cache_read_input_tokens + other.cache_read_input_tokens,
        )

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )


class ModelTurn(BaseModel):
    """What the model produced in response to one request."""

    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    stop_reason: str = "end_turn"
    usage: Usage = Field(default_factory=Usage)
    response_id: str | None = None
    response_model: str | None = None


class ToolassayError(Exception):
    """Base class for errors toolassay reports to the user rather than crashing on."""


class ConfigError(ToolassayError):
    """A server config, case file, or price file could not be loaded."""
