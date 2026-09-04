"""The adapter interface every model provider implements.

An adapter owns two things a provider does differently: how tool definitions are declared,
and how conversation history is shaped. Everything else (the loop, scoring, telemetry,
cost) works on the neutral types in ``toolassay.core``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from toolassay.core import ModelTurn, ToolDefinition, ToolResult

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class AdapterSettings(BaseModel):
    """Provider-neutral knobs. Adapters map them onto their own request parameters."""

    model_config = ConfigDict(extra="forbid")

    model: str
    effort: Effort | None = "low"
    max_tokens: int = Field(default=4096, ge=1)
    strict_tools: bool = True
    timeout_seconds: float = Field(default=120.0, gt=0)


class Conversation(ABC):
    """One conversation with the model, holding provider-native history internally."""

    @abstractmethod
    async def send_user(self, text: str) -> ModelTurn:
        """Append a user message and request the next model turn."""

    @abstractmethod
    async def send_tool_results(self, results: Sequence[ToolResult]) -> ModelTurn:
        """Append tool results for the pending tool calls and request the next model turn."""


class ModelAdapter(ABC):
    """Factory for conversations against one provider and model."""

    provider_name: ClassVar[str]
    """Value for the ``gen_ai.provider.name`` span attribute, for example ``anthropic``."""

    model: str

    @abstractmethod
    def start(self, *, system: str | None, tools: Sequence[ToolDefinition]) -> Conversation:
        """Begin a conversation with the given system prompt and tool definitions."""

    async def aclose(self) -> None:
        """Release network resources. Adapters without any can keep the default."""
        return None
