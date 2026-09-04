"""Connecting to MCP servers over stdio or streamable HTTP and calling their tools."""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Mapping
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Annotated, Any, Literal

import httpx2
import yaml
from mcp import types as mcp_types
from mcp.client import Client
from mcp.client.stdio import StdioServerParameters, get_default_environment
from mcp.client.streamable_http import streamable_http_client
from mcp.server.mcpserver import MCPServer
from mcp.shared.exceptions import MCPError
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from toolassay.core import ConfigError, ToolCall, ToolDefinition, ToolResult

TransportKind = Literal["pipe", "tcp", "inproc"]


class StdioServerConfig(BaseModel):
    """Launch a server as a subprocess and talk to it over stdin and stdout."""

    model_config = ConfigDict(extra="forbid")

    transport: Literal["stdio"] = "stdio"
    command: str = Field(min_length=1)
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None


class HttpServerConfig(BaseModel):
    """Connect to a server that is already running over streamable HTTP."""

    model_config = ConfigDict(extra="forbid")

    transport: Literal["http"]
    url: str = Field(min_length=1)
    headers: dict[str, str] = Field(default_factory=dict)


ServerConfig = Annotated[StdioServerConfig | HttpServerConfig, Field(discriminator="transport")]

_config_adapter: TypeAdapter[StdioServerConfig | HttpServerConfig] = TypeAdapter(ServerConfig)

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

BUILTIN_VARIABLES = {
    "TOOLASSAY_PYTHON": sys.executable,
}
"""Variables expanded before the environment is consulted.

``${TOOLASSAY_PYTHON}`` is the interpreter running toolassay, so a config can launch a
server with ``-m some.module`` without depending on which ``python`` is on PATH.
"""


def expand_variables(text: str, environ: Mapping[str, str] | None = None) -> str:
    """Replace ``${NAME}`` with a built-in variable or an environment variable."""
    env = os.environ if environ is None else environ

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in BUILTIN_VARIABLES:
            return BUILTIN_VARIABLES[name]
        if name in env:
            return env[name]
        raise ConfigError(f"environment variable {name} is referenced but not set")

    return _ENV_PATTERN.sub(replace, text)


def _expand_tree(value: Any, environ: Mapping[str, str] | None) -> Any:
    if isinstance(value, str):
        return expand_variables(value, environ)
    if isinstance(value, list):
        return [_expand_tree(item, environ) for item in value]
    if isinstance(value, dict):
        return {key: _expand_tree(item, environ) for key, item in value.items()}
    return value


def parse_server_config(
    raw: Any, *, base_dir: Path | None = None, environ: Mapping[str, str] | None = None
) -> StdioServerConfig | HttpServerConfig:
    """Validate a parsed config mapping, expanding variables and resolving a relative cwd."""
    if not isinstance(raw, dict):
        raise ConfigError("server config must be a mapping with a 'transport' key")
    try:
        config = _config_adapter.validate_python(_expand_tree(raw, environ))
    except ValidationError as exc:
        raise ConfigError(f"server config is invalid:\n{exc}") from exc
    if isinstance(config, StdioServerConfig) and config.cwd is not None:
        cwd = Path(config.cwd)
        if not cwd.is_absolute() and base_dir is not None:
            cwd = base_dir / cwd
        config = config.model_copy(update={"cwd": str(cwd)})
    return config


def load_server_config(path: Path) -> StdioServerConfig | HttpServerConfig:
    """Load a YAML (or JSON) server config from disk."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read server config {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"server config {path} is not valid YAML: {exc}") from exc
    return parse_server_config(raw, base_dir=path.parent)


def redacted_config(config: StdioServerConfig | HttpServerConfig) -> dict[str, Any]:
    """The config as stored in run artifacts: header and env values are masked."""
    data = config.model_dump(mode="json")
    for key in ("headers", "env"):
        if data.get(key):
            data[key] = dict.fromkeys(data[key], "***")
    return data


def render_tool_result(result: mcp_types.CallToolResult) -> str:
    """Flatten an MCP tool result into the text a model sees."""
    parts: list[str] = []
    for block in result.content:
        if isinstance(block, mcp_types.TextContent):
            parts.append(block.text)
        elif isinstance(block, mcp_types.ImageContent):
            parts.append(f"[image: {block.mime_type}, {len(block.data)} base64 chars]")
        elif isinstance(block, mcp_types.AudioContent):
            parts.append(f"[audio: {block.mime_type}, {len(block.data)} base64 chars]")
        elif isinstance(block, mcp_types.ResourceLink):
            parts.append(f"[resource link: {block.uri}]")
        elif isinstance(block, mcp_types.EmbeddedResource):
            resource = block.resource
            if isinstance(resource, mcp_types.TextResourceContents):
                parts.append(resource.text)
            else:
                parts.append(f"[embedded resource: {resource.uri}]")
    if not parts and result.structured_content is not None:
        parts.append(json.dumps(result.structured_content, sort_keys=True))
    return "\n".join(parts)


class McpToolServer:
    """An open connection to one MCP server.

    Use as an async context manager. The target is a server config (stdio or HTTP) or,
    for tests, an in-process ``MCPServer`` instance.
    """

    def __init__(self, target: StdioServerConfig | HttpServerConfig | MCPServer) -> None:
        self._target = target
        self._stack: AsyncExitStack | None = None
        self._client: Client | None = None

    @property
    def transport_kind(self) -> TransportKind:
        if isinstance(self._target, StdioServerConfig):
            return "pipe"
        if isinstance(self._target, HttpServerConfig):
            return "tcp"
        return "inproc"

    @property
    def address(self) -> str | None:
        """A human-readable description of where the server lives (for spans and output)."""
        if isinstance(self._target, StdioServerConfig):
            return " ".join([self._target.command, *self._target.args])
        if isinstance(self._target, HttpServerConfig):
            return self._target.url
        return None

    async def __aenter__(self) -> McpToolServer:
        stack = AsyncExitStack()
        try:
            client = await stack.enter_async_context(self._build_client(stack))
        except BaseException:
            await stack.aclose()
            raise
        self._stack = stack
        self._client = client
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._client = None

    def _build_client(self, stack: AsyncExitStack) -> Client:
        target = self._target
        if isinstance(target, MCPServer):
            return Client(target)
        if isinstance(target, StdioServerConfig):
            params = StdioServerParameters(
                command=target.command,
                args=target.args,
                env={**get_default_environment(), **target.env},
                cwd=target.cwd,
            )
            return Client(params)
        if target.headers:
            http_client = httpx2.AsyncClient(
                headers=target.headers,
                follow_redirects=True,
                timeout=httpx2.Timeout(30.0, read=300.0),
            )
            stack.push_async_callback(http_client.aclose)
            return Client(streamable_http_client(target.url, http_client=http_client))
        return Client(target.url)

    @property
    def client(self) -> Client:
        if self._client is None:
            raise RuntimeError("McpToolServer must be entered before use")
        return self._client

    async def list_tools(self) -> list[ToolDefinition]:
        """Discover tools via tools/list, following pagination."""
        tools: list[ToolDefinition] = []
        cursor: str | None = None
        while True:
            page = await self.client.list_tools(cursor=cursor)
            for tool in page.tools:
                tools.append(
                    ToolDefinition(
                        name=tool.name,
                        description=tool.description or "",
                        input_schema=dict(tool.input_schema),
                    )
                )
            cursor = page.next_cursor
            if not cursor:
                return tools

    async def call_tool(self, call: ToolCall) -> ToolResult:
        """Invoke a tool. Protocol-level errors become error results; transport failures raise."""
        try:
            result = await self.client.call_tool(call.name, call.arguments)
        except MCPError as exc:
            return ToolResult(call_id=call.id, name=call.name, text=str(exc), is_error=True)
        return ToolResult(
            call_id=call.id,
            name=call.name,
            text=render_tool_result(result),
            is_error=bool(result.is_error),
        )
