from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from mcp import types as mcp_types

from toolassay.core import ConfigError, ToolCall
from toolassay.server import (
    HttpServerConfig,
    McpToolServer,
    StdioServerConfig,
    expand_variables,
    load_server_config,
    parse_server_config,
    redacted_config,
    render_tool_result,
    safe_url,
)

from .support import connect


def test_expand_variables_builtin_env_and_missing() -> None:
    assert expand_variables("${TOOLASSAY_PYTHON}") == sys.executable
    assert expand_variables("x-${TOKEN}-y", {"TOKEN": "abc"}) == "x-abc-y"
    with pytest.raises(ConfigError, match="MISSING_VAR"):
        expand_variables("${MISSING_VAR}", {})


def test_parse_stdio_config_resolves_relative_cwd(tmp_path: Path) -> None:
    config = parse_server_config(
        {"transport": "stdio", "command": "srv", "args": ["-x"], "cwd": "sub"},
        base_dir=tmp_path,
    )
    assert isinstance(config, StdioServerConfig)
    assert config.cwd == str(tmp_path / "sub")
    assert config.args == ["-x"]


def test_parse_http_config_expands_headers() -> None:
    config = parse_server_config(
        {"transport": "http", "url": "http://h/mcp", "headers": {"Authorization": "Bearer ${T}"}},
        environ={"T": "secret"},
    )
    assert isinstance(config, HttpServerConfig)
    assert config.headers == {"Authorization": "Bearer secret"}
    assert redacted_config(config)["headers"] == {"Authorization": "***"}


def test_parse_config_rejects_bad_shapes() -> None:
    with pytest.raises(ConfigError, match="mapping"):
        parse_server_config(["nope"])
    with pytest.raises(ConfigError, match="invalid"):
        parse_server_config({"transport": "carrier-pigeon", "url": "x"})
    with pytest.raises(ConfigError, match="invalid"):
        parse_server_config({"transport": "stdio", "command": "x", "bogus": 1})


def test_load_server_config_from_examples(examples_dir: Path) -> None:
    config = load_server_config(examples_dir / "server-v1.yaml")
    assert isinstance(config, StdioServerConfig)
    assert config.command == sys.executable
    assert config.args[-2:] == ["--contracts", "v1"]
    assert config.document is not None
    assert config.document["command"] == "${TOOLASSAY_PYTHON}"


def test_render_tool_result_prefers_text_then_structured() -> None:
    result = mcp_types.CallToolResult(
        content=[
            mcp_types.TextContent(type="text", text="a"),
            mcp_types.TextContent(type="text", text="b"),
        ]
    )
    assert render_tool_result(result) == "a\nb"
    structured = mcp_types.CallToolResult(content=[], structuredContent={"k": [1, 2]})
    assert render_tool_result(structured) == '{"k": [1, 2]}'
    image = mcp_types.CallToolResult(
        content=[mcp_types.ImageContent(type="image", data="abcd", mimeType="image/png")]
    )
    assert render_tool_result(image) == "[image: image/png, 4 base64 chars]"


async def test_in_process_list_and_call() -> None:
    async with connect("v2") as v2_connection:
        tools = await v2_connection.list_tools()
        assert [t.name for t in tools] == [
            "find_book",
            "get_book",
            "list_inventory",
            "reserve_book",
        ]
        inventory = next(t for t in tools if t.name == "list_inventory")
        assert set(inventory.input_schema["properties"]) == {"section", "limit"}
        assert (
            "required" not in inventory.input_schema
            or "section" not in inventory.input_schema["required"]
        )
        assert v2_connection.transport_kind == "inproc"

        result = await v2_connection.call_tool(
            ToolCall(id="1", name="get_book", arguments={"book_id": "BK-0007"})
        )
        assert not result.is_error
        assert '"Salt Roads"' in result.text

        unknown = await v2_connection.call_tool(ToolCall(id="2", name="no_such_tool", arguments={}))
        assert unknown.is_error
        assert "no_such_tool" in unknown.text

        bad_args = await v2_connection.call_tool(
            ToolCall(id="3", name="list_inventory", arguments={"limit": "many"})
        )
        assert bad_args.is_error


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_port(port: int, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.1)
    raise AssertionError(f"nothing listening on port {port}")


@pytest.fixture(scope="module")
def http_demo_server() -> Iterator[int]:
    port = _free_port()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "toolassay.demo.server",
            "--contracts",
            "v2",
            "--transport",
            "http",
            "--port",
            str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=os.environ.copy(),
    )
    try:
        _wait_for_port(port)
        yield port
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


async def test_stdio_transport_round_trip() -> None:
    config = StdioServerConfig(
        command=sys.executable, args=["-m", "toolassay.demo.server", "--contracts", "v1"]
    )
    async with McpToolServer(config) as connection:
        assert connection.transport_kind == "pipe"
        assert connection.address is not None and "toolassay.demo.server" in connection.address
        tools = await connection.list_tools()
        assert {t.name for t in tools} == {
            "find_book",
            "get_book",
            "list_inventory",
            "reserve_book",
        }
        result = await connection.call_tool(
            ToolCall(id="1", name="find_book", arguments={"query": "Quill"})
        )
        assert "The Quiet Lantern" in result.text
        assert "book_id" not in result.text


async def test_http_transport_round_trip(http_demo_server: int) -> None:
    config = HttpServerConfig(transport="http", url=f"http://127.0.0.1:{http_demo_server}/mcp")
    async with McpToolServer(config) as connection:
        assert connection.transport_kind == "tcp"
        tools = await connection.list_tools()
        assert len(tools) == 4
        result = await connection.call_tool(
            ToolCall(id="1", name="get_book", arguments={"book_id": "BK-0042"})
        )
        assert '"shelf": "P3"' in result.text


async def test_http_transport_with_headers(http_demo_server: int) -> None:
    config = HttpServerConfig(
        transport="http",
        url=f"http://127.0.0.1:{http_demo_server}/mcp",
        headers={"X-Demo": "1"},
    )
    async with McpToolServer(config) as connection:
        assert len(await connection.list_tools()) == 4


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://user:secret@host:8443/mcp?token=abc#frag", "https://host:8443/mcp"),
        ("http://${HOST}:8765/mcp", "http://${HOST}:8765/mcp"),
        ("host/mcp?x=1", "host/mcp"),
        ("https://host?token=s3cret", "https://host"),
        ("https://host#token=s3cret", "https://host"),
        ("https://u:p@host?x=1", "https://host"),
        ("https://Host.Example:8443", "https://Host.Example:8443"),
        (
            "${SCHEME}://user:s3cret@127.0.0.1:8765/mcp?token=s3cret#f",
            "${SCHEME}://127.0.0.1:8765/mcp",
        ),
        ("https://host#user:s3cret@evil", "https://host"),
        ("https://user:s3cret@host", "https://host"),
        ("//user:s3cret@host/mcp?x=1", "//host/mcp"),
        ("user:s3cret@host/mcp#f", "host/mcp"),
        ("", ""),
    ],
)
def test_safe_url_strips_userinfo_query_and_fragment(url: str, expected: str) -> None:
    assert safe_url(url) == expected
    assert "s3cret" not in safe_url(url)


def test_redacted_config_keeps_the_document_as_written() -> None:
    raw = {
        "transport": "stdio",
        "command": "server",
        "args": ["--token", "${TOKEN}"],
        "env": {"API_KEY": "${TOKEN}"},
    }
    config = parse_server_config(raw, environ={"TOKEN": "s3cret"})
    assert isinstance(config, StdioServerConfig)
    assert config.args == ["--token", "s3cret"]
    redacted = redacted_config(config)
    assert redacted["args"] == ["--token", "${TOKEN}"]
    assert redacted["env"] == {"API_KEY": "***"}
    assert "s3cret" not in json.dumps(redacted)
    assert McpToolServer(config).address == "server --token ${TOKEN}"

    http = parse_server_config(
        {"transport": "http", "url": "https://u:${TOKEN}@h/mcp?key=${TOKEN}"},
        environ={"TOKEN": "s3cret"},
    )
    assert isinstance(http, HttpServerConfig)
    assert http.url == "https://u:s3cret@h/mcp?key=s3cret"
    assert redacted_config(http)["url"] == "https://h/mcp"
    assert McpToolServer(http).address == "https://h/mcp"


def test_redacted_config_without_a_document_masks_what_it_can() -> None:
    config = HttpServerConfig(transport="http", url="https://a:b@h/mcp?q=1", headers={"X": "y"})
    assert config.document is None
    assert redacted_config(config) == {
        "transport": "http",
        "url": "https://h/mcp",
        "headers": {"X": "***"},
    }
    stdio = StdioServerConfig(command="python", args=["-m", "x"])
    assert McpToolServer(stdio).address == "python -m x"


def test_interpolated_scheme_still_redacts_userinfo_in_artifacts() -> None:
    http = parse_server_config(
        {"transport": "http", "url": "${SCHEME}://user:s3cret@host:8765/mcp?key=${TOKEN}"},
        environ={"SCHEME": "https", "TOKEN": "t0ken"},
    )
    assert isinstance(http, HttpServerConfig)
    assert http.url == "https://user:s3cret@host:8765/mcp?key=t0ken"
    redacted = json.dumps(redacted_config(http))
    assert "s3cret" not in redacted
    assert "t0ken" not in redacted
    assert redacted_config(http)["url"] == "${SCHEME}://host:8765/mcp"
    assert McpToolServer(http).address == "https://host:8765/mcp"
