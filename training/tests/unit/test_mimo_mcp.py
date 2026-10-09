"""Tests for the MiMo general_agent MCP client and the mimoagent tool glue."""

from __future__ import annotations

import socket
import sys
import threading
import time
import types
from typing import Any

import pytest

from training.examples.rl.harbor.datasets.mimo.mcp.client import (
    MCPServer,
    MCPToolError,
    call_tool,
    list_tools,
    servers_from_json,
)


def test_servers_from_json_round_trips_manifest_entries() -> None:
    servers = servers_from_json(
        '[{"name": "ledger", "url": "http://127.0.0.1:39101/mcp"}]'
    )
    assert servers == [MCPServer(name="ledger", url="http://127.0.0.1:39101/mcp")]


def test_call_tool_wraps_transport_failures() -> None:
    server = MCPServer(name="dead", url="http://127.0.0.1:1/mcp")
    with pytest.raises(MCPToolError):
        call_tool(server, "noop", {}, timeout=5)


def _install_fake_mimoagent() -> None:
    """Stand-in for mimoagent.tools.base: the adapter only needs the contract."""
    base = types.ModuleType("mimoagent.tools.base")

    class ToolException(Exception):
        pass

    class ToolOutput:
        def __init__(self, output: str, success: bool = True, **_: Any):
            self.output = output
            self.success = success

    class BaseTool:
        def __init__(self, config: dict | None = None):
            self.config = config or {}

        def get_function_definition(self) -> dict:
            return {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": self.get_function_parameters(),
                },
            }

    base.BaseTool = BaseTool
    base.ToolOutput = ToolOutput
    base.ToolException = ToolException
    pkg = types.ModuleType("mimoagent")
    tools_pkg = types.ModuleType("mimoagent.tools")
    sys.modules.setdefault("mimoagent", pkg)
    sys.modules["mimoagent.tools"] = tools_pkg
    sys.modules["mimoagent.tools.base"] = base


def _mcp_tools_module():
    _install_fake_mimoagent()
    sys.modules.pop("training.examples.rl.harbor.mimoagent.mcp_tools", None)
    from training.examples.rl.harbor.mimoagent import mcp_tools

    return mcp_tools


def test_tool_adapter_names_and_passes_schemas() -> None:
    mcp_tools = _mcp_tools_module()
    server = MCPServer(name="ledger", url="http://127.0.0.1:39101/mcp")
    specs = [
        {
            "name": "post_entry",
            "description": "Post a ledger entry.",
            "inputSchema": {
                "type": "object",
                "properties": {"amount": {"type": "number"}},
                "required": ["amount"],
            },
        }
    ]
    mcp_tools.mcp_client.list_tools = lambda s: specs  # stub the wire
    tools = mcp_tools.mimoagent_mcp_tools([server])
    assert len(tools) == 1
    tool = tools[0]
    assert tool.name == "mcp__ledger__post_entry"
    assert tool.description == "Post a ledger entry."
    assert tool.get_function_parameters()["required"] == ["amount"]


def test_tool_adapter_maps_results_and_errors() -> None:
    mcp_tools = _mcp_tools_module()
    server = MCPServer(name="ledger", url="http://127.0.0.1:39101/mcp")
    tool = mcp_tools.MCPHTTPTool(
        server=server, tool_name="post_entry", tool_description="", input_schema={}
    )
    mcp_tools.mcp_client.call_tool = lambda *a, **k: {
        "is_error": False,
        "content": "posted",
    }
    out = tool.execute({"amount": 3})
    assert out.output == "posted" and out.success
    mcp_tools.mcp_client.call_tool = lambda *a, **k: {
        "is_error": True,
        "content": "duplicate",
    }
    out = tool.execute({"amount": 3})
    assert not out.success and out.output == "duplicate"

    def boom(*a: Any, **k: Any) -> None:
        raise MCPToolError("ConnectionError: refused")

    mcp_tools.mcp_client.call_tool = boom
    with pytest.raises(mcp_tools.ToolException, match="call failed"):
        tool.execute({"amount": 3})
    with pytest.raises(mcp_tools.ToolException, match="object"):
        tool.execute(["not", "a", "dict"])


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def mcp_server() -> MCPServer:
    """A real streamable-HTTP MCP server (FastMCP + uvicorn) in a thread."""
    pytest.importorskip("mcp")
    uvicorn = pytest.importorskip("uvicorn")
    from mcp.server.fastmcp import FastMCP

    app = FastMCP("test-server")

    @app.tool()
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @app.tool()
    def boom() -> str:
        """Always fail."""
        raise ValueError("nope")

    port = _free_port()
    config = uvicorn.Config(
        app.streamable_http_app(), host="127.0.0.1", port=port, log_level="error"
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 15
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    if not server.started:
        pytest.fail("test MCP server did not start")
    yield MCPServer(name="test", url=f"http://127.0.0.1:{port}/mcp")
    server.should_exit = True
    thread.join(timeout=10)


def test_client_against_a_real_mcp_server(mcp_server: MCPServer) -> None:
    tools = list_tools(mcp_server)
    by_name = {tool["name"]: tool for tool in tools}
    assert set(by_name) == {"add", "boom"}
    assert by_name["add"]["description"] == "Add two numbers."
    assert by_name["add"]["inputSchema"]["type"] == "object"

    result = call_tool(mcp_server, "add", {"a": 2, "b": 3})
    assert not result["is_error"]
    assert "5" in result["content"]

    failed = call_tool(mcp_server, "boom", {})
    assert failed["is_error"]
    assert failed["content"]
