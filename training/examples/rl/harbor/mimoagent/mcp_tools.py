"""Adapt a task's local MCP services into mimoagent tools.

The MCP client itself lives with the dataset adapter
(``datasets/mimo/mcp/client.py`` — the dataset defines these services); this
is the harness glue that exposes them in mimoagent's tool registry, named
``mcp__<server>__<fn>`` like MiMo's own recipe proxy names them.

prepare_tasks copies this file and the client into the task image as flat
siblings of the driver; the import falls back to the cookbook package path
for unit tests.
"""

from __future__ import annotations

from typing import Any

from mimoagent.tools.base import BaseTool, ToolException, ToolOutput

try:
    import mcp_client  # in-image flat copy
except ImportError:  # cookbook package path (tests)
    from training.examples.rl.harbor.datasets.mimo.mcp import client as mcp_client

_DEFAULT_CALL_TIMEOUT = 120


class MCPHTTPTool(BaseTool):
    """One tool of one local MCP server, called over one-shot HTTP sessions."""

    def __init__(
        self,
        *,
        server: mcp_client.MCPServer,
        tool_name: str,
        tool_description: str,
        input_schema: dict[str, Any],
        timeout: int = _DEFAULT_CALL_TIMEOUT,
    ):
        super().__init__({})
        self._server = server
        self._tool_name = tool_name
        self._full_name = f"mcp__{server.name}__{tool_name}"
        self._tool_description = (
            tool_description or f"MCP tool {tool_name} on server {server.name}."
        )
        self._input_schema = input_schema or {"type": "object", "properties": {}}
        self._timeout = timeout

    @property
    def name(self) -> str:
        return self._full_name

    @property
    def description(self) -> str:
        return self._tool_description

    def get_function_parameters(self) -> dict[str, Any]:
        return self._input_schema

    def execute(self, params: Any, context: dict[str, Any] | None = None) -> ToolOutput:
        del context
        if not isinstance(params, dict):
            raise ToolException(
                f"{self._full_name} expects an object of arguments, got {type(params).__name__}"
            )
        try:
            result = mcp_client.call_tool(
                self._server, self._tool_name, params, timeout=self._timeout
            )
        except mcp_client.MCPToolError as exc:
            raise ToolException(f"{self._full_name} call failed: {exc}") from exc
        return ToolOutput(
            output=result["content"] or "(empty result)",
            success=not result["is_error"],
        )


def mimoagent_mcp_tools(
    servers: list[mcp_client.MCPServer],
    *,
    timeout: int = _DEFAULT_CALL_TIMEOUT,
) -> list[MCPHTTPTool]:
    """List every server's tools and build one mimoagent tool per MCP tool."""
    tools: list[MCPHTTPTool] = []
    for server in servers:
        for spec in mcp_client.list_tools(server):
            tools.append(
                MCPHTTPTool(
                    server=server,
                    tool_name=spec["name"],
                    tool_description=spec["description"],
                    input_schema=spec["inputSchema"],
                    timeout=timeout,
                )
            )
    return tools


__all__ = ["MCPHTTPTool", "mimoagent_mcp_tools"]
