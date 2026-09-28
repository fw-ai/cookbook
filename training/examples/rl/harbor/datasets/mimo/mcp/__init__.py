"""MCP client for MiMo general_agent task sidecars.

The general_agent split ships four streamable-HTTP MCP servers per task,
started by the task bundle in the sidecar container on 127.0.0.1:39101-39104
(manifest.json lists them). This package is the lightweight driver for those
local services: list and call tools over one-shot sessions, stateless, same
semantics as the bundle's own mcp_bridge.py.
"""

from training.examples.rl.harbor.datasets.mimo.mcp.client import (
    MCPServer,
    MCPToolError,
    call_tool,
    list_tools,
)

__all__ = ["MCPServer", "MCPToolError", "call_tool", "list_tools"]
