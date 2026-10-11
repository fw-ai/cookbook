"""One-shot streamable-HTTP MCP client for a task's local sidecar servers.

Stdlib + the official ``mcp`` SDK only, so this file can be copied standalone
into a task image by whichever harness prepare step needs it (it must never
import cookbook modules). Each operation opens a fresh session and closes
it — the task servers are local and per-trial, so no pooling is needed,
matching the one-shot semantics of the bundle's own ``mcp_bridge.py``.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any


class MCPToolError(Exception):
    """The MCP server could not be reached or the call itself failed."""


@dataclass(frozen=True)
class MCPServer:
    """One local MCP service from the task bundle's manifest."""

    name: str
    url: str


async def _with_session(url: str, timeout: float, op: Any) -> Any:
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(url, timeout=timeout) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await op(session)


def _run(url: str, timeout: float, op: Any) -> Any:
    try:
        return asyncio.run(_with_session(url, timeout, op))
    except MCPToolError:
        raise
    except Exception as exc:
        raise MCPToolError(f"{type(exc).__name__}: {exc}") from exc


def list_tools(server: MCPServer, *, timeout: float = 120) -> list[dict[str, Any]]:
    """Return the server's tools as plain dicts (name, description, inputSchema)."""

    async def op(session: Any) -> list[dict[str, Any]]:
        resp = await session.list_tools()
        return [
            {
                "name": tool.name,
                "description": tool.description or "",
                "inputSchema": tool.inputSchema or {"type": "object", "properties": {}},
            }
            for tool in resp.tools
        ]

    return _run(server.url, timeout, op)


def _content_to_text(result: Any) -> str:
    """Flatten an MCP CallToolResult's content blocks to a single text string."""
    parts = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
        else:
            data = getattr(block, "data", None)
            parts.append(data if isinstance(data, str) else str(block))
    return "\n".join(parts)


def call_tool(
    server: MCPServer,
    tool_name: str,
    arguments: dict[str, Any] | None,
    *,
    timeout: float = 120,
) -> dict[str, Any]:
    """Call one tool. Returns {"is_error", "content", "structured"?}."""

    async def op(session: Any) -> dict[str, Any]:
        result = await session.call_tool(tool_name, arguments or {})
        payload: dict[str, Any] = {
            "is_error": bool(getattr(result, "isError", False)),
            "content": _content_to_text(result),
        }
        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            payload["structured"] = structured
        return payload

    return _run(server.url, timeout, op)


def servers_from_json(raw: str) -> list[MCPServer]:
    """Parse a JSON list of {"name", "url"} as carried by the harness env."""
    servers = []
    for item in json.loads(raw):
        servers.append(MCPServer(name=str(item["name"]), url=str(item["url"])))
    return servers


__all__ = [
    "MCPServer",
    "MCPToolError",
    "call_tool",
    "list_tools",
    "servers_from_json",
]
