"""Deterministic MCP clients used by workflows instead of autonomous agents."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Protocol

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class ToolGateway(Protocol):
    """Minimal injectable contract used by Revit workflows and tests."""

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call one named tool and return its decoded payload."""


@dataclass(frozen=True, slots=True)
class StdioMCPGateway:
    """Open a short-lived MCP stdio session for each deterministic tool call.

    Revit state lives in the Revit document rather than the Node MCP process, so
    independent sessions avoid cross-event-loop lifetime bugs and make every
    workflow step independently retryable.
    """

    command: str
    args: tuple[str, ...]
    cwd: str | None = None
    timeout_seconds: float = 120.0

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        params = StdioServerParameters(
            command=self.command,
            args=list(self.args),
            cwd=self.cwd,
        )
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(
                read_stream,
                write_stream,
                read_timeout_seconds=timedelta(seconds=self.timeout_seconds),
            ) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments=arguments)
        if getattr(result, "isError", False) or getattr(result, "is_error", False):
            raise RuntimeError(_content_text(result) or f"MCP tool {name!r} failed")
        text = _content_text(result)
        if not text:
            structured = getattr(result, "structuredContent", None)
            if structured is None:
                structured = getattr(result, "structured_content", None)
            return structured if structured is not None else {}
        payload = decode_tool_text(text)
        _raise_for_payload_error(name, payload)
        return payload


def _content_text(result: Any) -> str:
    parts: list[str] = []
    for item in getattr(result, "content", []) or []:
        text = getattr(item, "text", None)
        if text:
            parts.append(str(text))
    return "\n".join(parts).strip()


def decode_tool_text(text: str) -> Any:
    """Decode direct JSON or the first JSON value embedded in tool prose."""
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        starts = [
            index for index, char in enumerate(stripped)
            if char in "[{"
        ]
        for start in starts:
            try:
                value, _end = decoder.raw_decode(stripped[start:])
                return value
            except json.JSONDecodeError:
                continue
    return stripped


def _raise_for_payload_error(name: str, payload: Any) -> None:
    if isinstance(payload, str):
        lowered = payload.lower()
        if " failed:" in lowered or lowered.startswith("error"):
            raise RuntimeError(payload)
        return
    if not isinstance(payload, dict):
        return
    success = payload.get("Success", payload.get("success"))
    if success is False:
        message = payload.get("Message", payload.get("message", payload))
        raise RuntimeError(f"MCP tool {name!r} failed: {message}")
    if payload.get("error"):
        raise RuntimeError(f"MCP tool {name!r} failed: {payload['error']}")


def response_items(payload: Any) -> list[Any]:
    """Return a Revit AIResult response list with casing compatibility."""
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    response = payload.get("Response", payload.get("response", []))
    if isinstance(response, list):
        return response
    return [response] if response is not None else []
