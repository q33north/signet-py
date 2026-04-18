"""Tool registry for Signet's live conversation tool use.

Aggregates TOOL_DEFINITIONS and a unified execute_tool dispatcher across
all tool submodules (filesystem, web, ...). Interfaces should import from
here rather than reaching into individual modules.
"""
from __future__ import annotations

from signet.tools import filesystem, web

TOOL_DEFINITIONS: list[dict] = [
    *filesystem.TOOL_DEFINITIONS,
    *web.TOOL_DEFINITIONS,
]

_EXECUTORS = [filesystem.execute_tool, web.execute_tool]
_NAMES_BY_MODULE = {
    id(filesystem.execute_tool): {t["name"] for t in filesystem.TOOL_DEFINITIONS},
    id(web.execute_tool): {t["name"] for t in web.TOOL_DEFINITIONS},
}


def execute_tool(name: str, tool_input: dict) -> str:
    """Dispatch a tool call to the module that owns it."""
    for executor in _EXECUTORS:
        if name in _NAMES_BY_MODULE[id(executor)]:
            return executor(name, tool_input)
    return f"Unknown tool: {name}"


__all__ = ["TOOL_DEFINITIONS", "execute_tool"]
