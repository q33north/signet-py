"""Filesystem tools for Signet's tool use.

Provides read_file, list_directory, search_files, and file_info tools
that Signet can invoke during conversation via the Anthropic tool use API.
Access is restricted to configured allowed paths.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import structlog

from signet.config import settings

log = structlog.get_logger()


def _is_allowed(path: Path) -> bool:
    """Check if a path falls within the allowed directories."""
    resolved = path.resolve()
    return any(
        resolved == allowed or allowed in resolved.parents
        for allowed in settings.allowed_paths
    )


def _check_path(path_str: str) -> Path:
    """Validate and resolve a path. Raises ValueError if disallowed."""
    path = Path(path_str).expanduser().resolve()
    if not _is_allowed(path):
        raise ValueError(
            f"Access denied: {path} is outside allowed directories. "
            f"Allowed: {[str(p) for p in settings.allowed_paths]}"
        )
    return path


def read_file(path: str, max_lines: int = 500, offset: int = 0) -> str:
    """Read a file's contents. Returns text or error message."""
    try:
        p = _check_path(path)
        if not p.exists():
            return f"Error: {path} does not exist"
        if not p.is_file():
            return f"Error: {path} is not a file"
        if p.stat().st_size > 1_000_000:
            return f"Error: {path} is too large ({p.stat().st_size:,} bytes). Use offset/max_lines to read a portion."

        lines = p.read_text(errors="replace").splitlines()
        total = len(lines)
        selected = lines[offset : offset + max_lines]
        content = "\n".join(selected)

        if total > offset + max_lines:
            content += f"\n\n[... {total - offset - max_lines} more lines. Use offset={offset + max_lines} to continue.]"

        log.info("tool.read_file", path=str(p), lines=len(selected))
        return content

    except ValueError as e:
        return str(e)
    except Exception as e:
        log.exception("tool.read_file_error", path=path)
        return f"Error reading {path}: {e}"


def list_directory(path: str, max_items: int = 100, pattern: str = "") -> str:
    """List contents of a directory. Optionally filter by glob pattern."""
    try:
        p = _check_path(path)
        if not p.exists():
            return f"Error: {path} does not exist"
        if not p.is_dir():
            return f"Error: {path} is not a directory"

        if pattern:
            items = sorted(p.glob(pattern))
        else:
            items = sorted(p.iterdir())

        lines = []
        for item in items[:max_items]:
            if item.name.startswith("."):
                continue
            kind = "d" if item.is_dir() else "f"
            size = ""
            if item.is_file():
                s = item.stat().st_size
                if s > 1_000_000:
                    size = f" ({s / 1_000_000:.1f}M)"
                elif s > 1000:
                    size = f" ({s / 1000:.0f}K)"
            lines.append(f"[{kind}] {item.name}{size}")

        result = f"{path}/ ({len(lines)} items)\n" + "\n".join(lines)
        if len(items) > max_items:
            result += f"\n[... {len(items) - max_items} more items]"

        log.info("tool.list_directory", path=str(p), items=len(lines))
        return result

    except ValueError as e:
        return str(e)
    except Exception as e:
        log.exception("tool.list_directory_error", path=path)
        return f"Error listing {path}: {e}"


def search_files(
    path: str,
    query: str,
    *,
    glob: str = "**/*",
    max_results: int = 20,
    max_context_lines: int = 3,
) -> str:
    """Search file contents using ripgrep (rg) or fallback grep."""
    try:
        p = _check_path(path)
        if not p.exists():
            return f"Error: {path} does not exist"

        # Try ripgrep first, fall back to grep
        try:
            result = subprocess.run(
                ["rg", "--no-heading", "-n", "--max-count=5",
                 f"--max-filesize=1M", "-g", glob,
                 f"--context={max_context_lines}", query, str(p)],
                capture_output=True, text=True, timeout=10,
            )
            output = result.stdout
        except FileNotFoundError:
            result = subprocess.run(
                ["grep", "-rn", f"--include={glob}",
                 f"-C{max_context_lines}", query, str(p)],
                capture_output=True, text=True, timeout=10,
            )
            output = result.stdout

        if not output.strip():
            return f"No matches for '{query}' in {path}"

        lines = output.strip().splitlines()
        if len(lines) > max_results * (1 + 2 * max_context_lines):
            lines = lines[: max_results * (1 + 2 * max_context_lines)]
            lines.append(f"\n[... more results truncated]")

        log.info("tool.search_files", path=str(p), query=query, matches=len(lines))
        return "\n".join(lines)

    except ValueError as e:
        return str(e)
    except subprocess.TimeoutExpired:
        return f"Search timed out in {path}"
    except Exception as e:
        log.exception("tool.search_files_error", path=path, query=query)
        return f"Error searching {path}: {e}"


def file_info(path: str) -> str:
    """Get metadata about a file or directory."""
    try:
        p = _check_path(path)
        if not p.exists():
            return f"Error: {path} does not exist"

        stat = p.stat()
        kind = "directory" if p.is_dir() else "file"
        size = stat.st_size

        info = [
            f"Path: {p}",
            f"Type: {kind}",
            f"Size: {size:,} bytes",
        ]

        if p.is_file():
            info.append(f"Extension: {p.suffix or '(none)'}")
        elif p.is_dir():
            children = list(p.iterdir())
            info.append(f"Children: {len(children)}")

        from datetime import datetime
        mtime = datetime.fromtimestamp(stat.st_mtime)
        info.append(f"Modified: {mtime.strftime('%Y-%m-%d %H:%M')}")

        return "\n".join(info)

    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"Error: {e}"


# ── Tool definitions for the Anthropic API ─────────────────

TOOL_DEFINITIONS = [
    {
        "name": "read_file",
        "description": (
            "Read the contents of a file. Use this to look at code, configs, "
            "data files, analysis outputs, etc. Access is limited to allowed directories."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file",
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Maximum lines to read (default 500)",
                    "default": 500,
                },
                "offset": {
                    "type": "integer",
                    "description": "Line offset to start reading from (default 0)",
                    "default": 0,
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "list_directory",
        "description": (
            "List the contents of a directory. Shows files and subdirectories "
            "with sizes. Optionally filter by glob pattern."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the directory",
                },
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to filter results (e.g. '*.py', '**/*.csv')",
                    "default": "",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "search_files",
        "description": (
            "Search for text patterns in files within a directory. "
            "Uses ripgrep for fast searching. Good for finding code, "
            "config values, or specific content."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory to search in",
                },
                "query": {
                    "type": "string",
                    "description": "Text or regex pattern to search for",
                },
                "glob": {
                    "type": "string",
                    "description": "File glob pattern (e.g. '*.py', '**/*.R')",
                    "default": "**/*",
                },
            },
            "required": ["path", "query"],
        },
    },
    {
        "name": "file_info",
        "description": "Get metadata about a file or directory (size, type, modification date).",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file or directory",
                },
            },
            "required": ["path"],
        },
    },
]


def execute_tool(name: str, tool_input: dict) -> str:
    """Execute a tool by name and return the result string."""
    handlers = {
        "read_file": lambda inp: read_file(
            inp["path"],
            max_lines=inp.get("max_lines", 500),
            offset=inp.get("offset", 0),
        ),
        "list_directory": lambda inp: list_directory(
            inp["path"],
            pattern=inp.get("pattern", ""),
        ),
        "search_files": lambda inp: search_files(
            inp["path"],
            inp["query"],
            glob=inp.get("glob", "**/*"),
        ),
        "file_info": lambda inp: file_info(inp["path"]),
    }

    handler = handlers.get(name)
    if not handler:
        return f"Unknown tool: {name}"

    return handler(tool_input)
