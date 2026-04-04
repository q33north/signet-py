"""Tests for filesystem tools."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from signet.tools.filesystem import (
    TOOL_DEFINITIONS,
    _is_allowed,
    execute_tool,
    file_info,
    list_directory,
    read_file,
    search_files,
)


@pytest.fixture
def sandbox(tmp_path):
    """Create a temp directory and patch allowed_paths to include it."""
    # Create some test files
    (tmp_path / "hello.py").write_text("print('hello world')\n")
    (tmp_path / "data.csv").write_text("name,value\nalpha,1\nbeta,2\n")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "nested.txt").write_text("nested content\n")
    (tmp_path / "big.txt").write_text("line\n" * 1000)

    with patch("signet.tools.filesystem.settings") as mock_settings:
        mock_settings.allowed_paths = [tmp_path]
        yield tmp_path


class TestReadFile:
    def test_read_small_file(self, sandbox):
        result = read_file(str(sandbox / "hello.py"))
        assert "print('hello world')" in result

    def test_read_with_offset(self, sandbox):
        result = read_file(str(sandbox / "data.csv"), offset=1, max_lines=1)
        assert "alpha,1" in result
        assert "name,value" not in result

    def test_read_with_max_lines(self, sandbox):
        result = read_file(str(sandbox / "big.txt"), max_lines=10)
        assert "more lines" in result

    def test_read_nonexistent(self, sandbox):
        result = read_file(str(sandbox / "nope.txt"))
        assert "does not exist" in result

    def test_read_directory(self, sandbox):
        result = read_file(str(sandbox / "subdir"))
        assert "not a file" in result


class TestListDirectory:
    def test_list_contents(self, sandbox):
        result = list_directory(str(sandbox))
        assert "hello.py" in result
        assert "data.csv" in result
        assert "subdir" in result

    def test_list_with_pattern(self, sandbox):
        result = list_directory(str(sandbox), pattern="*.py")
        assert "hello.py" in result
        assert "data.csv" not in result

    def test_list_nonexistent(self, sandbox):
        result = list_directory(str(sandbox / "nope"))
        assert "does not exist" in result


class TestSearchFiles:
    def test_search_finds_content(self, sandbox):
        result = search_files(str(sandbox), "hello")
        assert "hello" in result

    def test_search_no_matches(self, sandbox):
        result = search_files(str(sandbox), "xyznonexistent")
        assert "No matches" in result


class TestFileInfo:
    def test_file_info(self, sandbox):
        result = file_info(str(sandbox / "hello.py"))
        assert "file" in result
        assert ".py" in result

    def test_dir_info(self, sandbox):
        result = file_info(str(sandbox))
        assert "directory" in result


class TestAccessControl:
    def test_disallowed_path(self, sandbox):
        result = read_file("/etc/passwd")
        assert "Access denied" in result

    def test_allowed_path(self, sandbox):
        result = read_file(str(sandbox / "hello.py"))
        assert "Access denied" not in result


class TestExecuteTool:
    def test_execute_read_file(self, sandbox):
        result = execute_tool("read_file", {"path": str(sandbox / "hello.py")})
        assert "hello world" in result

    def test_execute_list_directory(self, sandbox):
        result = execute_tool("list_directory", {"path": str(sandbox)})
        assert "hello.py" in result

    def test_execute_file_info(self, sandbox):
        result = execute_tool("file_info", {"path": str(sandbox / "hello.py")})
        assert "file" in result

    def test_execute_unknown_tool(self, sandbox):
        result = execute_tool("delete_everything", {})
        assert "Unknown tool" in result


class TestToolDefinitions:
    def test_all_tools_have_required_fields(self):
        for tool in TOOL_DEFINITIONS:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            assert tool["input_schema"]["type"] == "object"

    def test_tool_count(self):
        assert len(TOOL_DEFINITIONS) == 4

    def test_all_tools_executable(self):
        """Every defined tool should have a handler in execute_tool."""
        for tool in TOOL_DEFINITIONS:
            # Provide all required params so we test dispatch, not input validation
            inp = {"path": "/nonexistent", "query": "test"}
            result = execute_tool(tool["name"], inp)
            assert "Unknown tool" not in result
