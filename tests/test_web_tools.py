"""Tests for web tools (fetch_url, pubmed_search, biorxiv_search)."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from signet.providers.biorxiv import Preprint
from signet.providers.pubmed import PubMedArticle
from signet.tools import TOOL_DEFINITIONS as ALL_TOOL_DEFINITIONS
from signet.tools import execute_tool as dispatch_tool
from signet.tools.web import (
    TOOL_DEFINITIONS,
    _rewrite_github_url,
    biorxiv_search,
    execute_tool,
    fetch_url,
    pubmed_search,
)


# Mocks: patch() auto-wraps async targets with AsyncMock. Pass plain values
# as return_value and AsyncMock yields them when awaited. side_effect with a
# plain callable returning a value also works and lets us capture args.


class TestGitHubURLRewrite:
    def test_blob_url(self):
        url = "https://github.com/foo/bar/blob/main/src/baz.py"
        out = _rewrite_github_url(url)
        assert out == ["https://raw.githubusercontent.com/foo/bar/main/src/baz.py"]

    def test_blob_url_with_branch(self):
        url = "https://github.com/foo/bar/blob/feature/x/README.md"
        out = _rewrite_github_url(url)
        assert out == ["https://raw.githubusercontent.com/foo/bar/feature/x/README.md"]

    def test_bare_repo_url(self):
        url = "https://github.com/foo/bar"
        out = _rewrite_github_url(url)
        assert "https://raw.githubusercontent.com/foo/bar/main/README.md" in out
        assert "https://raw.githubusercontent.com/foo/bar/master/README.md" in out

    def test_repo_with_trailing_slash(self):
        out = _rewrite_github_url("https://github.com/foo/bar/")
        assert any("README.md" in u for u in out)

    def test_repo_with_git_suffix(self):
        out = _rewrite_github_url("https://github.com/foo/bar.git")
        assert any("foo/bar/main/README.md" in u for u in out)

    def test_non_github_url_untouched(self):
        url = "https://example.com/path"
        assert _rewrite_github_url(url) == [url]

    def test_github_non_blob_untouched(self):
        url = "https://github.com/foo/bar/issues/1"
        assert _rewrite_github_url(url) == [url]


class TestFetchURL:
    def test_fetches_plain_url(self):
        with patch("signet.tools.web.web.fetch_page", return_value="hello world"):
            result = fetch_url("https://example.com/")
        assert "hello world" in result

    def test_truncates_long_content(self):
        long = "x" * 30000
        with patch("signet.tools.web.web.fetch_page", return_value=long):
            result = fetch_url("https://example.com/", max_chars=100)
        assert "truncated" in result
        assert len(result) < 1000

    def test_github_blob_rewritten(self):
        captured = {}

        def fake_fetch(u):
            captured["url"] = u
            return "# readme"

        with patch("signet.tools.web.web.fetch_page", side_effect=fake_fetch):
            result = fetch_url("https://github.com/foo/bar/blob/main/README.md")

        assert captured["url"] == "https://raw.githubusercontent.com/foo/bar/main/README.md"
        assert "readme" in result
        assert "resolved" in result

    def test_bare_repo_falls_back_to_master(self):
        """main/README returns None, should try master/README next."""
        calls = []

        def fake_fetch(u):
            calls.append(u)
            if "/main/" in u:
                return None
            return "master readme"

        with patch("signet.tools.web.web.fetch_page", side_effect=fake_fetch):
            result = fetch_url("https://github.com/foo/bar")

        assert len(calls) == 2
        assert "master readme" in result

    def test_fetch_failure_returns_error(self):
        with patch("signet.tools.web.web.fetch_page", return_value=None):
            result = fetch_url("https://example.com/")
        assert "Error" in result


class TestPubMedSearch:
    def test_formats_articles(self):
        articles = [
            PubMedArticle(
                pmid="12345",
                title="A study of things",
                abstract="We studied things. Things were found.",
                authors=["Smith J", "Jones A"],
                journal="Nature",
                pub_date="2025 Jan",
                doi="10.1038/x",
            )
        ]
        with patch(
            "signet.tools.web.pubmed.search_and_fetch",
            return_value=articles,
        ):
            result = pubmed_search("cancer")

        assert "A study of things" in result
        assert "PMID: 12345" in result
        assert "Nature" in result
        assert "Smith J" in result
        assert "10.1038/x" in result

    def test_empty_results(self):
        with patch(
            "signet.tools.web.pubmed.search_and_fetch",
            return_value=[],
        ):
            result = pubmed_search("xyzzy")
        assert "No PubMed results" in result


class TestBioRxivSearch:
    def test_formats_preprints(self):
        preprints = [
            Preprint(
                doi="10.1101/xyz",
                title="Preprint title",
                authors="Smith J; Jones A",
                abstract="An abstract.",
                category="bioinformatics",
                date="2026-04-01",
            )
        ]
        with patch(
            "signet.tools.web.biorxiv.search_preprints",
            return_value=preprints,
        ):
            result = biorxiv_search("agents")
        assert "Preprint title" in result
        assert "10.1101/xyz" in result
        assert "bioinformatics" in result

    def test_empty_results(self):
        with patch(
            "signet.tools.web.biorxiv.search_preprints",
            return_value=[],
        ):
            result = biorxiv_search("xyzzy")
        assert "No bioRxiv results" in result


class TestExecuteTool:
    def test_dispatches_fetch_url(self):
        with patch("signet.tools.web.web.fetch_page", return_value="page"):
            result = execute_tool("fetch_url", {"url": "https://example.com/"})
        assert "page" in result

    def test_dispatches_pubmed(self):
        with patch(
            "signet.tools.web.pubmed.search_and_fetch",
            return_value=[],
        ):
            result = execute_tool("pubmed_search", {"query": "q"})
        assert "No PubMed" in result

    def test_dispatches_biorxiv(self):
        with patch(
            "signet.tools.web.biorxiv.search_preprints",
            return_value=[],
        ):
            result = execute_tool("biorxiv_search", {"query": "q"})
        assert "No bioRxiv" in result

    def test_unknown_tool(self):
        assert "Unknown tool" in execute_tool("no_such_tool", {})


class TestToolDefinitions:
    def test_all_have_required_fields(self):
        for tool in TOOL_DEFINITIONS:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            assert tool["input_schema"]["type"] == "object"

    def test_tool_count(self):
        assert len(TOOL_DEFINITIONS) == 3

    def test_aggregated_registry_includes_web(self):
        names = {t["name"] for t in ALL_TOOL_DEFINITIONS}
        assert {"fetch_url", "pubmed_search", "biorxiv_search"}.issubset(names)
        assert {"read_file", "list_directory", "search_files", "file_info"}.issubset(names)

    def test_aggregated_dispatch_routes_to_web(self):
        with patch("signet.tools.web.web.fetch_page", return_value="hi"):
            result = dispatch_tool("fetch_url", {"url": "https://example.com/"})
        assert "hi" in result
