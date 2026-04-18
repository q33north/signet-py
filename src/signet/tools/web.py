"""Web tools for Signet's tool use.

Exposes three capabilities to live conversation:
- fetch_url: read a web page (with GitHub raw-URL rewriting for repos)
- pubmed_search: query PubMed and return article summaries
- biorxiv_search: query bioRxiv preprints by keyword

The underlying providers are async; these wrappers bridge back to the
sync tool pipeline via a short-lived worker thread.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import re
from typing import Coroutine

import structlog

from signet.providers import biorxiv, pubmed, web

log = structlog.get_logger()


def _run_async(coro: Coroutine) -> object:
    """Run an async coroutine from sync code, even with a parent event loop."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


# ── GitHub URL rewriting ──────────────────────────────────
# Converts human-facing GitHub URLs into raw.githubusercontent.com so
# trafilatura isn't fighting the JS-rendered UI.

_GITHUB_BLOB = re.compile(
    r"^https?://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+?)(?:\?.*)?$"
)
_GITHUB_REPO = re.compile(r"^https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$")


def _rewrite_github_url(url: str) -> list[str]:
    """Return one or more URLs to try for a github.com link.

    - blob URLs collapse to a single raw URL.
    - bare repo URLs expand to common README locations across branches.
    - anything else is returned unchanged.
    """
    m = _GITHUB_BLOB.match(url)
    if m:
        owner, repo, branch, path = m.groups()
        return [f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"]

    m = _GITHUB_REPO.match(url)
    if m:
        owner, repo = m.groups()
        return [
            f"https://raw.githubusercontent.com/{owner}/{repo}/main/README.md",
            f"https://raw.githubusercontent.com/{owner}/{repo}/master/README.md",
        ]

    return [url]


# ── Tool implementations ──────────────────────────────────


def fetch_url(url: str, max_chars: int = 20000) -> str:
    """Fetch a URL and return clean text. GitHub links are rewritten to raw."""
    candidates = _rewrite_github_url(url)
    last_err = "unknown error"
    for candidate in candidates:
        try:
            text = _run_async(web.fetch_page(candidate))
        except Exception as e:
            last_err = str(e)
            continue
        if text:
            log.info("tool.fetch_url", url=candidate, length=len(text))
            if len(text) > max_chars:
                text = text[:max_chars] + f"\n\n[... truncated, {len(text) - max_chars} more chars]"
            prefix = ""
            if candidate != url:
                prefix = f"(resolved {url} -> {candidate})\n\n"
            return prefix + text
        last_err = "no extractable content"

    return f"Error fetching {url}: {last_err}"


def pubmed_search(query: str, max_results: int = 5) -> str:
    """Search PubMed and return formatted article summaries."""
    try:
        articles = _run_async(
            pubmed.search_and_fetch(query, max_results=max_results)
        )
    except Exception as e:
        log.exception("tool.pubmed_search_error", query=query[:50])
        return f"Error searching PubMed: {e}"

    if not articles:
        return f"No PubMed results for '{query}'."

    sections = [f"PubMed results for '{query}' ({len(articles)} of top {max_results}):"]
    for a in articles:
        authors = ", ".join(a.authors[:3])
        if len(a.authors) > 3:
            authors += f" et al. ({len(a.authors)} total)"
        abstract = (a.abstract[:600] + "...") if len(a.abstract) > 600 else a.abstract
        lines = [
            f"\n## {a.title}",
            f"PMID: {a.pmid} | {a.journal} | {a.pub_date}",
        ]
        if a.doi:
            lines.append(f"DOI: {a.doi}")
        if a.pmc_id:
            lines.append(f"PMC: {a.pmc_id} (open access)")
        if authors:
            lines.append(f"Authors: {authors}")
        lines.append(f"URL: {a.url}")
        if abstract:
            lines.append(f"\n{abstract}")
        sections.append("\n".join(lines))

    return "\n".join(sections)


def biorxiv_search(query: str, days: int = 30, max_results: int = 5) -> str:
    """Search recent bioRxiv preprints by keyword."""
    try:
        preprints = _run_async(
            biorxiv.search_preprints(query, days=days, max_results=max_results)
        )
    except Exception as e:
        log.exception("tool.biorxiv_search_error", query=query[:50])
        return f"Error searching bioRxiv: {e}"

    if not preprints:
        return f"No bioRxiv results for '{query}' in the last {days} days."

    sections = [f"bioRxiv results for '{query}' (last {days} days, {len(preprints)} hits):"]
    for p in preprints:
        abstract = (p.abstract[:500] + "...") if len(p.abstract) > 500 else p.abstract
        lines = [
            f"\n## {p.title}",
            f"DOI: {p.doi} | {p.category} | {p.date}",
            f"Authors: {p.authors[:200]}",
            f"URL: {p.url}",
        ]
        if abstract:
            lines.append(f"\n{abstract}")
        sections.append("\n".join(lines))

    return "\n".join(sections)


# ── Tool definitions for the Anthropic API ────────────────

TOOL_DEFINITIONS = [
    {
        "name": "fetch_url",
        "description": (
            "Fetch a web page and return its readable text content. "
            "Use this when the user shares a link (blog post, docs, paper, GitHub repo/file, etc.) "
            "and you need to see what's there. GitHub URLs are automatically rewritten to raw form "
            "so you get source code or README text, not the rendered UI."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch (http or https).",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters to return (default 20000).",
                    "default": 20000,
                },
            },
            "required": ["url"],
        },
    },
    {
        "name": "pubmed_search",
        "description": (
            "Search PubMed for biomedical literature and return titles, abstracts, "
            "authors, DOIs, and links. Use this for peer-reviewed papers, clinical studies, "
            "or when you need a well-indexed source across biomedical journals."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "PubMed query string (supports MeSH terms and field tags).",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max articles to return (default 5).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "biorxiv_search",
        "description": (
            "Search recent bioRxiv preprints by keyword. Good for very recent work that "
            "hasn't hit peer review yet. Note: bioRxiv has no real search endpoint, so this "
            "filters recent preprints locally — keep the window narrow for precision."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords to match in title and abstract.",
                },
                "days": {
                    "type": "integer",
                    "description": "How many days back to search (default 30).",
                    "default": 30,
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max preprints to return (default 5).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
]


def execute_tool(name: str, tool_input: dict) -> str:
    """Execute a web tool by name and return the result string."""
    handlers = {
        "fetch_url": lambda inp: fetch_url(
            inp["url"],
            max_chars=inp.get("max_chars", 20000),
        ),
        "pubmed_search": lambda inp: pubmed_search(
            inp["query"],
            max_results=inp.get("max_results", 5),
        ),
        "biorxiv_search": lambda inp: biorxiv_search(
            inp["query"],
            days=inp.get("days", 30),
            max_results=inp.get("max_results", 5),
        ),
    }
    handler = handlers.get(name)
    if not handler:
        return f"Unknown tool: {name}"
    return handler(tool_input)
