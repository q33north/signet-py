"""bioRxiv API client for preprint discovery and retrieval."""
from __future__ import annotations

from datetime import date, timedelta

import httpx
import structlog
from pydantic import BaseModel, Field

log = structlog.get_logger()

_BASE = "https://api.biorxiv.org"
_TIMEOUT = 20.0


class Preprint(BaseModel):
    """A bioRxiv preprint."""
    doi: str
    title: str
    authors: str = ""
    abstract: str = ""
    category: str = ""
    date: str = ""
    version: str = "1"
    server: str = "biorxiv"

    @property
    def pdf_url(self) -> str:
        return f"https://www.biorxiv.org/content/{self.doi}v{self.version}.full.pdf"

    @property
    def url(self) -> str:
        return f"https://www.biorxiv.org/content/{self.doi}v{self.version}"


async def recent_preprints(
    *,
    days: int = 7,
    server: str = "biorxiv",
    max_results: int = 30,
) -> list[Preprint]:
    """Fetch recent preprints from the last N days.

    Args:
        days: How many days back to search.
        server: "biorxiv" or "medrxiv".
        max_results: Cap on returned results.
    """
    end = date.today()
    start = end - timedelta(days=days)
    interval = f"{start.isoformat()}/{end.isoformat()}"

    url = f"{_BASE}/details/{server}/{interval}/0/json"

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(url)
            resp.raise_for_status()

        data = resp.json()
        collection = data.get("collection", [])

        preprints = [
            Preprint(
                doi=item.get("doi", ""),
                title=item.get("title", ""),
                authors=item.get("authors", ""),
                abstract=item.get("abstract", ""),
                category=item.get("category", ""),
                date=item.get("date", ""),
                version=str(item.get("version", "1")),
                server=server,
            )
            for item in collection[:max_results]
            if item.get("doi")
        ]

        log.info("biorxiv.fetched", count=len(preprints), interval=interval)
        return preprints

    except Exception:
        log.exception("biorxiv.fetch_error", interval=interval)
        return []


async def search_preprints(
    query: str,
    *,
    days: int = 30,
    server: str = "biorxiv",
    max_results: int = 10,
) -> list[Preprint]:
    """Search recent preprints by keyword matching on title and abstract.

    The bioRxiv API doesn't have a search endpoint, so we fetch recent
    preprints and filter locally. For broad topic monitoring this works
    fine; for precise searches, use PubMed.
    """
    preprints = await recent_preprints(
        days=days, server=server, max_results=200
    )

    query_lower = query.lower()
    terms = query_lower.split()

    scored = []
    for p in preprints:
        searchable = f"{p.title} {p.abstract}".lower()
        hits = sum(1 for t in terms if t in searchable)
        if hits > 0:
            scored.append((hits, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = [p for _, p in scored[:max_results]]

    log.info("biorxiv.search", query=query[:50], results=len(results))
    return results


async def fetch_abstract(doi: str) -> Preprint | None:
    """Fetch details for a specific preprint by DOI."""
    url = f"{_BASE}/details/biorxiv/{doi}/na/json"

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(url)
            resp.raise_for_status()

        data = resp.json()
        collection = data.get("collection", [])
        if not collection:
            return None

        item = collection[-1]  # latest version
        return Preprint(
            doi=item.get("doi", doi),
            title=item.get("title", ""),
            authors=item.get("authors", ""),
            abstract=item.get("abstract", ""),
            category=item.get("category", ""),
            date=item.get("date", ""),
            version=str(item.get("version", "1")),
        )

    except Exception:
        log.exception("biorxiv.abstract_error", doi=doi)
        return None
