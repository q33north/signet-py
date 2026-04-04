"""Web page fetcher with clean text extraction."""
from __future__ import annotations

import structlog
import httpx
from trafilatura import extract

log = structlog.get_logger()

_TIMEOUT = 15.0
_MAX_BODY = 500_000  # 500 KB limit for raw HTML


async def fetch_page(url: str) -> str | None:
    """Fetch a URL and return clean readable text, or None on failure."""
    try:
        async with httpx.AsyncClient(
            timeout=_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": "Signet/1.0 (research agent)"},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()

        html = resp.text[:_MAX_BODY]
        text = extract(html, include_links=False, include_tables=True)

        if not text:
            log.warning("web.extraction_empty", url=url)
            return None

        log.info("web.fetched", url=url, length=len(text))
        return text

    except httpx.HTTPStatusError as e:
        log.warning("web.http_error", url=url, status=e.response.status_code)
        return None
    except Exception:
        log.exception("web.fetch_error", url=url)
        return None
