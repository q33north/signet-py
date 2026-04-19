"""Web page fetcher with clean text extraction."""
from __future__ import annotations

import structlog
import httpx
from trafilatura import extract

log = structlog.get_logger()

_TIMEOUT = 15.0
_MAX_BODY = 500_000  # 500 KB limit for raw body

# Content types that are already plain text — skip HTML extraction.
_PLAIN_TYPES = (
    "text/plain",
    "text/markdown",
    "text/x-markdown",
    "application/json",
    "application/x-yaml",
    "text/yaml",
    "text/csv",
)


def _is_plain(content_type: str) -> bool:
    ct = content_type.lower().split(";", 1)[0].strip()
    if ct in _PLAIN_TYPES:
        return True
    # Source-code-ish content types from e.g. GitHub raw
    if ct.startswith("text/") and ct not in ("text/html", "text/xml"):
        return True
    return False


async def fetch_page(url: str) -> str | None:
    """Fetch a URL and return readable text, or None on failure.

    HTML goes through trafilatura for article extraction. Plain-text
    responses (raw markdown, source files, READMEs from raw.githubusercontent,
    JSON, YAML, etc.) are returned as-is — trafilatura would strip them to
    nothing.
    """
    try:
        async with httpx.AsyncClient(
            timeout=_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": "Signet/1.0 (research agent)"},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()

        body = resp.text[:_MAX_BODY]
        content_type = resp.headers.get("content-type", "")

        if _is_plain(content_type):
            log.info("web.fetched_plain", url=url, length=len(body), content_type=content_type)
            return body or None

        text = extract(body, include_links=False, include_tables=True)

        if not text:
            log.warning("web.extraction_empty", url=url, content_type=content_type)
            return None

        log.info("web.fetched", url=url, length=len(text))
        return text

    except httpx.HTTPStatusError as e:
        log.warning("web.http_error", url=url, status=e.response.status_code)
        return None
    except Exception:
        log.exception("web.fetch_error", url=url)
        return None
