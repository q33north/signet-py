"""Web page fetcher with clean text extraction."""
from __future__ import annotations

import asyncio

import structlog
import httpx
from trafilatura import extract

log = structlog.get_logger()

_TIMEOUT = 30.0
_MAX_BODY = 500_000  # 500 KB limit for HTML/text
_MAX_PDF_BYTES = 20_000_000  # 20 MB — typical papers are 2-10 MB

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

_DOC_SUFFIXES = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/msword": ".docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/vnd.ms-powerpoint": ".pptx",
}


def _is_plain(content_type: str) -> bool:
    ct = content_type.lower().split(";", 1)[0].strip()
    if ct in _PLAIN_TYPES:
        return True
    # Source-code-ish content types from e.g. GitHub raw
    if ct.startswith("text/") and ct not in ("text/html", "text/xml"):
        return True
    return False


def _doc_suffix(content_type: str, url: str) -> str | None:
    """Return a Docling-compatible suffix if the response is a supported document."""
    ct = content_type.lower().split(";", 1)[0].strip()
    if ct in _DOC_SUFFIXES:
        return _DOC_SUFFIXES[ct]
    url_lower = url.lower().split("?", 1)[0]
    for ext in (".pdf", ".docx", ".pptx"):
        if url_lower.endswith(ext):
            return ext
    return None


async def fetch_page(url: str) -> str | None:
    """Fetch a URL and return readable text, or None on failure.

    HTML goes through trafilatura for article extraction. Plain-text
    responses (raw markdown, source files, READMEs from raw.githubusercontent,
    JSON, YAML, etc.) are returned as-is. PDF/DOCX/PPTX responses are
    converted to markdown via Docling in a worker thread.
    """
    try:
        async with httpx.AsyncClient(
            timeout=_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": "Signet/1.0 (research agent)"},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")

        suffix = _doc_suffix(content_type, url)
        if suffix:
            pdf_bytes = resp.content
            if len(pdf_bytes) > _MAX_PDF_BYTES:
                log.warning(
                    "web.doc_too_large",
                    url=url,
                    bytes=len(pdf_bytes),
                    suffix=suffix,
                )
                return None

            from signet.knowledge.ingest import convert_bytes_to_markdown

            try:
                text = await asyncio.to_thread(
                    convert_bytes_to_markdown, pdf_bytes, suffix=suffix
                )
            except Exception:
                log.exception("web.doc_convert_error", url=url, suffix=suffix)
                return None

            if not text:
                log.warning("web.doc_empty", url=url, suffix=suffix)
                return None

            log.info("web.fetched_doc", url=url, length=len(text), suffix=suffix)
            return text

        body = resp.text[:_MAX_BODY]

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
