"""Tests for signet.providers.web.fetch_page — plain/HTML/PDF routing."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from signet.providers.web import _doc_suffix, _is_plain, fetch_page


def _mock_client(*, content: bytes, text: str, content_type: str):
    """Return a patched httpx.AsyncClient whose get() yields a prepared response."""
    resp = MagicMock()
    resp.content = content
    resp.text = text
    resp.headers = {"content-type": content_type}
    resp.raise_for_status = MagicMock(return_value=None)

    client = MagicMock()
    client.get = AsyncMock(return_value=resp)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client


class TestIsPlain:
    def test_plain_text(self):
        assert _is_plain("text/plain; charset=utf-8")

    def test_markdown(self):
        assert _is_plain("text/markdown")

    def test_python_source(self):
        # text/x-python etc. are source-code content types; should be plain
        assert _is_plain("text/x-python")

    def test_html_is_not_plain(self):
        assert not _is_plain("text/html")

    def test_pdf_is_not_plain(self):
        assert not _is_plain("application/pdf")


class TestDocSuffix:
    def test_pdf_content_type(self):
        assert _doc_suffix("application/pdf", "https://example.com/x") == ".pdf"

    def test_docx_content_type(self):
        ct = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        assert _doc_suffix(ct, "https://example.com/x") == ".docx"

    def test_pdf_url_without_content_type(self):
        assert _doc_suffix("application/octet-stream", "https://arxiv.org/pdf/2504.08066.pdf") == ".pdf"

    def test_pdf_url_via_arxiv_style(self):
        # arxiv's /pdf/ URLs often don't end in .pdf, only content-type flags them
        assert _doc_suffix("application/pdf", "https://arxiv.org/pdf/2504.08066") == ".pdf"

    def test_html_returns_none(self):
        assert _doc_suffix("text/html", "https://example.com/") is None


class TestFetchPagePlain:
    async def test_returns_plain_body(self):
        client = _mock_client(content=b"hi", text="hi there", content_type="text/plain")
        with patch("signet.providers.web.httpx.AsyncClient", return_value=client):
            out = await fetch_page("https://example.com/")
        assert out == "hi there"


class TestFetchPageHTML:
    async def test_extracts_html(self):
        html = "<html><body><article><p>Paragraph.</p></article></body></html>"
        client = _mock_client(content=html.encode(), text=html, content_type="text/html")
        with patch("signet.providers.web.httpx.AsyncClient", return_value=client):
            out = await fetch_page("https://example.com/")
        assert out is not None
        assert "Paragraph" in out


class TestFetchPagePDF:
    async def test_routes_pdf_content_type_to_docling(self):
        pdf_bytes = b"%PDF-1.4 fake bytes"
        client = _mock_client(content=pdf_bytes, text="garbage", content_type="application/pdf")

        with patch("signet.providers.web.httpx.AsyncClient", return_value=client):
            with patch(
                "signet.knowledge.ingest.convert_bytes_to_markdown",
                return_value="# Extracted\n\nBody.",
            ) as conv:
                out = await fetch_page("https://arxiv.org/pdf/2504.08066")

        conv.assert_called_once()
        args, kwargs = conv.call_args
        assert args[0] == pdf_bytes
        assert kwargs["suffix"] == ".pdf"
        assert out is not None
        assert "Extracted" in out

    async def test_routes_pdf_url_without_content_type(self):
        pdf_bytes = b"%PDF-1.7"
        client = _mock_client(
            content=pdf_bytes, text="", content_type="application/octet-stream"
        )

        with patch("signet.providers.web.httpx.AsyncClient", return_value=client):
            with patch(
                "signet.knowledge.ingest.convert_bytes_to_markdown",
                return_value="paper text",
            ):
                out = await fetch_page("https://example.com/paper.pdf")

        assert out == "paper text"

    async def test_rejects_oversized_pdf(self):
        # 25 MB > 20 MB cap
        pdf_bytes = b"x" * 25_000_000
        client = _mock_client(content=pdf_bytes, text="", content_type="application/pdf")

        with patch("signet.providers.web.httpx.AsyncClient", return_value=client):
            with patch(
                "signet.knowledge.ingest.convert_bytes_to_markdown",
            ) as conv:
                out = await fetch_page("https://example.com/huge.pdf")

        conv.assert_not_called()
        assert out is None

    async def test_docling_failure_returns_none(self):
        pdf_bytes = b"%PDF"
        client = _mock_client(content=pdf_bytes, text="", content_type="application/pdf")

        with patch("signet.providers.web.httpx.AsyncClient", return_value=client):
            with patch(
                "signet.knowledge.ingest.convert_bytes_to_markdown",
                side_effect=RuntimeError("docling blew up"),
            ):
                out = await fetch_page("https://example.com/x.pdf")

        assert out is None
