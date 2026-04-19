"""Tests for Discord attachment handling helpers."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from signet.interfaces.discord import _doc_suffix_for_attachment, _read_attachment


def _fake_attachment(
    *,
    filename: str,
    content_type: str | None,
    data: bytes = b"",
    size: int | None = None,
) -> MagicMock:
    att = MagicMock()
    att.filename = filename
    att.content_type = content_type
    att.size = size if size is not None else len(data)
    att.read = AsyncMock(return_value=data)
    return att


class TestDocSuffixForAttachment:
    def test_pdf_by_content_type(self):
        att = _fake_attachment(filename="paper.bin", content_type="application/pdf")
        assert _doc_suffix_for_attachment(att) == ".pdf"

    def test_pdf_by_filename(self):
        att = _fake_attachment(filename="paper.PDF", content_type=None)
        assert _doc_suffix_for_attachment(att) == ".pdf"

    def test_docx(self):
        ct = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        att = _fake_attachment(filename="notes.docx", content_type=ct)
        assert _doc_suffix_for_attachment(att) == ".docx"

    def test_image_is_unsupported(self):
        att = _fake_attachment(filename="pic.png", content_type="image/png")
        assert _doc_suffix_for_attachment(att) is None


class TestReadAttachment:
    async def test_text_attachment(self):
        att = _fake_attachment(
            filename="notes.txt",
            content_type="text/plain",
            data=b"hello world",
        )
        out = await _read_attachment(att)
        assert out == "hello world"

    async def test_pdf_attachment_goes_through_docling(self):
        att = _fake_attachment(
            filename="paper.pdf",
            content_type="application/pdf",
            data=b"%PDF-1.4",
        )
        with patch(
            "signet.knowledge.ingest.convert_bytes_to_markdown",
            return_value="# Paper\n\nContent.",
        ) as conv:
            out = await _read_attachment(att)

        conv.assert_called_once()
        args, kwargs = conv.call_args
        assert args[0] == b"%PDF-1.4"
        assert kwargs["suffix"] == ".pdf"
        assert "Paper" in out

    async def test_pdf_too_large_skipped(self):
        att = _fake_attachment(
            filename="huge.pdf",
            content_type="application/pdf",
            data=b"",
            size=25_000_000,
        )
        with patch(
            "signet.knowledge.ingest.convert_bytes_to_markdown",
        ) as conv:
            out = await _read_attachment(att)

        conv.assert_not_called()
        assert out is None

    async def test_unsupported_attachment(self):
        att = _fake_attachment(
            filename="photo.png",
            content_type="image/png",
            data=b"\x89PNG",
        )
        out = await _read_attachment(att)
        assert out is None

    async def test_docling_failure_returns_none(self):
        att = _fake_attachment(
            filename="paper.pdf",
            content_type="application/pdf",
            data=b"%PDF",
        )
        with patch(
            "signet.knowledge.ingest.convert_bytes_to_markdown",
            side_effect=RuntimeError("nope"),
        ):
            out = await _read_attachment(att)
        assert out is None
